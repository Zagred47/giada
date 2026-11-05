"""Evaluation-only diagnostic completion; original failed floors are immutable."""
import hashlib,json,os,zipfile
from pathlib import Path
import numpy as np
from . import task41n_independent_confirmation as n

PLAN='experiments/task41n_recovery_v1.json'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_source(root,archive,cache):
    plan=json.loads((root/PLAN).read_text());assert sha(archive)==plan['source_sha256']
    cache.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in plan['members']:
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==plan['members'][name],name
            target=cache/name;assert target.resolve().is_relative_to(cache.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    spec,*_=n.contract(root)
    assert json.loads((cache/'preregistration.json').read_text())==spec
    assert json.loads((cache/'native_interface_report.json').read_text())['valid']
    freeze=json.loads((cache/'execution_freeze.json').read_text())
    assert freeze['eligible']==plan['eligible'] and freeze['fresh_accessed'] is False
    floors=json.loads((cache/'canonical_floors.json').read_text())
    assert [(r['mode'],r['policy'],sum(not x['passed'] for x in r['rows'])) for r in floors]==[
        ('imposed','primary',0),('imposed','voltage_solve64',0),('autonomous','primary',2),('autonomous','voltage_solve64',0)]
    return plan,spec,freeze,floors

def classification(floors,measurements_passed):
    floor_ok=all(r['passed'] for r in floors)
    return dict(original_canonical_floors_passed=floor_ok,original_confirmation_passed=bool(floor_ok and measurements_passed),
        original_confirmation_blocked_by_failed_fp32_floor=not floor_ok,
        decision='DIAGNOSTIC_COMPLETION_ORIGINAL_CONFIRMATION_BLOCKED' if not floor_ok else ('CONFIRMATION_PASSED_REQUIRES_AUDIT' if measurements_passed else 'CONFIRMATION_NO_GO'),
        task42_authorized=False,speedup_claim=False)

def run(torch,root,archive,out,device='cuda'):
    out.mkdir(exist_ok=False);cache=out.parent/'verified_original_41n'
    plan,spec,freeze,floors=verify_source(root,archive,cache)
    original={name:sha(cache/name) for name in plan['members']}
    os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    _,cfg,calcium,base=n.contract(root);cfg=n.old_native.x.calibration_config(spec,cfg)
    nets,parent=n.load_models(torch,root,device)
    assert list(nets)==freeze['eligible'] # Verify, never redo model/implementation selection.
    weights={name:n.weights_hash(net) for name,net in nets.items()};parent_hash=n.weights_hash(parent)
    assert all(torch.equal(v,nets['dual'].parent.state_dict()[k]) for k,v in parent.state_dict().items())
    hardware=n.parity(torch,nets,device,spec)
    assert hardware['eligible']==freeze['eligible'] and all(r['bitwise_equal'] for r in hardware['rows'])
    n.p.save_json(out/'hardware_verification.json',hardware)
    n.p.save_json(out/'recovery_plan.json',plan)
    n.p.save_json(out/'original_canonical_floors.json',floors)
    n.p.save_json(out/'source_reuse.json',dict(archive_sha256=plan['source_sha256'],members=original,training_updates=0,native_regenerated=False,selection_recomputed=False))
    with np.load(cache/'native_shadow_traces.npz') as z:data=dict(z)
    meta=json.loads((cache/'cases.json').read_text());assert meta==n.old_native.x.design(spec)
    refs={}
    for policy in ('primary','voltage_solve64'):
        with np.load(cache/f'reference_autonomous_{policy}.npz') as z:refs[policy]=dict(z)
    reused=json.loads((cache/'causal_results.json').read_text())
    assert all('/imposed/' in key for key in reused) and len(reused)==10
    exact=list(json.loads((cache/'fresh_execution_parity.json').read_text()));reports={};dual=None
    for name,net in {**nets,'H_control':parent,'dual_fp32':nets['dual']}.items():
        pred=n.m.rollout(torch,net,data,(cfg,calcium,base),device,'primary' if name=='dual_fp32' else 'voltage_solve64')
        assert all(np.isfinite(x).all() for x in pred.values())
        if name=='dual':dual=pred
        if name in nets:
            same=all(n.array_bits_equal(pred[k],dual[k]) for k in pred)
            exact.append(dict(mode='autonomous',model=name,bitwise_equal=same))
        for reference,ref in refs.items():
            measured=n.m.measure(pred,ref,data,meta,spec,cfg,'autonomous','41n')
            reports[name+'/autonomous/'+reference]=dict(summary=n.m.summary(measured),metrics=measured)
        np.savez_compressed(out/f'traces_{name}_autonomous.npz',**pred)
        n.p.save_json(out/'autonomous_results.json',reports);n.p.save_json(out/'execution_parity.json',exact)
        print('[GIADA41n recovery] '+name+' '+str(reports[name+'/autonomous/primary']['summary']),flush=True)
    stresses=n.evaluate_stress(torch,nets,spec,out,device)
    costs=n.benchmark(torch,nets,parent,data,(cfg,calcium,base),spec,out,device)
    assert weights=={name:n.weights_hash(net) for name,net in nets.items()} and parent_hash==n.weights_hash(parent)
    assert original=={name:sha(cache/name) for name in plan['members']}
    measurements_passed=all(reports['dual/autonomous/'+ref]['summary']['passed'] for ref in refs) and all(reused['dual/imposed/'+ref]['summary']['passed'] for ref in refs) and stresses['dual']['passed']
    result=dict(valid=True,complete=True,task='41n',scope='Previously blocked evaluation-only diagnostic completion; not a new independent confirmation',
        **classification(floors,measurements_passed),candidate_measurements_passed=measurements_passed,
        summaries={k:v['summary'] for k,v in {**reused,**reports}.items()},
        stress={k:dict(passed=v['passed'],passed_count=sum(r['metrics']['passed'] for r in v['rows']),total=len(v['rows']),exact_to_dual=v['exact_to_dual']) for k,v in stresses.items()},
        cost_rows=len(costs),execution_parity=exact,equivalent_implementations=[name for name in nets if all(r['bitwise_equal'] for r in exact if r['model']==name) and stresses[name]['exact_to_dual']],
        source_sha256=plan['source_sha256'],source_members_unchanged=True,weights_before=weights,weights_after={name:n.weights_hash(net) for name,net in nets.items()},
        training_updates_performed=0,selection_recomputed=False,native_regenerated=False,imposed_evaluation_recomputed=False,thresholds_changed=False,
        all_parameters_finite=all(all(torch.isfinite(v).all() for v in net.parameters()) for net in [*nets.values(),parent]),
        failed_original_floor_checks=[dict(mode=r['mode'],policy=r['policy'],failed_indices=[i for i,x in enumerate(r['rows']) if not x['passed']]) for r in floors if not r['passed']])
    n.p.save_json(out/'final_report.json',result);return result
