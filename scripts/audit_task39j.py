"""Read-only scientific audit of frozen 39j; never trains or changes thresholds."""
import hashlib,io,json,sys,zipfile
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39j_atomic_relaxation as j
from task39i_cpu_scanner import install
p=j.p
OUT=ROOT/'experiments/results/task39j_audited_v1'
ARCHIVE=ROOT/'artifacts/task39j_primary_external/giada_task39j_result.zip'
def sha(x):return hashlib.sha256(x).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def finite(x):
    if torch.is_tensor(x):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(map(finite,x.values()))
    if isinstance(x,(list,tuple)):return all(map(finite,x))
    return True
def key(r):return (r['role'],r['family'],r['case'],r['seed'])
def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert sha(ARCHIVE.read_bytes())=='827e45ab4181867f006d26bb70d1d47546b03ba568c9d1f37a90e22b6fabc536'
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for n in z.namelist():
            assert Path(n).name==n,n
            target=OUT/n;data=z.read(n)
            if target.exists():assert target.read_bytes()==data
            else:target.write_bytes(data)
    spec=read('preregistration.json');quality=json.loads((ROOT/p.CONFIG).read_text())['quality']
    assert sha((ROOT/p.CONFIG).read_bytes())==spec['quality_config_sha256']
    assert (OUT/'preregistration.json').read_bytes()==(ROOT/j.CONFIG).read_bytes()
    frozen=read('selection_freeze.json');payload={k:v for k,v in frozen.items() if k!='freeze_sha256'}
    assert sha(json.dumps(payload,sort_keys=True).encode())==frozen['freeze_sha256']
    assert sha((OUT/'frozen_checkpoint.pt').read_bytes())==frozen['checkpoint_sha256']
    assert not frozen['stress_used_for_selection'] and not frozen['independent_confirmation_accessed']
    states=torch.load(OUT/'frozen_checkpoint.pt',weights_only=True,map_location='cpu')
    curves=read('training_curves.json');checkpoints={};checks={}
    for step in spec['checkpoints']:
        state=torch.load(OUT/f'checkpoint_{step}.pt',weights_only=True,map_location='cpu')
        opt=torch.load(OUT/f'optimizer_{step}.pt',weights_only=False,map_location='cpu')
        assert finite(state) and finite(opt)
        checkpoints[step]=state;checks[str(step)]={'parameters_and_adam_finite':True}
    weights=j.parent_weights(torch,ROOT,spec,'cpu')
    selection=[];models=[]
    for aidx,a in enumerate(j.arms()):
        model=j.make_model(torch,weights,a,spec['seeds'],'cpu');model.load_state_dict(states[a['name']]);models.append(model)
        for k,v in weights.items():
            if not a['parent_trainable']:assert torch.equal(states[a['name']][k],v),(a,k)
        for member,seed in enumerate(spec['seeds']):
            index=aidx*3+member;chosen=frozen['chosen_steps'][index]
            for name,_ in model.named_parameters():assert torch.equal(states[a['name']][name][member],checkpoints[chosen][a['name']][name][member])
            # Frozen panels carry the original lexicographic rank components.
            selection.append(dict(arm=a['name'],seed=seed,step=chosen,rank=frozen['ranks'][index]))
    rows=read('recurrence_evaluation.json');by_model=defaultdict(list)
    for r in rows:by_model[r['model']].append(r)
    reference=by_model['joint_wfrozen39i_I3'];assert len(reference)==144
    keys={key(r) for r in reference};assert len(keys)==144
    for name,rr in by_model.items():assert len(rr)==144 and {key(r) for r in rr}==keys,name
    # Recompute qualification from all saved numerical components, not summary booleans.
    def qualify(m):
        cc=[]
        for c in m['channels']:
            e=c['event'];ep=e['status']=='not_evaluable' or (e['status']=='evaluable' and e['peak_error_ms']<=quality['peak_timing_max_ms']+1e-10 and max(e['onset_error_ms'],e['offset_error_ms'])<=quality['onset_offset_max_ms']+1e-10)
            passed=all(c[field]<=quality[limit] for field,limit in [('nrmse','current_nrmse_max'),('nmax','current_nmax_max'),('signed_charge_normalized','signed_charge_error_max'),('absolute_charge_normalized','absolute_charge_error_max')]) and ep and c['blocked_current_passed']
            assert bool(passed)==c['passed'];cc.append(passed)
        g=m['gate'];gp=g['finite'] and g['occupancy_violations']==0 and g['rmse']<=quality['gate_rmse_max'] and g['max_abs']<=quality['gate_max_abs_max']
        assert bool(gp)==g['passed'];return m['finite'] and all(cc) and gp
    for r in rows:assert qualify(r['metrics'])==r['metrics']['passed']
    prior=json.loads((ROOT/'experiments/results/task39i_recovered_v1/recurrence_evaluation.json').read_text())
    prior={key(r):r for r in prior if r['model']=='joint_wI3'}
    assert keys==set(prior)
    baseline_mismatch=[key(r) for r in reference if r['metrics']['passed']!=prior[key(r)]['metrics']['passed']]
    assert not baseline_mismatch
    base={key(r):r for r in reference};summary={};fixed={};remaining={}
    for name,rr in by_model.items():
        summary[name]={};fixed[name]=[];remaining[name]=[]
        for role in ('development','stress_diagnostic'):
            group=[r for r in rr if r['role']==role]
            events=Counter(c['event']['status'] for r in group for c in r['metrics']['channels'])
            summary[name][role]=dict(passed=sum(r['metrics']['passed'] for r in group),total=len(group),
                per_seed={str(s):sum(r['metrics']['passed'] for r in group if r['seed']==s) for s in spec['seeds']},event_status=dict(events),
                max_nrmse=max(c['nrmse'] for r in group for c in r['metrics']['channels']),max_nmax=max(c['nmax'] for r in group for c in r['metrics']['channels']))
        for r in rr:
            b=base[key(r)]
            if not b['metrics']['passed'] and r['metrics']['passed']:
                fixed[name].append(dict(case=key(r),channels=[dict(channel=c['channel'],before=c['event'],after=r['metrics']['channels'][idx]['event']) for idx,c in enumerate(b['metrics']['channels']) if not c['passed']]))
            if not r['metrics']['passed']:remaining[name].append(dict(case=key(r),channels=[c for c in r['metrics']['channels'] if not c['passed']],gate=r['metrics']['gate']))
            if name.startswith('joint_wJ') and name!='joint_wJ8':
                for idx,c in enumerate(r['metrics']['channels']):
                    if c['channel'] not in ('Nap_Et2','K_Pst'):assert c==b['metrics']['channels'][idx],(name,key(r),c['channel'])
    # Stored training pools are independently reproducible from registered seeds.
    label_errors={}
    for support in ('uniform','sensitive'):
        with np.load(OUT/f'fit_{support}.npz') as z:
            expected=j.anchors(spec['fit_seed'],spec['anchors'],support=='sensitive')
            assert all(np.array_equal(z[k],v) for k,v in expected.items() if k not in ('u','z','initial'))
            label_errors[support]={k:float(np.max(np.abs(z[k]-expected[k]))) for k in ('u','z','initial')}
            assert max(label_errors[support].values())<1e-14 # Cross-platform FP64 exp roundoff, not a qualification tolerance.
    dataset=[]
    for r in json.loads((ROOT/spec['dataset']/'dataset_manifest.json').read_text()):
        assert p.digest(ROOT/spec['dataset']/r['file'])==r['sha256'];dataset.append(r['file'])
    report=dict(valid=True,archive_sha256=sha(ARCHIVE.read_bytes()),checks=checks,quality_unchanged=True,selection=selection,
                full_key_alignment=True,baseline_39i_pass_fail_mismatches=baseline_mismatch,summaries=summary,fixed=fixed,remaining=remaining,
                unchanged_other_nine_channels_exact=True,fit_recipe_reproduced=True,label_roundoff=label_errors,dataset_hashes_verified=len(dataset),
                paired_contrasts=read('paired_contrasts.json'),limitations=['24 unique trajectories/role repeated over 3 training seeds; not72 independent protocols','Development reused across many tasks; stress diagnostic is not sealed confirmation','Stored scalar qualification independently recomputed; CPU trajectory replay separate','Fixed fullbatch sees8x examples/update','No speed benchmark or autonomous membrane validation'])
    p.save_json(OUT/'audit.json',report)
    print(json.dumps(dict(summary=summary,selection=selection)),flush=True)
    # Independent CPU implementation: frozen J1 only, all trajectories and the unchanged evaluator.
    # J1 is the simplest sufficient arm, chosen without stress-based selection.
    replay=OUT/'cpu_J1_replay';replay.mkdir(exist_ok=True)
    install(j.i.old,torch,[models[0]],replay,spec['numeric_atol'])
    scanner=j.i.old.Scanner(torch,[models[0]],2,spec['recurrence_chunk'],'cpu');replayed=[]
    for role in ('development','stress_diagnostic'):
        for family in json.loads((ROOT/p.CONFIG).read_text())['families']:
            with np.load(ROOT/spec['dataset']/f'{role}__{family}.npz') as z:a={k:z[k] for k in z.files}
            gates=scanner.run(a['voltage'],a['bridge_ca'],a['initial_gates'])[0]
            for member,seed in enumerate(spec['seeds']):
                _,currents=p.readout(a['voltage'],gates[:,member],a['multipliers'],a['reversals'])
                for case in range(2):
                    m=p.evaluate_currents(a['current_targets'][:,case],currents[:,case],.1,quality,a['gate_targets'][:,case],gates[:,member,case],a['multipliers'][case]==0)
                    replayed.append(dict(role=role,family=family,case=case,seed=seed,metrics=m))
            print('CPU replay '+role+'/'+family,flush=True)
    gpu={key(r):r for r in by_model['joint_wJ1']}
    mismatches=[key(r) for r in replayed if r['metrics']['passed']!=gpu[key(r)]['metrics']['passed']]
    p.save_json(replay/'recurrence_evaluation.json',replayed)
    p.save_json(replay/'audit.json',dict(valid=not mismatches,pass_fail_mismatches=mismatches,passed_by_role={role:sum(r['metrics']['passed'] for r in replayed if r['role']==role) for role in ('development','stress_diagnostic')},not_bitwise_equivalence=True))
    print('CPU audit mismatches '+str(mismatches),flush=True)
if __name__=='__main__':main()
