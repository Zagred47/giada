"""Post-hoc audit of immutable Task40b, no training or threshold modification."""
import hashlib,json,sys,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task40b_calcium_precision as b
from task39i_cpu_scanner import install
from audit_task40 import reasons
t,p,j=b.t,b.p,b.j
OUT=ROOT/'experiments/results/task40b_audited_v1'
def read(n):return json.loads((OUT/n).read_text())
def save(n,v):p.save_json(OUT/n,v)
def key(r):return (r['role'],r['family'],r['case'],r['seed'])
def finite(x):
    if torch.is_tensor(x):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(map(finite,x))
    return True
def compare(a,c):
    if isinstance(a,dict):
        assert a.keys()==c.keys()
        for k in a:compare(a[k],c[k])
    elif isinstance(a,list):
        assert len(a)==len(c)
        for x,y in zip(a,c):compare(x,y)
    elif isinstance(a,float):assert np.isclose(a,c,rtol=2e-6,atol=1e-12),(a,c)
    else:assert a==c,(a,c)
def qualify(m,q):
    channels=[]
    for c in m['channels']:channels.append(not reasons(c,q))
    gate=m.get('gate');gp=True
    if gate:
        gp=gate['finite'] and gate['occupancy_violations']==0 and gate['rmse']<=q['gate_rmse_max'] and gate['max_abs']<=q['gate_max_abs_max']
        assert bool(gp)==gate['passed']
    value=bool(m['finite'] and all(channels) and gp)
    assert value==m['passed'];return value

def main():
    torch.set_num_threads(1)
    archive=ROOT/'artifacts/task40b_primary_external/giada_task40b_result.zip'
    assert p.digest(archive)=='8a0df5c1c165d7dd083f6454d254179e2ded0cf5ce7c75fe44e73a551171f4db'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert Path(name).name==name
            assert (OUT/name).read_bytes()==z.read(name),name
    spec=read('preregistration.json');assert (OUT/'preregistration.json').read_bytes()==(ROOT/b.CONFIG).read_bytes()
    q=spec['current_quality'];assert q==json.loads((ROOT/t.CONFIG).read_text())['current_quality']
    for path,sha in spec['sources'].items():assert p.digest(ROOT/path)==sha,path
    freeze=read('selection_freeze.json');payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    assert p.digest(OUT/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    assert not freeze['stress_used_for_selection'] and not freeze['independent_confirmation_accessed']
    parent=t.load_models(torch,ROOT,'cpu')['J1'];parent_state=parent.state_dict()
    frozen=torch.load(OUT/'frozen_checkpoint.pt',weights_only=True,map_location='cpu')
    curves=read('training_curves.json');steps=spec['checkpoints'];checks={};checkpoints={}
    for step in steps:
        c=torch.load(OUT/f'checkpoint_{step}.pt',weights_only=False,map_location='cpu');assert c['step']==step and finite(c)
        checkpoints[step]=c;checks[str(step)]=dict(parameters_and_adam_finite=True,optimizer_step=float(next(iter(c['optimizer']['state'].values()))['step']))
        assert checks[str(step)]['optimizer_step']==step
    definitions=b.arms();models={};selection=[]
    for k,a in enumerate(definitions):
        name=a['name'];model=b.make_model(torch,parent,a,spec['seeds'],'cpu');model.load_state_dict(frozen[name]);models[name]=model
        for n,v in parent_state.items():assert torch.equal(v,frozen[name][n]),(name,n)
        for s,seed in enumerate(spec['seeds']):
            member=k*3+s;expected=min(curves,key=lambda c:c['selection_score'][member]);chosen=freeze['chosen_steps'][member]
            assert expected['step']==chosen and expected['selection_score'][member]==freeze['selection_scores'][member]
            w=checkpoints[chosen]['correction'][member];assert torch.equal(w[:a['basis']],frozen[name]['ca_correction'][s])
            assert not torch.count_nonzero(w[a['basis']:])
            assert not torch.count_nonzero(w[:,np.array(b.mask(a))==0])
            selection.append(dict(arm=name,seed=seed,chosen=chosen,selection_score=freeze['selection_scores'][member],active_weights=a['basis']*sum(b.mask(a))))
    label_check={}
    for file,seed,n in [('fit_atomic.npz',spec['fit_seed'],4096),('selection_atomic.npz',spec['selection_seed'],256)]:
        expected=b.atomic(seed,n)
        with np.load(OUT/file) as z:
            assert set(z.files)==set(expected)
            error={k:float(np.max(np.abs(z[k]-v))) for k,v in expected.items()}
            assert max(error.values())<1e-13,error;label_check[file]=error
    # Recompute atomic selection losses independently on CPU at all registered checkpoints.
    with np.load(OUT/'selection_atomic.npz') as z:data={k:z[k] for k in z.files}
    held=b.batch_pool(torch,b.pool(torch,parent,data,'cpu'),definitions);idx=torch.arange(256)[None].expand(21,-1)
    with torch.inference_mode():
        baseline=b.terms(torch,torch.zeros(21,64,8),held,idx).clamp_min(1e-15)
        errors=[]
        for curve in curves:
            actual=b.terms(torch,checkpoints[curve['step']]['correction'],held,idx)
            score=(actual/baseline).mean(-1).numpy();expected=np.array(curve['selection_score'])
            errors.append(float(np.max(np.abs(score-expected))))
            # CPU Torch2.4 versus GPU Torch2.11 raw parent kernels: diagnostic
            # loss arithmetic tolerance only, never a scientific quality gate.
            assert np.allclose(score,expected,rtol=5e-4,atol=1e-7),(curve['step'],errors[-1])
    rows=read('recurrence_evaluation.json');reports=read('causal_results.json');cases=read('cases.json')
    for r in rows:qualify(r['metrics'],q)
    for report in reports.values():
        for r in report['cases']:qualify(r['current'],q)
        for r in report['interventions']:qualify(r['metrics'],q)
    summaries={};remaining=[];original_stress=[]
    for name in ['B0']+list(models):
        selected=[r for r in rows if r['model']=='joint_w'+name];assert len(selected)==144
        summaries[name]={role:dict(passed=sum(r['metrics']['passed'] for r in selected if r['role']==role),total=72,
            per_seed={str(s):sum(r['metrics']['passed'] for r in selected if r['role']==role and r['seed']==s) for s in spec['seeds']}) for role in ('development','stress_diagnostic')}
        for r in selected:
            if not r['metrics']['passed']:remaining.append(dict(arm=name,role=r['role'],family=r['family'],case=r['case'],seed=r['seed'],channels=[c for c in r['metrics']['channels'] if not c['passed']]))
            if r['role']=='stress_diagnostic' and ((r['family'],r['case'],r['seed']) in [('same_v_history',1,17),('activation_boundary',0,29)]):
                original_stress.append(dict(arm=name,**{k:r[k] for k in ('family','case','seed','metrics')}))
        for mode in ('imposed','autonomous'):
            report=reports[name+'/'+mode];rr=report['cases'];dd=report['interventions']
            summaries[name][mode]=dict(trajectory_passed=sum(r['passed'] for r in rr),trajectory_total=len(rr),
                differential_passed=sum(r['metrics']['passed'] for r in dd),differential_total=len(dd),
                max_differential_nrmse=max(c['nrmse'] for r in dd for c in r['metrics']['channels']),
                worst_voltage_rmse_mv=max(r['voltage_rmse_mv'] for r in rr),worst_calcium_rmse_mM=max(r['calcium_rmse_mM'] for r in rr),
                spike_count_failures=sum(not r['spike_count_equal'] for r in rr),worst_spike_timing_ms=max(r['spike_timing_error_ms'] or 0 for r in rr),
                voltage_domain_exits=sum(r['voltage_domain_exits'] for r in rr),calcium_domain_exits=sum(r['calcium_domain_exits'] for r in rr),
                failing_channels=dict(Counter(c['channel'] for r in dd for c in r['metrics']['channels'] if not c['passed'])))
    save('stress_details.json',dict(original_two_cases=original_stress,remaining=remaining))
    print('Checkpoint, frozen parent, selection, and scalar qualifications verified',flush=True)
    with np.load(OUT/'reference_traces.npz') as z:refs={k:z[k] for k in z.files}
    oldspec,cfg,calcium,base=t.contract(ROOT);data,meta=t.design(ROOT,oldspec)
    recomputed=0;subtraction_changed=[]
    with np.load(OUT/'autonomous_traces.npz') as z:
        for name in ['B0']+list(models):
            pred={k:z[name+'__'+k] for k in ('voltage','calcium','gates','current')};assert all(np.isfinite(v).all() for v in pred.values())
            ref={k:refs['autonomous__'+k].copy() for k in pred}
            for k in ('voltage','calcium'):ref[k][:,:32]=data[k][:32].T
            actual=t.metrics(pred,ref,data,meta,q,cfg['gates'],'autonomous')
            compare(actual,{k:v for k,v in reports[name+'/autonomous'].items() if k!='interventions'})
            rr=refs['autonomous__current'];pp=pred['current']
            for row in reports[name+'/autonomous']['interventions']:
                c=row['case'];s=spec['seeds'].index(row['seed']);origin=cases[c]['source_index']
                m=p.evaluate_intervention(rr[:,origin],rr[:,c],pp[:,s,origin],pp[:,s,c],.1,q);compare(m,row['metrics']);recomputed+=1
                m64=p.evaluate_currents(rr[:,c].astype('float64')-rr[:,origin],pp[:,s,c].astype('float64')-pp[:,s,origin],.1,q)
                if m64['passed']!=m['passed']:subtraction_changed.append((name,c,row['seed']))
            print('Trace audit '+name,flush=True)
    assert not subtraction_changed
    old=json.loads((ROOT/'experiments/results/task40_audited_v1/audit.json').read_text())
    for mode in ('imposed','autonomous'):
        expected=old['summary']['J1/all/'+mode];actual=reports['B0/'+mode]
        assert expected['flags']=={str((r['case'],r['seed'])):r['metrics']['passed'] for r in actual['interventions']}
        assert expected['trajectory_passed']==actual['passed_cases']
    contrasts=read('paired_contrasts.json');temp=OUT/'contrast_recompute';temp.mkdir(exist_ok=True)
    # Existing evaluator requires recurrence file; preserve all source bytes.
    (temp/'recurrence_evaluation.json').write_bytes((OUT/'recurrence_evaluation.json').read_bytes())
    b.paired_contrasts(temp,spec,reports);compare(contrasts,json.loads((temp/'paired_contrasts.json').read_text()))
    audit=dict(valid=True,thresholds_changed=False,training_performed=False,post_hoc=True,archive_sha256=p.digest(archive),
        frozen_sha256=freeze['checkpoint_sha256'],checkpoint_checks=checks,selection=selection,selection_cpu_max_error=max(errors),
        label_recipe_max_errors=label_check,parent_bitwise_unchanged=True,inactive_weights_zero=True,
        all_saved_parameters_and_traces_finite=True,baseline_task40_reproduced=True,
        recurrence_scalar_checks=len(rows),causal_scalar_cases=4224,intervention_scalar_cases=2688,
        autonomous_trajectories_recomputed=2112,autonomous_interventions_recomputed=recomputed,subtraction64_verdict_changes=subtraction_changed,
        imposed_trace_recomputed=False,imposed_audit_scope='Stored scalar components and baseline flags; imposed raw traces were not saved',
        summaries=summaries,paired_contrasts=contrasts['contrasts'],cpu_recurrence_pending=True,
        task41_preparation_authorized=False,independent_confirmation=False)
    save('audit.json',audit)
    # Independent learned-coefficient cache with unchanged FP32 serial recurrence.
    replay=OUT/'cpu_B6_replay';replay.mkdir(exist_ok=True)
    install(j.i.old,torch,[models['B6']],replay,1e-5)
    scanner=j.i.old.Scanner(torch,[models['B6']],2,32,'cpu');replayed=[]
    quality=json.loads((ROOT/p.CONFIG).read_text());dataset=ROOT/json.loads((ROOT/j.CONFIG).read_text())['dataset']
    for role in ('development','stress_diagnostic'):
        for family in quality['families']:
            with np.load(dataset/f'{role}__{family}.npz') as z:a={k:z[k] for k in z.files}
            gates=scanner.run(a['voltage'],a['bridge_ca'],a['initial_gates'])[0]
            for s,seed in enumerate(spec['seeds']):
                _,current=p.readout(a['voltage'],gates[:,s],a['multipliers'],a['reversals'])
                for c in range(2):
                    mm=p.evaluate_currents(a['current_targets'][:,c],current[:,c],.1,q,a['gate_targets'][:,c],gates[:,s,c],a['multipliers'][c]==0)
                    replayed.append(dict(role=role,family=family,case=c,seed=seed,metrics=mm))
            print('CPU B6 '+role+'/'+family,flush=True)
    gpu={key(r):r for r in rows if r['model']=='joint_wB6'}
    mismatches=[key(r) for r in replayed if r['metrics']['passed']!=gpu[key(r)]['metrics']['passed']]
    p.save_json(replay/'recurrence_evaluation.json',replayed)
    assert not mismatches,mismatches
    # Frozen ablations only: test whether the stress repair actually needs u as well as z.
    ablations={}
    for label,columns in [('B6_z_only',slice(0,4)),('B6_u_only',slice(4,8))]:
        model=b.make_model(torch,parent,definitions[5],spec['seeds'],'cpu');model.load_state_dict(frozen['B6'])
        with torch.no_grad():model.ca_correction[:,:,columns]=0
        scanner=j.i.old.Scanner(torch,[model],2,32,'cpu');rr=[]
        for family in ('same_v_history','activation_boundary'):
            with np.load(dataset/f'stress_diagnostic__{family}.npz') as z:a={k:z[k] for k in z.files}
            gg=scanner.run(a['voltage'],a['bridge_ca'],a['initial_gates'])[0]
            for s,seed in enumerate(spec['seeds']):
                _,ii=p.readout(a['voltage'],gg[:,s],a['multipliers'],a['reversals'])
                for c in range(2):rr.append(dict(family=family,case=c,seed=seed,metrics=p.evaluate_currents(a['current_targets'][:,c],ii[:,c],.1,q,a['gate_targets'][:,c],gg[:,s,c],a['multipliers'][c]==0)))
        ablations[label]=rr
    save('stress_ablation.json',dict(post_hoc=True,training_performed=False,candidate_eligible=False,results=ablations))
    audit.update(cpu_recurrence_pending=False,cpu_recurrence_pass_fail_mismatches=mismatches,
        cpu_B6_passed={role:sum(r['metrics']['passed'] for r in replayed if r['role']==role) for role in ('development','stress_diagnostic')},
        preferred_arm='B6',selection_reason='Only registered package passing all opened development,stress,causal trajectory and response gates across all three seeds',
        task41_preparation_authorized=True,task41_executed=False,canonical_fallback_added=False,
        limitations=['Post-hoc package nomination uses opened stress; Task41 must use new protocols/seeds and freeze B6 before access',
        'B6 has256 active Ca correction weights versus B1 128; not a capacity-matched proof of unique equilibrium causation',
        'No competitive speedup established for B6; Task41 must measure matched ionic and integrated cost',
        'Historical32native episodes plus56formula intervention cases do not establish whole-morphology validity'])
    save('audit.json',audit)
    print(json.dumps(dict(valid=True,preferred='B6',summaries=summaries,cpu=audit['cpu_B6_passed'])),flush=True)
if __name__=='__main__':main()
