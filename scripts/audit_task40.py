"""Post-hoc immutable trace audit. Never changes Task40 thresholds or selection."""
import hashlib, json, zipfile, sys
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.giada_teacher import task40_causal_reinsertion as t

OUT=ROOT/'experiments/results/task40_audited_v1'
SHA='3bec8ca3acce9b6d3858eb6f3e2caefff6c0859745cf51f3531553e937e5b706'
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def reasons(row,q):
    result=[]
    for key,limit in [('nrmse','current_nrmse_max'),('nmax','current_nmax_max'),('signed_charge_normalized','signed_charge_error_max'),('absolute_charge_normalized','absolute_charge_error_max')]:
        if row[key]>q[limit]:result.append(key)
    e=row['event']
    if e['status']=='missing_event':result.append('missing_event')
    elif e['status']=='evaluable':
        for key,limit in [('peak_error_ms','peak_timing_max_ms'),('onset_error_ms','onset_offset_max_ms'),('offset_error_ms','onset_offset_max_ms')]:
            if e[key]>q[limit]+1e-10:result.append(key)
    if not row['blocked_current_passed']:result.append('blocked')
    assert bool(result)==(not row['passed'])
    return result

def main():
    archive=ROOT/'artifacts/task40_primary_external/giada_task40_result.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            assert '/' not in info.filename and '\\' not in info.filename
            path=OUT/info.filename;raw=z.read(info)
            if path.exists():assert path.read_bytes()==raw
            else:path.write_bytes(raw)
    read=lambda n:json.loads((OUT/n).read_text())
    spec=read('preregistration.json');q=spec['current_quality'];cases=read('cases.json');reports=read('causal_results.json')
    assert spec==json.loads((ROOT/t.CONFIG).read_text())
    with np.load(OUT/'primary_traces.npz') as z:pred={k:z[k] for k in z.files}
    with np.load(OUT/'reference_traces.npz') as z:ref={k:z[k] for k in z.files}
    failed=[];summary={};all_delta=[];recomputed=0;recompute_errors=[]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            assert np.isclose(a,b,rtol=2e-6,atol=1e-12),(a,b)
            recompute_errors.append(abs(a-b))
        else:assert a==b,(a,b)
    for name in ('I3','J1'):
        for mode in ('imposed','autonomous'):
            label=f'{name}/all/{mode}';pp=pred[f'{name}__all__{mode}__current'];rr=ref[mode+'__current']
            flags={};fail_channels=Counter();fail_reasons=Counter();interventions=Counter()
            for row in reports[label]['interventions']:
                s=spec['seeds'].index(row['seed']);c=row['case'];o=cases[c]['source_index']
                mm=t.p.evaluate_intervention(rr[:,o],rr[:,c],pp[:,s,o],pp[:,s,c],.1,q)
                compare(mm,row['metrics']);recomputed+=1
                # Cast BEFORE subtraction: unchanged physical criteria, audit numerical cancellation.
                rd=rr[:,c].astype('float64')-rr[:,o].astype('float64');pd=pp[:,s,c].astype('float64')-pp[:,s,o].astype('float64')
                mm64=t.p.evaluate_currents(rd,pd,.1,q)
                assert mm64['passed']==mm['passed'],('Subtraction precision changed verdict',label,c,row['seed'])
                flags[str((c,row['seed']))]=mm['passed']
                all_delta.append(dict(model=name,mode=mode,case=c,seed=row['seed'],metrics=mm64))
                if mm['passed']:continue
                interventions[cases[c]['intervention']]+=1
                detail=[]
                for k,ch in enumerate(mm['channels']):
                    if ch['passed']:continue
                    why=reasons(ch,q);fail_channels[ch['channel']]+=1;fail_reasons.update(why)
                    eb=pp[:,s,o].astype('float64')-rr[:,o];ei=pp[:,s,c].astype('float64')-rr[:,c]
                    delta_error=ei[:,k]-eb[:,k]
                    assert np.allclose(delta_error,pd[:,k]-rd[:,k],atol=1e-15,rtol=0)
                    detail.append(dict(**ch,failed_metrics=why,subtraction64=mm64['channels'][k],
                        baseline_current_scale=float(np.abs(rr[:,o,k]).max()),
                        effect_to_baseline_peak=float(np.abs(rd[:,k]).max()/max(np.abs(rr[:,o,k]).max(),q['current_scale_floor_ma_cm2'])),
                        baseline_error_rmse=float(np.sqrt(np.mean(eb[:,k]**2))),
                        intervention_error_rmse=float(np.sqrt(np.mean(ei[:,k]**2))),
                        differential_error_bias=float(delta_error.mean()),
                        baseline_error_first=float(eb[0,k]),intervention_error_first=float(ei[0,k]),
                        differential_error_last=float(delta_error[-1]),
                        quarter_nrmse=[float(np.sqrt(np.mean(x*x))/ch['scale_ma_cm2']) for x in np.array_split(delta_error,4)]))
                failed.append(dict(model=name,mode=mode,case=c,seed=row['seed'],metadata=cases[c],channels=detail))
            summary[label]=dict(trajectory_passed=reports[label]['passed_cases'],trajectory_total=reports[label]['total'],
                differential_passed=sum(flags.values()),differential_total=len(flags),
                failing_channels=dict(fail_channels),failing_metrics=dict(fail_reasons),failing_interventions=dict(interventions),flags=flags)
    # Differential numerical floor: formula32 vs same formula64, not just single trajectories.
    rr=ref['autonomous__current'];gg=ref['golden__current'];floors=[]
    for c in range(32,len(cases)):
        o=cases[c]['source_index']
        mm=t.p.evaluate_intervention(gg[:,o],gg[:,c],rr[:,o].astype('float64'),rr[:,c].astype('float64'),.1,q)
        floors.append(dict(case=c,metadata=cases[c],metrics=mm))
    write('differential_formula_floor.json',floors)
    # All pointwise arithmetic remains diagnostic: compare frozen candidates directly with FP64 reference too.
    against64=[]
    for name in ('I3','J1'):
        pp=pred[name+'__all__autonomous__current']
        for s,seed in enumerate(spec['seeds']):
            for c in range(32,len(cases)):
                o=cases[c]['source_index']
                mm=t.p.evaluate_intervention(gg[:,o],gg[:,c],pp[:,s,o].astype('float64'),pp[:,s,c].astype('float64'),.1,q)
                against64.append(dict(model=name,case=c,seed=seed,metrics=mm))
    write('candidate_vs_float64_formula.json',against64)
    write('failure_details.json',failed);write('differential_recomputed.json',all_delta)
    # Trajectory-level safe summaries, preserving every original metric in causal_results.json.
    trajectory={}
    for label,r in reports.items():
        trajectory[label]=dict(passed=r['passed_cases'],total=r['total'],worst_V_rmse=max(x['voltage_rmse_mv'] for x in r['cases']),
            worst_Ca_rmse=max(x['calcium_rmse_mM'] for x in r['cases']),spike_count_failures=sum(not x['spike_count_equal'] for x in r['cases']),
            worst_spike_timing=max((x['spike_timing_error_ms'] or 0) for x in r['cases']),
            voltage_domain_exits=sum(x['voltage_domain_exits'] for x in r['cases']),calcium_domain_exits=sum(x['calcium_domain_exits'] for x in r['cases']))
    audit=dict(valid=True,archive_sha256=SHA,thresholds_changed=False,recomputed_interventions=recomputed,
        subtraction64_verdicts_identical=True,recomputed_max_numeric_difference=max(recompute_errors),summary=summary,trajectory=trajectory,
        differential_formula32_floor_failed_cases=[x['case'] for x in floors if not x['metrics']['passed']],
        versus_float64_failed={n:sum(not x['metrics']['passed'] for x in against64 if x['model']==n) for n in ('I3','J1')},
        matched_I3_J1_failure_flags={mode:summary[f'I3/all/{mode}']['flags']==summary[f'J1/all/{mode}']['flags'] for mode in ('imposed','autonomous')},
        task41_authorized=False,post_hoc=True,training_performed=False)
    write('audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('summary','trajectory')},indent=2))
    for label,row in summary.items():print(label,json.dumps({k:v for k,v in row.items() if k!='flags'}))
    for row in failed:
        if row['model']=='J1' and row['mode']=='autonomous':print('FAIL',row['case'],row['seed'],row['metadata'],[(c['channel'],c['failed_metrics'],c['nrmse'],c['nmax'],c['scale_ma_cm2']) for c in row['channels']])
if __name__=='__main__':main()
