"""CPU-only re-evaluation of all immutable47 current traces; no model execution."""
import collections,gzip,hashlib,io,json,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher.multi_event_evaluation import evaluate_events,event_metrics,original_event_pass,amplitude_pass,_events
F=ROOT/'experiments/results/task47_multi_event_v1'

class ArrayWriter:
    def __init__(self,path):self.stream=gzip.open(path,'wt',encoding='utf-8');self.stream.write('[');self.first=True
    def add(self,row):
        if not self.first:self.stream.write(',')
        self.first=False;self.stream.write(json.dumps(row,separators=(',',':')))
    def close(self):self.stream.write(']');self.stream.close()

def mutants(ref,q):
    base=ref[0];signal=abs(ref-base);amplitude=float(signal.max());scale=max(float(abs(ref).max()),q['current_scale_floor_ma_cm2']);idx=int(signal.argmax())
    if amplitude<.25*scale or amplitude<10*q['current_scale_floor_ma_cm2'] or idx in (0,len(ref)-1):return [],'reference_not_temporally_supported'
    events=_events(signal,.1*amplitude,q['peak_plateau_relative_band']);closed=[e for e in events if not e['right_censored']]
    shifted=np.full_like(ref,base);shifted[10:]=ref[:-10]
    result=[('identity',ref.copy(),True),('delay_1ms',shifted,False),('amplitude_1p5x',base+1.5*(ref-base),False)]
    if closed:
        erased=ref.copy();first=closed[0];erased[first['start']:first['end']+1]=base;result.append(('deleted_episode',erased,False))
    quiet=signal<.1*amplitude;change=np.diff(np.r_[False,quiet,False].astype('int8'));starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)-1
    gaps=[(a,b) for a,b in zip(starts,ends) if b-a>=2 and a>0 and b<len(ref)-1]
    if not gaps:
        # The final quiet tail is usable too, but keep the inserted sample interior.
        gaps=[(max(int(a),1),min(int(b),len(ref)-2)) for a,b in zip(starts,ends) if min(int(b),len(ref)-2)-max(int(a),1)>=2]
    if gaps:
        a,b=max(gaps,key=lambda v:v[1]-v[0]);spike=ref.copy();spike[(a+b)//2]=base+np.sign(ref[idx]-base)*.5*amplitude;result.append(('spurious_episode',spike,False))
    return result,None

def main():
    start=time.time();freeze=json.loads((F/'freeze.json').read_text());assert freeze['valid']
    for name,sha in freeze['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Frozen evaluator source changed: '+name
    assert not (F/'summary.json').exists(),'Do not overwrite an existing study'
    p=json.loads((ROOT/'experiments/task47_preregistration_v1.json').read_text());q=p['quality'];assert q==freeze['quality_snapshot']
    archive=ROOT/'artifacts/task47_retry_external/task47_final.zip'
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==freeze['archive_sha256']
    records=ArrayWriter(F/'channel_results.json.gz');details=ArrayWriter(F/'event_details.json.gz');controls=ArrayWriter(F/'saved_reference_controls.json.gz')
    stats=collections.defaultdict(lambda:dict(total=0,eligible=0,original_passed=0,multi_event_passed=0,new_failures=0,recovered=0));coverage=collections.Counter();control_stats=collections.defaultdict(lambda:dict(total=0,as_expected=0,unexpected=0));control_skips=collections.Counter();case_rows=[];failed_cases=[];control_failures=[];channels_total=0;original_reproduced=0
    with zipfile.ZipFile(archive) as z:
        original=json.loads(z.read('results.json'));indexed={(r['geometry'],r['mode'],r['horizon_ms'],r['case'],r['seed'],r['arm']):r for r in original};original_report=json.loads(z.read('final_report.json'))
        for geo in p['geometries']:
            name=geo['id'];print('[47 multi-event] loading reference '+name,flush=True)
            with np.load(io.BytesIO(z.read(f'reference_{name}.npz'))) as saved:reference=saved['current'][:,0,0].copy()
            # Negative controls depend on the reference only; no model or seed selection.
            for case in range(reference.shape[1]):
                for node in range(reference.shape[2]):
                    for ch in range(11):
                        ref=reference[:,case,node,ch];variants,reason=mutants(ref,q)
                        if reason:control_skips[reason]+=1;continue
                        names={v[0] for v in variants}
                        for absent in {'identity','delay_1ms','amplitude_1p5x','deleted_episode','spurious_episode'}-names:control_skips[absent+'_not_constructible']+=1
                        for label,pred,want in variants:
                            ev=evaluate_events(ref,pred,.1,q,details=False);amp=amplitude_pass(ref,pred,.1,q);passed=bool(ev['passed'] and amp);ok=passed==want
                            old=event_metrics(ref,pred,.1,q['current_scale_floor_ma_cm2'],q['peak_plateau_relative_band'])
                            row=dict(geometry=name,case=case,node=node,channel=ch,control=label,expected_pass=want,observed_pass=passed,as_expected=ok,amplitude_passed=amp,original_passed=bool(original_event_pass(old,q) and amp),**ev);row['multi_event_timing_passed']=row.pop('passed');controls.add(row)
                            control_stats[label]['total']+=1;control_stats[label]['as_expected']+=int(ok);control_stats[label]['unexpected']+=int(not ok)
                            if not ok:control_failures.append(row)
            print('[47 multi-event] saved-reference controls complete '+name,flush=True)
            for mode in ('imposed','autonomous'):
                print('[47 multi-event] loading '+name+'/'+mode,flush=True)
                with np.load(io.BytesIO(z.read(f'traces_{name}_{mode}.npz'))) as saved:predictions=saved['current']
                for horizon in p['prefix_horizons_ms']:
                    count=round(horizon/.1)
                    for case in range(reference.shape[1]):
                        for si,seed in enumerate(p['seeds']):
                            for ai,arm in enumerate(p['arms']):
                                oldrow=indexed[name,mode,horizon,case,seed,arm];multi_current=True;reasons=collections.Counter();changed=[]
                                for node,oldnode in enumerate(oldrow['per_node']):
                                    node_ok=oldnode['gate'] is None or oldnode['gate']['passed']
                                    for ch,oldchannel in enumerate(oldnode['channels']):
                                        ref=reference[:count,case,node,ch];pred=predictions[:count,si,ai,case,node,ch]
                                        old=event_metrics(ref,pred,.1,q['current_scale_floor_ma_cm2'],q['peak_plateau_relative_band']);assert old==oldchannel['event'],'Original event mismatch'
                                        original_reproduced+=1;ev=evaluate_events(ref,pred,.1,q,details=True)
                                        amp=bool(oldchannel['nrmse']<=q['current_nrmse_max'] and oldchannel['nmax']<=q['current_nmax_max'] and oldchannel['signed_charge_normalized']<=q['signed_charge_error_max'] and oldchannel['absolute_charge_normalized']<=q['absolute_charge_error_max'] and oldchannel['blocked_current_passed'])
                                        assert bool(amp and original_event_pass(old,q))==oldchannel['passed'],'Original channel gate mismatch'
                                        passed=bool(amp and ev['passed']);node_ok=node_ok and passed;channels_total+=1;coverage[name+'/'+mode+'/'+arm+'/'+ev['status']]+=1
                                        key=dict(geometry=name,mode=mode,horizon_ms=horizon,case=case,seed=seed,arm=arm,node=node,channel=oldchannel['channel'])
                                        compact={k:v for k,v in ev.items() if k not in ('pairs','unmatched_reference_events','unmatched_candidate_events','threshold_ma_cm2','passed')}
                                        records.add(dict(**key,amplitude_passed=amp,original_channel_passed=oldchannel['passed'],multi_event_channel_passed=passed,original_event=old,**compact))
                                        if not passed or passed!=oldchannel['passed']:
                                            details.add(dict(**key,amplitude_passed=amp,original_channel_passed=oldchannel['passed'],multi_event_channel_passed=passed,original_event=old,multi_event=ev))
                                        if not ev['passed']:reasons[ev['status']]+=1
                                        if passed!=oldchannel['passed'] or not passed:changed.append(dict(node=node,channel=oldchannel['channel'],original_passed=oldchannel['passed'],multi_event_passed=passed,status=ev['status'],missing_events=ev['missing_events'],spurious_events=ev['spurious_events'],max_peak_error_ms=ev.get('max_peak_error_ms'),max_onset_error_ms=ev.get('max_onset_error_ms'),max_offset_error_ms=ev.get('max_offset_error_ms')))
                                    multi_current=multi_current and node_ok
                                eligible=oldrow['reference_valid'];before=bool(oldrow['current_passed'] and oldrow['compartment_passed']);after=bool(multi_current and oldrow['compartment_passed']);pool=stats[str(horizon)+'/'+mode+'/'+arm]
                                pool['total']+=1;pool['eligible']+=int(eligible);pool['original_passed']+=int(eligible and before);pool['multi_event_passed']+=int(eligible and after);pool['new_failures']+=int(eligible and before and not after);pool['recovered']+=int(eligible and after and not before)
                                row=dict(geometry=name,mode=mode,horizon_ms=horizon,case=case,seed=seed,arm=arm,reference_valid=eligible,original_passed=before,multi_event_passed=after,compartment_passed=oldrow['compartment_passed'],event_failure_types=dict(reasons),failing_or_changed_channels=changed)
                                case_rows.append(row)
                                if not after or before!=after:failed_cases.append(row)
                del predictions
                print('[47 multi-event] '+name+'/'+mode+' evaluated',flush=True)
    records.close();details.close();controls.close()
    assert len(case_rows)==len(original) and original_reproduced==channels_total
    for h in p['prefix_horizons_ms']:
        for mode in ('imposed','autonomous'):
            for arm in p['arms']:assert stats[str(h)+'/'+mode+'/'+arm]['original_passed']==original_report['summary'][str(h)][arm][mode]['passed']
    null_ok=all(v['multi_event_passed']==v['eligible'] and v['eligible']>0 for k,v in stats.items() if k.endswith('/none'))
    controls_ok=not control_failures and all(v['total']>0 and v['unexpected']==0 for v in control_stats.values())
    valid=bool(null_ok and controls_ok)
    result=dict(valid=valid,complete=True,task='47',evaluation='multi-event v1, post-hoc frozen before full corpus',archive_sha256=freeze['archive_sha256'],frozen_P_sha256=freeze['frozen_P_sha256'],training_performed=False,selection_performed=False,thresholds_changed=False,original_evaluator_modified=False,original_verdict_retained=True,original_events_and_counts_reproduced=True,case_comparisons=len(case_rows),channel_comparisons=channels_total,summary=dict(stats),event_coverage=dict(coverage),negative_controls=dict(control_stats),negative_control_skips=dict(control_skips),unexpected_controls=control_failures,formula_null_control_passed=null_ok,new_evaluator_controls_passed=controls_ok,elapsed_seconds=time.time()-start,task48_authorized=False,independent_confirmation_claimed=False,speedup_claimed=False,limitations='Same immutable saved47cohort, with prior knowledge ofCa_HVAborderline case. New timing semantics, same numerical thresholds. Threshold episodes are not every biological response; one-sample support mismatches remain failures. Censored/unsupported timing explicitly reported. Amplitude/state/native scores unchanged. No original verdict overwritten or automatic48authorization.')
    write_json(F/'case_results.json',case_rows);write_json(F/'failure_audit.json',failed_cases)
    files=['channel_results.json.gz','event_details.json.gz','saved_reference_controls.json.gz','case_results.json','failure_audit.json']
    result['files']={name:dict(bytes=(F/name).stat().st_size,sha256=hashlib.sha256((F/name).read_bytes()).hexdigest()) for name in files}
    write_json(F/'summary.json',result)
    print(json.dumps(dict(valid=valid,channel_comparisons=channels_total,negative_controls=control_stats,summary=stats)),flush=True)
if __name__=='__main__':main()
