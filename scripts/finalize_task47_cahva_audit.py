"""Quantify threshold-identity instability; diagnostic only, no gate waiver."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from audit_task47_cahva import F,describe
from src.giada_teacher.task37_ionic_playground import event_metrics
from src.giada_teacher.task47_independent_confirmation import design
from research_memory.mirror import write_json

def local_peaks(ref,pred,starts,dt=.1):
    """Diagnostic pulse windows fixed by input, not tuned on candidate output."""
    result=[];base=float(ref[0]);rr=abs(ref.astype('float64')-base);pp=abs(pred.astype('float64')-base);global_amp=rr.max();threshold=.1*global_amp
    for index,start in enumerate(starts):
        end=int(starts[index+1]) if index+1<len(starts) else len(ref)
        ri=int(start+rr[start:end].argmax());pi=int(start+pp[start:end].argmax())
        result.append(dict(pulse_index=index,start_index=int(start),end_index_exclusive=end,reference_peak_index=ri,candidate_peak_index=pi,peak_error_ms=abs(ri-pi)*dt,reference_peak_fraction=float(rr[ri]/global_amp),candidate_peak_fraction=float(pp[pi]/global_amp),reference_above_original_threshold=bool(rr[ri]>=threshold),candidate_above_original_threshold=bool(pp[pi]>=threshold),signed_peak_amplitude_error_ma_cm2=float(pred[pi]-ref[ri]),candidate_relative_peak_amplitude_error=float(abs(pp[pi]-rr[ri])/max(rr[ri],1e-7)),diagnostic_only=True))
    return result

def main():
    p=json.loads((ROOT/'experiments/task47_preregistration_v1.json').read_text());initial=json.loads((F/'trace_audit_initial.json').read_text());q=p['quality']
    with np.load(F/'case8_node4_evidence.npz') as n:evidence=dict(n)
    ref=evidence['reference_current'][:,0];rr=abs(ref.astype('float64')-float(ref[0]));threshold=.1*rr.max()
    rows,injection,_=design(p['geometries'][0],p);assert rows[8]['protocol']==initial['case'];inj=injection[:,8,4]
    starts=np.flatnonzero((inj[1:]>0)&(inj[:-1]==0))+1;starts=starts[starts*.1>20]
    samples=[];counterfactuals=[];matching=[];rounding=[]
    for mode in ('imposed','autonomous'):
        for si,seed in enumerate((17,29,43)):
            pred=evidence[mode+'_current'][:,si,2,0]
            local=local_peaks(ref,pred,starts);matching.append(dict(mode=mode,seed=seed,pulses=local))
            # Compare numeric precision without altering the threshold fraction.
            for dtype in ('float32','float64'):
                old=event_metrics(ref.astype(dtype),pred.astype(dtype),.1,1e-7,.001);rounding.append(dict(mode=mode,seed=seed,dtype=dtype,event=old))
            if seed!=17:continue
            d=describe(ref,pred);different=d['active_mask_disagreement_indices'];assert different==[1983]
            index=different[0];actual=float(pred[index]);reference=float(ref[index]);margin_ref=float(rr[index]-threshold);margin_pred=abs(actual-float(ref[0]))-threshold
            samples.append(dict(mode=mode,index=index,time_ms=index*.1,reference_current_ma_cm2=reference,candidate_current_ma_cm2=actual,absolute_error_ma_cm2=abs(actual-reference),original_threshold_ma_cm2=threshold,reference_threshold_margin_ma_cm2=margin_ref,candidate_threshold_margin_ma_cm2=margin_pred,reference_fraction_of_global_peak=float(rr[index]/rr.max()),candidate_fraction_of_global_peak=float(abs(actual-float(ref[0]))/rr.max()),weak_peak_relative_error=float(abs(actual-reference)/rr[index]),reference_active_samples_in_last_island=1,float32_ulp_ma_cm2=float(abs(np.spacing(ref[index]))),absolute_error_in_reference_float32_ULPs=float(abs(actual-reference)/abs(np.spacing(ref[index])))))
            restored=pred.copy();restored[index]=ref[index]
            event=event_metrics(ref,restored,.1,1e-7,.001);assert event['offset_error_ms']==0
            counterfactuals.append(dict(mode=mode,intervention='Diagnostic only: replace candidate sample1983 with saved reference value. No saved rollout/checkpoint altered.',changed_samples=1,maximum_intervention_ma_cm2=float(abs(actual-reference)),event_after=event))
    assert all(r['event'].get('offset_error_ms')==44.900000000000006 for r in rounding if r['seed']==17)
    assert all(v['pulses'][-1]['reference_peak_index']==v['pulses'][-1]['candidate_peak_index']==1983 for v in matching)
    assert np.array_equal(evidence['imposed_current'][:,0,1,0],evidence['imposed_current'][:,0,2,0]) and np.array_equal(evidence['imposed_current'][:,0,2,0],evidence['imposed_current'][:,0,3,0])
    # The signal at the borderline still exists; a real local delay is detectable.
    last=int(starts[-1]);shifted=evidence['imposed_current'][:,0,2,0].copy();old=shifted.copy();shifted[last+10:]=old[last:-10];shifted[last:last+10]=old[last]
    shifted_probe=local_peaks(ref,shifted,starts)[-1];assert shifted_probe['peak_error_ms']==1.
    nearest=F/'case8_node4_evidence.npz'
    final=dict(valid=True,task='47',audit_kind='Saved-trace forensic, not a new confirmation or evaluator replacement',archive_sha256=initial['archive_sha256'],evidence_npz_sha256=hashlib.sha256(nearest.read_bytes()).hexdigest(),original_failures_reproduced=True,official_task47_verdict_unchanged='NO_GO:143/144at500and1000ms per mode',thresholds_changed=False,training_performed=False,selection_performed=False,model_outputs_modified=False,evaluator_code_modified=False,
        numerical_definition=dict(dt_ms=.1,threshold_fraction_of_reference_global_transient=.1,onset_offset_limit_ms=q['onset_offset_max_ms'],peak_limit_ms=q['peak_timing_max_ms'],sample_clock='current[k] is the saved ionic update stage at V[k], gates[k+1]; the original evaluator labels k*dt. No relative time shift applied.'),
        factual_diagnosis='Small real amplitude approximation at the final weak response crosses the global10% classification boundary. Global last-active-index then compares different response islands (198.3vs153.4ms), inflating that support difference into44.9ms of apparent offset error. The weak response peak is at198.3ms in both model and reference; not a44.9ms physical relaxation delay. Not solely floating-point rounding.',
        evaluator_scope_issue='event_metrics explicitly assumes one transient/window, but Task47 passes an entire irregular eight-pulse burst. It returns an aggregate envelope, not matched-event offsets. The arithmetic matches the registered rule; this audit does not waive its failure.',
        inherited_control='For seed17 imposed Ca_HVA current, C1/P/R41 are bitwise identical at every sample. Thus the amplitude bias is inherited rather than introduced by P repair or axial coupling; failure is already present at coupling0 and imposed drive.',
        actionable_next_step='Keep original report and143/144verdict. Prospectively specify multi-event matching and separate threshold-support mismatch from temporal displacement, with known-delay/amplitude controls and unchanged numerical tolerances; re-evaluate saved traces before any further training. Do not retrospectively grant144/144 or authorize48.',
        threshold_samples=samples,per_pulse_diagnostics=matching,precision_controls=rounding,single_sample_counterfactuals=counterfactuals,known_local_delay_negative_control=shifted_probe,assumptions=['Per-pulse windows are fixed from the existing input pulse start to the next pulse start, not tuned on model output; diagnostic only.','Same local peak times do not prove all waveform shapes or all biological behaviors equivalent.','The small amplitude error is real; the declared aggregate timing gate still fails.'])
    write_json(F/'audit.json',final)
    # Queryable exact evidence for the critical traces; original signed currents preserved.
    write_json(F/'critical_traces.json',dict(dt_ms=.1,units='mA/cm2, signed inward currents negative',reference=ref.tolist(),imposed_seed17=evidence['imposed_current'][:,0,2,0].tolist(),autonomous_seed17=evidence['autonomous_current'][:,0,2,0].tolist(),injection_ma_cm2=inj.tolist(),all_sample_times='sample_index*0.1ms'))
    plot(evidence,threshold,starts)
    print(json.dumps(dict(diagnosis=final['factual_diagnosis'],threshold_samples=samples,matched_peak_errors=[(r['mode'],r['seed'],[v['peak_error_ms'] for v in r['pulses']]) for r in matching])),flush=True)

def plot(evidence,threshold,starts):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref=evidence['reference_current'][:,0];pred=evidence['imposed_current'][:,0,2,0];auto=evidence['autonomous_current'][:,0,2,0];time=np.arange(len(ref))*.1;base=float(ref[0])
    fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained')
    axes[0].plot(time,ref*1e6,label='Reference');axes[0].plot(time,pred*1e6,'--',label='P imposed');axes[0].plot(time,auto*1e6,':',label='P autonomous');axes[0].set(xlim=(80,220),ylabel='Signed Ca_HVA current (nA/cm²)',title='Saved Task47: case8, distal node4, seed17, sodium blocked, coupling0')
    for start in starts:axes[0].axvline(start*.1,color='grey',alpha=.15)
    axes[0].legend()
    for value,name,style in ((ref,'Reference','-'),(pred,'P imposed','--'),(auto,'P autonomous',':')):axes[1].plot(time,(abs(value.astype('float64')-base)-threshold)*1e9,style,label=name)
    axes[1].axhline(0,color='black',lw=.8);axes[1].set(xlim=(197.9,198.7),ylim=(-100,35),ylabel='Margin above 10% threshold (pA/cm²)',title='One borderline reference sample above threshold; local peaks occur together')
    for j,value in enumerate((ref,pred,auto)):
        mask=abs(value-base)>=.1*float(abs(ref-ref[0]).max());axes[2].plot(time,np.where(mask,j+1,np.nan),'|',label=('Reference','P imposed','P autonomous')[j])
    axes[2].set(xlim=(95,205),ylim=(.5,3.5),yticks=[1,2,3],yticklabels=['Reference','P imposed','P autonomous'],xlabel='Time (ms)',title='Original global threshold mask: final active sample jumps between separate responses')
    fig.savefig(F/'cahva_threshold_audit.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
