"""Post-replay interpretation of new SK_E2 threshold episodes; no rule change."""
import io,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.multi_event_evaluation import _events
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task47_multi_event_v1'
if __name__=='__main__':
    q=json.loads((ROOT/'experiments/task47_preregistration_v1.json').read_text())['quality'];archive=ROOT/'artifacts/task47_retry_external/task47_final.zip'
    with zipfile.ZipFile(archive) as z:
        with np.load(io.BytesIO(z.read('reference_47_chain5.npz'))) as src:ref=src['current'][:,0,0,9,4,9].copy()
        with np.load(io.BytesIO(z.read('traces_47_chain5_autonomous.npz'))) as src:pred=src['current'][:,:,:,9,4,9].copy()
    base=ref[0];rr=abs(ref-base);threshold=.1*float(rr.max());re=_events(rr,threshold,q['peak_plateau_relative_band']);out=[]
    for si,seed in enumerate((17,29,43)):
        for ai,arm in enumerate(('none','C1','P','R41')):
            pp=abs(pred[:,si,ai]-base);events=_events(pp,threshold,q['peak_plateau_relative_band']);different=np.flatnonzero((rr>=threshold)!=(pp>=threshold))
            out.append(dict(seed=seed,arm=arm,reference_events=[dict(start=e['start'],end=e['end'],samples=e['end']-e['start']+1) for e in re],candidate_events=[dict(start=e['start'],end=e['end'],samples=e['end']-e['start']+1) for e in events],mask_disagreement_samples=different.tolist(),threshold_ma_cm2=threshold,max_abs_error_ma_cm2=float(abs(ref-pred[:,si,ai]).max()),reference_margin_near_onset_ma_cm2=(rr[1337:1346].astype('float64')-threshold).tolist(),candidate_margin_near_onset_ma_cm2=(pp[1337:1346].astype('float64')-threshold).tolist()))
    write_json(F/'new_support_audit.json',dict(valid=True,post_hoc_interpretation_only=True,source='47_chain5,case9,distalnode4,autonomous,SK_E2',sample_indices_near_onset=list(range(1337,1346)),dt_ms=.1,rows=out,rule_or_threshold_changed=False,limitation='Spurious means extra threshold-connected component, not independently established new biological response. Frozen-v1counts retained; no smoothing/gap merging introduced after outcome inspection.'))
    print(json.dumps([r for r in out if r['arm']=='P']),flush=True)
