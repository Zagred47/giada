"""Threshold-episode matching, preserving Task47 numeric acceptance tolerances."""
import numpy as np
from .task37_ionic_playground import event_metrics

def _events(signal,threshold,band):
    mask=signal>=threshold;changes=np.diff(np.r_[False,mask,False].astype(np.int8));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)-1
    events=[]
    for start,end in zip(starts,ends):
        values=signal[start:end+1];amplitude=float(values.max());peak=np.flatnonzero(values>=amplitude*(1-band))+start
        events.append(dict(start=int(start),end=int(end),peaks=peak,right_censored=bool(end==len(signal)-1)))
    return events

def _distance(left,right):
    pos=np.searchsorted(left,right);before=left[np.maximum(pos-1,0)];after=left[np.minimum(pos,len(left)-1)]
    return int(np.minimum(abs(right-before),abs(right-after)).min())

def _pair(ref,pred,dt,peak_limit,edge_limit):
    return dict(peak_error_ms=float(_distance(ref['peaks'],pred['peaks'])*dt),onset_error_ms=abs(ref['start']-pred['start'])*dt,offset_error_ms=abs(ref['end']-pred['end'])*dt,offset_censored_lower_bound=bool(ref['right_censored'] or pred['right_censored']),reference_start_index=ref['start'],reference_end_index=ref['end'],candidate_start_index=pred['start'],candidate_end_index=pred['end'])

def _align(ref,pred,dt,peak_limit,edge_limit):
    n,m=len(ref),len(pred)
    if n==m:return [dict(reference_event=i,candidate_event=i,**_pair(a,b,dt,peak_limit,edge_limit)) for i,(a,b) in enumerate(zip(ref,pred))],[],[]
    if not n or not m:return [],list(range(n)),list(range(m))
    states={(0,0):(0,0.,[])}
    def keep(key,value):
        current=states.get(key)
        if current is None or (-value[0],value[1])<(-current[0],current[1]):states[key]=value
    for i in range(n+1):
        for j in range(m+1):
            matched,cost,path=states[i,j]
            if i<n and j<m:
                row=_pair(ref[i],pred[j],dt,peak_limit,edge_limit)
                distance=row['peak_error_ms']/peak_limit+(row['onset_error_ms']+row['offset_error_ms'])/edge_limit
                keep((i+1,j+1),(matched+1,cost+distance,path+[(i,j,row)]))
            if i<n:keep((i+1,j),(matched,cost,path+[(i,None,None)]))
            if j<m:keep((i,j+1),(matched,cost,path+[(None,j,None)]))
    path=states[n,m][2]
    return [dict(reference_event=i,candidate_event=j,**row) for i,j,row in path if row is not None],[i for i,j,row in path if j is None],[j for i,j,row in path if i is None]

def evaluate_events(reference,prediction,dt,quality,details=True):
    reference=np.asarray(reference);prediction=np.asarray(prediction)
    if reference.ndim!=1 or prediction.shape!=reference.shape or not len(reference) or dt<=0:raise ValueError('Aligned nonempty one-dimensional currents required')
    if not np.isfinite(reference).all():raise ValueError('Nonfinite reference')
    if not np.isfinite(prediction).all():return dict(passed=False,status='nonfinite',reference_events=0,candidate_events=0,matched_events=0,missing_events=0,spurious_events=0)
    base=reference[0];rr=abs(reference-base);pp=abs(prediction-base);amplitude=float(rr.max());scale=max(float(abs(reference).max()),quality['current_scale_floor_ma_cm2']);floor=quality['current_scale_floor_ma_cm2'];idx=int(rr.argmax())
    supported=amplitude>=.25*scale and amplitude>=10*floor
    if not supported or idx in (0,len(rr)-1):
        spurious=not supported and float(pp.max())>=.25*scale and float(pp.max())>=10*floor
        return dict(passed=not spurious,status='spurious_on_unsupported_reference' if spurious else 'not_evaluable',reason='amplitude_unsupported' if not supported else 'global_reference_peak_at_boundary',reference_events=0,candidate_events=None,matched_events=0,missing_events=0,spurious_events=int(spurious),max_peak_error_ms=None,max_onset_error_ms=None,max_offset_error_ms=None,censored_offsets=0)
    threshold=.1*amplitude;ref=_events(rr,threshold,quality['peak_plateau_relative_band']);pred=_events(pp,threshold,quality['peak_plateau_relative_band'])
    peak_limit=quality['peak_timing_max_ms'];edge_limit=quality['onset_offset_max_ms'];pairs,missing,spurious=_align(ref,pred,dt,peak_limit,edge_limit)
    maxima={key:max((r[key] for r in pairs),default=None) for key in ('peak_error_ms','onset_error_ms','offset_error_ms')}
    temporal=all(r['peak_error_ms']<=peak_limit+1e-10 and r['onset_error_ms']<=edge_limit+1e-10 and r['offset_error_ms']<=edge_limit+1e-10 for r in pairs)
    passed=not missing and not spurious and temporal
    result=dict(passed=bool(passed),status='matched' if passed else 'event_support_mismatch' if missing or spurious else 'timing_failure',reference_events=len(ref),candidate_events=len(pred),matched_events=len(pairs),missing_events=len(missing),spurious_events=len(spurious),max_peak_error_ms=maxima['peak_error_ms'],max_onset_error_ms=maxima['onset_error_ms'],max_offset_error_ms=maxima['offset_error_ms'],censored_offsets=sum(r['offset_censored_lower_bound'] for r in pairs))
    if details:result.update(pairs=pairs,unmatched_reference_events=missing,unmatched_candidate_events=spurious,threshold_ma_cm2=threshold)
    return result

def amplitude_pass(reference,prediction,dt,quality):
    scale=max(float(abs(reference).max()),quality['current_scale_floor_ma_cm2']);delta=prediction-reference
    return bool(np.sqrt(np.mean(delta**2))/scale<=quality['current_nrmse_max'] and abs(delta).max()/scale<=quality['current_nmax_max'] and abs(delta.sum())/(len(reference)*scale)<=quality['signed_charge_error_max'] and abs((abs(prediction)-abs(reference)).sum())/(len(reference)*scale)<=quality['absolute_charge_error_max'])

def original_event_pass(event,quality):
    return event['status']=='not_evaluable' or event['status']=='evaluable' and event['peak_error_ms']<=quality['peak_timing_max_ms']+1e-10 and max(event['onset_error_ms'],event['offset_error_ms'])<=quality['onset_offset_max_ms']+1e-10
