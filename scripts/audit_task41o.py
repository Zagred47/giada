"""Read-only forensic audit of the completed41n stress failures and memory."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41n_independent_confirmation as n
from research_memory.payload_store import open_readonly
from research_memory.mirror import write_json

def main():
    path=ROOT/'artifacts/task41n_recovery_kaggle_external/task41n_recovery_final.zip'
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest=='6d2557b43a3026b35a6976a26b8ee33f1008ba6d61a57772a7d17e9f3f363a63'
    spec=n.contract(ROOT)[0];meta=n.stress_data(spec)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        with np.load(io.BytesIO(z.read('fresh_stress_inputs.npz'))) as a:data=dict(a)
        with np.load(io.BytesIO(z.read('stress_dual.npz'))) as a:gates=a['gates']
        measured=json.loads(z.read('fresh_stress_results.json'))['dual']['rows']
    v,truth=data['voltage'],data['target_gates'];target=n.p.readout(v,truth,data['multipliers'],n.p.ionic.REVERSALS)[1]
    failures=[];rescues={};local_numeric_differences=0
    for s,seed in enumerate(spec['seeds']):
        pred=n.p.readout(v,gates[:,s],data['multipliers'],n.p.ionic.REVERSALS)[1]
        for k,row in enumerate(meta):
            original=measured[s*len(meta)+k]['metrics']
            check=n.p.evaluate_currents(target[:,k],pred[:,k],.1,spec['quality'],truth[:,k],gates[:,s,k])
            if check!=original:
                def differences(a,b,path=''):
                    if isinstance(a,dict):
                        return sum((differences(a[key],b[key],path+'/'+key) for key in a),[])
                    if isinstance(a,list):return sum((differences(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
                    if isinstance(a,(float,int)) and not isinstance(a,bool):
                        return [] if np.isclose(a,b,rtol=1e-12,atol=1e-20) else [(path,a,b)]
                    return [] if a==b else [(path,a,b)]
                delta=differences(check,original)
                local_numeric_differences+=len(delta)
                assert check['passed']==original['passed'] and [c['passed'] for c in check['channels']]==[c['passed'] for c in original['channels']],('classification mismatch',s,k)
            if check['passed']:continue
            channels=[]
            for j,c in enumerate(check['channels']):
                if c['passed']:continue
                rr=target[:,k,j];pp=pred[:,k,j];base=rr[0];amp=abs(rr-base).max();level=.1*amp
                ri=np.flatnonzero(abs(rr-base)>=level);pi=np.flatnonzero(abs(pp-base)>=level)
                crossing=int(ri[-1]);slope=float(abs(rr[min(crossing+1,len(rr)-1)]-rr[max(0,crossing-1)])/ .2)
                channels.append(dict(channel=c['channel'],metrics=c,reference_offset_ms=float(crossing*.1),prediction_offset_ms=float(pi[-1]*.1),
                    crossing_slope_ma_cm2_per_ms=slope,current_error_at_crossing_ma_cm2=float(pp[crossing]-rr[crossing]),
                    linearized_timing_error_ms=None if slope==0 else float(abs(pp[crossing]-rr[crossing])/slope)))
            failures.append(dict(model_seed=seed,case=k,**{x:row[x] for x in ('family','input_seed','history','panel')},channels=channels,gate=check['gate']))
        # Replace one channel's state only, not all currents: diagnostic oracle.
        for j,name in enumerate(n.p.ionic.CHANNELS):
            patched=gates[:,s].copy();lo,hi=n.p.OFFSETS[j:j+2];patched[...,lo:hi]=truth[...,lo:hi]
            current=n.p.readout(v,patched,data['multipliers'],n.p.ionic.REVERSALS)[1]
            passes=[n.p.evaluate_currents(target[:,k],current[:,k],.1,spec['quality'],truth[:,k],patched[:,k])['passed'] for k in range(len(meta))]
            rescues.setdefault(name,[]).extend(passes)
    query="SELECT stable_code,name FROM v_records WHERE table_key='findings' AND (stable_code LIKE '%41m%' OR stable_code LIKE '%41l%' OR stable_code LIKE '%39j%') ORDER BY stable_code LIMIT 30"
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:
        patterns=[dict(r) for r in con.execute(query)]
    report=dict(valid=True,task='41o',source_sha256=digest,baseline_passed=72-len(failures),total=72,failures=failures,
        single_channel_oracle_passed={k:sum(v) for k,v in rescues.items()},memory_query=query,memory_matches=patterns,
        local_readout_numeric_differences=local_numeric_differences,archived_classifications_reproduced=True,
        interpretation='Post-hoc Windows/NumPy localization, not a released model or independent confirmation. Float32 readout reductions can differ across platforms; all archived pass/fail predicates reproduced. GPU pinned replay required for release. Oracle channel swaps diagnose necessity/sufficiency only on this opened panel.',thresholds_changed=False)
    dest=ROOT/'experiments/results/task41o_causal_v1';dest.mkdir(exist_ok=True)
    write_json(dest/'audit.json',report)
    print(json.dumps(dict(baseline=report['baseline_passed'],single_channel_oracle_passed=report['single_channel_oracle_passed'],memory_matches=len(patterns))))
if __name__=='__main__':main()
