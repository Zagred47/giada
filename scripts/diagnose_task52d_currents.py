"""Trace-only residual stratification. Does not refit or select a model."""
import gzip,io,json,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES=['Ca_HVA','Ca_LVAst','NaTa_t','NaTs2_t','Nap_Et2','Ih','Im','K_Pst','K_Tst','SK_E2','SKv3_1']
def main():
    out={}
    with zipfile.ZipFile(ROOT/'artifacts/task52d_kaggle_external/task52d_final.zip') as z:
        ref=dict(np.load(io.BytesIO(z.read('reference.npz'))))
        for arm in ['frozen52cDE','trajectory']:
            trace=dict(np.load(io.BytesIO(z.read(arm+'/traces.npz'))))
            rows=json.loads(gzip.decompress(z.read(arm+'/evaluation.json.gz')))
            info={}
            for mode,idx in [('compact',4 if arm=='frozen52cDE' else 0),('compact_imposed',5 if arm=='frozen52cDE' else 1)]:
                pred=trace['current'][:,:,idx];target=ref['current'];channels={}
                for j,name in enumerate(NAMES):
                    p=pred[...,j];r=target[...,j];windows={}
                    for label,lo,hi in [('initial',0,1),('first1ms',0,10),('first10ms',0,100),('late500to1000',5000,10000),('full',0,10000)]:
                        pp=p[lo:hi];rr=r[lo:hi];den=np.sum(np.abs(rr))
                        windows[label]={'rmse':float(np.sqrt(np.mean((pp-rr)**2))),'absolute_magnitude_ratio':float(np.sum(np.abs(pp))/max(den,1e-30))}
                    selected=[r['current']['channels'][j] for r in rows if r['arm']==mode and r['horizon_ms']==1000]
                    ev=[r['current']['multi_event'][j] for r in rows if r['arm']==mode and r['horizon_ms']==1000]
                    channels[name]={'windows':windows,'missing_events':sum(e.get('missing_events',0) for e in ev),'reference_events':sum(e.get('reference_events',0) for e in ev),'timing_failure_cases':sum(e.get('status')=='timing_failure' for e in ev),'median_nmax':float(np.median([r['nmax'] for r in selected]))}
                info[mode]=channels
            out[arm]=info
        out['learning_curve']=json.loads(z.read('learning_curve.json'))
        out['qualification']='Descriptive residuals on reused evaluation panel; not independent hypothesis confirmation. Initial current error may combine initialization/readout, not solely recurrence.'
    dest=ROOT/'experiments/results/task52d_audit_v1/residuals.json';dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['trajectory'],indent=2))
if __name__=='__main__':main()
