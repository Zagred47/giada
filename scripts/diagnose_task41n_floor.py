"""Read-only decomposition of saved canonical floor failures; no selection."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from src.giada_teacher import task41n_independent_confirmation as n
ROOT=Path(__file__).resolve().parents[1]
def main():
    receipt=json.loads((ROOT/'artifacts/task41n_primary_external/external_backup_receipt.json').read_text());archive=Path(receipt['local'])
    with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
    spec,cfg,calcium,base=n.contract(ROOT);cfg=n.old_native.x.calibration_config(spec,cfg)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        meta=json.loads(z.read('cases.json'))
        def load(name):
            with np.load(io.BytesIO(z.read(name))) as a:return dict(a)
        data=load('native_shadow_traces.npz')
        refs={policy:load('reference_autonomous_'+policy+'.npz') for policy in ('primary','voltage_solve64')}
        target=n.t.numpy_reference(data,cfg,calcium,base)
        floors=json.loads(z.read('canonical_floors.json'));failed=[]
        actions=[(k,r) for k,r in enumerate(meta) if r['intervention']!='none']
        for floor in floors:
            if floor['mode']!='autonomous':continue
            ref=refs[floor['policy']]
            for index,metric in enumerate(floor['rows']):
                if metric['passed']:continue
                assert index>=len(meta),'Unexpected individual trajectory failure'
                case,row=actions[index-len(meta)];source=int(row['source_index'])
                channels=[]
                for ch,value in enumerate(metric['channels']):
                    if value['passed']:continue
                    expected=target['current'][:,case,ch]-target['current'][:,source,ch]
                    actual=ref['current'][:,0,case,ch]-ref['current'][:,0,source,ch]
                    mixed=refs['voltage_solve64']['current'][:,0,case,ch]-refs['voltage_solve64']['current'][:,0,source,ch]
                    err=actual-expected;worst=int(np.argmax(np.abs(err)))
                    channels.append(dict(channel=value['channel'],nrmse=value['nrmse'],nmax=value['nmax'],
                        scale=value['scale_ma_cm2'],rmse=value['rmse_ma_cm2'],normalization_floor=spec['quality'],
                        target_response_peak=float(np.max(np.abs(expected))),original_error_peak=float(np.max(np.abs(err))),
                        peak_error_tick=worst,peak_error_time_ms=.1*worst,
                        mixed_rmse=float(np.sqrt(np.mean((mixed-expected)**2))),
                        response_current_inputs_finite=bool(np.isfinite([*expected,*actual,*mixed]).all())))
                trajectories={}
                for policy,rr in refs.items():
                    trajectories[policy]={}
                    for label,ix in [('baseline',source),('intervention',case)]:
                        trajectories[policy][label]={key:dict(rmse=float(np.sqrt(np.mean((rr[key][:,0,ix]-target[key][:,ix])**2))),max_abs=float(np.max(np.abs(rr[key][:,0,ix]-target[key][:,ix])))) for key in ('voltage','calcium','gates','current')}
                recomputed=n.p.evaluate_intervention(target['current'][:,source],target['current'][:,case],ref['current'][:,0,source],ref['current'][:,0,case],.1,spec['quality'])
                assert recomputed['passed']==metric['passed']
                numeric_diffs=[]
                def compare(a,b,path=''):
                    if isinstance(a,dict):
                        assert a.keys()==b.keys()
                        for key in a:compare(a[key],b[key],path+'/'+str(key))
                    elif isinstance(a,list):
                        assert len(a)==len(b)
                        for j,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(j))
                    elif isinstance(a,(float,int)) and not isinstance(a,bool):
                        if a!=b:
                            numeric_diffs.append(dict(path=path,saved=b,recomputed=a,absolute_difference=abs(a-b),relative_difference=abs(a-b)/max(abs(a),abs(b),1e-300)))
                    else:assert a==b,(path,a,b)
                compare(recomputed,metric)
                failed.append(dict(floor_index=index,case=case,source=source,metadata=row,channels=channels,trajectory_errors=trajectories,evaluator_pass_readback_equal=True,cross_environment_metric_differences=numeric_diffs))
        report=dict(valid=True,archive_sha256=receipt['remote']['sha256'],diagnostic_only=True,training_updates=0,thresholds_changed=False,
            canonical_floors=[dict(mode=x['mode'],policy=x['policy'],passed=x['passed'],total=len(x['rows']),failed=sum(not r['passed'] for r in x['rows'])) for x in floors],failures=failed,
            input_dtypes={k:str(v.dtype) for k,v in data.items()},
            interpretation='Failures are intervention differences, not individual trajectories. Canonical equations in FP32 vs NumPy64; exact metric readback, same causal data/time grid, mixed64 control. Need separate diagnostic completion from original confirmation validity.')
        out=ROOT/'experiments/results/task41n_audited_v1/floor_diagnosis.json';out.write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(failures=[dict(case=r['case'],source=r['source'],metadata=r['metadata'],channels=r['channels'],trajectory_errors=r['trajectory_errors']) for r in failed]),indent=2))
if __name__=='__main__':main()
