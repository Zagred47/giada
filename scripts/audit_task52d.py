"""Read-only audit of frozen Task52d artifacts and paired Task52c DE control."""
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True

def main():
    folder = ROOT / 'artifacts/task52d_kaggle_external'
    receipt = json.loads((folder / 'kaggle_receipt.json').read_text())
    archive = folder / receipt['archive']
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == receipt['sha256'] and archive.stat().st_size == receipt['bytes']
    result = {'archive_sha256': digest, 'archive_bytes': archive.stat().st_size}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        result['report'] = {k:v for k,v in json.loads(z.read('final_report.json')).items() if k != 'summary'}
        result['checkpoint_finiteness'] = {name:finite(torch.load(io.BytesIO(z.read(name)), map_location='cpu', weights_only=True)) for name in z.namelist() if name.endswith('.pt')}
        assert all(result['checkpoint_finiteness'].values())
        result['selection'] = json.loads(z.read('selection_freeze.json'))
        ref = dict(np.load(io.BytesIO(z.read('reference.npz'))))
        result['arms'] = {}
        for name in ['frozen52cDE','control','balance','trajectory','combined']:
            trace = dict(np.load(io.BytesIO(z.read(name+'/traces.npz'))))
            assert all(np.isfinite(v).all() for v in trace.values())
            rows = json.loads(gzip.decompress(z.read(name+'/evaluation.json.gz')))
            modes = {}
            for mode in sorted({r['arm'] for r in rows}):
                modes[mode] = {}
                for horizon in [100,500,1000]:
                    rr = [r for r in rows if r['arm']==mode and r['horizon_ms']==horizon]
                    modes[mode][horizon] = {'passed':sum(r['passed'] for r in rr),'total':len(rr),'voltage_rmse_mv':float(np.sqrt(np.mean([r['voltage_rmse_mv']**2 for r in rr]))),'calcium_rmse_mM':float(np.sqrt(np.mean([r['calcium_rmse_mM']**2 for r in rr])))}
            channels = {}
            for mode in ['compact','compact_imposed']:
                rr = [r for r in rows if r['arm']==mode and r['horizon_ms']==1000]
                channels[mode] = {str(k):{'median_nrmse':float(np.median([r['current']['channels'][k]['nrmse'] for r in rr])),'passed':sum(r['current']['channels'][k]['passed'] for r in rr)} for k in range(11)}
            idx = 5 if name=='frozen52cDE' else 1
            err = trace['current'][:,:,idx].astype(float)-ref['current']
            net = err.sum(-1)
            by_case = [{'seed':r['seed'],'case':r['case'],'protocol':r['protocol'],'intervention':r['intervention'],'voltage_rmse_mv':r['voltage_rmse_mv']} for r in rows if r['arm']=='compact' and r['horizon_ms']==1000]
            cost = json.loads(z.read(name+'/cost.json'))['samples_ms']
            result['arms'][name] = {'modes':modes,'channels':channels,'imposed_net_current_rmse':float(np.sqrt(np.mean(net**2))),'imposed_channel_error_rss':float(np.sqrt(np.mean(err**2,axis=(0,1,2)).sum())),'cases':by_case,'cost_median_ms':{k:float(np.median(v)) for k,v in cost.items()}}
        base = result['arms']['frozen52cDE']
        for name in ['control','balance','trajectory','combined']:
            arm = result['arms'][name]
            delta = [a['voltage_rmse_mv']-b['voltage_rmse_mv'] for a,b in zip(arm['cases'],base['cases'])]
            arm['vs_52cDE']={'voltage_improved_cases':sum(d < -1e-9 for d in delta),'voltage_regressed_cases':sum(d > 1e-9 for d in delta),'max_case_worsening_mv':max(delta),'net_current_relative_change':arm['imposed_net_current_rmse']/base['imposed_net_current_rmse']-1}
    dest = ROOT/'experiments/results/task52d_audit_v1'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'V':v['modes']['compact'],'net':v['imposed_net_current_rmse'],'delta':v.get('vs_52cDE')} for k,v in result['arms'].items()},indent=2))

if __name__ == '__main__':
    main()
