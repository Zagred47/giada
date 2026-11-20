import gzip,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/task52_kaggle_external';archive=folder/'task52_final.zip'
receipt=json.loads((folder/'kaggle_receipt.json').read_text())
original_receipt=json.loads((folder/'kaggle_receipt.original.json').read_text()) if (folder/'kaggle_receipt.original.json').exists() else receipt
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
assert archive.stat().st_size==receipt['bytes']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    report=json.loads(z.read('final_report.json'));plan=json.loads(z.read('preregistration.json'));cases=json.loads(z.read('cases.json'));rows=json.loads(gzip.decompress(z.read('evaluation.json.gz')))
    responses=json.loads(gzip.decompress(z.read('interventions.json.gz')))
    finite={}
    for name in ('P.pt','C1.pt','R41.pt'):
        state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True);finite[name]=all(bool(torch.isfinite(v).all()) for v in state.values())
    with np.load(io.BytesIO(z.read('traces.npz'))) as arrays:
        traces={k:arrays[k] for k in arrays.files}
    arm_summary={}
    for arm in plan['arms']:
        rr=[r for r in rows if r['arm']==arm and r['horizon_ms']==1000]
        family={}
        for fam,indices in plan['families'].items():
            cells=[r['current']['channels'][k] for r in rr for k in indices]
            family[fam]=dict(median_nrmse=float(np.median([c['nrmse'] for c in cells])),max_nrmse=max(c['nrmse'] for c in cells),passed=sum(c['passed'] for c in cells),total=len(cells))
        arm_summary[arm]=dict(summary=report['summary'][arm],calcium_pooled_rmse_mM=float(np.sqrt(np.mean([r['calcium_rmse_mM']**2 for r in rr]))),families=family,intervention_passed=sum(r['current']['conservative_passed'] for r in responses if r['arm']==arm and r['horizon_ms']==1000))
    pair={}
    for left,right in [('canonical_J','canonical64'),('P_J','P'),('compact_reset','compact')]:
        l=plan['arms'].index(left);r=plan['arms'].index(right)
        pair[left+'__'+right]={k:dict(rmse=float(np.sqrt(np.mean((traces[k][:,:,l].astype(float)-traces[k][:,:,r])**2))),max_abs=float(abs(traces[k][:,:,l].astype(float)-traces[k][:,:,r]).max())) for k in ('voltage','calcium','current')}
    cost=json.loads(z.read('cost.json'))
assert all(finite.values()) and all(np.isfinite(a).all() for a in traces.values())
out=dict(valid=True,archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'],finite_saved_parameters=finite,all_traces_finite=True,weights_unchanged=report['weights_unchanged'],arm_summary=arm_summary,paired_differences=pair,cost=cost,receipt_corrected='metadata_correction' in receipt,wrapper_metadata_discrepancy=dict(receipt_updates=original_receipt['training_updates_performed'],scientific_updates=report['training_updates'],receipt_selection=original_receipt['selection_recomputed'],scientific_selection=report['selection_recomputed']))
dest=ROOT/'experiments/results/task52_audit_v1';dest.mkdir(exist_ok=True);(dest/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
for arm,a in arm_summary.items():
    r=a['summary']['1000'];print(arm,'V',r['pooled_voltage_rmse'],'Ca',a['calcium_pooled_rmse_mM'],'passed',r['passed'],'current_nrmse',{k:round(v['median_nrmse'],5) for k,v in a['families'].items()},'interventions',a['intervention_passed'])
print('PAIRS',pair);print('COST',cost);print('METADATA',out['wrapper_metadata_discrepancy'])
