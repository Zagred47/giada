"""Recompute Task52e rejection reasons from saved development metrics only."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NAMES=['Ca_HVA','Ca_LVAst','NaTa_t','NaTs2_t','Nap_Et2','Ih','Im','K_Pst','K_Tst','SK_E2','SKv3_1']
def main():
    folder=ROOT/'artifacts/task52e_kaggle_external';receipt=json.loads((folder/'kaggle_receipt.json').read_text());archive=folder/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        b=json.loads(z.read('validation_baseline.json'));curve=json.loads(z.read('learning_curve.json'));p=json.loads(z.read('preregistration.json'))
        B=np.array(b['channels']);T=np.array(b['trajectory']);S=np.array(b['score']);rows=[]
        for step in curve:
            C=np.array(step['channel_errors']);V=np.array(step['trajectory_error']);violC=C>B*(1+1e-6)+1e-12;violV=V>T*(1+1e-6)+1e-12
            eligible=~(violC.any(1)|violV.any(1));assert eligible.tolist()==step['eligible']
            for i,arm in enumerate(p['arms']):
                sl=slice(i*3,(i+1)*3);ratio=C[sl]/B[sl];vr=V[sl]/T[sl]
                rows.append(dict(arm=arm['id'],update=step['update'],eligible=eligible[sl].tolist(),current_mse_mean_ratio=dict(zip(NAMES,ratio.mean(0).tolist())),voltage_mse_mean_ratio=float(vr[:,0].mean()),calcium_mse_mean_ratio=float(vr[:,1].mean()),common_score_mean_ratio=float((np.array(step['score'])[sl]/S[sl]).mean()),per_seed=[dict(seed=p['seeds'][j],blockers=[NAMES[k] for k in np.flatnonzero(violC[i*3+j])]+[name for k,name in enumerate(['voltage','calcium']) if violV[i*3+j,k]],current_mse_ratio=dict(zip(NAMES,ratio[j].tolist())),voltage_mse_ratio=float(vr[j,0]),calcium_mse_ratio=float(vr[j,1])) for j in range(3)]))
        report=json.loads(z.read('final_report.json'))
        result=dict(valid=True,source_sha256=receipt['sha256'],all_rejections_reproduced=True,rows=rows,selection=json.loads(z.read('selection_freeze.json')),credit_probe=json.loads(z.read('credit_assignment_probe.json')),final_summary=report['summary'],qualification='Ratios are development MSE, not RMSE or held-out acceptance. Rejected checkpoints were not evaluated on the final coupled panel. No new training or selection performed.')
    dest=ROOT/'experiments/results/task52e_audit_v1';dest.mkdir(exist_ok=True);(dest/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        if r['arm'] in ('combined','family'):
            print(r['arm'],r['update'],[(x['seed'],x['blockers']) for x in r['per_seed']])
if __name__=='__main__':main()
