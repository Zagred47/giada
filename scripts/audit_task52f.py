"""Audit completed52f without rerunning training, selection or evaluation."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
NAMES=['Ca_HVA','Ca_LVAst','NaTa_t','NaTs2_t','Nap_Et2','Ih','Im','K_Pst','K_Tst','SK_E2','SKv3_1']
def main():
    folder=ROOT/'artifacts/task52f_resume_colab_external';receipt=json.loads((folder/'external_backup_receipt.json').read_text());archive=Path(receipt['local'])
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
    assert receipt['remote']['label']=='final'
    checks=[]
    def visit(value,path):
        if torch.is_tensor(value):
            okay=bool(torch.isfinite(value).all());assert okay,path;checks.append(path)
        elif isinstance(value,dict):
            for k,v in value.items():visit(v,path+'/'+str(k))
        elif isinstance(value,(list,tuple)):
            for k,v in enumerate(value):visit(v,path+'/'+str(k))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.pt'):visit(torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True),name)
        report=json.loads(z.read('final_report.json'));p=json.loads(z.read('preregistration.json'));freeze=json.loads(z.read('selection_freeze.json'))
        assert report['complete'] and report['training_updates']==800 and report['resumed_from']==600 and report['new_training_updates']==200
        baseline=json.loads(z.read('validation_baseline.json'));B=np.array(baseline['channels']);T=np.array(baseline['trajectory']);S=np.array(baseline['score']);curve=json.loads(z.read('learning_curve.json'));rows=[]
        for point in curve:
            C=np.array(point['channel_errors']);V=np.array(point['trajectory_error']);vc=C>B*(1+1e-6)+1e-12;vv=V>T*(1+1e-6)+1e-12;eligible=~(vc.any(1)|vv.any(1));assert eligible.tolist()==point['eligible']
            for i,arm in enumerate(p['arms']):
                sl=slice(i*3,(i+1)*3);ratio=C[sl]/B[sl];vr=V[sl]/T[sl]
                rows.append(dict(arm=arm['id'],update=point['update'],eligible=eligible[sl].tolist(),score_ratio=float((np.array(point['score'])[sl]/S[sl]).mean()),current_mse_ratio=dict(zip(NAMES,ratio.mean(0).tolist())),voltage_mse_ratio=float(vr[:,0].mean()),calcium_mse_ratio=float(vr[:,1].mean()),per_seed=[dict(seed=p['seeds'][j],blockers=[NAMES[k] for k in np.flatnonzero(vc[i*3+j])]+[name for k,name in enumerate(['voltage','calcium']) if vv[i*3+j,k]]) for j in range(3)]))
        arms={}
        for arm in p['arms']:
            name=arm['id'];points=[r for r in rows if r['arm']==name]
            arms[name]=dict(last=points[-1],best_score_diagnostic=min(points,key=lambda r:r['score_ratio']),eligible_seed_checkpoints=sum(sum(r['eligible']) for r in points),selected=report['summary'][name],cost=json.loads(z.read(name+'/cost.json')))
            assert report['summary'][name]=={k:v for k,v in report['summary']['frozen52d_trajectory'].items() if k in report['summary'][name]}
        result=dict(valid=True,backup=receipt,finite_tensor_count=len(checks),finite_pt_files=sorted(set(c.split('/')[0] for c in checks)),finite_parameters_independently_verified=True,completion={k:report[k] for k in ('complete','training_updates','resumed_from','new_training_updates','scientific_acceptance_passed','thresholds_changed','abstraction_changed')},all_selected_fallback=all(freeze['baseline_fallback']),freeze=freeze,arms=arms,rows=rows,encoder_calibration=json.loads(z.read('encoder_calibration.json')),credit_probe=json.loads(z.read('credit_assignment_probe.json')),controls=report['summary']['frozen52d_trajectory'],qualification='Development ratios are MSE, not RMSE. Diagnostic best is not an eligible selected candidate. Coupled panel evaluates baseline fallbacks; no unseen-final-checkpoint performance is inferred.')
    dest=ROOT/'experiments/results/task52f_audit_v1';dest.mkdir(exist_ok=True);(dest/'audit.json').write_text(json.dumps(result,indent=2))
    for name,arm in arms.items():
        print(name,json.dumps({k:arm['last'][k] for k in ('score_ratio','voltage_mse_ratio','calcium_mse_ratio','current_mse_ratio','per_seed')}));print('best',arm['best_score_diagnostic']['update'],arm['best_score_diagnostic']['score_ratio'],'eligible',arm['eligible_seed_checkpoints'],'cost',arm['cost'])
    print('finite',len(checks),'calibration',result['encoder_calibration'],'probe',result['credit_probe'])
if __name__=='__main__':main()
