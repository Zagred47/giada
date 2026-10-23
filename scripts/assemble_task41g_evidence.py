"""Assemble available 41g evidence without inventing a complete final archive."""
import hashlib,io,json,statistics,zipfile
from pathlib import Path

R=Path(__file__).resolve().parents[1];D=R/'artifacts/task41g_primary_external';O=R/'experiments/results/task41g_audited_v1'
def sha(b):return hashlib.sha256(b).hexdigest()
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def finite(torch,x):
    if isinstance(x,torch.Tensor):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(torch,v) for v in x.values())
    if isinstance(x,(tuple,list)):return all(finite(torch,v) for v in x)
    return True
def main():
    O.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((D/'external_backup_receipt.json').read_text());path=Path(receipt['local'])
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
    salv=D/'salvaged/recovery_final.zip.terminal.partial'
    salvage=json.loads((D/'salvage_receipt.json').read_text())
    final_salv=next(x for x in salvage if Path(x['source']).name=='recovery_final.zip.terminal.partial')
    recovered={x['member']:x for x in final_salv['members']}
    manifest=[];pt=[]
    import torch
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        names=set(z.namelist())
        for info in z.infolist():
            if info.filename.endswith('.json'):
                raw=z.read(info.filename)
                manifest.append(dict(member=info.filename,source='archive',archive=str(path),bytes=len(raw),sha256=sha(raw)))
            elif info.filename.endswith('.pt'):
                raw=z.read(info.filename);v=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                okay=finite(torch,v);assert okay
                pt.append(dict(member=info.filename,sha256=sha(raw),bytes=len(raw),all_tensors_finite=okay));del raw,v
        for name in sorted(recovered):
            if name in names:continue
            raw=(salv/name).read_bytes();assert len(raw)==recovered[name]['bytes']
            json.loads(raw)
            manifest.append(dict(member=name,source='salvaged_crc_valid_member',path=str(salv/name),bytes=len(raw),sha256=sha(raw)))
        freeze=json.loads(z.read('selection_freeze.json'));claimed=freeze.pop('freeze_sha256')
        assert sha(json.dumps(freeze,sort_keys=True).encode())==claimed
        assert sha(z.read('frozen_checkpoint.pt'))==freeze['checkpoint_sha256']
        assert not freeze['opened_regressions_used_for_selection']
        spec=json.loads(z.read('preregistration.json'));old=json.loads((R/'experiments/task41f_preregistration_v1.json').read_text())
        for k in ('quality','compartment_gates','cost'):assert spec[k]==old[k]
        curves=json.loads(z.read('training_curves.json'));assert curves[-1]['step']==60000
        projection=json.loads(z.read('projection_data_resolution_probe.json'))
        projection_summary=[]
        for n in (64,128):
            for shape in ('linear','smoothC2'):
                for count in (1024,4096,16384):
                    rows=[r for r in projection['rows'] if r['basis']==n and r['shape']==shape and r['FIT_count']==count]
                    projection_summary.append(dict(basis=n,shape=shape,FIT_count=count,median_mse=statistics.median(r['heldout_logit_mse'] for r in rows),worst_max=max(r['heldout_logit_max'] for r in rows)))
        grad=[]
        for row in curves:
            if not row['gradients']:continue
            diag=row['gradients'];ratios=[min(.1,1e6*a/max(p,1e-30)) for p,a in zip(diag['point_gradient'],diag['auxiliary_gradient'])]
            grad.append(dict(step=row['step'],median_auxiliary_to_point_on_active_slot=statistics.median(ratios),min_ratio=min(ratios),max_ratio=max(ratios),slot_every=spec['feedback_every'],tail_examples=row['tail_examples'],atomic_examples=row['atomic_examples'],cosine_min=min(diag['cosine']),cosine_max=max(diag['cosine'])))
    final=json.loads((salv/'final_report.json').read_text());assert final['valid'] and final['decision']=='PRESERVATION_REPAIR_NO_GO' and not final['passing_arms']
    assert not final['thresholds_changed'] and not final['task42_authorized']
    precision=json.loads((salv/'precision_execution_cost.json').read_text())
    summary=dict(valid=True,scientific_finished=True,final_report_recovered=True,final_archive_complete=False,
        final_archive_expected_sha256=json.loads((D/'job_status.json').read_text())['archive_sha256'],
        latest_runtime_check='Exact Colab assignment absent on 2026-10-23; no runtime creation/restart attempted.',
        archive=receipt,crc_all_archive_members_valid=True,checkpoints=pt,freeze_sha256=claimed,
        freeze_checkpoint_sha256=freeze['checkpoint_sha256'],all_saved_tensors_finite=True,updates=60000,
        raw_json_reports=manifest,thresholds_unchanged=True,nominee=final['nominee'],decision=final['decision'],
        projection_summary=projection_summary,gradient_dose=grad,
        precision_cost=[{k:v for k,v in x.items() if k!='samples'} for x in precision],
        missing='Complete final archive container and any non-JSON final-only members not recovered; original intermediate archive/checkpoints and all available final reports retained.',
        retraining=False,run_interrupted=False)
    save('evidence_inventory.json',summary)
    save('final_report_recovered.json',final)
    print(json.dumps({k:summary[k] for k in ('valid','scientific_finished','final_report_recovered','final_archive_complete','updates','nominee','decision')}))
    print(json.dumps(dict(checkpoints=len(pt),raw_reports=len(manifest),projection=projection_summary,gradient_dose=grad)))
if __name__=='__main__':main()
