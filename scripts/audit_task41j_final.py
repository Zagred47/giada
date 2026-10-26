"""Verify the final archive and compare every arm to same-panel frozen H."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from research_memory.mirror import write_json
from compare_task41h_recovery import failures,difference,finite


def main():
    receipt=json.loads((ROOT/'artifacts/task41j_kaggle_external/pipeline_final.json').read_text())
    assert receipt['valid'] and not receipt['technical_failure']
    path=Path(receipt['local'])
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        read=lambda name:json.loads(z.read(name))
        report=read('final_report.json');freeze=read('selection_freeze.json')
        payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
        assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']
        assert finite(torch.load(io.BytesIO(z.read('frozen_checkpoint.pt')),map_location='cpu',weights_only=True))
        assert report['valid'] and not report['thresholds_changed'] and not report['selection_recomputed']
        assert report['training_updates_performed']==0
        assert read('preregistration.json')==json.loads((ROOT/'experiments/task41j_preregistration_v1.json').read_text())
        names=['H_nominee']+[a['name'] for a in freeze['arms']];failed={n:{} for n in names};dev=read('development_gpu.json')
        for n in names:failed[n]['development']=failures(dev[n])
        for label in ('41','41c'):
            data=read('opened'+label+'/causal_results.json');stress=read('stress'+label+'.json')
            for n in names:
                failed[n]['stress'+label]=failures([r for r in stress['rows'] if r['model']==n])
                for mode in ('imposed','autonomous'):
                    row=data[n+'/'+mode]
                    failed[n][label+'/'+mode+'/trajectory']=failures(row['cases'])
                    failed[n][label+'/'+mode+'/responses']=failures(row['interventions'])
        hist=read('historical_regression/causal_results.json');rec=read('historical_regression/recurrence_evaluation.json')
        for n in names:
            for mode in ('imposed','autonomous'):
                failed[n]['historical/'+mode+'/trajectory']=failures(hist[n+'/'+mode]['cases'])
                failed[n]['historical/'+mode+'/responses']=failures(hist[n+'/'+mode]['interventions'])
            for role in ('development','stress_diagnostic'):
                failed[n]['recurrence/'+role]=failures([r for r in rec if r['model']=='joint_w'+n and r['role']==role])
        result=dict(valid=True,complete=True,report=report,backup=receipt,thresholds_changed=False,
            failures=failed,failed_evaluations={n:sum(map(len,rows.values())) for n,rows in failed.items()},
            changes_vs_H={n:{panel:difference(failed['H_nominee'][panel],rows) for panel,rows in failed[n].items()} for n in names[1:]},
            costs=read('cost.json'),distillation=read('distillation_diagnostics.json'),
            limits=['Development differs from41i/41h: use current same-panel H, not unmatched historical totals.',
                    'Whole operator paired times only; cross-VM absolute times are not paired.',
                    'No opened-panel reselection or automatic Task42 authorization.'])
    folder=ROOT/'experiments/results/task41j_audited_v1';folder.mkdir(exist_ok=True)
    write_json(folder/'comparison_with_H.json',result)
    print(json.dumps(dict(valid=True,decision=report['decision'],nominee=report['nominee'],failed=result['failed_evaluations'],backup_sha256=receipt['remote']['sha256'])),flush=True)


if __name__=='__main__':main()
