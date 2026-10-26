"""Read-only scientific audit of a verified, finished Task41i recovery."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from research_memory.mirror import write_json
from compare_task41h_recovery import failures,difference,finite


def main():
    ext=ROOT/'artifacts/task41i_kaggle_external'
    receipt=json.loads((ext/'pipeline_final.json').read_text())
    assert receipt['valid'] and not receipt['technical_failure']
    path=Path(receipt['local'])
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        read=lambda name:json.loads(z.read(name))
        report=read('final_report.json');freeze=read('selection_freeze.json')
        repair=json.loads((ROOT/'experiments/task41i_evaluation_recovery_v1.json').read_text())
        assert report['valid'] and report['training_updates_performed']==0 and not report['selection_recomputed']
        assert not report['thresholds_changed']
        assert read('preregistration.json')==json.loads((ROOT/'experiments/task41i_preregistration_v1.json').read_text())
        assert freeze['freeze_sha256']==repair['freeze_sha256']
        assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==repair['checkpoint_sha256']
        assert finite(torch.load(io.BytesIO(z.read('frozen_checkpoint.pt')),map_location='cpu',weights_only=False))
        names=['H_nominee',report['nominee']]+[n for n in report['summaries'] if '__' in n]
        failed={n:{} for n in names}
        dev=read('development_gpu.json')
        for n in names:failed[n]['development']=failures(dev[n])
        for panel in ('41','41c'):
            data=read('opened'+panel+'/causal_results.json');stress=read('stress'+panel+'.json')
            for n in names:
                failed[n]['stress'+panel]=failures([r for r in stress['rows'] if r['model']==n])
                for mode in ('imposed','autonomous'):
                    row=data[n+'/'+mode]
                    failed[n][panel+'/'+mode+'/trajectory']=failures(row['cases'])
                    failed[n][panel+'/'+mode+'/responses']=failures(row['interventions'])
        data=read('historical_regression/causal_results.json');rec=read('historical_regression/recurrence_evaluation.json')
        for n in names:
            for mode in ('imposed','autonomous'):
                row=data[n+'/'+mode]
                failed[n]['historical/'+mode+'/trajectory']=failures(row['cases'])
                failed[n]['historical/'+mode+'/responses']=failures(row['interventions'])
            for role in ('development','stress_diagnostic'):
                failed[n]['recurrence/'+role]=failures([r for r in rec if r['model']=='joint_w'+n and r['role']==role])
        baseline=failed['H_nominee']
        changes={n:{panel:difference(baseline[panel],rows) for panel,rows in failed[n].items()} for n in names[1:]}
        costs=read('cost.json');paired_cost=[]
        for row in costs:
            for n in names[1:]:
                if n in row['comparisons'] and 'H_nominee' in row['comparisons'][n]:
                    paired_cost.append(dict(batch_per_seed=row['batch_per_seed'],actual_sites=row['actual_sites'],scope=row['scope'],model=n,
                                            paired_vs_H=row['comparisons'][n]['H_nominee']))
        old=json.loads((ROOT/'experiments/results/task41h_recovery_v1/audit_completion.json').read_text())
        result=dict(valid=True,complete=True,report=report,backup=receipt,selected_checkpoint_unchanged=True,
                    selection_unchanged=True,training_updates_performed=0,thresholds_changed=False,
                    old41h_audit=old,failures=failed,changes_vs_H_same_panels=changes,
                    failed_evaluations={n:sum(map(len,panels.values())) for n,panels in failed.items()},
                    paired_cost_vs_H=paired_cost,whole_chain_cost=read('whole_chain_execution_cost.json'),
                    limitations=['New development cases differ from41h; compare I to the H control rerun on the SAME41i panel, not old development counts.',
                                 'Raw times across VM assignments are not paired speed comparisons.',
                                 'Numerical/execution variants are diagnostic and did not select the learned nominee. No automatic Task42 authorization.'])
    folder=ROOT/'experiments/results/task41i_recovery_v1';folder.mkdir(exist_ok=True)
    write_json(folder/'comparison_with_41h.json',result)
    print(json.dumps(dict(valid=True,decision=report['decision'],nominee=report['nominee'],failed=result['failed_evaluations'],backup_sha256=receipt['remote']['sha256'])),flush=True)


if __name__=='__main__':main()
