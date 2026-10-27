"""Verify recovery archive and audit all frozen arms, without reselection."""
import hashlib,io,json,statistics,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research_memory.mirror import write_json
from compare_task41h_recovery import failures,difference,finite


def main():
    receipt=json.loads((ROOT/'artifacts/task41k_kaggle_external/pipeline_final.json').read_text())
    assert receipt['valid'] and not receipt['technical_failure']
    path=Path(receipt['local'])
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
    prior=json.loads((ROOT/'experiments/results/task41j_audited_v1/comparison_with_H.json').read_text())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        read=lambda name:json.loads(z.read(name))
        report=read('final_report.json');freeze=read('selection_freeze.json')
        assert report['valid'] and report['training_updates_performed']==0 and not report['selection_recomputed']
        assert not report['thresholds_changed'] and not report['task42_authorized']
        assert read('preregistration.json')==json.loads((ROOT/'experiments/task41k_preregistration_v1.json').read_text())
        payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
        assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']
        tensor_checks={}
        for name in z.namelist():
            if name.endswith('.pt'):
                assert finite(torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False)),name
                tensor_checks[name]=True
        names=['H_nominee']+[a['name'] for a in freeze['arms']];failed={n:{} for n in names}
        dev=read('development_gpu.json')
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
        costs=read('cost.json')
        def speeds(rows):
            return [dict(scope=r['scope'],batch_per_seed=r['batch_per_seed'],
                vs_formula={n:statistics.median(r['samples']['formula']['wall_ms'])/statistics.median(v['wall_ms']) for n,v in r['samples'].items() if n!='formula'}) for r in rows]
        fixed_comparison={}
        for n in names[1:]:
            fixed_comparison[n]={}
            comparator='J_chain_joint' if not n.startswith('K_copy') else 'J_compact64'
            for panel,rows in failed[n].items():
                if panel=='development':continue # new development seeds: no paired historical comparison
                fixed_comparison[n][panel]=difference(prior['failures'][comparator][panel],rows)
        compact=[]
        for n in names:
            rows=failed[n];s=report['summaries'][n]
            compact.append(dict(arm=n,failed_evaluations=sum(map(len,rows.values())),
                development=s['development']['passed'],stress41=s['stress']['41']['passed'],stress41c=s['stress']['41c']['passed'],
                response_failures=sum(len(v) for k,v in rows.items() if k.startswith(('41/','41c/')) and k.endswith('/responses')),
                historical_failures=sum(len(v) for k,v in rows.items() if k.startswith('historical/')),
                recurrence_failures=sum(len(v) for k,v in rows.items() if k.startswith('recurrence/')),
                introduced_vs_H={k:len(difference(failed['H_nominee'][k],v)['introduced']) for k,v in rows.items() if difference(failed['H_nominee'][k],v)['introduced']}))
        result=dict(valid=True,complete=True,report=report,backup=receipt,tensor_checks=tensor_checks,
            chosen_steps=freeze['chosen_steps'],failures=failed,compact=compact,
            changes_vs_H={n:{p:difference(failed['H_nominee'][p],v) for p,v in rows.items()} for n,rows in failed.items() if n!='H_nominee'},
            changes_vs_J_fixed_panels=fixed_comparison,speeds=speeds(costs),J_speeds=speeds(prior['costs']),
            costs=costs,paired_contrasts=read('paired_contrasts.json'),curves=read('training_curves.json'),
            limits=['Development differs: use same-run H; fixed panels alone permit paired J comparison.',
                    'Failed evaluations are not unique biological cases; same case may fail multiple tests.',
                    'Speed ratios are paired within each run, not direct paired cross-run hardware comparisons.',
                    'No repair or compression arm globally qualified; no Task42 authorization.'])
        folder=ROOT/'experiments/results/task41k_audited_v1';folder.mkdir(exist_ok=True)
        write_json(folder/'comparison_with_J.json',result)
        print(json.dumps(dict(valid=True,decision=report['decision'],nominee=report['nominee'],compact=compact,speeds=result['speeds'],chosen_steps=freeze['chosen_steps'],all_checkpoints_finite=True,backup_sha256=receipt['remote']['sha256'])),flush=True)


if __name__=='__main__':main()
