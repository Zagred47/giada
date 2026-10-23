"""Verify the recovered final bundle and compare every registered panel to41g."""
import hashlib,io,json,statistics,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research_memory.mirror import write_json

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def finite(v):
    if torch.is_tensor(v):return bool(torch.isfinite(v).all())
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,(tuple,list)):return all(finite(x) for x in v)
    return True
def failures(rows):
    result=[]
    for row in rows:
        metric=row.get('metrics',row)
        if metric['passed']:continue
        result.append(dict(identity={k:row[k] for k in ('case','seed','family','input_seed','original_case','role') if k in row},
            failed_channels=[r for r in metric.get('channels',[]) if not r['passed']],
            other_metrics={k:v for k,v in metric.items() if k!='channels'}))
    return result
def identities(rows):return {json.dumps(r['identity'],sort_keys=True):r for r in rows}
def difference(old,new):
    a,b=identities(old),identities(new)
    return dict(resolved=[a[k] for k in a.keys()-b.keys()],introduced=[b[k] for k in b.keys()-a.keys()],remaining=[b[k] for k in a.keys()&b.keys()])
def compact_cost(rows):
    output=[]
    for row in rows:
        samples=row['samples'];medians={}
        for name,values in samples.items():
            medians[name]=statistics.median(values['wall_ms'] if isinstance(values,dict) else values)
        output.append({k:v for k,v in row.items() if k!='samples'}|dict(median_wall_ms=medians))
    return output

def main():
    folder=ROOT/'experiments/results/task41h_recovery_v1';ext=ROOT/'artifacts/task41h_evaluation_primary_external'
    receipt=json.loads((ext/'external_backup_receipt.json').read_text());status=json.loads((ext/'job_status.json').read_text())
    assert status['state']=='finished' and status.get('worker_exit_code')==0 and not status.get('technical_failure')
    assert receipt['valid'] and receipt['remote']['label']=='final'
    path=Path(receipt['local']);assert sha(path)==receipt['remote']['sha256']
    old=json.loads((ROOT/'experiments/results/task41g_audited_v1/final_report_recovered.json').read_text())
    old_receipt=json.loads((ROOT/'artifacts/task41g_primary_external/final_reconstruction_receipt.json').read_text())
    old_path=Path(old_receipt['local']);assert sha(old_path)==old_receipt['sha256']
    with zipfile.ZipFile(old_path) as prior:
        old_cost=compact_cost(json.loads(prior.read('cost.json')))
        old_execution=compact_cost(json.loads(prior.read('execution_matrix.json')))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        read=lambda name:json.loads(z.read(name))
        report=read('final_report.json');spec=read('preregistration.json');freeze=read('selection_freeze.json')
        assert report['valid'] and report['training_updates_performed']==0 and not report['thresholds_changed']
        original=json.loads((ROOT/'experiments/task41h_preregistration_v1.json').read_text());assert spec==original
        assert read('selection_parity.json')['all60_rank_vectors_bitwise_equal']
        payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
        assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']
        tensor_checks={}
        for name in z.namelist():
            if name.endswith('.pt'):
                assert finite(torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False)),name
                tensor_checks[name]=hashlib.sha256(z.read(name)).hexdigest()
        seal=json.loads((folder/'input_seal.json').read_text())
        for name,digest in seal['checkpoint_sha256'].items():assert tensor_checks[name]==digest
        details={name:{} for name in report['summaries']};floors={}
        for panel in ('41','41c'):
            data=read('opened'+panel+'/causal_results.json')
            for key,row in data.items():
                name,mode=key.split('/');dest=details.setdefault(name,{})
                dest[panel+'/'+mode+'/responses']=failures(row['interventions'])
                dest[panel+'/'+mode+'/trajectories']=failures(row['cases'])
            del data
            stress=read('stress'+panel+'.json')
            floors['stress'+panel]=all(r['passed'] for r in stress['formula_floor'])
            for name in details:details[name]['stress'+panel]=failures([r for r in stress['rows'] if r['model']==name])
            del stress
            for mode in ('imposed','autonomous'):
                floor=read('opened'+panel+'/floor_'+mode+'.json')
                floors[panel+'/'+mode]=all(r['passed'] for r in floor['cases']+floor['interventions'])
            floors[panel+'/native']=all(r['passed'] for r in read('opened'+panel+'/native_phase_floor.json'))
        hist=read('historical_regression/causal_results.json')
        for key,row in hist.items():
            name,mode=key.split('/');dest=details.setdefault(name,{})
            dest['historical/'+mode+'/responses']=failures(row['interventions'])
            dest['historical/'+mode+'/trajectories']=failures(row['cases'])
        del hist
        recurrence=read('historical_regression/recurrence_evaluation.json')
        for name in details:
            for role in ('development','stress_diagnostic'):
                details[name]['recurrence/'+role]=failures([r for r in recurrence if r['model']=='joint_w'+name and r['role']==role])
        del recurrence
        dev=read('development_gpu.json')
        for name,rows in dev.items():details[name]['development_new_41h']=failures(rows)
        old_names={'G_nominee':original['task41g_nominee'],'G_tail_diagnostic':original['task41g_diagnostic']}
        reproduction={}
        for current,previous in old_names.items():
            reproduction[current]={k:report['summaries'][current][k]==old['summaries'][previous][k] for k in ('panels','stress','historical','recurrence')}
        deltas={name:{base:{panel:difference(details[base][panel],rows) for panel,rows in panels.items() if panel in details[base]}
            for base in old_names} for name,panels in details.items() if name in {a['name'] for a in spec['arms']} or name.startswith('aggregate_')}
        rawcost=read('cost.json');cost=compact_cost(rawcost);execution=compact_cost(read('execution_matrix.json'));compact=read('compact_execution_cost.json')
        same_run_speed=[]
        for row in rawcost:
            samples=row['samples'];name=report['nominee']
            same_run_speed.append(dict(batch=row['batch_per_seed'],scope=row['scope'],candidate=name,
                speedup_vs_41g_nominee=statistics.median(a/b for a,b in zip(samples['G_nominee']['wall_ms'],samples[name]['wall_ms'])),
                speedup_vs_41g_tail_diagnostic=statistics.median(a/b for a,b in zip(samples['G_tail_diagnostic']['wall_ms'],samples[name]['wall_ms'])),
                speedup_vs_formula=statistics.median(a/b for a,b in zip(samples['formula']['wall_ms'],samples[name]['wall_ms'])),
                quality_qualified=name in report['passing_arms']))
        result=dict(valid=True,complete_final_backup_verified=True,all_tensors_finite=True,unchanged_training_checkpoints=tensor_checks,
            backup=receipt,report=report,nominee=report['nominee'],old_nominee=old['nominee'],old_summaries=old['summaries'],
            baseline_reproduction=reproduction,floors=floors,all_floors_passed=all(floors.values()),failures=details,deltas=deltas,
            selection=freeze,paired_contrasts=read('paired_contrasts.json'),ionic_cost=cost,execution_matrix=execution,compact_execution_cost=compact,
            same_run_speed=same_run_speed,old41g_cost=old_cost,old41g_execution=old_execution,
            training_updates_performed=0,thresholds_changed=False,
            limits=['Old and new development use different seed families: compare new41h candidates to G controls evaluated on the same41h development, not old development counts as paired.',
                '41g controls are measured again within41h. Historical timing ratios from different VM assignments are descriptive, not a paired speed benchmark.',
                'Nominee is fixed by original stratified selection. Posthoc best counts do not replace it. Alternate numerical policies remain diagnostic.'])
        write_json(folder/'comparison_with_41g.json',result);write_json(folder/'final_report.json',report)
        print(json.dumps(dict(valid=True,nominee=report['nominee'],decision=report['decision'],passing=report['passing_arms'],floors=floors,reproduction=reproduction,
            summaries={k:report['summaries'][k] for k in (report['nominee'],'G_nominee','G_tail_diagnostic')})),flush=True)

if __name__=='__main__':main()
