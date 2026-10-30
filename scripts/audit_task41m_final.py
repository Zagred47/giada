"""Read-only41m audit. Original and matched references must never be pooled."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    import torch
    receipt=json.loads((ROOT/'artifacts/task41m_primary_external/external_backup_receipt.json').read_text())
    archive=Path(receipt['local']);assert receipt['valid'] and receipt['runtime_independent'] and receipt['remote']['label']=='final'
    with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
    assert archive.stat().st_size==receipt['remote']['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        read=lambda name:json.loads(z.read(name))
        report=read('final_report.json');spec=read('preregistration.json')
        assert report['valid'] and report['complete'] and report['all_parameters_finite']
        assert report['training_updates_performed']==0 and not report['selection_performed']
        assert not report['thresholds_changed'] and not report['abstraction_changed']
        weights=torch.load(io.BytesIO(z.read('frozen_models.pt')),map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for model in weights.values() for v in model.values())
        first,second=weights['swap000'],weights['swap100']
        assert first.keys()==second.keys()
        assert all(torch.equal(first[k],second[k]) for k in first if k!='mask')
        assert int((first['mask']!=second['mask']).sum())==2
        donor={k[len('donor.'):]:v for k,v in second.items() if k.startswith('donor.')}
        assert donor.keys()==weights['K_full'].keys()
        assert all(torch.equal(v,weights['K_full'][k]) for k,v in donor.items())
        reuse=read('reused_reports.json')
        for name,sha in spec['reuse']['members'].items():
            # Reused parsed values are checked against the immutable subset,
            # whose original member byte hashes were checked by the worker.
            with zipfile.ZipFile(ROOT/'artifacts/task41m_reuse.zip.bin') as prior:
                raw=prior.read(name);assert hashlib.sha256(raw).hexdigest()==sha
                assert json.loads(raw)==reuse['reports'][name]
        old=reuse['reports'];models=spec['models'];aux={}
        for model in models:
            aux[model]={}
            for panel in ('41','41c'):
                rows=[r for r in old['frozen_channel_attribution/imposed_stress'+panel+'.json']['rows'] if r['model']==model]
                assert len(rows)==36
                aux[model]['stress'+panel]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows),
                    failures=[r for r in rows if not r['metrics']['passed']])
            for role in ('development','stress_diagnostic'):
                rows=[r for r in old['frozen_channel_attribution/recurrence_evaluation.json'] if r['model']=='joint_w'+model and r['role']==role]
                assert len(rows)==72
                aux[model]['recurrence_'+role]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows),failures=[r for r in rows if not r['metrics']['passed']])
        aggregate=[];failures=[]
        for model in models:
            for policy in spec['policies']:
                for reference in ['primary']+([] if policy=='primary' else [policy]):
                    rows=[r for r in report['summaries'] if r['model']==model and r['mode']=='autonomous' and r['policy']==policy and r['reference']==reference]
                    assert {r['panel'] for r in rows}=={'41','41c','historical'}
                    row=dict(model=model,policy=policy,reference=reference,
                        trajectory_passed=sum(r['summary']['trajectory_passed'] for r in rows),trajectory_total=sum(r['summary']['trajectory_total'] for r in rows),
                        response_passed=sum(r['summary']['responses_passed'] for r in rows),response_total=sum(r['summary']['responses_total'] for r in rows),
                        imposed_aux_passed=sum(r['passed'] for r in aux[model].values()),imposed_aux_total=sum(r['total'] for r in aux[model].values()))
                    row['failed_evaluations']=row['trajectory_total']-row['trajectory_passed']+row['response_total']-row['response_passed']+row['imposed_aux_total']-row['imposed_aux_passed']
                    aggregate.append(row)
                    for rr in rows:
                        value=read(f"{rr['panel']}_autonomous_{policy}_{model}_vs_{reference}.json")
                        for response in value['metrics']['interventions']:
                            if not response['metrics']['passed']:
                                failures.append(dict(model=model,policy=policy,reference=reference,panel=rr['panel'],**response))
        cost=read('numerical_cost.json');timing=[]
        for row in cost:
            samples={k:np.asarray(v,dtype=float) for k,v in row['samples'].items()}
            key='voltage_solve64/swap100'
            timing.append(dict(batch_per_seed=row['batch_per_seed'],integrated_candidate_over_original_formula=float(np.median(samples[key]/samples['primary/formula'])),
                integrated_candidate_over_native_H=float(np.median(samples[key]/samples['primary/H_native'])),
                primary_transplant_over_native_H=float(np.median(samples['primary/swap100']/samples['primary/H_native'])),
                mixed_precision_over_primary_transplant=float(np.median(samples[key]/samples['primary/swap100'])),
                paired_saved_fraction=row['versus_original_formula'][key],diagnostic_only=True))
        floors=read('canonical_floors.json');assert all(r['passed'] for r in floors if not r['historical_imposed_floor_not_recomputed'])
        result=dict(valid=True,complete=True,archive=str(archive.relative_to(ROOT)),sha256=receipt['remote']['sha256'],bytes=archive.stat().st_size,
            zip_crc_verified=True,all_saved_tensors_finite=True,selective_models_differ_only_in_two_gate_mask=True,donor_identity_verified=True,
            elapsed_seconds=report['elapsed_seconds'],training_updates_performed=0,thresholds_changed=False,abstraction_changed=False,
            summaries=aggregate,failed_responses=failures,reused_imposed_diagnostics=aux,
            original_reports_sha256={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith('.json')},
            timing=timing,primary_cost=read('cost.json'),canonical_floor_passed=True,
            interpretation={
                'progress':'Under original FP32 reference: H3 failures, Ca-only primary2, Ca-only voltage/solve64 zero on1320autonomous trajectories+840responses+72stress+144recurrence evaluations. Opened diagnostic evidence, not independent replication.',
                'separable_causes':'Ca-only neural transplant repairs both Ca_HVA boundary stress failures without donor Nap/K_Tst imposed regressions. Coherent voltage/solve64 removes remaining autonomous failures; no new training.',
                'reference_warning':'Recentered32 Ca-only has3failed responses against original reference but0against matched reference. Joint64 has0against both, so its observed pass is not solely a reference switch.',
                'cost':'Two-network selective diagnostic is not a compact deployment. At642sites/seed integrated execution is about2.01times original formula. No qualified speedup.',
                'scope':'New development and independent fresh protocols were not evaluated; original FP32 release contract not silently changed. Task42 remains unauthorized. No proof of training-horizon causality or fundamental capacity limit.'})
        folder=ROOT/'experiments/results/task41m_audited_v1';folder.mkdir(parents=True,exist_ok=True)
        (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
        print(json.dumps(dict(valid=True,counts=aggregate,timing=timing),indent=2))

if __name__=='__main__':main()
