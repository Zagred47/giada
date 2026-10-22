"""Verify the byte-identical final archive and audit all paired outcomes."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
from audit_task41d_result import failed

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'artifacts/task41f_reserve_external'

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def finite(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return True

def identity(row):return tuple(row.get(k) for k in ('case','seed','family','input_seed'))

def transitions(before,after):
    a,b={identity(r) for r in before},{identity(r) for r in after}
    return dict(resolved=sorted(a-b,key=str),introduced=sorted(b-a,key=str),remaining=sorted(a&b,key=str))

def main():
    path=EXT/'recovery_final.zip';remote=json.loads((EXT/'remote_recovery_manifest.json').read_text())
    assert path.stat().st_size==remote['bytes']
    sha=digest(path);assert sha==remote['sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        print('Final archive SHA-256 and every member CRC valid',flush=True)
        load=lambda n:json.loads(z.read(n))
        params={n:finite(torch.load(io.BytesIO(z.read(n)),map_location='cpu',weights_only=False)) for n in z.namelist() if n.endswith('.pt')}
        assert all(params.values()) and 'checkpoint_60000.pt' in params
        spec=load('preregistration.json');report=load('final_report.json');freeze=load('selection_freeze.json')
        revision=load('revision.json')
        assert revision['revision']=='b8acd169d5b978311e4df327fac1d6e3c2f09be7' and not revision['dirty_runtime']
        assert spec==json.loads((ROOT/'experiments/task41f_preregistration_v1.json').read_text(encoding='utf-8'))
        unsigned={k:v for k,v in freeze.items() if k!='freeze_sha256'}
        assert hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
        assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']
        assert not freeze['opened_regressions_used_for_selection']
        assert report['valid'] and report['decision']=='PRECISION_RELAXATION_NO_GO'
        assert not report['thresholds_changed'] and not report['passing_arms']
        receipt=dict(valid=True,runtime_independent=True,local=str(path),bytes=path.stat().st_size,sha256=sha,
            remote=remote,byte_identical_to_remote=True,crc_all_members_valid=True,final_members_verified=len(z.infolist()),
            parameters_finite=params,freeze_sha256=freeze['freeze_sha256'],checkpoint_sha256=freeze['checkpoint_sha256'],
            decision=report['decision'],scientific_valid=report['valid'],technical_failure=False,revision=revision,
            all_training_updates_completed=int(torch.load(io.BytesIO(z.read('checkpoint_60000.pt')),map_location='cpu',weights_only=False)['step']),
            recovery='Independent terminal transport; original run and thresholds untouched.',
            previous_supervisor_receipt='Preserved separately: it did not verify the final transport.')
        (EXT/'final_recovery_receipt.json').write_text(json.dumps(receipt,indent=2))
        (EXT/'external_backup_receipt.json').write_text(json.dumps(receipt,indent=2))
        details={n:{'stress':{},'responses':{}} for n in report['summaries']}
        for panel in ('41','41c'):
            stress=load('stress'+panel+'.json');causal=load('opened'+panel+'/causal_results.json')
            for n in details:
                details[n]['stress'][panel]=failed(stress['rows'],n)
                details[n]['responses'][panel]=failed(causal[n+'/autonomous']['interventions'])
        historical=load('historical_regression/causal_results.json')
        for n in details:details[n]['responses']['40b']=failed(historical[n+'/autonomous']['interventions'])
        regressions={b:{n:{kind+'/'+panel:transitions(details[b][kind][panel],details[n][kind][panel])
            for kind in ('stress','responses') for panel in details[n][kind]} for n in details}
            for b in ('P110','E_Q0','E_H10')}
        precision=load('frozen_audit/precision.json')
        precision_summary=[dict(model=r['model'],policy=r['policy'],counts={k:dict(passed=sum(v['metrics']['passed'] for v in rows),total=len(rows)) for k,rows in r['reference_comparisons'].items()}) for r in precision]
        oracles=load('frozen_audit/coefficient_oracles.json')
        oracle_summary=[]
        for r in oracles:
            rows=[]
            for v in r['rows']:
                case=v['original_case'];ch=v['metrics']['channels'][0]
                rows.append(dict(seed=v['seed'],case={k:case.get(k) for k in ('case','family','panel','seed')},passed=v['metrics']['passed'],Ca_HVA=ch,crossing=v['Ca_crossing']))
            oracle_summary.append(dict(model=r['model'],oracle=r['oracle'],passed=sum(v['passed'] for v in rows),total=len(rows),rows=rows))
        projection=load('projection_data_resolution_probe.json')
        objectives={}
        for step in (0,60000):
            raw=load(f'objective_probe_{step}.json');base=raw['rows']['unchanged'];summary={}
            for direction in ('point_descent','response_descent'):
                summary[direction]={}
                for loss in base:
                    a=np.asarray(base[loss]);b=np.asarray(raw['rows'][direction][loss]);delta=b-a
                    summary[direction][loss]=dict(improved=int((delta<0).sum()),worsened=int((delta>0).sum()),equal=int((delta==0).sum()),median_relative_change=float(np.median(delta/np.maximum(abs(a),1e-30))),delta=delta.tolist())
            objectives[str(step)]=summary
        curves=load('training_curves.json');assert [v['step'] for v in curves]==spec['checkpoints']
        for curve in curves:
            development=load(f"development_{curve['step']}.json")
            for rank,rows in zip(curve['ranks'],development):
                assert rank[1]==sum(any(not r['channels'][k]['passed'] for k in (0,4)) for r in rows)
        budgets=[dict(step=r['step'],target_passing=sum(v[1]==0 for v in r['ranks']),introduced=sum(v[0] for v in r['ranks']),median_response=float(np.median(r['response'])),median_atomic=float(np.median(np.sum(r['point'],axis=-1)))) for r in curves]
        point0=np.asarray(curves[0]['point']);point1=np.asarray(curves[-1]['point'])
        optimization=dict(lr=spec['lr'],head_order=['u_Ca_m','u_Ca_h','u_Nap_m','z_Ca_m','z_Ca_h','z_Nap_m'],
            mean_normalized_point_initial=point0.mean(0).tolist(),mean_normalized_point_final=point1.mean(0).tolist(),
            total_point_final_over_initial=(point1.sum(-1)/point0.sum(-1)).tolist(),
            members_worse_point=int((point1.sum(-1)>point0.sum(-1)).sum()),
            response_final_over_initial=(np.asarray(curves[-1]['response'])/np.asarray(curves[0]['response'])).tolist(),
            caveat='Fixed initial per-head normalization; dimensionless scores, not absolute gate errors. Final60000 is not necessarily frozen selection. Association with regressions is not per-head causal attribution.')
        out=dict(valid=True,thresholds_changed=False,backup=receipt,decision=report['decision'],nominee=report['nominee'],summaries=report['summaries'],failure_details=details,
            regressions=regressions,precision=precision_summary,coefficient_oracles=oracle_summary,
            reproduction=load('frozen_audit/reproduction.json'),batch_sensitivity=load('frozen_audit/batch_sensitivity.json'),
            clamps=load('frozen_audit/clamps.json'),projection=projection,objective_directions=objectives,
            selection=freeze,selection_predicates_consistent=True,budget_curves=budgets,optimization=optimization,contrasts=load('paired_contrasts.json'),
            caution='Diagnostic opened panels are not independent confirmation; no alternate dtype, reference, nominee or threshold is promoted by this audit.')
        (EXT/'final_detailed_audit.json').write_text(json.dumps(out,indent=2))
        compact_objectives={s:{d:{k:{kk:vv for kk,vv in v.items() if kk!='delta'} for k,v in losses.items()} for d,losses in directions.items()} for s,directions in objectives.items()}
        print(json.dumps(dict(backup=receipt,objectives=compact_objectives,budgets=budgets,chosen_steps=freeze['chosen_steps'])),flush=True)

if __name__=='__main__':main()
