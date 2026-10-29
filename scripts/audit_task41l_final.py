"""Read saved evidence only; retain full source JSON separately in the mirror."""
import hashlib, io, json, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'artifacts/task41l_colab_primary_external/recovery_final.zip'
SHA='bd88755d1a4728180a5a3621d67f0f415101852baa3d261a4f465879bd301c7b'

def failed(rows):
    result=[]
    for row in rows:
        metric=row.get('metrics',row)
        if metric['passed']:continue
        result.append({k:row[k] for k in ('case','seed','family','role') if k in row}|
            dict(channels=[c for c in metric.get('channels',[]) if not c['passed']],gate=metric.get('gate')))
    return result

def main():
    import torch
    with ARCHIVE.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        report=read('final_report.json');freeze=read('selection_freeze.json')
        weights=torch.load(io.BytesIO(z.read('frozen_checkpoint.pt')),map_location='cpu',weights_only=True)
        def finite(x):
            if torch.is_tensor(x):return bool(torch.isfinite(x).all())
            if isinstance(x,dict):return all(finite(v) for v in x.values())
            if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
            return True
        assert finite(weights) and report['valid'] and not report['passing_arms']
        names=['H_nominee']+list(freeze['chosen_steps']);failures={n:{} for n in names}
        for panel,path in [('41','opened41/causal_results.json'),('41c','opened41c/causal_results.json'),('historical','historical_regression/causal_results.json')]:
            data=read(path)
            for name in names:
                for mode in ('imposed','autonomous'):
                    row=data[name+'/'+mode]
                    failures[name][panel+'/'+mode+'/responses']=failed(row['interventions'])
                    failures[name][panel+'/'+mode+'/trajectories']=failed(row['cases'])
        for panel in ('41','41c'):
            rows=read('stress'+panel+'.json')['rows']
            for name in names:failures[name]['stress'+panel]=failed([r for r in rows if r['model']==name])
        rows=read('historical_regression/recurrence_evaluation.json')
        for name in names:
            for role in ('development','stress_diagnostic'):
                failures[name]['recurrence/'+role]=failed([r for r in rows if r['model']=='joint_w'+name and r['role']==role])
        frozen=[]
        for name in sorted(z.namelist()):
            if name.startswith('frozen_channel_attribution/41') and name.endswith('.json'):
                row=read(name);frozen.append({k:row[k] for k in ('panel','policy','forcing','model','continuous','native_floor_qualified')}|dict(failures=failed(row['responses'])))
        stress={panel:{n:failed([r for r in read('frozen_channel_attribution/imposed_stress'+panel+'.json')['rows'] if r['model']==n]) for n in ('swap000','swap100','K_full')} for panel in ('41','41c')}
        recurrence=read('frozen_channel_attribution/recurrence_evaluation.json')
        result=dict(valid=True,source=str(ARCHIVE.relative_to(ROOT)),sha256=SHA,bytes=ARCHIVE.stat().st_size,
            all_frozen_parameters_finite=True,zip_crc_verified=True,report=report,chosen_steps=freeze['chosen_steps'],
            failures=failures,failed_evaluations_excluding_new_development={n:sum(map(len,v.values())) for n,v in failures.items()},
            frozen_diagnostics=frozen,frozen_stress=stress,
            frozen_recurrence={n:failed([r for r in recurrence if r['model']=='joint_w'+n]) for n in ('swap000','swap100','K_full')},
            numerical_floors=read('frozen_channel_attribution/paired_numerical_floors.json'),
            cost=read('cost.json'),source_checkpoint_sha256=hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest(),
            source_freeze_sha256=hashlib.sha256(z.read('selection_freeze.json')).hexdigest(),
            caveats=['New development differs from historical development; compare same-run H.',
                     'A numerical pass is not proof of independent confirmation or qualified speed.',
                     'Native-floor predicates cover voltage/calcium, not a new gate-equivalence claim.'])
        destination=ROOT/'experiments/results/task41l_audited_v1';destination.mkdir(parents=True,exist_ok=True)
        (destination/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        # A small, byte-identical subset for reusing invariant imposed diagnostics.
        subset=['frozen_channel_attribution/'+n for n in ('imposed_stress41.json','imposed_stress41c.json','recurrence_evaluation.json')]
        subset+=['final_report.json','preregistration.json']
        package=ROOT/'artifacts/task41m_reuse.zip.bin';manifest={}
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as target:
            for name in subset:
                raw=z.read(name);target.writestr(name,raw);manifest[name]=hashlib.sha256(raw).hexdigest()
        receipt=dict(source_sha256=SHA,source=str(ARCHIVE.relative_to(ROOT)),members=manifest,
            local=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest())
        (destination/'reuse_manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(dict(valid=True,counts=result['failed_evaluations_excluding_new_development'],reuse=receipt['sha256'])),flush=True)

if __name__=='__main__':main()
