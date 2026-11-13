"""45e evaluation-only recovery authorization/results; exclusive local SQLite."""
import argparse,json,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--blocked',action='store_true');ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task45e_evaluation_v1';folder.mkdir(exist_ok=True)
    before=folder/'memory_receipt.json' if a.result or a.blocked else repo/'experiments/results/task45e_protocol_v1/result_memory_receipt.json'
    baseline={} if a.validate_only else json.loads(before.read_text())['verification'];m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task45e-evaluation-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    if a.blocked:
        p=json.loads((repo/'experiments/task45e_evaluation_recovery_v1.json').read_text());assert not p['new_upload']
        put('observations','transport-blocked',Nome='45e recovery blocked beforeGPU; compact upload not authorized, existing output not mounted',Descrizione=json.dumps(dict(plan=p,compact_upload_rejected=True,upload_performed=False,scientific_protocol_unchanged=True,CPU_mount_failed=True,GPU_launched=False,error="AssertionError: ('task45e_final.zip', 0)",reason='Automatic review requires explicit permission for a new private Dataset payload. Existing-output alternative was attempted with CPU only: archived failed-run output is downloadable but not exposed as notebook input. Original full archive and compact local input preserved. No training, nomination or GPU evaluation repeated.')))
        put('decisions','await-upload-approval',Nome='Await explicit approval for private45e checkpoint Dataset',Motivazione='Request permission for70,230,258byte frozen checkpoint+completed-metric input on own Kaggle account. No native traces or DB. Do not bypass review, auto-upload, relaunch training or start unverifiedGPU. Original archive hash16cf035671bdc4fd877a381db3bfdbf3a7a448e6d075bfb811847f41730180a8 retained.',**{'Data':'2026-11-13'})
    elif not a.result:
        p=json.loads((repo/'experiments/task45e_evaluation_recovery_v1.json').read_text())
        put('observations','authorized',Nome='45e complete8missing evaluations only;10blocks retained',Descrizione=json.dumps(dict(user_authorized=True,source=p,training_performed=False,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,scope='Load final saved models, frozenP nomination; skip10completed blocks, evaluate8missing then all registered stress/cost. Preserve original runtime-limit archive and failed complete-run marker. Compare all arms to45d, no new fit or nomination.')))
        put('decisions','scope',Nome='45e recover evaluations without repeating training or selection',Esito='Continuare',Motivazione='Explicit user authorization. Existing frozen checkpoint source transported to a private Kaggle input, no biological/DB upload. Same immutable source archive and registered cost closure; SQLite backup remains local.',**{'Data':'2026-11-13'})
    else:
        r=json.loads((repo/'artifacts/task45e_evaluation_external/external_backup_receipt.json').read_text());assert r['valid'];path=Path(r['local'])
        art=put('artifacts','output',Nome='45e missing-evaluation recovery output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':r['remote']['sha256'],'Dimensione byte':r['remote']['bytes']});obs=[]
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):obs.append(put('observations','raw-'+name,Nome='45e evaluation '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        obs.append(put('observations','comparison',Nome='All45e vs45d arms, timing, regressions, stress and matched speed',Descrizione=(folder/'comparison.json').read_text(),**{'Artefatti dettagliati':[art]}))
        experiment=json.loads((repo/'experiments/results/task45e_protocol_v1/preregistration_receipt.json').read_text())['records']['experiments-giada-task45e-matrix-v1']
        put('findings','result',Nome='45e recovered evaluation and full45d comparison',Esito='Misto',Esperimenti=[experiment],Osservazioni=obs,Risultato=json.dumps(r['report']),Limitazioni='No new training/selection/thresholds. Completion is not scientific qualification; original45e3h limit and41limits retained; no automatic46/47 or speedup claim.')
    m.commit_batch()
    if a.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False,evaluation_only=True,selection_recomputed=False)
    write_json(folder/('result_memory_receipt.json' if a.result else 'memory_receipt.json'),receipt)
    if not a.result:write_json(folder/'preregistration_receipt.json',receipt)
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45e evaluation memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45e evaluation schema valid; no writes')
