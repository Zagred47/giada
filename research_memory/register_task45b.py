"""45b authorized protocol/result registration, SQLite only, no remote DB."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task45b_protocol_v1';p=json.loads((repo/'experiments/task45b_preregistration_v1.json').read_text())
    previous=json.loads((repo/'experiments/results/task45_audited_v1/memory_receipt.json').read_text())
    baseline={} if args.validate_only else (json.loads((folder/'preregistration_receipt.json').read_text())['verification'] if args.result else previous['verification'])
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task45b-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','rate-coverage',Nome='45b frozen41r rate attribution and constrained coverage repair',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate invalid stimulus, Nap_h rate regression, Ih rate/inf bias, support versus budget and timing conditioning. No abstraction/threshold change.'})
    if not args.result:
        protocol=put('protocols','factorial',Nome='45b native calibration, atomic identification, learned fit, coupled2x2x2',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Original gates at every prefix; valid physical/native references; no new previously passing stress or44/45 failures. Nomination only atomic DEV before coupled outcomes.','Regole di arresto':'Stop for immutable weight/source mismatch, nonfinite weights, failed preflight/null-control. Preserve partial outputs. No automatic relaunch or posthoc threshold adjustment.'})
        for hypothesis in p['hypotheses']:
            claim=previous['records']['claims-giada-task45-audit-'+hypothesis['id']+'-v1']
            put('observations','planned-'+hypothesis['id'],Nome='45b registered discriminator '+hypothesis['id'],Descrizione=json.dumps(hypothesis))
        for arm in p['arms']:
            for mode in ('imposed','autonomous'):
                key=arm+'-'+mode;aa=put('arms',key,Nome='45b '+key,Ruolo='Baseline' if arm in ('none','f000') else 'Trattamento',Protocollo=[protocol])
                put('runs',key,Nome='45b '+key,Stato='Pianificata',Braccio=[aa],Seed='17/29/43; all original45, safe common-current levels,44regressions; no47')
        for support in p['fit']['supports']:
            for budget in p['fit']['checkpoints']:
                aa=put('arms',support+'-'+str(budget),Nome='45b FIT '+support+' checkpoint'+str(budget),Ruolo='Trattamento',Protocollo=[protocol])
                put('runs',support+'-'+str(budget),Nome='45b paired mini scaling '+support+' '+str(budget),Stato='Pianificata',Braccio=[aa],Seed='17/29/43; shared frozen initial weights, independent moments, same full-batch indices')
        put('decisions','authorized',Nome='User authorizes45b on frozen41r reference, new learned sidecars diagnostic',Esito='Continuare',Motivazione='Explicit user approval supersedes proposed-not-launched status prospectively only.41r not modified; no new thresholds, architecture,47 access or release claim. DB backup local; one Kaggle console run with external ZIP verification.',**{'Data':'2026-11-10'})
    else:
        receipt=json.loads((repo/'artifacts/task45b_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent'];path=Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='45b full persisted final or partial output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        observations=[]
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):observations.append(put('observations','raw-'+name,Nome='45b raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        put('findings','outcome',Nome='45b factorial outcome; scoped audit required',Esito='Misto',Esperimenti=[exp],Osservazioni=observations,Risultato=json.dumps(receipt['report']),Limitazioni='No automatic sidecar promotion, original41 accepted limits retained. Technical failures separated from scientific failures; original45 invalid physical regimes retained. No speedup/47 claim.')
        for arm in p['arms']:
            for mode in ('imposed','autonomous'):put('runs',arm+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verification,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45b SQLite verified/exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45b schema valid; no writes')
