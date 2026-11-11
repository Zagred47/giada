"""45c finalized protocol and result registration; mirror SQLite exclusively."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task45c_protocol_v1';p=json.loads((repo/'experiments/task45c_preregistration_v1.json').read_text())
    previous=json.loads((repo/'experiments/results/task45b_audited_v1/final_memory_receipt.json').read_text())
    baseline={} if args.validate_only else (json.loads((folder/'preregistration_receipt.json').read_text())['verification'] if args.result else previous['verification'])
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task45c-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','preserving-kinetics',Nome='45c f011 kinetic repair with functional preservation',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Distinguish Nap compensation and preservation, flat Ih crossing sensitivity and actual Ca_LVAst.h tail error without changing abstraction or quality gates.'})
    if not args.result:
        protocol=put('protocols','same-head-matrix',Nome='45c same-readout refinements and coupled regression matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Original all-prefix gates and per-identity no-regression guards, native-floor/physical-domain eligibility. Fresh DEV-only freeze; report rejected arms without replacing them by baseline.','Regole di arresto':'Pinned source, fixture/hash/ownership, finite or endpoint/null-parity mismatch stops scientific interpretation; preserve partial artifacts. No automatic resubmission or threshold change.'})
        for arm in p['coupled_arms']:
            for mode in ('imposed','autonomous'):
                label=arm+'-'+mode;aa=put('arms',label,Nome='45c '+label,Ruolo='Baseline' if arm in ('none','C0','R41','R111') else 'Trattamento',Protocollo=[protocol])
                put('runs',label,Nome='45c '+label,Stato='Pianificata',Braccio=[aa],Seed='17/29/43; source45b opened native fixtures, all declared prefixes; no47')
        for h in p['hypotheses']:put('observations','planned-'+h['id'],Nome='45c finalized contrast '+h['id'],Descrizione=json.dumps(h))
        put('observations','numeric-design',Nome='45c FIT/DEV and numeric refinement choices finalized',Descrizione=json.dumps(dict(fitting=p['fitting'],cost=p['cost'],transport=p['source_transport'])))
        put('decisions','authorized',Nome='User authorizes executable45c, private Kaggle run and local-only DB backup',Esito='Continuare',Motivazione='Explicit approval on11Nov2026. Operationalizes prior design without altering thresholds, inference map or18states. Corrected Ca_LVAst.h primary target; NaTs2_t.h diagnostic frozen control. No automatic release or46/47 advancement.',**{'Data':'2026-11-11'})
    else:
        receipt=json.loads((repo/'artifacts/task45c_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent'];path=Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='45c full persisted final or partial experiment output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        obs=[]
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):obs.append(put('observations','raw-'+name,Nome='45c raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        put('findings','outcome',Nome='45c preservation and tail-repair result requires scoped audit',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(receipt['report']),Limitazioni='No automatic sidecar promotion, old accepted limits retained. All raw gate/timing/charge and regression results preserved. No independent47 or speedup qualification from evaluator timing.')
        for arm in p['coupled_arms']:
            for mode in ('imposed','autonomous'):put('runs',arm+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verification,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45c SQLite verified/exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45c schema valid; no writes')
