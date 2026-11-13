"""Preregister45e and preserve the audit; SQLite payload-v2 only."""
import argparse,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task45e_protocol_v1';folder.mkdir(exist_ok=True)
    prev=folder/'preregistration_receipt.json' if a.result else repo/'experiments/results/task45d_evaluation_v1/result_memory_receipt.json'
    baseline={} if a.validate_only else json.loads(prev.read_text())['verification'];m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task45e-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    p=json.loads((repo/'experiments/task45e_preregistration_v1.json').read_text())
    exp=put('experiments','matrix',Nome='45e C1-anchored repair, support and trajectory objective',Tipo='Esplorativo',Stato='Concluso' if a.result else 'Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate loss of prior heads from genuine Nap regression; compare FIT coverage/path objective and Ih rate-only versus equilibrium+rate; separately benchmark actual repaired models.'})
    if not a.result:
        protocol=put('protocols','matrix',Nome='45e identity-preserving ionic repair',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Unchanged per-identity quality/regression/DEV/stress gates. No threshold/abstraction changes. Off-target tensors bitwise identical toC1. Exact execution preflight separate from biological qualification.','Regole di arresto':'Stop technical interpretation on GPU/input/parity/finiteness/reference mismatch; preserve all partial output, no automatic restart or promotion.'})
        for arm in p['coupled_arms']:
            for mode in ('imposed','autonomous'):
                key=arm+'-'+mode;rr=put('arms',key,Nome='45e '+key,Ruolo='Baseline' if arm in ('none','C0','C1','R41','R111') else 'Trattamento',Protocollo=[protocol]);put('runs',key,Nome='45e '+key,Stato='Pianificata',Braccio=[rr],Seed='17/29/43; paired geometry/protocol streams;0.1ms;100/500/1000ms; no47')
        claims=[put('claims',h['id'],Nome='45e '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['contrast'],Limiti='Preregistered discriminating contrast, not an established explanation. '+p['execution_scope']) for h in p['hypotheses']]
        put('experiments','matrix',Ipotesi=claims)
        audit=(folder/'development_audit.json').read_text();obs=put('observations','45d-development-audit',Nome='45d identity audit:18/19 Nap-arm regressions target untouched Ca_LVAst.h',Descrizione=audit)
        put('findings','audit',Nome='45d composition confound narrows45e hypotheses',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='Archived Ncap/Nline each lose19 baseline-passing DEV identities:18tail_ramp/Ca_LVAst.h plus1staircase/Nap. Source materializes fromf011 rather thanC1.18Ca failures must not be attributed to Nap damage; P tests actual restoration. Remaining Nap case still requires repair.',Limitazioni='Opened45dDEV, not independent evidence; old failed nomination remains unchanged. No claim of solved Ih/quality/speed.')
    else:
        receipt=json.loads((repo/'artifacts/task45e_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid']
        art=put('artifacts','output',Nome='45e persisted output',Tipo='Report',Versione='v1',Percorso=receipt['local'],**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']});obs=[]
        with zipfile.ZipFile(receipt['local']) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):obs.append(put('observations','raw-'+name,Nome='45e '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        obs.append(put('observations','paired-comparison',Nome='45e all-arm paired audit and measured execution costs',Descrizione=(folder/'comparison.json').read_text(),**{'Artefatti dettagliati':[art]}))
        put('findings','result',Nome='45e all-arm outcome awaits comparative interpretation',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(receipt['report']),Limitazioni='Completion is not release qualification; original41 limits, failed controls and any partial blocks remain explicit.')
        for arm in p['coupled_arms']:
            for mode in ('imposed','autonomous'):put('runs',arm+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if a.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/('result_memory_receipt.json' if a.result else 'preregistration_receipt.json'),dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45e SQLite verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45e schema valid; no writes')
