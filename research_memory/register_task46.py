"""Task46 exploratory progression, frozenP and matched interventions; SQLite only."""
import argparse,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task46_protocol_v1';folder.mkdir(exist_ok=True)
    prev=folder/'preregistration_receipt.json' if a.result else repo/'experiments/results/task45e_evaluation_v1/memory_receipt.json'
    baseline={} if a.validate_only else json.loads(prev.read_text())['verification'];m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task46-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    p=json.loads((repo/'experiments/task46_preregistration_v1.json').read_text())
    exp=put('experiments','spatial-cost',Nome='Task46 frozenP spatial interventions and cost frontier',Tipo='Esplorativo',Stato='Concluso' if a.result else 'Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Frozen level-one transfer under location, density and channel block interventions; distinguish local approximation from coupling feedback and resident ionic/axial/transfer costs.45e remains incomplete.'})
    if not a.result:
        protocol=put('protocols','spatial-cost',Nome='Task46 paired native/formula/C1/P/41r',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Unchanged current/gate/voltage/calcium/event/native-floor gates. Paired identity judgments, not macro-average waiver. Timing and cost scope separate. No automatic47or release qualification.','Regole di arresto':'Wrong input/P hash, nonfinite state, failed execution parity stop technical interpretation. Native floor failures remain excluded judgments; no threshold changes, no refit. Preserve complete/partial output.'})
        for arm in p['arms']:
            for mode in ('imposed','autonomous'):
                key=arm+'-'+mode;rr=put('arms',key,Nome='46 '+key,Ruolo='Trattamento' if arm=='P' else 'Baseline',Protocollo=[protocol]);put('runs',key,Nome='46 '+key,Stato='Pianificata',Braccio=[rr],Seed='17/29/43 frozen model seeds; two geometries,24matched cases each;0.1ms and100/500/1000ms prefixes; no47')
        claims=[put('claims',h['id'],Nome='46 '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['contrast'],Limiti='Prospective exploratory contrast, not established causal explanation. Same abstraction/thresholds; incomplete45e and original41limits remain.') for h in p['hypotheses']]
        put('experiments','spatial-cost',Ipotesi=claims)
        put('observations','freeze-proof',Nome='P reconstructed bitwise from existing45cJ and45dNline inputs',Descrizione=(folder/'reconstruction_receipt.json').read_text())
        put('decisions','exploratory-progression',Nome='User proceeds to46without claiming45e completion',Esito='Continuare',Motivazione='Explicit instruction: vai task46. Freeze45eP, accept known limits for exploration only.45e recovery deferred, not scientifically passed. Use existing Kaggle inputs to reconstruct identicalP; no private payload upload, no training/selection, no47authorization.',**{'Data':'2026-11-13'})
        put('observations','inherited-errors',Nome='Inherited45e limits retained for46interpretation',Descrizione=json.dumps(dict(original45_P_passed=71,original45_total=72,partial_hyper_P_passed=52,partial_hyper_total=54,Ih_offset_ms=.9,registered_offset_limit_ms=.3,hyper_gate_max_abs=.04748988151550293,registered_gate_max_abs=.03,known_failed_case_voltage_rmse_mv=[.000045752015115611404,.00015576560335979177,.0001276119443397368],original45e_completed_blocks=10,total_blocks=18,missing=['onehypergeometry','regression44','stress','cost'],original45e_archive_sha256=p['freeze']['source45e_sha256'],not_a_release=True)))
    else:
        receipt=json.loads((repo/'artifacts/task46_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid']
        art=put('artifacts','output',Nome='Task46 persisted output',Tipo='Report',Versione='v1',Percorso=receipt['local'],**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']});obs=[]
        with zipfile.ZipFile(receipt['local']) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):obs.append(put('observations','raw-'+name,Nome='46 '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        obs.append(put('observations','comparison',Nome='46 paired geometry/intervention/timing/cost audit',Descrizione=(folder/'comparison.json').read_text(),**{'Artefatti dettagliati':[art]}))
        put('findings','result',Nome='46 exploratory spatial transfer and cost outcome',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(receipt['report']),Limitazioni='Original45e incomplete and41limits retained; no automatic47release/speedup. Native-invalid cases are not model evidence; timing and cost are scoped diagnostics.')
        for arm in p['arms']:
            for mode in ('imposed','autonomous'):put('runs',arm+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if a.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/('result_memory_receipt.json' if a.result else 'preregistration_receipt.json'),dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('46 memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('46 schema valid; no writes')
