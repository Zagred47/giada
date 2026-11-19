import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51j_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task52_preregistration_v1.json').read_text())
    archive=repo/'artifacts/task51j_kaggle_external/task51j_final.zip';receipt=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51j-confirmation-v1'];dest=repo/'experiments/results/task51j_audit_v1';dest.mkdir(exist_ok=True);obs=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as handle:handle.write(text)
            art=put('artifacts','51j-result',name,Nome='51j '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','51j-result',name,Nome='51j full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    comparison=(dest/'comparison.json').read_text()
    obs.append(put('observations','51j-result','comparison',Nome='51j complete comparison with51i',Descrizione=comparison))
    finding=put('findings','51j-result','comparison',Nome='51j confirms static preservation, not recurrent success',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='All checkpoints finite, final and checkpoint archives verified. New seeds61/73/89,48freshcases perseed. SGD final openingRMSE9.12345e-9, Adam epsilon1e-4 final1.07557e-8; both144/144 current-amplitude cases. DefaultAdam final5.12889e-5 also144/144 but drifts much more. SGD selects final5000, high-epsilon Adam selects1. Frozen control9.17310e-9. Static preservation confirmed without retuning.',Limitazioni='Only static full-state readout; compact recurrent dynamics from50/51 remain unqualified. No closed-loop or deployment speedup conclusion.')
    put('experiments','51j','confirmation',Stato='Concluso')
    for arm in ('frozen','adam_default','adam_large_eps','sgd'):put('runs','51j',arm,Stato='Completata')
    put('lessons','51j-result','learnability',Nome='Close static readout branch and test coupled compressed state',Lezione='SGD and high-epsilon Adam preserve the calibrated static solution on new data; this does not repair latent dynamics. Task52 must distinguish inherited recurrence error, static readout error and feedback amplification with frozen models and causal controls.',Risultati=[finding],Limiti='No new51subtasks; no silent promotion of full-state readout to compressed model.')
    e=put('experiments','52','closed-loop',Nome='Task52 exploratory compressed-state closed loop',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Frozen51compact + frozen51j readout controls, autonomous/imposed, Ca/family oracles and interventions.'})
    protocol=put('protocols','52','closed-loop',Nome='52 projection and preservation matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Original amplitude gates unchanged; report initial/final/selected separately.','Regole di arresto':'Nonfinite/hash failure stops, preserve checkpoints.'})
    claims=[put('claims','52',h['id'],Nome='52 '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','52','closed-loop',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','52',arm,Nome='52 '+arm,Ruolo='Trattamento',Protocollo=[protocol]);put('runs','52',arm,Nome='52 '+arm,Stato='Pianificata',Braccio=[br],Seed='17,29,43 frozen model seeds')
    put('decisions','52','proceed',Nome='Proceed52 optimization diagnosis',Esito='Continuare',Motivazione='Explicit user request: unchanged thresholds and abstraction, privateKaggle and external backups.',Risultati=[finding])
    put('decisions','52','close51-proceed52',Nome='Close51series and proceed52',Esito='Continuare',Motivazione='Explicit user decision: after51j proceed52 without more51subtasks; preserve all unresolved recurrent limits.',Risultati=[finding])


    print('51j results and52 plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task52_protocol_v1';folder.mkdir(exist_ok=True);r=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',r);write_json(dest/'memory_receipt.json',r);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51j/52 DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51j/52 schema valid')
