import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52f_resume_colab_v1/memory_receipt.json').read_text());source=json.loads((repo/'experiments/results/task52f_protocol_v1/preregistration_receipt.json').read_text());dest=repo/'experiments/results/task52f_audit_v1';p=json.loads((repo/'experiments/task52g_preregistration_v1.json').read_text());design=json.loads((repo/'experiments/task52g_hypotheses_v1.json').read_text());audit=json.loads((dest/'audit.json').read_text())
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={};obs=[]
    if not a.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52f-audit52g-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    with zipfile.ZipFile(audit['backup']['local']) as z:
        for name in z.namelist():
            if not (name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();path=dest/(name if name.endswith('.gz') else name+'.gz');path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with gzip.open(path,'wt',encoding='utf-8') as stream:stream.write(text)
            art=put('artifacts',name,Nome='52f completed '+name,Tipo='Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
            obs.append(put('observations',name,Nome='52f completed '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    obs.append(put('observations','audit',Nome='52f independent finite and rejection audit',Descrizione=json.dumps(audit)))
    oldexp=source['records']['experiments-giada-task52f-encoder-credit-v1']
    finding=put('findings','52f',Nome='52f longer credit helps but cannot preserve all currents',Esito='Misto',Esperimenti=[oldexp],Osservazioni=obs,Risultato=json.dumps(design['facts']),Limitazioni=audit['qualification'])
    if not a.validate_only:
        m.local_upsert('experiments',{'Stato':'Concluso'},record_id=oldexp)
        for key,rid in source['records'].items():
            if key.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    exp=put('experiments','52g',Nome='52g current preservation x gradient conflict',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Retain long-gradient V/Ca benefit without channel regressions; distinguish objective incentives from gradient conflict at fixed abstraction.'})
    proto=put('protocols','52g',Nome='52g paired2x2 full-credit protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Original quality, per-channel/V/Ca nonregression and fallback unchanged. All12members,800updates,identical fit/development streams.','Regole di arresto':'Stop nonfinite or contract failure; no automatic restart or53; cap52j.'})
    claims=[]
    for h in design['hypotheses']:claims.append(put('claims',h['id'],Nome='52g '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['test']+' Falsifier: '+h['falsifier']))
    put('experiments','52g',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52g '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52g '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired')
    put('lessons','52f',Nome='Distinguish rejected learning trajectories from deployed fallback',Lezione='52f all96seed-checkpoints rejected. Encoder fit loss improves14.8% but downstream effect is small; full-credit containsNap drift yet regresses othercurrents. Selected final reports are identical baseline fallbacks, not proof that training interventions had no effect.',Risultati=[finding])
    put('decisions','52g',Nome='Proceed52g, retain52j ceiling and frozen thresholds',Esito='Continuare',Motivazione='User authorized52g,DB,push,externalbackup. Train-only guard andprojection; inference architecture unchanged. No52k or automatic53.',Risultati=[finding])
    m.commit_batch()
    if a.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];folder=repo/'experiments/results/task52g_protocol_v1';folder.mkdir(exist_ok=True);write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));write_json(dest/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52f audit and52g preregistration verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52f/52g schema valid')
