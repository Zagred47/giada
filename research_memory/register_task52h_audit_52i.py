import gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;F=repo/'experiments/results/task52i_protocol_v1';prior=json.loads((repo/'experiments/results/task52h_protocol_v1/preregistration_receipt.json').read_text());pre=json.loads((F/'local_preflight.json').read_text());assert pre['valid']
    p=json.loads((repo/'experiments/task52i_preregistration_v1.json').read_text());source=repo/'artifacts/task52h_kaggle_external/task52h_final.zip';assert hashlib.sha256(source.read_bytes()).hexdigest()==p['warmstart']['source_sha256']
    m=IncrementalBatchMirror();assert m.baseline_hash==prior['verification']['snapshot_hash'];ids={}
    def put(table,key,**fields):
        code=f'{table}-giada-task52h-audit52i-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    observations=[]
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None;report=json.loads(z.read('final_report.json'));freeze=json.loads(z.read('selection_freeze.json'))
        for name in z.namelist():
            if not name.endswith(('.json','.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();dest=repo/'experiments/results/task52h_audit_v1'/(name if name.endswith('.gz') else name+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with gzip.open(dest,'wt',encoding='utf-8') as stream:stream.write(text)
            art=put('artifacts',name,Nome='52h completed '+name,Tipo='Report',Versione='v1',Percorso=dest.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(dest.read_bytes()).hexdigest(),'Dimensione byte':dest.stat().st_size})
            observations.append(put('observations',name,Nome='52h completed '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    oldexp=prior['records']['experiments-giada-task52g-audit52h-52h-v1']
    finding=put('findings','52h',Nome='52h eligible intermediate checkpoints but no final quality pass',Esito='Misto',Esperimenti=[oldexp],Osservazioni=observations,Risultato=json.dumps(dict(finite=report['finite_parameters'],updates=800,selected_updates=freeze['updates'],selected_nonfallback=sum(x>0 for x in freeze['updates']),scientific_acceptance_passed=False)),Limitazioni='5/12 models select trained weights; no arm passes complete quality. Late training loses eligibility. Same reused evaluation; not independent confirmation.')
    m.local_upsert('experiments',{'Stato':'Concluso'},record_id=oldexp)
    for code,rid in prior['records'].items():
        if code.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    exp=put('experiments','52i',Nome='52i preserving learned solutions',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate late step size and finite-step current regression using same52h warmstarts; unchanged10states and quality guards.'})
    proto=put('protocols','52i',Nome='52i decay x fit-only step acceptance',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':p['selection'],'Regole di arresto':'Stop nonfinite/contract failure. No threshold relaxation or52k; after52j proceed53 with limitations.'})
    claims=[]
    for key,statement,contrast in [('step_size','Late update size prevents retention of accurate solutions','decay vs constant; same initialization and gradients'),('finite_step','Finite Adam steps damage individual currents despite aggregate protection','actual step acceptance vs none; fit-only panel excluded from gradient training'),('interaction','Decay and finite-step protection together preserve useful learning','2x2 interaction; original development and full coupled evaluation decisive')]:
        claims.append(put('claims',key,Nome='52i '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,Limiti=contrast+'; fit-panel success alone is not sufficient.'))
    put('experiments','52i',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52i '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52i '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired')
    put('observations','preflight',Nome='52i local verification',Descrizione=json.dumps(pre))
    put('decisions','52i',Nome='Proceed52i with original thresholds and52j ceiling',Esito='Continuare',Motivazione='User approved. Same52h worst_current_guard selected warmstarts[300,0,300]; reset optimizer equally. Never use development/fresh for step acceptance. Kaggle and external backups; DB local.',Risultati=[finding])
    m.commit_batch();v=m.verify_incremental(prior['verification']);assert v['valid'];write_json(F/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));print('52h audit and52i preregistration verified',flush=True)
if __name__=='__main__':main()
