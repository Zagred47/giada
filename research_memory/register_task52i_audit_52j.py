import gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;F=repo/'experiments/results/task52j_protocol_v1';prior=json.loads((repo/'experiments/results/task52i_protocol_v1/preregistration_receipt.json').read_text());pre=json.loads((F/'local_preflight.json').read_text());assert pre['valid']
    p=json.loads((repo/'experiments/task52j_preregistration_v1.json').read_text());source=repo/'artifacts/task52i_kaggle_external/task52i_final.zip';assert hashlib.sha256(source.read_bytes()).hexdigest()==p['warmstart']['source_sha256'];m=IncrementalBatchMirror();assert m.baseline_hash==prior['verification']['snapshot_hash'];ids={}
    def put(table,key,**fields):
        code=f'{table}-giada-task52i-audit52j-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    obs=[]
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None;report=json.loads(z.read('final_report.json'));freeze=json.loads(z.read('selection_freeze.json'))
        for name in z.namelist():
            if not name.endswith(('.json','.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();dest=repo/'experiments/results/task52i_audit_v1'/(name if name.endswith('.gz') else name+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with gzip.open(dest,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts',name,Nome='52i completed '+name,Tipo='Report',Versione='v1',Percorso=dest.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(dest.read_bytes()).hexdigest(),'Dimensione byte':dest.stat().st_size});obs.append(put('observations',name,Nome='52i completed '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    oldexp=prior['records']['experiments-giada-task52h-audit52i-52i-v1'];finding=put('findings','52i',Nome='52i partial retention, unresolved autonomousV',Esito='Misto',Esperimenti=[oldexp],Osservazioni=obs,Risultato=json.dumps(dict(finite=report['finite_parameters'],selected_updates=freeze['updates'],scientific_acceptance_passed=False,best_1000ms_voltage_rmse=report['summary']['combined']['compact']['1000']['pooled_voltage_rmse'])),Limitazioni='All arms fail full quality; no eligible600/800checkpoint; fit-current guard can preserve currents without preserving autonomousV. No independent confirmation.')
    m.local_upsert('experiments',{'Stato':'Concluso'},record_id=oldexp)
    for code,rid in prior['records'].items():
        if code.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    exp=put('experiments','52j',Nome='52j final guard coverage diagnostic',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Distinguish current time coverage from missing autonomousV/Ca acceptance; last52subtask.'})
    proto=put('protocols','52j',Nome='52j current horizon x coupled guard',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':p['selection'],'Regole di arresto':'400attempts; stop nonfinite or contract error. After audit proceed53 with limitations; no52k.'})
    claims=[]
    for key,claim,contrast in [('coverage','128tick fit guard misses slow current drift','512vs128guard, same thresholds'),('coupling','Current-only acceptance misses autonomous V/Ca sensitivity','Coupled fit-only acceptance vs current-only'),('interaction','Broader current guard and coupled guard jointly preserve useful learning','Factorial interaction; unchanged full validation decisive')]:claims.append(put('claims',key,Nome='52j '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=claim,Limiti=contrast+'; reused data, not independent confirmation.'))
    put('experiments','52j',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52j '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52j '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired')
    put('observations','preflight',Nome='52j verified tests',Descrizione=json.dumps(pre));put('decisions','52j',Nome='Close52refinements after52j',Esito='Continuare',Motivazione='User authorized final52j. No threshold or abstraction change. No52k; transition53 with unresolved limitations after final audit.',Risultati=[finding])
    m.commit_batch();v=m.verify_incremental(prior['verification']);assert v['valid'];write_json(F/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));print('52i audit and52j registered',flush=True)
if __name__=='__main__':main()
