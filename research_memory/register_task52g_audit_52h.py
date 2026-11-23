import gzip,hashlib,json,zipfile
import numpy as np
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;F=repo/'experiments/results/task52h_protocol_v1';old=json.loads((repo/'experiments/results/task52g_resume640_v1/preregistration_receipt.json').read_text());pre=json.loads((F/'local_preflight.json').read_text());assert pre['valid']
    p=json.loads((repo/'experiments/task52h_preregistration_v1.json').read_text());path=repo/'artifacts/task52g_kaggle_external/task52g_final.zip';backup=json.loads((path.parent/'external_backup_receipt.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==backup['remote']['sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None;report=json.loads(z.read('final_report.json'));curve=json.loads(z.read('learning_curve.json'));base=json.loads(z.read('validation_baseline.json'));freeze=json.loads(z.read('selection_freeze.json'))
    last=curve[-1];ch=np.array(last['channel_errors']);b=np.array(base['channels']);audit=dict(execution_valid=report['execution_valid'],finite=report['finite_parameters'],updates=report['training_updates'],all_fallback=all(freeze['baseline_fallback']),Ih_regressions=int((ch[:,5]>b[:,5]*(1+1e-6)+1e-12).sum()),last_regressed_channels=(ch>b*(1+1e-6)+1e-12).sum(1).tolist(),backup=backup)
    m=IncrementalBatchMirror();assert m.baseline_hash==old['verification']['snapshot_hash'];ids={}
    def put(table,key,**fields):
        code=f'{table}-giada-task52g-audit52h-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    obs=put('observations','52g',Nome='52g completed800: all fallback, Ih12of12regressions',Descrizione=json.dumps(audit))
    detailed=[]
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(('.json','.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode()
            dest=repo/'experiments/results/task52g_audit_v1'/(name if name.endswith('.gz') else name+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with gzip.open(dest,'wt',encoding='utf-8') as stream:stream.write(text)
            art=put('artifacts',name,Nome='52g completed '+name,Tipo='Report',Versione='v1',Percorso=dest.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(dest.read_bytes()).hexdigest(),'Dimensione byte':dest.stat().st_size})
            detailed.append(put('observations',name,Nome='52g completed '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    source=json.loads((repo/'experiments/results/task52g_protocol_v1/preregistration_receipt.json').read_text());oldexp=source['records']['experiments-giada-task52f-audit52g-52g-v1']
    finding=put('findings','52g',Nome='52g guards reduce some regressions but no eligible checkpoint',Esito='Misto',Esperimenti=[oldexp],Osservazioni=[obs]+detailed,Risultato=json.dumps(audit),Limitazioni='Training trajectories differ; final selected fallback outputs are identical. No scientific promotion. Reused evaluation panel.')
    m.local_upsert('experiments',{'Stato':'Concluso'},record_id=oldexp)
    exp=put('experiments','52h',Nome='52h targeted Ih coverage x worst-current guard',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate fit coverage from mean-penalty dilution without changing inference abstraction or quality thresholds.'})
    proto=put('protocols','52h',Nome='52h paired factorial with feasibility gate',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':p['selection'],'Regole di arresto':'Stop full training if bounded fit-only Ih feasibility fails for any seed, or nonfinite/contract failure. Cap52j.'})
    claims=[]
    for name,claim,test in [('coverage','Ih drift reflects insufficient coverage of slow hyperpolarization/recovery windows','Targeted support contrast; same total updates and batch.'),('dilution','Mean channel penalty dilutes the worst current regression','Mean-plus-max contrast; original per-channel eligibility unchanged.'),('interaction','Coverage plus concentrated protection can retain V/Ca improvements without regressions','Factorial interaction and all11channels/V/Ca original guards.')]:
        claims.append(put('claims',name,Nome='52h '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=claim,Limiti=test+' Failure to improve eligible checkpoints weakens this explanation.'))
    put('experiments','52h',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52h '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52h '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired')
    put('observations','preflight',Nome='52h local tests and contract',Descrizione=json.dumps(pre))
    put('decisions','52h',Nome='Proceed52h on Kaggle with fit-only gate',Esito='Continuare',Motivazione='User approved four paired arms and bounded feasibility gate. Same10states, original52d initialization, frozen encoder/readout, thresholds and52j ceiling. No Colab launch.',Risultati=[finding])
    m.commit_batch();v=m.verify_incremental(old['verification']);assert v['valid'];write_json(F/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));print('52g audit and52h registered and verified',flush=True)
if __name__=='__main__':main()
