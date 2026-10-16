"""Register audited IE-P5 outputs and scoped evidence, SQLite only."""
import hashlib
import json
import statistics
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent; folder=repo/'experiments/results/ie_p5_kaggle_v1'
    r=json.loads((folder/'final_report.json').read_text()); audit=json.loads((folder/'result_audit.json').read_text())
    spec=json.loads((folder/'preregistration.json').read_text()); launch=json.loads((repo/'experiments/ie_p5_kaggle_launch_v1.json').read_text())
    assert r['valid'] and audit['valid'] and audit['forced_path_maps_replayed_cpu']==23 and audit['timing_cells_verified']==138
    assert r['code_revision']==audit['code_revision']==launch['code_revision']
    assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==r['selection']['checkpoint_sha256']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p5-{suffix}-v1',**fields})['record_id']
    artifacts=[]
    for p in sorted(folder.iterdir()):
        artifacts.append(put('artifacts','result-'+p.name,Nome='IE-P5 result '+p.name,Tipo='Report',Versione='v1',Percorso=p.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    for relative in ('experiments/ie_p5_result_v1.md','scripts/audit_ie_p5_result.py'):
        p=repo/relative
        artifacts.append(put('artifacts','result-'+p.name,Nome='IE-P5 '+p.name,Tipo='Report',Versione='v1',Percorso=relative,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    runs={}
    for name in list(spec['arms'])+['oracle-coefficients']:
        arm=put('arms',name,Nome='IE-P5 '+name)
        runs[name]=put('runs',name,Nome='IE-P5 Kaggle v1 '+name,Stato='Completata',Braccio=[arm],**{'Artefatti prodotti':artifacts,'Configurazione effettiva':json.dumps(launch),'Validità tecnica':'138 preflights, zero graph failures; 23 maps independently replayed CPU. Not a quality promotion.'})
    def family(name):
        if name.endswith('_table025'): return 'neural_table025'
        if name.endswith('_table1'): return 'neural_table1'
        return name.split('_seed')[0]
    evaluations={}
    for key,unit,target in [('wall','ms','>=10% paired saving at b642; independently require quality'),('fresh','dimensionless','RMSE<=.001 and max<=.01'),('path','dimensionless','Every family RMSE<=.005 and max<=.03'),('historical','mV','V RMSE<=.05, max<=.1; Ca max<=5e-6mM; opened diagnostic'),('oracle','mV','Old frozen models ONLY; no selection, timing or attribution to new refits')]:
        metric=put('metrics',key,Nome='IE-P5 '+key,Famiglia='Regressione',Direzione='Minimizzare',**{'Unità':unit})
        evaluations[key]=put('evaluations',key,Nome='IE-P5 '+key,Metrica=[metric],Versione='v1',Ruolo='Primaria' if key in ('fresh','path') else 'Diagnostica',Target=target)
    observations=[]
    def observation(key,name,value,details,run):
        observations.append(put('observations',key+'-'+name,Nome='IE-P5 '+key+' '+name,Valore=value,Run=[runs[run]],**{'Specifica di valutazione':[evaluations[key]],'Artefatti dettagliati':artifacts,'Descrizione':json.dumps(details)}))
    for row in r['rows']:
        observation('wall',f"{row['mode']}-{row['name']}-b{row['batch']}",statistics.median(row['timings']['wall_ms']),row,family(row['name']))
    for name,f in r['fresh'].items():
        observation('fresh',name,f['fresh_gate']['rmse'],f,family(name))
        observation('path',name,f['forced_path_gate']['rmse'],f,family(name))
        observation('historical',name,r['historical'][name]['voltage']['rmse'],r['historical'][name],family(name))
    oracle=json.loads((folder/'diagnostic_oracle_attribution.json').read_text())
    assert not oracle['selection_eligible']
    for name,details in oracle['metrics'].items(): observation('oracle',name,details['voltage']['rmse'],details,'oracle-coefficients')
    finding=put('findings','outcome',Nome='IE-P5 standard refit improves accuracy; balanced loses; tabulation trades memory for graph speed',Esito='Misto',Osservazioni=observations,**{'Risultato':(repo/'experiments/ie_p5_result_v1.md').read_text(),'Limitazioni':spec['exclusions']+' Budget/data/Adam-reset confounded versus parent; balanced terms not separately ablated; oracle swaps only old frozen models; no end-to-end speed claim.'})
    for key,status,outcome,reason in [('H_accuracy','Contraddetta nel dominio','Contraddice','Balanced package loses in all three paired seeds; not each term separately.'),('H_representation','Indeterminata','Indeterminata','Fine tables preserve predictions and improve graph cost with 9.25x memory, but no learned candidate passes quality; coarse tables fail conversion.'),('H_localization','Supportata nel dominio','Sostiene','Conditional old-frozen z and calcium-group oracle response; not unique attribution and not tested on new refits.')]:
        claim=put('claims',key,Nome='IE-P5 '+key,Stato=status)
        put('evidence',key,Nome='IE-P5 result '+key,Esito=outcome,Argomentazione=reason,**{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],'Limiti e spiegazioni alternative':spec['exclusions'],'Data valutazione':'2026-10-16'})
    put('experiments','accuracy-execution',Nome='IE-P5 accuracy repair and inference representation',Stato='Concluso')
    put('actions','execute',Nome='Execute IE-P5 matched accuracy and deployment experiment',Stato='Conclusa',Descrizione='Completed and audited; no automatic promotion or new run.')
    mirror.commit_batch(); write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot()); assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'result_registered':True,'airtable_accessed':False}))

if __name__=='__main__': main()
