"""Register all audited IE-P6 measurements and scoped hypothesis outcomes."""
import hashlib
import json
import statistics
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/ie_p6_kaggle_v1'
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    r=read(folder/'final_report.json');audit=read(folder/'result_audit.json');spec=read(folder/'preregistration.json');launch=read(repo/'experiments/ie_p6_kaggle_launch_v1.json')
    assert r['valid'] and audit['valid'] and audit['timing_cells_verified']==210 and audit['forced_path_maps_replayed_cpu']==35
    assert r['code_revision']==audit['code_revision']==launch['verified_code_revision']==launch['expected_code_revision']
    assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==r['selection']['checkpoint_sha256']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p6-{suffix}-v1',**fields})['record_id']
    artifacts=[]
    for p in sorted(folder.iterdir()):
        artifacts.append(put('artifacts','result-'+p.name,Nome='IE-P6 result '+p.name,Tipo='Report',Versione='v1',Percorso=p.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    for relative in ('experiments/ie_p6_result_v1.md','scripts/audit_ie_p6_result.py','experiments/ie_p6_kaggle_launch_v1.json'):
        p=repo/relative
        artifacts.append(put('artifacts','result-'+p.name,Nome='IE-P6 '+p.name,Tipo='Report',Versione='v1',Percorso=relative,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    runs={}
    for name in list(spec['arms'])+['oracle']:
        arm=put('arms',name,Nome='IE-P6 '+name)
        runs[name]=put('runs',name,Nome='IE-P6 Kaggle v1 '+name,Stato='Completata',Braccio=[arm],**{'Artefatti prodotti':artifacts,'Configurazione effettiva':json.dumps(launch),'Validità tecnica':'210preflights pass, zero graph failures,35maps replayed CPU. Six local maps pass registered quality and primary graph-cost gates, not a native confirmation.'})
    def family(name):return 'neural_table025' if name.endswith('_table025') else name.split('_seed')[0]
    evaluations={}
    for key,unit,target in [('wall','ms','Median30blocks x20calls; >=10%paired savings atb642 withpositive interval, plus independent quality'),('fresh','dimensionless','RMSE<=.001,max<=.01; allother gates required'),('path','dimensionless','Each6family RMSE<=.005,max<=.03; occupancy0'),('historical','mV','OpenedTask35 diagnostic: Vrmse<=.05,Vmax<=.1,Camax<=5e-6mM; no domain exits'),('oracle','mV','IE-P5standard parent-only exact swaps, not selection or deployment eligible')]:
        metric=put('metrics',key,Nome='IE-P6 '+key,Famiglia='Regressione',Direzione='Minimizzare',**{'Unità':unit})
        evaluations[key]=put('evaluations',key,Nome='IE-P6 '+key,Metrica=[metric],Versione='v1',Ruolo='Primaria' if key in ('fresh','path') else 'Diagnostica',Target=target)
    observations=[]
    def observation(key,name,value,details,run):
        observations.append(put('observations',key+'-'+name,Nome='IE-P6 '+key+' '+name,Valore=value,Run=[runs[run]],**{'Specifica di valutazione':[evaluations[key]],'Artefatti dettagliati':artifacts,'Descrizione':json.dumps(details)}))
    for row in r['rows']:
        observation('wall',f"{row['mode']}-{row['name']}-b{row['batch']}",statistics.median(row['timings']['wall_ms']),row,family(row['name']))
    for name,f in r['fresh'].items():
        observation('fresh',name,f['fresh_gate']['rmse'],f,family(name))
        observation('path',name,f['forced_path_gate']['rmse'],f,family(name))
        observation('historical',name,r['historical'][name]['voltage']['rmse'],{'metrics':r['historical'][name],'all_quality_gates':r['quality_qualified'][name]},family(name))
    oracle=read(folder/'diagnostic_oracle_attribution.json');assert not oracle['selection_eligible']
    for name,details in oracle['metrics'].items():observation('oracle',name,details['voltage']['rmse'],details,'oracle')
    assert len(observations)==333
    finding=put('findings','outcome',Nome='IE-P6 learned local maps pass scoped quality and graph cost; temporal objective fails',Esito='Misto',Osservazioni=observations,**{'Risultato':(repo/'experiments/ie_p6_result_v1.md').read_text(encoding='utf-8'),'Limitazioni':spec['exclusions']+' Local maps are learned tables, not dense neural speedup or fewer weights. Analytic LUT remains faster; one device/run and opened historical traces. Budget/data/reset/LR not individually isolated.'})
    decisions={
        'H_optimization':('Supportata nel dominio','Sostiene','Acontinuation improves fresh joint score3/3 by12.1–34.7%, quality2/3; not isolated budget cause; later local checkpoints can degrade.'),
        'H_coverage':('Supportata nel dominio','Sostiene','B-A registered joint worst-case score improves21.9–25.3%,median24.2%,direction3/3. Mean gate/path errors worsen3/3 and historicalV improves1/3; NOT universal accuracy improvement, quality1/3.'),
        'H_temporal':('Contraddetta nel dominio','Contraddice','C worse than A3/3 and D worse than B3/3 on joint fresh gate/path score. Scope is tested8step objective/weight, not recurrent learning generally.'),
        'H_interaction':('Contraddetta nel dominio','Contraddice','Positive interactions0.0823/0.1080/0.0505 for lower-is-better score: no proposed complementarity.'),
        'H_conflict':('Indeterminata','Indeterminata','Logit gradient84–219xbase and near-orthogonal; negativecosine only1/3. Scale imbalance plausible, not causal proof that conflicting gradients caused P5failure.'),
        'H_representation':('Supportata nel dominio','Sostiene','Six local learned maps pass ALL quality and registered10%median graph-cost criterion.11,084/21,964bytes vs9,420MLP. Analytic LUT remains faster25.3%; no ordinary-backend or end-to-end speed advantage claimed.'),
        'H_attribution':('Supportata nel dominio','Sostiene','On new P5parents exact z swaps strongly improve V/Ca3/3; calcium-group correction also helps Ca. Conditional oracle evidence, not unique attribution or inference candidate.')}
    for key,(status,outcome,reason) in decisions.items():
        claim=put('claims',key,Nome='IE-P6 '+key,Stato=status)
        put('evidence',key,Nome='IE-P6 result '+key,Esito=outcome,Argomentazione=reason,**{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],'Limiti e spiegazioni alternative':spec['exclusions'],'Data valutazione':'2026-10-16'})
    put('experiments','repair',Nome='IE-P6 coverage recurrence and learned local operator',Stato='Concluso')
    put('actions','execute',Nome='Execute IE-P6 paired repair and local representation notebook',Stato='Conclusa',Descrizione='Complete and independently audited; no new run or operational promotion automatically authorized.')
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'result_registered':True,'observations':len(observations),'hypotheses_evaluated':len(decisions),'airtable_accessed':False}))

if __name__=='__main__':main()
