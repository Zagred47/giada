"""Register audited IE-P4 outcomes, matched observations and scoped evidence."""
import hashlib
import json
import statistics
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/ie_p4_kaggle_v1'
    r=json.loads((folder/'final_report.json').read_text());audit=json.loads((folder/'result_audit.json').read_text())
    spec=json.loads((folder/'preregistration.json').read_text());launch=json.loads((repo/'experiments/ie_p4_kaggle_launch_v1.json').read_text())
    assert audit['valid'] and r['valid'] and audit['timing_cells_verified']==84 and audit['forced_path_members_replayed_cpu']==14
    assert r['code_revision']==launch['code_revision']==audit['code_revision']
    assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==r['selection']['checkpoint_sha256']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p4-{suffix}-v1',**fields})['record_id']
    artifacts=[]
    paths=list(sorted(folder.iterdir()))+[repo/'experiments/ie_p4_result_v1.md',repo/'scripts/audit_ie_p4_result.py',repo/'experiments/ie_p4_kaggle_launch_v1.json']
    for p in paths:
        artifacts.append(put('artifacts','result-'+p.stem,Nome='IE-P4 result '+p.name,Tipo='Report',Percorso=p.relative_to(repo).as_posix(),Versione='v1',**{
            'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    runs={}
    for name in spec['arms']:
        arm=put('arms',name,Nome='IE-P4 '+name)
        runs[name]=put('runs',name,Nome='IE-P4 Kaggle v1 '+name,Stato='Completata',Braccio=[arm],**{
            'Artefatti prodotti':artifacts,'Configurazione effettiva':json.dumps(launch),
            'Validità tecnica':'84preflights pass,84timing cells,zero graph errors,14forced-path maps replayedCPU. Technical validity is not learned-model quality.'})
    wall_metric=put('metrics','wall',Nome='IE-P4 resident core wall time',Famiglia='Regressione',Direzione='Minimizzare',**{'Unità':'ms','Formula':'Median30blocks x20calls; same fixed resident inputs and five-output ABI, compilation excluded.'})
    wall_eval=put('evaluations','wall',Nome='IE-P4 paired resident core cost',Metrica=[wall_metric],Versione='v1',Ruolo='Diagnostica',Target='>=10% saving vs exact vectorized formula atb642 in common graph mode, independently require quality.',**{'Aggregazione e pesi':'Within-run paired bootstrap, not hardware replication.'})
    gate_metric=put('metrics','gate',Nome='IE-P4 gate RMSE',Famiglia='Regressione',Direzione='Minimizzare',**{'Unità':'dimensionless','Formula':'sqrt(mean((prediction-reference)^2));17V-dependent physical gates.'})
    voltage_metric=put('metrics','voltage',Nome='IE-P4 historical closed-loop voltage RMSE',Famiglia='Regressione',Direzione='Minimizzare',**{'Unità':'mV','Formula':'RMSE against openedTask35 native traces over32episodes x400steps. Diagnostic only.'})
    fresh_eval=put('evaluations','fresh',Nome='IE-P4 fresh atomic transitions',Metrica=[gate_metric],Versione='v1',Ruolo='Primaria',Target='RMSE<=.001,max<=.01; other quality gates also required.')
    path_eval=put('evaluations','path',Nome='IE-P4 fresh imposed voltage paths',Metrica=[gate_metric],Versione='v1',Ruolo='Primaria',Target='Each familyRMSE<=.005,max<=.03; no occupancy violations.',**{'Aggregazione e pesi':'24paths400steps,6families; family metrics and per-gate/current/charge probes retained in observation JSON.'})
    voltage_eval=put('evaluations','historical',Nome='IE-P4 openedTask35 integrated diagnostic',Metrica=[voltage_metric],Versione='v1',Ruolo='Diagnostica',Target='Vrmse<=.05,Vmax<=.1,Camax<=5e-6mM; zero voltage-domain exits. Not new confirmation.')
    observations=[]
    for row in r['rows']:
        times=row['timings']['wall_ms'];assert len(times)==30 and min(times)>0
        control=next(x for x in r['rows'] if x['batch']==row['batch'] and x['mode']==row['mode'] and x['name']=='formula_vectorized')['timings']['wall_ms']
        assert abs(statistics.median(1-b/a for a,b in zip(control,times))-row['saving_vs_formula']['median_saved_fraction'])<1e-12
        family=row['name'].split('_seed')[0]
        observations.append(put('observations',f"{row['mode']}-{row['name']}-b{row['batch']}",Nome=f"IE-P4 {row['mode']} {row['name']} b{row['batch']}",Valore=statistics.median(times),Run=[runs[family]],**{
            'Specifica di valutazione':[wall_eval],'Artefatti dettagliati':artifacts,'Descrizione':json.dumps(row)}))
    for name,f in r['fresh'].items():
        family=name.split('_seed')[0]
        for suffix,evaluation,value,details in (
            ('fresh',fresh_eval,f['fresh_gate']['rmse'],f),
            ('path',path_eval,f['forced_path_gate']['rmse'],f),
            ('historical',voltage_eval,r['historical'][name]['voltage']['rmse'],r['historical'][name])):
            observations.append(put('observations',suffix+'-'+name,Nome='IE-P4 '+suffix+' '+name,Valore=value,Run=[runs[family]],**{
                'Specifica di valutazione':[evaluation],'Artefatti dettagliati':artifacts,
                'Descrizione':json.dumps({'metrics':details,'quality_qualified_all_gates':r['quality_qualified'][name]})}))
    finding=put('findings','outcome',Nome='IE-P4 internal supervision helps recurrence; LUT is competitive, MLPs not qualified',Esito='Misto',Osservazioni=observations,**{
        'Risultato':'Internal supervision improves forced paths6/6 and historicalV10.29–16.10x, but fresh one-step3/6 only. All12MLPs fail jointquality and are20.8–29.4% slower than exact formula in graph b642. LUTpassesquality and saves18.57%graph but loses8.60%ordinary. No deployment promotion.',
        'Limitazioni':spec['exclusions']+' Allselected checkpoints10000; more budget is plausible for accuracy, not a remedy for the unchanged inference-graph cost.'})
    evaluations={
        'H_direct':('Contraddetta nel dominio','Contraddice','No tested learned member passes jointquality or10%saving atb642. Scope is this family and budget, not all neural operators.'),
        'H_internal':('Indeterminata','Indeterminata','Mixed: forced-path errors improve6/6 and historicalV10.29–16.10x; fresh atomic one-step improves3/6 only. Strong recurrence signal, not universal accuracy gain.'),
        'H_capacity':('Contraddetta nel dominio','Contraddice','Width64 does not produce a quality-qualified model and costs more. No sufficient cost-error benefit under registered criteria; not a universal anti-scaling claim.'),
        'H_lut':('Supportata nel dominio','Sostiene','LUTdominates theseMLPs in quality and measured cost. Improvement over exact formula depends on backend/batch, not universal.')}
    for name,(status,outcome,reason) in evaluations.items():
        claim=put('claims',name,Nome='IE-P4 '+name,Stato=status)
        put('evidence',name,Nome='IE-P4 result '+name,Esito=outcome,Argomentazione=reason,**{
            'Affermazione valutata':[claim],'Risultati a sostegno':[finding],
            'Limiti e spiegazioni alternative':spec['exclusions'],'Data valutazione':'2026-10-16'})
    put('experiments','discrete',Nome='IE-P4 joint discrete ionic coefficient matrix',Stato='Concluso')
    put('actions','execute',Nome='Execute IE-P4 joint coefficient matrix in one notebook',Stato='Conclusa',Descrizione='Completed and audited. No next experiment or operational promotion automatically authorized.')
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'result_registered':True,'airtable_accessed':False}))

if __name__=='__main__':main()
