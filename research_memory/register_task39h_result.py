"""Retain complete39h evidence in SQLite, including all negative arms and probes."""
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39h_colab_v1'
    read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    audit=read('result_audit.json');final=read('final_report.json');freeze=read('selection_freeze.json')
    proposal=json.loads((repo/'experiments/task39h_residual_hypotheses.json').read_text())
    assert audit['valid'] and final['valid'] and all(v['all_finite'] for v in audit['numerical_verification'].values())
    m=ValidatedBatchMirror();ids={}
    @lru_cache(None)
    def ref(code):
        rr=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rr)==1,code
        return rr[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39h-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    paths=[p for p in folder.rglob('*') if p.is_file() and not p.name.startswith('memory_')]
    paths += [repo/'artifacts/task39h_primary_external/giada_task39h_result.zip',repo/'experiments/task39h_residual_hypotheses.json',repo/'scripts/analyze_task39h_result.py',Path(__file__),repo/'scripts/probe_task39h_frozen_residuals.py']
    arts={}
    for path in paths:
        rel=path.relative_to(repo).as_posix()
        arts[rel]=put('artifacts',rel,Nome='Task39h '+path.name,Tipo='Checkpoint' if path.suffix=='.pt' else 'Codice' if path.suffix=='.py' else 'Report',Versione='final120000',Percorso=rel,**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    def art(n):return arts[(folder/n).relative_to(repo).as_posix()]
    exp=ref('experiments-giada-task39h-temporal-suite-v1')
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task39h-temporal-suite-v1','Stato':'Concluso','Descrizione':proposal['outcome']})
    runs={}
    for i,a in enumerate(freeze['arms']):
        for j,seed in enumerate(freeze['seeds']):
            code=f'runs-giada-task39h-temporal-{a["name"]}-{seed}-v1';rid=ref(code);runs[('joint_w'+a['name'],seed)]=rid
            m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata','Artefatti prodotti':list(arts.values()),
                'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,chosen_step=freeze['chosen_steps'][i*3+j],revision=read('revision.json'),budget=120000)),
                'Hardware e ambiente':json.dumps(read('execution.json')),
                'Validità tecnica':json.dumps(dict(valid=True,exit_code=0,numerical_verification=audit['numerical_verification'],scientific_decision=final['decision']))})
    for a in ('A0','A6'):
        for seed in freeze['seeds']:
            runs[('joint_wfrozen39g_'+a,seed)]=ref(f'runs-giada-task39g-temporal-{a}-{seed}-retry2-v1')
    baseline=ref('runs-giada-task38-joint-transition-kaggle-local161-v1');obs=[]
    def observe(suffix,data,source,run=None,value=None,count=1,evaluation='quality'):
        fields={'Nome':'Task39h '+suffix,'Descrizione':json.dumps(data,separators=(',',':')),'Numerosità':count,
            'Artefatti dettagliati':[art(source)],'Specifica di valutazione':[ref('evaluations-giada-task39h-temporal-'+evaluation+'-v1')]}
        if run:fields['Run']=[run]
        if value is not None:fields['Valore']=value
        rid=put('observations',suffix,**fields);obs.append(rid)
    counts={}
    for mode,path in [('primary','recurrence_evaluation.json'),('common','common_selection_diagnostic/recurrence_evaluation.json')]:
        rows=read(path);counts[mode]=len(rows);group=defaultdict(list)
        for r in rows:group[(r['model'],r['role'],r['family'],r['seed'])].append(r)
        for (model,role,family,seed),rr in group.items():
            observe(f'{mode}-{model}-{role}-{family}-{seed}',dict(cases=rr,selection_eligible=mode=='primary' and role=='development'),path,
                baseline if model=='frozen_local161_SKcanonical' else runs[(model,seed)],sum(r['metrics']['passed'] for r in rr),len(rr))
    oracle=read('frozen_oracle_attribution.json');group=defaultdict(list)
    for r in oracle:group[(r['parent_arm'],r['precision'],r['intervention'],r['family'],r['seed'])].append(r)
    for (arm,precision,mode,family,seed),rr in group.items():
        observe(f'oracle-{arm}-{precision}-{mode}-{family}-{seed}',dict(cases=rr,selection_eligible=False,parent_task='39g'),'frozen_oracle_attribution.json',
            runs[('joint_wfrozen39g_'+arm,seed)],sum(r['metrics']['passed'] for r in rr),len(rr),'oracles')
    for label,path,parent_task in [('localized','residual_oracle_attribution.json','39g'),('h8probe','posthoc_h8_oracles.json','39h')]:
        rr=read(path);rr=rr['rows'] if isinstance(rr,dict) else rr
        for i,r in enumerate(rr):
            arm=('joint_wH8' if label=='h8probe' else 'joint_wfrozen39g_'+r['parent'])
            observe(f'{label}-{i}',dict(cases=[r],selection_eligible=False,parent_task=parent_task),path,runs[(arm,r['seed'])],int(r['metrics']['passed']),1,'oracles')
    for curve in read('training_curves.json'):
        for i,a in enumerate(freeze['arms']):
            for j,seed in enumerate(freeze['seeds']):
                k=i*3+j
                observe(f'curve-{a["name"]}-{seed}-{curve["step"]}',dict(step=curve['step'],panel=curve['panel'][k],elapsed_seconds=curve['elapsed_seconds']),
                    'training_curves.json',runs[('joint_w'+a['name'],seed)],curve['common_development_score'][k],evaluation='selection')
    for cost in read('resident_cost_benchmark.json')['rows']:
        observe(f'cost-{cost["arm"]}-{cost["seed"]}',cost,'resident_cost_benchmark.json',runs[('joint_w'+cost['arm'],cost['seed'])],cost['median_saving_fraction'],30,'cost')
    for p in sorted(folder.glob('*.json')):
        if p.name not in {'recurrence_evaluation.json','frozen_oracle_attribution.json'} and not p.name.startswith('memory_'):
            observe('diagnostic-'+p.name,read(p.name),p.name,evaluation='oracles')
    finding=put('findings','outcome',Nome='Task39h65/72; seven residuals and mixed generalization',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,
        **{'Risultato':json.dumps(proposal,ensure_ascii=False),'Limitazioni':audit['limits']})
    for name in ('relaxation_bias_accumulation','warm_finetuning_step_interference','mean_loss_hides_rare_rate_error','rare_recovery_support','event_conditioning_and_other_currents'):
        put('evidence','after39g-'+name,Nome='Task39h scoped evidence '+name,Esito='Indeterminata',
            **{'Affermazione valutata':[ref('claims-giada-task39g-result-residual-'+name+'-v1')],'Risultati a sostegno':[finding],
            'Argomentazione':json.dumps(dict(findings=proposal['findings'],contrasts=read('paired_contrasts.json'))),
            'Limiti e spiegazioni alternative':audit['limits'],'Data valutazione':'2026-10-19'})
    for h in proposal['hypotheses']:
        claim=put('claims','residual-'+h['id'],Nome='After39h '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['statement'],
            **{'Condizioni di falsificazione':h['falsifier'],'Limiti':'Post hoc; priority='+h['priority']+'; proposed test: '+h['test']})
        put('evidence','residual-'+h['id'],Nome='Task39h residual rationale '+h['id'],Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],'Argomentazione':json.dumps(proposal['findings']),
            'Limiti e spiegazioni alternative':h['falsifier'],'Data valutazione':'2026-10-19'})
    put('analyses','audit',Nome='Task39h complete audit and gradient/parent attribution',Origine='Post hoc',Versione='v1',Procedura='scripts/analyze_task39h_result.py',
        Configurazione=json.dumps(proposal),**{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':[finding]})
    put('lessons','temporal',Nome='Joint equilibrium/rate and low-occupancy SK accuracy require separate diagnostics',Lezione=json.dumps(proposal['findings']),Risultati=[finding],Limiti=audit['limits'])
    decision=put('decisions','no-go',Nome='Task39h NO-GO;Task40 remains blocked',Esito='Modificare',Risultati=[finding],Motivazione=proposal['outcome'],**{'Data':'2026-10-19'})
    put('actions','proposed39i',Nome='Proposed39i coupled equilibrium/relaxation and SK precision',Stato='Proposta',Descrizione=json.dumps(proposal['proposed_experiment']),**{'Decisioni di origine':[decision]})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task39h-temporal-execute-v1','Stato':'Conclusa','Descrizione':proposal['outcome']})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(obs))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    checks={}
    for mode,count in [('primary',1872),('common',1872),('oracle',1440),('localized',278),('h8probe',68)]:
        sql="SELECT count(*) cases,sum(json_array_length(json_extract(j.value,'$.metrics.channels'))) channels FROM observations o,json_each(o.Descrizione,'$.cases') j WHERE o.\"Codice stabile\" LIKE 'observations-giada-task39h-result-"+mode+"-%'"
        check=m.query(sql)['rows'][0];assert check['cases']==count and check['channels']==count*11,check;checks[mode]=check
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(obs),readback=checks,verification=v,airtable_accessed=False,drive_backup_updated=False,next_experiment_launched=False))
    print(json.dumps(dict(valid=True,records=len(ids),readback=checks)),flush=True)

if __name__=='__main__':main()
