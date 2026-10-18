"""Retain all Task39d cases, diagnostics and scoped hypotheses in SQLite only."""
import hashlib
import json
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39d_colab_v1'
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    audit=read('result_audit.json');final=read('final_report.json');freeze=read('selection_freeze.json')
    rows=read('recurrence_evaluation.json');diagnosis=read('residual_diagnosis.json')
    assert audit['valid'] and audit['checkpoint_hash_valid'] and len(rows)==1872 and final['valid']
    assert not final['thresholds_changed'] and not final['independent_confirmation_accessed']
    for data in (audit,diagnosis):
        assert all(hashlib.sha256((folder/n).read_bytes()).hexdigest()==s for n,s in data['sources'].items())
    archive=repo/'artifacts/giada_task39d_current_timing_colab_v1.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():
            if not n.endswith('/'):assert hashlib.sha256(z.read(n)).digest()==hashlib.sha256((folder/n).read_bytes()).digest()
    m=ValidatedBatchMirror();ids={};prefix='giada-task39d-current'
    def put(table,suffix,**fields):
        code=f'{table}-{prefix}-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def code_ref(code):
        rr=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rr)==1,code
        return rr[0]['record_id']
    def ref(table,suffix):return code_ref(f'{table}-{prefix}-{suffix}-v1')
    doc=repo/'experiments/task39d_result_and_next_hypotheses.md';document=doc.read_text(encoding='utf-8')
    paths=[p for p in sorted(folder.iterdir()) if p.is_file() and not p.name.startswith('memory_')]
    paths += [archive,doc,repo/'scripts/analyze_task39d_result.py',repo/'scripts/diagnose_task39d_result.py',Path(__file__).resolve()]
    arts={}
    for path in paths:
        arts[path.name]=put('artifacts','result-'+path.name,Nome='Task39d '+path.name,
            Tipo='Checkpoint' if path.suffix=='.pt' else 'Codice' if path.suffix=='.py' else 'Report',Versione='Colab v1 / frozen audit',
            Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=put('experiments','suite',Nome='Task39d current scaling, history, conditioning and timing',Stato='Concluso',
        Descrizione='Colab technically complete; max43/72 vs39c44/72. Local scaling improves extremes, strict NO-GO. Kagglev3 canceled queued, no training.')
    runs={}
    for i,d in enumerate(freeze['arms']):
        for j,seed in enumerate(freeze['seeds']):
            runs[('joint_w'+d['name'],seed)]=put('runs',f'{d["name"]}-seed{seed}',Nome=f'Task39d {d["name"]} seed{seed}',
                Stato='Completata',Seed=str(seed),Braccio=[ref('arms',d['name'])],
                **{'Configurazione effettiva':json.dumps(dict(arm=d,seed=seed,chosen_step=freeze['chosen_steps'][i*3+j],selection_score=freeze['scores'][i*3+j],revision=read('revision.json'),platform='Colab',kaggle_session_canceled=359739424)),
                   'Hardware e ambiente':json.dumps(read('execution.json')),'Artefatti prodotti':list(arts.values()),
                   'Validità tecnica':json.dumps(dict(valid=True,freeze_hash_valid=True,checkpoint_hash_valid=True,scientific_decision=final['decision']))})
    baseline=code_ref('runs-giada-task38-joint-transition-kaggle-local161-v1');obs=[]
    def observation(suffix,name,data,artifact,evaluation='quality',run=None,value=None,count=1):
        fields={'Nome':name,'Numerosità':count,'Descrizione':json.dumps(data,separators=(',',':')),
            'Specifica di valutazione':[ref('evaluations',evaluation)],'Artefatti dettagliati':[arts[artifact]]}
        if run:fields['Run']=[run]
        if value is not None:fields['Valore']=value
        rid=put('observations',suffix,**fields);obs.append(rid);return rid
    grouped=defaultdict(list)
    for r in rows:grouped[(r['model'],r['role'],r['family'],r['seed'])].append(r)
    for (model,role,family,seed),cases in grouped.items():
        observation(f'result-{model}-{role}-{family}-{seed}',f'Task39d {model}/{role}/{family}/{seed}',dict(cases=cases),
            'recurrence_evaluation.json',run=baseline if model=='frozen_local161_SKcanonical' else runs[(model,seed)],
            value=sum(r['metrics']['passed'] for r in cases),count=len(cases))
    for curve in read('training_curves.json'):
        for i,d in enumerate(freeze['arms']):
            for j,seed in enumerate(freeze['seeds']):
                observation(f'curve-{d["name"]}-{seed}-{curve["step"]}',f'Task39d {d["name"]} seed{seed} step{curve["step"]}',
                    dict(step=curve['step'],seed=seed,elapsed_seconds=curve['elapsed_seconds']), 'training_curves.json','selection',
                    runs[('joint_w'+d['name'],seed)],curve['common_development_score'][i*3+j])
    og=defaultdict(list);oracle=read('frozen_oracle_attribution.json')
    for r in oracle:og[(r['parent_arm'],r['precision'],r['intervention'],r['family'],r['seed'])].append(r)
    for (arm,precision,mode,family,seed),cases in og.items():
        observation(f'oracle-{arm}-{precision}-{mode}-{family}-{seed}',f'Task39d frozen39c oracle {arm}/{precision}/{mode}/{family}/{seed}',
            dict(cases=cases,selection_eligible=False),'frozen_oracle_attribution.json','oracles',count=len(cases),value=sum(r['metrics']['passed'] for r in cases))
    for cost in read('resident_cost_benchmark.json')['rows']:
        observation(f'cost-{cost["arm"]}-{cost["seed"]}',f'Task39d resident cost {cost["arm"]}/{cost["seed"]}',cost,
            'resident_cost_benchmark.json','cost',runs[('joint_w'+cost['arm'],cost['seed'])],cost['median_saving_fraction'],30)
    excluded={'recurrence_evaluation.json','frozen_oracle_attribution.json'}
    for path in sorted(folder.glob('*.json')):
        if path.name not in excluded and not path.name.startswith('memory_'):
            observation('diagnostic-'+path.name,'Task39d complete '+path.name,read(path.name),path.name,'oracles')
    finding=put('findings','outcome',Nome='Task39d improves extreme current errors but does not improve full development pass count',Esito='Misto',
        Esperimenti=[exp],Osservazioni=obs,**{'Risultato':document,'Limitazioni':audit['limits']+' '+final['limits']})
    for name in ('current_objective','history_exposure','conditioning','event_objective','tail_bias','timing_sensitivity'):
        put('evidence','after39c-'+name,Nome='Task39d scoped update: '+name,Esito='Indeterminata',
            **{'Affermazione valutata':[code_ref('claims-giada-task39c-dynamic-residual-'+name+'-v1')],
               'Risultati a sostegno':[finding],'Argomentazione':document,
               'Limiti e spiegazioni alternative':'No blanket rejection of broad hypothesis. Only registered package/contrast tested. See paired_contrasts and frozen oracles.',
               'Data valutazione':'2026-10-18'})
    hypotheses={
        'tail_precision_objective':('Absolute u loss and global selection underweight relative equilibrium errors amplified by current powers and weak-regime scales.',
            'Matched hybrid tail-aware u/current objective plus prospectively aligned selection fails to improve tail coefficients and strict current/charge outcomes.'),
        'stratified_coefficient_support':('Empirical FIT anchor coverage undersamples important V/logCa boundary regions even at high repeated-update exposure.',
            'At equal label/update counts, verified stratified FIT support provides no repeatable gain.'),
        'head_interference':('Shared u/z representation and gradients limit simultaneous equilibrium and rate accuracy.',
            'Parameter-matched separate u/z heads do not improve scoped errors and strict quality at matched compute/support.'),
        'long_variable_exposure':('The tested64-tick variable-input package is too short to train slow history-dependent errors.',
            'Longer variable rollouts at equal target-tick exposure do not improve long-history/recovery or timing metrics.'),
        'remaining_budget':('Optimization remains incomplete for some arms at120k, but scaling alone may not resolve objective mismatch.',
            'Preregistered larger checkpoints fail to improve paired strict metrics despite lower training/common scores.')}
    for name,(statement,falsifier) in hypotheses.items():
        claim=put('claims','residual-'+name,Nome='After39d: '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,
            **{'Condizioni di falsificazione':falsifier,'Limiti':'Post hoc proposal, no new experiment or independent confirmation yet. Same abstraction and thresholds.'})
        put('evidence','residual-'+name,Nome='After39d evidence/limits: '+name,Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],
               'Argomentazione':document,'Limiti e spiegazioni alternative':falsifier,'Data valutazione':'2026-10-18'})
    predecessors=[p['_record_id'] for p in diagnosis['precedents']]
    put('analyses','residual-audit',Nome='Task39d paired failure audit and SQLite precedent query',Origine='Post hoc',Versione='v1',
        Procedura='scripts/analyze_task39d_result.py; scripts/diagnose_task39d_result.py',
        Configurazione=json.dumps(dict(query=diagnosis['precedent_query'],parameters=diagnosis['precedent_codes'])),
        **{'Run analizzati':list(runs.values())+[baseline],'Input artefatti':list(arts.values()),'12 · Risultati scientifici':[finding]+predecessors})
    put('lessons','tails-not-pass-count',Nome='Task39d tail improvement and complete behavior are distinct outcomes',Lezione=document,
        Risultati=[finding],Limiti=audit['limits'])
    decision=put('decisions','post-result',Nome='Task39d NO-GO; preserve39c and propose39e',Esito='Modificare',Risultati=[finding],
        Motivazione=document,**{'Data':'2026-10-18'})
    put('actions','execute',Nome='Execute Task39d (migrated Kaggle queue to Colab)',Stato='Conclusa',
        Descrizione='Completed on Colab; Kagglev3 canceled queued.36 runs updated; strict scientific NO-GO, not technical failure.')
    put('actions','proposed39e',Nome='Proposed Task39e tail precision, support and interference matrix',Stato='Proposta',Descrizione=document,
        **{'Decisioni di origine':[decision],'Criteri di avvio':'User approval and concrete preregistration/implementation required; no launch authorized by ingestion.'})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(obs))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    sql='''SELECT count(*) cases,sum(json_array_length(json_extract(j.value,'$.metrics.channels'))) channels,
        sum(json_extract(j.value,'$.metrics.passed')) passed
        FROM observations o,json_each(o.Descrizione,'$.cases') j
        WHERE o."Codice stabile" LIKE 'observations-giada-task39d-current-result-%' '''
    check=m.query(sql)['rows'][0];assert check['cases']==1872 and check['channels']==20592,check
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(obs),recurrence_cases=1872,channel_metrics=20592,
        oracle_cases=len(oracle),verification=verification,readback=check,airtable_accessed=False,next_experiment_launched=False,drive_backup_updated=False))
    print(json.dumps(dict(valid=True,records=len(ids),observations=len(obs),readback=check,verification=verification)),flush=True)
if __name__=='__main__':main()
