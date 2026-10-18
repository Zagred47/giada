"""Idempotently retain Task39c results and the proposed residual diagnosis in SQLite."""
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent; folder=repo/'experiments/results/task39c_kaggle_v1'
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    audit=read('result_audit.json'); final=read('final_report.json'); freeze=read('selection_freeze.json')
    diagnosis=read('residual_diagnosis.json'); rows=read('recurrence_evaluation.json')
    assert audit['valid'] and audit['checkpoint_hash_valid'] and len(rows)==1872 and final['valid']
    assert not final['thresholds_changed'] and not freeze['independent_confirmation_accessed'] and not freeze['stress_used_for_selection']
    for data in (audit,diagnosis):
        assert all(hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha for name,sha in data['sources'].items())
    payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    m=ValidatedBatchMirror(); ids={}; prefix='giada-task39c-dynamic'
    def put(table,suffix,**fields):
        code=f'{table}-{prefix}-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id']; ids[code]=rid; return rid
    @lru_cache(None)
    def refcode(code):
        result=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(result)==1,code
        return result[0]['record_id']
    def ref(table,suffix):return refcode(f'{table}-{prefix}-{suffix}-v1')
    arts={}
    paths=[p for p in sorted(folder.iterdir()) if p.is_file() and p.name not in ('memory_receipt.json','memory_readback.json')]
    doc=repo/'experiments/task39c_result_and_residual_design.md'
    paths += [doc,repo/'scripts/diagnose_task39c_residuals.py',repo/'scripts/analyze_task39c_result.py',PathFile()]
    for path in paths:
        arts[path.name]=put('artifacts','result-'+path.name,Nome='Task39c '+path.name,
            Tipo='Checkpoint' if path.suffix=='.pt' else 'Codice' if path.suffix=='.py' else 'Report',Versione='Kaggle v1 / post-freeze audit',
            Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=put('experiments','suite',Nome='Task39c coefficient information, support and temporal identification',Stato='Concluso',
        Descrizione='Kaggle v1 technically complete; best new arm44/72 development; strict NO-GO, Task40 blocked. No independent confirmation.')
    runs={}
    for i,d in enumerate(freeze['arms']):
        for j,seed in enumerate(freeze['seeds']):
            runs[('joint_w'+d['name'],seed)]=put('runs',f'{d["name"]}-seed{seed}',Nome=f'Task39c {d["name"]} seed{seed}',
                Stato='Completata',Seed=str(seed),Braccio=[ref('arms',d['name'])],
                **{'Configurazione effettiva':json.dumps(dict(arm=d,seed=seed,chosen_step=freeze['chosen_steps'][i*3+j],selection_score=freeze['scores'][i*3+j],revision=read('revision.json'))),
                   'Hardware e ambiente':json.dumps(read('execution.json')),'Artefatti prodotti':list(arts.values()),
                   'Validità tecnica':json.dumps(dict(valid=True,freeze_hash_valid=True,checkpoint_hash_valid=True,scientific_decision=final['decision']))})
    baseline=refcode('runs-giada-task38-joint-transition-kaggle-local161-v1')
    obs=[]
    def observation(suffix,name,data,artifact,evaluation='quality',run=None,value=None,count=1):
        fields={'Nome':name,'Numerosità':count,'Descrizione':json.dumps(data,separators=(',',':')),
            'Specifica di valutazione':[ref('evaluations',evaluation)],'Artefatti dettagliati':[arts[artifact]]}
        if run:fields['Run']=[run]
        if value is not None:fields['Valore']=value
        rid=put('observations',suffix,**fields); obs.append(rid); return rid
    grouped=defaultdict(list)
    for row in rows:grouped[(row['model'],row['role'],row['family'],row['seed'])].append(row)
    for (model,role,family,seed),cases in grouped.items():
        observation(f'result-{model}-{role}-{family}-{seed}',f'Task39c {model}/{role}/{family}/seed{seed}',dict(cases=cases),
            'recurrence_evaluation.json',run=baseline if model=='frozen_local161_SKcanonical' else runs[(model,seed)],
            value=sum(r['metrics']['passed'] for r in cases),count=len(cases))
    for curve in read('training_curves.json'):
        for i,d in enumerate(freeze['arms']):
            for j,seed in enumerate(freeze['seeds']):
                observation(f'curve-{d["name"]}-{seed}-{curve["step"]}',f'Task39c {d["name"]} seed{seed} step{curve["step"]}',
                    dict(step=curve['step'],seed=seed,elapsed_seconds=curve['elapsed_seconds']), 'training_curves.json','selection',
                    runs[('joint_w'+d['name'],seed)],curve['common_development_score'][i*3+j])
    for probe in read('internal_probes.json'):
        observation(f'probe-{probe["arm"]}-{probe["seed"]}',f'Task39c coefficient probe {probe["arm"]} seed{probe["seed"]}',
            probe,'internal_probes.json','oracles',runs[('joint_w'+probe['arm'],probe['seed'])])
    oracle=read('frozen_oracle_attribution.json'); groups=defaultdict(list)
    for r in oracle:groups[(r['parent_arm'],r['precision'],r['intervention'],r['family'],r['seed'])].append(r)
    for (arm,precision,mode,family,seed),cases in groups.items():
        observation(f'oracle-{arm}-{precision}-{mode}-{family}-{seed}',f'Task39c frozen39b oracle {arm}/{precision}/{mode}/{family}/{seed}',
            dict(cases=cases,selection_eligible=False),'frozen_oracle_attribution.json','oracles',count=len(cases),value=sum(r['metrics']['passed'] for r in cases))
    for cost in read('resident_cost_benchmark.json')['rows']:
        observation(f'cost-{cost["arm"]}-{cost["seed"]}',f'Task39c resident ionic cost {cost["arm"]}/{cost["seed"]}',cost,
            'resident_cost_benchmark.json','cost',runs[('joint_w'+cost['arm'],cost['seed'])],cost['median_saving_fraction'],len(cost['paired_formula_candidate_ms']))
    # Full structured diagnostics are queryable in SQLite, not merely linked externally.
    excluded={'recurrence_evaluation.json','frozen_oracle_attribution.json','internal_probes.json','memory_receipt.json','memory_readback.json'}
    for path in sorted(folder.glob('*.json')):
        if path.name not in excluded:
            observation('diagnostic-'+path.name,'Task39c complete '+path.name,read(path.name),path.name,'oracles')
    document=doc.read_text(encoding='utf-8')
    finding=put('findings','outcome',Nome='Task39c dynamic information improves learnability; current and timing residuals remain',Esito='Misto',
        Esperimenti=[exp],Osservazioni=obs,**{'Risultato':document,'Limitazioni':final['limits']+' '+audit['limits']})
    hypotheses={
        'current_objective':('Global current scaling underweights weak-regime errors required by local strict evaluation.',
            'Matched local-scale objective fails to improve weak-regime current/charge errors without strong-regime regressions.'),
        'history_exposure':('Variable-input history and boundary/tail coverage remain insufficient despite long constant-input anchors.',
            'Verified targeted fit-only history coverage has no consistent benefit under matched target-tick and budget comparisons.'),
        'conditioning':('Fixed initial auxiliary weights and gradient interaction hinder late optimization.',
            'Fit-only phase calibration changes gradients but fails to improve paired strict outcomes; clipping alone is not causal evidence.'),
        'event_objective':('Pointwise objectives fail to prioritize charge and event-boundary accuracy.',
            'A fit-only charge/event objective package gives no consistent benefit at equal exposure under the unchanged original evaluator.'),
        'tail_bias':('A learnable linear logit component helps approximate tail coefficients at the same abstraction.',
            'Parameter-matched architecture control fails on targeted tails or introduces unacceptable regressions/cost.'),
        'timing_sensitivity':('Some timing failures reflect crossing sensitivity or precision rather than uniform phase shift.',
            'Frozen new-candidate FP64, exact swaps and trace perturbation audit show stable crossings and unchanged errors.')}
    predecessors=[p['_record_id'] for p in diagnosis['precedents']]
    for name,(statement,falsifier) in hypotheses.items():
        claim=put('claims','residual-'+name,Nome='After39c: '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,
            **{'Condizioni di falsificazione':falsifier,'Limiti':'Post hoc hypothesis; prospective39d proposal, not an established explanation. No change to abstraction or thresholds.'})
        put('evidence','residual-'+name,Nome='After39c evidence and limits: '+name,Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[finding]+predecessors,
               'Argomentazione':statement+' See residual_diagnosis.json and detailed result document.',
               'Limiti e spiegazioni alternative':falsifier+' Historical findings are scoped analogies, not proof.', 'Data valutazione':'2026-10-18'})
    for name in ('dynamic_information','optimization_budget','local_support','shared_capacity','loss_conditioning','numerical_timing'):
        put('evidence','prior-hypothesis-'+name,Nome='Task39c update of39b hypothesis '+name,Esito='Indeterminata',
            **{'Affermazione valutata':[refcode('claims-giada-task39b-architecture-residual-'+name+'-v1')],
               'Risultati a sostegno':[finding],'Argomentazione':document,
               'Limiti e spiegazioni alternative':'Broad hypothesis remains conditional; exact16 paired contrasts and scope are retained, not blanket confirmation or refutation.',
               'Data valutazione':'2026-10-18'})
    put('analyses','residual-audit',Nome='Task39c frozen residual audit and SQLite precedent retrieval',Origine='Post hoc',Versione='v1',
        Procedura='scripts/analyze_task39c_result.py; scripts/diagnose_task39c_residuals.py',
        Configurazione=json.dumps(dict(query=diagnosis['precedent_query'],parameters=diagnosis['precedent_codes'],scope=diagnosis['scope'])),
        **{'Run analizzati':list(runs.values())+[baseline],'Input artefatti':list(arts.values()),'12 · Risultati scientifici':[finding]+predecessors})
    put('lessons','objective-and-event-contract',Nome='Task39c gate tolerance does not guarantee current or timing tolerance',Lezione=document,
        Risultati=[finding],Limiti='Observed improvement is domain-limited; proposed explanations require paired interventions.')
    decision=put('decisions','post-result',Nome='Task39c NO-GO; propose39d residual-current diagnosis',Esito='Modificare',Risultati=[finding],
        Motivazione=document,**{'Data':'2026-10-18'})
    put('actions','execute',Nome='Execute Task39c on Kaggle',Stato='Conclusa',Descrizione='Completed and ingested; strict scientific NO-GO, Task40 blocked.')
    put('actions','proposed39d',Nome='Task39d proposed current and timing matrix',Stato='Proposta',Descrizione=document,
        **{'Decisioni di origine':[decision],'Criteri di avvio':'Concrete loss/support/preflight/budget preregistration and implementation still required; no new notebook launched by this registration.'})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(obs))),flush=True)
    m.commit_batch(); write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot()); verification=m.verify(); assert verification['valid']
    # Read back the nested case/channel records through the actual relational views.
    sql='''SELECT count(*) cases, sum(json_array_length(json_extract(j.value,'$.metrics.channels'))) channels,
        sum(json_extract(j.value,'$.metrics.passed')) passed
        FROM observations o,json_each(o.Descrizione,'$.cases') j
        WHERE o."Codice stabile" LIKE 'observations-giada-task39c-dynamic-result-%' '''
    check=m.query(sql)['rows'][0]; assert check['cases']==1872 and check['channels']==20592,check
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(obs),recurrence_cases=1872,channel_metrics=20592,
        oracle_cases=len(oracle),curves=len(read('training_curves.json'))*36,member_probes=36,cost_members=36,
        verification=verification,readback=check,airtable_accessed=False,task39d_launched=False,drive_backup_updated=False))
    print(json.dumps(dict(valid=True,records=len(ids),observations=len(obs),readback=check,verification=verification)),flush=True)


def PathFile():
    from pathlib import Path
    return Path(__file__).resolve()


if __name__=='__main__':main()
