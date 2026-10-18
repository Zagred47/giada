"""Idempotent complete Task39b result ingestion and scoped causal follow-up."""
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent; folder=repo/'experiments/results/task39b_kaggle_v2'
    read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    audit=read('result_audit.json'); final=read('final_report.json'); freeze=read('selection_freeze.json')
    assert audit['valid'] and audit['checkpoint_hash_valid'] and final['valid'] and audit['rows']==2016
    assert not final['thresholds_changed'] and not final['independent_confirmation_accessed']
    assert all(hashlib.sha256((folder/n).read_bytes()).hexdigest()==h for n,h in audit['sources'].items())
    m=ValidatedBatchMirror(); ids={}
    prefix='giada-task39b-architecture'
    def put(table,suffix,**fields):
        code=f'{table}-{prefix}-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(maxsize=None)
    def code_ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    def ref(table,suffix):return code_ref(f'{table}-{prefix}-{suffix}-v1')
    arts={}
    # Mutable receipts and DB query readback are not scientific inputs and avoid circular hashes.
    for path in sorted(folder.iterdir()):
        if path.name in ('memory_receipt.json','memory_exploration.json'):continue
        arts[path.name]=put('artifacts','result-'+path.name,Nome='Task39b '+path.name,Tipo='Checkpoint' if path.suffix=='.pt' else 'Report',
            Versione='Kaggle v2',Percorso=path.relative_to(repo).as_posix(),
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    docpath=repo/'experiments/task39b_result_and_causal_next_design.md'
    document=docpath.read_text(encoding='utf-8')
    arts['analysis_document']=put('artifacts','result-analysis-document',Nome='Task39b result and proposed39c',Tipo='Report',Versione='v1',
        Percorso=docpath.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(docpath.read_bytes()).hexdigest(),'Dimensione byte':docpath.stat().st_size})
    exp=put('experiments','suite',Nome='Task39b training and architecture diagnosis',Stato='Concluso',
        Descrizione='Kaggle v2 technically complete; no new arm passes strict quality; Task40 blocked. Full recurrence, budgets, probes and contrasts retained.')
    runs={}
    for i,d in enumerate(freeze['arms']):
        runs['joint_w'+d['name']]=put('runs',d['name'],Nome='Task39b completed '+d['name'],Stato='Completata',Braccio=[ref('arms',d['name'])],
            **{'Configurazione effettiva':json.dumps(dict(arm=d,seeds=freeze['seeds'],chosen_steps=freeze['chosen_steps'][3*i:3*i+3],selection_scores=freeze['scores'][3*i:3*i+3],revision=read('revision.json'))),
               'Hardware e ambiente':json.dumps(read('execution.json')),'Artefatti prodotti':list(arts.values()),
               'Validità tecnica':json.dumps({k:v for k,v in audit.items() if k not in ('models','budgets','sources')})})
    runs['joint_whistorical39_best']=code_ref('runs-giada-task39-learnability-factorized_expanded_fr-v1')
    runs['frozen_local161_SKcanonical']=code_ref('runs-giada-task38-joint-transition-kaggle-local161-v1')
    observations=[];groups=defaultdict(list)
    for row in read('recurrence_evaluation.json'):
        groups[(row['model'],row['role'],row['family'],row['seed'])].append(row)
    for (model,role,family,seed),rows in groups.items():
        observations.append(put('observations',f'result-{model}-{role}-{family}-{seed}',
            Nome=f'Task39b {model}/{role}/{family}/seed{seed}',Valore=sum(r['metrics']['passed'] for r in rows),Numerosità=len(rows),Run=[runs[model]],
            **{'Specifica di valutazione':[ref('evaluations','strict_quality')],'Artefatti dettagliati':[arts['recurrence_evaluation.json']],
               'Strato o sottogruppo':f'{role}/{family}/seed{seed}','Descrizione':json.dumps(dict(cases=rows),separators=(',',':'))}))
        if len(observations)%200==0:print(json.dumps(dict(stage='staging_cases',observations=len(observations))),flush=True)
    for row in read('training_curves.json'):
        for i,d in enumerate(freeze['arms']):
            for j,seed in enumerate(freeze['seeds']):
                observations.append(put('observations',f'curve-{d["name"]}-{seed}-{row["step"]}',Nome=f'Task39b {d["name"]} seed{seed} step{row["step"]}',
                    Valore=row['common_development_score'][i*3+j],Numerosità=1,Run=[runs['joint_w'+d['name']]],
                    **{'Specifica di valutazione':[ref('evaluations','selection')],'Artefatti dettagliati':[arts['training_curves.json']],
                       'Descrizione':json.dumps(dict(step=row['step'],seed=seed,elapsed_seconds=row['elapsed_seconds']))}))
    probes=read('internal_probes.json');coeff=read('coefficient_probe.json')['records']
    for i,d in enumerate(freeze['arms']):
        for j,seed in enumerate(freeze['seeds']):
            member=i*3+j
            data={k:(v[member] if isinstance(v,list) and len(v)==36 else v) for k,v in probes.items()}
            data['held_voltage_coefficient_probe']=[r for r in coeff if r['arm']==d['name'] and r['seed']==seed]
            observations.append(put('observations',f'probe-{d["name"]}-{seed}',Nome=f'Task39b internal probes {d["name"]} seed{seed}',Numerosità=1,
                Run=[runs['joint_w'+d['name']]],**{'Specifica di valutazione':[ref('evaluations','probes')],
                'Artefatti dettagliati':[arts['internal_probes.json'],arts['coefficient_probe.json']],'Descrizione':json.dumps(data)}))
    # Preserve remaining structured diagnostics inside SQLite as well as artifact references.
    for name in ('matched_history_probes.json','paired_contrasts.json','window_support.json','exposure.json','result_audit.json','selection_freeze.json'):
        observations.append(put('observations','diagnostic-'+name,Nome='Task39b complete '+name,Numerosità=1,
            **{'Specifica di valutazione':[ref('evaluations','probes')],'Artefatti dettagliati':[arts[name]],'Descrizione':json.dumps(read(name))}))
    finding=put('findings','outcome',Nome='Task39b scoped interventions insufficient; dynamic identification remains open',Esito='Misto',
        Esperimenti=[exp],Osservazioni=observations,**{'Risultato':document,'Limitazioni':final['limits']})
    for name in ('horizon','collocation','relative','grouped','multiscale','interaction'):
        cs=[c for c in final['contrasts']['contrasts'] if c['axis']==name or (name in ('grouped','multiscale') and c['treatment'].startswith(name))]
        if name=='interaction':cs=final['contrasts']['contrasts']
        # Failure of this acceptance conjunction does not falsify every broader mechanism.
        claim=put('claims',name,Nome='Task39b '+name,Stato='Indeterminata',
            **{'Limiti':'No registered joint contrast passed. Mixed partial gains; three seeds, fixed budget, only tested parametrizations. Wider causal hypothesis remains open.'})
        put('evidence','result-'+name,Nome='Task39b scoped contrast '+name,Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],'Argomentazione':json.dumps(cs),
               'Limiti e spiegazioni alternative':'Do not equate failure to reach joint thresholds with impossibility. Architecture comparisons differ in effective connectivity.','Data valutazione':'2026-10-18'})
    hypotheses={
        'dynamic_information':('Short state targets underweight equilibrium/relaxation errors, especially small fractions.','Paired coefficient supervision and held-multihorizon interventions both fail despite sufficient support and verified gradients.'),
        'optimization_budget':('Residual error can be reduced by further optimization; all36 selection scores still improve at60k.','Prospective matched budget ladder plateaus without relevant improvements on long regimes.'),
        'local_support':('Rare or local V/Ca/state support is insufficient at fixed sampled windows.','Matched targeted-support intervention fails within the same capacity/budget/objective.'),
        'shared_capacity':('Shared function capacity or interference limits some gates after information repair.','Neither matched separate heads nor wider shared model improve the affected regimes at matched exposure.'),
        'loss_conditioning':('Auxiliary gradient scales trade accuracy among gates/channels instead of improving the joint criterion.','Gradient-scale diagnostics and paired calibration show no material contribution.'),
        'numerical_timing':('A minority of failures arise from precision or timing/selection sensitivity rather than broad dynamic error.','Same frozen model FP64 and trace-level event audit leave errors unchanged; exact controls pass.')}
    newclaims={}
    for name,(statement,falsifier) in hypotheses.items():
        newclaims[name]=put('claims','residual-'+name,Nome='After39b: '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,
            **{'Condizioni di falsificazione':falsifier,'Limiti':'Post-result hypothesis for proposed39c, not an established causal finding. Same abstraction level retained.'})
    memory=read('memory_exploration.json'); predecessors=[x['_record_id'] for x in memory['precedents']]
    analysis=put('analyses','memory-and-failure-exploration',Nome='Task39b SQL precedent and frozen failure exploration',Origine='Post hoc',Versione='v1',
        Procedura='scripts/explore_task39b_memory.py; scripts/analyze_task39b_result.py; scripts/probe_task39b_frozen_coefficients.py',
        Configurazione=json.dumps(dict(queries=memory['queries'],precedent_codes=memory['precedent_codes'],scope='GIADA only; no Airtable')),
        **{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':[finding]+predecessors})
    put('lessons','conditional-repair-pattern',Nome='Task39b: diagnose information and conditioning before rejecting abstraction',Lezione=document,
        Risultati=[finding]+predecessors,Limiti='Precedents are domain-specific analogies; neither unlimited scaling nor a guaranteed fix is established.')
    decision=put('decisions','post-result',Nome='Task39b NO-GO; propose39c causal identification matrix',Esito='Modificare',Risultati=[finding],
        Motivazione=document,**{'Data':'2026-10-18'})
    put('actions','proposed39c',Nome='Task39c — proposed causal identification and training matrix',Stato='Proposta',Descrizione=document,
        **{'Decisioni di origine':[decision],'Criteri di avvio':'Freeze exact losses, sampling, budgets and common selection after technical benchmark; user has not requested launch in this turn.'})
    print(json.dumps(dict(stage='validated_batch_ready',records=len(ids),observations=len(observations))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(observations),recurrence_cases=2016,channel_metrics=22176,
        curves=180,member_probes=36,precedents=15,precedent_evidence_chains=23,open_hypotheses=6,verification=verification,
        airtable_accessed=False,task39c_launched=False,drive_backup_updated=False))
    print(json.dumps(dict(valid=True,records=len(ids),observations=len(observations),verification=verification)),flush=True)


if __name__=='__main__':main()
