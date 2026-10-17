"""Task38 actual results + prospective Task39, one validated local batch."""
import hashlib
import json
from collections import defaultdict
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent; folder=repo/'experiments/results/task38_kaggle_v2'
    read=lambda n:json.loads((folder/n).read_text())
    audit=read('result_audit.json'); final=read('final_report.json'); assert audit['valid'] and final['valid']
    spec_path=repo/'experiments/task39_learnability_preregistration_v1.json'; spec=json.loads(spec_path.read_text())
    m=ValidatedBatchMirror(); ids={}
    def put(table,suffix,task=39,**fields):
        prefix='joint-transition' if task==38 else 'learnability'
        code=f'{table}-giada-task{task}-{prefix}-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def artifact(path,task):
        rel=path.relative_to(repo).as_posix()
        return put('artifacts',rel,task=task,Nome=f'Task{task} '+rel,Percorso=rel,
            Tipo='Checkpoint' if path.suffix=='.pt' else ('Codice' if path.suffix=='.py' else 'Report'),Versione='v2' if task==38 else 'v1',
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    arts38={path.name:artifact(path,38) for path in sorted(folder.iterdir()) if path.name!='memory_receipt.json'}
    artifact(repo/'experiments/task38_result_v1.md',38)
    exp38=put('experiments','suite',task=38,Nome='Task38 joint explicit-state ionic transition',Stato='Concluso',
        Descrizione='Completed30k updates, all432 recurrence evaluations. Scientific NO-GO; technical completion valid. No independent confirmation.')
    protocol38=ref('protocols-giada-task38-joint-transition-suite-v1')
    partitions=[ref(f'partitions-giada-task37-ionic-playground-{r}-v1') for r in ('fit','fit_extra','development','stress_diagnostic')]
    runs38={}
    for model,suffix in [('joint_w32','w32'),('joint_w64','w64'),('frozen_local161_SKcanonical','local161')]:
        code='kaggle-'+suffix if suffix!='local161' else 'kaggle-local161'
        runs38[model]=put('runs',code,task=38,Nome='Task38 completed '+model,Stato='Completata',
            Braccio=[ref(f'arms-giada-task38-joint-transition-{suffix}-v1')],
            **{'Artefatti prodotti':list(arts38.values()),'Configurazione effettiva':json.dumps(final['summaries'][model]),
               'Validità tecnica':json.dumps(audit),'Hardware e ambiente':json.dumps(read('execution.json'))})
    metric=put('metrics','family_worst_current_nrmse',task=38,Nome='Task38 family-seed worst channel current NRMSE',
        Formula='max channel NRMSE across the two trajectories of one family, role and seed',Direzione='Minimizzare',**{'Unità':'dimensionless'})
    ev=put('evaluations','family_worst_current_nrmse',task=38,Nome='Task38 post-run diagnostic family strata',Versione='v1',
        Metrica=[metric],Protocollo=[protocol38],Partizioni=partitions[2:],Ruolo='Diagnostica',
        Target='Per-case values and gate statistics retained in description JSON, full per-channel metrics in hashed recurrence_evaluation artifact.',
        **{'Aggregazione e pesi':'Worst over channels/cases within each family/seed/role; not a new primary endpoint.'})
    grouped=defaultdict(list)
    for r in read('recurrence_evaluation.json'):grouped[(r['model'],r['role'],r['family'],r['seed'])].append(r)
    obs=[]
    for (model,role,family,seed),rows in grouped.items():
        detail=[]
        for row in rows:
            worst=max(row['metrics']['channels'],key=lambda c:c['nrmse'])
            detail.append(dict(case=row['case'],passed=row['metrics']['passed'],gate=row['metrics']['gate'],worst_current=worst))
        obs.append(put('observations',f'{model}-{role}-{family}-{seed}',task=38,Nome=f'Task38 {model}/{role}/{family}/seed{seed}',
            Valore=max(x['worst_current']['nrmse'] for x in detail),Numerosità=len(rows),Run=[runs38[model]],
            **{'Specifica di valutazione':[ev],'Artefatti dettagliati':[arts38['recurrence_evaluation.json']],
               'Strato o sottogruppo':f'{role}/{family}/seed{seed}','Descrizione':json.dumps(dict(cases=detail))}))
    finding=put('findings','outcome',task=38,Nome='Task38 learns one-step but joint recurrent quality insufficient',Esito='Misto',
        Esperimenti=[exp38],Osservazioni=obs,
        **{'Risultato':(repo/'experiments/task38_result_v1.md').read_text(encoding='utf-8'),'Limitazioni':audit['limitations']})
    statuses={'joint_learnability':('Indeterminata','Indeterminata'),'capacity':('Indeterminata','Indeterminata'),
        'budget':('Indeterminata','Indeterminata'),'recurrence_gap':('Supportata nel dominio','Sostiene'),
        'causal_memory':('Indeterminata','Indeterminata')}
    for name,(status,outcome) in statuses.items():
        claim=put('claims',name,task=38,Nome='Task38 '+name,Stato=status)
        put('evidence',name,task=38,Nome='Task38 scoped '+name,Esito=outcome,
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],
               'Argomentazione':'Loss declines through30k but strict recurrent pass fails. Memory contrast is present, not accurate enough. Width comparison has mixed compile/eager execution. Does not prove impossibility or that extra compute suffices.',
               'Data valutazione':'2026-10-18','Limiti e spiegazioni alternative':audit['limitations']})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-ionic-abstraction-20261017-task38-v1',
        'Nome':'Task38 — Aggiornamento appreso a stato esplicito','Stato':'Conclusa',
        'Descrizione':'Technically completed, scientific NO-GO; proceed only to diagnostic Task39, not Task40.'})
    code_paths=['experiments/task39_learnability_preregistration_v1.json','experiments/task39_design_v1.md',
        'src/giada_teacher/task39_learnability.py','scripts/run_task39_learnability.py','scripts/build_task39_notebook.py',
        'tests/test_task39_learnability.py','notebooks/giada_task39_learnability.ipynb','research_memory/register_task39_protocol.py']
    arts39={path:artifact(repo/path,39) for path in code_paths}
    claims=[put('claims',key,Nome='Task39 '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=value,
        **{'Limiti':spec['limits']}) for key,value in spec['hypotheses'].items()]
    exp39=put('experiments','suite',Nome='Task39 factorial learnability diagnosis',Stato='Preregistrato',Ipotesi=claims,
        Descrizione=spec['scope'],**{'Obiettivo informativo':json.dumps(spec['hypotheses'])})
    protocol39=put('protocols','suite',Nome='Task39 prospective factorial',Versione='v1',Modalità='Ricomposizione',
        Esperimento=[exp39],Procedura=json.dumps(spec,ensure_ascii=False),
        **{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[arts39[spec_path.relative_to(repo).as_posix()]],
           'Hash preregistrazione':hashlib.sha256(spec_path.read_bytes()).hexdigest(),'Criteri di successo':spec['quality'],
           'Regole di arresto':spec['preflights']+' '+spec['limits'],'Registrato il':'2026-10-18'})
    impl=put('implementations','worker',Nome='Task39 paired30-member worker',
        **{'Repository URL':'https://github.com/Zagred47/giada','Commit':'Scientific source hashes registered; actual full launch revision stored in worker revision.json and Kaggle source pin.',
           'Percorso entrypoint':'scripts/run_task39_learnability.py','Ambiente':spec['execution'],'Artefatti codice':list(arts39.values())})
    from src.giada_teacher.task39_learnability import arms
    definitions=arms(); arm_ids={}
    for definition in definitions:
        name=definition['name']
        model=put('models',name,Nome='Task39 '+name,Ruolo='Surrogate',Descrizione=json.dumps(definition)+' '+spec['architecture'])
        arm_ids[name]=put('arms',name,Nome='Task39 '+name,Ruolo='Trattamento',Protocollo=[protocol39],Modello=[model],
            Implementazione=[impl],Partizioni=partitions,**{'Configurazione residua':json.dumps(dict(arm=definition,seeds=spec['seeds'],checkpoints=spec['checkpoints']))})
        put('runs',name,Nome='Task39 planned '+name,Stato='Pianificata',Braccio=[arm_ids[name]],
            **{'Configurazione effettiva':'Prospective: three independent paired seeds in one shared Kaggle job. No execution or result claimed before launch.',
               'Hardware e ambiente':'CUDA T4 requested; actual hardware will be stored by worker.'})
    pairs=[]
    for axis,low,high in [('coverage','fit','expanded'),('mode','tf','fr'),('bias','joint','factorized')]:
        for a in definitions[:8]:
            if a[axis]!=low:continue
            b=next(b for b in definitions[:8] if b[axis]==high and all(a[k]==b[k] for k in ('coverage','mode','bias') if k!=axis))
            pairs.append((axis,a['name'],b['name']))
    pairs += [('current_supervision','joint_fit_tf_gate_only','joint_fit_tf'),('small_currents','joint_fit_fr','joint_fit_fr_relative')]
    for axis,a,b in pairs:
        put('contrasts',a+'-vs-'+b,Nome='Task39 '+axis+': '+a+' vs '+b,Protocollo=[protocol39],Bracci=[arm_ids[a],arm_ids[b]],
            **{'Contrasto e coefficienti':'Three paired seed severity ratios treatment/control; '+a+' -> '+b,
               'Soglia interpretativa':spec['planned_contrasts'],'Confondenti controllati':'Same initialization, windows, targets, optimizer and execution backend; only the declared factor differs.',
               'Correzione confronti multipli':'Exploratory preregistered diagnostics; no population-level significance or p-value claim.'})
    for name,formula in [('strict_pass','Conjunction of unchanged Task37 evaluator gates'),('recurrence_severity',spec['severity']),('budget_score','max(common development TF and FR loss),60k/30k')]:
        metric=put('metrics',name,Nome='Task39 '+name,Formula=formula)
        put('evaluations',name,Nome='Task39 '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol39],
            Partizioni=partitions[2:],Ruolo='Diagnostica',Target=spec['quality'],**{'Aggregazione e pesi':spec['severity'],'Gestione classi assenti':'Unsupported events not certified; weak-current strata never dropped.'})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-ionic-abstraction-20261017-task39-v1',
        'Nome':'Task39 — Learnability, bias e mini scaling','Stato':'Pronta','Descrizione':'Prospective10-arm paired diagnosis registered. Launch after local and CUDA preflights; no Task40 promotion without quality and audit.'})
    put('decisions','task38-to39',Nome='Task38 NO-GO authorizes diagnosis, not causal reinsertion',Esito='Modificare',
        Risultati=[finding],Motivazione='Proceed to preregistered Task39; Task40 stays quality-gated.',**{'Data':'2026-10-18'})
    print(json.dumps(dict(stage='validated_batch_ready',records=len(ids),observations=len(obs))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot()); verification=m.verify();assert verification['valid']
    out=repo/'experiments/results/task39_protocol_v1';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'memory_receipt.json',dict(valid=True,records=ids,task38_observations=len(obs),
        task39_result_observations=0,verification=verification,airtable_accessed=False))
    print(json.dumps(dict(valid=True,records=len(ids),observations=len(obs),airtable_accessed=False)),flush=True)


if __name__=='__main__':main()
