"""Idempotent Task39 result ingestion, complete queryable metrics, local only."""
import hashlib
import json
from collections import defaultdict
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent; folder = repo/'experiments/results/task39_kaggle_v1'
    read = lambda name: json.loads((folder/name).read_text(encoding='utf-8'))
    audit = read('result_audit.json'); final = read('final_report.json')
    assert audit['valid'] and final['valid'] and audit['rows'] == 1872
    m = ValidatedBatchMirror(); ids = {}
    def ref(table, suffix):
        code = f'{table}-giada-task39-learnability-{suffix}-v1'
        rows = m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows) == 1, code
        return rows[0]['record_id']
    def put(table, suffix, **fields):
        code = f'{table}-giada-task39-learnability-{suffix}-v1'
        rid = m.local_upsert(table, {'Codice stabile':code, **fields})['record_id']; ids[code] = rid
        return rid
    arts = {}
    for path in sorted(folder.iterdir()):
        if path.name == 'memory_receipt.json': continue
        rel = path.relative_to(repo).as_posix()
        arts[path.name] = put('artifacts', rel, Nome='Task39 '+path.name, Percorso=rel, Tipo='Report', Versione='v1',
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte':path.stat().st_size})
    exp = put('experiments','suite',Nome='Task39 factorial learnability diagnosis',Stato='Concluso',
        Descrizione='Technical success, scientific NO-GO. 1872 complete recurrence cases; Task40 blocked; no independent confirmation.')
    freeze = read('selection_freeze.json'); runs = {}
    for arm in freeze['arms']:
        name = arm['name']
        runs['joint_w'+name] = put('runs',name,Nome='Task39 completed '+name,Stato='Completata',Braccio=[ref('arms',name)],
            **{'Configurazione effettiva':json.dumps(arm),'Hardware e ambiente':json.dumps(read('execution.json')),
               'Validità tecnica':json.dumps({k:v for k,v in audit.items() if k not in ('models','sources')}),
               'Artefatti prodotti':list(arts.values())})
    # Historical controls retain their original Task38 run identity.
    for model,suffix in [('joint_whistorical38_w32','w32'),('joint_whistorical38_w64','w64'),('frozen_local161_SKcanonical','local161')]:
        code=f'runs-giada-task38-joint-transition-kaggle-{suffix}-v1'
        runs[model]=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'][0]['record_id']
    ev = ref('evaluations','strict_pass'); observations = []; groups = defaultdict(list)
    for row in read('recurrence_evaluation.json'):
        groups[(row['model'],row['role'],row['family'],row['seed'])].append(row)
    for (model,role,family,seed), rows in groups.items():
        observations.append(put('observations',f'result-{model}-{role}-{family}-{seed}',
            Nome=f'Task39 {model}/{role}/{family}/seed{seed}',Valore=sum(r['metrics']['passed'] for r in rows),
            Numerosità=len(rows),Run=[runs[model]],**{'Specifica di valutazione':[ev],
            'Artefatti dettagliati':[arts['recurrence_evaluation.json']], 'Strato o sottogruppo':f'{role}/{family}/seed{seed}',
            'Descrizione':json.dumps(dict(cases=rows),separators=(',',':'))}))
    curve_ev=ref('evaluations','budget_score'); curves=read('training_curves.json')
    for row in curves:
        for i,arm in enumerate(freeze['arms']):
            for j,seed in enumerate((17,29,43)):
                observations.append(put('observations',f'curve-{arm["name"]}-{seed}-{row["step"]}',
                    Nome=f'Task39 {arm["name"]}/seed{seed}/step{row["step"]}',Valore=row['common_development_score'][i*3+j],
                    Numerosità=1,Run=[runs['joint_w'+arm['name']]],**{'Specifica di valutazione':[curve_ev],
                    'Artefatti dettagliati':[arts['training_curves.json']], 'Descrizione':json.dumps(dict(step=row['step'],seed=seed,elapsed_seconds=row['elapsed_seconds']))}))
    document=(repo/'experiments/task39_failure_analysis_and_next_design.md').read_text(encoding='utf-8')
    finding=put('findings','outcome',Nome='Task39 partial repairs, no fully passing candidate',Esito='Misto',
        Esperimenti=[exp],Osservazioni=observations,**{'Risultato':document,'Limitazioni':final['limits']})
    for name in ('coverage','exposure','bias','current_supervision','small_currents','budget'):
        supported = name in ('exposure','small_currents','budget')
        claim=put('claims',name,Nome='Task39 '+name,Stato='Supportata nel dominio' if supported else 'Indeterminata')
        detail=final['budget_contrasts'] if name=='budget' else [c for c in final['planned_contrasts'] if c['factor']==('mode' if name=='exposure' else name)]
        put('evidence','result-'+name,Nome='Task39 scoped evidence '+name,Esito='Sostiene' if supported else 'Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],'Argomentazione':json.dumps(detail),
               'Limiti e spiegazioni alternative':'Scope-specific contrasts, not global solution. Budget only short loss; exposure supported only factorized. No independent test.', 'Data valutazione':'2026-10-18'})
    hypotheses={
        'long_horizon':'0.8ms training windows inadequately constrain long relaxation and history retention.',
        'state_identifiability':'Trajectory-only coverage leaves affine dynamics weakly identified over physical states.',
        'combined_loss':'Relative current supervision can complement factorized recurrent training; interaction untested.',
        'slow_optimization':'Small update fractions and slow increments are poorly conditioned under sigmoid parameterization.',
        'gate_interference':'Shared trunk may trade accuracy between channel kinetics; requires per-gate gradient probes.',
        'event_ambiguity':'Some temporal failures involve ambiguous or weak events; inspect traces without changing thresholds.'}
    new_claims=[]
    for name,statement in hypotheses.items():
        new_claims.append(put('claims','residual-'+name,Nome='Task39 residual: '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,
            **{'Limiti':'Post-result hypothesis; not tested causal attribution. See Task39 failure analysis and proposed39b.'}))
    put('decisions','post-result',Nome='Task39 NO-GO: propose39b, retain Task40 block',Esito='Modificare',Risultati=[finding],
        Motivazione=document,**{'Data':'2026-10-18'})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-ionic-abstraction-20261017-task39-v1',
        'Nome':'Task39 — Learnability, bias e mini scaling','Stato':'Conclusa','Descrizione':'Technically completed, scientific NO-GO. Task39b proposed, not launched; Task40 blocked.'})
    print(json.dumps(dict(stage='ready',observations=len(observations),cases=sum(map(len,groups.values())))),flush=True)
    m.commit_batch(); write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot())
    verification=m.verify(); assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(observations),
        recurrence_cases=1872,channel_metrics=20592,new_open_hypotheses=6,verification=verification,airtable_accessed=False))
    print(json.dumps(dict(valid=True,observations=len(observations),verification=verification)),flush=True)


if __name__=='__main__': main()
