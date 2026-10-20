"""Register recovered 39i evidence and the approved plateau investigation.

Only the versioned SQLite interface is used. Primary and diagnostic observations
remain separate; no recovered diagnostic checkpoint becomes selectable.
"""
import hashlib
import json
import shutil
from collections import defaultdict
from functools import lru_cache
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


HYPOTHESES = [
    ('objective_precision', 'The global coefficient/trajectory loss underweights the local accuracy required by shallow current crossings.',
     'Compare bounded full-relaxation current losses with coefficient losses on identical FIT data and architecture.',
     'No paired improvement in independent atomic holdouts or unchanged full-trace timing despite sufficient FIT convergence.'),
    ('local_representation', 'A small joint u/z correction can resolve localized residual bias without perturbing unrelated channels.',
     'Freeze the I3 parent; compare channel-local and parameter-matched generic residual features, with the same losses and data.',
     'The local residual does not improve held-out relaxation timing over its matched generic control.'),
    ('fine_optimization', 'Stochastic fine-tuning and persistent clipping prevent reaching the precision required after warm start.',
     'Compare fixed-FIT deterministic refinement against matched Adam, report gradient evaluations and clipping; do not equate clipping with proof.',
     'Deterministic refinement reaches a smaller FIT objective but does not improve holdout error, or cannot lower FIT loss.'),
    ('information_coverage', 'Uniform FIT sampling spends too little effort on long low-occupancy relaxations with sensitive crossings.',
     'Equal-count ordinary versus sensitivity-stratified independent FIT, with shared model, objective and optimizer.',
     'Matched coverage does not improve independent nearby voltages, initial states and durations.'),
    ('joint_interference', 'A correction learnable for isolated K_Pst/Nap_Et2 relaxations is lost during joint training.',
     'Atomic versus joint-channel optimization with identical initialization and per-channel budgets, followed by unchanged full-suite evaluation.',
     'Isolated fitting fails too, or joint fitting matches it without collateral regressions.'),
]


def main():
    repo = ROOT.parent
    source = repo/'artifacts/task39i_recovered_cpu'
    folder = repo/'experiments/results/task39i_recovered_v1'
    folder.mkdir(exist_ok=True)
    # Preserve all JSON diagnostics and the selected weights in versioned results.
    for path in list(source.rglob('*.json')) + [source/'frozen_checkpoint.pt']:
        if 'recovery_code' in path.parts:
            continue
        dest = folder/path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == path.read_bytes(), str(dest)
        else:
            shutil.copy2(path, dest)
    plateau_path = repo/'artifacts/task39i_plateau_analysis/memory_and_residual_queries.json'
    target = folder/'plateau_queries.json'
    if target.exists():
        assert target.read_bytes() == plateau_path.read_bytes()
    else:
        shutil.copy2(plateau_path, target)
    read = lambda name: json.loads((folder/name).read_text(encoding='utf-8'))
    final, freeze = read('final_report.json'), read('selection_freeze.json')
    finite = read('all_checkpoints_finite.json')
    assert final['valid'] and finite['valid'] and not final['thresholds_changed']
    assert not freeze['stress_used_for_selection'] and not freeze['independent_confirmation_accessed']
    assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest() == freeze['checkpoint_sha256']
    payload = {k:v for k,v in freeze.items() if k != 'freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest() == freeze['freeze_sha256']
    m = ValidatedBatchMirror(); ids = {}
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39i-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid
        return rid
    arts={}
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or path.name.startswith('memory_'):continue
        name=path.relative_to(folder).as_posix()
        arts[name]=put('artifacts',name,Nome='Task39i recovered '+name,Tipo='Checkpoint' if path.suffix=='.pt' else 'Report',
            Versione='recovered-v1',Percorso=path.relative_to(repo).as_posix(),
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=ref('experiments-giada-task39i-temporal-suite-v1')
    outcome='Recovered without retraining: best development66/72 versusH8 65/72;I3 stress62/72 versus61/72. Six remaining development failures are two constant-voltage relaxations x3seeds. All14registered contrasts unsupported;Task40blocked. CPU recovery preserves144/144H8pass/fail outcomes;no GPU speed claim.'
    limits='Development-informed, not independent confirmation.36members share data/parents. CPU recovery differs from CUDA environment; equivalence audited, not bitwise identity. ParentH8oracles are not I3oracles. Historical analogies and gradient/clipping correlations are not causal proof.'
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task39i-temporal-suite-v1','Stato':'Concluso','Descrizione':outcome})
    runs={}
    for index,a in enumerate(freeze['arms']):
        for j,seed in enumerate(freeze['seeds']):
            code=f'runs-giada-task39i-temporal-{a["name"]}-{seed}-v1';runs[('joint_w'+a['name'],seed)]=ref(code)
            m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata','Artefatti prodotti':list(arts.values()),
                'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,chosen_step=freeze['chosen_steps'][index*3+j],budget=120000,revision=read('revision.json'))),
                'Hardware e ambiente':json.dumps(dict(training=read('execution.json'),recovery=read('evaluation_recovery.json'))),
                'Validità tecnica':json.dumps(dict(valid=True,finite=finite,decision=final['decision'],retraining=False))})
    for label,task,arm in [('joint_wfrozen39h_H8','39h','H8'),('joint_wfrozen39g_A0','39g','A0'),('joint_wfrozen39g_A6','39g','A6')]:
        for seed in freeze['seeds']:
            suffix='-retry2' if task=='39g' else ''
            runs[(label,seed)]=ref(f'runs-giada-task{task}-temporal-{arm}-{seed}{suffix}-v1')
    baseline=ref('runs-giada-task38-joint-transition-kaggle-local161-v1')
    obs=[]; expected={}
    def observe(name,data,file,run=None,count=1,value=None,kind='oracles'):
        fields={'Nome':'Task39i '+name,'Descrizione':json.dumps(data,separators=(',',':')),'Numerosità':count,
                'Artefatti dettagliati':[arts[file]],'Specifica di valutazione':[ref('evaluations-giada-task39i-temporal-'+kind+'-v1')]}
        if run:fields['Run']=[run]
        if value is not None:fields['Valore']=value
        rid=put('observations',name,**fields);obs.append(rid)
    for path in sorted(folder.rglob('recurrence_evaluation.json')):
        name=path.relative_to(folder).as_posix();tag='primary' if path.parent==folder else path.parent.name
        rows=read(name);expected[tag]=len(rows);group=defaultdict(list)
        for r in rows:group[(r['model'],r['role'],r['family'],r['seed'])].append(r)
        for (model,role,family,seed),rr in group.items():
            run=baseline if model=='frozen_local161_SKcanonical' else runs[(model,seed)]
            observe(f'{tag}-{model}-{role}-{family}-{seed}',dict(cases=rr,selection_eligible=tag=='primary' and role=='development'),name,
                    run,len(rr),sum(r['metrics']['passed'] for r in rr),'quality')
    group=defaultdict(list)
    for r in read('frozen_oracle_attribution.json'):
        group[(r['parent_arm'],r['precision'],r['intervention'],r['family'],r['seed'])].append(r)
    for (arm,precision,mode,family,seed),rr in group.items():
        observe(f'oracle-{arm}-{precision}-{mode}-{family}-{seed}',dict(cases=rr,parent_task='39h',selection_eligible=False),
                'frozen_oracle_attribution.json',runs[('joint_wfrozen39h_H8',seed)],len(rr),sum(r['metrics']['passed'] for r in rr))
    for curve in read('training_curves.json'):
        for index,a in enumerate(freeze['arms']):
            for j,seed in enumerate(freeze['seeds']):
                k=index*3+j
                observe(f'curve-{a["name"]}-{seed}-{curve["step"]}',dict(step=curve['step'],panel=curve['panel'][k],elapsed_seconds=curve.get('elapsed_seconds'),evaluation_recovered=curve.get('evaluation_recovered',False)),
                    'training_curves.json',runs[('joint_w'+a['name'],seed)],value=curve['common_development_score'][k],kind='selection')
    for path in sorted(folder.glob('*.json')):
        if path.name in ('recurrence_evaluation.json','frozen_oracle_attribution.json','training_curves.json','plateau_queries.json') or path.name.startswith('memory_'):continue
        observe('diagnostic-'+path.name,read(path.name),path.name)
    plateau=read('plateau_queries.json')
    for q in plateau['queries']:
        observe('query-'+q['name'],dict(query=q,read_only=True,exploratory=True),'plateau_queries.json',count=len(q['rows']))
    for key,value in plateau.items():
        if key not in ('queries','limits'):observe('plateau-'+key,dict(value=value,posthoc=True),'plateau_queries.json')
    finding=put('findings','outcome',Nome='Task39i66/72 and localized relaxation plateau',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,
        Risultato=json.dumps(dict(outcome=outcome,comparison=read('comparison_with_39h.json'),contrasts=read('paired_contrasts.json'))),Limitazioni=limits)
    analysis=put('analyses','plateau',Nome='Task39i historical SQL and constant-voltage localization',Origine='Post hoc',Versione='v1',
        Procedura='scripts/explore_task39i_plateau.py',Configurazione=json.dumps(plateau),
        **{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':[finding]})
    for key,statement,test,falsifier in HYPOTHESES:
        claim=put('claims',key,Nome='After39i '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,
            **{'Condizioni di falsificazione':falsifier,'Limiti':limits+' Proposed test: '+test})
        put('evidence',key,Nome='Task39i rationale '+key,Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[finding],
               'Argomentazione':test+' See linked plateau analysis; hypothesis not yet established.',
               'Limiti e spiegazioni alternative':falsifier,'Data valutazione':'2026-10-20'})
    put('lessons','objective',Nome='Low mean error is not a certificate of relaxation timing',Lezione=outcome+' Global score improves35/36 while worst panel ratio worsens36/36;do not optimize one scalar blindly.',
        Risultati=[finding],Limiti=limits)
    decision=put('decisions','next',Nome='Task40 blocked; authorize39j atomic relaxation diagnosis',Esito='Modificare',Risultati=[finding],Motivazione=outcome,**{'Data':'2026-10-20'})
    put('actions','39j',Nome='Prepare39j matched atomic relaxation experiments',Stato='Proposta',Descrizione=json.dumps(HYPOTHESES),**{'Decisioni di origine':[decision]})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task39i-temporal-execute-v1','Stato':'Conclusa','Descrizione':outcome})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(obs))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    checks={}
    for tag,count in expected.items():
        sql="SELECT count(*) cases,sum(json_array_length(json_extract(j.value,'$.metrics.channels'))) channels FROM observations o,json_each(o.Descrizione,'$.cases') j WHERE o.\"Codice stabile\" LIKE 'observations-giada-task39i-result-"+tag+"-%'"
        row=m.query(sql)['rows'][0];assert row['cases']==count and row['channels']==count*11,row;checks[tag]=row
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(obs),readback=checks,verification=verification,airtable_accessed=False,drive_backup_updated=False))
    print(json.dumps(dict(valid=True,records=len(ids),readback=checks)),flush=True)


if __name__=='__main__':main()
