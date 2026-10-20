"""Full 39j result/audit registration through the versioned SQLite mirror."""
import hashlib,json
from collections import defaultdict
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39j_audited_v1'
    read=lambda n:json.loads((folder/n).read_text())
    audit=read('audit.json');attribution=read('attribution_audit.json');freeze=read('selection_freeze.json')
    assert audit['valid'] and attribution['valid'] and read('cpu_J1_replay/audit.json')['valid']
    m=ValidatedBatchMirror();ids={};arts={};obs=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39j-result-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or path.name.startswith('memory_'):continue
        name=path.relative_to(folder).as_posix();arts[name]=put('artifacts',name,Nome='Task39j '+name,Tipo='Checkpoint' if path.suffix=='.pt' else 'Report',Versione='v1',
            Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=ref('experiments-giada-task39j-atomic-suite-v1');runs={}
    for idx,a in enumerate(freeze['arms']):
        for k,seed in enumerate(freeze['seeds']):
            code=f'runs-giada-task39j-atomic-{a["name"]}-{seed}-v1';runs[('joint_w'+a['name'],seed)]=ref(code)
            m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata','Artefatti prodotti':list(arts.values()),
                'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,budget=10000,chosen_step=freeze['chosen_steps'][idx*3+k],actual_revision=read('revision.json'))),
                'Hardware e ambiente':json.dumps(read('execution.json')),'Validità tecnica':json.dumps(dict(valid=True,finite=True,cpu_J1_replay=read('cpu_J1_replay/audit.json')))})
    for seed in freeze['seeds']:runs[('joint_wfrozen39i_I3',seed)]=ref(f'runs-giada-task39i-temporal-I3-{seed}-v1')
    baseline=ref('runs-giada-task38-joint-transition-kaggle-local161-v1')
    def observe(name,data,file,run=None,value=None,count=1,kind='quality'):
        fields={'Nome':'Task39j '+name,'Descrizione':json.dumps(data,separators=(',',':')),'Numerosità':count,'Artefatti dettagliati':[arts[file]],
                'Specifica di valutazione':[ref('evaluations-giada-task39j-atomic-'+kind+'-v1')]}
        if run:fields['Run']=[run]
        if value is not None:fields['Valore']=value
        rid=put('observations',name,**fields);obs.append(rid)
    expected={}
    for tag,file in [('primary','recurrence_evaluation.json'),('cpu_J1','cpu_J1_replay/recurrence_evaluation.json')]:
        rr=read(file);expected[tag]=len(rr);groups=defaultdict(list)
        for r in rr:groups[(r.get('model','joint_wJ1'),r['role'],r['family'],r['seed'])].append(r)
        for (model,role,family,seed),rows in groups.items():
            run=baseline if model=='frozen_local161_SKcanonical' else runs[(model,seed)]
            observe(f'{tag}-{model}-{role}-{family}-{seed}',dict(cases=rows,cpu_replay=tag=='cpu_J1',independent_confirmation=False),file,run,
                sum(r['metrics']['passed'] for r in rows),len(rows))
    for c in read('training_curves.json'):
        for idx,a in enumerate(freeze['arms']):
            for k,seed in enumerate(freeze['seeds']):
                n=idx*3+k;observe(f'curve-{a["name"]}-{seed}-{c["step"]}',dict(step=c['step'],panel=c['panel'][n],fit=c['fit'][a['name']][k],
                    holdout=c['atomic_nonselectable'][a['name']][k],clipping_count=c['clipping_counts'][a['name']][k],elapsed_seconds=c['elapsed_seconds']),
                    'training_curves.json',runs[('joint_w'+a['name'],seed)],c['common_development_score'][n],kind='selection')
    for path in folder.rglob('*.json'):
        if path.name in ('recurrence_evaluation.json','training_curves.json') or path.name.startswith('memory_'):continue
        name=path.relative_to(folder).as_posix();observe('diagnostic-'+name,read(name),name,kind='atomic')
    limits='Repeated development,not independentconfirmation;24trajectories perrole x3trainingseeds. Two stressfailures remain;V/Ca imposed. No speedup/autonomouscompartment claim.8registeredcontrasts do not pass;freeze support is scoped,not universal.'
    findings={}
    for name,statement in attribution['findings'].items():
        findings[name]=put('findings',name,Nome='Task39j '+name,Esito='Misto',Esperimenti=[exp],Osservazioni=obs if name=='outcome' else [],
            Risultato=statement,Limitazioni=limits)
    put('analyses','audit',Nome='Task39j full aligned audit and independent CPU replay',Origine='Post hoc',Versione='v1',
        Procedura='scripts/audit_task39j.py; scripts/task39j_attribution_audit.py',Configurazione=json.dumps(attribution),
        **{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for key in ('objective_precision','local_representation','fine_optimization','information_coverage','joint_interference'):
        claim=ref('claims-giada-task39i-result-'+key+'-v1')
        explanation={'objective_precision':'Extra relaxationloss notnecessary:J1 passes. Cannot isolate newcoefficientobjective from residual/frozenparent package.',
            'local_representation':'Learnedlocalcorrection suffices, butglobalFourierJ4also passes;localitynecessity notestablished.',
            'fine_optimization':'StochasticJ1passeswithoutclipping;fullbatch notneeded and8xexposure confoundsoptimizerattribution.',
            'information_coverage':'Uniform1024anchorsalready suffice; sensitivecoverage addsno categoricaldevelopmentgain.',
            'joint_interference':'MatchedJ2/J8favors frozenparent underthisrecipe. J8improvesglobalcoefficienterror butselectionreturnsstep0;representationalnecessity notproved.'}[key]
        put('evidence',key,Nome='Task39j scoped evidence '+key,Esito='Indeterminata',**{'Affermazione valutata':[claim],
            'Risultati a sostegno':list(findings.values()),'Argomentazione':explanation,'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-10-20'})
    claim=put('claims','sufficient-package',Nome='FrozenI3 pluslearnedjointcorrection qualifiesdevelopment',Tipo='Ipotesi',Stato='Aperta',
        Enunciato=attribution['findings']['sufficient_package'],Limiti=limits)
    put('evidence','sufficient-package',Nome='Task39j observed package sufficiency',Esito='Sostiene',**{'Affermazione valutata':[claim],
        'Risultati a sostegno':[findings['outcome'],findings['sufficient_package']],'Argomentazione':'J1three seeds24/24;CPU144pass/failmatches;zero-correctioncontrolrestoresI3.',
        'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-10-20'})
    put('lessons','protect-learned-function',Nome='Protect learned function while correcting localized precision',Lezione=attribution['findings']['freeze_contrast'],Risultati=list(findings.values()),Limiti=limits)
    decision=put('decisions','prepare40',Nome='GO toTask40 preparation withfrozenJ1,not deployment',Esito='Continuare',Risultati=list(findings.values()),
        Motivazione=attribution['findings']['decision'],**{'Data':'2026-10-20'})
    put('actions','task40',Nome='PrepareTask40 causal reinsertion andmatchedcost',Stato='Proposta',Descrizione=attribution['findings']['decision']+' Carrytwo knownstressfailures;preservemixedprecisioncontract.',**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task39j-atomic-suite-v1','Stato':'Concluso','Descrizione':attribution['findings']['outcome']})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task39j-atomic-execute-v1','Stato':'Conclusa','Descrizione':attribution['findings']['outcome']})
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True);m.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid'];checks={}
    for tag,count in expected.items():
        sql="SELECT count(*) cases,sum(json_array_length(json_extract(j.value,'$.metrics.channels'))) channels FROM observations o,json_each(o.Descrizione,'$.cases') j WHERE o.\"Codice stabile\" LIKE 'observations-giada-task39j-result-"+tag+"-%'"
        row=m.query(sql)['rows'][0];assert row['cases']==count and row['channels']==count*11;checks[tag]=row
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,readback=checks,verification=v,airtable_accessed=False))
    print(json.dumps(dict(valid=True,records=len(ids),readback=checks)),flush=True)
if __name__=='__main__':main()
