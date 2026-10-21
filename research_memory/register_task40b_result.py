"""Complete Task40b result audit, effective erratum and scoped Task41 decision."""
import json,hashlib
from pathlib import Path
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task40b_audited_v1'
    read=lambda n:json.loads((folder/n).read_text())
    audit=read('audit.json');synthesis=read('synthesis.json');assert audit['valid'] and synthesis['valid'] and audit['task41_preparation_authorized']
    m=ValidatedBatchMirror();ids={};arts={};observations=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task40b-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    paths=[p for p in folder.rglob('*') if p.is_file() and 'contrast_recompute' not in p.parts and not p.name.startswith('memory_')]
    paths += [repo/'scripts/audit_task40b.py',repo/'scripts/task40b_audit_synthesis.py',repo/'tests/test_task40b_audit.py',Path(__file__),
        repo/'experiments/results/task40b_protocol_v1/documentation_errata.json']
    for path in paths:
        key=path.relative_to(folder).as_posix() if path.is_relative_to(folder) else path.name
        arts[key]=put('artifacts',key,Nome='Task40b '+key,Tipo='Checkpoint' if path.suffix=='.pt' else ('Codice' if path.suffix=='.py' else 'Report'),
            Versione='audited-v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=ref('experiments-giada-task40b-calcium-suite-v1');runs={}
    limits='Opened historical diagnostics and post-hoc package nomination, not independent confirmation. Unchanged scientific thresholds. B6 has256 active Ca correction weights versus B1 128; no universal unique-cause proof or speedup. Task41 preparation only; no Task41 executed, deployment or multicompartment authorization.'
    for name in audit['summaries']:
        for seed in (17,29,43):
            code=f'runs-giada-task40b-calcium-{name}-{seed}-v1';runs[(name,seed)]=ref(code)
            m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata','Artefatti prodotti':[arts['frozen_checkpoint.pt'],arts['causal_results.json'],arts['recurrence_evaluation.json'],arts['audit.json']],
                'Hardware e ambiente':json.dumps(read('execution.json')),'Validità tecnica':json.dumps(dict(valid=True,parameters_finite=True,baseline_reproduced=True,thresholds_changed=False,full_audit=True))})
    def obs(name,data,file,run=None,value=None,count=1,kind='quality'):
        fields={'Nome':'Task40b '+name,'Descrizione':json.dumps(data,separators=(',',':')),'Numerosità':count,
            'Artefatti dettagliati':[arts[file]],'Specifica di valutazione':[ref('evaluations-giada-task40b-calcium-'+kind+'-v1')]}
        if run:fields['Run']=[run]
        if value is not None:fields['Valore']=value
        rid=put('observations',name,**fields);observations.append(rid);return rid
    reports=read('causal_results.json')
    for label,report in reports.items():
        name,mode=label.split('/')
        for seed in (17,29,43):
            rows=[r for r in report['cases'] if r['seed']==seed]
            obs('trajectory-'+label+'-'+str(seed),dict(cases=rows),'causal_results.json',runs[(name,seed)],sum(r['passed'] for r in rows),len(rows))
            rows=[r for r in report['interventions'] if r['seed']==seed]
            obs('intervention-'+label+'-'+str(seed),dict(cases=rows),'causal_results.json',runs[(name,seed)],sum(r['metrics']['passed'] for r in rows),len(rows),'causal')
    recurrence=read('recurrence_evaluation.json')
    for model in sorted(set(r['model'] for r in recurrence)):
        for role in ('development','stress_diagnostic'):
            for seed in (17,29,43):
                rr=[r for r in recurrence if r['model']==model and r['role']==role and r['seed']==seed]
                obs('recurrence-'+model+'-'+role+'-'+str(seed),dict(cases=rr),'recurrence_evaluation.json',runs.get((model.removeprefix('joint_w'),seed)),sum(r['metrics']['passed'] for r in rr),len(rr))
    for label,rr in read('stress_ablation.json')['results'].items():
        obs('ablation-'+label,dict(cases=rr,post_hoc=True,candidate_eligible=False),'stress_ablation.json',None,sum(r['metrics']['passed'] for r in rr),len(rr))
    cpu=read('cpu_B6_replay/recurrence_evaluation.json')
    for seed in (17,29,43):
        rr=[r for r in cpu if r['seed']==seed]
        obs('cpu-recurrence-B6-'+str(seed),dict(cases=rr),'cpu_B6_replay/recurrence_evaluation.json',runs[('B6',seed)],sum(r['metrics']['passed'] for r in rr),len(rr))
    # Complete successful/failed numerical metrics are stored above exactly once.
    # Small diagnostic files retain curves, floors, selection, probes and limitations.
    exclude={'causal_results.json','recurrence_evaluation.json','stress_ablation.json','stress_details.json'}
    for path in folder.glob('*.json'):
        if path.name in exclude or path.name.startswith('memory_'):continue
        obs('diagnostic-'+path.name,read(path.name),path.name,kind='selection' if path.name in ('selection_freeze.json','training_curves.json') else 'causal')
    # Record exact original stress rows, including successful channel metrics.
    obs('two-original-stress',read('stress_details.json'),'stress_details.json',count=16)
    erratum=json.loads((repo/'experiments/results/task40b_protocol_v1/documentation_errata.json').read_text())
    obs('documentation-erratum',erratum,'documentation_errata.json')
    spec=read('preregistration.json')
    m.local_upsert('protocols',{'Codice stabile':'protocols-giada-task40b-calcium-suite-v1','Versione':'v1-audited-descriptive-erratum',
        'Procedura':json.dumps(dict(executed_preregistration=spec,documentation_erratum=erratum,
            effective_limits=[x for x in spec['limits'] if x!=erratum['inherited_text']],explicit_training_selection_authoritative=True)),
        'Hash preregistrazione':hashlib.sha256((folder/'preregistration.json').read_bytes()).hexdigest()})
    findings={name:put('findings',name,Nome='Task40b '+name,Esito='Misto',Esperimenti=[exp],Risultato=value,Limitazioni=limits,
        Osservazioni=observations if name=='outcome' else []) for name,value in synthesis['findings'].items()}
    put('analyses','audit',Nome='Task40b immutable checkpoint/trace/stress audit and Task40 comparison',Origine='Post hoc',Versione='v1',
        Procedura='scripts/audit_task40b.py; scripts/task40b_audit_synthesis.py',Configurazione=json.dumps(audit),
        **{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for e in synthesis['evidence']:
        put('evidence',e['id'],Nome='Task40b scoped '+e['id'],Esito=e['outcome'],
            **{'Affermazione valutata':[ref('claims-giada-task40-result-'+e['id']+'-v1')],
               'Risultati a sostegno':list(findings.values()),'Argomentazione':e['argument'],
               'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-10-21'})
    put('lessons','separate-stress-mechanisms',Nome='A differential repair need not repair an equilibrium-sensitive stress',
        Lezione='m+h relaxation correction fixes all22causal response failures and Ca_LVAst offset, but the Ca_HVA activation-boundary stress requires a further equilibrium correction in this frozen package. Preserve rare-case and timing gates; global aggregate severity can conceal specific progress.',Risultati=list(findings.values()),Limiti=limits)
    decision=put('decisions','nominate-B6-for41',Nome='Nominate frozen B6; GO to independent Task41 preparation only',Esito='Continuare',Risultati=list(findings.values()),
        Motivazione=json.dumps(synthesis['next_step'])+' '+limits,**{'Data':'2026-10-21'})
    put('actions','prepare41',Nome='Prepare independent Task41 level1 confirmation of B6',Stato='Proposta',Descrizione=json.dumps(synthesis['next_step']),**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task40b-calcium-suite-v1','Stato':'Concluso','Descrizione':synthesis['findings']['outcome']+' '+synthesis['findings']['decision']})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task40b-calcium-execute-v1','Stato':'Conclusa','Descrizione':synthesis['findings']['decision']})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(observations))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    checks={}
    for prefix,expected,keyname in [('trajectory-',4224,'current'),('intervention-',2688,'metrics'),('recurrence-',1296,'metrics'),('cpu-recurrence-',144,'metrics'),('ablation-',24,'metrics')]:
        sql="SELECT count(*) cases,sum(json_array_length(json_extract(j.value,'$."+keyname+".channels'))) channels FROM observations o,json_each(o.Descrizione,'$.cases') j WHERE o.\"Codice stabile\" LIKE 'observations-giada-task40b-result-"+prefix+"%'"
        row=m.query(sql)['rows'][0];assert row['cases']==expected and row['channels']==expected*11,row;checks[prefix]=row
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(observations),readback=checks,verification=v,
        task41_preparation_authorized=True,task41_executed=False,erratum_reconciled=True,airtable_accessed=False))
    print(json.dumps(dict(valid=True,readback=checks,records=len(ids))),flush=True)
if __name__=='__main__':main()
