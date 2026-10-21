"""Complete Task40 results and post-hoc audit; authoritative local mirror only."""
import json,hashlib
from functools import lru_cache
from collections import defaultdict
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task40_audited_v1'
    read=lambda n:json.loads((folder/n).read_text())
    audit=read('audit.json');attribution=read('attribution_audit.json');synthesis=read('synthesis.json')
    assert audit['valid'] and attribution['valid'] and synthesis['valid']
    assert not audit['task41_authorized'] and not read('final_report.json')['task41_preparation_authorized']
    m=ValidatedBatchMirror();ids={};arts={};observations=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task40-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    for path in [*sorted(folder.glob('*')),*(repo/'scripts'/n for n in ['audit_task40.py','task40_failure_attribution.py','task40_audit_synthesis.py']),Path(__file__)]:
        if not path.is_file() or path.name.startswith('memory_'):continue
        rel=path.relative_to(repo).as_posix();key=path.name
        arts[key]=put('artifacts',key,Nome='Task40 '+key,Tipo='Checkpoint' if path.suffix=='.pt' else ('Codice' if path.suffix=='.py' else 'Report'),
            Versione='v1',Percorso=rel,**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=ref('experiments-giada-task40-causal-suite-v1');protocol=ref('protocols-giada-task40-causal-suite-v1');runs={}
    limits='Post-hoc opened diagnostics, not independent confirmation. All original thresholds unchanged. Formula substitutions are attribution controls, not deployable candidates. Task41 not authorized. No training or model selection performed in this audit.'
    for label,report in read('causal_results.json').items():
        name,family,mode=label.split('/')
        for seed in (17,29,43):
            code=f'runs-giada-task40-causal-{name}-{family}-{mode}-{seed}-v1'
            runs[(label,seed)]=ref(code)
            m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata','Artefatti prodotti':[arts['causal_results.json'],arts['final_report.json'],arts[name+'_frozen.pt']],
                'Validità tecnica':json.dumps(dict(valid=True,scientific_suite_no_go=True,thresholds_changed=False)),
                'Hardware e ambiente':json.dumps(read('execution.json'))})
    def obs(name,data,file,run=None,value=None,count=1,kind='quality'):
        fields={'Nome':'Task40 '+name,'Descrizione':json.dumps(data,separators=(',',':')),'Numerosità':count,
            'Artefatti dettagliati':[arts[file]],'Specifica di valutazione':[ref('evaluations-giada-task40-causal-'+kind+'-v1')]}
        if run:fields['Run']=[run]
        if value is not None:fields['Valore']=value
        rid=put('observations',name,**fields);observations.append(rid);return rid
    for label,report in read('causal_results.json').items():
        for seed in (17,29,43):
            rows=[r for r in report['cases'] if r['seed']==seed]
            obs('trajectory-'+label+'-'+str(seed),dict(cases=rows), 'causal_results.json',runs[(label,seed)],sum(r['passed'] for r in rows),len(rows))
            if 'interventions' in report:
                rows=[r for r in report['interventions'] if r['seed']==seed]
                obs('intervention-'+label+'-'+str(seed),dict(cases=rows),'causal_results.json',runs[(label,seed)],sum(r['metrics']['passed'] for r in rows),len(rows),'causal')
    for file in ['attribution_controls.json','readout_controls.json','coefficient_controls.json']:
        for variant,rows in read(file).items():
            for seed in (17,29,43):
                selected=[r for r in rows if r['seed']==seed]
                obs(file+'-'+variant+'-'+str(seed),dict(cases=selected,diagnostic_oracle=True,candidate_eligible=False),file,runs[('J1/all/autonomous',seed)],sum(r['metrics']['passed'] for r in selected),len(selected),'causal')
    for r in read('failure_details.json'):
        if r['model']=='J1' and r['mode']=='autonomous':
            obs('failure-'+str(r['case'])+'-'+str(r['seed']),r,'failure_details.json',runs[('J1/all/autonomous',r['seed'])],0,1,'causal')
    # Retain complete diagnostics including all stress and floor metrics, costs, and successful channels.
    handled={'causal_results.json','attribution_controls.json','readout_controls.json','coefficient_controls.json'}
    for path in folder.glob('*.json'):
        if path.name in handled or path.name.startswith('memory_'):continue
        data=read(path.name)
        obs('diagnostic-'+path.name,data,path.name,kind='cost' if path.name=='cost.json' else 'causal')
    findings={k:put('findings',k,Nome='Task40 '+k,Esito='Misto',Esperimenti=[exp],Risultato=v,Limitazioni=limits,
        Osservazioni=observations if k=='outcome' else []) for k,v in synthesis['findings'].items()}
    analysis=put('analyses','audit',Nome='Task40 complete differential and kinetic attribution audit',Origine='Post hoc',Versione='v1',
        Procedura='scripts/audit_task40.py; scripts/task40_failure_attribution.py; scripts/task40_audit_synthesis.py',
        Configurazione=json.dumps(dict(audit=audit,attribution=attribution)),
        **{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for h in synthesis['hypotheses']:
        claim=put('claims',h['id'],Nome='After Task40: '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['statement'],
            **{'Condizioni di falsificazione':'Matched proposed contrast fails to improve independent FIT holdout and unchanged causal criteria: '+h['test'],
               'Limiti':limits+' Priority='+h['priority']+' Evidence='+h['evidence']})
        put('evidence',h['id'],Nome='Task40 scoped evidence '+h['id'],Esito='Sostiene' if h['id']=='relaxation_precision' else 'Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[findings['kinetic_control'],findings['causal_channel_control']],
               'Argomentazione':h['evidence'],'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-10-21'})
    # Original preregistered hypotheses receive evidence, without rewriting their meaning.
    for number,outcome,explanation in [
        (1,'Sostiene','J1 passes264/264 autonomous trajectory gates; this does not include intervention-response qualification.'),
        (2,'Sostiene','22failures localize to calcium. Differential formula floor passes56/56; calcium-only reproduces22.'),
        (3,'Sostiene','No SK current failures in the tested cases; no global/general stability claim.'),
        (4,'Contraddice','22/168 autonomous differential cases fail unchanged registered criteria.'),
        (5,'Indeterminata','Cost is recorded against restricted formula baseline; no competitive acceleration qualification.')]:
        put('evidence','registered-'+str(number),Nome='Task40 registered H'+str(number),Esito=outcome,
            **{'Affermazione valutata':[ref(f'claims-giada-task40-causal-{number}-v1')],
               'Risultati a sostegno':list(findings.values()),'Argomentazione':explanation,
               'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-10-21'})
    put('lessons','differential-precision',Nome='Trajectory fidelity does not guarantee intervention fidelity',Lezione='A small state-dependent kinetic error can pass all absolute trajectory gates yet fail differential intervention gates. Audit the differential numerical floor and probe equilibrium versus relaxation before increasing global capacity.',Risultati=list(findings.values()),Limiti=limits)
    decision=put('decisions','40b-not41',Nome='Task40 scientific NO-GO; propose targeted40b, not41',Esito='Continuare',Risultati=list(findings.values()),Motivazione='Continue only targeted diagnosis; NO-GO to Task41. '+synthesis['findings']['decision'],**{'Data':'2026-10-21'})
    put('actions','40b',Nome='Proposed Task40b calcium kinetic correction matrix',Stato='Proposta',Descrizione=json.dumps(synthesis['next_experiment']),**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task40-causal-suite-v1','Stato':'Concluso','Descrizione':synthesis['findings']['outcome']})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task40-causal-execute-v1','Stato':'Conclusa','Descrizione':synthesis['findings']['outcome']})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(observations))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    checks={}
    for prefix,expected in [('trajectory-',6336),('intervention-',672)]:
        sql="SELECT count(*) cases,sum(json_array_length(json_extract(j.value,'$."+('current' if prefix=='trajectory-' else 'metrics')+".channels'))) channels FROM observations o,json_each(o.Descrizione,'$.cases') j WHERE o.\"Codice stabile\" LIKE 'observations-giada-task40-result-"+prefix+"%'"
        row=m.query(sql)['rows'][0];assert row['cases']==expected and row['channels']==expected*11,row;checks[prefix]=row
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observations=len(observations),readback=checks,verification=v,airtable_accessed=False))
    print(json.dumps(dict(valid=True,readback=checks,records=len(ids))),flush=True)

from pathlib import Path
if __name__=='__main__':main()
