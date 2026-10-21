"""Register complete41c attribution, typed evidence and proposed41d; local only."""
import hashlib,json
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41c_audited_v1'
    read=lambda n:json.loads((folder/n).read_text())
    a,s=read('audit.json'),read('synthesis.json');assert a['valid'] and s['valid'] and not a['task42_authorized']
    assert read('stress_replay_validation.json')['valid']
    m=ValidatedBatchMirror();ids={};arts={};obs=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41c-audit-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    paths=[f for f in folder.iterdir() if f.is_file() and f.name!='memory_receipt.json']
    paths += [repo/'scripts/audit_task41c.py',repo/'scripts/probe_task41c_responses.py',repo/'scripts/synthesize_task41c_audit.py',Path(__file__)]
    for path in paths:
        arts[path.name]=put('artifacts',path.name,Nome='Task41c audit '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=ref('experiments-giada-task41bc-41c-v1')
    runs={seed:ref(f'runs-giada-task41bc-41c-P110-{seed}-v1') for seed in (17,29,43)}
    def observation(name,value,file,kind='quality',run=None):
        fields=dict(Nome='Task41c '+name,Descrizione=json.dumps(value,ensure_ascii=False,separators=(',',':')))
        fields.update({'Artefatti dettagliati':[arts[file]],'Specifica di valutazione':[ref(f'evaluations-giada-task41bc-41c-{kind}-v1')]})
        if run:fields['Run']=[runs[run]]
        oid=put('observations',name,**fields);obs.append(oid);return oid
    for path in folder.glob('*.json'):
        if path.name=='memory_receipt.json':continue
        observation(path.stem,read(path.name),path.name,'stress' if path.name.startswith('stress') else 'quality')
    for r in read('response_failures.json'):observation('response-'+str(r['case']['id'])+'-'+str(r['model_seed']),r,'response_failures.json',run=r['model_seed'])
    for r in read('stress_failures.json'):observation('stress-'+str(r['index'])+'-'+str(r['model_seed']),r,'stress_failures.json','stress',r['model_seed'])
    originals=[ref('observations-giada-task41c-completed-'+n+'-v1') for n in ('causal_results.json','fresh_stress_results.json','cost.json','formula_floor.json','native_current_phase_floor.json')]
    findings={name:put('findings',name,Nome='Task41c '+name,Esito='Misto',Esperimenti=[exp],Risultato=text,Limitazioni=s['limits'],Osservazioni=obs+originals) for name,text in s['findings'].items()}
    put('analyses','attribution',Nome='Task41c immutable failure replay and single-coefficient interventions',Origine='Post hoc',Versione='v1',Procedura='scripts/audit_task41c.py; scripts/probe_task41c_responses.py; scripts/synthesize_task41c_audit.py',Configurazione=json.dumps(a),**{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for e in s['original_claim_evidence']:
        code=f"claims-giada-task41bc-41c-{e['id']}-v1"
        m.local_upsert('claims',{'Codice stabile':code,'Stato':e['state'],'Limiti':e['argument']+' '+s['limits']})
        put('evidence','original-'+str(e['id']),Nome='Task41c evaluated H'+str(e['id']+1),Esito=e['outcome'],**{'Affermazione valutata':[ref(code)],'Risultati a sostegno':[findings[e['finding']]],'Argomentazione':e['argument'],'Limiti e spiegazioni alternative':s['limits'],'Data valutazione':'2026-10-21'})
    for h in s['hypotheses']:
        claim=put('claims',h['id'],Nome='Task41c residual '+h['id'],Tipo='Ipotesi',Stato=h['status'],Enunciato=h['statement'],Limiti=s['limits'],**{'Condizioni di falsificazione':h['falsifier']})
        put('evidence',h['id'],Nome='Task41c scoped evidence '+h['id'],Esito={'Supportata nel dominio':'Sostiene','Contraddetta nel dominio':'Contraddice','Aperta':'Indeterminata'}[h['status']],**{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[h['evidence']]],'Argomentazione':h['statement']+' '+h['falsifier'],'Limiti e spiegazioni alternative':s['limits'],'Data valutazione':'2026-10-21'})
    for name,text in [('fast_equilibrium_slow_offset','Ca_HVA repeats the earlier Nap pattern: a fast activation equilibrium bias rescales a slow current tail; do not assume a missing slow state merely from long offset error.'),('paired_regressions','Equal aggregate counts can hide a trade: B6 and P110 each334/336 responses, but two failures fixed and two introduced. Always compare case identities on the same inputs.'),('precision_generalization','Opened repair success does not guarantee independent response precision. Protect coefficient value and local response, then check autonomous feedback and historical long tails.'),('conditioning','A shallow current slope amplifies tiny current bias into milliseconds. Keep original pass/fail; diagnose residual/slope rather than relaxing timing gates.')]:
        put('lessons',name,Nome='Task41c '+name,Lezione=text,Risultati=list(findings.values()),Limiti=s['limits'])
    prior=[ref(link['code']) for link in s['memory_links'] if link['code'].startswith('findings-')]
    put('lessons','memory_links',Nome='Task41c historical patterns re-tested on current candidate',Lezione=json.dumps(s['memory_links']),Risultati=prior+list(findings.values()),Limiti=s['limits'])
    decision=put('decisions','propose41d',Nome='Keep41c NO-GO; propose41d targeted precision and feedback contrasts',Esito='Continuare',Motivazione=json.dumps(s['next_experiment'])+' '+s['limits'],Risultati=list(findings.values()),**{'Data':'2026-10-21'})
    put('actions','propose41d',Nome='Task41d proposed single-notebook matrix; not launched',Stato='Proposta',Descrizione=json.dumps(s['next_experiment']),**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41bc-41c-v1','Descrizione':s['findings']['outcome']+' '+s['findings']['decision'],'Stato':'Concluso'})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(obs))),flush=True)
    m.commit_batch();print('Committed; exporting full snapshot and verifying',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    counts=m.query("SELECT [Codice stabile] code,json_array_length(json_extract(Descrizione,'$.rows')) n FROM observations WHERE [Codice stabile] IN ('observations-giada-task41c-audit-response_counterfactuals-v1','observations-giada-task41c-audit-stress_counterfactuals-v1','observations-giada-task41c-audit-response_ablations-v1')")['rows']
    assert sorted(r['n'] for r in counts)==[12,35,60],counts
    claims=m.query("SELECT [Codice stabile] code,Stato FROM claims WHERE [Codice stabile] LIKE 'claims-giada-task41c-audit-%' OR [Codice stabile] LIKE 'claims-giada-task41bc-41c-%'")['rows'];assert len(claims)==11
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=verification,probe_readback=counts,claim_states=claims,original_observations_preserved=originals,airtable_accessed=False,backup_policy='local only',task42_authorized=False,next_experiment_launched=False))
    print('Full Task41c audit,typed evidence and proposal registered and verified',flush=True)
if __name__=='__main__':main()
