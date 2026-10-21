"""Record immutable Task41 failure audit and proposed, unlaunched Task41b."""
import hashlib,json
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41_audited_v1'
    read=lambda n:json.loads((folder/n).read_text())
    audit=read('audit.json');synthesis=read('synthesis.json')
    assert audit['valid'] and synthesis['valid'] and not audit['task42_authorized']
    m=ValidatedBatchMirror();ids={};arts={};observations=[]
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41-audit-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    files=[p for p in folder.iterdir() if p.is_file() and not p.name.startswith('memory_')]
    files += [repo/'scripts/audit_task41.py',repo/'scripts/task41_failure_probes.py',repo/'scripts/synthesize_task41_audit.py',repo/'tests/test_task41_audit.py',Path(__file__)]
    for path in files:
        arts[path.name]=put('artifacts',path.name,Nome='Task41 audit '+path.name,
            Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='audited-v1',Percorso=path.relative_to(repo).as_posix(),
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    exp=ref('experiments-giada-task41-independent-suite-v1')
    runs={}
    for mode in ('imposed','autonomous'):
        for seed in (17,29,43):runs[(mode,seed)]=ref(f'runs-giada-task41-independent-B6-{mode}-{seed}-v1')
    def obs(name,data,file,kind='quality',value=None,count=1,run=None):
        fields={'Nome':'Task41 audit '+name,'Descrizione':json.dumps(data,ensure_ascii=False,separators=(',',':')),
            'Numerosità':count,'Artefatti dettagliati':[arts[file]],
            'Specifica di valutazione':[ref('evaluations-giada-task41-independent-'+kind+'-v1')]}
        if value is not None:fields['Valore']=value
        if run:fields['Run']=[run]
        rid=put('observations',name,**fields);observations.append(rid);return rid
    # Original full numerical reports already exist in the mirror; retain those
    # observations and reference them, without making a second giant copy.
    originals=[ref('observations-giada-task41-completed-'+n+'-v1') for n in
        ('causal_results.json','fresh_stress_results.json','gpu_formula_floor.json','native_current_phase_floor.json','cost.json')]
    for file in sorted(folder.glob('*.json')):
        if file.name.startswith('memory_') or file.name=='stress_counterfactuals.json':continue
        obs('diagnostic-'+file.name,read(file.name),file.name,'cost' if file.name=='cost_audit.json' else 'quality')
    probes=read('stress_counterfactuals.json')
    for name,variant in probes['variants'].items():
        for seed in (17,29,43):
            rows=[r for r in variant['rows'] if r['model_seed']==seed]
            obs('probe-'+name+'-'+str(seed),dict(post_hoc=True,candidate_eligible=False,cases=rows),
                'stress_counterfactuals.json','stress',sum(r['metrics']['passed'] for r in rows),len(rows))
    obs('probe-provenance',{k:v for k,v in probes.items() if k!='variants'},'stress_counterfactuals.json','stress')
    limits=('Task41 remains independent frozen NO-GO. Audit probes use already opened cases and are not candidates, training, new confirmation or permission to waive thresholds. '
        'CPU replay is not bitwise GPU replication; pass flags/event times reproduced and numerical differences recorded. '
        'Single compartment,40ms coupled /1200ms imposed; no universal bifurcation coverage or quality-qualified speedup. New confirmation required after any repair. SQLite backup local only.')
    findings={}
    for name,result in synthesis['findings'].items():
        findings[name]=put('findings',name,Nome='Task41 '+name,Esito='Misto',Esperimenti=[exp],
            Risultato=result,Limitazioni=limits,Osservazioni=observations+originals if name=='outcome' else observations)
    analysis=put('analyses','immutable-replay',Nome='Task41 immutable metrics, numerical floors, event morphology and coefficient interventions',
        Origine='Post hoc',Versione='v1',Procedura='scripts/audit_task41.py; scripts/task41_failure_probes.py; scripts/synthesize_task41_audit.py',
        Configurazione=json.dumps(audit),**{'Run analizzati':list(runs.values()),'Input artefatti':list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for e in synthesis['original_claim_evidence']:
        code='claims-giada-task41-independent-'+e['id']+'-v1'
        m.local_upsert('claims',{'Codice stabile':code,'Stato':e['state'],'Limiti':e['argument']+' '+limits})
        put('evidence','original-'+e['id'],Nome='Task41 evaluated H'+e['id'],Esito=e['outcome'],
            **{'Affermazione valutata':[ref(code)],'Risultati a sostegno':[findings[e['finding']]],
            'Argomentazione':e['argument'],'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-10-21'})
    for h in synthesis['hypotheses']:
        claim=put('claims',h['id'],Nome='Task41 residual '+h['id'],Tipo='Ipotesi',Stato=h['status'],
            Enunciato=h['statement'],Limiti=limits,**{'Condizioni di falsificazione':h['falsifier']})
        put('evidence',h['id'],Nome='Task41 scoped evidence '+h['id'],
            Esito={'Supportata nel dominio':'Sostiene','Contraddetta nel dominio':'Contraddice','Aperta':'Indeterminata'}[h['status']],
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[h['evidence']]],
            'Argomentazione':h['statement']+' '+h['falsifier'],'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-10-21'})
    for name,text in [
        ('fast-gate-slow-tail','An observed slow current error can originate in the equilibrium of a fast multiplicative gate: Nap.m u-only oracle fixes the two long offsets. Do not equate long temporal scale with missing memory or h-rate error.'),
        ('multi-event-topology','A tiny secondary threshold crossing can move last-active offset by many milliseconds without shifting the main pulse. Preserve the registered failure and add connected-component/margin diagnostics; never silently smooth it away.'),
        ('specific-functions','Separate equilibrium bias, relaxation bias and numerical accumulation with frozen coefficient interventions before increasing network size or compute indiscriminately.'),
        ('memory-transfer','Past local corrections39j/40b motivate the next design; current-parent probes are still required because affected gates and error amplification differ.')]:
        put('lessons',name,Nome='Task41 '+name,Lezione=text,Risultati=list(findings.values()),Limiti=limits)
    # Preserve actual typed links to historical results; narrative patterns alone
    # are not evidence of causality in Task41.
    prior=[ref(r['code']) for r in synthesis['memory_links'] if r['code'].startswith('findings-')]
    put('lessons','historical-bridges',Nome='Task41 pattern: frozen local repair plus mechanism-specific probes',
        Lezione=json.dumps(synthesis['memory_links'],ensure_ascii=False),Risultati=prior+list(findings.values()),Limiti=limits)
    decision=put('decisions','no-go-next-diagnosis',Nome='Task41 NO-GO retained; propose diagnostic Task41b, not Task42',Esito='Continuare',
        Motivazione=json.dumps(synthesis['next_experiment'],ensure_ascii=False)+' '+limits,Risultati=list(findings.values()),**{'Data':'2026-10-21'})
    put('actions','propose41b',Nome='Task41b local precision factorial proposal; await implementation authorization',Stato='Proposta',
        Descrizione=json.dumps(synthesis['next_experiment'],ensure_ascii=False),**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41-independent-suite-v1','Stato':'Concluso',
        'Descrizione':synthesis['findings']['outcome']+' '+synthesis['findings']['decision']})
    print(json.dumps(dict(stage='staged',records=len(ids),observations=len(observations))),flush=True)
    m.commit_batch();print('Task41 audit committed; exporting full snapshot',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot())
    v=m.verify();assert v['valid']
    sql="SELECT count(*) cases,sum(json_array_length(json_extract(j.value,'$.metrics.channels'))) channels FROM observations o,json_each(o.Descrizione,'$.cases') j WHERE o.\"Codice stabile\" LIKE 'observations-giada-task41-audit-probe-%'"
    check=m.query(sql)['rows'][0];assert check['cases']==324 and check['channels']==3564,check
    states=m.query("SELECT \"Codice stabile\",Stato FROM claims WHERE \"Codice stabile\" LIKE 'claims-giada-task41-%'")['rows']
    assert len(states)==12
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,observation_count=len(observations),
        verification=v,probe_readback=check,claim_states=states,original_full_reports_preserved=originals,
        airtable_accessed=False,task42_authorized=False,next_experiment_launched=False))
    print(json.dumps(dict(valid=True,records=len(ids),readback=check)),flush=True)

if __name__=='__main__':main()
