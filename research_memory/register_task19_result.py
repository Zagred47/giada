"""Verify Task19 archive and register complete scalar results in local mirror."""
import hashlib
import json
import subprocess
import zipfile
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;archive=repo/'artifacts/giada_task19_single_gate_9f5d49f_37c01f4c.zip'
    root=repo/'experiments/results/task19_kaggle_9f5d49f';root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze=json.loads(z.read('selection_freeze.json'));claimed=freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==claimed
        assert not freeze['fresh_accessed']
        for k,v in freeze['checkpoint_hashes'].items():assert hashlib.sha256(z.read(k)).hexdigest()==v
        provenance=json.loads(z.read('code_provenance.json'));assert not provenance['dirty_runtime']
        for path,expected in provenance['sources'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',provenance['code_revision']+':'+path],cwd=repo)).hexdigest()==expected
        for name in z.namelist():
            if '/' not in name and name.endswith('.json'):(root/name).write_bytes(z.read(name))
    report=json.loads((root/'final_report.json').read_text())
    assert report['valid'] and report['task20_authorized'] and not report['fresh_used_for_selection']
    assert all(json.loads((root/n).read_text())['valid'] for n in ('native_audit.json','equivalence_preflight.json'))
    write_json(root/'archive_audit.json',dict(valid=True,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),CRC_valid=True,freeze_checkpoint_source_hashes_valid=True))
    m=ValidatedBatchMirror()
    def put(t,s,**f):return m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task19-{s}-v1',**f})['record_id']
    def ref(t,s):return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task19-{s}-v1'+"'")['rows'][0]['record_id']
    arts=[]
    for path in root.glob('*.json'):
        arts.append(put('artifacts','result-'+path.stem,**{'Nome':'Task19 '+path.name,'Tipo':'Report','Percorso':path.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'}))
    obs=[]
    def record(row,suffix,role):
        rid=put('runs',suffix,**{'Nome':'Task19 '+suffix,'Stato':'Completata','Braccio':[ref('arms',f'{row["channel"]}-w{row["width"]}-{row["objective"]}')],'Blocco':[ref('blocks',role)],'Seed':str(row['seed']),'Configurazione effettiva':f'width{row["width"]};step{row["step"]}','Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(report['environment']),'Validità tecnica':'Oracle, equivalenza, freeze, checkpoint e sorgenti verificati.'})
        for domain,metrics in row['metrics'].items():
            for k,v in metrics.items():
                if k=='finite':continue
                count=(2814 if role=='development' else 5642) if domain=='state_extrema' else (2048 if role=='development' or domain.startswith('ood') else 4096)
                obs.append(put('observations',suffix+'-'+domain+'-'+k,**{'Nome':f'Task19 {suffix} {domain} {k}','Valore':v,'Numerosità':count,'Run':[rid],'Specifica di valutazione':[ref('evaluations',k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':domain,'Descrizione':'Errore gate adimensionale; corrente mA/cm². Tuple, non repliche indipendenti. OOD diagnostico.'}))
        for horizon,metrics in row.get('rollout',{}).items():
            for k,v in metrics.items():
                if k=='finite':continue
                obs.append(put('observations',suffix+'-roll'+horizon+'-'+k,**{'Nome':f'Task19 {suffix} rollout{horizon} {k}','Valore':v,'Numerosità':256,'Run':[rid],'Specifica di valutazione':[ref('evaluations','rollout-'+k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':'held_voltage_'+horizon,'Descrizione':'128V×stati0/1, dt1ms; non voltaggio variabile.'}))
    for row in report['learned']:record(row,f'fresh-{row["channel"]}-{row["objective"]}-s{row["seed"]}','fresh')
    for row in json.loads((root/'development_ladder.json').read_text()):record(row,f'dev-{row["channel"]}-{row["objective"]}-w{row["width"]}-step{row["step"]}-s{row["seed"]}','development')
    findings=[]
    for suffix,title,outcome,result in (
        ('transfer','Ih/Im confermati3/3','Positivo','Rate-supervised Ih w32/15k e Im w16/15k passano tutti domini richiesti e rollout. WorstRMSE .000445393/.000639889; max .002435081/.004577217. Task20 autorizzata.'),
        ('supervision','Supervisione rate e controllo transition-only','Misto','Rate-supervised passa entrambi3/3; transition-only non passa universalmente. Contrasti completi per seed/capacità/budget conservati: non si assume miglioramento monotono in ogni metrica.'),
        ('compute','Accuratezza non equivale a speedup','Negativo','Batch4096 eagerCUDA: Ih MLP .512676ms vsformula .325900ms vsLUT .301732ms; Im MLP .508293ms vsformula .239067ms vsLUT .298428ms. Rete piùlenta in questo benchmark, LUT piùaccurata. Non inferire speedupneurone.'),
        ('ood','OOD separato e ancora non risolto','Misto','Dominio confermato[-135,75]mV; OOD negativo include comportamento canonico Ih vicino-154.9. Errori OOD conservati separatamente, nessun successo universale di estrapolazione.')):
        findings.append(put('findings',suffix,**{'Nome':title,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Osservazioni':obs,'Risultato':result,'Limitazioni':report['scope']}))
    for k in ('single_gate_transfer','rate_identifiability','capacity_budget','state_rollout','numerical_reference'):
        fid=findings[0 if k in ('single_gate_transfer','state_rollout') else 1 if k in ('rate_identifiability','capacity_budget') else 2]
        put('evidence',k,**{'Nome':'Task19 '+k,'Esito':'Sostiene' if k in ('single_gate_transfer','state_rollout','numerical_reference') else 'Indeterminata','Affermazione valutata':[ref('claims',k)],'Risultati a sostegno':[fid],'Argomentazione':'Esito e contrasti completi negli artefatti; non si deduce beneficio uniforme da sola promozione.','Limiti e spiegazioni alternative':report['scope']})
        if k in ('single_gate_transfer','state_rollout','numerical_reference'):put('claims',k,**{'Stato':'Supportata nel dominio'})
    put('decisions','to20',**{'Nome':'Task19 autorizza Task20 SK_E2','Esito':'Continuare','Risultati':findings,'Motivazione':'Criteri preregistrati rate3seed/2canali superati; fresh dopo freeze.','Condizioni di revisione':'SK_E2 calcio-dipendente separato, nessuna promessa speedup.'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':arts,'Validità tecnica':'COMPLETE exit0; auditCRC/hashes/native/equivalence validi; GO.'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,observations=len(obs),findings=len(findings),airtable_accessed=False)))

if __name__=='__main__':main()
