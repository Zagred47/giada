"""Verify Task20 archive and register complete scalar results in local mirror."""
import hashlib
import json
import subprocess
import zipfile
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;archive=repo/'artifacts/giada_task20_calcium_gate_f94993f_f76a0ffa.zip'
    root=repo/'experiments/results/task20_kaggle_f94993f';root.mkdir(exist_ok=True)
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
    assert report['valid'] and report['task21_authorized'] and not report['fresh_used_for_selection']
    assert all(json.loads((root/n).read_text())['valid'] for n in ('native_audit.json','equivalence_preflight.json'))
    write_json(root/'archive_audit.json',dict(valid=True,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),CRC_valid=True,freeze_checkpoint_source_hashes_valid=True))
    m=ValidatedBatchMirror()
    def put(t,s,**f):return m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task20-{s}-v1',**f})['record_id']
    def ref(t,s):return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task20-{s}-v1'+"'")['rows'][0]['record_id']
    arts=[]
    for path in root.glob('*.json'):
        arts.append(put('artifacts','result-'+path.stem,**{'Nome':'Task20 '+path.name,'Tipo':'Report','Percorso':path.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'}))
    obs=[]
    def record(row,suffix,role):
        rid=put('runs',suffix,**{'Nome':'Task20 '+suffix,'Stato':'Completata','Braccio':[ref('arms',f'{row["channel"]}-w{row["width"]}-{row["objective"]}')],'Blocco':[ref('blocks',role)],'Seed':str(row['seed']),'Configurazione effettiva':f'width{row["width"]};step{row["step"]}','Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(report['environment']),'Validità tecnica':'Oracle, equivalenza, freeze, checkpoint e sorgenti verificati.'})
        for domain,metrics in row['metrics'].items():
            for k,v in metrics.items():
                if k=='finite':continue
                count=(2814 if role=='development' else 5642) if domain=='state_extrema' else (2048 if role=='development' or domain.startswith('ood') else 4096)
                obs.append(put('observations',suffix+'-'+domain+'-'+k,**{'Nome':f'Task20 {suffix} {domain} {k}','Valore':v,'Numerosità':count,'Run':[rid],'Specifica di valutazione':[ref('evaluations',k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':domain,'Descrizione':'Errore gate adimensionale; corrente mA/cm². Tuple, non repliche indipendenti. OOD diagnostico.'}))
        for horizon,metrics in row.get('rollout',{}).items():
            for k,v in metrics.items():
                if k=='finite':continue
                obs.append(put('observations',suffix+'-roll'+horizon+'-'+k,**{'Nome':f'Task20 {suffix} rollout{horizon} {k}','Valore':v,'Numerosità':256,'Run':[rid],'Specifica di valutazione':[ref('evaluations','rollout-'+k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':'held_calcium_'+horizon,'Descrizione':'128cai×stati0/1, dt1ms; calcio imposto costante.'}))
    for row in report['learned']:record(row,f'fresh-{row["channel"]}-{row["objective"]}-s{row["seed"]}','fresh')
    for row in json.loads((root/'development_ladder.json').read_text()):record(row,f'dev-{row["channel"]}-{row["objective"]}-w{row["width"]}-step{row["step"]}-s{row["seed"]}','development')
    findings=[]
    for suffix,title,outcome,result in (
        ('transfer','SK_E2 confermato3/3','Positivo','Calcium-log/rate w16/15k,321parametri: tutti3seed passano; worst gateRMSE .000233993,max .001259843. Task21 autorizzata.'),
        ('input','Calcio necessario, log e lineare entrambi apprendibili','Positivo','Rate-supervised log e lineare passano3/3; soloV fallisce. Log e lineare contengono medesimo calcio; confronto encoding appaiato in paired_input_contrasts. Nessuna superiorità universale log; linear selezionato w32/30k piùbudget/capacità.'),
        ('supervision','Supervisione rate non necessaria in questo contratto','Misto','Calcio log e lineare transition-only passano tutti3seed. Tau1ms è nota e non appresa: i dt lunghi mostrano direttamente zInf. Non trasferire questo esito a gate con tauignota/lenta.'),
        ('paths','Percorsi di calcio imposto confermati3/3','Positivo','Quattro dwell1/5/25/100ms×1000passi: worstRMSE .000334543,max .001260396. Calcio imposto costante dentro ciascun ms, non calcio endogeno né CaDynamics.')):
        findings.append(put('findings',suffix,**{'Nome':title,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Osservazioni':obs,'Risultato':result,'Limitazioni':report['scope']}))
    for k,index,state in (('calcium_sufficiency',1,'Sostiene'),('input_encoding',1,'Indeterminata'),('rate_supervision',2,'Indeterminata'),('capacity_budget',2,'Indeterminata'),('imposed_calcium_path',3,'Sostiene')):
        put('evidence',k,**{'Nome':'Task20 '+k,'Esito':state,'Affermazione valutata':[ref('claims',k)],'Risultati a sostegno':[findings[index]],'Argomentazione':'Contrasti appaiati e dati completi preservati; successo non implica beneficio uniforme di ogni fattore.','Limiti e spiegazioni alternative':report['scope']})
        if state=='Sostiene':put('claims',k,**{'Stato':'Supportata nel dominio'})
    for p in report['imposed_calcium_paths']:
        rid=put('runs','path-s'+str(p['seed']),**{'Nome':'Task20 imposedpath seed'+str(p['seed']),'Stato':'Completata','Braccio':[ref('arms','calcium_log-w16-rate_supervised')],'Blocco':[ref('blocks','fresh')],'Seed':str(p['seed']),'Artefatti prodotti':arts,'Configurazione effettiva':'checkpoint congelato w16/15k; 64cai×stati0/1; dt1ms'})
        for row in p['rows']:
            for k in ('gate_rmse','gate_max_error','occupancy_violations'):
                obs.append(put('observations',f'path-s{p["seed"]}-dwell{row["dwell_ms"]}-{k}',**{'Nome':f'Task20 path{p["seed"]} dwell{row["dwell_ms"]} {k}','Valore':row[k],'Run':[rid],'Specifica di valutazione':[ref('evaluations','rollout-'+k)],'Numerosità':128000,'Artefatti dettagliati':arts,'Strato o sottogruppo':f'imposed_calcium_dwell{row["dwell_ms"]}','Descrizione':'1000timepoints×128tracce; non128000repliche indipendenti.'}))
    put('decisions','to21',**{'Nome':'Task20 autorizza Task21 dinamica lenta','Esito':'Continuare','Risultati':findings,'Motivazione':'Gatecalcio e percorsi imposti entro criteri3/3; fresh dopo freeze.','Condizioni di revisione':'CaDynamics/embedding/speedup non testati.'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':arts,'Validità tecnica':'COMPLETE exit0; auditCRC/hashes/native/equivalence validi; GO.'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,observations=len(obs),findings=len(findings),airtable_accessed=False)))

if __name__=='__main__':main()
