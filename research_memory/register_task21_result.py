"""Verify Task21 archive and register complete scalar results in local mirror."""
import hashlib
import json
import subprocess
import zipfile
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;archive=repo/'artifacts/giada_task21_slow_gate_ea1f758_a86f3094.zip'
    root=repo/'experiments/results/task21_kaggle_ea1f758';root.mkdir(exist_ok=True)
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
    assert report['valid'] and report['task22_authorized'] and not report['fresh_used_for_selection']
    assert all(json.loads((root/n).read_text())['valid'] for n in ('native_audit.json','equivalence_preflight.json'))
    write_json(root/'archive_audit.json',dict(valid=True,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),CRC_valid=True,freeze_checkpoint_source_hashes_valid=True))
    m=ValidatedBatchMirror()
    cache={}
    def put(t,s,**f):
        rid=m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task21-{s}-v1',**f})['record_id'];cache[t,s]=rid;return rid
    def ref(t,s):
        if (t,s) in cache:return cache[t,s]
        return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task21-{s}-v1'+"'")['rows'][0]['record_id']
    arts=[]
    for path in root.glob('*.json'):
        arts.append(put('artifacts','result-'+path.stem,**{'Nome':'Task21 '+path.name,'Tipo':'Report','Percorso':path.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'}))
    for k in ('persistence_rmse','formula_f32_rmse','learned_single_macro_update_rmse'):
        metric=put('metrics',k,**{'Nome':'Task21 '+k,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'adimensionale'})
        put('evaluations','rollout-'+k,**{'Nome':'Task21 rollout '+k,'Metrica':[metric],'Protocollo':[ref('protocols','paired')],'Versione':'v1','Ruolo':'Secondaria','Aggregazione e pesi':'Diagnostic only, never selects frozen model.'})
    obs=[]
    def record(row,suffix,role):
        rid=put('runs',suffix,**{'Nome':'Task21 '+suffix,'Stato':'Completata','Braccio':[ref('arms',f'{row["channel"]}-w{row["width"]}-{row["objective"]}')],'Blocco':[ref('blocks',role)],'Seed':str(row['seed']),'Configurazione effettiva':f'width{row["width"]};step{row["step"]}','Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(report['environment']),'Validità tecnica':'Oracle, equivalenza, freeze, checkpoint e sorgenti verificati.'})
        for domain,metrics in row['metrics'].items():
            for k,v in metrics.items():
                if k=='finite':continue
                count=(4422 if role=='development' else 8932) if domain=='state_extrema' else (2048 if role=='development' or domain.startswith('ood') else 4096)
                obs.append(put('observations',suffix+'-'+domain+'-'+k,**{'Nome':f'Task21 {suffix} {domain} {k}','Valore':v,'Numerosità':count,'Run':[rid],'Specifica di valutazione':[ref('evaluations',k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':domain,'Descrizione':'Errore h adimensionale; nessuna corrente testata. Tuple, non repliche indipendenti. OOD diagnostico.'}))
        for horizon,metrics in row.get('rollout',{}).items():
            for k,v in metrics.items():
                if k=='finite':continue
                obs.append(put('observations',suffix+'-roll'+horizon+'-'+k,**{'Nome':f'Task21 {suffix} rollout{horizon} {k}','Valore':v,'Numerosità':256,'Run':[rid],'Specifica di valutazione':[ref('evaluations','rollout-'+k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':'held_voltage_'+horizon,'Descrizione':'128V×stati0/1, dt1ms; voltaggio imposto costante.'}))
    for row in report['learned']:record(row,f'fresh-{row["channel"]}-{row["objective"]}-s{row["seed"]}','fresh')
    for row in json.loads((root/'development_ladder.json').read_text()):record(row,f'dev-{row["channel"]}-{row["objective"]}-w{row["width"]}-step{row["step"]}-s{row["seed"]}','development')
    findings=[]
    for suffix,title,outcome,result in (
        ('transfer','Gate lento confermato3/3','Positivo','Multiscale/rate3/3; worst primary-domain RMSE0.000776376, worst repeated1ms rolloutRMSE0.001448890 through10000ms. h-only, not entire Nap.'),
        ('duration','Supporto temporale multiscala decisivo','Positivo','Short/transition0/3, multiscale/transition3/3. Short/rate2/3, multiscale/rate3/3. Contrasti appaiati a width/budget fissi in paired_duration_contrasts.'),
        ('rates','Supervisione rate non universalmente superiore','Misto','Multiscale/transition passa3/3 con rollout10s RMSE0.000226-0.000499; multiscale/rate0.000738-0.001175. Selezioni width/budget diverse: non attribuire la differenza solo alla loss.'),
        ('floor','Errore residuo non spiegato dal solo float32','Positivo','Formula iterata float32 RMSE10s2.92665e-5, molto sotto errore learned. Macro learned e repeated learned simili. Persistence10s RMSE0.66952.'),
        ('scaling','Capacita/budget conservati per interrogazione','Inconcludente','Ladder completo e contrasti appaiati preservati; non inferire beneficio monotono da selezioni finali.')):
        findings.append(put('findings',suffix,**{'Nome':title,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Osservazioni':obs,'Risultato':result,'Limitazioni':report['scope']}))
    for k,index,state in (('slow_gate_transfer',0,'Sostiene'),('duration_identifiability',1,'Sostiene'),('rate_identifiability',2,'Indeterminata'),('capacity_budget',4,'Indeterminata'),('state_rollout',3,'Sostiene'),('numerical_reference',0,'Indeterminata')):
        put('evidence',k,**{'Nome':'Task21 '+k,'Esito':state,'Affermazione valutata':[ref('claims',k)],'Risultati a sostegno':[findings[index]],'Argomentazione':'Contrasti controllati; nessuna generalizzazione a dinamica endogena o intero canale.','Limiti e spiegazioni alternative':report['scope']})
        if state=='Sostiene':put('claims',k,**{'Stato':'Supportata nel dominio'})
    put('decisions','to22',**{'Nome':'Task21 autorizza Task22 condivisione controllata','Esito':'Continuare','Risultati':findings,'Motivazione':'Primario multiscale/rate entro tutte soglie3/3.','Condizioni di revisione':'Modelli ancora indipendenti; sharing da verificare.'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':arts,'Validità tecnica':'COMPLETE exit0; auditCRC/hashes/native/equivalence validi; GO.'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,observations=len(obs),findings=len(findings),airtable_accessed=False)))

if __name__=='__main__':main()
