"""Verify preserved Task22 outputs and register scalar development/fresh probes."""
import hashlib
import json
import subprocess
import zipfile
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent;archive=repo/'artifacts/giada_task22_controlled_sharing_eb494b8_9c3900fb.zip'
    root=repo/'experiments/results/task22_kaggle_eb494b8';root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze=json.loads(z.read('selection_freeze.json'));claimed=freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==claimed
        assert not freeze['fresh_accessed']
        for path,h in freeze['checkpoint_hashes'].items():assert hashlib.sha256(z.read(path)).hexdigest()==h
        provenance=json.loads(z.read('code_provenance.json'));assert not provenance['dirty_runtime']
        for path,h in provenance['sources'].items():assert hashlib.sha256(subprocess.check_output(['git','show',provenance['code_revision']+':'+path],cwd=repo)).hexdigest()==h
        for name in z.namelist():
            if '/' not in name and name.endswith('.json'):(root/name).write_bytes(z.read(name))
    report=json.loads((root/'final_report.json').read_text())
    assert report['valid'] and report['task23_authorized'] and not report['fresh_used_for_selection']
    assert json.loads((root/'process_status.json').read_text())['returncode']==0
    assert all(json.loads((root/n).read_text())['valid'] for n in ('native_audit.json','equivalence_preflight.json'))
    write_json(root/'archive_audit.json',dict(valid=True,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),CRC_valid=True,freeze_checkpoint_source_hashes_valid=True))
    m=ValidatedBatchMirror();cache={}
    def put(t,s,**f):
        rid=m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task22-{s}-v1',**f})['record_id'];cache[t,s]=rid;return rid
    def ref(t,s):
        if (t,s) in cache:return cache[t,s]
        rid=m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task22-{s}-v1'+"'")['rows'][0]['record_id']
        cache[t,s]=rid
        return rid
    arts=[]
    for p in root.glob('*.json'):
        arts.append(put('artifacts','result-'+p.stem,**{'Nome':'Task22 '+p.name,'Tipo':'Report','Percorso':p.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'}))
    obs=[]
    for role,rows in [('fresh',report['learned']),('development',json.loads((root/'development_ladder.json').read_text()))]:
        for r in rows:
            suffix=f'{role}-{r["family"]}-w{r["width"]}-step{r["step"]}-s{r["seed"]}'
            run=put('runs',suffix,**{'Nome':'Task22 '+suffix,'Stato':'Completata','Braccio':[ref('arms',r['family']+'-w'+str(r['width'])+'-rate_supervised')],'Blocco':[ref('blocks',role)],'Seed':str(r['seed']),'Artefatti prodotti':arts,'Configurazione effettiva':json.dumps({k:r[k] for k in ('family','width','step','seed')}),'Validità tecnica':'Freeze, sorgenti, checkpoint, oracle ed equivalenza verificati.'})
            groups=[(d,c,metrics,False) for d,channels in r['metrics'].items() for c,metrics in channels.items()]
            groups += [('rollout'+h,c,metrics,True) for c,hh in r.get('rollout',{}).items() for h,metrics in hh.items()]
            groups += [('swapped_identity',c,metrics,False) for c,metrics in r.get('swapped_identity',{}).items()]
            for domain,channel,metrics,roll in groups:
                for k,value in metrics.items():
                    if k=='finite':continue
                    obs.append(put('observations',suffix+'-'+domain+'-'+channel+'-'+k,**{'Nome':f'Task22 {suffix} {domain} {channel} {k}','Valore':value,'Run':[run],'Specifica di valutazione':[ref('evaluations',('rollout-' if roll else '')+k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':channel+'/'+domain,'Descrizione':'Gate isolato aVimposto; OOD e identità scambiata diagnostici. Confronti width allineati, parametri differenti.'}))
    findings=[]
    for suffix,title,outcome,result in [
        ('independent','Tre gate indipendenti confermati','Positivo','Independent passa3/3 su tutti domini/rollout richiesti;3558parametri perbundle.'),
        ('sharing','Condivisione non confermata al budget registrato','Negativo','Shared_heads1318 e conditioned1282parametri non passano il criterio congiuntivo. Non prova impossibilita di sharing in generale.'),
        ('diagnostics','Scaling e identita conservati per analisi','Inconcludente','Ladder e probe swapped dettagliati preservati; nessuna diagnosi automatica di capacity o gradient conflict.')]:
        findings.append(put('findings',suffix,**{'Nome':'Task22 '+title,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Osservazioni':obs,'Risultato':result,'Limitazioni':report['scope']}))
    put('evidence','sharing_learnability',**{'Nome':'Task22 sharing nel contratto testato','Esito':'Contraddice','Affermazione valutata':[ref('claims','sharing_learnability')],'Risultati a sostegno':[findings[1]],'Argomentazione':'Entrambe le famiglie condivise falliscono il gate congiuntivo; riferimento indipendente valido3/3. Confondenti di capacita totale dichiarati.'})
    put('claims','sharing_learnability',**{'Stato':'Contraddetta nel dominio'})
    put('decisions','to23',**{'Nome':'Task22 permette preparazione Task23 CaHVA+CaLVA','Esito':'Continuare','Risultati':findings,'Motivazione':'Independent reference confermato; composizione con baseline indipendente, sharing ancora sperimentale.','Condizioni di revisione':'CaLVA doppiooracle e gate isolati obbligatori; dipendenze sezioneIV per componenti aggiunte.'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':arts,'Validità tecnica':'Version2 COMPLETE exit0; reportvalido; indipendentiGO, sharingNO-GO nelbudget.'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,observations=len(obs),task23_preparation_authorized=True,airtable_accessed=False)))


if __name__=='__main__':main()
