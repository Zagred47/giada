"""Verified Task18 result, complete fresh scalar metrics; only local SQLite."""
import hashlib,json,subprocess,zipfile
from .mirror import Mirror,ROOT,write_json

def main():
    mirror=Mirror();repo=ROOT.parent;archive=repo/'artifacts/giada_task18_hh_transfer_a094fa9_ea87f7a2.zip'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest=='1f6de2183e5d2a458de2efc77871427778ba3ed8ebc0da71e2f896593c4428ba'
    root=repo/'experiments/results/task18_kaggle_a094fa9'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in ('final_report.json','native_audit.json','equivalence_preflight.json','selection_freeze.json','development_ladder.json','fresh_metrics.json','code_provenance.json','process_status.json','run_contract.json'):
            (root/name).write_bytes(z.read(name))
        freeze=json.loads(z.read('selection_freeze.json'));claimed=freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==claimed
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in freeze['checkpoints'].items())
        provenance=json.loads(z.read('code_provenance.json'))
        assert not provenance['dirty_runtime']
        for path,h in provenance['sources'].items():
            raw=subprocess.check_output(['git','-C',str(repo),'show',provenance['code_revision']+':'+path]);assert hashlib.sha256(raw).hexdigest()==h
    report=json.loads((root/'final_report.json').read_text())
    assert report['valid'] and not report['fresh_used_for_selection']
    assert json.loads((root/'native_audit.json').read_text())['valid']
    assert json.loads((root/'equivalence_preflight.json').read_text())['valid']
    write_json(root/'archive_audit.json',{'valid':True,'archive_sha256':digest,'archive_size_bytes':archive.stat().st_size,'CRC_valid':True,'source_hashes_valid':True,'freeze_and_checkpoint_hashes_valid':True})
    def put(t,s,**f):return mirror.local_upsert(t,{'Codice stabile':f'{t}-giada-task18-result-{s}-v1',**f})['record_id']
    def ref(t,s):
        r=mirror.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task18-{s}-v1'+"'")['rows'];assert len(r)==1;return r[0]['record_id']
    arts=[]
    for p in root.iterdir():
        arts.append(put('artifacts',p.stem,**{'Nome':'Task18 risultato — '+p.name,'Tipo':'Report','Percorso':p.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'}))
    arts.append(put('artifacts','archive',**{'Nome':'Task18 archivio completo','Tipo':'Report','Percorso':archive.relative_to(repo).as_posix(),'SHA-256':digest,'Dimensione byte':archive.stat().st_size,'Versione':'v1'}))
    obs=[]
    for i,r in enumerate(report['learned']):
        suffix=f"{r['channel']}-{r['objective']}-s{r['seed']}"
        run=put('runs',suffix,**{'Nome':'Task18 — '+suffix,'Stato':'Completata','Braccio':[ref('arms',f"{r['channel']}-w{r['width']}-{r['objective']}")],'Blocco':[ref('blocks','fresh')],'Seed':str(r['seed']),'Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(report['environment']),'Configurazione effettiva':f"width={r['width']};step={r['step']}; selection development only",'Validità tecnica':'Audit, CRC, codice, freeze e checkpoint validi; test mai usato per selezione.'})
        for domain,mm in r['metrics'].items():
            for key in ('gate_rmse','gate_max_error','inf_rmse','log_tau_rmse','open_rmse','occupancy_violations'):
                obs.append(put('observations',suffix+'-'+domain+'-'+key,**{'Nome':'Task18 — '+suffix+' '+domain+' '+key,'Valore':mm[key],'Numerosità':4096 if domain=='in_support' else 2048,'Run':[run],'Specifica di valutazione':[ref('evaluations',key)],'Artefatti dettagliati':arts,'Strato o sottogruppo':domain,'Descrizione':'Occupazioni adimensionali, non mV. Tutti i valori originali in final_report; OOD diagnostico separato.'}))
        print(f'[SQLite Task18] modelli {i+1}/18',flush=True)
    finding=put('findings','transfer',**{'Nome':'Task18 — Ca/Na positivi; K_Pst non promosso','Esito':'Misto','Esperimenti':[ref('experiments','matrix')],'Osservazioni':obs,'Risultato':'Rate-supervised Ca_HVA e NaTa_t passano3/3; K_Pst0/3: RMSE0.001192–0.001446, massimo0.01521–0.02875. LUT2049 passa per tutti. Training114.39s.','Incertezza':'Tre seed; width16/32; massimo15k passi. Nessuna prova di impossibilità architetturale.','Limitazioni':report['scope']})
    ev=[]
    for key,state,outcome in [('transfer','Contraddetta nel dominio','Contraddice'),('numerical_reference','Supportata nel dominio','Sostiene'),('identifiability','Indeterminata','Indeterminata'),('capacity_budget','Indeterminata','Indeterminata')]:
        mirror.local_upsert('claims',{'Codice stabile':f'claims-giada-roadmap-task18-{key}-v1','Stato':state})
        ev.append(put('evidence',key,**{'Nome':'Task18 evidenza — '+key,'Esito':outcome,'Affermazione valutata':[ref('claims',key)],'Risultati a sostegno':[finding],'Argomentazione':'Promozione congiuntiva fallita soloK; riferimento numerico positivo. Cinetiche migliorano in vari contrasti, ma configurazioni selezionate possono differire e ladder non stabilisce causalmente un limite assoluto.','Limiti e spiegazioni alternative':'Budget, coda di tensione, peso loss e capacità da separare; nessun peso condiviso.','Data valutazione':'2026-10-04'}))
    put('decisions','18b',**{'Nome':'Task18 — diagnosi18b prima diTask19','Esito':'Modificare','Risultati':[finding],'Valutazioni delle evidenze':ev,'Motivazione':'Il test è valido, il potassio non passa; preservare successi Ca/Na. Non rilassare soglie; nuove tuplefresh dopo nuovofreeze.','Condizioni di revisione':'Tutti i seed K superano soglie originali con conferma dedicata della coda interna.','Data':'2026-10-04'})
    mirror.local_upsert('runs',{'Codice stabile':'runs-giada-roadmap-task18-kaggle-orchestration-v1','Stato':'Completata','Artefatti prodotti':arts,'Validità tecnica':'Exit0; CRC, provenance, oracle, vectorization, freeze and hashes valid. Scientific partial NO-GO, not execution failure.'})
    mirror.local_upsert('experiments',{'Codice stabile':'experiments-giada-roadmap-task18-matrix-v1','Stato':'Concluso'})
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'observations':len(obs),'airtable_accessed':False}))

if __name__=='__main__':main()
