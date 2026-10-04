"""Task18c verified metrics and Task18d causal preregistration, SQLite only."""
import hashlib,json,subprocess,zipfile
from .mirror import Mirror,ROOT,write_json

def main():
    repo=ROOT.parent;m=Mirror();archive=repo/'artifacts/giada_task18c_potassium_c1a8e0b_e6158d0f.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()=='e16025c95e7395bd5baa0c6b50afd96a9c7d2358e9d96360d7250e4c4a2dd985'
    root=repo/'experiments/results/task18c_kaggle_c1a8e0b'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        f=json.loads(z.read('selection_freeze.json'));h=f.pop('freeze_sha256');assert hashlib.sha256(json.dumps(f,sort_keys=True,separators=(',',':')).encode()).hexdigest()==h
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in f['checkpoint_hashes'].items())
        p=json.loads(z.read('code_provenance.json'));assert not p['dirty_runtime']
        for k,v in p['sources'].items():assert hashlib.sha256(subprocess.check_output(['git','show',p['code_revision']+':'+k],cwd=repo)).hexdigest()==v
        for name in ('final_report.json','native_audit.json','equivalence_preflight.json','selection_freeze.json','development_ladder.json','paired_development_contrasts.json','code_provenance.json','process_status.json','run_contract.json'):(root/name).write_bytes(z.read(name))
    (root/'error_localization.json').write_bytes((repo/'artifacts/task18c_localization.json').read_bytes())
    r=json.loads((root/'final_report.json').read_text());assert r['valid'] and not r['fresh_used_for_selection']
    def put(t,s,**f):return m.local_upsert(t,{'Codice stabile':f'{t}-giada-{s}-v1',**f})['record_id']
    def ref(t,s):return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-{s}-v1'+"'")['rows'][0]['record_id']
    arts=[]
    for path in list(root.iterdir())+[archive]:arts.append(put('artifacts','task18c-result-'+path.stem,**{'Nome':'Task18c '+path.name,'Tipo':'Report','Percorso':path.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'}))
    obs=[]
    for row in r['results']:
        seed=row['seed'];run=put('runs',f'task18c-result-s{seed}',**{'Nome':f'Task18c frozen seed{seed}','Stato':'Completata','Braccio':[ref('arms','task18c-tail_and_kink-top0-lr0.003')],'Blocco':[ref('blocks','task18c-fresh')],'Seed':str(seed),'Artefatti prodotti':arts,'Configurazione effettiva':'w32 step30000 tail_and_kink top0 lr0.003; common development selection','Validità tecnica':'CRC, freeze/checkpoints, clean pinned source hashes valid; scientific partialNO-GO.'})
        for domain,mm in row['metrics'].items():
            for key in ('gate_rmse','gate_max_error','inf_rmse','log_tau_rmse','open_rmse','occupancy_violations'):
                obs.append(put('observations',f'task18c-s{seed}-{domain}-{key}',**{'Nome':f'Task18c seed{seed} {domain} {key}','Valore':mm[key],'Numerosità':2048 if domain=='ood_negative' else 4096,'Run':[run],'Specifica di valutazione':[ref('evaluations','task18c-'+key)],'Artefatti dettagliati':arts,'Strato o sottogruppo':domain,'Descrizione':'Gate/rate adimensionali; OOD solo diagnostico.'}))
        print(f'[SQLite] Task18c seed{seed} registered',flush=True)
    finding=put('findings','task18c-partial',**{'Nome':'Task18c RMSE/coda risolti; massimo locale mTau resta','Esito':'Misto','Esperimenti':[ref('experiments','task18c-matrix')],'Osservazioni':obs,'Risultato':'3/3uniform e coda passano; kink2/3. Seed17 max.010521>.01. Probe tutti m aV−59.931 dt25; tauoracle errore<.000258. Parent appaiato migliora uniform/coda/kink; OOD negativo peggiora e resta diagnostico.','Limitazioni':'Fresh18c ora consumato per diagnosi. Contrasti fattoriali misti; winner non prova ogni intervento utile. No embedded transfer.'})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task18c-matrix-v1','Stato':'Concluso'})
    m.local_upsert('runs',{'Codice stabile':'runs-giada-task18c-kaggle-orchestration-v1','Stato':'Completata','Artefatti prodotti':arts,'Validità tecnica':'Exit0; audit/freeze/hash validi; scientific partialNO-GO.'})
    cfgpath=repo/'experiments/task18d_potassium_architecture.json';cfg=json.loads(cfgpath.read_text());claims=[]
    for k,text in cfg['hypotheses'].items():claims.append(put('claims','task18d-'+k,**{'Nome':'Task18d '+k,'Enunciato':text,'Tipo':'Ipotesi','Stato':'Aperta','Limiti':'K_Pst held-voltage; treseed; contrasti appaiati e fresh nuovo.'}))
    ex=put('experiments','task18d-matrix',**{'Nome':'GIADA Task18d local K_Pst repair','Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':claims,'Descrizione':'Smooth vs cusp feature vs branch heads;9models; frozen-parent initialization; Adam reset matched.','Obiettivo informativo':'Distinguere struttura della curva da ulteriore ottimizzazione; controllo estremi stato.'})
    pr=put('protocols','task18d-paired',**{'Nome':'Task18d paired factorial','Esperimento':[ex],'Versione':'v1','Modalità':'Componente isolato','Procedura':json.dumps(cfg),'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'Regole di arresto':'Non promuovereTask19 né rilassare soglie se freshfallisce.'})
    armids=[]
    for s in cfg['sampling']:
        for k in cfg['topk_weights']:
            for lr in cfg['learning_rates']:armids.append(put('arms',f'task18d-{s}-top{k}-lr{lr}',**{'Nome':f'Task18d {s} top{k} lr{lr}','Ruolo':'Trattamento','Protocollo':[pr],'Descrizione':cfg['initialization'],'Configurazione residua':cfg['loss']}))
    ev=[]
    for k,v in cfg['gates'].items():
        metric=put('metrics','task18d-'+k,**{'Nome':'Task18d '+k,'Famiglia':'Regressione' if k!='occupancy_violations' else 'Fisica','Unità':'adimensionale','Direzione':'Minimizzare','Formula':'hh_family_transfer.measurements '+k})
        ev.append(put('evaluations','task18d-'+k,**{'Nome':'Task18d '+k,'Metrica':[metric],'Protocollo':[pr],'Versione':'v1','Ruolo':'Primaria','Target':str(v),'Popolazione e regioni':'Uniform,coda,kink separati; tutti3seed','Aggregazione e pesi':'Coniuntiva; OODdiagnostico'}))
    for i,(k,text) in enumerate(cfg['hypotheses'].items()):
        pred=put('predictions','task18d-'+k,**{'Nome':'Task18d '+k,'Origine':'Preregistrata','Ipotesi':[claims[i]],'Risultato atteso':text,'Specifica di valutazione':[ev[0]]})
        put('contrasts','task18d-'+k,**{'Nome':'Task18d '+k,'Bracci':armids,'Predizioni':[pred],'Specifiche di valutazione':ev,'Contrasto e coefficienti':text,'Confondenti controllati':'Pesi, seed, tuplecount, stati/dt, minibatch, Adam reset identici; singolo fattore varia per contrasto.','Soglia interpretativa':'Delta matched; promozionefreshoriginalthresholds; nessun pvalue.'})
    for role in ('fit','development','fresh'):put('blocks','task18d-'+role,**{'Nome':'Task18d '+role,'Protocollo':[pr],'Seed':json.dumps(cfg['data_seeds']),'Descrizione':'Tuple nuove; test prima congelato; strati separati.'})
    put('decisions','task18c-to18c',**{'Nome':'18d prima di19','Esito':'Modificare','Risultati':[finding],'Motivazione':'Errore mTau locale; confronto strutturale e griglia stati estremi.','Condizioni di revisione':cfg['promotion'],'Data':'2026-10-04'})
    put('artifacts','task18d-prereg',**{'Nome':'Task18d prereg','Tipo':'Report','Percorso':cfgpath.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'Versione':'v1'})
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid'];print(json.dumps({'valid':True,'observations':len(obs),'arms':len(armids),'airtable_accessed':False}))

if __name__=='__main__':main()
