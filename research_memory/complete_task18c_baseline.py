"""Preserve the matched frozen-parent scalar baseline and verify 18d protocol."""
import hashlib,json
from .mirror import Mirror,ROOT,write_json
def main():
    m=Mirror();r=json.loads((ROOT.parent/'experiments/results/task18c_kaggle_c1a8e0b/final_report.json').read_text())
    def ref(t,s):return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-{s}-v1'+"'")['rows'][0]['record_id']
    def put(t,s,**f):return m.local_upsert(t,{'Codice stabile':f'{t}-giada-{s}-v1',**f})['record_id']
    obs=[]
    for n,seed in enumerate((17,29,43)):
        put('runs',f'task18c-result-s{seed}',**{'Configurazione effettiva':'w32 step30000 tail_and_kink top0 lr0.003; common development selection'})
        run=put('runs',f'task18c-parent-s{seed}',**{'Nome':f'Task18c frozen18b parent seed{seed}','Stato':'Completata','Braccio':[ref('arms','task18b-w32-tail_enriched-tau0.1')],'Blocco':[ref('blocks','task18c-fresh')],'Seed':str(seed),'Configurazione effettiva':'Frozen18b width32step60000; evaluated on same Task18c fresh after freeze.','Validità tecnica':'Parent checkpoint SHA checked; matched fresh baseline.'})
        for domain,values in r['parent_same_fresh'].items():
            for key in ('gate_rmse','gate_max_error','inf_rmse','log_tau_rmse','open_rmse','occupancy_violations'):
                obs.append(put('observations',f'task18c-parent-s{seed}-{domain}-{key}',**{'Nome':f'Task18c parent seed{seed} {domain} {key}','Valore':values[n][key],'Numerosità':2048 if domain=='ood_negative' else 4096,'Run':[run],'Specifica di valutazione':[ref('evaluations','task18c-'+key)],'Strato o sottogruppo':domain,'Descrizione':'Matched frozen parent; fresh never used for selection.'}))
        print(f'[SQLite] Task18c matched parent seed{seed}',flush=True)
    put('findings','task18c-baseline',**{'Nome':'Task18c matched parent comparison','Esito':'Misto','Esperimenti':[ref('experiments','task18c-matrix')],'Osservazioni':obs,'Risultato':'Refit improves mean and max in uniform/tail/kink in all3seeds. Negative extrapolation RMSE worsens in all3seeds; not a universal improvement.'})
    put('findings','task18c-partial',**{'Limitazioni':'Fresh18c consumed for diagnosis; three seeds and finite samples. Parent quantified separately; OOD negative worsens. No embedded substitution.'})
    put('decisions','task18c-to18c',**{'Nome':'Task18d prima di19'})
    cfg=ROOT.parent/'experiments/task18d_potassium_architecture.json';c=json.loads(cfg.read_text());put('protocols','task18d-paired',**{'Procedura':json.dumps(c),'Hash preregistrazione':hashlib.sha256(cfg.read_bytes()).hexdigest()})
    put('artifacts','task18d-prereg',**{'SHA-256':hashlib.sha256(cfg.read_bytes()).hexdigest()})
    for key in c['gates']:put('evaluations','task18d-'+key,**{'Popolazione e regioni':'Uniform,coda,kink e griglia state_extrema separati; tutti3seed'})
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid'];print(json.dumps({'valid':True,'matched_parent_observations':len(obs)}))
if __name__=='__main__':main()
