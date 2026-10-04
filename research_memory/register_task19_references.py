"""Register Task19 LUT references and CUDA benchmark as queryable observations."""
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror as StagedMirror

def main():
    m=StagedMirror();r=json.loads((ROOT.parent/'experiments/results/task19_kaggle_9f5d49f/final_report.json').read_text())
    def ref(t,s):return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task19-{s}-v1'+"'")['rows'][0]['record_id']
    def put(t,s,**f):return m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task19-{s}-v1',**f})['record_id']
    arts=[ref('artifacts','result-final_report'),ref('artifacts','result-gpu_benchmark')]
    n=0
    for row in r['numerical']:
        suffix='reference-'+row['channel']+'-'+row['family']+'-'+row['domain']
        size=row['family'].split('_')[2]
        run=put('runs',suffix,**{'Nome':'Task19 '+suffix,'Stato':'Completata','Braccio':[ref('arms','lut'+size)],'Blocco':[ref('blocks','fresh')],'Configurazione effettiva':'LUT f64; griglia V; nessun training; stesse tuplefresh','Artefatti prodotti':arts})
        for k,v in row['metrics'].items():
            if k=='finite':continue
            put('observations',suffix+'-'+k,**{'Nome':'Task19 '+suffix+' '+k,'Valore':v,'Run':[run],'Specifica di valutazione':[ref('evaluations',k)],'Artefatti dettagliati':arts,'Strato o sottogruppo':row['domain'],'Numerosità':5642 if row['domain']=='state_extrema' else 4096,'Descrizione':'LUTf64 accuratezza isolata; non confondere col timingTorchf32.'});n+=1
    metric=put('metrics','cuda_latency',**{'Nome':'Task19 latenza GPU eager','Famiglia':'Regressione','Unità':'ms','Direzione':'Minimizzare','Formula':'Mediana di5misure CUDAevents;100forward/misura;warmup10','Casi degeneri':'Solo batch4096 residenteGPU; no speedupNEURON.'})
    evaluation=put('evaluations','cuda_latency',**{'Nome':'Task19 timingGPU','Metrica':[metric],'Protocollo':[ref('protocols','paired')],'Versione':'v1','Ruolo':'Secondaria','Popolazione e regioni':'4096tuple; stessi outputs gate/inf/tau','Aggregazione e pesi':'Non usato per promozione.'})
    for row in r['gpu_benchmark']['records']:
        suffix='benchmark-'+row['channel']+'-'+row['family']
        run=put('runs',suffix,**{'Nome':'Task19 '+suffix,'Stato':'Completata','Blocco':[ref('blocks','fresh')],'Hardware e ambiente':json.dumps(r['environment']),'Configurazione effettiva':'seed17 selezionato o riferimentoTorch;batch4096;10warmup;5x100forward','Artefatti prodotti':arts})
        for k,ev in [('median_forward_ms',evaluation),('gate_rmse',ref('evaluations','gate_rmse'))]:
            put('observations',suffix+'-'+k,**{'Nome':'Task19 '+suffix+' '+k,'Valore':row[k],'Run':[run],'Specifica di valutazione':[ev],'Numerosità':4096,'Artefatti dettagliati':arts,'Strato o sottogruppo':'eager_CUDA_resident','Descrizione':'Timing e accuratezza congiunti, non endtoendspeedup.'});n+=1
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,reference_observations=n,airtable_accessed=False)))

if __name__=='__main__':main()
