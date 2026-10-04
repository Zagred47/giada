"""Complete the queryable Task21 numerical reference measurements."""
import json
from .mirror import ROOT


def register(m):
    report=json.loads((ROOT.parent/'experiments/results/task21_kaggle_ea1f758/final_report.json').read_text(encoding='utf-8'))
    cache={}
    def ref(t,s):
        if (t,s) in cache:return cache[t,s]
        return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task21-{s}-v1'+"'")['rows'][0]['record_id']
    def put(t,s,**f):
        rid=m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task21-{s}-v1',**f})['record_id'];cache[t,s]=rid;return rid
    arts=[ref('artifacts','result-final_report'),ref('artifacts','result-fresh_metrics')];obs=[]
    for row in report['numerical']:
        suffix='reference-'+row['channel']+'-'+row['family']+'-'+row['domain'];size=row['family'].split('_')[2]
        run=put('runs',suffix,**{'Nome':'Task21 '+suffix,'Stato':'Completata','Braccio':[ref('arms','lut'+size)],'Blocco':[ref('blocks','fresh')],
            'Configurazione effettiva':'Canonical Nap_h LUT voltage grid; support labels duplicate physical reference, not independent replicas.','Artefatti prodotti':arts})
        for k,v in row['metrics'].items():
            if k=='finite':continue
            obs.append(put('observations',suffix+'-'+k,**{'Nome':'Task21 '+suffix+' '+k,'Valore':v,'Run':[run],'Specifica di valutazione':[ref('evaluations',k)],
                'Artefatti dettagliati':arts,'Strato o sottogruppo':row['domain'],'Numerosità':8932 if row['domain']=='state_extrema' else 4096,'Descrizione':'No CUDA latency or speedup claim.'}))
    finding=put('findings','numerical',**{'Nome':'Task21 LUT baseline accurata','Esito':'Positivo','Esperimenti':[ref('experiments','matrix')],'Osservazioni':obs,'Risultato':'Both LUT513/2049 pass all in-support criteria. Reference labels repeat the same physical approximation; no independent replication claim.'})
    put('evidence','numerical_reference',**{'Esito':'Sostiene','Risultati a sostegno':[finding]})
