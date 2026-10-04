"""Complete Task20 scalar reference/path links in the next validated batch."""
import json
from .mirror import ROOT


def register(m):
    report=json.loads((ROOT.parent/'experiments/results/task20_kaggle_f94993f/final_report.json').read_text(encoding='utf-8'))
    def ref(t,s):
        return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{t}-giada-roadmap-task20-{s}-v1'+"'")['rows'][0]['record_id']
    def put(t,s,**f):
        return m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task20-{s}-v1',**f})['record_id']
    arts=[ref('artifacts','result-final_report'),ref('artifacts','result-fresh_metrics')]
    for row in report['numerical']:
        suffix='reference-'+row['channel']+'-'+row['family']+'-'+row['domain']
        size=row['family'].split('_')[2]
        run=put('runs',suffix,**{'Nome':'Task20 '+suffix,'Stato':'Completata','Braccio':[ref('arms','lut'+size)],'Blocco':[ref('blocks','fresh')],
            'Configurazione effettiva':'Canonical SK_E2 LUT on logcalcium grid; encoding labels are duplicate physical references, not independent replicas.','Artefatti prodotti':arts})
        for k,v in row['metrics'].items():
            if k=='finite':continue
            put('observations',suffix+'-'+k,**{'Nome':'Task20 '+suffix+' '+k,'Valore':v,'Run':[run],'Specifica di valutazione':[ref('evaluations',k)],
                'Artefatti dettagliati':arts,'Strato o sottogruppo':row['domain'],'Numerosità':5642 if row['domain']=='state_extrema' else 4096,
                'Descrizione':'Canonical numerical reference; no measured CUDA latency, no speedup claim.'})
    observations=[r['record_id'] for r in m.query("SELECT record_id FROM v_records WHERE table_key='observations' AND stable_code LIKE 'observations-giada-roadmap-task20-%'",limit=10000)['rows']]
    put('findings','paths',**{'Osservazioni':observations})
    put('experiments','matrix',**{'Descrizione':'36 independent models:3input encodings×2objectives×2widths×3seeds; one physical SK_E2 gate. CUDA training, no latency benchmark. Imposed-calcium paths.'})
