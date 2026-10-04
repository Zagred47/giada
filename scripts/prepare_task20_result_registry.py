"""Mechanically reuse verified result-registration structure, with SK_E2 findings."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'research_memory/register_task19_result.py').read_text(encoding='utf-8')
source=source.replace('Task19','Task20').replace('task19','task20').replace('task20_single_gate_9f5d49f_37c01f4c','task20_calcium_gate_f94993f_f76a0ffa').replace('task20_kaggle_9f5d49f','task20_kaggle_f94993f').replace("report['task20_authorized']","report['task21_authorized']")
start=source.index('    for suffix,title,outcome,result in (')
end=source.index("        findings.append(",start)
source=source[:start]+'''    for suffix,title,outcome,result in (
        ('transfer','SK_E2 confermato3/3','Positivo','Calcium-log/rate w16/15k,321parametri: tutti3seed passano; worst gateRMSE .000233993,max .001259843. Task21 autorizzata.'),
        ('input','Calcio necessario, log e lineare entrambi apprendibili','Positivo','Rate-supervised log e lineare passano3/3; soloV fallisce. Log e lineare contengono medesimo calcio; confronto encoding appaiato in paired_input_contrasts. Nessuna superiorità universale log; linear selezionato w32/30k piùbudget/capacità.'),
        ('supervision','Supervisione rate non necessaria in questo contratto','Misto','Calcio log e lineare transition-only passano tutti3seed. Tau1ms è nota e non appresa: i dt lunghi mostrano direttamente zInf. Non trasferire questo esito a gate con tauignota/lenta.'),
        ('paths','Percorsi di calcio imposto confermati3/3','Positivo','Quattro dwell1/5/25/100ms×1000passi: worstRMSE .000334543,max .001260396. Calcio imposto costante dentro ciascun ms, non calcio endogeno né CaDynamics.')):
''' + source[end:]
start=source.index("    for k in (")
end=source.index("    put('experiments','matrix'",start)
source=source[:start]+'''    for k,index,state in (('calcium_sufficiency',1,'Sostiene'),('input_encoding',1,'Indeterminata'),('rate_supervision',2,'Indeterminata'),('capacity_budget',2,'Indeterminata'),('imposed_calcium_path',3,'Sostiene')):
        put('evidence',k,**{'Nome':'Task20 '+k,'Esito':state,'Affermazione valutata':[ref('claims',k)],'Risultati a sostegno':[findings[index]],'Argomentazione':'Contrasti appaiati e dati completi preservati; successo non implica beneficio uniforme di ogni fattore.','Limiti e spiegazioni alternative':report['scope']})
        if state=='Sostiene':put('claims',k,**{'Stato':'Supportata nel dominio'})
    for p in report['imposed_calcium_paths']:
        rid=put('runs','path-s'+str(p['seed']),**{'Nome':'Task20 imposedpath seed'+str(p['seed']),'Stato':'Completata','Braccio':[ref('arms','calcium_log-w16-rate_supervised')],'Blocco':[ref('blocks','fresh')],'Seed':str(p['seed']),'Artefatti prodotti':arts,'Configurazione effettiva':'checkpoint congelato w16/15k; 64cai×stati0/1; dt1ms'})
        for row in p['rows']:
            for k in ('gate_rmse','gate_max_error','occupancy_violations'):
                obs.append(put('observations',f'path-s{p["seed"]}-dwell{row["dwell_ms"]}-{k}',**{'Nome':f'Task20 path{p["seed"]} dwell{row["dwell_ms"]} {k}','Valore':row[k],'Run':[rid],'Specifica di valutazione':[ref('evaluations','rollout-'+k)],'Numerosità':128000,'Artefatti dettagliati':arts,'Strato o sottogruppo':f'imposed_calcium_dwell{row["dwell_ms"]}','Descrizione':'1000timepoints×128tracce; non128000repliche indipendenti.'}))
    put('decisions','to21',**{'Nome':'Task20 autorizza Task21 dinamica lenta','Esito':'Continuare','Risultati':findings,'Motivazione':'Gatecalcio e percorsi imposti entro criteri3/3; fresh dopo freeze.','Condizioni di revisione':'CaDynamics/embedding/speedup non testati.'})
''' + source[end:]
source=source.replace('held_voltage_','held_calcium_').replace('128V×stati0/1, dt1ms; non voltaggio variabile.','128cai×stati0/1, dt1ms; calcio imposto costante.')
(root/'research_memory/register_task20_result.py').write_text(source,encoding='utf-8')
print('Task20 result registry generated')
