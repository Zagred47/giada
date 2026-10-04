"""Generate verified Task21 result registrar from the established template."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'research_memory/register_task20_result.py').read_text(encoding='utf-8').replace('Task20','Task21').replace('task20','task21').replace('calcium_gate_f94993f_f76a0ffa','slow_gate_ea1f758_a86f3094').replace('kaggle_f94993f','kaggle_ea1f758').replace('task21_authorized','task22_authorized')
s=s.replace('(2814 if role==\'development\' else 5642)','(4422 if role==\'development\' else 8932)')
s=s.replace('Errore gate adimensionale; corrente mA/cm².','Errore h adimensionale; nessuna corrente testata.').replace('held_calcium_','held_voltage_').replace('128cai×stati0/1','128V×stati0/1').replace('calcio imposto costante','voltaggio imposto costante')
s=s.replace("    findings=[]", "    findings=[]")
s=s.replace("    def put(t,s,**f):return m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task21-{s}-v1',**f})['record_id']", "    cache={}\n    def put(t,s,**f):\n        rid=m.local_upsert(t,{'Codice stabile':f'{t}-giada-roadmap-task21-{s}-v1',**f})['record_id'];cache[t,s]=rid;return rid")
s=s.replace('    def ref(t,s):return m.query(', '    def ref(t,s):\n        if (t,s) in cache:return cache[t,s]\n        return m.query(')
a=s.index('    findings=[]');b=s.index("    put('experiments','matrix'",a)
s=s[:a]+'''    findings=[]
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
''' +s[b:]
# Extra floor metrics need their own secondary specifications, not primary gates.
a=s.index('    obs=[]')
s=s[:a]+'''    for k in ('persistence_rmse','formula_f32_rmse','learned_single_macro_update_rmse'):
        metric=put('metrics',k,**{'Nome':'Task21 '+k,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'adimensionale'})
        put('evaluations','rollout-'+k,**{'Nome':'Task21 rollout '+k,'Metrica':[metric],'Protocollo':[ref('protocols','paired')],'Versione':'v1','Ruolo':'Secondaria','Aggregazione e pesi':'Diagnostic only, never selects frozen model.'})
''' +s[a:]
(R/'research_memory/register_task21_result.py').write_text(s,encoding='utf-8')
print('Task21 registrar generated')
