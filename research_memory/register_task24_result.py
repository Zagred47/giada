"""Register every verified Task24 scalar, retaining the scientific NO-GO."""
import hashlib
import json
import statistics
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def scalars(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items(): yield from scalars(item, path+(key,))
    elif isinstance(value, list):
        for i, item in enumerate(value): yield from scalars(item, path+(str(i),))
    elif isinstance(value, (int, float, bool)): yield path, float(value)


def main():
    root = ROOT.parent/'experiments/results/task24_kaggle_4fad2c6'
    report = json.loads((root/'final_report.json').read_text())
    audit = json.loads((root/'result_audit.json').read_text())
    assert audit['valid'] and audit['checkpoint_and_source_hashes_valid']
    assert report['valid'] and not report['task25_preparation_authorized']
    cfg = json.loads((root/'run_contract.json').read_text())
    m = ValidatedBatchMirror(); cache = {}; observations = []; subsets = {}
    def put(table, key, **fields):
        rid=m.local_upsert(table, {'Codice stabile':f'{table}-giada-roadmap-task24-{key}-v1', **fields})['record_id']
        cache[table,key]=rid
        return rid
    def ref(table,key):
        if (table,key) not in cache:
            cache[table,key]=m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{table}-giada-roadmap-task24-{key}-v1'+"'")['rows'][0]['record_id']
        return cache[table,key]
    artifacts={}
    for path in root.glob('*.json'):
        artifacts[path.name]=put('artifacts','result-'+path.stem,**{'Nome':'Task24 '+path.name,'Tipo':'Report','Percorso':path.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'})
    def evaluation(path, group):
        key=path[-1]
        if group=='path_gates' and 'per_channel' in path: group='path_channel_gates'
        if group in ('gates','held_gates') and 'pair' in path: group='pair_gates'
        if group and key in cfg[group]:return ref('evaluations',group+'-'+key)
        code='diagnostic-'+key
        if ('evaluations',code) not in cache:
            metric=put('metrics',code,**{'Nome':'Task24 '+key,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'mA/cm2' if 'ma_cm2' in key else 'adimensionale','Formula':'Verified scalar probe, diagnostic unless explicitly preregistered.'})
            put('evaluations',code,**{'Nome':'Task24 '+key,'Metrica':[metric],'Protocollo':[ref('protocols','paired')],'Versione':'v1','Ruolo':'Secondaria','Aggregazione e pesi':'Preserve family/seed/width/budget/domain/channel strata.'})
        return cache['evaluations',code]
    def record(run,suffix,value,artifact,family='',group=None):
        for path, scalar in scalars(value):
            rid=put('observations',suffix+'-'+'-'.join(path),**{'Nome':'Task24 '+suffix+' '+ '/'.join(path),'Valore':scalar,'Run':[run],'Specifica di valutazione':[evaluation(path,group)],'Artefatti dettagliati':[artifacts[artifact]],'Strato o sottogruppo':'/'.join(path[:-1]),'Descrizione':'Original immutable outcome; fresh never used for model selection; OOD and swaps diagnostic.'})
            observations.append(rid);subsets.setdefault(family,[]).append(rid)
    for role, rows in [('fresh',report['learned']),('development',json.loads((root/'development_ladder.json').read_text()))]:
        for row in rows:
            suffix=f'{role}-{row["family"]}-w{row["width"]}-step{row["step"]}-s{row["seed"]}'
            artifact='final_report.json' if role=='fresh' else 'development_ladder.json'
            run=put('runs',suffix,**{'Nome':'Task24 '+suffix,'Stato':'Completata','Braccio':[ref('arms',row['family']+'-w'+str(row['width']))],'Blocco':[ref('blocks',role)],'Seed':str(row['seed']),'Artefatti prodotti':[artifacts[artifact]],'Configurazione effettiva':json.dumps({k:row[k] for k in ('family','width','step','seed')}),'Validità tecnica':'CRC, checkpoint/source/freeze SHA, native oracle, vectorization verified.'})
            record(run,suffix,row['metrics'],artifact,row['family'],'gates')
            for name,group in [('held_rollout','held_gates'),('path_rollout','path_gates'),('swapped_identity','gates')]:
                if name in row: record(run,suffix+'-'+name,row[name],artifact,row['family'],group)
            record(run,suffix+'-summary',{k:row[k] for k in ('score','passed','parameter_count','swapped_identity_rmse_ratio') if k in row},artifact,row['family'])
    for name,rows in [('final_report.json',report['numerical']),('paired_architecture_contrasts.json',json.loads((root/'paired_architecture_contrasts.json').read_text())),('paired_scaling_contrasts.json',json.loads((root/'paired_scaling_contrasts.json').read_text())),('native_audit.json',json.loads((root/'native_audit.json').read_text())['rows']),('equivalence_preflight.json',json.loads((root/'equivalence_preflight.json').read_text())['rows']),('inference_benchmark.json',report['inference_benchmark']['rows'])]:
        for n,row in enumerate(rows):
            key=name[:-5]+'-'+str(n)
            run=put('runs',key,**{'Nome':'Task24 '+key,'Stato':'Completata','Artefatti prodotti':[artifacts[name]],'Validità tecnica':'Verified diagnostic, no post-hoc selection.'})
            record(run,key,row,name)
    findings=[]
    specifications=[('composition','Negativo','Independent bundle fails 3/3. All ten violations are Nap_Et2 h or combined gate RMSE: seed17 held1000 h=.00206812>.002; seeds29/43 additionally one-step and long-held criteria.', 'independent'),('sharing','Misto','Shared-heads and conditioned pass3/3; conditioned420 vs independent1116 parameters. Confirmed sharing remains false because independent reference fails; not evidence sharing is harmful.','conditioned'),('sodium_oracle','Positivo','All three canonical sodium mechanisms pass native rates, update and current audits. No execution error.',''),('path_transfer','Misto','Both shared families pass fast and slow imposed paths and long holds. Independent bundle violates long-held Nap h criteria, so global conjunction not passed.','independent')]
    for key,outcome,text,family in specifications:
        finding=put('findings',key,**{'Nome':'Task24 '+key,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Osservazioni':subsets.get(family,[]),'Risultato':text,'Limitazioni':cfg['limits']});findings.append(finding)
        verdict='Sostiene' if key=='sodium_oracle' else 'Contraddice' if key=='composition' else 'Indeterminata'
        put('evidence',key,**{'Nome':'Task24 '+key,'Esito':verdict,'Affermazione valutata':[ref('claims',key)],'Risultati a sostegno':[finding],'Argomentazione':text})
        put('claims',key,**{'Stato':'Supportata nel dominio' if key=='sodium_oracle' else 'Contraddetta nel dominio' if key=='composition' else 'Indeterminata'})
    put('decisions','to24b',**{'Nome':'Task24b independent sodium optimization diagnosis','Esito':'Modificare','Risultati':findings,'Motivazione':'Task25 stays blocked. Distinguish optimizer oscillation, channel gradient clipping coupling, width/budget and numerical long-held drift. New fresh confirmation after new freeze; original thresholds retained.','Condizioni di revisione':'No reclassification of original NO-GO; no fresh-driven checkpoint selection.'})
    ratios=[row['swapped_identity_rmse_ratio'] for row in report['learned']]
    minimum=min(ratios)
    scaling=json.loads((root/'paired_scaling_contrasts.json').read_text())
    medians={f:{factor:statistics.median([r['improvement_fraction'] for r in scaling if r['family']==f and r['factor']==factor]) for factor in ('budget','width')} for f in cfg['families']}
    for key,outcome,text,verdict in [
        ('identity','Positivo' if minimum>=10 else 'Negativo',f'Minimum swapped identity RMSE ratio across9systems={minimum}; threshold10. This does not make a failing architecture pass.','Sostiene' if minimum>=10 else 'Contraddice'),
        ('scaling','Misto','Paired development median improvements: '+json.dumps(medians)+'. Larger budget/width is not uniformly beneficial; all raw contrasts preserved.','Indeterminata'),
        ('compute','Positivo','Frozen rate+gate forward timings measured on the same GPU atbatch1/128/1024; repetition-level samples preserved. No full-neuron speedup or selection on latency claimed.','Sostiene')]:
        finding=put('findings',key,**{'Nome':'Task24 '+key,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Risultato':text,'Limitazioni':cfg['limits']})
        put('evidence',key,**{'Nome':'Task24 '+key,'Esito':verdict,'Affermazione valutata':[ref('claims',key)],'Risultati a sostegno':[finding],'Argomentazione':text})
        put('claims',key,**{'Stato':'Supportata nel dominio' if verdict=='Sostiene' else 'Contraddetta nel dominio' if verdict=='Contraddice' else 'Indeterminata'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':list(artifacts.values()),'Validità tecnica':'COMPLETE exit0. Scientific independent NO-GO; both shared families3/3.'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,observations=len(observations),airtable_accessed=False)))


if __name__=='__main__': main()
