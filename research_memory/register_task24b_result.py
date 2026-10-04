"""Complete scalar evidence, paired contrasts and causal interpretations."""
import hashlib
import json
import statistics
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task24_result import scalars


def main():
    root=ROOT.parent/'experiments/results/task24b_kaggle_bde23cc'
    report=json.loads((root/'final_report.json').read_text());audit=json.loads((root/'result_audit.json').read_text())
    assert audit['valid'] and audit['checkpoint_and_source_hashes_valid'] and report['task25_preparation_authorized']
    cfg=json.loads((root/'run_contract.json').read_text());m=ValidatedBatchMirror();cache={};observations=[]
    def put(table,key,**fields):
        rid=m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task24b-{key}-v1',**fields})['record_id'];cache[table,key]=rid;return rid
    def ref(table,key):
        if (table,key) not in cache:cache[table,key]=m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{table}-giada-roadmap-task24b-{key}-v1'+"'")['rows'][0]['record_id']
        return cache[table,key]
    artifacts={p.name:put('artifacts','result-'+p.stem,**{'Nome':'Task24b '+p.name,'Tipo':'Checkpoint' if p.suffix=='.pt' else 'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'}) for p in root.iterdir() if p.suffix in ('.json','.pt')}
    def evaluation(path,group=None):
        key=path[-1]
        if group in ('gates','held_gates') and 'pair' in path:group='pair_gates'
        if group=='path_gates' and 'per_channel' in path:group='path_channel_gates'
        if group and key in cfg[group]:return ref('evaluations',group+'-'+key)
        code='diagnostic-'+key
        if ('evaluations',code) not in cache:
            metric=put('metrics',code,**{'Nome':'Task24b '+key,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'mA/cm2' if 'ma_cm2' in key else 'adimensionale','Formula':'Immutable scalar diagnostic; interpretation given by artifact/path. Not an unregistered promotion gate.'})
            put('evaluations',code,**{'Nome':'Task24b '+key,'Metrica':[metric],'Protocollo':[ref('protocols','paired')],'Versione':'v1','Ruolo':'Secondaria','Aggregazione e pesi':'Preserve arm,width,seed,budget,channel,precision,replacement and domain.'})
        return ref('evaluations',code)
    def record(run,suffix,value,artifact,group=None):
        for path,value in scalars(value):
            code=suffix+'-'+hashlib.sha256('/'.join(path).encode()).hexdigest()[:20]
            observations.append(put('observations',code,**{'Nome':'Task24b '+suffix+' '+ '/'.join(path),'Valore':value,'Run':[run],'Specifica di valutazione':[evaluation(path,group)],'Artefatti dettagliati':[artifacts[artifact]],'Strato o sottogruppo':'/'.join(path[:-1]),'Descrizione':'Verified source/freeze/checkpoint hashes; no old or new fresh selection.'}))
    for role,rows in [('fresh',report['learned']),('development',json.loads((root/'development_ladder.json').read_text()))]:
        for row in rows:
            arm=row['arm'];suffix=f'{role}-{arm}-w{row["width"]}-step{row["step"]}-s{row["seed"]}'
            if arm.startswith('frozen_') and ('arms',arm) not in cache:put('arms',arm,**{'Nome':'Task24b '+arm,'Ruolo':'Baseline','Protocollo':[ref('protocols','paired')],'Descrizione':'Frozen original Task24 sharing model; no retraining.'})
            armref=ref('arms',arm) if arm.startswith('frozen_') else ref('arms',arm+'-w'+str(row['width']))
            artifact='final_report.json' if role=='fresh' else 'development_ladder.json'
            run=put('runs',suffix,**{'Nome':'Task24b '+suffix,'Stato':'Completata','Braccio':[armref],'Blocco':[ref('blocks',role)],'Seed':str(row['seed']),'Artefatti prodotti':[artifacts[artifact]],'Configurazione effettiva':json.dumps({k:row[k] for k in ('arm','width','step','seed')}),'Validità tecnica':'COMPLETE; native and seed/Adam equivalence valid.'})
            record(run,suffix,row['metrics'],artifact,'gates')
            for key,group in [('held_rollout','held_gates'),('composed_hold','held_gates'),('path_rollout','path_gates')]:
                if key in row:record(run,suffix+'-'+key,row[key],artifact,group)
            record(run,suffix+'-summary',{k:v for k,v in row.items() if k not in ('metrics','held_rollout','composed_hold','path_rollout')},artifact)
    for name in ('paired_contrasts.json','gradient_probes.json','frozen_rate_precision_attribution.json','native_audit.json','equivalence_preflight.json'):
        run=put('runs',name[:-5],**{'Nome':'Task24b '+name,'Stato':'Completata','Artefatti prodotti':[artifacts[name]],'Validità tecnica':'Verified diagnostic; attribution is post hoc on the old immutable model, not selectable.'})
        record(run,name[:-5],json.loads((root/name).read_text()),name)
    contrasts=json.loads((root/'paired_contrasts.json').read_text())
    late={factor:statistics.median([r['improvement_fraction'] for r in contrasts if r['factor']==factor and r.get('step',60000)==60000]) for factor in ('schedule','clipping','budget','width')}
    texts={
      'annealing':('Supportata nel dominio','Positivo','Late schedule paired median at60k='+str(late['schedule'])+'. Small decay w16 passes3/3; constant w32 also passes. Schedule not sole sufficient explanation; budget matters.'),
      'clipping':('Contraddetta nel dominio','Negativo','No gradient clipping activation: all bundle/channel frequencies zero. Matched clipped variants are identical. Proposed clipping-coupling explanation unsupported in this run.'),
      'budget_capacity':('Supportata nel dominio','Positivo','Paired60kvs30k median improvement='+str(late['budget'])+'; width32vs16 at60k='+str(late['width'])+'. Four arms pass3/3. No universal monotone scaling claim.'),
      'rate_attribution':('Supportata nel dominio','Positivo','Old Nap_h: tau replacement most reduces1000ms error; inf replacement most reduces10000ms error. Both analytic rates produce FP32 floor9.38e-6/2.66e-5. Contributions interact; not additive.'),
      'precision':('Contraddetta nel dominio','Negativo','Identical learned rates castFP64 do not reduce old error by50%; rates dominate. Exact-rate FP64 floor~1e-14, FP32 floor~1e-5, far below old~.002 error.'),
      'selection':('Indeterminata','Misto','Held-aware and one-step-only development selections are identical for all4arms. No checkpoint-choice change is attributable to adding held-aware selection. Primary rule confirmed3/3; no causal selection-rule repair claim.')}
    findings=[]
    for key,(state,outcome,text) in texts.items():
        finding=put('findings',key,**{'Nome':'Task24b '+key,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Risultato':text,'Limitazioni':cfg['limits']});findings.append(finding)
        put('evidence',key,**{'Nome':'Task24b '+key,'Esito':'Sostiene' if state.startswith('Supportata') else 'Contraddice' if state.startswith('Contraddetta') else 'Indeterminata','Affermazione valutata':[ref('claims',key)],'Risultati a sostegno':[finding],'Argomentazione':text})
        put('claims',key,**{'Stato':state})
    findings.append(put('findings','repair-and-sharing',**{'Nome':'Task24b repair and sharing','Esito':'Positivo','Esperimenti':[ref('experiments','matrix')],'Risultato':'Primary decay_channel1116params passes3/3 fresh conjunction. Frozen conditioned420passes3/3 (62.37% fewer); shared_heads1516passes but no compression gain. Original Task24 NO-GO remains unchanged. No measured runtime acceleration.'}))
    put('decisions','to25',**{'Nome':'Prepare original Task25 heterogeneous mechanisms','Esito':'Continuare','Risultati':findings,'Motivazione':'Recovered independent sodium reference plus calcium pair allow imposed-V heterogeneous composition. No autonomous voltage authorization.','Condizioni di revisione':'Task25 requires frozen checkpoints, common input/path and per-mechanism gates, not merely total current.'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':list(artifacts.values()),'Validità tecnica':'Exit0, primary3/3; all6arms3/3. Scientific valid.'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,observations=len(observations),late_contrasts=late,airtable_accessed=False)))


if __name__=='__main__':main()
