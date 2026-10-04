"""Persist Task25 gates, currents, controls, contrasts and scoped conclusions."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task24_result import scalars


def summary(value):
    """Repeated panel details remain losslessly in indexed JSON artifacts."""
    if isinstance(value,list):return [summary(v) for v in value]
    if not isinstance(value,dict):return value
    result={k:summary(v) for k,v in value.items() if k!='panels'}
    if 'panels' in value:
        panels=value['panels'];result['cancellation_count_max']=max(p['cancellation_count'] for p in panels)
        for key in ('cancellation_total_normalized_rmse','cancellation_individual_normalized_rmse'):
            values=[p[key] for p in panels if p[key] is not None]
            if values:result[key+'_max']=max(values)
    return result


def main():
    folder=ROOT.parent/'experiments/results/task25_kaggle_bdb75e6';report=json.loads((folder/'final_report.json').read_text());audit=json.loads((folder/'result_audit.json').read_text());cfg=json.loads((folder/'run_contract.json').read_text())
    assert audit['valid'] and audit['checkpoint_and_source_hashes_valid'] and report['composition_passed'] and not report['fresh_used_for_selection']
    m=ValidatedBatchMirror();cache={};observations=[];byarm={};diagnostics=[]
    def put(table,key,**fields):
        rid=m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task25-{key}-v1',**fields})['record_id'];cache[table,key]=rid;return rid
    def ref(table,key):
        if (table,key) not in cache:cache[table,key]=m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{table}-giada-roadmap-task25-{key}-v1'+"'")['rows'][0]['record_id']
        return cache[table,key]
    artifacts={p.name:put('artifacts','result-'+p.stem,**{'Nome':'Task25 '+p.name,'Tipo':'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'}) for p in folder.iterdir() if p.suffix=='.json'}
    def evaluation(path,group=None):
        key=path[-1]
        if group and 'currents' in path:group='current_gates' if group=='gates' else 'path_current_gates'
        if group and key in cfg[group]:return ref('evaluations',group+'-'+key)
        code='diagnostic-'+key
        if ('evaluations',code) not in cache:
            metric=put('metrics',code,**{'Nome':'Task25 '+key,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'mA/cm2' if 'ma_cm2' in key else 'adimensionale','Formula':'Verified diagnostic scalar; exact meaning given by indexed artifact and stratum.'})
            put('evaluations',code,**{'Nome':'Task25 '+key,'Metrica':[metric],'Protocollo':[ref('protocols','paired')],'Versione':'v1','Ruolo':'Secondaria','Aggregazione e pesi':'Seed/arm/domain/channel/held horizon/path kept separate; panel constants in configuration, raw panels in immutable report.'})
        return ref('evaluations',code)
    def record(run,key,value,artifact,group=None):
        ids=[]
        for path,v in scalars(value):
            ids.append(put('observations',key+'-'+hashlib.sha256('/'.join(path).encode()).hexdigest()[:20],**{'Nome':'Task25 '+key+' '+ '/'.join(path),'Valore':v,'Run':[run],'Specifica di valutazione':[evaluation(path,group)],'Artefatti dettagliati':[artifacts[artifact]],'Strato o sottogruppo':'/'.join(path[:-1]),'Descrizione':'No training/new selection; frozen source hashes verified. Full repeated panels preserved in linked report, not silently discarded.'}))
        observations.extend(ids);return ids
    for row in report['learned']:
        key=row['arm']+'-s'+str(row['seed']);run=put('runs',key,**{'Nome':'Task25 '+key,'Stato':'Completata','Braccio':[ref('arms',row['arm'])],'Blocco':[ref('blocks','fresh')],'Seed':str(row['seed']),'Artefatti prodotti':[artifacts['final_report.json']],'Validità tecnica':'COMPLETE exit0; CRC/freeze/source/native verified. Scientific gate recorded separately.'})
        ids=[]
        for phase,group in [('metrics','gates'),('held_rollout','held_gates'),('held_fp64_diagnostic',None),('path_rollout','path_gates'),('swapped_identity',None)]:ids.extend(record(run,key+'-'+phase,summary(row[phase]),'final_report.json',group))
        ids.extend(record(run,key+'-outcome',{k:row[k] for k in ('passed','parameter_count','identity_rmse_ratio')},'final_report.json'))
        # Normalize all measured one-step panel values; retain supplied parameters as run metadata.
        for domain,metrics in row['metrics'].items():
            for index,panel in enumerate(metrics['currents']['panels']):
                values={k:v for k,v in panel.items() if k not in ('gbar_multipliers','reversals_mv')}
                ids.extend(record(run,key+'-'+domain+'-panel'+str(index),values,'final_report.json'))
        byarm.setdefault(row['arm'],[]).extend(ids)
    for name,value in [('numerical',report['numerical']),('paired_contrasts',report['paired_contrasts']),('native_audit',json.loads((folder/'native_audit.json').read_text()))]:
        run=put('runs',name,**{'Nome':'Task25 '+name,'Stato':'Completata','Artefatti prodotti':[artifacts['native_audit.json' if name=='native_audit' else 'final_report.json']],'Validità tecnica':'Diagnostic only, no candidate selection.'})
        diagnostics.extend(record(run,name,summary(value),'native_audit.json' if name=='native_audit' else 'final_report.json'))
    findings=[]
    specs={
      'heterogeneous_composition':('Supportata nel dominio','Positivo','Independent1860parameters passes3/3 onall5channels,allrequired domains,paths,held horizons andworst current panels. Five mechanisms atimposedV only.',byarm['independent']),
      'compression_reuse':('Supportata nel dominio','Misto','Calcium-only compression1556passes3/3. Sodium-compressed1164 andboth-compressed860 pass2/3: seed43 activation-boundary mRMSE NaTa=.001118951,NaTs=.001078819 exceed .001. Allcurrent/path/held gatespass. No globalcompression orspeedup claim.',byarm['calcium_compressed']+byarm['sodium_compressed']+byarm['both_compressed']),
      'cancellation':('Indeterminata','Misto','Opposite-sign current panels and cancellation strata recorded. Individual/worst-panel gates prevent total-onlypromotion. Compressed sodium gate failures are not evidence cancellation causes them; total/currentpass doesnot imply gatespass.',diagnostics),
      'identity':('Supportata nel dominio','Positivo','Minimum identity-swapped gateRMSE ratio across12systems='+str(audit['minimum_identity_rmse_ratio'])+' (>10). Diagnostic,notpromotion.',[]),
      'precision':('Indeterminata','Misto','Same learnedFP32rates used inFP32/FP64 heldsolvers; formulaFP32/LUT controls recorded. Compressed failures are one-stepactivation m errors,not a failure ofnative execution. No precision rescue orposthoccandidate selection.',diagnostics),
      'path_transfer':('Supportata nel dominio','Positivo','Allfourbundles pass every fast/slow imposedpath,allseeds andeveryheldhorizon. Compression stillfails twoone-stepmcriteria. Not autonomousvoltage or1msendpoint sufficiency.',byarm['independent'])}
    for key,(state,outcome,text,ids) in specs.items():
        f=put('findings',key,**{'Nome':'Task25 '+key,'Esito':outcome,'Esperimenti':[ref('experiments','matrix')],'Osservazioni':ids,'Risultato':text,'Limitazioni':cfg['limits']});findings.append(f)
        put('evidence',key,**{'Nome':'Task25 '+key,'Esito':'Sostiene' if state.startswith('Supportata') else 'Indeterminata','Affermazione valutata':[ref('claims',key)],'Risultati a sostegno':[f],'Argomentazione':text});put('claims',key,**{'Stato':state})
    put('decisions','to26',**{'Nome':'Prepare original Task26 joint vs independent','Esito':'Continuare','Risultati':findings,'Motivazione':'Valid independent heterogeneous composition. Calcium sharing reuse verified; do not silentlypromote sodiumcompression. Task26 ismatchedlearnability comparison,newfit/development/fresh,nottestdrivenrepair.','Condizioni di revisione':'No fullionblock orclosedloop authorization. Task24/24b immutable decisions retained.'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':list(artifacts.values()),'Validità tecnica':'COMPLETE exit0; independent3/3 andcalciumcompressed3/3; othercompressed2/3. Evaluation73.92s; total~150s.'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,observations=len(observations),findings=6,task26_preparation_authorized=True,airtable_accessed=False)))
    from .link_task25_evidence import main as link_evidence
    link_evidence()


if __name__=='__main__':main()
