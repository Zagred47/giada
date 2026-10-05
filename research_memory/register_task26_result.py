"""Persist Task26 observations, contrasts, probes and scoped conclusions."""
import hashlib
import json
import statistics

from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task24_result import scalars
from .register_task25_result import summary


def main():
    repo=ROOT.parent
    folder=repo/'experiments/results/task26_kaggle_2a6bd21'
    report=json.loads((folder/'final_report.json').read_text(encoding='utf-8'))
    freeze=json.loads((folder/'selection_freeze.json').read_text(encoding='utf-8'))
    status=json.loads((folder/'process_status.json').read_text(encoding='utf-8'))
    cfg=json.loads((folder/'run_contract.json').read_text(encoding='utf-8'))
    import src.giada_teacher.current_supervision_comparison as next_task
    widths,hashes=next_task.verify_parent(repo,next_task.config(repo))
    assert status=={'phase':'gpu','returncode':0} and report['valid'] and report['comparison_passed']
    assert not report['fresh_used_for_selection'] and not report['current_supervision_used']
    assert report['per_family_passed']=={'independent':True,'shared_heads':True}
    for family in report['selection']:
        name=report['selection'][family]['checkpoint']
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==freeze['checkpoint_hashes'][name]
    mirror=ValidatedBatchMirror();cache={};byfamily={f:[] for f in cfg['families']}
    def put(table,key,**fields):
        rid=mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task26-{key}-v1',**fields})['record_id']
        cache[table,key]=rid
        return rid
    def ref(table,key):
        if (table,key) not in cache:
            code=f'{table}-giada-roadmap-task26-{key}-v1'
            cache[table,key]=mirror.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'][0]['record_id']
        return cache[table,key]
    artifacts={}
    for path in folder.iterdir():
        if path.suffix not in ('.json','.pt'):continue
        artifacts[path.name]=put('artifacts','result-'+path.stem,**{'Nome':'Task26 '+path.name,
            'Tipo':'Checkpoint' if path.suffix=='.pt' else 'Report','Percorso':path.relative_to(repo).as_posix(),
            'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'})
    def evaluation(metric,group):
        if group and metric in cfg[group]:return ref('evaluations',group+'-'+metric)
        key='diagnostic-'+metric
        if ('evaluations',key) not in cache:
            mid=put('metrics',key,**{'Nome':'Task26 '+metric,'Famiglia':'Regressione',
                'Direzione':'Minimizzare','Unità':'adimensionale',
                'Formula':'Scalare diagnostico. Definizione completa in run_contract.json e final_report.json.'})
            put('evaluations',key,**{'Nome':'Task26 '+metric,'Metrica':[mid],
                'Protocollo':[ref('protocols','paired')],'Versione':'v1','Ruolo':'Secondaria',
                'Aggregazione e pesi':'Seed, famiglia, dominio, canale, orizzonte e path mantenuti distinti.'})
        return ref('evaluations',key)
    observations=0
    def record(run,key,value,artifact,group=None):
        nonlocal observations
        ids=[]
        for path,v in scalars(summary(value)):
            name='/'.join(path)
            scalar_group=('current_gates' if group=='gates' else 'path_current_gates') if 'currents' in path else group
            id=put('observations',key+'-'+hashlib.sha256(name.encode()).hexdigest()[:20],**{
                'Nome':'Task26 '+key+' '+name,'Valore':v,'Run':[run],
                'Specifica di valutazione':[evaluation(path[-1],scalar_group)],
                'Artefatti dettagliati':[artifacts[artifact]],'Strato o sottogruppo':'/'.join(path[:-1]),
                'Descrizione':'Misura frozen del checkpoint selezionato. Pannelli ripetuti preservati integralmente nel report collegato.'})
            ids.append(id);observations+=1
        return ids
    for row in report['learned']:
        family=row['arm'];key=family+'-s'+str(row['seed'])
        arm=ref('arms',family+'-w'+str(row['width']))
        run=put('runs',key,**{'Nome':'Task26 '+key,'Stato':'Completata','Braccio':[arm],
            'Blocco':[ref('blocks','fresh')],'Seed':str(row['seed']),
            'Artefatti prodotti':[artifacts['final_report.json'],artifacts[report['selection'][family]['checkpoint']]],
            'Validità tecnica':'Kaggle COMPLETE, worker exit0; freeze, parent and selected checkpoint hashes verified.'})
        ids=[]
        for field,group in [('metrics','gates'),('held_rollout','held_gates'),('held_fp64_diagnostic',None),
                            ('path_rollout','path_gates'),('swapped_identity',None)]:
            ids.extend(record(run,key+'-'+field,row[field],'final_report.json',group))
        ids.extend(record(run,key+'-outcome',{k:row[k] for k in ('passed','parameter_count','step','width')},'final_report.json'))
        byfamily[family].extend(ids)
    probes=[]
    for name in ('gradient_probes.json','clipping_probes.json','cost_probes.json','timing_probes.json',
                 'equivalence_preflight.json','native_audit.json','fresh_paired_contrasts.json'):
        run=put('runs','diagnostic-'+name.split('.')[0],**{'Nome':'Task26 '+name,'Stato':'Completata',
            'Artefatti prodotti':[artifacts[name]],'Validità tecnica':'Probe non selezionabile; artefatto completo preservato.'})
        probes.extend(record(run,'diagnostic-'+name.split('.')[0],json.loads((folder/name).read_text(encoding='utf-8')),name))
    for name,data in [('paired_contrasts',report['paired_contrasts']),('timing',report['timing'])]:
        run=put('runs','diagnostic-'+name,**{'Nome':'Task26 '+name,'Stato':'Completata',
            'Artefatti prodotti':[artifacts['final_report.json']],'Validità tecnica':'Probe non selezionabile.'})
        probes.extend(record(run,'diagnostic-'+name,data,'final_report.json'))
    counts=report['selected_parameter_counts'];compression=1-counts['shared_heads']/counts['independent']
    paired=report['paired_contrasts']
    medians={factor:statistics.median([row['improvement_fraction'] for row in paired if row['factor']==factor])
             for factor in ('sharing','capacity','budget','approx_parameter_matched')}
    gradients=json.loads((folder/'gradient_probes.json').read_text(encoding='utf-8'))
    negative=sum(v<0 for probe in gradients for matrix in probe['cosines'] for i,row in enumerate(matrix) for j,v in enumerate(row) if i<j)
    total=sum(1 for probe in gradients for matrix in probe['cosines'] for i,row in enumerate(matrix) for j,_ in enumerate(row) if i<j)
    specs={
        'joint_learnability':('Positivo','Supportata nel dominio',
            f'Both independently trained families pass 3/3 seed. Selected independent width{widths["independent"]} '+
            f'has {counts["independent"]} parameters; shared width{widths["shared_heads"]} has {counts["shared_heads"]} '+
            f'({compression:.1%} fewer). V imposed; all fresh/held/path/current gates pass.',
            byfamily['independent']+byfamily['shared_heads']),
        'capacity':('Misto','Indeterminata',
            f'Paired median development capacity improvement={medians["capacity"]:.4f}; individual/group variation retained. '+
            'Selected widths differ across families; no general capacity law inferred.',probes),
        'budget':('Misto','Indeterminata',
            f'Paired median 60k-vs30k development improvement={medians["budget"]:.4f}. Both families selected step60000. '+
            'Mini scaling law is local to this training schedule.',probes),
        'gradient_interference':('Misto','Indeterminata',
            f'{negative}/{total} off-diagonal shared-trunk gradient cosine values are negative across fixed probes. '+
            'This is a local conflict diagnostic, not a causal failure attribution.',probes),
        'current_masking':('Positivo','Supportata nel dominio',
            'Every current panel and individual channel passed; promotion never relied on a cancelling total alone. '+
            'The panel-level cancellation and current errors are preserved in final_report.json.',
            byfamily['independent']+byfamily['shared_heads']),
        'compute':('Misto','Indeterminata',
            f'Shared has {compression:.1%} fewer selected parameters. GPU timing and theoretical MAC probes are retained. '+
            'No full-neuron acceleration claim; GateD remains incomplete.',probes)}
    findings=[]
    for key,(outcome,state,text,ids) in specs.items():
        finding=put('findings',key,**{'Nome':'Task26 '+key,'Esito':outcome,
            'Esperimenti':[ref('experiments','matrix')],'Osservazioni':ids,'Risultato':text,'Limitazioni':cfg['limits']})
        findings.append(finding)
        put('evidence',key,**{'Nome':'Task26 '+key,'Esito':'Sostiene' if outcome=='Positivo' else 'Indeterminata',
            'Affermazione valutata':[ref('claims',key)],'Risultati a sostegno':[finding],'Argomentazione':text})
        put('claims',key,**{'Stato':state})
    put('decisions','to27',**{'Nome':'Prepare original Task27 current supervision factorial','Esito':'Continuare',
        'Risultati':findings,'Motivazione':'Independent and shared heads both pass 3/3 on fixed five-channel support; '+
        'current supervision is isolated in Task27 with new paired fit/development/fresh.',
        'Condizioni di revisione':'Task26 did not train on currents; no GateD/closed-loop/full neuron claim.'})
    put('experiments','matrix',**{'Stato':'Concluso'})
    put('runs','kaggle-orchestration',**{'Stato':'Completata','Artefatti prodotti':list(artifacts.values()),
        'Validità tecnica':'Kaggle COMPLETE, worker exit0, source/freeze/selected checkpoint hashes verified. Both families3/3.'})
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'observations':observations,'findings':len(findings),
                      'compression_fraction':compression,'airtable_accessed':False,'parent_hashes':hashes}))


if __name__=='__main__':main()
