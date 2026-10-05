"""Record the compiled Gate D NO-GO without erasing the eager positive result."""
import hashlib
import json

from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    folder=repo/'experiments/results/gate_d_compiled_kaggle_e66dfea'
    report=json.loads((folder/'final_report.json').read_text(encoding='utf-8'))
    audit=json.loads((folder/'result_audit.json').read_text(encoding='utf-8'))
    cfg=json.loads((folder/'run_contract.json').read_text(encoding='utf-8'))
    assert audit['valid'] and not audit['compiled_confirmation_passed']
    assert report['valid'] and not report['gate_d_robustly_supported'] and not report['task29_authorized']
    assert all(not r['material_reduction_passed'] for r in report['rows'])
    assert not report['training_performed'] and not report['fresh_used_for_selection']
    mirror=ValidatedBatchMirror();cache={}
    def put(table,key,**fields):
        rid=mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-gated-compiled-{key}-v1',**fields})['record_id']
        cache[table,key]=rid
        return rid
    def ref(table,key):
        if (table,key) not in cache:
            code=f'{table}-giada-gated-compiled-{key}-v1'
            rows=mirror.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
            assert len(rows)==1,code
            cache[table,key]=rows[0]['record_id']
        return cache[table,key]
    artifacts={}
    for path in sorted(folder.iterdir()):
        if path.suffix not in ('.json','.zip'):continue
        artifacts[path.name]=put('artifacts','compiled-'+path.stem,**{
            'Nome':'Gate D compiled '+path.name,'Tipo':'Checkpoint' if path.suffix=='.zip' else 'Report',
            'Percorso':path.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'Dimensione byte':path.stat().st_size,'Versione':'v1'})
    observations=[]
    for row in report['rows']:
        family=row['family']
        run=put('runs','compiled-'+family,**{
            'Nome':'Gate D compiled '+family,'Stato':'Completata','Seed':'17,29,43',
            'Artefatti prodotti':[artifacts['final_report.json'],artifacts['artifact_bundle.zip']],
            'Validità tecnica':'Kaggle COMPLETE/exit0, same Inductor backend, numerical compiled/eager equivalence and paired CUDA timings independently audited.'})
        for key,value,unit in (
            ('exact_compiled_median_ms',row['exact_compiled_median_ms'],'ms'),
            ('hybrid_compiled_median_ms',row['hybrid_compiled_median_ms'],'ms'),
            ('paired_compute_reduction_fraction',row['paired_compute_reduction_fraction'],'frazione'),
            ('exact_gate_difference',row['exact_gate_difference'],'adimensionale'),
            ('hybrid_gate_difference',row['hybrid_gate_difference'],'adimensionale')):
            metric=put('metrics',family+'-'+key,**{
                'Nome':'Gate D compiled '+family+' '+key,'Famiglia':'Regressione',
                'Direzione':'Minimizzare' if key!='paired_compute_reduction_fraction' else 'Massimizzare',
                'Unità':unit,'Formula':'Exact paired definition and samples preserved in final_report.json.'})
            evaluation=put('evaluations',family+'-'+key,**{
                'Nome':'Gate D compiled '+family+' '+key,'Metrica':[metric],
                'Protocollo':[ref('protocols','compiled')],'Versione':'v1',
                'Ruolo':'Primaria' if key=='paired_compute_reduction_fraction' else 'Secondaria',
                'Target':'0.10' if key=='paired_compute_reduction_fraction' else 'diagnostic',
                'Aggregazione e pesi':'Same device, dtype, output and compiler backend; 40 alternating paired repetitions.'})
            observations.append(put('observations',family+'-'+key,**{
                'Nome':'Gate D compiled '+family+' '+key,'Valore':float(value),'Run':[run],
                'Specifica di valutazione':[evaluation],
                'Artefatti dettagliati':[artifacts['final_report.json']],
                'Strato o sottogruppo':family+'/batch-642',
                'Descrizione':'Post-eager symmetric Inductor confirmation, not retroactive amendment.'}))
    finding=put('findings','compiler-no-go',**{
        'Nome':'Gate D compiled comparison reverses eager speed advantage','Esito':'Negativo',
        'Esperimenti':[ref('experiments','compiled')],'Osservazioni':observations,
        'Risultato':'Independent exact 0.3969 ms vs hybrid 0.6891 ms (hybrid ~73.6% slower); shared_heads exact 0.3752 ms vs hybrid 0.7862 ms (hybrid ~109.5% slower). Both compiled-vs-eager numerical comparisons pass. Eager 42.6–43.0% speedup was kernel-launch sensitive. No material compute reduction; Gate D incomplete and Task29 unauthorized.',
        'Limitazioni':cfg['limits']+' Compiled accuracy flags also false, likely due boundary occupancy; exact counts not in artifact. Compute failure is independently decisive.'})
    put('evidence','compiler-no-go',**{
        'Nome':'Gate D compiled evidence contradicts material speedup','Esito':'Contraddice',
        'Affermazione valutata':[ref('claims','compiler-robustness')],
        'Risultati a sostegno':[finding],
        'Argomentazione':'Same T4, float32, batch 642, all 18 gates plus 11 currents, fullgraph Inductor on both branches; 40 paired CUDA timings per family. Both reductions are negative.'})
    put('claims','compiler-robustness',**{'Stato':'Contraddetta nel dominio'})
    put('experiments','compiled',**{'Stato':'Concluso'})
    put('decisions','post-compiled',**{
        'Nome':'Do not promote Task29 after compiled Gate D NO-GO','Esito':'Modificare',
        'Risultati':[finding],
        'Motivazione':'Teacher-forced composition is accurate, but the optimized analytic block is faster in both registered families. Eager win is not robust.',
        'Condizioni di revisione':'A new architecture or implementation with its own preregistered fair optimized comparator and accuracy gates could reopen Gate D. Do not retroactively relax 10%.'})
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'gate_d_completed':False,'task29_authorized':False,
                      'observations':len(observations),'airtable_accessed':False}))


if __name__=='__main__':main()
