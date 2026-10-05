"""Register eager Gate D outcome and separately preregister compiled confirmation."""
import hashlib
import json

from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    folder=repo/'experiments/results/gate_d_compute_kaggle_59c91e7'
    report=json.loads((folder/'final_report.json').read_text(encoding='utf-8'))
    audit=json.loads((folder/'result_audit.json').read_text(encoding='utf-8'))
    cfg=json.loads((repo/'experiments/task28c_gate_d_compiled_confirmation.json').read_text(encoding='utf-8'))
    from src.giada_teacher.gate_d_compiled_confirmation import verify_parent
    verify_parent(repo,cfg)
    assert audit['valid'] and audit['eager_registered_passed'] and report['valid']
    assert not audit['compiled_baseline_tested']
    mirror=ValidatedBatchMirror()
    def put(table,key,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-gated-compiled-{key}-v1',**fields})['record_id']
    artifacts={}
    for path in sorted(folder.iterdir()):
        if path.suffix not in ('.json','.zip'):continue
        artifacts[path.name]=put('artifacts','eager-'+path.stem,**{
            'Nome':'Gate D eager '+path.name,'Tipo':'Checkpoint' if path.suffix=='.zip' else 'Report',
            'Percorso':path.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'Dimensione byte':path.stat().st_size,'Versione':'v1'})
    eager_findings=[]
    for row in report['rows']:
        family,batch=row['family'],row['batch_size']
        run=put('runs',f'eager-{family}-{batch}',**{
            'Nome':f'Gate D eager {family} batch {batch}','Stato':'Completata',
            'Seed':'17,29,43','Artefatti prodotti':[artifacts['final_report.json']],
            'Validità tecnica':'Kaggle COMPLETE/exit0; exact GPU NumPy preflight, paired CUDA timings and source hashes independently audited.'})
        metric=put('metrics',f'eager-reduction-{family}-{batch}',**{
            'Nome':f'Gate D eager paired reduction {family} batch {batch}',
            'Famiglia':'Regressione','Direzione':'Massimizzare','Unità':'frazione',
            'Formula':'Median of (exact-hybrid)/exact across 60 paired CUDA events.'})
        evaluation=put('evaluations',f'eager-reduction-{family}-{batch}',**{
            'Nome':f'Gate D eager reduction {family} batch {batch}',
            'Metrica':[metric],'Versione':'v1','Ruolo':'Primaria' if batch==642 else 'Secondaria',
            'Target':'0.10' if batch==642 else 'diagnostic',
            'Aggregazione e pesi':'Within-family, alternating execution order, no cross-family averaging.'})
        observation=put('observations',f'eager-{family}-{batch}-gain',**{
            'Nome':f'Gate D eager {family} batch {batch} gain',
            'Valore':row['median_paired_compute_reduction_fraction'],'Run':[run],
            'Specifica di valutazione':[evaluation],
            'Artefatti dettagliati':[artifacts['final_report.json']],
            'Strato o sottogruppo':f'{family}/batch-{batch}',
            'Descrizione':'Exact and hybrid medians and all 60 paired timings retained in linked report. Accuracy passed.'})
        if batch==642:
            eager_findings.append(put('findings',f'eager-{family}',**{
                'Nome':f'Gate D eager {family} passes materiality',
                'Esito':'Positivo','Osservazioni':[observation],
                'Risultato':f"{row['median_paired_compute_reduction_fraction']:.3%} reduction; exact {row['exact_median_ms']:.4f} ms, hybrid {row['hybrid_median_ms']:.4f} ms; accuracy passed.",
                'Limitazioni':'Eager PyTorch only. Formula kernel-fusion sensitivity not yet tested.'}))
    claim=put('claims','compiler-robustness',**{
        'Nome':'Gate D advantage survives symmetric Inductor compilation',
        'Tipo':'Ipotesi','Stato':'Aperta',
        'Enunciato':'Both-arm hybrid remains >=10% faster than all-formula baseline when both are compiled with the same Inductor backend at batch 642.',
        'Limiti':cfg['limits'],'Condizioni di falsificazione':cfg['promotion']})
    experiment=put('experiments','compiled',**{
        'Nome':'Gate D symmetric compiled confirmation after eager GO',
        'Tipo':'Confermativo','Stato':'Preregistrato','Ipotesi':[claim],
        'Descrizione':cfg['motivation'],'Obiettivo informativo':cfg['promotion']})
    protocol=put('protocols','compiled',**{
        'Nome':'Gate D two-sided Inductor full block','Esperimento':[experiment],
        'Versione':'v1','Modalità':'Ricomposizione','Procedura':json.dumps(cfg,ensure_ascii=False),
        'Criteri di successo':cfg['promotion'],
        'Hash preregistrazione':hashlib.sha256((repo/'experiments/task28c_gate_d_compiled_confirmation.json').read_bytes()).hexdigest(),
        'Regole di arresto':'Compiler failure is technical non-comparability; <10% in either family is scientific non-promotion.'})
    put('decisions','await-compiled',**{
        'Nome':'Hold Task29 pending symmetric compiled Gate D',
        'Esito':'Continuare','Risultati':eager_findings,
        'Motivazione':'Both eager families exceed 10% with accuracy intact, but analytical branch may benefit disproportionately from kernel fusion.',
        'Condizioni di revisione':'Only symmetric compiled confirmation, or another fairly optimized same-device comparator, can support robust Gate D promotion.'})
    mirror.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'eager_families':len(eager_findings),
                      'compiled_preregistered':True,'airtable_accessed':False}))


if __name__=='__main__':main()
