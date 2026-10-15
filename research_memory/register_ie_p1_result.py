"""Audit timing aggregates and record IE-P1 v3 outcome, SQLite only."""
import hashlib
import json
import statistics
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    directory=repo/'experiments/results/ie_p1_v3_kaggle_b177d01'
    report=json.loads((directory/'final_report.json').read_text())
    assert report['valid'] and not report['candidate_promoted'] and not report['training_performed']
    assert not report['whole_simulator_profile_complete'] and len(report['rows'])==9
    assert (directory/'code_revision.txt').read_text().strip()=='b177d01bb693da05224d961d0ba2719dcc94bd78'
    assert len(report['numerical_errors'])==27
    assert all(all(a<=b for a,b in zip(row['maximum_errors'],row['limits'])) for row in report['numerical_errors'])
    for row in report['rows']:
        full=row['timings']['membrane_core']['wall_ms']
        cached=row['timings']['membrane_core_cached_gates']['wall_ms']
        assert len(full)==len(cached)==30 and all(x>0 for x in full+cached)
        assert abs(statistics.median(1-b/a for a,b in zip(full,cached))-
                   row['gate_buffer_diagnostic']['median_saved_fraction'])<1e-12
    mirror=ValidatedBatchMirror()
    for code in ('runs-giada-ie-p1-kaggle-v1','runs-giada-ie-p1-kaggle-v2'):
        mirror.local_upsert('runs',{'Codice stabile':code,'Stato':'Fallita'})
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p1-result-{suffix}-v1',**fields})['record_id']
    metric=put('metrics','paired-savings',Nome='IE-P1 paired gate-buffer resident time saving',
        Famiglia='Regressione',Direzione='Descrittiva',**{'Unità':'fraction',
        'Formula':'median_j(1 - wall_cached_j / wall_full_j); paired timing blocks, not ratio of component medians.'})
    evaluation=put('evaluations','paired-savings',Nome='IE-P1 resident diagnostic cost evaluation',
        Metrica=[metric],Versione='v1',Ruolo='Diagnostica',Target='Map opportunity; no candidate speed promotion.',**{
        'Aggregazione e pesi':'30 paired blocks x10 calls, separate arm/batch, three seeds vectorized; within-run bootstrap interval.'})
    artifacts=[]
    for path in directory.iterdir():
        if path.is_file():
            artifacts.append(put('artifacts',path.stem.replace('_','-'),Nome='IE-P1 v3 '+path.name,Tipo='Report',
                Percorso=path.relative_to(repo).as_posix(),Versione='v3',**{
                    'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
    run=mirror.local_upsert('runs',{'Codice stabile':'runs-giada-ie-p1-kaggle-v3','Stato':'Completata',
        'Artefatti prodotti':artifacts,'Validità tecnica':'All 27 compiler numerical preflights and 9 benchmark cells passed; timing medians independently recomputed. Prospective v2 compiler amendment applied; v1/v2 incidents preserved.'})['record_id']
    obs=[]
    for row in report['rows']:
        obs.append(put('observations',f"{row['arm']}-{row['batch_per_seed']}",
            Nome=f"IE-P1 gate-buffer diagnostic {row['arm']} batch{row['batch_per_seed']}",
            Valore=row['gate_buffer_diagnostic']['median_saved_fraction'],Run=[run],**{
                'Specifica di valutazione':[evaluation],
                'Artefatti dettagliati':artifacts,'Descrizione':json.dumps(row,ensure_ascii=False)}))
    finding=put('findings','cost-margin',Nome='Gate-only replacement has limited margin against compiled formulas',
        Esito='Misto',Osservazioni=obs,**{
            'Risultato':'T4 FP32, three seeds vectorized, batch642: formula full 0.464987ms vs cached-gate core 0.392049ms, paired saving 15.92% (within-run bootstrap 14.62–16.42%). Independent/shared savings 48.23%/55.60%. Main benchmark valid; no learned speedup demonstrated.',
            'Limitazioni':'Cached gates are noncausal diagnostic buffers. Compiler fusion makes differences workload-specific, not a universal speedup ceiling or additive operation cost. Host events/plasticity/RNG, transfers and full simulator excluded; no candidate trained.'})
    put('decisions','target-review',Nome='Use measured gate margin before next ionic training',Esito='Continuare',
        Risultati=[finding],**{'Motivazione':'Complete IE-03 target selection and IE-04/05 isolated contract. Gate-only work can remove hybrid overhead, but massive gains against formulas require evaluating remaining readout/solver and execution costs. Do not infer that retained formulas alone cause the slowdown.',
        'Condizioni di revisione':'Whole-simulator cost map and isolated native playground remain incomplete. No candidate promotion or Task36 authorization.'})
    mirror.local_upsert('experiments',{'Codice stabile':'experiments-giada-ie-p1-profile-v1','Stato':'Concluso'})
    mirror.local_upsert('actions',{'Codice stabile':'actions-giada-ie-02-v1','Stato':'In corso',
        'Criteri di avvio':'Resident IE-P1 phase complete; host event/RNG and full-simulator profiling still open. Finding '+finding})
    mirror.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'ie_p1_complete':True,'whole_ie02_complete':False,'airtable_accessed':False}))


if __name__=='__main__':main()
