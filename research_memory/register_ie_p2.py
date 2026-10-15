"""Register IE-P2 prepared protocol or audited result in one local batch."""
import argparse
import hashlib
import json
import math
import statistics
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;path=repo/'experiments/ie_p2_residual_cost_preregistration_v1.json'
    spec=json.loads(path.read_text())
    assert hashlib.sha256((repo/spec['parent_report']).read_bytes()).hexdigest()==spec['parent_sha256']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p2-{suffix}-v1',**fields})['record_id']
    experiment=put('experiments','residual',Nome='IE-P2 resident residual cost and execution matrix',
        Stato='Preregistrato',Descrizione=spec['scope'],**{'Obiettivo informativo':spec['decision']})
    protocol=put('protocols','residual',Nome='IE-P2 formula block and execution counterfactuals',Versione='v1',
        Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec,ensure_ascii=False),**{
            'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),
            'Criteri di successo':json.dumps(spec['hypotheses']),
            'Regole di arresto':spec['validation']})
    artifacts=[]
    for relative in ('experiments/ie_p2_residual_cost_preregistration_v1.json','experiments/ie_p2_design_v1.md',
                     'notebooks/ie_p2_residual_cost.ipynb','scripts/run_ie_p2_residual_cost.py',
                     'src/giada_teacher/ie_residual_cost.py'):
        p=repo/relative
        artifacts.append(put('artifacts',p.stem,Nome='IE-P2 '+p.name,Tipo='Report',Percorso=relative,
            Versione='v1',**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    if args.result:
        launch=json.loads((repo/'experiments/ie_p2_kaggle_launch_v1.json').read_text())
        folder=repo/'experiments/results/ie_p2_kaggle_v1'
        report=json.loads((folder/'final_report.json').read_text())
        assert report['valid'] and not report['training_performed'] and not report['candidate_promoted']
        assert report['code_revision']==launch['code_revision']
        assert len([r for r in report['rows'] if r['mode']=='compiled'])==27
        assert all(r['valid'] for r in report['preflights'])
        for row in report['rows']:
            times=row['timings']['wall_ms']
            assert len(times)==30 and all(math.isfinite(t) and t>0 for t in times)
            if 'saving_vs_full' in row:
                full=next(x for x in report['rows'] if x['mode']==row['mode'] and
                          x['batch_per_seed']==row['batch_per_seed'] and x['variant']=='full')['timings']['wall_ms']
                assert abs(statistics.median(1-b/a for a,b in zip(full,times))-
                           row['saving_vs_full']['median_saved_fraction'])<1e-12
        for row in report['execution_contrasts']:
            def wall(mode):
                return next(x for x in report['rows'] if x['batch_per_seed']==row['batch'] and
                            x['mode']==mode and x['variant']=='full')['timings']['wall_ms']
            assert abs(statistics.median(1-b/a for a,b in zip(wall('compiled'),wall('cuda_graph')))-
                       row['median_saved_fraction'])<1e-12
        for p in sorted(folder.iterdir()):
            if p.is_file():
                artifacts.append(put('artifacts','result-'+p.stem,Nome='IE-P2 output '+p.name,Tipo='Report',
                    Percorso=p.relative_to(repo).as_posix(),Versione='v1',**{
                        'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
        receipt=repo/'experiments/ie_p2_kaggle_launch_v1.json'
        put('artifacts','launch',Nome='IE-P2 launch receipt',Tipo='Report',Percorso=receipt.relative_to(repo).as_posix(),
            Versione='v1',**{'SHA-256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'Dimensione byte':receipt.stat().st_size})
        arm=put('arms','matrix',Nome='IE-P2 resident formula interventions',Ruolo='Oracle diagnostico',Protocollo=[protocol])
        run=put('runs','kaggle',Nome='IE-P2 Kaggle v1',Stato='Completata',Braccio=[arm],**{
            'Artefatti prodotti':artifacts,'Configurazione effettiva':json.dumps(launch),
            'Validità tecnica':'Compiled preflights and all 27 compiled timing cells pass; optional CUDA graph failures are preserved separately.'})
        metric=put('metrics','wall',Nome='IE-P2 resident wall time per ensemble step',Famiglia='Regressione',
            Direzione='Minimizzare',**{'Unità':'ms','Formula':'Median synchronized wall_ms across 30 timing blocks; fixed inputs, same diagnostic ABI for cache interventions.'})
        evaluation=put('evaluations','wall',Nome='IE-P2 same-input resident cost',Metrica=[metric],Versione='v1',
            Ruolo='Diagnostica',Target='Opportunity map only; no learned candidate promotion.',**{'Aggregazione e pesi':spec['uncertainty']})
        observations=[]
        for row in report['rows']:
            observations.append(put('observations',f"{row['mode']}-{row['variant']}-{row['batch_per_seed']}",
                Nome=f"IE-P2 {row['mode']} {row['variant']} b{row['batch_per_seed']}",
                Valore=statistics.median(row['timings']['wall_ms']),Run=[run],**{
                    'Specifica di valutazione':[evaluation],'Artefatti dettagliati':artifacts,
                    'Descrizione':json.dumps(row)}))
        put('findings','opportunity',Nome='IE-P2 block and execution opportunity map',Esito='Misto',
            Osservazioni=observations,**{'Risultato':json.dumps(report.get('targeting',{})),
            'Limitazioni':spec['decision']+' '+spec['exclusions']})
        mirror.local_upsert('experiments',{'Codice stabile':'experiments-giada-ie-p2-residual-v1','Stato':'Concluso'})
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'result_registered':args.result,'airtable_accessed':False}))


if __name__=='__main__':main()
