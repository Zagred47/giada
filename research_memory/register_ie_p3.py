"""Local-only IE-P3 preregistration, before opening fresh atomic outcomes."""
import hashlib
import json
import argparse
import math
import statistics
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;path=repo/'experiments/ie_p3_readout_preregistration_v1.json'
    spec=json.loads(path.read_text());mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p3-{suffix}-v1',**fields})['record_id']
    experiment=put('experiments','readout',Nome='IE-P3 paired exact and learned ionic readout',Stato='Preregistrato',
        Descrizione=spec['scope'],**{'Obiettivo informativo':spec['decision']})
    protocol=put('protocols','readout',Nome='IE-P3 atomic aperture learnability and matched execution',Versione='v1',
        Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec,ensure_ascii=False),**{
        'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),
        'Criteri di successo':json.dumps(spec['quality'])+'; paired saving >=10% with positive CI versus vectorized algebra at b642; no deployment authorization.',
        'Regole di arresto':spec['validation']})
    artifacts=[]
    for relative in ('experiments/ie_p3_readout_preregistration_v1.json','experiments/ie_p3_design_v1.md',
                     'src/giada_teacher/ie_readout_comparison.py','scripts/run_ie_p3_readout.py',
                     'notebooks/ie_p3_readout.ipynb','tests/test_ie_readout_comparison.py'):
        p=repo/relative
        artifacts.append(put('artifacts',p.stem,Nome='IE-P3 '+p.name,Tipo='Report',Percorso=relative,Versione='v1',**{
            'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    arms={}
    for name in spec['arms']:
        arms[name]=put('arms',name,Nome='IE-P3 '+name,Ruolo='Baseline' if 'algebra' in name else 'Trattamento',Protocollo=[protocol])
    put('actions','execute',Nome='Execute IE-P3 single notebook with paired six-member training',Stato='Pronta',
        Descrizione=spec['decision']+' '+spec['exclusions'])
    if args.result:
        folder=repo/'experiments/results/ie_p3_kaggle_v1'
        report=json.loads((folder/'final_report.json').read_text())
        launch=json.loads((repo/'experiments/ie_p3_kaggle_launch_v1.json').read_text())
        assert report['valid'] and report['code_revision']==launch['code_revision']
        assert len(report['rows'])==48 and len(report['preflights'])==51
        assert all(x['valid'] for x in report['preflights']) and not report['graph_failures']
        assert not report['candidate_promoted'] and not report['deployment_authorized']
        assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==report['selection']['checkpoint_sha256']
        assert json.loads((folder/'selection_freeze.json').read_text())==report['selection']
        assert (folder/'preregistration.json').read_bytes()==path.read_bytes()
        for p in sorted(folder.iterdir()):
            artifacts.append(put('artifacts','result-'+p.stem,Nome='IE-P3 result '+p.name,Tipo='Report',
                Percorso=p.relative_to(repo).as_posix(),Versione='v1',**{
                    'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
        for relative in ('experiments/ie_p3_kaggle_launch_v1.json','experiments/ie_p3_result_v1.md'):
            p=repo/relative
            artifacts.append(put('artifacts',p.stem,Nome='IE-P3 '+p.name,Tipo='Report',Percorso=relative,Versione='v1',**{
                'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
        runs={name:put('runs',name,Nome='IE-P3 Kaggle v1 '+name,Stato='Completata',Braccio=[arm],**{
            'Artefatti prodotti':artifacts,'Configurazione effettiva':json.dumps(launch),
            'Validità tecnica':'All51 preflights pass,48 timing cells,zero graph failures. Scientific quality gates evaluated separately.'}) for name,arm in arms.items()}
        metric=put('metrics','wall',Nome='IE-P3 wall time per resident core call',Famiglia='Regressione',Direzione='Minimizzare',**{
            'Unità':'ms','Formula':'Median of30 synchronized blocks x20 calls, paired fixed inputs. Actual unpadded single-model inference.'})
        evaluation=put('evaluations','wall',Nome='IE-P3 resident execution cost',Metrica=[metric],Versione='v1',Ruolo='Diagnostica',
            Target='Cost only; quality independently required.',**{'Aggregazione e pesi':'Within-run paired bootstrap; not independent hardware replication.'})
        observations=[]
        for row in report['rows']:
            times=row['timings']['wall_ms'];assert len(times)==30 and all(math.isfinite(x) and x>0 for x in times)
            control=next(x for x in report['rows'] if x['mode']==row['mode'] and x['batch']==row['batch'] and x['name']=='vectorized_algebra')['timings']['wall_ms']
            assert abs(statistics.median(1-b/a for a,b in zip(control,times))-row['saving_vs_vectorized']['median_saved_fraction'])<1e-12
            family=row['name'].split('_seed')[0]
            observations.append(put('observations',f"{row['mode']}-{row['name']}-b{row['batch']}",Nome=f"IE-P3 {row['mode']} {row['name']} b{row['batch']}",
                Valore=statistics.median(times),Run=[runs[family]],**{'Specifica di valutazione':[evaluation],'Artefatti dettagliati':artifacts,'Descrizione':json.dumps(row)}))
        quality_metric=put('metrics','aperture',Nome='IE-P3 fresh atomic aperture RMSE',Famiglia='Regressione',Direzione='Minimizzare',**{'Unità':'dimensionless','Formula':'sqrt(mean((predicted-aperture minus exact-aperture)^2)) across independent fresh states and11channels'})
        quality_evaluation=put('evaluations','aperture',Nome='IE-P3 fresh atomic readout',Metrica=[quality_metric],Versione='v1',Ruolo='Primaria',Target='RMSE<=.005 and max_abs<=.03; closed-loop diagnostic gates additionally required.')
        for k,row in enumerate(report['fresh_atomic']):
            name=f'learned_width{16 if k<3 else 32}_seed{spec["seeds"][k%3]}'
            observations.append(put('observations','fresh-'+name,Nome='IE-P3 fresh '+name,Valore=row['rmse'],Run=[runs[name.split('_seed')[0]]],**{
                'Specifica di valutazione':[quality_evaluation],'Artefatti dettagliati':artifacts,
                'Descrizione':json.dumps({'fresh':row,'historical_closed_loop':report['historical_diagnostics'][name],'qualified':report['quality_qualified'][name]})}))
        put('findings','result',Nome='IE-P3 exact vectorization wins; tested learned readouts fail fidelity and cost',Esito='Misto',Osservazioni=observations,**{
            'Risultato':'Atb642 graph exactvectorized .053139ms vs legacy .102743ms:48.26% paired saving. All6 learned readouts fail quality and are87.6–108.0% slower than vectorized algebra. Best fresh apertureRMSE .02092 vs registered .005; historicalV .544–.915mV vs .001mV exact. No candidate promoted.',
            'Limitazioni':spec['exclusions']+' Allseeds select3000 updates; this suggests unresolved convergence, not proof that more training suffices. No full ionic operator tested.'})
        put('experiments','readout',Nome='IE-P3 paired exact and learned ionic readout',Stato='Concluso',Descrizione=spec['scope'],**{'Obiettivo informativo':spec['decision']})
        put('actions','execute',Nome='Execute IE-P3 single notebook with paired six-member training',Stato='Conclusa',Descrizione='Run completed, audited and archived. No new experiment authorized by result alone.')
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'preregistered':True,'airtable_accessed':False}))

if __name__=='__main__':main()
