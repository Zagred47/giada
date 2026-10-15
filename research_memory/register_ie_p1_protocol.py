"""Register the prepared IE-P1 profile and historical N2 reference observations."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec_path = repo/'experiments/ie_p1_ionic_cost_preregistration_v1.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    mirror = ValidatedBatchMirror()
    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-ie-p1-{suffix}-v1', **fields})['record_id']
    experiment = put('experiments', 'profile', **{'Nome': 'IE-P1 resident ionic gate-cost intervention',
        'Stato': 'Preregistrato', 'Descrizione': spec['scope'], 'Obiettivo informativo': spec['decision']})
    protocol = put('protocols', 'profile', **{'Nome': 'IE-P1 paired GPU profile', 'Esperimento': [experiment],
        'Versione': 'v1', 'Modalità': 'Ricomposizione', 'Procedura': json.dumps(spec,ensure_ascii=False),
        'Criteri di successo': spec['decision'], 'Regole di arresto': spec['failure'],
        'Hash preregistrazione': hashlib.sha256(spec_path.read_bytes()).hexdigest()})
    for relative in ('experiments/giada_modular_surrogate_roadmap.md', 'experiments/cc_n2_result_v1.md',
                     'experiments/ie_p1_ionic_cost_preregistration_v1.json', 'notebooks/ie_p1_ionic_cost.ipynb',
                     'src/giada_teacher/ie_ionic_cost.py', 'scripts/run_ie_p1_ionic_cost.py'):
        path = repo/relative
        put('artifacts', path.stem.replace('_','-'), **{'Nome': 'IE direction / profile '+path.name,
            'Tipo': 'Report', 'Percorso': relative, 'Versione': 'v1',
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size})
    for i in (2,3,4):
        mirror.local_upsert('actions', {'Codice stabile': f'actions-giada-ie-{i:02d}-v1',
            'Stato': 'Pronta', 'Criteri di avvio': 'IE-P1 protocol '+protocol+
            '; resident core only. Whole-simulator exclusions and IE-05 remain open.'})
    prior = mirror.query("SELECT record_id FROM v_records WHERE stable_code='runs-giada-cc-n2-kaggle-v1'")['rows'][0]['record_id']
    refs = json.loads((repo/'experiments/results/cc_n2_kaggle_469661b/baseline_evaluation.json').read_text())
    for i,row in enumerate(refs):
        put('observations', f'n2-reference-{i:03d}', **{'Nome': f"CC-N2 reference {row['arm']} {row['seed']} {row['role']} {row['horizon_ms']}ms",
            'Run': [prior], 'Valore': row['voltage_pooled_rmse_mv'],
            'Descrizione': json.dumps({'diagnostic_kind':'historical_reference',**row},ensure_ascii=False)})
    launch_path = repo/'experiments/ie_p1_kaggle_launch_v1.json'
    if launch_path.is_file():
        launch = json.loads(launch_path.read_text(encoding='utf-8'))
        assert launch['run_requested']
        arm = put('arms', 'execution', **{'Nome': 'IE-P1 frozen resident cost matrix',
            'Ruolo': 'Oracle diagnostico', 'Protocollo': [protocol],
            'Configurazione residua': json.dumps(launch,ensure_ascii=False)})
        put('runs', 'kaggle', **{'Nome': 'IE-P1 Kaggle v1', 'Braccio': [arm], 'Stato': 'In corso',
            'Configurazione effettiva': json.dumps(launch,ensure_ascii=False),
            'Validità tecnica': 'Run requested; numerical/compile preflights must pass before timing interpretation.'})
        put('artifacts', 'launch', **{'Nome': 'IE-P1 launch receipt', 'Tipo': 'Report',
            'Percorso': launch_path.relative_to(repo).as_posix(), 'Versione': 'v1',
            'SHA-256': hashlib.sha256(launch_path.read_bytes()).hexdigest(), 'Dimensione byte': launch_path.stat().st_size})
        mirror.local_upsert('experiments',{'Codice stabile':'experiments-giada-ie-p1-profile-v1','Stato':'In corso'})
        mirror.local_upsert('actions',{'Codice stabile':'actions-giada-ie-02-v1','Stato':'In corso'})
    mirror.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'protocol_prepared':True,'launch_registered':launch_path.is_file(),'airtable_accessed':False}))


if __name__ == '__main__':
    main()
