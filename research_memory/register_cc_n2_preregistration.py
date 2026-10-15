"""Register only the approved N2 protocol/implementation; SQLite exclusive."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    path=repo/'experiments/cc_n2_unified_cell_preregistration_v1.json'
    spec=json.loads(path.read_text(encoding='utf-8'))
    assert hashlib.sha256((repo/spec['parent_report']).read_bytes()).hexdigest()==spec['parent_sha256']
    mirror=ValidatedBatchMirror()
    experiment=mirror.query("SELECT record_id FROM v_records WHERE stable_code='experiments-giada-cc-n2-v1'")['rows']
    assert len(experiment)==1
    exp=experiment[0]['record_id']
    mirror.local_upsert('experiments',{'Codice stabile':'experiments-giada-cc-n2-v1',
        'Stato':'Preregistrato','Descrizione':spec['interpretation'],
        'Obiettivo informativo':'Nine arms x three paired seeds: explicit kinetic control, latent dimension, physical readout, initialization. No sealed candidate promotion.'})
    protocol=mirror.local_upsert('protocols',{'Codice stabile':'protocols-giada-cc-n2-v1',
        'Nome':'CC-05..08 native unified ionic cell learnability matrix',
        'Esperimento':[exp],'Versione':'v1','Modalità':'Ricomposizione',
        'Procedura':json.dumps(spec,ensure_ascii=False),'Criteri di successo':spec['selection']+' '+spec['stopping'],
        'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),'Regole di arresto':spec['stopping']})['record_id']
    for task in spec['roadmap_ids']:
        mirror.local_upsert('actions',{'Codice stabile':f'actions-giada-cc-{task.lower()}-v1',
            'Stato':'Pronta','Criteri di avvio':'User approved N2 after N1 explanation. Protocol '+protocol+
            '; numeric equivalence preflight before training; native targets; no synapse competency claim.'})
    for suffix,relative in [('protocol',path.relative_to(repo).as_posix()),
        ('design','experiments/cc_n2_design_v1.md'),('runtime','src/giada_teacher/cc_n2_unified_cell.py'),
        ('runner','scripts/run_cc_n2_unified_cell.py'),('notebook','notebooks/cc_n2_unified_cell.ipynb'),
        ('local-audit','experiments/cc_n2_local_audit_v1.json')]:
        source=repo/relative
        mirror.local_upsert('artifacts',{'Codice stabile':f'artifacts-giada-cc-n2-{suffix}-v1',
            'Nome':'CC-N2 '+suffix,'Tipo':'Report','Percorso':relative,'Versione':'v1',
            'SHA-256':hashlib.sha256(source.read_bytes()).hexdigest(),'Dimensione byte':source.stat().st_size})
    launch_path=repo/'experiments/cc_n2_kaggle_launch_v1.json'
    if launch_path.is_file():
        launch=json.loads(launch_path.read_text(encoding='utf-8'))
        assert launch['run_requested'] and launch['scientific_result']=='pending'
        arm=mirror.local_upsert('arms',{'Codice stabile':'arms-giada-cc-n2-execution-v1',
            'Nome':'CC-N2 paired native and vectorized exploratory matrix',
            'Ruolo':'Oracle diagnostico','Protocollo':[protocol],
            'Configurazione residua':json.dumps(launch,ensure_ascii=False)})['record_id']
        run=mirror.local_upsert('runs',{'Codice stabile':'runs-giada-cc-n2-kaggle-v1',
            'Nome':'CC-N2 native and 27-member GPU matrix — Kaggle v1',
            'Braccio':[arm],'Stato':'In corso',
            'Hardware e ambiente':'Kaggle T4, native NEURON 8.2.7 and CUDA in distinct subprocesses',
            'Configurazione effettiva':json.dumps(launch,ensure_ascii=False),
            'Validità tecnica':'Run requested through Kaggle MCP. No scientific result at registration.'})['record_id']
        mirror.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-cc-n2-launch-v1',
            'Nome':'CC-N2 Kaggle launch v1','Tipo':'Report','Percorso':launch_path.relative_to(repo).as_posix(),
            'Versione':'v1','SHA-256':hashlib.sha256(launch_path.read_bytes()).hexdigest(),
            'Dimensione byte':launch_path.stat().st_size})
        mirror.local_upsert('experiments',{'Codice stabile':'experiments-giada-cc-n2-v1','Stato':'In corso'})
        for task in spec['roadmap_ids']:
            mirror.local_upsert('actions',{'Codice stabile':f'actions-giada-cc-{task.lower()}-v1','Stato':'In corso'})
    mirror.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'tasks':spec['roadmap_ids'],'airtable_accessed':False,'run_launched':False}))


if __name__=='__main__':main()
