"""Register CC-N1 protocol and reviewable implementation in SQLite only."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/cc_n1_contract_memory_preregistration_v1.json'
    spec = json.loads(path.read_text(encoding='utf-8'))
    assert spec['roadmap_ids'] == ['CC-01', 'CC-02', 'CC-03', 'CC-04']
    for relative, expected in spec['sources'].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected
    mirror = ValidatedBatchMirror()
    experiment_rows = mirror.query(
        "SELECT record_id FROM v_records WHERE table_key='experiments' "
        "AND stable_code='experiments-giada-cc-n1-v1'")['rows']
    assert len(experiment_rows) == 1
    experiment = experiment_rows[0]['record_id']
    mirror.local_upsert('experiments', {
        'Codice stabile': 'experiments-giada-cc-n1-v1',
        'Stato': 'Preregistrato',
        'Descrizione': 'CC-01..04 approved by user. Full versioned notebook, native/CUDA subprocess runner and numeric preregistration prepared. No optimizer training. Task35 is historical diagnostic support; new native cases only anchor the numerical interface. Local analytic and CPU torch checks are implementation preflights, not independent scientific confirmation.'})
    artifacts = []
    for suffix, relative in (
        ('dropshot', 'experiments/giada_compartment_compression_dropshot_v1.md'),
        ('n1-preregistration', path.relative_to(repo).as_posix()),
        ('n1-notebook', 'notebooks/cc_n1_contract_memory.ipynb'),
        ('n1-runner', 'scripts/run_cc_n1_contract_memory.py'),
        ('n1-runtime', 'src/giada_teacher/cc_n1_contract_memory.py'),
        ('n1-local-audit', 'experiments/cc_n1_local_implementation_audit_v1.json')):
        artifact = repo / relative
        artifacts.append(mirror.local_upsert('artifacts', {
            'Codice stabile': f'artifacts-giada-cc-{suffix}-v1',
            'Nome': f'GIADA CC {suffix}', 'Tipo': 'Report',
            'Percorso': relative, 'Versione': 'v1',
            'SHA-256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'Dimensione byte': artifact.stat().st_size})['record_id'])
    protocol = mirror.local_upsert('protocols', {
        'Codice stabile': 'protocols-giada-cc-n1-v1',
        'Nome': 'CC-N1 contract, cost and controlled memory matrix',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Source/native-floor failure blocks GPU judgment. Compile failure produces incomplete cost verdict. Diagnostics and failures always saved. No training or candidate promotion in N1.'})['record_id']
    for identifier in spec['roadmap_ids']:
        mirror.local_upsert('actions', {
            'Codice stabile': f'actions-giada-cc-{identifier.lower()}-v1',
            'Stato': 'Pronta',
            'Criteri di avvio': 'User approved N1; versioned protocol and matching code/notebook frozen before Kaggle launch. Native phase precedes GPU; same-process NEURON/CUDA forbidden. Protocol record '+protocol})
    for suffix, name, unit, formula in (
        ('collision-voltage', 'Endpoint separation for identical causal view', 'mV',
         'max_j V_j(h)-min_j V_j(h), within fixed panel/tail/future current and identical initial V/Ca; deterministic minimax error lower bound is separation/2.'),
        ('collision-calcium', 'Calcium endpoint separation for identical causal view', 'mM',
         'max_j Ca_j(h)-min_j Ca_j(h), within exactly matched visible inputs and forcing; separation/2 lower-bounds deterministic minimax endpoint error.'),
        ('resident-latency', 'Resident membrane-core wall latency', 'ms',
         'Median synchronized block-wall time / 10 autonomous core steps; three seeds vectorized; excludes synaptic RNG/event generation. Batch/compilation scope reported explicitly.'),
        ('native-floor', 'Native versus discrete formula voltage floor', 'mV',
         'Pooled and worst episode RMSE of voltage under identical current, conductance panel and physical-state intervention.')):
        mirror.local_upsert('metrics', {
            'Codice stabile': f'metrics-giada-cc-n1-{suffix}-v1', 'Nome': name,
            'Famiglia': 'Regressione', 'Direzione': 'Descrittiva', 'Unità': unit,
            'Formula': formula})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'preregistered_tasks': spec['roadmap_ids'],
                      'artifacts': len(artifacts), 'airtable_accessed': False,
                      'experiment_launched_by_registration': False}))


if __name__ == '__main__':
    main()
