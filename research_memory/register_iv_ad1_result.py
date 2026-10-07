"""Register the immutable IV-A1/A2/D1 v1-v3 results in local SQLite only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    paths = [repo / f'experiments/results/iv_ad1_v{i}_final_report.json' for i in (1, 2, 3)]
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    assert reports[0]['diagnosis'] == 'PASSIVE_SINGLE_INTERFACE_NO_GO'
    assert reports[1]['diagnosis'] == 'PASSIVE_AXIAL_NO_GO'
    assert reports[2]['diagnosis'] == 'PASSIVE_AXIAL_PREREQUISITE_CONFIRMED'
    assert reports[2]['valid'] and reports[2]['iv_a1_a2_passed']
    assert reports[2]['iv_d1_opened'] and reports[2]['iv_d1_passed']
    assert reports[2]['task36_authorized'] and not reports[2]['active_compartments_tested']
    assert len(reports[2]['paired_cases']) == 12
    assert all(row['passed'] for row in reports[2]['paired_cases'])
    assert reports[2]['convergence_passed']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-ad1-{suffix}-result-v1', **fields
        })['record_id']

    artifacts = []
    for index, path in enumerate(paths, start=1):
        artifacts.append(put('artifacts', f'v{index}-report', **{
            'Nome': f'IV-A1/A2/D1 v{index} native report', 'Tipo': 'Report',
            'Percorso': path.relative_to(repo).as_posix(),
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'Dimensione byte': path.stat().st_size, 'Versione': f'v{index}'}))
    run = put('runs', 'v3-new-geometries', **{
        'Nome': 'IV-A1/A2/D1 passive native confirmation v3',
        'Stato': 'Completata', 'Artefatti prodotti': artifacts,
        'Validità tecnica': 'V1 and v2 diagnostic failures preserved; v3 new geometry matrix was preregistered before Kaggle native execution; no active model evaluated.'})
    metric = put('metrics', 'native-discrete-voltage', **{
        'Nome': 'IV-D1 native versus independent discrete solver max absolute voltage error',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'Maximum absolute difference across all steps, sites, geometries and protocols.'})
    evaluation = put('evaluations', 'v3-axial-matrix', **{
        'Nome': 'IV-D1 prospective three-geometry by four-protocol native matrix',
        'Metrica': [metric], 'Versione': 'v3', 'Ruolo': 'Primaria',
        'Target': 'Every row passes preregistered native/discrete, current balance, negative controls; independent equilibrium and dt convergence pass.',
        'Aggregazione e pesi': 'No averaging hides a failing geometry/protocol; 12 independent cases.'})
    maximum = max(row['max_native_discrete_error_mv'] for row in reports[2]['paired_cases'])
    observation = put('observations', 'v3-axial-max-error', **{
        'Nome': 'IV-D1 v3 maximal native/discrete voltage discrepancy',
        'Valore': maximum, 'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': artifacts,
        'Descrizione': f'12/12 rows passed; maximum {maximum:.9g} mV; equilibrium native error {reports[2]["equilibrium_native_error_mv"]:.9g} mV; convergence passed.'})
    finding = put('findings', 'passive-axial-confirmed', **{
        'Nome': 'Passive two-compartment axial interface confirmed; Task36 entry open',
        'Esito': 'Positivo', 'Osservazioni': [observation],
        'Risultato': 'IV-A1/A2 passed. IV-D1 passed in 12/12 new geometry/protocol cases against native NEURON; capacitive and axial current balance, disconnected and equal-current-density symmetric controls, equilibrium and temporal convergence passed. V1 failed a continuous-time fixed-step gate, v2 failed an unequal-area equal-absolute-current control; both prior reports remain immutable.',
        'Limitazioni': 'Passive two-node interface only; no active ion channels, calcium, synapses, surrogate models, larger trees, full cell or speedup claim.'})
    put('decisions', 'task36-entry', **{
        'Nome': 'IV-D1 gate passed; authorize separate active two-compartment Task36',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'Prospective v3 new geometries passed unchanged numeric tolerances with corrected equal-density physical control. The passive prerequisite is complete.',
        'Condizioni di revisione': 'Task36 must have its own frozen active-channel/synapse protocol and independent model assessment; IV-D1 is not Task36.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_d1_passed': True,
                      'task36_authorized': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
