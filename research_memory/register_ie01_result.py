"""Register audited CC-N2 results and the approved IE direction, SQLite only."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/cc_n2_kaggle_469661b'
    read = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    audit, report = read('result_audit.json'), read('final_report.json')
    assert audit['valid'] and audit['selection_reproduced'] and not audit['candidate_promoted']
    for name, row in audit['files'].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == row['sha256']
    mirror = ValidatedBatchMirror()
    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-ie01-{suffix}-v1', **fields})['record_id']
    artifacts = []
    for path in sorted(folder.iterdir()):
        if path.is_file():
            artifacts.append(put('artifacts', path.name.replace('.', '-'), **{
                'Nome': 'CC-N2 audited '+path.name, 'Tipo': 'Dataset' if path.suffix == '.npz' else 'Report',
                'Percorso': path.relative_to(repo).as_posix(), 'Versione': 'v1',
                'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size}))
    run = mirror.local_upsert('runs', {'Codice stabile': 'runs-giada-cc-n2-kaggle-v1',
        'Stato': 'Completata', 'Artefatti prodotti': artifacts,
        'Validità tecnica': audit['audit_scope'] + '; 108 evaluations independently recomputed, all selection minima verified.'})['record_id']
    metric = put('metrics', 'voltage', **{'Nome': 'CC-N2 autonomous pooled voltage RMSE',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'sqrt(mean((Vpred - Vnative)^2)) over time and episodes, separately per member/role/horizon.'})
    evaluation = put('evaluations', 'voltage', **{'Nome': 'CC-N2 frozen autonomous V/Ca diagnostic gate',
        'Metrica': [metric], 'Versione': 'v1', 'Ruolo': 'Diagnostica',
        'Target': 'V pooled <=0.05 mV, worst episode <=0.1 mV, worst Ca <=5e-6 mM; finite and zero violations.',
        'Aggregazione e pesi': 'All 27 members, 24 episodes per role, development/diagnostic and 8/30ms separately; no seed selection.'})
    observations = []
    for i, row in enumerate(report['results']):
        observations.append(put('observations', f'evaluation-{i:03d}', **{
            'Nome': f"CC-N2 {row['role']} {row['arm']} seed{row['seed']} {row['horizon_ms']}ms",
            'Valore': row['voltage_pooled_rmse_mv'], 'Run': [run],
            'Specifica di valutazione': [evaluation], 'Artefatti dettagliati': artifacts,
            'Descrizione': json.dumps(row, ensure_ascii=False)}))
    # Keep exploratory memory effects and learning curves as queryable observations;
    # they are not independent confirmations or primary voltage-gate measurements.
    for kind, filename, field in [('memory', 'memory_interventions.json', 'score'),
                                  ('learning', 'learning_curves.json', 'score'),
                                  ('initial-probe', 'gate_probes.json', 'development_gate_probe_rmse')]:
        for i, row in enumerate(read(filename)):
            put('observations', f'{kind}-{i:03d}', **{
                'Nome': f"CC-N2 {kind} {row['arm']} seed{row['seed']} item{i}",
                'Valore': row[field], 'Run': [run], 'Artefatti dettagliati': artifacts,
                'Descrizione': json.dumps({'diagnostic_kind': kind, **row}, ensure_ascii=False)})
    finding = put('findings', 'n2', **{'Nome': 'CC-N2 complete: explicit and compressed controls fail CC fidelity',
        'Esito': 'Negativo', 'Osservazioni': observations,
        'Risultato': 'No member passes CC gates at 8 or 30ms. Diagnostic 8ms mean voltage RMSE explicit18 0.251518mV; conductance_z8 0.620890mV. The explicit-state control also fails, so state compression alone cannot explain all errors. Selection and saved-array metrics independently verified.',
        'Limitazioni': 'Exploratory native ionic core without synapses. No sealed confirmation, speed measurement, universal impossibility or causal attribution to training budget. Initial-state probes do not certify evolved latent information.'})
    decision = put('decisions', 'ie-priority', **{'Nome': 'Prioritize measured ionic-block efficiency before whole-compartment compression',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'User-approved modular efficiency direction IE-01..16. First reconcile N2, profile costs, define ionic interface, then isolate learnability and runtime. Preserve Task35 and CC history.',
        'Condizioni di revisione': 'N3..N6 suspended; no automatic training or Task36 launch. OpenEvolve on DigitalOcean user-reported, not verified; future RunPod integration conditional on evaluator and budget.'})
    mirror.local_upsert('experiments', {'Codice stabile': 'experiments-giada-cc-n2-v1', 'Stato': 'Concluso'})
    for suffix in ('05', '06', '07', '08'):
        mirror.local_upsert('actions', {'Codice stabile': f'actions-giada-cc-cc-{suffix}-v1', 'Stato': 'Conclusa',
            'Criteri di avvio': 'Diagnostic matrix completed; CC-B fidelity not passed. No candidate promoted; result '+finding})
    for i in range(1, 17):
        mirror.local_upsert('actions', {'Codice stabile': f'actions-giada-ie-{i:02d}-v1',
            'Nome': f'IE-{i:02d} — see original dropshot VI-bis', 'Stato': 'Conclusa' if i == 1 else 'Proposta',
            'Criteri di avvio': 'Approved direction in experiments/giada_modular_surrogate_roadmap.md section VI-bis. Numeric preregistration required before each new experiment. Decision '+decision})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'airtable_accessed': False, 'n2_evaluations': len(observations),
                      'ie01_complete': True, 'new_training_performed': False}))


if __name__ == '__main__':
    main()
