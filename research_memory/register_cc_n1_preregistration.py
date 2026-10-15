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
        ('n1-progress', 'experiments/cc_n1_progress_v1.md'),
        ('n1-result', 'experiments/cc_n1_result_v1.md'),
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
    amendment = repo / 'experiments/cc_n1_native_sampling_amendment_v1.json'
    if amendment.is_file():
        changed = json.loads(amendment.read_text())
        for relative, expected in changed['prior_sources'].items():
            assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected
        amended_protocol = mirror.local_upsert('protocols', {
            'Codice stabile': 'protocols-giada-cc-n1-native-sampling-v1',
            'Nome': 'CC-N1 native current sampling amendment; thresholds unchanged',
            'Esperimento': [experiment], 'Versione': 'v1-native-sampling-amendment',
            'Modalità': 'Ricomposizione', 'Procedura': json.dumps(changed, ensure_ascii=False),
            'Criteri di successo': changed['decision'],
            'Hash preregistrazione': hashlib.sha256(amendment.read_bytes()).hexdigest(),
            'Regole di arresto': changed['reuse_policy']})['record_id']
        primary_path = repo / 'experiments/cc_n1_native_memory_primary_amendment_v2.json'
        native_protocol = amended_protocol
        primary = None
        if primary_path.is_file():
            primary = json.loads(primary_path.read_text())
            native_protocol = mirror.local_upsert('protocols', {
                'Codice stabile': 'protocols-giada-cc-n1-native-memory-primary-v2',
                'Nome': 'CC-N1 direct native memory matrix; analytic floor retained',
                'Esperimento': [experiment], 'Versione': 'v2-native-memory-primary',
                'Modalità': 'Ricomposizione', 'Procedura': json.dumps(primary, ensure_ascii=False),
                'Criteri di successo': primary['decision'],
                'Hash preregistrazione': hashlib.sha256(primary_path.read_bytes()).hexdigest(),
                'Regole di arresto': primary['interpretation']})['record_id']
        gpu_protocol = native_protocol
        gpu_amendment_path = repo / 'experiments/cc_n1_gpu_compile_amendment_v1.json'
        if gpu_amendment_path.is_file():
            gpu_amendment = json.loads(gpu_amendment_path.read_text())
            gpu_protocol = mirror.local_upsert('protocols', {
                'Codice stabile': 'protocols-giada-cc-n1-gpu-compile-v1',
                'Nome': 'CC-N1 fullgraph metadata fix and verified native reuse',
                'Esperimento': [experiment], 'Versione': 'v1-gpu-compile', 'Modalità': 'Ricomposizione',
                'Procedura': json.dumps(gpu_amendment, ensure_ascii=False),
                'Criteri di successo': gpu_amendment['decision'],
                'Hash preregistrazione': hashlib.sha256(gpu_amendment_path.read_bytes()).hexdigest(),
                'Regole di arresto': gpu_amendment['reuse_policy']})['record_id']
        failed_observations = []
        result_runs, result_artifacts = {}, {}
        for version, status, effective_protocol in [(1, 'Fallita', protocol),
                                                     (2, 'Fallita' if primary else 'In corso', amended_protocol),
                                                     (3, 'Completata', native_protocol),
                                                     (4, 'In corso', gpu_protocol)]:
            launch_path = repo / f'experiments/cc_n1_kaggle_launch_v{version}.json'
            if not launch_path.is_file(): continue
            launch = json.loads(launch_path.read_text())
            produced = []
            if version == 1:
                paths = [launch_path, *[repo / p for p in changed['prior_sources']]]
            elif version == 2:
                paths = [launch_path, amendment] + ([repo / p for p in primary['prior_sources']] if primary else [])
            else:
                paths = [launch_path, amendment, primary_path]
                result_roots = list((repo / 'experiments/results').glob(f'cc_n1_v{version}_kaggle_*'))
                if result_roots:
                    assert len(result_roots) == 1
                    paths += [p for p in sorted(result_roots[0].iterdir()) if p.is_file()]
                    if (result_roots[0] / 'final_report.json').is_file(): status = 'Completata'
                if version == 4: paths.append(gpu_amendment_path)
            for source in paths:
                produced.append(mirror.local_upsert('artifacts', {
                    'Codice stabile': f'artifacts-giada-cc-n1-run{version}-' + source.stem.replace('_', '-') + '-v1',
                    'Nome': 'CC-N1 '+source.stem, 'Tipo': 'Dataset' if source.suffix == '.npz' else 'Report',
                    'Percorso': source.relative_to(repo).as_posix(), 'Versione': 'v'+str(version),
                    'SHA-256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'Dimensione byte': source.stat().st_size})['record_id'])
            produced = list(dict.fromkeys(produced))
            arm = mirror.local_upsert('arms', {
                'Codice stabile': f'arms-giada-cc-n1-execution-v{version}',
                'Nome': f'CC-N1 integrated diagnostic execution v{version}',
                'Ruolo': 'Oracle diagnostico', 'Protocollo': [effective_protocol],
                'Configurazione residua': json.dumps(launch, ensure_ascii=False)})['record_id']
            run = mirror.local_upsert('runs', {
                'Codice stabile': f'runs-giada-cc-n1-kaggle-v{version}',
                'Nome': f'CC-N1 Kaggle version {version}', 'Stato': status,
                'Braccio': [arm], 'Artefatti prodotti': produced,
                'Hardware e ambiente': 'Kaggle NvidiaTeslaT4; isolated NEURON 8.2.7 and CUDA processes',
                'Configurazione effettiva': json.dumps(launch, ensure_ascii=False),
                'Validità tecnica': ('Native formula-memory floor failed before CUDA; initial gate states exact, two calcium floors above unchanged limit. No learned model judged.'
                                    if status == 'Fallita' else 'Native memory matrix and fresh synaptic baseline valid; v3 compiled cost failed symbolic NumPy metadata index; v4 verifies fixed compilation. See per-phase reports, not process completion alone.')})['record_id']
            result_runs[version], result_artifacts[version] = run, produced
            anchor_path = repo / f'experiments/results/cc_n1_v{version}_native_memory_anchors.json'
            if status == 'Fallita' and anchor_path.is_file():
                raw = json.loads(anchor_path.read_text())
                for variable, unit, key in [('voltage', 'mV', 'voltage_rmse_mv'),
                                             ('calcium', 'mM', 'cai_rmse_mM')]:
                    metric = mirror.local_upsert('metrics', {
                        'Codice stabile': f'metrics-giada-cc-n1-anchor-{variable}-v1',
                        'Nome': f'Native versus analytic memory continuation {variable} RMSE',
                        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': unit,
                        'Formula': f'RMSE {variable} over 10ms native/analytic continuation from identical imported 18 gate states, V/Ca, parameters and future current.'})['record_id']
                    evaluation = mirror.local_upsert('evaluations', {
                        'Codice stabile': f'evaluations-giada-cc-n1-anchor-{variable}-v1',
                        'Nome': f'CC-N1 {variable} memory-map floor', 'Metrica': [metric],
                        'Versione': 'v1', 'Ruolo': 'Diagnostica',
                        'Target': 'Unchanged original native/analytic floor: voltage per-episode <=0.5mV; calcium <=1e-6mM. Failure does not judge a learned model.',
                        'Aggregazione e pesi': 'Per intervention and version; no averaging away failed calcium cases.'})['record_id']
                    for row in raw['rows']:
                        failed_observations.append(mirror.local_upsert('observations', {
                            'Codice stabile': f'observations-giada-cc-n1-native-v{version}-{row["row_index"]}-{variable}-v1',
                            'Nome': f'CC-N1 native v{version} row {row["row_index"]} {variable}',
                            'Valore': row[key], 'Run': [run], 'Specifica di valutazione': [evaluation],
                            'Artefatti dettagliati': produced, 'Descrizione': json.dumps(row, ensure_ascii=False)})['record_id'])
        if primary:
            mirror.local_upsert('findings', {
                'Codice stabile': 'findings-giada-cc-n1-sampling-cause-not-supported-v1',
                'Nome': 'Per-tick current sampling did not remove the off-state analytic/native discrepancy',
                'Esito': 'Negativo', 'Osservazioni': failed_observations,
                'Risultato': primary['finding'],
                'Limitazioni': 'Two reopened eight-case native diagnostic runs. Specific sampling hypothesis not supported; not a universal diagnosis of all numerical orderings. No learned network tested.'})
        native_result = repo / 'experiments/results/cc_n1_v3_kaggle_9b6a581/memory_analysis.json'
        if native_result.is_file():
            memory = json.loads(native_result.read_text())
            assert memory['valid'] and memory['collision_target_source'] == 'canonical_NEURON_direct_continuations'
            obs = []
            for variable, key in [('voltage', 'voltage_endpoint_separation_mv'),
                                  ('calcium', 'calcium_endpoint_separation_mM')]:
                metric = mirror.local_upsert('metrics', {
                    'Codice stabile': f'metrics-giada-cc-n1-collision-{variable}-v1',
                    'Nome': f'Endpoint {variable} separation for identical causal view',
                    'Famiglia': 'Regressione', 'Direzione': 'Descrittiva',
                    'Unità': 'mV' if variable == 'voltage' else 'mM',
                    'Formula': f'max native {variable}(h)-min native {variable}(h); fixed panel, tail, current; minimax lower bound is separation/2.'})['record_id']
                evaluation = mirror.local_upsert('evaluations', {
                    'Codice stabile': f'evaluations-giada-cc-n1-native-collision-{variable}-v1',
                    'Nome': f'CC-N1 native {variable} collision matrix', 'Metrica': [metric],
                    'Versione': 'v1', 'Ruolo': 'Primaria',
                    'Target': 'Positive separation under exactly identical visible V/Ca and future forcing refutes that view. Past clamp-current histories are NOT included in the matched V/Ca view.',
                    'Aggregazione e pesi': 'Per tail x panel x future current x horizon. Each group has 12 histories; no aggregate sufficiency claim.'})['record_id']
                for i, row in enumerate(memory['collisions']):
                    obs.append(mirror.local_upsert('observations', {
                        'Codice stabile': f'observations-giada-cc-n1-collision-{variable}-{i:03d}-v1',
                        'Nome': f'CC-N1 native collision {variable} group {i:03d}',
                        'Valore': row[key], 'Run': [result_runs[3]],
                        'Specifica di valutazione': [evaluation], 'Artefatti dettagliati': result_artifacts[3],
                        'Descrizione': json.dumps(row, ensure_ascii=False)})['record_id'])
            finding = mirror.local_upsert('findings', {
                'Codice stabile': 'findings-giada-cc-n1-native-memory-witness-v1',
                'Nome': 'Native compartment futures differ under identical instantaneous and recent V/Ca views',
                'Esito': 'Positivo', 'Osservazioni': obs,
                'Risultato': '288 native continuations; 60/96 tail/panel/current/horizon groups exceed registered detectability. Max V endpoint separation 3.26397mV; even a 200ms common V/Ca tail leaves a tested 1.47787mV separation (sodium_x4 active current). Native duplicate and hidden-state transplantation are exact. Physical memory must be retained or inferred with a separately validated interface.',
                'Limitazioni': 'Controlled gate-state interventions from analytic V/Ca clamp histories; not prevalence in physiological trajectories. Matched view contains V/Ca history only, not past clamp current or all possible observations. Does not identify minimum latent dimension or prove a neural cell learnable. Analytic memory floor remains failed in some cases.'})['record_id']
            claim = mirror.local_upsert('claims', {
                'Codice stabile': 'claims-giada-cc-n1-v-ca-history-insufficient-v1',
                'Nome': 'Instantaneous V/Ca and tested fixed recent V/Ca windows are insufficient',
                'Tipo': 'Ipotesi', 'Stato': 'Supportata nel dominio',
                'Enunciato': 'In the controlled history-intervention domain, a deterministic predictor using only instantaneous V/Ca or the tested identical recent V/Ca windows cannot reproduce every native future within an error smaller than half the matched endpoint separation.',
                'Condizioni di falsificazione': 'Witness would be invalid if visible V/Ca or physical parameters/future current differed, native intervention import failed, or native replay were inconsistent.',
                'Limiti': 'Specific observation interface and history family only. No lower bound for models with persistent latent memory or richer input history.'})['record_id']
            mirror.local_upsert('evidence', {
                'Codice stabile': 'evidence-giada-cc-n1-native-memory-witness-v1',
                'Nome': 'Native matched-pair evidence for V/Ca-view insufficiency',
                'Esito': 'Sostiene', 'Affermazione valutata': [claim],
                'Risultati a sostegno': [finding],
                'Argomentazione': 'Exactly equal visible views but different native endpoint targets give an explicit deterministic-prediction lower bound D/2. No inference from a failed decoder is used.',
                'Limiti e spiegazioni alternative': 'The native witness bypasses the inadmissible off-state analytic-map floor; past clamp-current history is outside the matched view.',
                'Data valutazione': '2026-10-15'})
        latest_roots = list((repo / 'experiments/results').glob('cc_n1_v4_kaggle_*'))
        latest_report = json.loads((latest_roots[0] / 'final_report.json').read_text()) \
            if latest_roots and (latest_roots[0] / 'final_report.json').is_file() else None
        complete = bool(latest_report and latest_report['valid'])
        if complete:
            assert latest_report['code_revision'] == 'dc175f9d47d8a7f974f4c6a19a0468f89a22f6a6'
            assert not latest_report['training_performed'] and not latest_report['speedup_claim']
            assert latest_report['cc02']['cost_map_valid']
            gpu = json.loads((latest_roots[0] / 'gpu_report.json').read_text())
            assert gpu['valid'] and not gpu['compile_failures'] and len(gpu['rows']) == 18
            cost_observations = []
            for suffix, key, unit in [('wall-latency', 'median_wall_ms_per_step_ensemble', 'ms'),
                                      ('normalized-latency', 'ns_per_compartment_step', 'ns'),
                                      ('parameters', 'parameter_count_per_seed', 'parameters'),
                                      ('state-size', 'persistent_core_state_values_per_compartment', 'scalar values')]:
                metric = mirror.local_upsert('metrics', {
                    'Codice stabile': f'metrics-giada-cc-n1-final-{suffix}-v1',
                    'Nome': 'CC-N1 '+suffix, 'Famiglia': 'Regressione',
                    'Direzione': 'Descrittiva', 'Unità': unit,
                    'Formula': key + '; report exact seed count, batch, dtype and core exclusions; three seeds vectorized.'})['record_id']
                evaluation = mirror.local_upsert('evaluations', {
                    'Codice stabile': f'evaluations-giada-cc-n1-final-{suffix}-v1',
                    'Nome': 'CC-N1 matched '+suffix, 'Metrica': [metric],
                    'Versione': 'v1', 'Ruolo': 'Diagnostica',
                    'Target': 'Map baseline costs; no speedup promotion. Native event/RNG generation is outside resident-core timing.',
                    'Aggregazione e pesi': 'Separate arm, batch and eager/compiled; 20 synchronized timing blocks of 10 steps, median per ensemble step.'})['record_id']
                for row in gpu['rows']:
                    label = f'{row["arm"]}-{row["mode"]}-b{row["batch_per_seed"]}'
                    cost_observations.append(mirror.local_upsert('observations', {
                        'Codice stabile': f'observations-giada-cc-n1-{suffix}-{label}-v1',
                        'Nome': f'CC-N1 {suffix} {label}', 'Valore': row[key],
                        'Run': [result_runs[4]], 'Specifica di valutazione': [evaluation],
                        'Artefatti dettagliati': result_artifacts[4],
                        'Descrizione': json.dumps(row, ensure_ascii=False)})['record_id'])
            cost_finding = mirror.local_upsert('findings', {
                'Codice stabile': 'findings-giada-cc-n1-compiled-baseline-cost-v1',
                'Nome': 'Current learned-rate hybrids do not reduce compiled membrane-core latency',
                'Esito': 'Negativo', 'Osservazioni': cost_observations,
                'Risultato': 'At B=642 per seed, compiled ensemble-step median: formula 0.4238351ms, independent 0.80748165ms (6260 weights/seed), shared_heads 0.90410175ms (644 weights/seed). Three seeds per arm; current state remains 20 scalars (V,Ca,18 gates), excluding synaptic state. GPU profiler CUDA event counts 18/22/23 are diagnostic. Parameter compression does not establish compute compression.',
                'Limitazioni': 'Kaggle Tesla T4/PyTorch2.11 FP32, resident core and synchronized wall timing. Host event/RNG generation excluded. No whole-neuron/end-to-end speed claim; no new architecture trained. Uncertainty/material-speed gate reserved for CC-18/19.'})['record_id']
            h8 = mirror.query("SELECT record_id FROM v_records WHERE table_key='claims' AND stable_code='claims-giada-cc-h8-v1'")['rows'][0]['record_id']
            mirror.local_upsert('evidence', {
                'Codice stabile': 'evidence-giada-cc-n1-cost-vs-new-compression-h8-v1',
                'Nome': 'Baseline cost does not yet decide the future aggregated architecture hypothesis',
                'Esito': 'Indeterminata', 'Affermazione valutata': [h8],
                'Risultati a sostegno': [cost_finding],
                'Argomentazione': 'N1 measures existing hybrids and formulas; future compressed-cell H8 has not been tested. Existing shared-head parameter savings did not yield the desired latency improvement.',
                'Limiti e spiegazioni alternative': 'Cannot extrapolate the cost of a future recurrent cell from the current rate-module architecture.'})
            # The staged native finding may not yet be in SQLite; resolve its deterministic staged ID.
            native_id = mirror.local_upsert('findings', {
                'Codice stabile': 'findings-giada-cc-n1-native-memory-witness-v1',
                'Nome': 'Native compartment futures differ under identical instantaneous and recent V/Ca views'})['record_id']
            mirror.local_upsert('decisions', {
                'Codice stabile': 'decisions-giada-cc-n1-complete-prepare-n2-v1',
                'Nome': 'CC-N1 complete; prepare unified cell and persistent memory tests',
                'Esito': 'Continuare', 'Data': '2026-10-15',
                'Risultati': [native_id, cost_finding],
                'Motivazione': 'Native collisions establish a need for memory in the tested V/Ca observation interface; existing rate hybrids are accurate but slower than compiled formulas. N2 can now distinguish sharing from state compression with explicit-state control, latent ladder, physical-bias contrasts and initialization tests.',
                'Descrizione': 'Prepare CC-05..08; no N2 launch or architecture winner authorized by N1. Preserve off-state analytic floor failures and native targets as primary in those conditions.',
                'Condizioni di revisione': 'Preregister N2 data/support/capacity/budget and matched controls before training. Do not infer minimum latent dimension or whole-neuron speedup.'})
        mirror.local_upsert('experiments', {
            'Codice stabile': 'experiments-giada-cc-n1-v1', 'Stato': 'Concluso' if complete else 'In corso'})
        for identifier in spec['roadmap_ids']:
            mirror.local_upsert('actions', {
                'Codice stabile': f'actions-giada-cc-{identifier.lower()}-v1',
                'Stato': 'Conclusa' if complete or identifier != 'CC-02' and native_result.is_file() else 'In corso'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'preregistered_tasks': spec['roadmap_ids'],
                      'artifacts': len(artifacts), 'airtable_accessed': False,
                      'experiment_launched_by_registration': False}))


if __name__ == '__main__':
    main()
