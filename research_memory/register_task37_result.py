"""Register Task37 actual dataset, evaluators, controls, limits and decision."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task37_playground_v2'
    read = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    report, audit, spec, cases, manifest = [read(n) for n in ('final_report.json', 'result_audit.json', 'preregistration.json', 'cases.json', 'dataset_manifest.json')]
    assert report['valid'] and audit['valid'] and audit['target_arrays_regenerated']
    assert not report['training_performed'] and not report['confirmation_accessed']
    assert audit['trajectories_verified'] == 168 and audit['unit_tests_passed'] == 9
    assert json.loads((repo / 'experiments/task37_ionic_playground_preregistration_v1.json').read_text()) == spec
    for x in manifest:
        assert hashlib.sha256((folder / x['file']).read_bytes()).hexdigest() == x['sha256']
    mirror = ValidatedBatchMirror()
    ids = {}
    def put(table, suffix, **fields):
        code = f'{table}-giada-task37-ionic-playground-{suffix}-v1'
        rid = mirror.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        ids[code] = rid
        return rid
    def ref(code):
        rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code='" + code + "'")['rows']
        assert len(rows) == 1, code
        return rows[0]['record_id']
    def plan(number, **fields):
        code = f'actions-giada-ionic-abstraction-20261017-task{number}-v1'
        return mirror.local_upsert('actions', {'Codice stabile': code, **fields})['record_id']
    paths = [p for p in folder.iterdir() if p.name != 'memory_receipt.json'] + [
        repo / 'experiments/task37_ionic_playground_preregistration_v1.json',
        repo / 'experiments/task37_history_matching_amendment_v2.json',
        repo / 'experiments/task37_playground_design_v1.md', repo / 'experiments/task37_result_v1.md',
        repo / 'scripts/run_task37_playground.py', repo / 'scripts/audit_task37_playground.py',
        repo / 'tests/test_task37_ionic_playground.py', repo / 'src/giada_teacher/task37_ionic_playground.py',
        repo / 'research_memory/register_task37_result.py']
    # Preserve JSON history of the technical-only first materialization.
    paths += list((repo / 'experiments/results/task37_playground_v1').glob('*.json'))
    artifacts = {}
    for p in sorted(paths):
        relative = p.relative_to(repo).as_posix()
        kind = {'.npz': 'Dataset', '.py': 'Codice', '.png': 'Figura'}.get(p.suffix, 'Report')
        artifacts[relative] = put('artifacts', relative, Nome='Task37 ' + relative, Tipo=kind,
            Versione='v2' if 'playground_v2/' in relative else 'v1', Percorso=relative,
            **{'SHA-256': hashlib.sha256(p.read_bytes()).hexdigest(), 'Dimensione byte': p.stat().st_size})
    art = lambda name: artifacts[(folder / name).relative_to(repo).as_posix()]
    code_arts = [a for k, a in artifacts.items() if k.endswith('.py')]
    all_arts = list(artifacts.values())
    implementation = put('implementations', 'generator-evaluator', Nome='Task37 imposed-path generator and ionic evaluators',
        **{'Repository URL': 'https://github.com/Zagred47/giada', 'Commit': 'source hashes in code_provenance.json and result_audit.json',
           'Percorso entrypoint': 'scripts/run_task37_playground.py',
           'Ambiente': 'CPU NumPy/PyTorch; isolated matplotlib dependencies for plots. No NEURON/CUDA co-import.', 'Artefatti codice': code_arts})
    generator = put('generators', 'canonical', Nome='Task37 canonical ionic path generator', Versione='v2',
        Procedura=spec['phase'], **{'Distribuzioni input': json.dumps(spec['families']),
           'Seed e RNG': json.dumps(spec['roles']), 'Variabili registrate': spec['storage'],
           'Artefatti codice': code_arts, 'Descrizione': spec['initial_states'] + ' History pairs follow explicit v2 matching amendment.'})
    dataset = put('datasets', 'v2', Nome='Task37 ionic development dataset v2', Versione='v2',
        Fingerprint=hashlib.sha256((folder / 'dataset_manifest.json').read_bytes()).hexdigest(),
        Schema=spec['storage'], Copertura=json.dumps({k: report[k] for k in ('family_count', 'trajectory_count', 'rare_trajectory_count', 'total_biological_ms')}),
        Generatore=[generator], Artefatti=[art(x['file']) for x in manifest] + [art('dataset_manifest.json'), art('cases.json')],
        **{'Numero esempi': 168, 'Descrizione': 'Count is independent trajectory designs, not time points. 23,685,200 scalar current targets; all roles are development-accessible, no final confirmation generated.'})
    partitions = {}
    for role, info in spec['roles'].items():
        selected = [c for c in cases if c['role'] == role]
        partitions[role] = put('partitions', role, Nome='Task37 ' + role,
            Ruolo='Fit' if role in ('fit', 'fit_extra') else ('Development' if role == 'development' else 'Diagnostico'),
            Dataset=[dataset], **{'Regola di assegnazione': json.dumps(info),
                'Gruppi seed episodi snapshot': json.dumps([c['id'] for c in selected]),
                'Hash appartenenza': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                'Restrizioni di accesso': 'Not a sealed/final test. No point-wise split; complete histories and matched pairs kept together.',
                'Descrizione': f'{len(selected)} trajectories; shared protocol family is intentional, exact histories do not overlap.'})
    root_claims = [ref('claims-giada-ionic-abstraction-20261017-' + k + '-v1') for k in ('joint_update', 'rare_coverage', 'rare_compute', 'memory_sufficiency', 'behavioral_fidelity')]
    exp = put('experiments', 'suite', Nome='NEW Task37 ionic playground and evaluator validation', Stato='Concluso',
        Ipotesi=root_claims, Descrizione=spec['scope'], **{'Obiettivo informativo': spec['evaluator_acceptance']})
    base_protocol = put('protocols', 'suite', Nome='Task37 v1 numeric contract', Versione='v1',
        Modalità='Ricomposizione', Esperimento=[exp], Procedura=json.dumps(spec, ensure_ascii=False),
        **{'Hash preregistrazione': hashlib.sha256((repo / 'experiments/task37_ionic_playground_preregistration_v1.json').read_bytes()).hexdigest(),
           'Artefatto preregistrazione': [artifacts['experiments/task37_ionic_playground_preregistration_v1.json']],
           'Partizioni autorizzate': list(partitions.values()), 'Criteri di successo': spec['evaluator_acceptance'],
           'Regole di arresto': spec['limits'], 'Registrato il': '2026-10-17'})
    amendment = json.loads((repo / 'experiments/task37_history_matching_amendment_v2.json').read_text())
    protocol = put('protocols', 'history-v2', Nome='Task37 v2 matched-history clarification', Versione='v2',
        Modalità='Ricomposizione', Esperimento=[exp], Procedura=json.dumps(amendment),
        **{'Versione precedente': [base_protocol],
           'Artefatto preregistrazione': [artifacts['experiments/task37_history_matching_amendment_v2.json']],
           'Partizioni autorizzate': list(partitions.values()), 'Criteri di successo': spec['evaluator_acceptance'],
           'Regole di arresto': spec['limits'], 'Registrato il': '2026-10-17'})
    playground = put('playgrounds', 'suite', Nome='Task37 controlled ionic-memory playground',
        **{'Variabili controllate': json.dumps(spec['roles']) + spec['parameter_policy'],
           'Difficoltà': '12 regimes; 0.1ms transitions through 10s histories; rare coverage explicitly stratified.',
           'Limiti di trasferimento': spec['limits'], '10 · Esperimenti': [exp]})
    arm = put('arms', 'canonical-reference', Nome='Task37 canonical reference, storage conversion and fault controls',
        Ruolo='Baseline', Protocollo=[protocol], **{'Configurazione residua': 'No learned candidate. Numerical/storage/evaluator results only.'})
    run = put('runs', 'v2', Nome='Task37 complete v2 CPU materialization and independent audit', Stato='Completata', Braccio=[arm],
        **{'Artefatti prodotti': all_arts, 'Configurazione effettiva': json.dumps(report),
           'Validità tecnica': json.dumps(audit), 'Hardware e ambiente': json.dumps(read('code_provenance.json')['runtime'])})
    metric_definitions = {
        'storage_current_nrmse': ('dimensionless', 'sqrt(mean((stored_current-reference_current)^2))/max(max(abs(reference_current)),1e-7)', 'Minimizzare'),
        'storage_gate_rmse': ('dimensionless', 'sqrt(mean((stored_gate-reference_gate)^2))', 'Minimizzare'),
        'fault_detected': ('boolean', '1 if the registered faulty control is rejected, 0 otherwise', 'Massimizzare'),
        'history_future_contrast': ('mA/cm2', 'max abs(I_historyA-I_historyB) in shared V/Ca tail, with common initial state and parameters', 'Descrittiva')}
    evals = {}
    for key, (unit, formula, direction) in metric_definitions.items():
        # Descriptive contrasts do not have a preferred scientific optimum.
        values = {'Nome': 'Task37 ' + key, 'Formula': formula, 'Unità': unit}
        if direction != 'Descrittiva': values['Direzione'] = direction
        metric = put('metrics', key, **values)
        evals[key] = put('evaluations', key, Nome='Task37 ' + key, Versione='v1', Protocollo=[protocol],
            Metrica=[metric], Partizioni=list(partitions.values()), Ruolo='Diagnostica',
            Target='Reference/storage or registered-control result, NOT learned model performance. ' + json.dumps(spec['quality']),
            **{'Gestione classi assenti': 'Unsupported transients explicitly not_evaluable; never an emergent behavior pass.',
               'Aggregazione e pesi': 'Keep channel/trajectory/role; no pooling to waive critical failures.'})
    observations = []
    conversion = read('storage_conversion_evaluation.json')
    for row in conversion:
        details = row['storage_conversion']
        data_art = art(f"{row['role']}__{row['family']}.npz")
        case_id = row['case_id'].replace('/', '-')
        for channel in details['channels']:
            observations.append(put('observations', f"storage-{case_id}-{channel['channel']}",
                Nome=f"Task37 storage {row['case_id']} {channel['channel']}", Valore=channel['nrmse'], Run=[run],
                **{'Specifica di valutazione': [evals['storage_current_nrmse']],
                   'Artefatti dettagliati': [data_art, art('storage_conversion_evaluation.json')],
                   'Strato o sottogruppo': f"{row['role']}/{row['family']}/{channel['channel']}",
                   'Descrizione': json.dumps(channel) + ' One trajectory, not independent time samples.'}))
        observations.append(put('observations', f'storage-gate-{case_id}', Nome='Task37 storage gates ' + row['case_id'],
            Valore=details['gate']['rmse'], Run=[run], **{'Specifica di valutazione': [evals['storage_gate_rmse']],
            'Artefatti dettagliati': [data_art, art('storage_conversion_evaluation.json')], 'Descrizione': json.dumps(details['gate'])}))
    for name, result in read('evaluator_controls.json')['controls_detected'].items():
        observations.append(put('observations', 'control-' + name, Nome='Task37 detection ' + name, Valore=int(not result['passed']),
            Run=[run], **{'Specifica di valutazione': [evals['fault_detected']], 'Artefatti dettagliati': [art('evaluator_controls.json')], 'Descrizione': json.dumps(result)}))
    for i, pair in enumerate(read('matched_history_probe.json')):
        observations.append(put('observations', f'history-{i}', Nome=f'Task37 matched history pair{i}',
            Valore=pair['current_future_delta_ma_cm2'], Run=[run], **{'Specifica di valutazione': [evals['history_future_contrast']],
            'Artefatti dettagliati': [art('matched_history_probe.json')], 'Descrizione': json.dumps(pair)}))
    for name, description in [('integrity', 'All 60 NPZ hashes and arrays regenerated; 168 exact history signatures distinct.'),
                              ('phase', '512 phase/dtype points versus existing prefix; no new NEURON claim.'),
                              ('sensitivity', '8 defective output controls and actual recurrent reset detected; reference passes.'),
                              ('qualitative', 'Canonical matched-history plot visually reviewed with separate units; no model behavior judged.')]:
        put('quality', name, Nome='Task37 ' + name, Esito='Superato', Regola=description,
            Run=[run], Dataset=[dataset], Implementazioni=[implementation],
            **{'Artefatti diagnostici': [art('result_audit.json')], 'Dettagli': description})
    finding = put('findings', 'outcome', Nome='Task37 isolated suite passes; causal behavior confirmation remains separate',
        Esito='Misto', Esperimenti=[exp], Osservazioni=observations,
        **{'Risultato': (repo / 'experiments/task37_result_v1.md').read_text(encoding='utf-8'),
           'Limitazioni': spec['limits'], 'Incertezza': 'Known fixtures demonstrate limited evaluator sensitivity, not completeness on unknown regimes.'})
    # A narrow confirmed proposition does not close the broader abstraction hypotheses.
    claim = put('claims', 'known-fault-sensitivity', Nome='Task37 registered ionic faults are detected', Tipo='Ipotesi',
        Stato='Supportata nel dominio', Enunciato='The implemented suite rejects the eight registered faulty output fixtures and an actual recurrent reset while accepting the canonical reference/storage.',
        **{'Limiti': 'No learned candidate tested; no general emergent-spiking or bifurcation evaluation proven.'})
    evidence = put('evidence', 'known-fault-sensitivity', Nome='Task37 scoped evaluator evidence', Esito='Sostiene',
        **{'Affermazione valutata': [claim], 'Risultati a sostegno': [finding],
           'Argomentazione': 'Control rejection, 9 unit tests, exact target regeneration and phase preflight passed.',
           'Data valutazione': '2026-10-17', 'Limiti e spiegazioni alternative': spec['limits']})
    incident_run = put('runs', 'v1-technical', Nome='Task37 v1 missing plotting dependency', Stato='Completata',
        **{'Configurazione effettiva': (repo / 'experiments/results/task37_playground_v1/failure_report.json').read_text(),
           'Validità tecnica': 'Technical-only preflight stop after data generation; no model result or thresholds changed.'})
    put('anomalies', 'v1-plotting', Nome='Task37 initial plotting dependency absent', Stato='Spiegata', Run=[incident_run],
        Atteso='Full evaluator fixture generation', Osservato='matplotlib import failed',
        **{'Possibili cause': 'Local plotting dependency absent; isolated install and full v2 regeneration completed.'})
    action37 = plan(37, Nome='Task37 — Playground di memoria e interventi', Stato='Conclusa',
        Descrizione='Isolated imposed-V/Ca suite complete with numeric/storage/control audit; coupled behavior remains Task40/41.')
    action38 = plan(38, Nome='Task38 — Aggiornamento appreso a stato esplicito', Stato='Pronta',
        Descrizione='Ready for implementation and architecture/optimizer/GPU preflight. Training itself requires prospective Task38 protocol and all numerical checks.')
    plan(36, Nome='Task36 — Contratto e scala di astrazione ionica', Stato='Conclusa', Descrizione='Completed source/interface audit. Historical exTask36 remains Task72; no new model performance was claimed.')
    decision = put('decisions', 'proceed', Nome='Task37 GO to Task38 implementation with preflights', Esito='Continuare',
        Motivazione=report['decision'], Risultati=[finding], **{'Valutazioni delle evidenze': [evidence],
            'Data': '2026-10-17', '13 · Azioni di ricerca': [action37, action38],
            'Condizioni di revisione': 'Reference/source mismatch or new evaluator blind spot blocks training until repaired; new model behavior still judged in its own scope.'})
    print(json.dumps({'stage': 'validated_batch_ready', 'records': len(ids), 'observations': len(observations)}), flush=True)
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    verification = mirror.verify()
    assert verification['valid']
    write_json(folder / 'memory_receipt.json', dict(valid=True, task=37, records=ids, dataset=dataset,
        partitions=partitions, generator=generator, playground=playground, decision=decision,
        observations=len(observations), task37_action=action37, next_action=action38,
        sqlite_verify=verification, airtable_accessed=False, training_performed=False))
    print(json.dumps({'valid': True, 'records': len(ids), 'observations': len(observations), 'airtable_accessed': False}), flush=True)


if __name__ == '__main__':
    main()
