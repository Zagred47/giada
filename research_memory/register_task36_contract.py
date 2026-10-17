"""Register new roadmap Task36, its future plan and reconciled IE-P7 locally."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec = json.loads((repo / 'experiments/task36_ionic_abstraction_contract_v1.json').read_text(encoding='utf-8'))
    folder = repo / 'experiments/results/task36_ionic_contract_v1'
    audit = json.loads((folder / 'contract_audit.json').read_text(encoding='utf-8'))
    p7folder = repo / 'experiments/results/ie_p7_kaggle_v4'
    p7 = json.loads((p7folder / 'final_report.json').read_text())
    p7audit = json.loads((p7folder / 'result_audit.json').read_text())
    assert audit['valid'] and p7audit['valid'] and audit['decision'] == 'GO_TASK37_PREPARATION_ONLY'
    for path, info in audit['sources'].items():
        assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == info['sha256']
    mirror = ValidatedBatchMirror()
    registered = {}
    def put(table, suffix, **fields):
        code = f'{table}-giada-ionic-abstraction-20261017-{suffix}-v1'
        rid = mirror.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        registered[code] = rid
        return rid
    def p7put(table, suffix, **fields):
        code = f'{table}-giada-ie-p7-{suffix}-v1'
        rid = mirror.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        registered[code] = rid
        return rid
    arts, artifact_ids = [], {}
    artifact_paths = sorted(p7folder.iterdir()) + [repo / p for p in audit['sources']] + [
        folder / 'contract_audit.json', repo / 'experiments/task36_result_v1.md',
        repo / 'experiments/ie_p7_result_v1.md', repo / 'experiments/giada_task_number_crosswalk.md',
        repo / 'research_memory/register_task36_contract.py']
    for p in dict.fromkeys(artifact_paths):
        relative = p.relative_to(repo).as_posix()
        artifact_ids[relative] = put('artifacts', relative, Nome='Task36 evidence ' + relative, Tipo='Report',
            Versione='v1', Percorso=relative, **{'SHA-256': hashlib.sha256(p.read_bytes()).hexdigest(), 'Dimensione byte': p.stat().st_size})
        arts.append(artifact_ids[relative])
    models = []
    for level, state in [(1, '18 physical gates'), (2, 'local latent states per ionic family'), (3, 'one local shared ionic latent state')]:
        models.append(put('models', f'level{level}', Nome=f'Planned ionic abstraction level {level}', Versione='planned-v1', Ruolo='Surrogate',
            Descrizione='Proposed representation, not a trained or validated instance.',
            **{'Contratto di stato': state + '; separate state per physical compartment', 'Contratto di input e output': json.dumps(spec['interface'], ensure_ascii=False)}))
    claims = []
    for h in spec['hypotheses']:
        claims.append(put('claims', h['id'], Nome='Ionic abstraction ' + h['id'], Tipo='Ipotesi', Stato='Aperta',
            Enunciato=h['statement'], Descrizione=h['contrast'],
            **{'Condizioni di falsificazione': h['falsifier'], 'Modelli oggetto': models,
               'Limiti': 'Proposed hypothesis; no causal confirmation from Task36 documentary audit.'}))
    exp = put('experiments', 'task36-contract', Nome='NEW Task36 ionic interface and evidence contract audit', Stato='Concluso',
        Ipotesi=claims, Descrizione=spec['mode'], **{'Obiettivo informativo': spec['future_run_gate']['task36_completion']})
    contract_path = repo / 'experiments/task36_ionic_abstraction_contract_v1.json'
    protocol = put('protocols', 'task36-contract', Nome='Task36 source audit and future evaluation contract', Versione='v1',
        Modalità='Ricomposizione', Esperimento=[exp], Procedura=json.dumps(spec, ensure_ascii=False),
        **{'Hash preregistrazione': hashlib.sha256(contract_path.read_bytes()).hexdigest(),
           'Registrato il': '2026-10-17', 'Criteri di successo': spec['future_run_gate']['task36_completion'],
           'Regole di arresto': 'No new training before Task37 support, evaluators, prospective thresholds and defective-control preflight. Task36 is source audit, not a model performance test.',
           'Artefatto preregistrazione': [artifact_ids[contract_path.relative_to(repo).as_posix()]]})
    evaluations = []
    for e in spec['evaluation_requirements']:
        metric = put('metrics', e['id'], Nome='Ionic contract ' + e['id'], Descrizione=e['metric'],
            **{'Unità': e['unit'], 'Parametri': 'Planned operational definition; exact thresholds/windows must be preregistered in Task37.',
               'Casi degeneri': 'Absent events or inadequate support => not evaluable; near-zero current requires absolute scale, not unstable relative division.'})
        evaluations.append(put('evaluations', e['id'], Nome='Task36 planned evaluation ' + e['id'], Versione='planned-v1',
            Metrica=[metric], Protocollo=[protocol], Ruolo='Diagnostica' if e['id'] == 'qualitative' else 'Primaria',
            Target='Specification only: numeric pass thresholds pending Task37; not an executable validated evaluator.',
            **{'Popolazione e regioni': 'Per ionic channel and regime; single then true coupled compartments',
               'Aggregazione e pesi': e['aggregation'], 'Gestione classi assenti': 'Not evaluable, never silent pass'}))
    actions = {}
    for number, title in audit['roadmap_actions'].items():
        n = int(number)
        deps = [actions[str(n-1)]] if str(n-1) in actions else []
        actions[number] = put('actions', f'task{n}', Nome=f'Task{n} — {title}',
            Stato='Conclusa' if n == 36 else 'Proposta',
            Descrizione=('Current original dropshot revision 2026-10-17; planned, not executed. '
                + ('Historical content exTask' + str(n-36) + '; historical record identities preserved.' if n >= 72 else 'Ionic abstraction program.')),
            **{'Dipende da': deps, 'Esperimenti previsti': [exp] if n == 36 else [],
               'Criteri di avvio': 'Task-specific prerequisites in current dropshot; dependency here is roadmap progression, not notebook count.',
               'Risultato atteso': 'Auditable scoped decision, with quality/cost/technical outcomes distinguished.'})
    arm = put('arms', 'task36-audit', Nome='Task36 documentary/source audit control', Ruolo='Baseline', Protocollo=[protocol])
    run = put('runs', 'task36-audit', Nome='Task36 local source/interface reconciliation', Stato='Completata', Braccio=[arm],
        **{'Artefatti prodotti': arts, 'Configurazione effettiva': json.dumps(audit, ensure_ascii=False),
           'Validità tecnica': '22 source hashes verified; 11 channels/18 gate inventory and Ca->SK phase verified. No new training, test or NEURON simulation.'})
    finding = put('findings', 'task36-contract', Nome='Task36 contract ready; Task37 evaluation work remains', Esito='Misto', Esperimenti=[exp],
        **{'Risultato': (repo / 'experiments/task36_result_v1.md').read_text(encoding='utf-8'),
           'Limitazioni': json.dumps(audit['open_evaluation_gaps'], ensure_ascii=False),
           'Incertezza': 'No evidence yet for accuracy/speed of new abstraction levels. Task37 preparation authorized only.'})
    # Close the pre-existing IE-P7 record identities; do not fork hypotheses.
    p7exp = p7put('experiments', 'confirmation', Nome='IE-P7 frozen native confirmation and operational cost', Stato='Concluso')
    p7run = p7put('runs', 'v4-complete', Nome='IE-P7 v4 complete and audited', Stato='Completata',
        **{'Artefatti prodotti': arts, 'Configurazione effettiva': json.dumps(p7audit),
           'Validità tecnica': '112 timing cells/preflights, 14 quality decisions recomputed; 8 CPU native-trace replays. Not a new NEURON execution or timing run.'})
    p7evals = {}
    for key, unit in [('native_v', 'mV'), ('native_ca', 'mM'), ('core_cost', 'ms'), ('operational_cost', 'ms')]:
        metric = p7put('metrics', key, Nome='IE-P7 ' + key, Famiglia='Regressione', Direzione='Minimizzare', **{'Unità': unit})
        p7evals[key] = p7put('evaluations', key, Nome='IE-P7 audited ' + key, Metrica=[metric], Versione='v1',
            Ruolo='Primaria', Target='Original frozen IE-P7 gates and scope; not new abstraction thresholds.')
    obs = []
    for name, q in p7['native_quality'].items():
        for i, metrics in enumerate(q['episodes']):
            for key, source in [('native_v', 'voltage_rmse_mv'), ('native_ca', 'calcium_max_mm')]:
                obs.append(p7put('observations', f'v4-{key}-{name}-{i}', Nome=f'IE-P7 v4 {name} case{i} {key}',
                    Valore=metrics[source], Run=[p7run], Numerosità=1,
                    **{'Specifica di valutazione': [p7evals[key]], 'Artefatti dettagliati': arts,
                       'Strato o sottogruppo': f'{name}/case{i}', 'Descrizione': json.dumps(metrics)}))
    for key, rows in [('core_cost', p7['core_rows']), ('operational_cost', p7['operational']['rows'])]:
        for row in rows:
            values = [s['wall_ms'] for s in row['samples']] if key == 'operational_cost' else row['timings']['wall_ms']
            import statistics
            obs.append(p7put('observations', f"v4-{key}-{row['batch']}-{row['mode']}-{row['name']}",
                Nome=f"IE-P7 {key} {row['name']} {row['mode']} b{row['batch']}",
                Valore=statistics.median(values), Numerosità=len(values), Run=[p7run],
                **{'Specifica di valutazione': [p7evals[key]], 'Artefatti dettagliati': arts, 'Descrizione': json.dumps(row)}))
    p7finding = p7put('findings', 'v4-outcome', Nome='IE-P7 native quality retained; operational speed gate not met', Esito='Misto',
        Osservazioni=obs, Esperimenti=[p7exp], **{'Risultato': (repo / 'experiments/ie_p7_result_v1.md').read_text(encoding='utf-8'),
        'Limitazioni': p7['exclusions'] + ' No general behavioral claim; CPU replay validates saved traces, not new native generation.'})
    for key in ('H_native', 'H_recurrence', 'H_cost', 'H_dilution', 'H_frontier'):
        status, outcome = ('Indeterminata', 'Indeterminata') if key == 'H_frontier' else ('Supportata nel dominio', 'Sostiene')
        claim = p7put('claims', key, Nome='IE-P7 ' + key, Stato=status)
        p7put('evidence', key, Nome='IE-P7 v4 ' + key, Esito=outcome,
            Argomentazione='Scoped native/atomic quality and resident cost retained; staged operational overhead dilutes benefit. No proof of dominance over analytic LUT or general latent-model claims.',
            **{'Affermazione valutata': [claim], 'Risultati a sostegno': [p7finding], 'Data valutazione': '2026-10-17',
               'Limiti e spiegazioni alternative': p7['exclusions']})
    p7put('actions', 'execute', Nome='Execute IE-P7 frozen confirmation and operational cost notebook', Stato='Conclusa')
    print(json.dumps({'stage': 'validated_batch_ready', 'records': len(registered), 'observations': len(obs)}), flush=True)
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    verified = mirror.verify()
    assert verified['valid']
    receipt = dict(valid=True, task=36, airtable_accessed=False, new_claims=claims,
        evaluation_specifications=evaluations, task_actions=actions, task36_run=run,
        task36_finding=finding, ie_p7_observations=len(obs), registered_record_ids=registered,
        decision='GO_TASK37_PREPARATION_ONLY', sqlite_verify=verified)
    write_json(folder / 'memory_receipt.json', receipt)
    print(json.dumps({'valid': True, 'records': len(registered), 'ie_p7_observations': len(obs), 'airtable_accessed': False}), flush=True)


if __name__ == '__main__':
    main()
