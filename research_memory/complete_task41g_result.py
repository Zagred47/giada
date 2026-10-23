"""Verified final archive, history-coverage evidence, and revised proposed41h.

Run only after register_task41g_result completed. No remote writes or launch.
"""
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT, write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview, PreviewComplete


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task41g_audited_v1'
    completed = json.loads((folder / 'audit_completion.json').read_text())
    prior = json.loads((folder / 'memory_receipt.json').read_text())
    assert prior['valid'] and prior['raw_original_bytes_verified']
    assert completed['valid'] and completed['final_archive_complete'] and not completed['launched']
    preview = '--validate-only' in sys.argv
    m = ResultPreview() if preview else IncrementalBatchMirror()
    ids = {}

    @lru_cache(None)
    def ref(code):
        if code in ids:
            return ids[code]
        table = {'finding': 'findings', 'decision': 'decisions'}.get(code.split('-', 1)[0], code.split('-', 1)[0])
        assert table in m.contract.tables, table
        rows = m.query("SELECT _record_id AS record_id FROM [" + table + "] WHERE [Codice stabile]='" + code.replace("'", "''") + "'")['rows']
        assert len(rows) == 1, code
        return rows[0]['record_id']

    def put(table, suffix, **fields):
        code = f'{table}-giada-task41g-completion-{suffix}-v1'
        rid = m.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        ids[code] = rid
        return rid

    archive = Path(completed['recovery']['local'])
    archive_id = put('artifacts', 'final-archive', Nome='Task41g complete final ZIP, byte-identical recovery',
        Tipo='Report', Versione='v1', Percorso=archive.relative_to(repo).as_posix(),
        **{'SHA-256': completed['recovery']['sha256'], 'Dimensione byte': archive.stat().st_size})
    artifacts = [archive_id]
    observations = []
    files = [folder / 'audit_completion.json', folder / 'synthesis_verification.json', folder / 'saved_response_audit.json',
        folder / 'early_memory_queries.json', folder / 'early_memory_queries.sql',
        repo / 'artifacts/task41g_primary_external/final_reconstruction_receipt.json',
        repo / 'scripts/complete_task41g_audit.py', repo / 'scripts/verify_task41g_synthesis.py',
        repo / 'scripts/reconstruct_task41g_final_archive.py', repo / 'scripts/query_task41g_early_memory.py',
        repo / 'scripts/audit_task41g_saved_responses.py', repo / 'research_memory/record_local_backup.py', Path(__file__)]
    files += [repo / row['path'] for row in completed['source_evidence']]
    for path in files:
        raw = path.read_bytes()
        aid = put('artifacts', path.name, Nome='Task41g completion ' + path.name,
            Tipo='Codice' if path.suffix == '.py' else 'Report', Versione='v1',
            Percorso=path.relative_to(repo).as_posix(), **{'SHA-256': hashlib.sha256(raw).hexdigest(), 'Dimensione byte': len(raw)})
        artifacts.append(aid)
        if path.suffix == '.json':
            observations.append(put('observations', path.name, Nome='Task41g completion evidence ' + path.name,
                Descrizione=raw.decode('utf-8'), **{'Artefatti dettagliati': [aid, archive_id]}))
    exp = ref('experiments-giada-task41g-factorial-v1')
    history = completed['history_finding']
    history_id = put('findings', 'history-contract', Nome='Task41g constant-tail versus return-ramp coverage gap',
        Esito='Misto', Esperimenti=[exp], Osservazioni=observations,
        Risultato=history['observed'] + ' ' + history['interpretation'], Limitazioni=history['boundary'])
    voltage = completed['voltage_precision_finding']
    voltage_id = put('findings', 'voltage-precision-contract', Nome='Task41g V64 probe jointly changes state and solver arithmetic',
        Esito='Misto', Esperimenti=[exp], Osservazioni=observations,
        Risultato=voltage['observed'], Limitazioni=voltage['limits'])
    responses_id = put('findings', 'saved-causal-responses', Nome='Task41g residual Nap errors measure responses to potassium-channel block',
        Esito='Misto', Esperimenti=[exp], Osservazioni=observations,
        Risultato='Saved traces independently reproduce both failed intervention metrics. Case122 seed17 blocksSK_E2 during reversed_staircase; Nap response NRMSE0.000761 imposed versus0.014015 autonomous. Case108 seed43 blocksK_Pst during irregular_train;0.000346 imposed versus0.010064 autonomous. These are differential causal-response errors, not the absolute current trajectory error.',
        Limitazioni=completed['saved_response_audit']['interpretation'] + ' Offline trace reduction only; selected nominee unchanged.')
    voltage_claim = put('claims', 'voltage-state-versus-solver', Nome='Task41g voltage accumulator versus Newton cancellation',
        Tipo='Ipotesi', Stato='Aperta', Enunciato=voltage['hypothesis'],
        **{'Condizioni di falsificazione': voltage['falsifier']}, Limiti=voltage['limits'] + ' ' + voltage['test'])
    put('evidence', 'voltage-state-versus-solver', Nome='Task41g numerical contract motivates split precision factorial',
        Esito='Indeterminata', **{'Affermazione valutata': [voltage_claim],
        'Risultati a sostegno': [voltage_id, responses_id, ref('findings-giada-task41g-result-precision-v1')],
        'Argomentazione': voltage['test'], 'Limiti e spiegazioni alternative': voltage['limits'], 'Data valutazione': '2026-10-23'})
    recovery_id = put('findings', 'final-recovery', Nome='Task41g final scientific archive safely recovered',
        Esito='Positivo', Esperimenti=[exp], Osservazioni=observations,
        Risultato='Final ZIP SHA-256 equals remote manifest; all181 members CRC-valid; all65 original JSON reports match the imported inventory. No restart or retraining. 258 summary checks passed.',
        Limitazioni='Technical recovery does not change scientific NO-GO, nominee or thresholds. Older partial-backup statements are historical and superseded by this result.')
    claim = put('claims', 'history-coverage', Nome='Task41g trajectory coverage rather than voltage-range extrapolation',
        Tipo='Ipotesi', Stato='Aperta', Enunciato=history['hypothesis'],
        **{'Condizioni di falsificazione': history['falsifier']}, Limiti=history['boundary'])
    put('evidence', 'history-coverage', Nome='Task41g source-contract evidence motivating history ablation',
        Esito='Indeterminata', **{'Affermazione valutata': [claim], 'Risultati a sostegno': [history_id],
        'Argomentazione': history['interpretation'], 'Limiti e spiegazioni alternative': history['falsifier'],
        'Data valutazione': '2026-10-23'})
    put('lessons', 'coverage', Nome='Voltage coverage is not coverage of return histories',
        Lezione=history['observed'] + ' Compare matched constant and ramp histories before attributing failure to abstract-model incapacity.',
        Risultati=[history_id, ref('findings-giada-task41g-result-dose-v1'), ref('findings-giada-task41f-result-equilibrium-v1')],
        Limiti=history['boundary'])
    for pattern in completed['early_memory_patterns']:
        put('lessons', pattern['id'], Nome='Task41g historical transfer: ' + pattern['id'],
            Lezione=pattern['evidence'] + ' ' + pattern['transfer'],
            Risultati=[ref(code) for code in pattern['findings']] + [history_id],
            Limiti='Exploratory analogy, not new causal replication. ' + pattern['transfer'])
    proposal = json.dumps(completed['next_experiment'], ensure_ascii=False)
    m.local_upsert('decisions', {'Codice stabile': 'decisions-giada-task41g-result-propose41h-v1',
        'Motivazione': 'Revised after source-history audit; supersedes initial8arm capacity-first proposal, preserved in synthesis.json. ' + proposal,
        'Risultati': [ref('findings-giada-task41g-result-' + k + '-v1') for k in
            ('outcome', 'quality', 'best_tail', 'residuals', 'precision', 'preservation', 'projection', 'dose', 'speed')] + [history_id, voltage_id, responses_id, recovery_id]})
    m.local_upsert('actions', {'Codice stabile': 'actions-giada-task41g-result-propose41h-v1',
        'Nome': 'Task41h revised10arm proposal, not prepared or launched', 'Stato': 'Proposta',
        'Descrizione': proposal, 'Criteri di avvio': 'User authorization and completed preregistration/preflight; unchanged thresholds and official reference.'})
    m.local_upsert('actions', {'Codice stabile': 'actions-giada-task41g-result-final-container-recovery-v1',
        'Nome': 'Task41g final ZIP recovered and verified outside runtime', 'Stato': 'Conclusa',
        'Descrizione': json.dumps(completed['recovery'], ensure_ascii=False)})
    obsolete = 'Contenitore ZIP finale completo non recuperato; nessun riavvio o retraining.'
    replacement = 'Aggiornamento conclusivo: ZIP finale completo recuperato con SHA identico al manifest remoto e CRC verificati; nessun riavvio o retraining.'
    for table, field, prefix in [('findings', 'Limitazioni', 'findings-giada-task41g-result-'),
        ('claims', 'Limiti', 'claims-giada-task41g-result-'), ('claims', 'Limiti', 'claims-giada-task41f-result-'),
        ('evidence', 'Limiti e spiegazioni alternative', 'evidence-giada-task41g-result-'),
        ('lessons', 'Limiti', 'lessons-giada-task41g-result-'), ('experiments', 'Descrizione', 'experiments-giada-task41g-')]:
        rows = m.query(f"SELECT [Codice stabile] code,[{field}] value FROM [{table}] WHERE [Codice stabile] LIKE '{prefix}%'")['rows']
        for row in rows:
            if obsolete in (row['value'] or ''):
                m.local_upsert(table, {'Codice stabile': row['code'], field: row['value'].replace(obsolete, replacement)})
    m.local_upsert('analyses', {'Codice stabile': 'analyses-giada-task41g-result-audit-v1',
        'Configurazione': json.dumps(dict(thresholds_changed=False, retraining=False, new_experiment_launched=False,
            final_archive_complete=True, completion='audit_completion.json', raw_json_reports=65))})
    runs = m.query("SELECT [Codice stabile] code FROM runs WHERE [Codice stabile] LIKE 'runs-giada-task41g-%'")['rows']
    assert len(runs) == 36
    for row in runs:
        m.local_upsert('runs', {'Codice stabile': row['code'],
            'Artefatti prodotti': [ref('artifacts-giada-task41g-result-archive-v1'), ref('artifacts-giada-task41g-result-final_report_recovered.json-v1'), archive_id],
            'Validità tecnica': '60000updates; seven checkpoints finite including Adam; freeze and checkpoint SHA verified; complete final archive SHA/CRC verified outside runtime; scientific NO-GO unchanged.'})
    put('analyses', 'completion', Nome='Task41g archive completion and history-aware hypothesis refinement',
        Origine='Post hoc', Versione='v1', Procedura='complete_task41g_audit.py; source contract inspection; no simulations or training',
        Configurazione=json.dumps(dict(thresholds_changed=False, final_archive_complete=True, proposed_only=True)),
        **{'Run analizzati': [ref(r['code']) for r in runs], 'Input artefatti': artifacts,
        '12 · Risultati scientifici': [history_id, voltage_id, responses_id, recovery_id]})
    print('Committing final recovery and refined hypotheses', flush=True)
    change = m.commit_batch()
    if preview:
        return
    verification = m.verify_incremental(prior['verification'])
    assert verification['valid']
    m.export_snapshot_to(ROOT / 'data/airtable_snapshot.json')
    write_json(folder / 'completion_memory_receipt.json', dict(valid=True, verification=verification,
        change=change, records=ids, final_archive_complete=True, raw_reports=65, prior41f_reports_preserved=67,
        thresholds_changed=False, launched=False, airtable_accessed=False, backup_policy='local only'))
    write_json(folder / 'LATEST.json', dict(current_audit='audit_completion.json',
        current_memory_receipt='completion_memory_receipt.json', prior_scientific_synthesis='synthesis.json',
        superseded_fields='Initial backup-incomplete status and initial8arm next-experiment proposal only; scientific results unchanged.',
        final_archive_complete=True, new_experiment_launched=False))
    print('Task41g completion registered and verified', flush=True)


if __name__ == '__main__':
    try:
        main()
    except PreviewComplete:
        print('Preview passed; no DB writes')
