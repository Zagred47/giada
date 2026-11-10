"""Explicit user-authorized exploratory transition; no task42 run or gate waiver."""
import argparse
import hashlib
import json
import zipfile
from .mirror import ROOT, write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview, PreviewComplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    repo = ROOT.parent
    folder = repo / 'experiments/results/task41v_protocol_v1'
    target = repo / 'experiments/task42_exploratory_transition_v1.json'
    receipt_path = folder / 'task42_transition_memory_receipt.json'
    assert args.validate_only or not receipt_path.exists(), 'Transition already recorded; inspect receipt.'
    baseline = json.loads((folder / 'detailed_audit_memory_receipt.json').read_text())['verification']
    original = json.loads((folder / 'preregistration_receipt.json').read_text())['records']
    audit = json.loads((folder / 'detailed_audit.json').read_text())
    incumbent = audit['arms']['41v']['incumbent']
    heads = repo / 'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz'
    final = json.loads((repo / 'artifacts/task41v_kaggle_external/external_backup_receipt.json').read_text())
    with zipfile.ZipFile(final['local']) as z:
        member_sha = hashlib.sha256(z.read('frozen_incumbent.pt')).hexdigest()
    freeze = dict(format='giada-exploratory-transition-v1', date='2026-11-10',
        candidate='41r/timing_frozen_k',
        logical_weights_sha256='a8c8240648022024faa4aac66a5f4d0adba6952cea647cc6494594c6e5ca08e6',
        baseline_weights_sha256='d5b7a92d2d0fe350c79e97879003570afefc8eb4b45026e148ef5c4a338d0cd0',
        heads_path=heads.relative_to(repo).as_posix(), heads_file_sha256=hashlib.sha256(heads.read_bytes()).hexdigest(),
        full_state_archive=final['local'], full_state_archive_sha256=final['remote']['sha256'],
        full_state_member='frozen_incumbent.pt', full_state_member_sha256=member_sha,
        source_revision='6734cc2137f6ba3235956876cb07e7019a544bb7',
        stress=incumbent['opened_stress'], native=incumbent['summaries'],
        known_offset_failures_ms=dict(K_Pst=[0.7,1.1,4.4,4.5], Nap_Et2=[0.8,1.8]),
        offset_threshold_ms=0.3, original_fp32_failed_checks=[218,234],
        task41_original_gate_passed=False, task42_exploratory_authorized=True,
        task42_qualified_release=False, thresholds_changed=False, abstraction_changed=False,
        task42_launched=False, future_tasks_automatically_authorized=False,
        decision='Stop serial41 repairs; advance under explicit accepted limitations, not retroactive qualification.',
        conditions=['Keep teacher/formula and frozen41r controls paired.',
                    'Validate passive/axial reference floors, units and balances before interpreting active transfer.',
                    'Separate inherited local failures, new failures and coupling amplification by case identity.',
                    'No privileged teacher corrections online, hidden refit, new solver or abstraction change.',
                    'Keep original quality gates and full failure reporting; no aggregate cancellation of regressions.',
                    'No general ELM superiority, teacher equivalence or integrated speedup claim.',
                    'Reopen41 repairs only for a concrete downstream blocker or genuinely new discriminating hypothesis.'],
        speed_scope='Historical~8x isolated resident ionic core; current integrated benchmark not a qualified speedup.',
        roadmap='experiments/giada_modular_surrogate_roadmap.md')
    limits = 'Exploratory continuation only;41 original NO-GO remains. Original numeric thresholds and six failures retained; no run or automatic downstream promotion.'
    m = ResultPreview() if args.validate_only else IncrementalBatchMirror()
    if not args.validate_only:
        assert m.baseline_hash == baseline['snapshot_hash']
    ids = {}
    def put(table, suffix, **fields):
        code = f'{table}-giada-task42-transition-{suffix}-v1'
        rid = m.local_upsert(table, {'Codice stabile':code, **fields})['record_id']
        ids[code] = rid
        return rid
    artifact = put('artifacts','roadmap',Nome='Original dropshot amended10 November2026',Tipo='Report',Versione='2026-11-10',
        Percorso=freeze['roadmap'], **{'SHA-256':hashlib.sha256((repo / freeze['roadmap']).read_bytes()).hexdigest()})
    obs = put('observations','freeze',Nome='Frozen41r identity, accepted failures and exploratory42 authorization',Descrizione=json.dumps(freeze), **{'Artefatti dettagliati':[artifact]})
    finding = put('findings','limits',Nome='41 retained as useful approximation, not fully qualified surrogate',Esito='Misto',
        Esperimenti=[original['experiments-giada-task41v-factorial-v1']],Osservazioni=[obs],
        Risultato=freeze['decision']+' '+freeze['speed_scope'],Limitazioni=limits)
    decision = put('decisions','authorize',Nome='Authorize42 exploration with frozen41r; do not pass41 retroactively',Esito='Continuare',
        Motivazione=json.dumps(freeze),Risultati=[finding], **{'Data':'2026-11-10','Condizioni di revisione':'Concrete downstream blocker or new discriminating hypothesis; preserve original gates.'})
    put('actions','prepare42',Nome='Prepare Task42 axial interface and paired systems as exploratory branch',Stato='Proposta',
        Descrizione=json.dumps(freeze), **{'Decisioni di origine':[decision]})
    put('lessons','permission_vs_qualification',Nome='Research progression is distinct from passing a scientific gate',
        Lezione='A documented user-authorized exploratory branch may proceed with known failures, without rewriting thresholds, historical outcomes or qualification claims.',Risultati=[finding],Limiti=limits)
    m.commit_batch()
    if args.validate_only:
        return
    verification = m.verify_incremental(baseline)
    assert verification['valid']
    write_json(target, freeze)
    write_json(receipt_path,dict(valid=True,verification=verification,records=ids,airtable_accessed=False,task42_launched=False))
    m.export_snapshot_to(ROOT / 'data/airtable_snapshot.json')
    print('Task42 exploratory decision and41r freeze verified; no run launched.',flush=True)


if __name__ == '__main__':
    try:
        main()
    except PreviewComplete:
        print('Task42 transition schema preview valid; no writes')
