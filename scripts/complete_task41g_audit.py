"""Finalize recovery evidence and a scoped prospective 41h proposal; no training."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task41g_audited_v1'


def main():
    recovery = json.loads((ROOT / 'artifacts/task41g_primary_external/final_reconstruction_receipt.json').read_text())
    assert recovery['valid'] and recovery['byte_identical_to_remote'] and recovery['crc_all_members_valid']
    inventory = json.loads((OUT / 'evidence_inventory.json').read_text())
    digest = hashlib.sha256()
    with Path(recovery['local']).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    assert digest.hexdigest() == recovery['expected']['sha256']
    reports = []
    with zipfile.ZipFile(recovery['local']) as archive:
        assert {n for n in archive.namelist() if n.endswith('.json')} == {r['member'] for r in inventory['raw_json_reports']}
        for row in inventory['raw_json_reports']:
            digest = hashlib.sha256()
            with archive.open(row['member']) as stream:
                while block := stream.read(8 * 1024 * 1024):
                    digest.update(block)
            assert digest.hexdigest() == row['sha256'], row['member']
            reports.append(dict(member=row['member'], sha256=row['sha256']))
        for row in inventory['checkpoints']:
            assert row['all_tensors_finite']
            assert hashlib.sha256(archive.read(row['member'])).hexdigest() == row['sha256']
    sources = []
    for name in ('task41g_preservation.py', 'task41c_frozen_confirmation.py', 'task41g_evaluation.py', 'task40_causal_reinsertion.py', 'task37_ionic_playground.py'):
        path = ROOT / 'src/giada_teacher' / name
        sources.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(
        valid=True,
        supersedes='Backup incompleteness and initial 41h proposal in synthesis.json; scientific counts and thresholds unchanged.',
        final_archive_complete=True,
        recovery=recovery,
        original_reports_match_final_archive=reports,
        all_65_original_reports_verified=True,
        final_checkpoint_hashes_match_finite_audit=inventory['checkpoints'],
        summary_checks=json.loads((OUT / 'synthesis_verification.json').read_text()),
        source_evidence=sources,
        saved_response_audit=json.loads((OUT / 'saved_response_audit.json').read_text()),
        voltage_precision_finding={
            'observed': 'precision_core casts V to the selected voltage dtype and calls membrane with conductance/injection/synapses cast to that dtype; it therefore changes both persistent V precision and membrane arithmetic. membrane uses the original12Newton iterations with residual denom*nv-numer+nmda*nv, with numer containing cap*v.',
            'hypothesis': 'Cancellation in the Newton residual and/or rounding of recurrent voltage increments contributes to tiny autonomous current errors. The current V64 probe does not distinguish these mechanisms.',
            'test': 'Frozen2x2: persistentV32/64 x membrane-solve32/64, gate network and readout held fixed. Return casts implement persistent-state precision explicitly. Keep12Newton iterations, dt and reference unchanged; measure residuals/increments, all quality panels and paired cost. If arithmetic dominates, test an algebraically recentered FP32 residual using voltage differences, without changing the equation or iteration count.',
            'falsifier': 'Split precision gives no reproducible effect on the new checkpoints or recentering does not reduce discrepancy; the original probe could reflect an interaction or another part of the V path.',
            'limits': 'Source-level numerical risk, not a measured cancellation attribution. Do not claim FP64 is necessary or sufficient for all failures. Canonical-reference parity/floor must be checked for every arithmetic variant.'
        },
        early_memory_patterns=[
            dict(id='identifiability_not_size',
                findings=['finding-task2b-h-physical-tau-identifiability-repaired-v1'],
                evidence='Task2b development score improved33.4x with rate supervision; multi-horizon independently repaired rollout. Constant-voltage isolated h only.',
                transfer='The right labels/horizons may identify dynamics that a small endpoint loss leaves ambiguous. Today rate supervision already exists: test information in path-aware tails, not mechanically add the old loss again.'),
            dict(id='conflict_not_pcgrad',
                findings=['finding-task3b-symmetric-rates-stabilize-shared-mh-v1'],
                evidence='Initial gradient conflict66.7%, yet PCGrad worsened score82.8%. Symmetric rate supervision improved61.5%; doubling budget improved63.1%.',
                transfer='Negative cosine is a diagnostic, not sufficient justification for gradient surgery. Prioritize controlled objective exposure and mini scaling, retain interventions with evidence.'),
            dict(id='support_before_width',
                findings=['finding-task3d-voltage-support-conditional-shape-repair-v1', 'finding-task3e-full-repair-sealed-generalization-v1'],
                evidence='Support plus conditional shape prior repaired width23 with97.9% score improvement; width46 worsened66.1%. The full package later passed an independent sealed confirmation.',
                transfer='That was voltage-support extrapolation, unlike the current within-range history gap. It motivates distinguishing coverage and conditioning before increasing width, not assuming an identical cause.')
        ],
        history_finding={
            'observed': 'Task41g tail_data uses a constant held voltage uniform[-85,-30]mV, initial gates from equilibrium at uniform[-15,35]mV. Its analytic tail samples stop at <=400ms. The remaining slow_boundary_return stress includes a 13-23ms depolarization, a 30ms return ramp, then a rest in[-48,-37]mV; it lasts1200ms and includes initial states at-95/-42mV.',
            'interpretation': 'The resting voltage is already inside the training voltage interval. Coverage of voltage values is not coverage of histories, initial states or weak late crossings. This is an observed contract gap, not proof that the gap caused the residual error.',
            'hypothesis': 'Training on constant-voltage tails underweights equilibrium errors accumulated during a return ramp and at late weak crossings, so the objective does not identify the required local precision.',
            'falsifier': 'At matched exposure, horizon, initial-state distribution and parameter count, adding new return-ramp histories does not improve held-out crossing error or the local signed-error diagnostic.',
            'boundary': 'Do not fit the old failed protocol/seed; generate new fit/development families. Do not infer that extra hidden memory or changed kinetics is necessary.'
        },
        next_experiment={
            'task': '41h proposed only; not prepared or launched',
            'thresholds_changed': False,
            'reference_changed': False,
            'frozen_diagnostics': [
                'Q0, frozen41g nominee and diagnostic41g best-tail: V32 vs V64, same official FP32 reference, all panels. Gate/readout precision unchanged. Test equivalent more stable FP32 arithmetic separately; no change to physical equation or dt.',
                'Split persistent voltage precision from membrane-solve precision in a2x2 frozen comparison. The existing V64 probe changes both. Keep the original12Newton iterations; inspect cancellation and near-zero increments. A recentered FP32 Newton residual is a prospective equivalent-arithmetic control, not a new membrane model.',
                'On new41g checkpoints, diagnostic oracle swaps u_m/z_m/h for Ca_HVA, signed current error and local slope at offset. These swaps are never deployment candidates.',
                'Audit all eligible and ineligible tail anchors, initial-state/history coverage and gradient exposure. Quantify late-current amplitudes instead of removing small events.'
            ],
            'training_matrix': {
                'primary_axes': {
                    'history': ['constant tail', 'matched mixture of constant and return-ramp tails'],
                    'auxiliary_every_updates': [256, 16],
                    'FIT_anchor_count': [4096, 16384]
                },
                'primary_arms': 8,
                'primary_basis': 64,
                'conditional_capacity_arms': 'Repeat the all-baseline and full-treatment arms at basis128: two extra arms; estimates capacity only at these two settings, not its complete factorial interactions.',
                'total_arms': 10,
                'seeds': [17, 29, 43],
                'members': 30,
                'parent': 'Q0 frozen',
                'trainable_scope': 'Only learned Ca_HVA.m equilibrium correction; retain all18 recurrent states and existing abstraction. No canonical kinetics in deployed inference.',
                'controls': 'Same initial weights for shared coefficients, paired minibatch streams, same optimizer/schedule/update budget, same thresholds. Match history arms on horizon up to1200ms, sample count, V/Ca/multiplier distributions and initial-state coverage; history shape is the intended difference. Keep the old41g checkpoint as a frozen control, not mislabel the new baseline as exact41g replication.',
                'compute': 'Batch/vectorize independent arms if equivalence preflight passes. Precompute teacher and frozen-parent quantities. For long constant tails, use the exact composition of the existing recurrence in training after forward/gradient equivalence checks; do not require full12000step backpropagation on every auxiliary slot. For prescribed-voltage ramps with frozen z and only equilibrium m trainable, a weighted composition of the existing affine recurrence may retain the entire causal path without a long Python backward loop; verify forward and gradient equivalence, do not apply this shortcut to endogenous-voltage feedback. This is training execution, not a new analytic inference fallback. Record GPU seconds, actual examples, unique exposure, gradient norms/cosines/clipping and per-regime progress. Do not claim equal GPU cost merely from equal update counts. Common auxiliary computation slots may isolate gradient frequency from evaluation overhead.',
                'checkpoints': [512, 2048, 8192, 30000, 60000],
                'selection': 'New disjoint development protocols/seeds. On the same saved checkpoints and development panel, prospectively compare the original aggregate selection rule with a regime-stratified rule; designate the primary rule before training/evaluation. This adds no retraining and isolates selection from weights. All acceptance thresholds stay fixed. Never select the rule by looking at old stress outcomes. Old opened stress cases remain regression diagnostics. No retrospective promotion of best-tail41g; a repaired nominee needs fresh independent confirmation.'
            },
            'execution_sidecar': 'Same frozen weights: legacy versus compact execution, reuse64hat features, compute only active heads and parent once; preserve separate tanh operations. Optional sparse2-neighbor basis evaluation requires boundary and rollout parity. Benchmark step and whole-loop at1/642/4096 on same GPU with paired order/warmup. Cross precision with execution so FP64 cost is visible. Retain the original formula and LUT controls and the10% saving requirement. Report an additional formula baseline with matched membrane arithmetic; never credit a shared solver optimization solely to the neural ionic block. The official quality reference remains unchanged.',
            'interpretation': 'History effect=>coverage; auxiliary-frequency effect=>effective objective exposure; FIT-count effect=>support density; conditional128 effect=>local resolution; oracle-only gains=>learnability/optimization gap; V64 gains=>numerical feedback contribution. Interactions and regressions are results, not reasons to loosen gates.',
            'stop_rules': 'Reference/parity failure is technical, not scientific NO-GO. Keep thresholds, dt, physical interface and abstraction fixed. No promised definitive resolution, no Task42 authorization.'
        },
        scientific_no_go_unchanged=True,
        launched=False,
        retraining=False,
        runtime_restarted=False,
        airtable_accessed=False,
        DB_backup='local only'
    )
    (OUT / 'audit_completion.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(valid=True, final_archive_complete=True, original_reports=len(reports), proposed_arms=10, launched=False)))


if __name__ == '__main__':
    main()
