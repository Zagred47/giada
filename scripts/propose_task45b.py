"""Evidence-linked45b proposal only; no notebook submission or training."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    audit=json.loads((ROOT/'experiments/results/task45_audited_v1/audit.json').read_text())
    def probe(channel):return next(r for r in audit['frozen_coefficient_probes'] if r['channel']==channel and r['seed']==29 and r['voltage_mv']==-106.)
    nap,ih=probe('Nap_Et2'),probe('Ih')
    plan=dict(task='45b',status='PROPOSED_NOT_LAUNCHED',thresholds_changed=False,abstraction_changed=False,
        frozen_reference_sha256=audit['weights_sha256'],execution_authorized=False,training_authorized=False,
        original45_preserved=True,original41_gate_passed=False,independent47_accessed=False,
        evidence=dict(Nap_h=nap,Ih_m=ih,max_constant_clamp_fp32_error=audit['max_fp32_vs_same_coeff_exact_constant_clamp_error']),
        hypotheses=[
            dict(id='protocol_domain',priority='confirmed_protocol_issue',claim='Original disconnected hyperpolarization exceeds the registered physical domain, even for the native/formula teacher.',
                 test='Calibrate negative current using native teacher only. Freeze common fixed-current settings across geometry/coupling before inspecting candidate outcomes. Suggested calibration targets -90/-110/-130mV are stimulus difficulty targets, NOT relaxed acceptance thresholds.',
                 discriminator='Original nine out-of-domain checks remain reported; calibrated reference stays within original physical bounds and native numerical floors.'),
            dict(id='nap_rate_regression',priority='highest_supported_local_mechanism',claim='The41r learned Nap_h rate readout regressed outside the opened repair coverage; equilibrium alone is not the primary error at the probed voltage.',
                 test='Paired learned-head rollback factorial: keep41r everywhere, restore only Nap_h update-logit head, only equilibrium-logit head (negative causal control), or both from the verified pre41r learned baseline. No analytic runtime replacement in selectable candidates.',
                 discriminator='Rate-only rollback rescues state/current response while equilibrium-only rollback does not; test old41r pass cases to expose reintroduced regressions.'),
            dict(id='ih_rate_bias',priority='high_supported_local_mechanism',claim='Ih transient error reflects an existing learned rate bias, not a new41r change or necessarily missing memory.',
                 test='Frozen coefficient probes and factorial oracle controls for x_inf versus update probability on common voltage paths; constrained Ih kinetic readouts (inf and rate) as a learned sidecar only after FIT/DEV freeze.',
                 discriminator='At -106mV/seed29 the100ms gate error is .01271; canonical-rate-only diagnostic reduces it to .0000287, equilibrium-only does not.'),
            dict(id='support_vs_budget',priority='unresolved_training_cause',claim='Negative-tail coverage, optimization budget or frozen-feature conditioning may limit a constrained rate-head repair.',
                 test='Equal-count nominal vs tail-balanced FIT support crossed with B vs4B updates, using paired initialization/minibatch indices and checkpoints from the same two training trajectories per seed. Same objective across arms. Diagnostic least-squares projection onto the same frozen readout features bounds optimization error.',
                 discriminator='Coverage effect at fixed budget versus budget effect at fixed coverage. Least-squares success with SGD failure points to optimization; failure of both does not justify assuming more compute suffices.'),
            dict(id='timing_conditioning',priority='needs_path_probe',claim='Flat Ih onset/offset crossings can turn small current bias into large registered timing error.',
                 test='Measure current residual and reference slope at the unchanged10%-transient crossings; compare predicted shift deltaI/slope to measured shift. Also save pulse-local event identities as diagnostic sidecar.',
                 discriminator='Concentration at flat crossings supports sensitivity of the observable; registered global timing failures still count. No event-window retuning or threshold waiver.'),
            dict(id='fp32_accumulation',priority='low_for_dominant_Nap_error',claim='FP32 recurrence roundoff is the main source of the observed large Nap_h discrepancy.',
                 test='Same frozen FP32 coefficients, FP32 versus FP64 recurrence and exact constant-voltage solution; a small saved-path precision control, not a new selectable precision policy.',
                 discriminator='Current constant-clamp probe maximum roundoff .0000722, versus Nap_h learned-coefficient error .0761 at -106mV; therefore roundoff alone is not supported as the dominant explanation there.')],
        phases=[
            dict(name='A native calibration',detail='Teacher-only calibration and freeze. One fixed-current panel preserves coupling counterfactuals; do not independently retune each geometry and call that a pure coupling effect.'),
            dict(name='B atomic identification',detail='Common V clamps and replay of saved45 V/Ca paths; x_inf/rate2x2 oracle diagnostics per gate, initial-state and100/500/1000ms probes; teacher-reset one-step versus free gate recurrence. Oracles are never selectable.'),
            dict(name='C constrained learned repair',detail='Proposed small FIT/DEV-only support×budget matrix on the relevant rate readouts, plus fixed-feature least-squares diagnostic. Numeric budgets, split hashes and candidate nomination rule must be preregistered before execution.'),
            dict(name='D paired coupled confirmation of mechanism',detail='Small2x2x2 factorial: Nap_h learned-rate rollback on/off, pre-frozen Ih inf-readout repair on/off, Ih rate-readout repair on/off. Eight contrasts include unchanged41r. Keep original current algebra/Ca/axial solver and18 states. Compare imposed/autonomous and0/.25/1 coupling with identical streams. Final case/budget caps must be preregistered; no indiscriminate extra arms.')],
        regression_guards=[
            'Keep all72 interpretable original45 cases unchanged as opened regression fixtures; safe calibration must not erase the difficult -106mV examples.',
            'Retain original nine out-of-domain cases as labelled diagnostics, never reclassify them as model passes.',
            'Use43/44 regressions and original41r previously-passing stress cases; original six41 failures remain explicit accepted limitations, not retroactively promoted.',
            'Same original current/gate/charge/timing/physical/native-floor thresholds. No aggregate cancellation of prefix or qualitative failures.',
            'Record hidden gate states as well as currents/V; small voltage error does not excuse wrong Nap_h memory.',
            'No analytic oracle in a nominated network; no abstraction reduction. Keep41r checkpoint immutable.',
            'No independent47 data access or automatic release/promotion. This is a proposal, not an executed45b.'
        ])
    write_json(ROOT/'experiments/task45b_proposed_contrasts_v1.json',plan)
    print('Task45b evidence-linked proposal generated; no execution authorized.')
if __name__=='__main__':main()
