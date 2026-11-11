"""Design45c from audited45b; intentionally no notebook dispatch or fitting."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    folder=ROOT/'experiments/results/task45b_audited_v1'
    audit=json.loads((folder/'audit.json').read_text());freeze=json.loads((folder/'f011_freeze.json').read_text())
    prior=json.loads((ROOT/'experiments/task45b_preregistration_v1.json').read_text())
    assert audit['valid'] and freeze['valid'] and {r['dominant_gate'] for r in audit['corrected_gate_failures']}=={'Ca_LVAst.h'}
    hypotheses=[
        dict(id='nap_compensation',claim='Global Nap_h rate rollback breaks a learned equilibrium/rate compensation on nominal histories.',
            test='Nap2x2: rate-only versus joint equilibrium/rate, crossed with functional preservation guard off/on. All start f011, not the old baseline.',
            discriminator='If joint+guard preserves old pass identities while rescuing negative-tail gate memory, compensation/preservation is supported; low coefficient RMSE alone does not count.'),
        dict(id='nap_information_vs_conditioning',claim='Frozen-feature conditioning or inadequate rare-history coverage, not missing memory state, limits the repair.',
            test='One common nominal+negative-tail FIT pool; same frozen-feature least-squares diagnostic and1/4 nonlinear refinements. Probe coefficient, free recurrence and functional losses separately.',
            discriminator='Compare ideal FIT projection, actual quantized forward and cross-history DEV. Do not infer that more data/steps necessarily cures a failed readout space.'),
        dict(id='ih_crossing_residual',claim='A tiny equilibrium/rate bias amplified at a flat crossing explains the remaining Ih offset.',
            test='Ih coefficient-only versus crossing-aware joint-readout refinement, same FIT support. Observe the actual intermediate node1 as well as root/distal; probe inf-only/rate-only oracles diagnostically.',
            discriminator='Reduced actual signed onset/offset under original event evaluator, with correct current amplitude and no new pass-case failures. DeltaI/slope alone is not a success criterion.'),
        dict(id='calva_tail',claim='Ca_LVAst.h kinetic-map extrapolation causes the hidden-state failures near -130mV.',
            test='Ca_LVAst.h rate-only versus joint equilibrium/rate readout fits on nominal+tail FIT; inf/rate2x2 oracles and teacher-reset/free recurrence first. Identify real owning module through Swap mask, not hardcoded parent.',
            discriminator='A kinetic oracle must explain the gate error before attributing it to optimization. Preserve Ca current, calcium feedback and nominal Ca_LVAst gating.'),
        dict(id='gate_identity_negative_control',claim='NaTs2_t.h is the dominant failed hidden gate in the six45b safe-panel cases.',
            test='Probe channel/state IDs for both Ca_LVAst.h(index3) and NaTs2_t.h(index7) on all six failures; NaTs2_t.h stays frozen as negative attribution control unless new evidence says otherwise.',
            discriminator='45b actual18-state traces contradict this claim: dominant index3, not7. No unnecessary NaTs2_t.h training based on a naming mistake.'),
        dict(id='repair_interaction',claim='Repairs selected independently on DEV compose without new coupled feedback or inherited-pass regressions.',
            test='Combine pre-frozen DEV-selected Nap/Ih/Ca_LVAst repairs and compare with each component under identical imposed/autonomous, coupling and prefix cases.',
            discriminator='Combined arm must retain the union of acquired pass identities, not only improve an aggregate average. Distinguish inherited f011 failures from new composition failures.')]
    arms=[dict(id='C0',role='baseline',change='f011 unchanged'),
        *[dict(id=f'N{r}{g}',role='Nap causal contrast',change=('joint inf/rate' if r else 'rate only')+(' + functional preservation' if g else ' without preservation')) for r in (0,1) for g in (0,1)],
        dict(id='I0',role='Ih control',change='joint inf/rate, coefficient-only objective'),dict(id='I1',role='Ih treatment',change='same heads/support plus causal crossing-aware objective'),
        dict(id='L0',role='Ca_LVAst contrast',change='h rate-only readout'),dict(id='L1',role='Ca_LVAst contrast',change='h equilibrium+rate readouts'),
        dict(id='J',role='composition',change='DEV-frozen best nonregressing component of each family, no coupled-outcome selection')]
    plan=dict(task='45c',version='design-v1',status='DESIGN_PREPARED_NOT_EXECUTABLE',date='2026-11-11',
        preparation_authorized=True,remote_execution_authorized=False,notebook_implemented=False,training_performed=False,
        baseline=freeze,reference41r=prior['freeze'],quality=prior['quality'],compartment_gates=prior['compartment_gates'],passive_gates=prior['passive_gates'],
        dt_ms=.1,seeds=[17,29,43],thresholds_changed=False,abstraction_changed=False,independent47_accessed=False,
        corrected_target=dict(primary='Ca_LVAst.h',primary_index=3,negative_control='NaTs2_t.h',negative_control_index=7,reason=audit['correction']),
        hypotheses=hypotheses,arms=arms,additional_diagnostic_controls=['all-formula','41r unchanged','45b/f111 known rollback-regression control'],
        fixed_structure='Same18explicit gate states, recurrence map, current algebra, calcium and axial solver. Only existing relevant kinetic readout weights/biases can change; trunk, all other heads and NaTs2_t.h frozen. No online oracle, new basis, V-dependent head switching or altered precision policy.',
        phase_order=['Verify f011 byte/logical hashes and gate-index ownership; CPU/GPU materialization parity before scientific evaluation.',
            'Diagnostic coefficient/oracle probes on opened45b paths, including intermediate Ih-failed node. No new fitted weights during this phase.',
            'Generate separate FIT/DEV local histories with new seeds, including nominal/recovery, negative-tail and long relaxation; freeze hashes. Opened45b data are diagnostic audit fixtures, not an independent confirmation.',
            'Prepare eight component treatments in paired vectorized groups, checkpoint1/4 refinements along same trajectories; report finite/conditioning/actual-forward diagnostics.',
            'DEV no-regression nomination and weight hashes; combine selected components only after this freeze.',
            'All coupled candidates and controls in one notebook, imposed/autonomous and coupling0/.25/1; original45 and safe45b panels,44 and original41 stress regressions. No47 access.'],
        fitting_design=dict(budget_checkpoints=[1,4],budget_unit='nonlinear frozen-feature refinement, not SGD steps',
            candidate_solver='FIT-only whitened regularized least-squares/Gauss-Newton in existing head space; coefficients plus exact causal gate/current sensitivities. Actual FP32 materialized models checked at each checkpoint.',
            paired_fit_seeds=[9453101,9453113,9453127],paired_dev_seeds=[9453201,9453211,9453229],
            nominal_voltage_mv=[-100,60],tail_voltage_mv=[-135,-100],nominal_fraction=.5,horizons_ms=[1,8,40,100,500,1000],
            preservation='Guarded Nap arms preserve functional responses, event crossings and hidden-state accuracy on nominal FIT histories against the teacher; a separate bounded f011 imitation term may preserve existing compensation. Guard acceptance is tested case-by-case on DEV with original gates, not assumed from loss.',
            saturation='Exact0/1 coefficient labels are censored and counts reported; evaluator probabilities/states are never clipped to hide failure.'),
        nomination='Highest DEV all-prefix joint pass count, no new failure on any f011-passing DEV case, then smallest worst original-gate severity, then earlier checkpoint/registered arm order. Baseline fallback if no eligible repair. No old stress or coupled-audit outcomes used to select weights. Report all candidates including rejected ones.',
        metrics=['Original current nRMSE/nmax/signed and absolute charge/onset/offset/peak criteria.','Per-gate RMSE/max/occupancy with channel+state names, not only pooled V or current.','Original V/Ca/native-floor/physical-domain criteria at every100/500/1000ms prefix.','Case-identity regression maps, event support and signed crossing shifts; intermediate-node slope/baseline diagnostics.','Parameter count and resident ionic/integrated execution costs separately; no speedup claim from batching overhead alone.'],
        source_transport='Reuse verified45b native/calibration fixtures via a bounded compact archive, preserving SHA/CRC/member hashes. f011 heads already exported as a small verified NPZ, not a4.6GB initialization dependency. Verify actual mounted input before GPU run; never use ambiguous archive.zip discovery.',
        limits=['45c is still a design, not an executed experiment or complete runnable notebook.','Numerical weights for losses/regularization, exact FIT history counts, transport manifest and executable preflight must be finalized and preregistered before fitting/dispatch.','No promise of zero regressions; that is the empirical acceptance requirement.','No restoration of original41 qualification, no automatic46/47 advancement.'])
    write_json(ROOT/'experiments/task45c_design_v1.json',plan)
    print('45c design prepared; no fitting, notebook launch or new results claimed.',flush=True)

if __name__=='__main__':main()
