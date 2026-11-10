"""Generate the prospective45b contract; never changes original45 gates/results."""
import itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    old=json.loads((ROOT/'experiments/task45_preregistration_v1.json').read_text())
    proposal=json.loads((ROOT/'experiments/task45b_proposed_contrasts_v1.json').read_text())
    p={k:old[k] for k in ('freeze','seeds','dt_ms','duration_ms','coupling','geometries','quality','compartment_gates','passive_gates','prefix_horizons_ms')}
    p.update(task='45b',version='v1',exploratory=True,execution_authorized=True,training_authorized=True,
        authorization='User explicitly requested45b implementation, SQLite registration, push and Kaggle launch.',
        thresholds_changed=False,abstraction_changed=False,independent47_accessed=False,
        hypotheses=proposal['hypotheses'],arms=['none']+['f'+''.join(map(str,b)) for b in itertools.product((0,1),repeat=3)],
        factorial_axes=['Nap_h learned rate-head rollback','Ih learned inf-head repair','Ih learned rate-head repair'],
        reference='f000 = unchanged41r; none = all-formula execution control',
        panels=['original45','safe_hyper','regression44'],
        original45_protocols=old['protocols'],
        calibration=dict(target_minima_mv=[-90.,-110.,-130.],amplitude_bracket_ma_cm2=[-.008,0.],bisection_steps=12,
            geometry=old['geometries'][0]['id'],coupling=0.,
            rule='Native-only disconnected root chain4, same conditioning and40:200ms pulse. Bisect for target minimum over whole1000ms. Keep safe side (minimum >=target). Freeze resulting three current densities, COMMON to every geometry and coupling. Do not inspect candidate results before freeze. Other geometries retain explicit physical-domain checks.'),
        fit=dict(points=2048,dev_points_per_bin=256,fit_seed=45001,dev_seed=45002,
            nominal_bounds_mv=[-100.,60.],tail_bounds_mv=[-135.,-100.],supports=['nominal','tail_balanced'],
            tail_fraction=.5,checkpoints=[500,2000],optimizer='Adam',learning_rate=.01,gradient_clip=1.,ridge=1e-8,
            trainable_columns=[10,27],gate=10,
            objective='Mean squared raw equilibrium/update-logit error, two heads independent; ridge on whitened delta coordinates. Labels from float64 canonical coefficients at quantized float32 V; Ca fixed1e-4 (Ih is voltage-only). Frozen trunk and every other head.',
            pairing='Same uniform quantiles and full-batch update indices; independent Adam moments and per support/seed/head gradient clipping. Both checkpoint budgets from same trajectory, no restarts.',
            conditioning='FIT-only SVD whitening with singular-value floor1e-5; identical transformed feature basis for Adam and ridge least-squares diagnostic. LS is not selectable.',
            development='Independent stratified quantiles per nominal/tail bin, disjoint float32 voltage values; persist arrays and hashes before fitting.',
            nomination='One common support/budget for all3seeds, minimum worst(seed,voltage-bin,head) development logit RMSE; ties earlier checkpoint then nominal. No coupled/stress outcomes used. If no finite candidate, stop, do not silently nominate oracle.'),
        atomic=dict(voltages_mv=[-135.,-130.,-120.,-110.,-106.,-100.,-90.,-80.,-65.,-40.,0.,40.],
            initial_gate_states=[0.,.5,1.],horizons_ms=[100,500,1000],gates=[9,10],
            controls=['learned','oracle_inf','oracle_rate','oracle_both','Nap_inf_rollback_negative_control','Nap_rate_rollback','Nap_both_rollback'],
            path='Replay original45 regenerated, saved formula V/Ca histories: all9 cases, root/distal in all3 geometries. Free-gate recurrence vs teacher-reset one-step; same coefficients FP32 vs FP64. Canonical coefficient oracles diagnostic only, never selectable.'),
        regressions=dict(original45='All27 physical cases/81model-seed cases retained, including9 invalid regimes;72 interpretable identities unchanged.',
            task44='Opened44 three geometries,200ms, same protocols/coupling; eight learned factorial arms and formula control, imposed/autonomous.',
            task41='Original72 stress fixtures for frozen41r and all8 factorial materialized models, same original quality gates. Previously passing identities must not regress. Six known failures remain explicit; no retroactive qualification.'),
        persistence='Checkpoint optimizer/weights at500/2000; freeze before coupled evaluation. Native NPZ and progress per panel/topology. All original full-resolution metrics saved. Targeted full-resolution V/Ca, Ih/Nap currents and gate9/10 for every arm; all gates/currents for f000/f111. Formula/native references saved. No duplicated frozen-source cache in ZIP. Partial ZIP persisted on failure and external SHA/CRC readback; DB bundle local only.',
        native_input_policy='Regenerate opened45 and44 native fixtures with identical versioned inputs in native-only subprocess; avoids re-uploading1.2GB old result archive and ambiguous Kaggle mounted ZIPs. Not new independent data or retraining. New safe protocol calibrated before GPU candidate outcomes.',
        execution='One notebook, isolated native/GPU subprocesses. Full support/seed/head training matrix vectorized; coupled9arms×physicalcases×3seeds resident. Split original/safe/regression panels to cap memory. Original FP32 ionic/FP64 voltage policy unchanged. No runtime analytic coefficient oracle in learned arms.',
        limits=['No thresholds changed; original41/45 failures not erased.','Learned repairs are experimental sidecars, not replacement of frozen41r nominee.','No Task47 data, no automatic46 authorization, no speedup or release claim.','Single-transient timing evaluator unchanged; slope/pulse probes diagnostic only.','Same architecture and18explicit gates, no new features/states, current/Ca/axial equations unchanged.'])
    write_json(ROOT/'experiments/task45b_preregistration_v1.json',p)

if __name__=='__main__':main()
