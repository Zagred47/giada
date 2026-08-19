# HayFlow 05k - frozen candidate micro-rollout

Status: completed on Kaggle; autoregressive gate failed.

The 05j-o preregistered fresh test confirmed all three frozen decoder seeds as
one-step candidates. This experiment now evaluates the same unchanged
checkpoints autoregressively for 2, 4 and 8 ms from the frozen branch boundary.
All 32 fresh-test pairs and both arms are retained.

The teacher state encoder may initialize the first boundary state only. Every
later step feeds back the model voltage and recurrent H2 state; future teacher
states are never injected. The authentic synaptic front-end remains outside
the surrogate, exactly as planned for deployment: only its A/B rise-decay
state and causal `U_realized` events enter the core. NMDA block is evaluated
with the autoregressive model voltage. No future teacher voltage, ion,
mechanism or calcium state is exposed.
No seed is selected after observing rollout outcomes: all three seeds are
reported, and at least two must pass every horizon.

For each horizon a seed must improve endpoint RMSE by at least five percent
over the better of autoregressive H2 and persistence, keep median branching
retention in [0.5, 2.0], remain below the H2 maximum segment error and avoid
non-finite or physically invalid voltages. The equal-weight ensemble is
descriptive and does not participate in the gate.

A pass could have authorized only a separate limited rollout-aware training
canary. The observed result was instead a robust failure: zero of three seeds
passed all horizons. Endpoint RMSE was 13.78--16.58 mV at 2 ms and
62.72--95.47 mV at 8 ms, with a strong positive drift and non-physical
voltages at 8 ms. Persistence remained substantially better. No training or
mass dataset generation is authorized.

The registered next step is the diagnostic-only
`05k_b_autoregressive_failure_reassessment`.

Expected artifact: `hayflow_hines_frozen_candidate_micro_rollout.zip`.
