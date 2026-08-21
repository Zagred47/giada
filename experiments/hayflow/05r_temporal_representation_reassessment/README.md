# HayFlow 05r - temporal-representation reassessment

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `7126ab8540934a60483bb4bcf3d64a9e5e6cc90364322a4df0b3ad70b84dff64`;
- artifact-index SHA-256: `2b2e731006373ba1ec53a0cbd8099e9a269b1e095e75039d35c043bf8de88912`;
- final-report SHA-256: `5c876547680eaee2bd8c12e73f63bf381294c3a9cf7e181fcc34468182c019c5`;
- all five indexed members passed size and digest verification.

The frozen checkpoints reproduced with exactly 0.0 mV error and the complete
state sketch reproduced within `2.38e-7`.  Mechanism-state removal degraded
8 ms RMSE by 4.93% median and using mechanism states alone improved over zero
state by 5.21%; both effects were positive in all three seeds.  Calcium/ion
and synapse-state effects were only 0.64% and 0.17%.  A +1 ms teacher-state
shift was immaterial (0.17%), whereas +4 ms was material (2.39%).  The result
therefore registers `TEMPORAL_STATE_SIGNAL_LOCALIZED_TO_ONE_CATEGORY` and
authorizes one bounded `05s_mechanism_states_state_encoder_canary`.

05q verified that the joint model depends materially on the identity of its
initial teacher state, but not materially on the explicit axial block.  05r
uses the same frozen joint checkpoints to determine what the compressed state
represents and how time-specific it is.

The verified 64-dimensional sketch is decomposed additively by canonical
teacher-state category using the original signed projection and the original
per-segment denominator.  The sum of category contributions must reconstruct
both the complete recomputed sketch and the stored 05p sketch within 1e-5.
For every seed the joint checkpoint is evaluated with:

- the complete initial-state sketch;
- each category removed separately;
- each category supplied alone against the zero-state reference;
- the authentic state shifted forward by one or four teacher milliseconds.

The time shifts deliberately use future teacher state only as a frozen
diagnostic counterfactual.  They are never proposed as deployment inputs and
cannot authorize a trained model.  Category materiality requires at least 2%
median degradation, regenerative non-inferiority and positive evidence in at
least two seeds.

Only train-derived development windows are evaluated.  No optimization,
checkpoint selection, validation/test access or fresh-test access occurs.
The experiment can identify only the next bounded state-encoder canary or a
projection-stability reassessment.

Expected artifact: `hayflow_temporal_representation_reassessment.zip`.
