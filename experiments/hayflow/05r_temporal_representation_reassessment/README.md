# HayFlow 05r - temporal-representation reassessment

Status: implemented; independent Kaggle execution pending.

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
