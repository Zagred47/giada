# HayFlow 05k-b - autoregressive failure reassessment

Status: completed on Kaggle; closed-loop state distribution shift confirmed.

The frozen 05j-n decoder family passed the preregistered one-step fresh test
but failed the 05k closed-loop test for every seed. Error is already worse
than persistence at 2 ms and develops a large positive voltage drift by 8 ms.

This experiment does not train, tune, select or reinstate a model. It applies
a fixed causal intervention matrix to the same three frozen checkpoints:

- reset the complete teacher boundary state at every step (oracle);
- clamp only voltage to the teacher boundary while latent state remains closed-loop;
- reset only latent/calcium/synaptic recurrent state while voltage remains predicted;
- prevent decoder output from feeding back into H2;
- apply the decoder residual only at the first step.

The interventions distinguish temporal one-step support from voltage feedback,
latent-state drift and repeated residual compounding. Teacher interventions are
diagnostic oracles and are not deployable inputs. The consumed fresh test may
not be used to select a repaired model; any future candidate must be developed
on train/development data and evaluated on a new sealed test.

Expected artifact: `hayflow_hines_autoregressive_failure_reassessment.zip`.

All 44 indexed members were verified. Resetting the complete teacher boundary
state reduced median 8 ms error by 96.0%. Voltage-only and latent-only resets
each recovered roughly 70--77%, whereas suppressing decoder feedback recovered
only about 8%. The failure is therefore coupled voltage/latent closed-loop
shift, not simple residual compounding. No candidate or training was
authorized; the fresh test is consumed for diagnosis.
