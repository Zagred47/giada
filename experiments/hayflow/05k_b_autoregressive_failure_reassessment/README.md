# HayFlow 05k-b - autoregressive failure reassessment

Status: implemented and locally verified; Kaggle execution pending.

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
