# HayFlow 06b-b - causal voltage/STATE coupling forensic

Status: preregistered; not yet executed.

The completed 06b canary showed that its causal mechanism-STATE updater learns
a reproducible signal but retains only `44.3%` of the gain obtained when the
teacher endpoint voltage is supplied. This experiment asks the smallest causal
question implied by that result: can a compact predictor of the next 1 ms
voltage change recover a material part of that missing STATE performance?

All six 06b STATE updater checkpoints are loaded from the exact verified
artifact and frozen. The only trainable component is a shared per-segment
voltage bridge. It receives current voltage and axial differences, current
mechanism STATE, local ions, realized causal input, morphology and region. Its
training target is `V_(t+1) - V_t` on train-derived roles.

The primary `predicted_endpoint` mode feeds the bridge prediction into the
frozen endpoint-conditioned STATE updater. It is compared with the frozen
causal updater, the teacher-endpoint oracle, and a capacity-identical shuffled
control that destroys the transition-level alignment of the same predicted
delta voltage. Therefore a positive result must reflect causally aligned
voltage information, not merely extra parameters or another optimization run.

The 1/2/4/8 ms analysis recursively advances mechanism STATE on common nested
development windows. It deliberately uses teacher `V_t` at every millisecond,
because 06b-b does not yet contain an autonomous voltage state. This is a
component forensic, not a complete neuron rollout.

A pass authorizes only `06c_coupled_voltage_state_micro_canary`. Validation,
test outcomes, full-neuron training, fresh-test generation and mass data remain
prohibited.
