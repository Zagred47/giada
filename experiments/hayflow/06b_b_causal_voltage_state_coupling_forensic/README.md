# HayFlow 06b-b - causal voltage/STATE coupling forensic

Status: completed; artifact independently verified; voltage representation
forensic required before coupled training.

Verified artifact:

- archive SHA-256: `d652c3fdf088569b212c6fc710185ab4f870857e5e84cc2947459b7f456bb349`;
- artifact-index SHA-256: `824cc0fdfb977c69fed7bbf3dfcea691f6c1346c84fadedd988aa20c12c56986`;
- final-report SHA-256: `41a1fc65c2bc1b812e06758f171983e8201b2627905fdd6a7b0813c442bacc2e`;
- all 14 indexed members passed size and digest verification.

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

The executed bridge produced a real causal one-step signal but did not pass the
full gate. Predicted endpoint paths improved frozen STATE performance over the
causal updater by `2.75--4.91` percentage points in all three seeds and beat the
same predictions shuffled across transitions by `8.79--14.38` points. They
recovered `33.3--49.3%` of the teacher-endpoint oracle gap. Thus additional
capacity alone cannot explain the gain: transition-aligned voltage information
is useful.

Median global voltage gain was only `7.12%` against the registered `10%` gate,
although active-voltage gains were `12.77--15.18%`. More importantly, at 8 ms
the predicted path was worse than the frozen causal updater in two of three
seeds (`-9.57%`, `-5.80%`, `+3.50%`). The registered diagnosis is therefore
`CAUSAL_VOLTAGE_BRIDGE_NOT_PREDICTIVE`, and 06c is not authorized.

The next bounded experiment is
`06b_c_voltage_bridge_representation_forensic`. It must separate optimization
budget from missing topology/axial support while keeping the six STATE
checkpoints frozen, retaining the shuffled control and remaining train-only.
