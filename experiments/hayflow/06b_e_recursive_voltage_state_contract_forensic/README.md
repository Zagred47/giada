# HayFlow 06b-e - recursive voltage/STATE contract forensic

The registered 06b-d experiment established learnable one-step voltage/STATE
coupling but failed its primary 8 ms transfer gate. This follow-up deliberately
does not add another trainable architecture. It restores the exact 1500-step
06b-d checkpoints and evaluates one frozen `2 x 2 x 2` boundary matrix.

The crossed factors are teacher versus predicted voltage feedback, teacher
versus predicted mechanism-STATE feedback, and current teacher ions versus the
initial ion context held through the window. All five frozen 06b-d objective
and schedule arms are evaluated on the same 16 nested train-derived development
windows, paired seeds and realized external input sequence.

This single matrix separates exposure bias, voltage recurrence, missing ion
dynamics, factor interactions, optimizer-trajectory effects and causal
specificity. No model is trained or selected. The fully fed-back cell is still
not a complete autonomous neuron: realized external input remains known at
each step and synaptic internal state is not closed.

Notebook: `notebooks/06b_e_recursive_voltage_state_contract_forensic.ipynb`.

## Registered result

The returned archive from revision `cc94d49` passed ZIP CRC and all ten indexed
member-size and SHA-256 checks. The preregistered primary outcome is decisive:
feeding predicted mechanism STATE instead of resetting to teacher STATE raises
8 ms normalized STATE RMSE by 481% at the median, in the same direction for all
three seeds. Mechanism-STATE exposure is therefore a genuine temporal limit.

The direct voltage measurements require a systems-level safety amendment to
the automatic diagnosis. With predicted voltage, predicted STATE and held
ions, STATE still improves over persistence by 25.56% at the median. Voltage,
however, is 46.46% worse than persistence in every seed, reaches roughly
26.0 mV RMSE and produces 314 endpoint values outside the registered physical
range. Predicted voltage is partially compensating STATE error while becoming
physically wrong; this cannot be accepted as successful recurrence.

The formal primary diagnosis remains `MECHANISM_STATE_EXPOSURE_PRIMARY_LIMIT`.
The systems-level decision is stricter:
`JOINT_MECHANISM_STATE_EXPOSURE_AND_VOLTAGE_FEEDBACK_LIMITS`. This is recorded
transparently as a post-execution safety amendment, not retroactively presented
as a preregistered voltage gate.

The other controls narrow the next intervention. Holding ions adds only 0.98%
median STATE error and is not a first-order limit at 8 ms. Constant versus
cosine differs by only 0.64 points and is not identified. Authentic joint
coupling retains a strong 7.11-point STATE advantage over the shuffled control
under full feedback, so its causal signal should be preserved.

Only one bounded train-only joint repair matrix is authorized. It must cross a
mechanism-STATE exposure repair with direct stable-voltage recurrence
constraints and retain a shuffled causal control. A STATE-only repair, 06c,
full training, fresh-test generation and mass-data generation remain
prohibited.
