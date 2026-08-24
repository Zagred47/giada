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
