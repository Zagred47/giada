# GIADA paper-scale validation track

This experiment family is isolated from the sequential Kaggle architecture
research under `experiments/hayflow/`. Its purpose is confirmation by data
scale, not architecture selection.

The starting observation is the valid 06b-c information-matched comparison:
the 8,985-parameter GIADA voltage bridge reduced authentic one-step soma RMSE
from a Branch-ELM median of 14.824 mV to 9.577 mV on the small train-derived
development support, a paired median reduction of 34.94%. The result ranked
the two compact voltage paths but did not establish a complete neuron
surrogate and was not directly comparable to the 64 biological hours of the
published NeuronIO train/validation corpus.

RunPod S0--S4 asks whether that ranking survives increasing amounts of teacher
time while keeping the numerical input tensor, target, sample order, loss,
optimizer family, and seeds aligned. No result in this track authorizes a new
architecture. Architecture hypotheses remain in the Kaggle/Allen-Zhu track.

Operational source, configs, and instructions live in `runpod_scale/` and
`src/giada_runpod/`. Generated HDF5 shards and checkpoints never enter Git.
