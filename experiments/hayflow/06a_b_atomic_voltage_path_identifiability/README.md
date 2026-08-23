# HayFlow 06a-b - atomic voltage-path identifiability

Status: implemented and preregistered; awaiting Kaggle execution.

The valid 06a pilot missed its registered 2% one-step gate, but two causes were
still confounded: both learning curves were improving at 300 steps, and neither
input represented the voltage trajectory inside the millisecond. Voltage-gated
mechanism kinetics depend on that trajectory, not only on its endpoints.

This forensic crosses two fixed factors. Each voltage context is trained once
to 1200 optimizer steps and evaluated at its best calibration checkpoint within
the first 300 and 1200 steps. The endpoint control receives eight values from a
linear interpolation between teacher boundary voltages. The path arm receives
the authentic teacher voltage at the same eight offsets. Their input width and
parameter count are identical, and the automatically narrowed hidden layer
keeps both models at or below the 7,238-parameter 06a ceiling.

Only the original train-derived fit, calibration and development roles are
read. The microtrace is explicitly privileged diagnostic information and makes
no deployment claim. Recursive state is evaluated on one common set of 8 ms
windows; 1, 2 and 4 ms results are prefixes of those exact windows rather than
independently sampled sets.

The preregistered 2% absolute gate and 1-percentage-point factor effects decide
whether optimization budget, the intra-ms voltage path, both, or neither
explain the 06a result. The experiment cannot authorize full training, held-out
access, fresh tests, capacity growth or mass data generation.
