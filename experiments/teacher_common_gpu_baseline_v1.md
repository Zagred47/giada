# GIADA Task 0.6 — common GPU baseline

- Seeds: `[17, 29, 43]`
- Batch size: **1024**
- Maximum optimizer steps: **10,000**
- Checkpoints: `[0, 100, 300, 1000, 3000, 10000]`
- Learning-rate grid: `[0.01, 0.003, 0.001]`
- Primary training precision: **float32 without AMP**
- Teacher/reference precision: **float64**

## Fairness firewall

Learned arms receive identical split membership, minibatch indices, batch boundaries, seeds, optimizer family, learning-rate trial count, checkpoint budgets and metrics. Analytic baselines receive the identical evaluation rows but no fictitious optimizer budget.

Development data may select hyperparameters and checkpoints. Sealed tests are opened only after code, weights, normalization and thresholds are frozen; they select nothing.

## GPU timing

Eager and `torch.compile` results are reported separately. CUDA events and synchronization are mandatory; compile time, steady-state kernel latency, end-to-end throughput and peak memory are distinct measurements.

Contract valid: **True**
