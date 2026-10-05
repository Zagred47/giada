# Gate D: compiled bottleneck forensic

The diagnostic was preregistered and run on Kaggle's Tesla T4 at commit
`b19aa65fec6d6d1ce200f6c590e5d2cd376fbfb0`. It preserves the parent
compiled Gate D NO-GO; it is not a new promotion test.

| Family | Five formulas | Five learned channels | Full formulas | Full hybrid | Kernel events / four calls, full exact → hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 0.1404 ms | 0.3833 ms | 0.3338 ms | 0.5948 ms | 56 → 80 |
| Shared heads | 0.1732 ms | 0.6273 ms | 0.4234 ms | 0.8682 ms | 55 → 87 |

The learned core alone is slower (2.73× independent, 3.62× shared heads),
and the whole hybrid remains slower (1.78× and 2.05×). The full hybrid's
top CUDA events include Volta SGEMM kernels absent from the top exact events;
the formula path uses fused Triton pointwise kernels. This supports a
small-batch GEMM/launch/fusion cost explanation, but is not proof that any
single kernel or mechanism accounts for all of the latency gap. Component
latencies cannot be added because Inductor fuses compositions differently.

The output decomposition, 40-sample timing traces, source hashes, ZIP CRC,
code revision and scientific status were independently audited in
`scripts/audit_gate_d_forensic_result.py`. The earlier compiled trial had
already failed the Gate D compute criterion; this run localizes why.

Next hypothesis: only a revised primitive that reduces GEMM/launch overhead
or amortizes it over a larger valid workload could challenge the compiled
formula control. Such a candidate requires a newly preregistered, same-backend
and accuracy-gated comparison. Task29 remains unauthorized.
