# HayFlow 06a-b - atomic voltage-path identifiability

Status: completed; artifact independently verified; optimization factor identified.

Verified artifact:

- archive SHA-256: `935c5114c553ddc8658032cf8cac10f868f28929055d87c636967de398f01b1f`;
- artifact-index SHA-256: `95d348e1bc7d4a7709592f32fc41354544993ca715e284fbb07df498469f52f5`;
- final-report SHA-256: `4a4bdfa7660fe8f128c7e15a9be148ebd1a8876fa836a86b53a7ee471461ff22`;
- all 13 indexed members passed size and digest verification.

The run is valid and registered as `ATOMIC_STATE_WAS_OPTIMIZATION_LIMITED`.
Increasing the fixed budget from 300 to 1200 steps raised one-step improvement
over persistence from `2.82%` to `16.12%` in the linear-endpoint arm and from
`2.88%` to `15.96%` in the teacher-microtrace arm. The paired optimization
effects (`13.31` and `13.08` percentage points) exceed the preregistered
one-point threshold. The teacher path was `0.16` points worse than linear
endpoint interpolation, so intra-ms path information was not identified.

The result is not confined to abundant coordinates. At the long budget,
semantic-macro gain is about `7.44%` and active-coordinate gain about `16.60%`.
Nested long-budget rollout gains are `14.1%`, `16.9%`, `17.4%` and `21.1%` for
the linear endpoint arm at 1/2/4/8 ms, with no non-finite or state-domain
violations. Neither calibration curve reached the registered plateau.

This experiment remains privileged: both arms know the teacher endpoint
voltage. It establishes that the detailed intra-ms path adds no benefit once
that endpoint is known, but it does not establish that the purely causal
start-voltage updater is learnable at the longer budget. The registered next
step, `06b_optimized_explicit_state_updater_canary`, must therefore restore a
causal deployment-compatible arm and retain the endpoint arm only as a paired
diagnostic reference. It must also use multiple seeds and report semantic-group
robustness. Full-neuron training and held-out access remain unauthorized.

The downloaded artifact contains one harmless documentary inconsistency:
`voltage_path_contract.json` inherited the old 06a names in `arms`. The correct
operational field, `voltage_context_arms`, names the two arms actually executed,
and all reports/checkpoints agree with it. The implementation now overrides the
legacy field for future runs; no rerun of 06a-b is required.
