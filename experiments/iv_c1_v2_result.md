# IV-C1 v2 — state timing remains unresolved

The prospectively preregistered v2 run at `f28d3bc` was technically valid: clean checkout, process exit 0, report identical to ZIP copy. Calibration selected conductance phase 0 (4.01e-18 uS maximum error; phase 1: 7.76e-5 uS). All 12 confirmation schedule×voltage cells matched conductance (max 1.28e-17 uS), current (1.11e-15 nA), and charge (7.38e-15 nA ms) to numerical precision. Clamp deviation was at most 0.000207 mV, and all three negative controls passed.

**IV-C1 nevertheless remains failed.** Every cell missed the unchanged A/B-state gate; maximum error remained 0.7537896848, the normalized single AMPA jump. The v2 left-continuous event-time hypothesis therefore did not generalize. The cause is not yet established: NetCon delivery may align differently for different scheduled timestamps or the state recorder may have a distinct intra-step ordering. The v1/v2 confirmation schedules are now diagnostic data and cannot be reused as a sealed test for a revised rule.

Next action: a forensic native state trace around each event, including the actual `h.t` sample and separate A/B values for each synapse; only after that audit should a new prospective state/event contract and disjoint confirmation schedule be prepared. IV-C2, IV-C3 and Task33 remain unauthorized. Report and ZIP: `experiments/results/iv_c1_v2_kaggle_f28d3bc/`.
