# IE-P7 compilation correction (version 3)

Version 2 passed native interface/formula gates, completed fresh and native
candidate evaluations, and saved 56 resident-core timing rows/preflights with
zero graph failures. It did not measure the staged operational workload:
Torch Dynamo failed its own guard on a NumPy-backed area scalar converted
inside the recurrent function under inference_mode.

Hoist float(area_um2) into stepper preparation and capture a Python float.
No equations, thresholds, candidate weights, events, RNG, dt, or budgets change.
This conversion is immutable setup metadata, not work belonging to each tick.
The regression deletes area metadata after constructing the stepper; eager and
compiled CPU execution must still agree. All six IE-P7 unit tests pass.
Real pinned GPU Inductor/CUDA Graph preflights remain mandatory in the rerun.

All six learned local maps passed native confirmation in version 2. This is
not a completed end-to-end speed result or automatic promotion. Archived
native_quality separates absolute_passed, noninferiority_passed, spike_passed
and the combined passed field. Some original hybrid controls meet absolute
gates but fail noninferiority; the short log reports absolute gates only.

Kaggle version 3 is pinned to 8d4852d68d2be154ee81d953ba52eb496b5584a0,
T4 and the same Docker image. Original preregistration remains byte-identical.
