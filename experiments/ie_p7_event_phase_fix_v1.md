# IE-P7 — EMS fixed-step event phase correction

The first run stopped at disjoint native calibration, before model evaluation
or timing. This is not evidence against any learned operator.

The independent generator and shadow both used continuous event-time receptor
convolution. Canonical EMS NMODL instead explicitly multiplies A/B by exp(-dt/tau)
once per state update. Off-grid NET_RECEIVE delivery and that update have a
fixed-step ordering. Plasticity still uses the true event timestamp.

Native NEURON 8.2.7 calibration shows three releases: exc_2 at 13.45 ms and
inh_1 at 6.45 and 14.45 ms. At the first recorded sample, the latter first
release has zero decay, whereas the others have one full dt of decay.
Accumulated floating midpoint clocks explain nominal half-grid ties falling
on different sides. Rounding event times or adding an arbitrary global shift
would be incorrect.

Correction: keep prescribed event times, RNG calls and plasticity unchanged;
use the native EMS fixed-step receptor age in both independent generation and
shadow audit. This mode is opt-in for IE-P7; historical Task33 defaults remain
unchanged. No checkpoint, quality threshold, event schedule, dt, or candidate
selection changed. Original preregistration bytes are preserved.

Archived calibration trace regression reconstructs every receptor A/B state
within 1e-12; the old convolution differs by over .05 in receptor state.
Using native receptor states diagnostically gives formula V RMSE
2.1382655646397123e-6 mV, V maximum 4.506646334334619e-6 mV, versus the original
.04487592915681609 and .2119264656031845 mV. Native states are not used to feed
the rerun candidates. The independent forcing must pass the unchanged native
audit and formula floor before the candidate phase.

Six local unit tests passed, including archived native half-grid regression,
prefix causality, frozen operators and eager/compiled CPU equivalence.
A native-only Kaggle preflight precedes the full pinned rerun. New native
confirmation outcomes are not used for tuning or selecting operators.
