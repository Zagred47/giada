# Task30 first run: technically valid, not decision-grade

The Kaggle run at commit `fb08789` completed and the passive-only NEURON
solver matched the registered semi-implicit passive solver within
`1.41e-12 mV`. The formula reference remained within `[-76, -6.53] mV`.
The reported 8ms primary errors were tiny, and 20–80ms autonomous errors
were also small in the registered secondary diagnostics.

However, an independent protocol audit found **zero injected-current
samples before the 8ms primary horizon in all 32 episodes**. The low and
high pulses began at 10ms; the paired pulse began at 8ms, after the last
state counted in the primary metric. Thus the nominal primary pass cannot
establish active-stimulus voltage retention. The result is technically valid
but **not decision-grade**. No Task31 or speed promotion follows from it.

The prospective Task30b confirmation uses new initial voltages and calcium
levels, moves active stimuli to 1–8ms, requires exposure in all 24 non-rest
episodes, and keeps the frozen checkpoints and 8ms numerical thresholds
unchanged. Its protocol is `experiments/task30b_active_exposure_confirmation.json`.
