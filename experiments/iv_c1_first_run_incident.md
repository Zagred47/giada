# IV-C1 first-run technical interruption

The first Kaggle run at revision `8093d42` did not reach calibration: `VClamp` does not expose `rs` in NEURON 8.2.7. The native/frozen receptor hypothesis was **not judged**. The failure report and ZIP are preserved in `experiments/results/iv_c1_kaggle_8093d42/`.

The continuation changes the clamp object to `SEClamp`, whose `dur1`, `amp1` and `rs` are documented in the [official NEURON point-process reference](https://nrn.readthedocs.io/en/9.0.1/progref/modelspec/programmatic/mechanisms/mech.html). This is an execution-interface fix; the IV-C1 event schedules, receptor parameters, calibration/confirmation split, thresholds and decision rule are unchanged.
