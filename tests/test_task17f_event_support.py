import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.giada_teacher.roadmap_task17f_event_supported_confirmation import (
    Task17fConfig, compare_trials, make_spec, pilot_specs, resolve_spec, support_report,
)
from src.hayflow_teacher.event_extractor import extract_events

REPS = {"soma": 0, "ais": 640, "trunk": 361, "hot_zone": 387, "nexus": 437, "tuft": 460}


class Calibrator:
    def select_synapse_cluster(self, target, count, distance, mode):
        return SimpleNamespace(center_segment_id=469, target_segment_id=REPS[target],
                               synapse_ids=tuple(range(count)))


def test_nmda_detector_follows_actual_cluster_and_preserves_thresholds():
    config = Task17fConfig()
    spec = pilot_specs(config, "nmda_spike")[0]
    _, actions, signals, definitions, contract = resolve_spec(spec, Calibrator(), REPS, config)
    assert signals["event_probe"] == 469
    nmda = next(d for d in definitions if d.kind == "nmda_spike")
    assert (nmda.segment_id, nmda.signal, nmda.threshold, nmda.min_duration_ms) == (469, "event_probe", -40, 1)
    times = np.arange(0, 5, 0.025)
    traces = {name: np.full(len(times), -80.) for name in signals}
    traces["event_probe"][40:120] = -30
    events = extract_events(times, traces, definitions)
    assert [(e["kind"], e["segment_id"]) for e in events] == [("nmda_spike", 469)]
    assert sum(len(a) for a in actions.values()) == 20
    assert contract["synapse_ids"] == list(range(10))


def test_paired_current_is_explicit_and_weights_are_not_changed():
    spec = pilot_specs(Task17fConfig(), "calcium_spike", 2.0)[3]
    _, actions, _, _, contract = resolve_spec(spec, Calibrator(), REPS, Task17fConfig())
    currents = [a for rows in actions.values() for a in rows if a.kind == "somatic_current"]
    assert len(currents) == 2 and all(a.amplitude_na == 2 for a in currents)
    assert all(a.weight_multiplier == 1 for rows in actions.values() for a in rows if a.kind == "synaptic_event")
    assert contract["spec"]["candidate"]["pair_with_somatic_spike"] is False


def test_support_cannot_pass_by_pooling_seeds_or_censored_events():
    seeds = Task17fConfig().seeds
    kinds = ("somatic_spike", "calcium_spike", "nmda_spike")
    rows = [{"seed": seed, "gbar_multiplier": 1.,
             "event_contract": {"spec": {"target_kind": kind}}, "events": [
        {"kind": kind, "right_censored": False}]} for seed in seeds for kind in kinds]
    assert support_report(rows, seeds)["valid"]
    rows[-1]["events"][-1]["right_censored"] = True
    assert not support_report(rows, seeds)["valid"]
    rows[-1]["gbar_multiplier"] = 1.5
    assert not support_report(rows, seeds)["valid"]


def test_freeze_configuration_rejects_posthoc_seed_or_threshold_changes():
    Task17fConfig().validate()
    with pytest.raises(ValueError):
        replace(Task17fConfig(), candidate_voltage_rmse_limit_mv=3).validate()
    assert not set(Task17fConfig().seeds) & set(Task17fConfig().pilot_seeds)


def test_causal_contrast_rejects_probe_input_changes_and_false_positives():
    trace = {str(s): {"v": [0., 0.], "m": [0., 0.], "h": [0., 0.],
                      "ica_hva": [0., 0.]} for s in (0, 387, 460, 469)}
    reference = {"traces": trace, "schedule_sha256": "a", "events": [],
                 "event_traces": {"soma": [0., 0.]}, "release_audit": {"events": []}}
    other = copy.deepcopy(reference)
    assert compare_trials(reference, other, Task17fConfig())["passed"]
    other["events"] = [{"kind": "somatic_spike", "onset_ms": 1., "right_censored": False}]
    assert not compare_trials(reference, other, Task17fConfig())["passed"]
    other["schedule_sha256"] = "changed"
    with pytest.raises(RuntimeError):
        compare_trials(reference, other, Task17fConfig())
