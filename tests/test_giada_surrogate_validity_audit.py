import json
from pathlib import Path

from src.hayflow_data import InputAction
from src.giada_runpod.config import load_scale_config
from src.giada_runpod.store import LeanShardWriter, validate_lean_shard
from src.giada_runpod.surrogate_validity_audit import (
    audit_corpus,
    audit_shard,
    verify_output_spike_corpus,
)
from src.giada_runpod.teacher import detect_neuronio_spike_peaks
from src.giada_runpod.hybrid_inputs import (
    NEURONIO_COMPATIBLE_TARGET_PROTOCOLS,
    sample_neuronio_contextualized_target_actions,
)


def _write_shard(root: Path, *, break_events: bool = False) -> Path:
    path = root / "shards" / "shard-00000.h5"
    metadata = {
        "storage_profile": "soma_paper",
        "mechanism_group_names": ["m0", "m1"],
        "ion_names": ["i0"],
    }
    writer = LeanShardWriter(
        path,
        segment_count_per_transition=1,
        mechanism_group_count=2,
        ion_count=1,
        schema_metadata=metadata,
    )
    events = [{
        "kind": "synaptic_event",
        "synapse_id": 10,
        "segment_id": 37,
        "offset_ms": 0.25,
        "release_success": True,
        "released_quantity": 1.0,
        "ampa_state_increment": 0.5,
        "nmda_state_increment": 0.2,
        "inhibitory_state_increment": 0.0,
    }, {
        "kind": "synaptic_event",
        "synapse_id": 11,
        "segment_id": 38,
        "offset_ms": 0.75,
        "release_success": False,
        "released_quantity": 0.0,
        "ampa_state_increment": 0.0,
        "nmda_state_increment": 0.0,
        "inhibitory_state_increment": 0.0,
    }]
    rows = [(-70.0, -60.0, -20.0), (-60.0, -50.0, -50.0)]
    for step, (start, end, maximum) in enumerate(rows):
        writer.append({
            "segment_id": [0],
            "voltage_t_mv": [start],
            "voltage_t_plus_1_mv": [end],
            "voltage_min_mv": [min(start, end)],
            "voltage_max_mv": [maximum],
            "parent_delta_t_mv": [0.0],
            "mean_child_delta_t_mv": [0.0],
            "mechanism_state_t": [[0.1, 0.2]],
            "ion_state_t": [[0.3]],
            "causal_drive": [[1.0] + [0.0] * 11],
            "trajectory_index": 5,
            "step_index": step,
            "seed": 123,
            "split_code": 0,
            "scheduled_event_count": 2 if step == 0 else 0,
            "realized_event_count": 1 if step == 0 else 0,
            "high_resolution_sample_count": 41,
        }, events if step == 0 and not break_events else [])
    writer.close(expected_transition_count=2)
    return path


def test_shard_distinguishes_presynaptic_events_and_spike_recoverability(tmp_path):
    report = audit_shard(_write_shard(tmp_path))
    assert report["valid"]
    assert report["event_table_rows"] == 2
    assert report["release_success_count"] == 1
    assert report["release_failure_count"] == 1
    assert report["realized_successful_events"] == 1
    assert report["hidden_above_minus25_excursions_detectable_from_max"] == 1
    assert report["endpoint_minus55_upcrossings"] == 1
    assert not report["explicit_output_spike_times_present"]
    assert report["step_continuity_gaps"] == 0
    assert report["voltage_boundary_discontinuities"] == 0


def test_missing_event_rows_are_a_blocker(tmp_path):
    report = audit_shard(_write_shard(tmp_path, break_events=True))
    assert not report["valid"]
    assert any("scheduled_event_count" in item for item in report["blockers"])


def test_corpus_resolves_composite_and_samples_read_only(tmp_path):
    component = tmp_path / "targeted"
    _write_shard(component)
    composite = tmp_path / "composite"
    composite.mkdir()
    (composite / "composite_manifest.json").write_text(json.dumps({
        "components": [{"component_id": "targeted", "root": "../targeted"}]
    }), encoding="utf-8")
    report = audit_corpus(composite, sample_shards_per_component=1)
    assert report["valid"] and report["read_only"]
    assert report["components"]["targeted"]["sampled_shards"] == 1
    assert not report["interpretation"]["original_neuronio_spike_times_recoverable_exactly"]


def test_neuronio_peak_detector_handles_interior_and_boundary_peaks():
    times = [index * 0.125 for index in range(9)]
    interior = [-70, -60, -10, -20, -40, -50, -60, -65, -66]
    spikes, previous = detect_neuronio_spike_peaks(times, interior)
    assert spikes == [{"offset_ms": 0.25, "peak_voltage_mv": -10.0}]
    boundary = [-5, -20, -40, -50, -60, -65, -66, -67, -68]
    spikes, _ = detect_neuronio_spike_peaks(
        times, boundary, previous_voltage_mv=-30.0
    )
    assert spikes == [{"offset_ms": 0.0, "peak_voltage_mv": -5.0}]
    assert previous == -65.0


def test_output_spike_extension_roundtrips(tmp_path):
    path = tmp_path / "shards" / "spike.h5"
    writer = LeanShardWriter(
        path,
        segment_count_per_transition=1,
        mechanism_group_count=1,
        ion_count=1,
        schema_metadata={"storage_profile": "soma_paper"},
        store_output_spikes=True,
    )
    writer.append({
        "segment_id": [0], "voltage_t_mv": [-60.0],
        "voltage_t_plus_1_mv": [-50.0], "parent_delta_t_mv": [0.0],
        "mean_child_delta_t_mv": [0.0], "mechanism_state_t": [[0.1]],
        "ion_state_t": [[0.2]], "causal_drive": [[0.0] * 12],
        "trajectory_index": 0, "step_index": 0, "seed": 1, "split_code": 1,
        "scheduled_event_count": 0, "realized_event_count": 0,
        "high_resolution_sample_count": 9,
    }, [], output_spikes=[{"offset_ms": 0.25, "peak_voltage_mv": 30.0}])
    completion = writer.close(expected_transition_count=1)
    report = audit_shard(path)
    validation = validate_lean_shard(path, expected_transition_count=1)
    assert completion["output_spike_count"] == 1
    assert report["explicit_output_spike_times_present"]
    assert report["explicit_output_spike_rows"] == 1
    assert report["output_spike_detector"] == (
        "neuronio_local_maximum_above_minus25mv_at_0.125ms"
    )
    assert validation["valid"]
    corpus = verify_output_spike_corpus(tmp_path)
    assert corpus["valid"]
    assert corpus["shards"] == 1
    assert corpus["transitions"] == 1
    assert corpus["explicit_output_spikes"] == 1
    assert corpus["shards_with_spikes"] == 1
    assert corpus["high_resolution_sample_count_histogram"] == {"9": 1}


def test_surrogate_validity_configs_are_all_test_and_balanced():
    config_root = Path(__file__).parents[1] / "runpod_scale" / "configs"
    background = load_scale_config(config_root / "surrogate_validity_eval_background.yml")
    targeted = load_scale_config(config_root / "surrogate_validity_eval_targeted.yml")
    assert background.validation_trajectory_fraction == 1.0
    assert targeted.validation_trajectory_fraction == 1.0
    assert background.trajectory_count == 60
    assert targeted.trajectory_count == 3000
    contextual = load_scale_config(
        config_root / "surrogate_validity_eval_neuronio_contextual.yml"
    )
    assert contextual.validation_trajectory_fraction == 1.0
    assert contextual.trajectory_duration_ms == 720
    assert contextual.trajectory_count == 400
    assert tuple(contextual.input_protocols) == NEURONIO_COMPATIBLE_TARGET_PROTOCOLS


def test_contextual_target_has_visible_history_and_shifted_stimulus(monkeypatch):
    background = {1: (InputAction("synaptic_event", 0.0, synapse_id=0),)}
    target = {20: (InputAction("synaptic_event", 0.25, synapse_id=10),)}
    monkeypatch.setattr(
        "src.giada_runpod.hybrid_inputs.sample_neuronio_actions",
        lambda *args, **kwargs: (background, {"sampler": "test"}),
    )
    monkeypatch.setattr(
        "src.giada_runpod.hybrid_inputs.sample_hybrid_actions",
        lambda *args, **kwargs: (target, {"stimulus_step": 20}),
    )
    mapping = object()
    schedule, metadata = sample_neuronio_contextualized_target_actions(
        720,
        mapping,
        seed=1,
        protocol=NEURONIO_COMPATIBLE_TARGET_PROTOCOLS[0],
    )
    assert 1 in schedule
    assert 520 in schedule
    assert metadata["contextual_stimulus_step"] == 520
