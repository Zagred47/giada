import json
from pathlib import Path

from src.giada_runpod.store import LeanShardWriter
from src.giada_runpod.surrogate_validity_audit import audit_corpus, audit_shard


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

