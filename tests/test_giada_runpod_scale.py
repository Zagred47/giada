import json
from pathlib import Path

import numpy as np

from src.giada_runpod.config import ScaleConfig
from src.giada_runpod.neuronio_inputs import DendriticSynapseMap, sample_neuronio_actions
from src.giada_runpod.planning import build_shard_plan, load_shard_plan, write_shard_plan
from src.giada_runpod.store import LeanShardWriter, validate_lean_shard
from src.giada_runpod.training import LeanSomaCorpus


def test_s1_plan_is_exact_disjoint_and_roundtrips(tmp_path: Path) -> None:
    config = ScaleConfig(stage="s1", target_transitions=600_000)
    shards = build_shard_plan(config)
    assert sum(row.expected_transition_count for row in shards) == 600_000
    trajectories = [trajectory for shard in shards for trajectory in shard.trajectories]
    assert len(trajectories) == 100
    assert len({row.trajectory_id for row in trajectories}) == 100
    assert {row.split for row in trajectories} == {"train", "validation"}
    path = tmp_path / "plan.json"
    write_shard_plan(path, config, shards)
    loaded_config, loaded_shards = load_shard_plan(path)
    assert loaded_config == config
    assert loaded_shards == shards


def test_neuronio_sampler_is_seeded_and_preserves_canonical_event_fields() -> None:
    count = 639
    mapping = DendriticSynapseMap(
        segment_ids=np.arange(1, count + 1),
        segment_lengths_um=np.linspace(2.0, 20.0, count),
        is_basal=np.arange(count) < 250,
        excitatory_synapse_ids=np.arange(count),
        inhibitory_synapse_ids=np.arange(count, 2 * count),
    )
    first, first_meta = sample_neuronio_actions(120, mapping, seed=123)
    second, second_meta = sample_neuronio_actions(120, mapping, seed=123)
    assert first_meta == second_meta
    assert {
        step: [row.to_dict() for row in actions] for step, actions in first.items()
    } == {
        step: [row.to_dict() for row in actions] for step, actions in second.items()
    }
    assert all(0 <= step < 120 for step in first)
    assert all(row.offset_ms == 0.0 for actions in first.values() for row in actions)


def test_lean_shard_is_atomic_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "shard.h5"
    writer = LeanShardWriter(
        path,
        segment_count_per_transition=1,
        mechanism_group_count=3,
        ion_count=2,
        schema_metadata={"project": "GIADA"},
        compression="lzf",
        chunk_transitions=2,
    )
    for step in range(3):
        writer.append(
            {
                "segment_id": np.asarray([0]),
                "voltage_t_mv": np.asarray([-76.0 + step]),
                "voltage_t_plus_1_mv": np.asarray([-75.5 + step]),
                "parent_delta_t_mv": np.asarray([0.0]),
                "mean_child_delta_t_mv": np.asarray([0.1]),
                "mechanism_state_t": np.zeros((1, 3)),
                "ion_state_t": np.zeros((1, 2)),
                "causal_drive": np.zeros((1, 12)),
                "trajectory_index": 0,
                "step_index": step,
                "seed": 1,
                "split_code": 0,
                "scheduled_event_count": 0,
                "realized_event_count": 0,
            },
            [],
        )
    completion = writer.close(expected_transition_count=3)
    assert path.is_file() and not path.with_suffix(".h5.partial").exists()
    report = validate_lean_shard(path, expected_transition_count=3)
    assert report["valid"]
    assert report["sha256"] == completion["sha256"]


def test_corpus_sampling_with_replacement_avoids_h5py_duplicate_index_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    path = root / "shards" / "shard-00000.h5"
    metadata = {
        "storage_profile": "soma_paper",
        "mechanism_group_names": ["g0", "g1", "g2"],
        "ion_names": ["i0", "i1"],
        "causal_drive_features": [f"u{i}" for i in range(12)],
        "segment_ids": [0],
        "mechanism_presence": [[1, 1, 0]],
        "segment_static": [[0.0] * 7],
        "region_names": ["soma"],
        "segment_region_ids": [0],
    }
    writer = LeanShardWriter(
        path,
        segment_count_per_transition=1,
        mechanism_group_count=3,
        ion_count=2,
        schema_metadata=metadata,
        chunk_transitions=2,
    )
    for step, split in enumerate((0, 0, 1)):
        writer.append(
            {
                "segment_id": [0],
                "voltage_t_mv": [-76.0 + step],
                "voltage_t_plus_1_mv": [-75.0 + step],
                "parent_delta_t_mv": [0.0],
                "mean_child_delta_t_mv": [0.0],
                "mechanism_state_t": [[0.2, 0.8, 0.0]],
                "ion_state_t": [[0.01, 1.0]],
                "causal_drive": [[0.0] * 12],
                "trajectory_index": step,
                "step_index": 0,
                "seed": step,
                "split_code": split,
                "scheduled_event_count": 0,
                "realized_event_count": 0,
            },
            [],
        )
    writer.close(expected_transition_count=3)
    corpus = LeanSomaCorpus(root)
    try:
        sample = corpus.sample_raw(0, 100, np.random.default_rng(7))
        assert sample["voltage_t_mv"].shape == (100,)
    finally:
        corpus.close()
