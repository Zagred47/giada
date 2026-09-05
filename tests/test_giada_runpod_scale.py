import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.giada_runpod.config import ScaleConfig
from src.giada_runpod.neuronio_inputs import (
    PILOT_PROTOCOLS,
    DendriticSynapseMap,
    neuronio_input_config_for_protocol,
    sample_neuronio_actions,
)
from src.giada_runpod.planning import build_shard_plan, load_shard_plan, write_shard_plan
from src.giada_runpod.store import LeanShardWriter, validate_lean_shard
from src.giada_runpod.training import LeanSomaCorpus
from src.giada_runpod import teacher as teacher_module
from src.giada_runpod.corpus_audit import audit_soma_corpus


def test_s1_plan_is_exact_disjoint_and_roundtrips(tmp_path: Path) -> None:
    config = ScaleConfig(
        stage="s1",
        target_transitions=600_000,
        trajectories_per_shard=1,
    )
    shards = build_shard_plan(config)
    assert len(shards) == 100
    assert all(len(shard.trajectories) == 1 for shard in shards)
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


def test_s1b_support_pilot_is_paired_within_every_factorial_cell() -> None:
    config = ScaleConfig(
        stage="s1b_pilot",
        target_transitions=96_000,
        trajectories_per_shard=1,
        validation_trajectory_fraction=0.5,
        purpose="input_support_pilot",
        input_protocols=PILOT_PROTOCOLS,
    )
    trajectories = [
        row for shard in build_shard_plan(config) for row in shard.trajectories
    ]
    assert len(trajectories) == 16
    for protocol in PILOT_PROTOCOLS:
        rows = [row for row in trajectories if row.protocol == protocol]
        assert len(rows) == 2
        assert {row.split for row in rows} == {"train", "validation"}


def test_s1b_factorial_protocols_condition_only_canonical_support() -> None:
    for protocol in PILOT_PROTOCOLS:
        config = neuronio_input_config_for_protocol(protocol)
        config.validate()
        assert 0.0 <= config.basal_excitatory_per_100ms[0]
        assert config.basal_excitatory_per_100ms[1] <= 800.0
        assert -600.0 <= config.basal_inhibitory_difference_per_100ms[0]
        assert config.basal_inhibitory_difference_per_100ms[1] <= 200.0
        assert set(config.rate_intervals_ms) <= set(
            neuronio_input_config_for_protocol(
                "neuronio_nmda_ergodic_v1"
            ).rate_intervals_ms
        )
        assert set(config.smoothing_sigmas_ms) <= set(
            neuronio_input_config_for_protocol(
                "neuronio_nmda_ergodic_v1"
            ).smoothing_sigmas_ms
        )


def test_s1b_factorial_protocols_sample_without_inverted_rate_bounds() -> None:
    count = 639
    mapping = DendriticSynapseMap(
        segment_ids=np.arange(1, count + 1),
        segment_lengths_um=np.linspace(2.0, 20.0, count),
        is_basal=np.arange(count) < 250,
        excitatory_synapse_ids=np.arange(count),
        inhibitory_synapse_ids=np.arange(count, 2 * count),
    )
    for protocol in PILOT_PROTOCOLS:
        for seed in range(32):
            _, metadata = sample_neuronio_actions(
                40,
                mapping,
                seed=seed,
                config=neuronio_input_config_for_protocol(protocol),
                protocol=protocol,
            )
            assert metadata["protocol"] == protocol


def test_runpod_teacher_session_uses_base_contract_without_calibration_artifacts(
    tmp_path: Path, monkeypatch,
) -> None:
    captured = {}

    def fake_base_init(self, elm_repo, teacher_repo, **kwargs):
        captured.update(kwargs)
        self.seed = int(kwargs["seed"])

    monkeypatch.setattr(
        teacher_module.DiagnosticDatasetSession, "__init__", fake_base_init
    )
    session = teacher_module.RunPodCausalTeacherSession(
        tmp_path / "giada",
        tmp_path / "teacher",
        output_dir=tmp_path / "runtime",
        seed=123,
    )
    assert captured["output_dir"] == tmp_path / "runtime"
    assert captured["expected_teacher_hashes"]
    assert session.active_random123_seed == 123
    assert not hasattr(session, "calibration_source")


def test_runpod_teacher_session_tracks_snapshot_random123_seed(monkeypatch) -> None:
    observed = {}

    def fake_configure(self, seed, sequences):
        observed["seed"] = seed
        observed["sequences"] = list(sequences)

    monkeypatch.setattr(
        teacher_module.DiagnosticDatasetSession, "_configure_rngs", fake_configure
    )
    session = object.__new__(teacher_module.RunPodCausalTeacherSession)
    session._configure_rngs(456, [1.0, 2.0])
    assert observed == {"seed": 456, "sequences": [1.0, 2.0]}
    assert session.active_random123_seed == 456


def test_ordered_segment_voltages_reads_values_not_mapping_keys() -> None:
    live_segments = {
        2: SimpleNamespace(v=-72.0),
        0: SimpleNamespace(v=-76.0),
        1: SimpleNamespace(v=-74.0),
    }
    observed = teacher_module.ordered_segment_voltages(live_segments)
    np.testing.assert_array_equal(observed, [-76.0, -74.0, -72.0])


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


def test_soma_corpus_audit_reports_split_activity_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    path = root / "shards" / "shard-00000.h5"
    metadata = {
        "storage_profile": "soma_paper",
        "mechanism_group_names": ["g0"],
        "ion_names": ["i0"],
        "causal_drive_features": ["u0"],
        "segment_ids": [0],
        "mechanism_presence": [[1]],
        "segment_static": [[0.0]],
        "region_names": ["soma"],
        "segment_region_ids": [0],
    }
    writer = LeanShardWriter(
        path,
        segment_count_per_transition=1,
        mechanism_group_count=1,
        ion_count=1,
        schema_metadata=metadata,
        chunk_transitions=2,
    )
    for step, (split, start, end) in enumerate(
        ((0, -56.0, -54.0), (0, -70.0, -69.5), (1, -70.0, -64.0))
    ):
        writer.append(
            {
                "segment_id": [0],
                "voltage_t_mv": [start],
                "voltage_t_plus_1_mv": [end],
                "parent_delta_t_mv": [0.0],
                "mean_child_delta_t_mv": [0.0],
                "mechanism_state_t": [[0.2]],
                "ion_state_t": [[0.01]],
                "causal_drive": [[0.0]],
                "trajectory_index": step,
                "step_index": 0,
                "seed": step,
                "split_code": split,
                "scheduled_event_count": 1,
                "realized_event_count": 1,
            },
            [],
        )
    writer.close(expected_transition_count=3)
    plan_path = root / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "trajectories": [
                            {
                                "trajectory_index": 0,
                                "split": "train",
                                "protocol": "p0",
                            },
                            {
                                "trajectory_index": 1,
                                "split": "train",
                                "protocol": "p1",
                            },
                            {
                                "trajectory_index": 2,
                                "split": "validation",
                                "protocol": "p1",
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = audit_soma_corpus(root, plan_path=plan_path)
    assert report["valid"]
    assert report["splits"]["train"]["transition_count"] == 2
    assert report["splits"]["train"]["somatic_upcrossings_minus55mv"] == 1
    assert report["splits"]["validation"]["absolute_delta_ge_5mv_count"] == 1
    assert report["protocol_splits"]["p0"]["train"]["transition_count"] == 1
    assert report["protocol_splits"]["p1"]["validation"][
        "absolute_delta_ge_5mv_count"
    ] == 1
