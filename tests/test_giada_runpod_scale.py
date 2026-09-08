import hashlib
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
from src.giada_runpod.hybrid_inputs import (
    HYBRID_PROTOCOLS,
    PROTOCOL_REPAIR_PROTOCOLS,
    PRODUCTION_BACKGROUND_PROTOCOLS,
    PRODUCTION_TARGET_PROTOCOLS,
    hybrid_protocol_spec,
    sample_hybrid_actions,
)
from src.giada_runpod.planning import build_shard_plan, load_shard_plan, write_shard_plan
from src.giada_runpod.store import LeanShardWriter, validate_lean_shard
from src.giada_runpod.training import (
    LeanSomaCorpus,
    MatchedTrainingConfig,
    PaperScaleMatchedTrainer,
)
from src.giada_runpod import teacher as teacher_module
from src.giada_runpod.corpus_audit import audit_soma_corpus
from src.giada_runpod.production_corpus import (
    PRODUCTION_PROFILES,
    fingerprint_validated_shards,
)


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


def test_s1c_hybrid_plan_keeps_causal_arms_in_same_snapshot_group() -> None:
    config = ScaleConfig(
        stage="s1c_hybrid_pilot",
        target_transitions=15_360,
        trajectory_duration_ms=80,
        trajectories_per_shard=1,
        storage_profile="spatial_probe",
        sampled_segments_per_transition=7,
        validation_trajectory_fraction=0.5,
        purpose="giada_hybrid_pilot",
        input_protocols=HYBRID_PROTOCOLS,
    )
    rows = [row for shard in build_shard_plan(config) for row in shard.trajectories]
    assert len(rows) == 192
    assert {row.split for row in rows} == {"train", "validation"}
    assert not (
        {row.seed for row in rows if row.split == "train"}
        & {row.seed for row in rows if row.split == "validation"}
    )
    groups = {}
    for row in rows:
        groups.setdefault(row.pair_id, []).append(row)
    assert len(groups) == 96
    for group in groups.values():
        assert len({row.seed for row in group}) == 1
        assert len({row.split for row in group}) == 1
        family = group[0].protocol_family
        assert {row.protocol_arm for row in group} == {
            hybrid_protocol_spec(protocol).arm
            for protocol in HYBRID_PROTOCOLS
            if hybrid_protocol_spec(protocol).family == family
        }


def test_s1c_hybrid_actions_are_deterministic_and_causally_matched() -> None:
    count = 639
    mapping = DendriticSynapseMap(
        segment_ids=np.arange(1, count + 1),
        segment_lengths_um=np.linspace(2.0, 20.0, count),
        is_basal=np.arange(count) < 250,
        excitatory_synapse_ids=np.arange(count),
        inhibitory_synapse_ids=np.arange(count, 2 * count),
    )
    materialized = {}
    for protocol in HYBRID_PROTOCOLS:
        first, metadata = sample_hybrid_actions(
            80, mapping, seed=9200001, protocol=protocol
        )
        second, _ = sample_hybrid_actions(
            80, mapping, seed=9200001, protocol=protocol
        )
        encoded = {
            step: [row.to_dict() for row in actions]
            for step, actions in first.items()
        }
        assert encoded == {
            step: [row.to_dict() for row in actions]
            for step, actions in second.items()
        }
        assert metadata["canonical_synaptic_weights_unchanged"]
        materialized[protocol] = encoded
    assert len(materialized["giada_hybrid_nmda_n8_v1"][20]) == 8
    assert len(materialized["giada_hybrid_nmda_n12_v1"][20]) == 12
    assert all(
        step >= 19
        for protocol, schedule in materialized.items()
        if not hybrid_protocol_spec(protocol).family.startswith("background_")
        for step in schedule
    )


def test_s1d_repair_plan_is_complete_paired_and_split_disjoint() -> None:
    config = ScaleConfig(
        stage="s1d_protocol_repair_pilot",
        target_transitions=11_520,
        trajectory_duration_ms=80,
        trajectories_per_shard=1,
        storage_profile="spatial_probe",
        sampled_segments_per_transition=7,
        validation_trajectory_fraction=0.5,
        purpose="giada_protocol_repair_pilot",
        input_protocols=PROTOCOL_REPAIR_PROTOCOLS,
    )
    rows = [row for shard in build_shard_plan(config) for row in shard.trajectories]
    assert len(rows) == 144
    assert not (
        {row.seed for row in rows if row.split == "train"}
        & {row.seed for row in rows if row.split == "validation"}
    )
    groups = {}
    for row in rows:
        groups.setdefault(row.pair_id, []).append(row)
    assert len(groups) == 32
    for group in groups.values():
        family = group[0].protocol_family
        assert len({row.seed for row in group}) == 1
        assert len({row.split for row in group}) == 1
        assert {row.protocol_arm for row in group} == {
            hybrid_protocol_spec(protocol).arm
            for protocol in PROTOCOL_REPAIR_PROTOCOLS
            if hybrid_protocol_spec(protocol).family == family
        }


def test_s1d_repair_actions_match_immutable_calibrations() -> None:
    count = 639
    mapping = DendriticSynapseMap(
        segment_ids=np.arange(1, count + 1),
        segment_lengths_um=np.linspace(2.0, 20.0, count),
        is_basal=np.arange(count) < 250,
        excitatory_synapse_ids=np.arange(count),
        inhibitory_synapse_ids=np.arange(count, 2 * count),
    )
    schedules = {}
    for protocol in PROTOCOL_REPAIR_PROTOCOLS:
        first, metadata = sample_hybrid_actions(
            80, mapping, seed=9_250_001, protocol=protocol
        )
        second, _ = sample_hybrid_actions(
            80, mapping, seed=9_250_001, protocol=protocol
        )
        encoded = {
            step: [row.to_dict() for row in actions]
            for step, actions in first.items()
        }
        assert encoded == {
            step: [row.to_dict() for row in actions]
            for step, actions in second.items()
        }
        assert metadata["canonical_synaptic_weights_unchanged"]
        schedules[protocol] = first

    negative = schedules["giada_repair_somatic_3na_negative_v1"][20][0]
    positive = schedules["giada_repair_somatic_6na_positive_v1"][20][0]
    assert negative.amplitude_na == 3.0
    assert positive.amplitude_na == 6.0

    assist = schedules["giada_repair_bap_assist_only_n12_b3_w400_v1"]
    assert set(assist) == {20, 21, 22}
    assert all(len(assist[step]) == 12 for step in assist)
    assert {
        action.synapse_id for actions in assist.values() for action in actions
    } == set(range(764, 781, 2)) | {872, 874, 936}

    p2 = schedules["giada_repair_bap_p2_factor3_soma_only_v1"]
    p3 = schedules["giada_repair_bap_p3_factor3_soma_only_v1"]
    assert set(p2) == {19, 20}
    assert set(p3) == {19, 20, 21}
    assert all(actions[0].amplitude_na == 9.0 for actions in p2.values())
    assert all(actions[0].amplitude_na == 9.0 for actions in p3.values())

    historical = schedules["giada_repair_bap_p3_factor2_combined_v1"]
    assert set(historical) == {19, 20, 21, 22}
    assert sum(
        action.kind == "somatic_current"
        for actions in historical.values()
        for action in actions
    ) == 3
    assert sum(
        action.kind == "synaptic_event"
        for actions in historical.values()
        for action in actions
    ) == 36


def test_s1e_hybrid_production_plan_has_exact_registered_composition() -> None:
    background = ScaleConfig(
        stage="s1e_hybrid_background",
        target_transitions=360_000,
        trajectory_duration_ms=6000,
        trajectories_per_shard=1,
        validation_trajectory_fraction=0.2,
        purpose="giada_hybrid_production_background",
        input_protocols=PRODUCTION_BACKGROUND_PROTOCOLS,
    )
    targeted = ScaleConfig(
        stage="s1e_hybrid_targeted",
        target_transitions=240_000,
        trajectory_duration_ms=80,
        trajectories_per_shard=25,
        validation_trajectory_fraction=0.2,
        purpose="giada_hybrid_production_targeted",
        input_protocols=PRODUCTION_TARGET_PROTOCOLS,
    )
    background_rows = [
        row for shard in build_shard_plan(background) for row in shard.trajectories
    ]
    targeted_rows = [
        row for shard in build_shard_plan(targeted) for row in shard.trajectories
    ]
    assert len(background_rows) == 60
    assert len(targeted_rows) == 3000
    for protocol in PRODUCTION_BACKGROUND_PROTOCOLS:
        rows = [row for row in background_rows if row.protocol == protocol]
        assert sum(row.split == "train" for row in rows) == 24
        assert sum(row.split == "validation" for row in rows) == 6
    for protocol in PRODUCTION_TARGET_PROTOCOLS:
        rows = [row for row in targeted_rows if row.protocol == protocol]
        assert sum(row.split == "train" for row in rows) == 200
        assert sum(row.split == "validation" for row in rows) == 50
    assert not (
        {row.seed for row in targeted_rows if row.split == "train"}
        & {row.seed for row in targeted_rows if row.split == "validation"}
    )
    groups = {}
    for row in targeted_rows:
        groups.setdefault(row.pair_id, []).append(row)
    for group in groups.values():
        assert len({row.seed for row in group}) == 1
        assert len({row.split for row in group}) == 1
        family = group[0].protocol_family
        expected_arms = {
            hybrid_protocol_spec(protocol).arm
            for protocol in PRODUCTION_TARGET_PROTOCOLS
            if hybrid_protocol_spec(protocol).family == family
        }
        assert {row.protocol_arm for row in group} == expected_arms


def test_s1e_target_registry_excludes_failed_s1c_transcriptions() -> None:
    assert "giada_hybrid_somatic_spike_v1" not in PRODUCTION_TARGET_PROTOCOLS
    assert "giada_hybrid_bap_combined_v1" not in PRODUCTION_TARGET_PROTOCOLS
    assert "giada_repair_somatic_6na_positive_v1" in PRODUCTION_TARGET_PROTOCOLS
    assert "giada_repair_bap_p2_factor3_combined_v1" in PRODUCTION_TARGET_PROTOCOLS


def test_s1e_training_config_forbids_validation_checkpoint_selection() -> None:
    MatchedTrainingConfig(
        required_composite_stage="s1e_hybrid_production",
        checkpoint_selection="final_preregistered",
    ).validate()
    try:
        MatchedTrainingConfig(checkpoint_selection="best_validation").validate()
    except ValueError as error:
        assert "checkpoint selection" in str(error)
    else:  # pragma: no cover
        raise AssertionError("validation checkpoint selection was accepted")


def test_s2_hybrid_plan_is_exact_sixfold_expansion_without_seed_leakage() -> None:
    background = ScaleConfig(
        stage="s2_hybrid_background",
        target_transitions=2_160_000,
        trajectory_duration_ms=6000,
        trajectories_per_shard=1,
        root_seed=9_500_001,
        validation_trajectory_fraction=0.2,
        purpose="giada_hybrid_production_background",
        input_protocols=PRODUCTION_BACKGROUND_PROTOCOLS,
    )
    targeted = ScaleConfig(
        stage="s2_hybrid_targeted",
        target_transitions=1_440_000,
        trajectory_duration_ms=80,
        trajectories_per_shard=25,
        root_seed=9_600_001,
        validation_trajectory_fraction=0.2,
        purpose="giada_hybrid_production_targeted",
        input_protocols=PRODUCTION_TARGET_PROTOCOLS,
    )
    background_rows = [
        row for shard in build_shard_plan(background) for row in shard.trajectories
    ]
    targeted_rows = [
        row for shard in build_shard_plan(targeted) for row in shard.trajectories
    ]
    assert len(background_rows) == 360
    assert len(targeted_rows) == 18_000
    assert background.shard_count == 360
    assert targeted.shard_count == 720
    for protocol in PRODUCTION_BACKGROUND_PROTOCOLS:
        rows = [row for row in background_rows if row.protocol == protocol]
        assert sum(row.split == "train" for row in rows) == 144
        assert sum(row.split == "validation" for row in rows) == 36
    for protocol in PRODUCTION_TARGET_PROTOCOLS:
        rows = [row for row in targeted_rows if row.protocol == protocol]
        assert sum(row.split == "train" for row in rows) == 1200
        assert sum(row.split == "validation" for row in rows) == 300
    train_seeds = {row.seed for row in background_rows + targeted_rows if row.split == "train"}
    validation_seeds = {
        row.seed for row in background_rows + targeted_rows if row.split == "validation"
    }
    assert not train_seeds & validation_seeds
    assert PRODUCTION_PROFILES["s2"]["splits"] == {
        "train": 2_880_000,
        "validation": 720_000,
    }


def test_s2_training_scales_exposure_and_preregisters_breadth() -> None:
    frozen_hashes = {
        "background_plan_sha256": "0" * 64,
        "background_validation_sha256": "1" * 64,
        "targeted_plan_sha256": "2" * 64,
        "targeted_validation_sha256": "3" * 64,
        "composite_manifest_sha256": "4" * 64,
        "production_audit_sha256": "5" * 64,
        "shard_marker_fingerprint_sha256": "6" * 64,
    }
    config = MatchedTrainingConfig(
        seeds=(61017, 61029, 61043, 61071, 61103),
        training_steps=18_000,
        checkpoints=(100, 300, 600, 1000, 1800, 3000, 6000, 18_000),
        required_composite_stage="s2_hybrid_production",
        minimum_seed_wins=4,
        minimum_family_wins=5,
        minimum_protocol_wins=12,
        scaling_reference_seeds=(61017, 61029, 61043),
        expected_corpus_hashes=frozen_hashes,
    )
    config.validate()
    assert config.training_steps * config.batch_size / 2_880_000 == 25.6


def test_s3_hybrid_plan_is_exact_eightfold_s2_expansion() -> None:
    background = ScaleConfig(
        stage="s3_hybrid_background",
        target_transitions=17_280_000,
        trajectory_duration_ms=6000,
        trajectories_per_shard=1,
        root_seed=9_700_001,
        validation_trajectory_fraction=0.2,
        purpose="giada_hybrid_production_background",
        input_protocols=PRODUCTION_BACKGROUND_PROTOCOLS,
    )
    targeted = ScaleConfig(
        stage="s3_hybrid_targeted",
        target_transitions=11_520_000,
        trajectory_duration_ms=80,
        trajectories_per_shard=25,
        root_seed=9_800_001,
        validation_trajectory_fraction=0.2,
        purpose="giada_hybrid_production_targeted",
        input_protocols=PRODUCTION_TARGET_PROTOCOLS,
    )
    background_rows = [
        row for shard in build_shard_plan(background) for row in shard.trajectories
    ]
    targeted_rows = [
        row for shard in build_shard_plan(targeted) for row in shard.trajectories
    ]
    assert len(background_rows) == 2_880
    assert len(targeted_rows) == 144_000
    assert background.shard_count == 2_880
    assert targeted.shard_count == 5_760
    for protocol in PRODUCTION_BACKGROUND_PROTOCOLS:
        rows = [row for row in background_rows if row.protocol == protocol]
        assert sum(row.split == "train" for row in rows) == 1_152
        assert sum(row.split == "validation" for row in rows) == 288
    for protocol in PRODUCTION_TARGET_PROTOCOLS:
        rows = [row for row in targeted_rows if row.protocol == protocol]
        assert sum(row.split == "train" for row in rows) == 9_600
        assert sum(row.split == "validation" for row in rows) == 2_400
    train_seeds = {
        row.seed for row in background_rows + targeted_rows if row.split == "train"
    }
    validation_seeds = {
        row.seed
        for row in background_rows + targeted_rows
        if row.split == "validation"
    }
    assert not train_seeds & validation_seeds
    assert PRODUCTION_PROFILES["s3"]["splits"] == {
        "train": 23_040_000,
        "validation": 5_760_000,
    }


def test_s3_training_preserves_exposure_and_evaluates_full_validation() -> None:
    frozen_hashes = {
        "background_plan_sha256": "0" * 64,
        "background_validation_sha256": "1" * 64,
        "targeted_plan_sha256": "2" * 64,
        "targeted_validation_sha256": "3" * 64,
        "composite_manifest_sha256": "4" * 64,
        "production_audit_sha256": "5" * 64,
        "shard_marker_fingerprint_sha256": "6" * 64,
    }
    config = MatchedTrainingConfig(
        seeds=(61017, 61029, 61043, 61071, 61103),
        training_steps=144_000,
        checkpoints=(
            100, 300, 600, 1000, 1800, 3000, 6000, 18_000,
            36_000, 72_000, 144_000,
        ),
        evaluation_sample_limit=5_760_000,
        required_composite_stage="s3_hybrid_production",
        minimum_seed_wins=4,
        minimum_family_wins=5,
        minimum_protocol_wins=12,
        require_spike_transition_advantage=True,
        scaling_reference_seeds=(61017, 61029, 61043),
        expected_corpus_hashes=frozen_hashes,
    )
    config.validate()
    assert config.training_steps * config.batch_size / 23_040_000 == 25.6
    assert config.evaluation_sample_limit == 5_760_000


def test_frozen_s3_training_config_matches_preregistration() -> None:
    import yaml

    values = yaml.safe_load(
        Path("runpod_scale/configs/s3_matched_training.yml").read_text(
            encoding="utf-8"
        )
    )["giada_matched_training"]
    config = MatchedTrainingConfig.from_mapping(values)
    assert config.required_composite_stage == "s3_hybrid_production"
    assert config.training_steps == 144_000
    assert config.training_steps * config.batch_size / 23_040_000 == 25.6
    assert config.evaluation_sample_limit == 5_760_000
    assert config.require_spike_transition_advantage
    assert config.expected_corpus_hashes[
        "shard_marker_fingerprint_sha256"
    ] == "4196dbfd08f2f6d75963caa14a6b7d0b1de7e1a771a55611bd442a3c52ea4fb0"


def test_corpus_fingerprint_covers_markers_and_physical_shards(tmp_path: Path) -> None:
    composite = tmp_path / "composite"
    composite.mkdir()
    components = []
    for component_id in ("background", "targeted"):
        component = tmp_path / component_id
        (component / "status").mkdir(parents=True)
        (component / "shards").mkdir()
        shard = component / "shards" / "shard-00000.h5"
        shard.write_bytes(f"physical-{component_id}".encode())
        marker = {
            "shard_id": "shard-00000",
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        }
        (component / "status" / "shard-00000.done.json").write_text(
            json.dumps(marker, sort_keys=True), encoding="utf-8"
        )
        components.append({"component_id": component_id, "root": f"../{component_id}"})
    (composite / "composite_manifest.json").write_text(
        json.dumps({"components": components}), encoding="utf-8"
    )
    progress = []
    first = fingerprint_validated_shards(
        composite, progress=lambda index, total: progress.append((index, total))
    )
    assert first["valid"]
    assert first["shard_count"] == 2
    assert len(first["marker_fingerprint_sha256"]) == 64
    assert progress == [(1, 2), (2, 2)]
    (tmp_path / "targeted" / "shards" / "shard-00000.h5").write_bytes(b"changed")
    second = fingerprint_validated_shards(composite)
    assert not second["valid"]
    assert second["physical_mismatch_count"] == 1


def test_training_corpus_verification_progress_is_throttled(capsys) -> None:
    for index in (1, 2, 199, 200, 201, 400, 401, 432):
        PaperScaleMatchedTrainer._report_corpus_fingerprint_progress(index, 432)
    assert capsys.readouterr().out.splitlines() == [
        "[GIADA RunPod][training corpus verification] 1/432 shards",
        "[GIADA RunPod][training corpus verification] 200/432 shards",
        "[GIADA RunPod][training corpus verification] 400/432 shards",
        "[GIADA RunPod][training corpus verification] 432/432 shards",
    ]


def test_s1e_training_verifies_sealed_composite_contract(tmp_path: Path) -> None:
    root = tmp_path / "composite"
    root.mkdir()
    audit_path = root / "production_audit.json"
    audit_path.write_text(
        json.dumps({"valid": True, "blockers": []}), encoding="utf-8"
    )
    (root / "composite_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "giada-runpod-composite-corpus-v1",
                "stage": "s1e_hybrid_production",
                "valid": True,
                "total_transition_count": 600000,
                "split_transition_counts": {"train": 480000, "validation": 120000},
                "production_audit": "production_audit.json",
            }
        ),
        encoding="utf-8",
    )
    report = PaperScaleMatchedTrainer._validate_corpus_contract(
        root,
        MatchedTrainingConfig(
            required_composite_stage="s1e_hybrid_production"
        ),
    )
    assert report["verified"]
    assert len(report["manifest_sha256"]) == 64
    assert len(report["production_audit_sha256"]) == 64


def test_s2_training_rejects_stage_substitution(tmp_path: Path) -> None:
    root = tmp_path / "composite"
    root.mkdir()
    (root / "production_audit.json").write_text(
        json.dumps({"valid": True, "blockers": []}), encoding="utf-8"
    )
    (root / "composite_manifest.json").write_text(
        json.dumps({
            "schema_version": "giada-runpod-composite-corpus-v1",
            "stage": "s1e_hybrid_production",
            "valid": True,
            "total_transition_count": 600_000,
            "production_audit": "production_audit.json",
        }),
        encoding="utf-8",
    )
    try:
        PaperScaleMatchedTrainer._validate_corpus_contract(
            root,
            MatchedTrainingConfig(
                required_composite_stage="s2_hybrid_production"
            ),
        )
    except RuntimeError as error:
        assert "wrong composite stage" in str(error)
    else:  # pragma: no cover
        raise AssertionError("S1e corpus was accepted as S2")


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


def test_logical_composite_reads_both_components_without_index_collision(
    tmp_path: Path,
) -> None:
    metadata = {
        "storage_profile": "soma_paper",
        "mechanism_group_names": ["g0"],
        "ion_names": ["i0"],
        "causal_drive_features": [f"u{i}" for i in range(12)],
        "segment_ids": [0],
        "mechanism_presence": [[1]],
        "segment_static": [[0.0]],
        "region_names": ["soma"],
        "segment_region_ids": [0],
    }
    component_roots = []
    for component_index, component_id in enumerate(("background", "targeted")):
        root = tmp_path / component_id
        component_roots.append(root)
        writer = LeanShardWriter(
            root / "shards" / "shard-00000.h5",
            segment_count_per_transition=1,
            mechanism_group_count=1,
            ion_count=1,
            schema_metadata=metadata,
            chunk_transitions=2,
        )
        for local_index, split in enumerate((0, 1)):
            writer.append(
                {
                    "segment_id": [0],
                    "voltage_t_mv": [-70.0],
                    "voltage_t_plus_1_mv": [-69.0 + component_index],
                    "parent_delta_t_mv": [0.0],
                    "mean_child_delta_t_mv": [0.0],
                    "mechanism_state_t": [[0.2]],
                    "ion_state_t": [[0.01]],
                    "causal_drive": [[0.0] * 12],
                    "trajectory_index": local_index,
                    "step_index": 0,
                    "seed": component_index * 100 + local_index,
                    "split_code": split,
                    "scheduled_event_count": 0,
                    "realized_event_count": 0,
                },
                [],
            )
        writer.close(expected_transition_count=2)
        (root / "validation_report.json").write_text(
            json.dumps({"valid": True, "validated_shard_count": 1, "validated_transition_count": 2}),
            encoding="utf-8",
        )
        (root / "plan.json").write_text(
            json.dumps({
                "shards": [{
                    "trajectories": [
                        {"trajectory_index": 0, "split": "train", "protocol": f"{component_id}_train"},
                        {"trajectory_index": 1, "split": "validation", "protocol": f"{component_id}_validation"},
                    ]
                }]
            }),
            encoding="utf-8",
        )
    composite = tmp_path / "composite"
    composite.mkdir()
    (composite / "composite_manifest.json").write_text(
        json.dumps({
            "schema_version": "giada-runpod-composite-corpus-v1",
            "valid": True,
            "components": [
                {"component_id": name, "root": f"../{name}", "plan": "plan.json"}
                for name in ("background", "targeted")
            ],
        }),
        encoding="utf-8",
    )
    report = audit_soma_corpus(composite)
    assert report["valid"]
    assert report["shard_count"] == 2
    assert report["splits"]["train"]["transition_count"] == 2
    assert report["splits"]["validation"]["transition_count"] == 2
    corpus = LeanSomaCorpus(composite)
    try:
        assert corpus.train_count == 2
        assert corpus.validation_count == 2
        labeled = list(corpus.iter_raw(1, 2, include_labels=True))
        assert {
            str(label)
            for chunk in labeled
            for label in chunk["_component_label"]
        } == {"background", "targeted"}
        assert {
            str(label)
            for chunk in labeled
            for label in chunk["_protocol_label"]
        } == {"background_validation", "targeted_validation"}
    finally:
        corpus.close()
