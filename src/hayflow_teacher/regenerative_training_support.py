"""Outcome-blind train acquisition with a sealed fresh-test plan for 05j-m."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_data import ProtocolTrajectory, validate_hdf5_store, write_json
from ..hayflow_data.targeted_protocols import action_schedule_from_json
from .audit import git_commit, sha256_file
from .audit_runtime import PINNED_TEACHER_COMMIT
from .bap_support_topup import BapValidationSupportTopupSession, _sha256_with_progress
from .diagnostic_dataset import DiagnosticDatasetSession
from .diagnostic_dataset_v1 import canonical_json_sha256
EXPECTED_05JI_ARCHIVE_SHA256 = (
    "18235e9f643064df90fd7e305450635b5c255253a4958ce5b639433a558a8275"
)
EXPECTED_05JI_INDEX_SHA256 = (
    "70a0a72d51c52fd1fd0eb5d1c72c1332d8114cb241c8a9ec3bcaf82ac9839bce"
)
EXPECTED_05JI_FINAL_SHA256 = (
    "edb8ffcb60e12ab593f8761d2bf17fe8fefcf496479b090a8d445e8e47cbeeaf"
)
EXPECTED_05JL_ARCHIVE_SHA256 = (
    "f4685a3322dd868aaf2c271e7e8cb949e5ba81ca8d6b3631eb470fe473adb40c"
)
EXPECTED_05JL_INDEX_SHA256 = (
    "dbe30639cb88dfafcf2349a56d91489bf765aa425c0d47c35fefb973cef58c82"
)
EXPECTED_05JL_FINAL_SHA256 = (
    "2eed425e145307ad6f4a93f035a21490ae749397163281b3b61b22e14cb785f0"
)
REGENERATIVE_TRAINING_SCHEMA_VERSION = "05j-m-regenerative-training-support-v1"


@dataclass(frozen=True)
class RegenerativeTrainingSupportConfig:
    train_pair_count: int = 96
    fresh_test_pair_count: int = 32
    minimum_train_near_pair_count: int = 72
    episode_duration_ms: int = 12
    post_branch_ms: int = 6
    conditioning_ms: int = 4
    train_seed_start: int = 900_001
    fresh_test_seed_start: int = 1_100_001
    minimum_teacher_distance_mv: float = 0.01
    near_lower_mv: float = -45.0
    near_upper_mv: float = -20.0

    def validate(self) -> None:
        if self.train_pair_count < self.minimum_train_near_pair_count:
            raise ValueError("train pair count is below the near-regenerative minimum")
        if self.fresh_test_pair_count < 16:
            raise ValueError("fresh test requires at least 16 independent pairs")
        if self.episode_duration_ms <= self.post_branch_ms + 1:
            raise ValueError("episode duration leaves no post-branch follow-up")
        if min(
            self.conditioning_ms,
            self.train_seed_start,
            self.fresh_test_seed_start,
            self.minimum_teacher_distance_mv,
        ) <= 0:
            raise ValueError("positive acquisition values must be positive")
        if not self.near_lower_mv < self.near_upper_mv:
            raise ValueError("near-regenerative bounds are reversed")
        train = set(range(self.train_seed_start, self.train_seed_start + self.train_pair_count))
        test = set(
            range(
                self.fresh_test_seed_start,
                self.fresh_test_seed_start + self.fresh_test_pair_count,
            )
        )
        if train & test:
            raise ValueError("train and fresh-test seed namespaces overlap")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "RegenerativeTrainingSupportConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def _plan_rows(protocols: Sequence[ProtocolTrajectory]) -> List[Dict[str, Any]]:
    rows = []
    for trajectory in sorted(protocols, key=lambda item: item.trajectory_id):
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "category": trajectory.category,
                "protocol": trajectory.protocol,
                "protocol_id": trajectory.protocol_id,
                "protocol_variant": trajectory.protocol_variant,
                "seed": int(trajectory.seed),
                "duration_ms": int(trajectory.duration_ms),
                "split": trajectory.split,
                "stimulus_onset_step": int(trajectory.stimulus_onset_step),
                "snapshot_source": trajectory.snapshot_source,
                "metadata": dict(trajectory.metadata),
                "actions": {
                    str(step): [action.to_dict() for action in actions]
                    for step, actions in sorted(trajectory.actions_by_step.items())
                },
            }
        )
    return rows


def build_regenerative_support_plans(
    templates: Sequence[Mapping[str, Any]],
    config: RegenerativeTrainingSupportConfig,
) -> Tuple[List[ProtocolTrajectory], List[ProtocolTrajectory], Dict[str, Any]]:
    """Build train and fresh-test plans before observing either acquisition."""

    config.validate()
    frozen = [dict(row) for row in templates]
    if not frozen:
        raise ValueError("05j-i selected template catalog is empty")
    identities = {(str(row["candidate_id"]), int(row["branch_step"])) for row in frozen}
    if len(identities) != len(frozen):
        raise ValueError("05j-i selected templates contain duplicate identities")

    def build(role: str, pair_count: int, seed_start: int, split: str):
        protocols: List[ProtocolTrajectory] = []
        pairs = []
        for pair_index in range(pair_count):
            template = frozen[pair_index % len(frozen)]
            branch_step = int(template["branch_step"])
            if branch_step + config.post_branch_ms >= config.episode_duration_ms:
                raise ValueError("template branch leaves insufficient follow-up")
            source = action_schedule_from_json(template["input_schedule"])
            prefix = {step: actions for step, actions in source.items() if step < branch_step}
            seed = int(seed_start + pair_index)
            pair_id = f"05jm-{role}-pair-{pair_index:04d}"
            snapshot_id = f"05jm-{role}-snapshot-{pair_index:04d}"
            trajectory_ids = []
            for arm in ("low", "high"):
                branch_actions = action_schedule_from_json(
                    {str(branch_step): template[f"{arm}_actions"]}
                )[branch_step]
                schedule = {**prefix, branch_step: branch_actions}
                trajectory_id = f"{pair_id}-{arm}"
                trajectory_ids.append(trajectory_id)
                trajectory = ProtocolTrajectory(
                    trajectory_id=trajectory_id,
                    category="branching",
                    protocol=f"05jm_{template['family']}",
                    protocol_id=f"05jm-{template['candidate_id']}-step{branch_step}",
                    protocol_variant=f"near_regenerative_causal_{arm}_arm",
                    seed=seed,
                    duration_ms=config.episode_duration_ms,
                    split=split,
                    actions_by_step=schedule,
                    stimulus_onset_step=min(schedule, default=branch_step),
                    negative_control=arm == "low",
                    snapshot_source=snapshot_id,
                    metadata={
                        "episode_id": trajectory_id,
                        "snapshot_id": snapshot_id,
                        "branch_pair_id": pair_id,
                        "branching_distance": "near_regenerative_expansion",
                        "branch_id": str(template["branch_id"]),
                        "source_candidate_id": str(template["candidate_id"]),
                        "source_family": str(template["family"]),
                        "branch_step": branch_step,
                        "branch_arm": arm,
                        "event_probe_segment_id": int(template["event_probe_segment_id"]),
                        "event_probe_region": str(template["event_probe_region"]),
                        "selected_synapse_ids": list(map(int, template["selected_synapse_ids"])),
                        "acquisition_role": role,
                        "outcome_blind_plan": True,
                        "all_episodes_retained": True,
                    },
                )
                trajectory.validate()
                protocols.append(trajectory)
            pairs.append(
                {
                    "branch_pair_id": pair_id,
                    "snapshot_id": snapshot_id,
                    "seed": seed,
                    "source_candidate_id": str(template["candidate_id"]),
                    "source_family": str(template["family"]),
                    "branch_step": branch_step,
                    "trajectory_ids": trajectory_ids,
                }
            )
        return protocols, pairs

    train, train_pairs = build(
        "train", config.train_pair_count, config.train_seed_start, "train"
    )
    fresh, fresh_pairs = build(
        "fresh_test", config.fresh_test_pair_count, config.fresh_test_seed_start, "test"
    )
    train_seeds = {row.seed for row in train}
    fresh_seeds = {row.seed for row in fresh}
    if train_seeds & fresh_seeds:
        raise RuntimeError("train and fresh-test protocol seeds overlap")
    train_rows, fresh_rows = _plan_rows(train), _plan_rows(fresh)
    contract = {
        "schema_version": REGENERATIVE_TRAINING_SCHEMA_VERSION,
        "policy": "fixed_templates_outcome_blind_train_and_sealed_fresh_test",
        "configuration": asdict(config),
        "train": {
            "pair_count": len(train_pairs),
            "episode_count": len(train),
            "transition_count": sum(row.duration_ms for row in train),
            "seed_start": min(train_seeds),
            "seed_end": max(train_seeds),
            "pairs": train_pairs,
            "protocol_plan_sha256": canonical_json_sha256({"protocols": train_rows}),
        },
        "fresh_test": {
            "pair_count": len(fresh_pairs),
            "episode_count": len(fresh),
            "transition_count": sum(row.duration_ms for row in fresh),
            "seed_start": min(fresh_seeds),
            "seed_end": max(fresh_seeds),
            "pairs": fresh_pairs,
            "protocol_plan_sha256": canonical_json_sha256({"protocols": fresh_rows}),
            "protocols": fresh_rows,
            "outcomes_generated": False,
            "must_not_be_loaded_during_training": True,
        },
        "seed_overlap": sorted(train_seeds & fresh_seeds),
        "all_plans_frozen_before_train_acquisition": True,
    }
    contract["contract_sha256"] = canonical_json_sha256({"contract": contract})
    return train, fresh, contract


class RegenerativeTrainingSupportSession(BapValidationSupportTopupSession):
    """Generate only train outcomes while sealing the independent test plan."""

    def __init__(
        self,
        *args: Any,
        artifact_05ji_source: Path,
        artifact_05jl_source: Path,
        acquisition_config: RegenerativeTrainingSupportConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        acquisition_config.validate()
        self.artifact_05ji_source = Path(artifact_05ji_source).resolve()
        self.artifact_05jl_source = Path(artifact_05jl_source).resolve()
        self.acquisition_config = acquisition_config
        self.train_protocols: List[ProtocolTrajectory] = []
        self.fresh_test_protocols: List[ProtocolTrajectory] = []
        self.acquisition_contract: Dict[str, Any] = {}
        self.provenance: Dict[str, Any] = {}

    @staticmethod
    def _verify_artifact(
        source: Path,
        *,
        archive_sha256: str,
        index_sha256: str,
        final_sha256: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = Path(source)
        verified = []
        payloads: Dict[str, bytes] = {}
        if source.is_file():
            if sha256_file(source) != archive_sha256:
                raise RuntimeError(f"artifact archive SHA-256 mismatch: {source.name}")
            with zipfile.ZipFile(source) as archive:
                index_names = [name for name in archive.namelist() if name.endswith("artifact_index.json")]
                if len(index_names) != 1:
                    raise RuntimeError("artifact index is ambiguous")
                index_bytes = archive.read(index_names[0])
                if hashlib.sha256(index_bytes).hexdigest() != index_sha256:
                    raise RuntimeError("artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                prefix = index_names[0][: -len("artifact_index.json")]
                for row in index["artifacts"]:
                    relative = str(row["path"]).replace("\\", "/")
                    payload = archive.read(prefix + relative)
                    if len(payload) != int(row["size_bytes"]) or hashlib.sha256(payload).hexdigest() != str(row["sha256"]):
                        raise RuntimeError(f"artifact member mismatch: {relative}")
                    if relative in {"final_report.json", "boundary_template_pilot.json", "confirmation_plan.json"}:
                        payloads[relative] = payload
                    verified.append(relative)
            source_kind, archive_hash = "original_zip", archive_sha256
        else:
            matches = [
                path for path in source.rglob("artifact_index.json")
                if hashlib.sha256(path.read_bytes()).hexdigest() == index_sha256
            ]
            if len(matches) != 1:
                raise RuntimeError("extracted artifact index is absent or ambiguous")
            root, index_bytes = matches[0].parent, matches[0].read_bytes()
            index = json.loads(index_bytes)
            for row in index["artifacts"]:
                relative = str(row["path"])
                path = root / relative
                if not path.is_file() or path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != str(row["sha256"]):
                    raise RuntimeError(f"extracted artifact member mismatch: {relative}")
                if relative in {"final_report.json", "boundary_template_pilot.json", "confirmation_plan.json"}:
                    payloads[relative] = path.read_bytes()
                verified.append(relative)
            source_kind, archive_hash = "kaggle_extracted_directory", None
        final_bytes = payloads.get("final_report.json")
        if final_bytes is None or hashlib.sha256(final_bytes).hexdigest() != final_sha256:
            raise RuntimeError("artifact final report SHA-256 mismatch")
        contract = {
            "source_path": str(source),
            "source_kind": source_kind,
            "archive_sha256": archive_hash,
            "artifact_index_sha256": index_sha256,
            "final_report_sha256": final_sha256,
            "verified_member_count": len(verified),
            "all_indexed_members_verified": len(verified)
            == int(index.get("artifact_count", len(index["artifacts"]))),
        }
        return contract, {name: json.loads(value) for name, value in payloads.items()}

    def verify_prerequisites(self) -> Dict[str, Any]:
        contract_i, payload_i = self._verify_artifact(
            self.artifact_05ji_source,
            archive_sha256=EXPECTED_05JI_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JI_INDEX_SHA256,
            final_sha256=EXPECTED_05JI_FINAL_SHA256,
        )
        contract_l, payload_l = self._verify_artifact(
            self.artifact_05jl_source,
            archive_sha256=EXPECTED_05JL_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JL_INDEX_SHA256,
            final_sha256=EXPECTED_05JL_FINAL_SHA256,
        )
        report_i, report_l = payload_i["final_report.json"], payload_l["final_report.json"]
        blockers = []
        if report_i.get("diagnosis") != "NEAR_REGENERATIVE_CONFIRMATION_SUPPORT_ACQUIRED":
            blockers.append("05j-i did not acquire the registered support")
        if int(report_i.get("realized_stratum_counts", {}).get("near_regenerative", 0)) < 18:
            blockers.append("05j-i near-regenerative support is insufficient")
        if report_l.get("diagnosis") != "FIT_ONLY_SAFETY_GATE_DOES_NOT_RESCUE_EXTERNAL_DECODER":
            blockers.append("05j-l diagnosis is unexpected")
        if report_l.get("candidate_authorized") is not False:
            blockers.append("05j-l unexpectedly authorized a candidate")
        pilot = payload_i.get("boundary_template_pilot.json", {})
        if not pilot.get("valid") or pilot.get("canonical_weight_scaling_used") is not False:
            blockers.append("05j-i frozen template pilot is invalid")
        if not pilot.get("selected_templates"):
            blockers.append("05j-i contains no frozen selected templates")
        self.provenance = {
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05ji": contract_i,
            "artifact_05jl": contract_l,
            "frozen_template_count": len(pilot.get("selected_templates", [])),
            "source_templates": pilot.get("selected_templates", []),
            "05ji_is_now_development_support": True,
            "05jl_is_post_result_diagnostic_only": True,
        }
        write_json(self.output_dir / "prerequisite_verification.json", self.provenance)
        if blockers:
            raise RuntimeError(f"05j-m prerequisite blockers: {blockers}")
        return self.provenance

    def build_acquisition_plans(self) -> Tuple[List[ProtocolTrajectory], Dict[str, Any]]:
        if not self.provenance.get("valid"):
            raise RuntimeError("verify_prerequisites() must pass first")
        train, fresh, contract = build_regenerative_support_plans(
            self.provenance["source_templates"], self.acquisition_config
        )
        base_episodes = self._normalized_episodes(
            self.pd.read_parquet(self.base_dataset / "episodes.parquet")
        )
        occupied = {int(row["seed"]) for row in base_episodes}
        planned = {row.seed for row in (*train, *fresh)}
        if occupied & planned:
            raise RuntimeError("05j-m planned seeds overlap the base dataset")
        contract.update(
            {
                "teacher_commit": PINNED_TEACHER_COMMIT,
                "base_seed_overlap": sorted(occupied & planned),
                "artifact_05ji": self.provenance["artifact_05ji"],
                "artifact_05jl": self.provenance["artifact_05jl"],
            }
        )
        contract.pop("contract_sha256", None)
        contract["contract_sha256"] = canonical_json_sha256({"contract": contract})
        self.train_protocols, self.fresh_test_protocols = train, fresh
        self.acquisition_contract = contract
        self.topup_plan = contract["train"]
        self._bind_protocol_registry(train)
        write_json(self.output_dir / "acquisition_contract.json", contract)
        write_json(
            self.output_dir / "sealed_fresh_test_plan.json",
            {
                "schema_version": REGENERATIVE_TRAINING_SCHEMA_VERSION,
                "fresh_test": contract["fresh_test"],
                "outcomes_generated": False,
                "training_access_forbidden": True,
                "plan_frozen_before_train_acquisition": True,
            },
        )
        return train, contract

    def generate_training_shard(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        protocols = list(protocols)
        if protocols != self.train_protocols:
            raise RuntimeError("training protocols differ from the frozen plan")
        required_snapshots = {str(row.metadata["snapshot_id"]) for row in protocols}
        if required_snapshots != set(self.snapshot_bank):
            raise RuntimeError("05j-m snapshot bank does not match the train plan")
        zero_targets = {"train": 0}
        write_json(
            self.output_dir / "planning_budget_report.json",
            {
                "schema_version": REGENERATIVE_TRAINING_SCHEMA_VERSION,
                "role": "standalone near-regenerative training shard",
                "effective_positive_targets": zero_targets,
                "effective_hard_negative_targets": zero_targets,
                "minimum_positive_targets": zero_targets,
                "minimum_hard_negative_targets": zero_targets,
            },
        )
        self.targeted_preflight_report = {
            "valid": True,
            "policy": "fixed_train_batch_all_pairs_retained",
            "protocol_plan_sha256": self.acquisition_contract["train"]["protocol_plan_sha256"],
        }
        write_json(self.output_dir / "targeted_preflight_report.json", self.targeted_preflight_report)
        self.release_rows = []
        self._bind_protocol_registry(protocols)
        self._collect_release_rows = True
        try:
            manifest = DiagnosticDatasetSession.generate_dataset(self, protocols)
        finally:
            self._collect_release_rows = False
        self.pd.DataFrame(self._parquet_safe_rows(self.release_rows)).to_parquet(
            self.output_dir / "release_outcomes.parquet", index=False
        )
        table_report = self._write_targeted_tables(protocols)
        manifest.update(
            {
                "schema_version": REGENERATIVE_TRAINING_SCHEMA_VERSION,
                "dataset_role": "train_only_near_regenerative_support_shard",
                "compatible_base_schema_version": "1.1.2",
                "selection_policy": "fixed_templates_all_train_pairs_retained",
                "acquisition_contract": "acquisition_contract.json",
                "sealed_fresh_test_plan": "sealed_fresh_test_plan.json",
                "table_report": table_report,
            }
        )
        self.dataset_manifest = manifest
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        self._write_artifact_index()
        return manifest

    def validate_training_shard(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        protocols = list(protocols)
        self._bind_protocol_registry(protocols)
        structural = validate_hdf5_store(self.transition_path)
        transition_sha = _sha256_with_progress(self.transition_path, label="shard train near-regenerative")
        replay = self._exhaustive_sequential_replay()
        transition_index = self.pd.read_parquet(self.output_dir / "transition_index.parquet")
        by_key = {
            (str(row.trajectory_id), int(row.step_index)): int(row.transition_id)
            for row in transition_index.itertuples(index=False)
        }
        import h5py

        pair_rows = []
        with h5py.File(self.transition_path, "r") as handle:
            for row in self.acquisition_contract["train"]["pairs"]:
                low_id, high_id = row["trajectory_ids"]
                step = int(row["branch_step"])
                low_index, high_index = by_key[(low_id, step)], by_key[(high_id, step)]
                low_t = handle["states/voltage/t"][low_index, :]
                high_t = handle["states/voltage/t"][high_index, :]
                low_t1 = handle["states/voltage/t_plus_1"][low_index, :]
                high_t1 = handle["states/voltage/t_plus_1"][high_index, :]
                peak = float(max(np.max(low_t1), np.max(high_t1)))
                pair_rows.append(
                    {
                        **dict(row),
                        "low_transition_id": low_index,
                        "high_transition_id": high_index,
                        "same_boundary_voltage_max_error_mv": float(np.max(np.abs(low_t - high_t))),
                        "teacher_distance_rmse_mv": float(np.sqrt(np.mean((low_t1 - high_t1) ** 2))),
                        "target_peak_mv": peak,
                        "realized_stratum": (
                            "regenerative"
                            if peak >= self.acquisition_config.near_upper_mv
                            else "near_regenerative"
                            if peak >= self.acquisition_config.near_lower_mv
                            else "subthreshold"
                        ),
                    }
                )
        self.pd.DataFrame(pair_rows).to_parquet(
            self.output_dir / "training_pairs.parquet", index=False
        )
        near_count = sum(row["realized_stratum"] == "near_regenerative" for row in pair_rows)
        pair_contract_valid = bool(
            len(pair_rows) == self.acquisition_config.train_pair_count
            and all(row["same_boundary_voltage_max_error_mv"] <= 1e-5 for row in pair_rows)
            and all(row["teacher_distance_rmse_mv"] >= self.acquisition_config.minimum_teacher_distance_mv for row in pair_rows)
        )
        split_values = sorted(set(map(str, transition_index["split"])))
        fresh_seeds = set(
            range(
                self.acquisition_config.fresh_test_seed_start,
                self.acquisition_config.fresh_test_seed_start
                + self.acquisition_config.fresh_test_pair_count,
            )
        )
        generated_seeds = set(map(int, transition_index["seed"]))
        fresh_absent = not bool(fresh_seeds & generated_seeds)
        artifact_valid = bool(
            structural.get("valid")
            and replay.get("valid")
            and pair_contract_valid
            and split_values == ["train"]
            and fresh_absent
        )
        support_sufficient = near_count >= self.acquisition_config.minimum_train_near_pair_count
        report = {
            "schema_version": REGENERATIVE_TRAINING_SCHEMA_VERSION,
            "valid": artifact_valid,
            "artifact_valid": artifact_valid,
            "scientific_train_support_sufficient": support_sufficient,
            "diagnosis": (
                "TRAIN_SUPPORT_ACQUIRED_FRESH_TEST_PLAN_SEALED"
                if support_sufficient
                else "TRAIN_SUPPORT_REALIZED_NEAR_COUNT_INSUFFICIENT"
            ),
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "code_revision": git_commit(self.elm_repo),
            "transition_store_sha256": transition_sha,
            "structural": structural,
            "exhaustive_replay": replay,
            "pair_contract_valid": pair_contract_valid,
            "registered_train_pair_count": self.acquisition_config.train_pair_count,
            "retained_train_pair_count": len(pair_rows),
            "retained_train_episode_count": len(protocols),
            "minimum_train_near_pair_count": self.acquisition_config.minimum_train_near_pair_count,
            "realized_train_stratum_counts": {
                name: sum(row["realized_stratum"] == name for row in pair_rows)
                for name in ("regenerative", "near_regenerative", "subthreshold")
            },
            "generated_splits": split_values,
            "fresh_test": {
                "protocol_plan_sha256": self.acquisition_contract["fresh_test"]["protocol_plan_sha256"],
                "pair_count": self.acquisition_config.fresh_test_pair_count,
                "outcomes_generated": False,
                "seeds_absent_from_training_shard": fresh_absent,
                "must_not_be_loaded_during_training": True,
            },
            "candidate_authorized": False,
            "micro_rollout_authorized": False,
            "full_training_authorized": False,
            "methodology": {
                "05ji_reclassified_as_development": True,
                "train_and_fresh_test_plans_frozen_before_outcomes": True,
                "all_train_pairs_retained": True,
                "canonical_netcon_weights_preserved": True,
                "weight_multiplier_sweeps_used": False,
                "causal_realized_release_logged": True,
                "fresh_test_outcomes_generated": False,
                "candidate_training_performed": False,
                "rollout_performed": False,
            },
            "next_step": (
                "05j_n_refit_decoder_on_expanded_train_support_before_fresh_test"
                if support_sufficient
                else "05j_m_b_expand_outcome_blind_train_acquisition"
            ),
        }
        write_json(self.output_dir / "validation_report.json", report)
        write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index({"transition_dataset.h5": transition_sha})
        if not artifact_valid:
            raise RuntimeError("05j-m train shard failed integrity, replay, or isolation")
        return report


__all__ = [
    "EXPECTED_05JI_ARCHIVE_SHA256",
    "EXPECTED_05JI_INDEX_SHA256",
    "EXPECTED_05JI_FINAL_SHA256",
    "EXPECTED_05JL_ARCHIVE_SHA256",
    "EXPECTED_05JL_INDEX_SHA256",
    "EXPECTED_05JL_FINAL_SHA256",
    "REGENERATIVE_TRAINING_SCHEMA_VERSION",
    "RegenerativeTrainingSupportConfig",
    "RegenerativeTrainingSupportSession",
    "build_regenerative_support_plans",
]
