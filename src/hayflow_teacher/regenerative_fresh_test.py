"""Generate the sealed 05j-m fresh test after the registered 05j-n gate."""

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
from .regenerative_training_support import _plan_rows


EXPECTED_05JM_ARCHIVE_SHA256 = (
    "70010063448be6de8c7478037398f58fbd2d4cd8d81a14fade1d1561b47bcf20"
)
EXPECTED_05JM_INDEX_SHA256 = (
    "90a46ddab7d27cc00f70e177f30ba97e8147a14f8fca390b809ed8eb81c868a1"
)
EXPECTED_05JM_FINAL_SHA256 = (
    "27eb696bfee58efa118a6929d8630ccc8141f558ed42d9394d6e0e8a00fed795"
)
EXPECTED_05JN_ARCHIVE_SHA256 = (
    "1d7e5aab979be1c7bf5dcbc5016861a5186769b4ad98a18edd556366419d89f4"
)
EXPECTED_05JN_INDEX_SHA256 = (
    "c3c4199b0a6da73b8f0ec3b986e22a7f2a9999cc74947039ac72811583c8e610"
)
EXPECTED_05JN_FINAL_SHA256 = (
    "4575983a8a81b2ee6b0160bcde06738a00ec689bb6c8c3f8d4957333b563c5cd"
)
EXPECTED_FRESH_PROTOCOL_PLAN_SHA256 = (
    "31b42befaa2a3cf6b76e304a45daef133371f87f2c069a76ade030c4a68ec642"
)
REGENERATIVE_FRESH_TEST_SCHEMA_VERSION = "05j-o-regenerative-fresh-test-v1"


@dataclass(frozen=True)
class RegenerativeFreshTestConfig:
    pair_count: int = 32
    episode_count: int = 64
    transition_count: int = 768
    episode_duration_ms: int = 12
    conditioning_ms: int = 4
    seed_start: int = 1_100_001
    minimum_teacher_distance_mv: float = 0.01
    boundary_state_atol: float = 1e-5

    def validate(self) -> None:
        if (self.pair_count, self.episode_count, self.transition_count) != (
            32,
            64,
            768,
        ):
            raise ValueError("05j-o cardinalities are preregistered")
        if self.episode_count != 2 * self.pair_count:
            raise ValueError("fresh-test episodes do not form complete pairs")
        if self.transition_count != self.episode_count * self.episode_duration_ms:
            raise ValueError("fresh-test transition count is inconsistent")
        if min(
            self.conditioning_ms,
            self.seed_start,
            self.minimum_teacher_distance_mv,
            self.boundary_state_atol,
        ) <= 0:
            raise ValueError("fresh-test configuration values must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "RegenerativeFreshTestConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def _verify_artifact(
    source: Path,
    *,
    archive_sha256: str,
    index_sha256: str,
    final_sha256: str,
    capture: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Verify every indexed member and return only explicitly captured bytes."""

    source = Path(source).expanduser().resolve()
    captured: Dict[str, bytes] = {}
    if source.is_file():
        if sha256_file(source) != archive_sha256:
            raise RuntimeError(f"artifact archive SHA-256 mismatch: {source.name}")
        with zipfile.ZipFile(source) as archive:
            indices = [name for name in archive.namelist() if name.endswith("artifact_index.json")]
            if len(indices) != 1:
                raise RuntimeError("artifact index is absent or ambiguous")
            index_bytes = archive.read(indices[0])
            if hashlib.sha256(index_bytes).hexdigest() != index_sha256:
                raise RuntimeError("artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            prefix = indices[0][: -len("artifact_index.json")]
            for row in index["artifacts"]:
                relative = str(row["path"]).replace("\\", "/")
                payload = archive.read(prefix + relative)
                if (
                    len(payload) != int(row["size_bytes"])
                    or hashlib.sha256(payload).hexdigest() != str(row["sha256"])
                ):
                    raise RuntimeError(f"artifact member mismatch: {relative}")
                if relative in capture:
                    captured[relative] = payload
        source_kind, archive_hash = "original_zip", archive_sha256
    elif source.is_dir():
        indices = [
            path
            for path in source.rglob("artifact_index.json")
            if hashlib.sha256(path.read_bytes()).hexdigest() == index_sha256
        ]
        if len(indices) != 1:
            raise RuntimeError("extracted artifact index is absent or ambiguous")
        root = indices[0].parent
        index = json.loads(indices[0].read_bytes())
        for row in index["artifacts"]:
            relative = str(row["path"])
            path = root / relative
            if (
                not path.is_file()
                or path.stat().st_size != int(row["size_bytes"])
                or sha256_file(path) != str(row["sha256"])
            ):
                raise RuntimeError(f"artifact member mismatch: {relative}")
            if relative in capture:
                captured[relative] = path.read_bytes()
        source_kind, archive_hash = "kaggle_extracted_directory", None
    else:
        raise RuntimeError(f"artifact source does not exist: {source}")
    final_bytes = captured.get("final_report.json")
    if final_bytes is None or hashlib.sha256(final_bytes).hexdigest() != final_sha256:
        raise RuntimeError("artifact final report SHA-256 mismatch")
    missing = sorted(set(capture) - set(captured))
    if missing:
        raise RuntimeError(f"required artifact members are absent: {missing}")
    contract = {
        "source_path": str(source),
        "source_kind": source_kind,
        "archive_sha256": archive_hash,
        "artifact_index_sha256": index_sha256,
        "final_report_sha256": final_sha256,
        "verified_member_count": len(index["artifacts"]),
        "all_indexed_members_verified": len(index["artifacts"])
        == int(index.get("artifact_count", len(index["artifacts"]))),
    }
    return contract, captured


def protocol_from_frozen_row(row: Mapping[str, Any]) -> ProtocolTrajectory:
    actions = action_schedule_from_json(row.get("actions", {}))
    metadata = dict(row.get("metadata", {}))
    trajectory = ProtocolTrajectory(
        trajectory_id=str(row["trajectory_id"]),
        category=str(row["category"]),
        protocol=str(row["protocol"]),
        protocol_id=str(row.get("protocol_id", "")),
        protocol_variant=str(row.get("protocol_variant", "")),
        seed=int(row["seed"]),
        duration_ms=int(row["duration_ms"]),
        split=str(row["split"]),
        actions_by_step=actions,
        stimulus_onset_step=int(row.get("stimulus_onset_step", 0)),
        negative_control=str(metadata.get("branch_arm")) == "low",
        snapshot_source=str(row.get("snapshot_source", metadata.get("snapshot_id", ""))),
        metadata=metadata,
    )
    trajectory.validate()
    return trajectory


class RegenerativeFreshTestSession(BapValidationSupportTopupSession):
    """Open the sealed plan once and generate every registered teacher outcome."""

    def __init__(
        self,
        *args: Any,
        artifact_05jm_source: Path,
        artifact_05jn_source: Path,
        fresh_test_config: RegenerativeFreshTestConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        fresh_test_config.validate()
        self.artifact_05jm_source = Path(artifact_05jm_source).resolve()
        self.artifact_05jn_source = Path(artifact_05jn_source).resolve()
        self.fresh = fresh_test_config
        self.fresh_protocols: List[ProtocolTrajectory] = []
        self.fresh_pairs: List[Dict[str, Any]] = []
        self.authorization: Dict[str, Any] = {}

    def open_authorized_plan(self) -> Tuple[List[ProtocolTrajectory], Dict[str, Any]]:
        # The 05j-n decision is verified before the sealed 05j-m plan is parsed.
        contract_n, payload_n = _verify_artifact(
            self.artifact_05jn_source,
            archive_sha256=EXPECTED_05JN_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JN_INDEX_SHA256,
            final_sha256=EXPECTED_05JN_FINAL_SHA256,
            capture=("final_report.json",),
        )
        report_n = json.loads(payload_n["final_report.json"])
        blockers = []
        if not report_n.get("valid") or not report_n.get("fresh_test_generation_authorized"):
            blockers.append("05j-n did not authorize fresh-test generation")
        if report_n.get("diagnosis") != "REFIT_PASSES_DEVELOPMENT_FRESH_TEST_GENERATION_AUTHORIZED":
            blockers.append("05j-n diagnosis is incompatible with fresh-test opening")
        if int(report_n.get("decoder_refit", {}).get("passing_seed_count", -1)) < 2:
            blockers.append("05j-n robust seed gate did not pass")
        if blockers:
            raise RuntimeError(f"05j-o authorization blockers: {blockers}")

        contract_m, payload_m = _verify_artifact(
            self.artifact_05jm_source,
            archive_sha256=EXPECTED_05JM_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JM_INDEX_SHA256,
            final_sha256=EXPECTED_05JM_FINAL_SHA256,
            capture=(
                "final_report.json",
                "acquisition_contract.json",
                "sealed_fresh_test_plan.json",
            ),
        )
        report_m = json.loads(payload_m["final_report.json"])
        acquisition = json.loads(payload_m["acquisition_contract.json"])
        sealed = json.loads(payload_m["sealed_fresh_test_plan.json"])
        fresh = sealed["fresh_test"]
        plan_hash = canonical_json_sha256({"protocols": fresh["protocols"]})
        seeds = sorted({int(row["seed"]) for row in fresh["protocols"]})
        expected_seeds = list(
            range(self.fresh.seed_start, self.fresh.seed_start + self.fresh.pair_count)
        )
        blockers = []
        if report_m.get("diagnosis") != "TRAIN_SUPPORT_ACQUIRED_FRESH_TEST_PLAN_SEALED":
            blockers.append("05j-m is not the registered sealed-plan artifact")
        if report_n.get("artifact_05jm", {}).get("artifact_index_sha256") != EXPECTED_05JM_INDEX_SHA256:
            blockers.append("05j-n was trained against a different 05j-m artifact")
        if plan_hash != EXPECTED_FRESH_PROTOCOL_PLAN_SHA256 or plan_hash != fresh.get("protocol_plan_sha256"):
            blockers.append("fresh-test protocol plan SHA-256 mismatch")
        if int(fresh.get("pair_count", -1)) != self.fresh.pair_count:
            blockers.append("fresh-test pair count changed")
        if int(fresh.get("episode_count", -1)) != self.fresh.episode_count:
            blockers.append("fresh-test episode count changed")
        if fresh.get("outcomes_generated") is not False:
            blockers.append("05j-m unexpectedly contains fresh-test outcomes")
        if seeds != expected_seeds:
            blockers.append("fresh-test seed namespace changed")
        if acquisition.get("fresh_test", {}).get("protocol_plan_sha256") != plan_hash:
            blockers.append("acquisition contract and sealed plan disagree")
        if blockers:
            raise RuntimeError(f"05j-o sealed-plan blockers: {blockers}")
        protocols = [protocol_from_frozen_row(row) for row in fresh["protocols"]]
        if _plan_rows(protocols) != fresh["protocols"]:
            raise RuntimeError("fresh-test protocol roundtrip changed the frozen plan")
        self.fresh_protocols = protocols
        self.fresh_pairs = [dict(row) for row in fresh["pairs"]]
        self.topup_plan = fresh
        self._bind_protocol_registry(protocols)
        self.authorization = {
            "schema_version": REGENERATIVE_FRESH_TEST_SCHEMA_VERSION,
            "valid": True,
            "artifact_05jm": contract_m,
            "artifact_05jn": contract_n,
            "protocol_plan_sha256": plan_hash,
            "pair_count": len(self.fresh_pairs),
            "episode_count": len(protocols),
            "transition_count": sum(row.duration_ms for row in protocols),
            "seed_start": min(seeds),
            "seed_end": max(seeds),
            "05jn_verified_before_plan_parsing": True,
            "all_frozen_protocols_retained": True,
            "selection_after_outcomes_permitted": False,
            "retraining_after_outcomes_permitted": False,
            "code_revision": git_commit(self.elm_repo),
        }
        write_json(self.output_dir / "fresh_test_authorization.json", self.authorization)
        write_json(
            self.output_dir / "opened_fresh_test_plan.json",
            {
                "schema_version": REGENERATIVE_FRESH_TEST_SCHEMA_VERSION,
                "fresh_test": fresh,
                "opened_after_05jn_freeze": True,
            },
        )
        return protocols, self.authorization

    def generate_fresh_test_shard(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        protocols = list(protocols)
        if protocols != self.fresh_protocols:
            raise RuntimeError("fresh-test protocols differ from the sealed plan")
        required_snapshots = {str(row.metadata["snapshot_id"]) for row in protocols}
        if required_snapshots != set(self.snapshot_bank):
            raise RuntimeError("fresh-test snapshot bank does not match the plan")
        zero_targets = {"test": 0}
        write_json(
            self.output_dir / "planning_budget_report.json",
            {
                "schema_version": REGENERATIVE_FRESH_TEST_SCHEMA_VERSION,
                "role": "sealed fresh-test teacher shard",
                "effective_positive_targets": zero_targets,
                "effective_hard_negative_targets": zero_targets,
                "minimum_positive_targets": zero_targets,
                "minimum_hard_negative_targets": zero_targets,
            },
        )
        self.targeted_preflight_report = {
            "valid": True,
            "policy": "all_preregistered_fresh_test_pairs_retained",
            "protocol_plan_sha256": self.authorization["protocol_plan_sha256"],
        }
        write_json(
            self.output_dir / "targeted_preflight_report.json",
            self.targeted_preflight_report,
        )
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
                "schema_version": REGENERATIVE_FRESH_TEST_SCHEMA_VERSION,
                "dataset_role": "preregistered_regenerative_fresh_test",
                "compatible_base_schema_version": "1.1.2",
                "selection_policy": "all_32_frozen_pairs_retained",
                "protocol_plan_sha256": self.authorization["protocol_plan_sha256"],
                "table_report": table_report,
            }
        )
        self.dataset_manifest = manifest
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        self._write_artifact_index()
        return manifest

    def validate_fresh_test_shard(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        protocols = list(protocols)
        self._bind_protocol_registry(protocols)
        structural = validate_hdf5_store(self.transition_path)
        transition_sha = _sha256_with_progress(
            self.transition_path, label="fresh test rigenerativo"
        )
        replay = self._exhaustive_sequential_replay()
        transition_index = self.pd.read_parquet(
            self.output_dir / "transition_index.parquet"
        )
        by_key = {
            (str(row.trajectory_id), int(row.step_index)): int(row.transition_id)
            for row in transition_index.itertuples(index=False)
        }
        import h5py

        pair_rows = []
        with h5py.File(self.transition_path, "r") as handle:
            for row in self.fresh_pairs:
                low_id, high_id = row["trajectory_ids"]
                step = int(row["branch_step"])
                low_index, high_index = by_key[(low_id, step)], by_key[(high_id, step)]
                low_t = handle["states/voltage/t"][low_index, :]
                high_t = handle["states/voltage/t"][high_index, :]
                low_t1 = handle["states/voltage/t_plus_1"][low_index, :]
                high_t1 = handle["states/voltage/t_plus_1"][high_index, :]
                pair_rows.append(
                    {
                        **dict(row),
                        "low_transition_id": low_index,
                        "high_transition_id": high_index,
                        "same_boundary_voltage_max_error_mv": float(
                            np.max(np.abs(low_t - high_t))
                        ),
                        "teacher_distance_rmse_mv": float(
                            np.sqrt(np.mean((low_t1 - high_t1) ** 2))
                        ),
                        "target_peak_mv": float(max(np.max(low_t1), np.max(high_t1))),
                    }
                )
        self.pd.DataFrame(pair_rows).to_parquet(
            self.output_dir / "fresh_test_pairs.parquet", index=False
        )
        split_values = sorted(set(map(str, transition_index["split"])))
        seeds = sorted(set(map(int, transition_index["seed"])))
        expected_seeds = list(
            range(self.fresh.seed_start, self.fresh.seed_start + self.fresh.pair_count)
        )
        pair_contract_valid = bool(
            len(pair_rows) == self.fresh.pair_count
            and all(
                row["same_boundary_voltage_max_error_mv"]
                <= self.fresh.boundary_state_atol
                for row in pair_rows
            )
            and all(
                row["teacher_distance_rmse_mv"]
                >= self.fresh.minimum_teacher_distance_mv
                for row in pair_rows
            )
        )
        valid = bool(
            structural.get("valid")
            and replay.get("valid")
            and pair_contract_valid
            and split_values == ["test"]
            and seeds == expected_seeds
            and len(protocols) == self.fresh.episode_count
            and len(transition_index) == self.fresh.transition_count
        )
        report = {
            "schema_version": REGENERATIVE_FRESH_TEST_SCHEMA_VERSION,
            "valid": valid,
            "diagnosis": "FRESH_TEST_TEACHER_OUTCOMES_GENERATED_AWAITING_FROZEN_MODEL_EVALUATION",
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "code_revision": git_commit(self.elm_repo),
            "protocol_plan_sha256": self.authorization["protocol_plan_sha256"],
            "transition_store_sha256": transition_sha,
            "structural": structural,
            "exhaustive_replay": replay,
            "pair_contract_valid": pair_contract_valid,
            "pair_count": len(pair_rows),
            "episode_count": len(protocols),
            "transition_count": len(transition_index),
            "generated_splits": split_values,
            "generated_seed_start": min(seeds),
            "generated_seed_end": max(seeds),
            "all_frozen_pairs_retained": True,
            "selection_performed": False,
            "retraining_performed": False,
            "candidate_model_authorized": False,
            "micro_rollout_authorized": False,
            "full_training_authorized": False,
        }
        write_json(self.output_dir / "teacher_fresh_test_report.json", report)
        write_json(self.output_dir / "validation_report.json", report)
        self._write_artifact_index({"transition_dataset.h5": transition_sha})
        if not valid:
            raise RuntimeError("05j-o fresh-test teacher shard failed validation")
        return report


__all__ = [
    "EXPECTED_05JM_ARCHIVE_SHA256",
    "EXPECTED_05JM_INDEX_SHA256",
    "EXPECTED_05JM_FINAL_SHA256",
    "EXPECTED_05JN_ARCHIVE_SHA256",
    "EXPECTED_05JN_INDEX_SHA256",
    "EXPECTED_05JN_FINAL_SHA256",
    "EXPECTED_FRESH_PROTOCOL_PLAN_SHA256",
    "REGENERATIVE_FRESH_TEST_SCHEMA_VERSION",
    "RegenerativeFreshTestConfig",
    "RegenerativeFreshTestSession",
    "protocol_from_frozen_row",
]
