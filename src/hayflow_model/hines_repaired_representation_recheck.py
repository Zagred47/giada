"""Notebook-05j train/development recheck of the repaired representation."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hayflow_hines import HINES_SYNAPTIC_FEATURE_NAMES
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_representation_forensics import robust_bounded_features
from .hines_synaptic_domain_repair import HinesSynapticDomainRepair

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05IC_ARCHIVE_SHA256 = (
    "dd2fc744760196b8fc5fd52f980a18a6d533ff0816103592725d6ad5c4e2e6df"
)
EXPECTED_05IC_INDEX_SHA256 = (
    "1554a04b9c6361f68169a616e785c395deee724ac38ba911e7cb4825e6568ad3"
)
EXPECTED_05IC_FINAL_SHA256 = (
    "5faca3a30ca132c4aa54e53997cea68f3a0c4b1e267e01924588250b48bd795e"
)


@dataclass(frozen=True)
class HinesRepairedRepresentationRecheckConfig:
    required_train_pair_count: int = 12
    required_development_pair_count: int = 1
    minimum_joint_passing_seed_count: int = 2
    expected_seed_count: int = 3
    expected_input_family_count: int = 3
    forbid_heldout_input_extraction: bool = True
    forbid_heldout_target_materialization: bool = True

    def validate(self) -> None:
        positive = (
            self.required_train_pair_count,
            self.required_development_pair_count,
            self.minimum_joint_passing_seed_count,
            self.expected_seed_count,
            self.expected_input_family_count,
        )
        if min(positive) <= 0:
            raise ValueError("05j recheck counts must be positive")
        if self.minimum_joint_passing_seed_count > self.expected_seed_count:
            raise ValueError("minimum passing seeds cannot exceed expected seeds")
        if not self.forbid_heldout_input_extraction:
            raise ValueError("05j must not extract held-out inputs")
        if not self.forbid_heldout_target_materialization:
            raise ValueError("05j must keep held-out targets sealed")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRepairedRepresentationRecheckConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def summarize_robust_family_gate(
    runs: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    *,
    expected_seed_count: int,
    minimum_joint_passing_seed_count: int,
) -> Sequence[Dict[str, Any]]:
    """Summarize the preregistered 2-of-3 joint train/development gate."""

    rows = []
    for family in families:
        selected = [row for row in runs if row["family"] == family]
        if not selected:
            raise ValueError(f"no 05j runs were produced for family {family}")
        joint = sum(
            bool(row["train_passed"] and row["development_passed"])
            for row in selected
        )
        rows.append({
            "family": family,
            "seed_count": len(selected),
            "joint_passing_seed_count": int(joint),
            "robust_family_passed": bool(
                len(selected) == int(expected_seed_count)
                and joint >= int(minimum_joint_passing_seed_count)
            ),
            "median_train_rmse_mv": float(np.median([
                row["train"]["aggregate_voltage_rmse_mv"] for row in selected
            ])),
            "median_development_rmse_mv": float(np.median([
                row["development"]["aggregate_voltage_rmse_mv"]
                for row in selected
            ])),
        })
    return rows


class HinesRepairedRepresentationRecheck(HinesSynapticDomainRepair):
    """05j bounded heads over repaired train/development representations only."""

    def __init__(
        self,
        *args: Any,
        recheck_config: HinesRepairedRepresentationRecheckConfig,
        artifact_05ic_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        recheck_config.validate()
        self.recheck = recheck_config
        self.artifact_05ic_source = Path(artifact_05ic_source).resolve()
        self.artifact_05ic_contract: Dict[str, Any] = {}
        self.recheck_roles: Dict[str, Any] = {}
        self.recheck_normalizers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _one_suffix(names: Sequence[str], suffix: str) -> str:
        matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one 05i-c {suffix}, found {matches}")
        return matches[0]

    def _read_verified_05ic(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05ic_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05IC_ARCHIVE_SHA256:
                raise RuntimeError("05i-c archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[: -len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05IC_INDEX_SHA256:
                    raise RuntimeError("05i-c artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                verified = {}
                for row in index["artifacts"]:
                    member = root + str(row["path"]).replace("\\", "/")
                    if member not in names:
                        raise RuntimeError(f"missing indexed 05i-c member: {row['path']}")
                    payload = archive.read(member)
                    observed = hashlib.sha256(payload).hexdigest()
                    if observed != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05i-c indexed member mismatch: {row['path']}")
                    verified[str(row["path"])] = observed
                final_bytes = archive.read(root + "final_report.json")
            kind = "original_zip"
        elif source.is_dir():
            indices = list(source.rglob("artifact_index.json"))
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05i-c artifact_index.json")
            index_bytes = indices[0].read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05IC_INDEX_SHA256:
                raise RuntimeError("extracted 05i-c artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            root = indices[0].parent
            verified = {}
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file():
                    raise RuntimeError(f"missing extracted 05i-c member: {row['path']}")
                observed = sha256_file(path)
                if observed != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05i-c member mismatch: {row['path']}")
                verified[str(row["path"])] = observed
            final_bytes = (root / "final_report.json").read_bytes()
            archive_hash = None
            kind = "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05i-c source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05IC_FINAL_SHA256:
            raise RuntimeError("05i-c final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05IC_INDEX_SHA256,
            "verified_member_count": len(verified),
            "all_indexed_members_verified": len(verified) == int(index["artifact_count"]),
            "final_report_sha256": EXPECTED_05IC_FINAL_SHA256,
        }

    def prepare_repaired_representation_recheck(self) -> Dict[str, Any]:
        base = self.prepare_synaptic_domain_repair()
        report, contract = self._read_verified_05ic()
        blockers = []
        if report.get("diagnosis") != "SYNAPTIC_DOMAIN_INPUT_CONTRACT_REPAIRED":
            blockers.append(f"unexpected 05i-c diagnosis: {report.get('diagnosis')}")
        if report.get("input_contract_passed") is not True:
            blockers.append("05i-c input contract did not pass")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05i-c dataset fingerprint mismatch")
        if not report.get("bounded_recency_roundtrip", {}).get("representation_contract_passed"):
            blockers.append("05i-c bounded-recency contract did not pass")
        if not report.get("synaptic_domain_floor_audit", {}).get("domain_floor_contract_passed"):
            blockers.append("05i-c domain-floor contract did not pass")
        if not report.get("repaired_frozen_h2_audit", {}).get("repaired_h2_input_contract_passed"):
            blockers.append("05i-c frozen-H2 contract did not pass")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05i-c artifact verification is incomplete")
        heldout = report.get("heldout_contract", {})
        if heldout.get("boundary_targets_materialized") or heldout.get("event_targets_materialized"):
            blockers.append("05i-c held-out future-target contract was not sealed")
        if tuple(self.forensics.input_families) != ("h2", "causal", "h2_causal"):
            blockers.append("05j input families differ from the registered 05h controls")
        if len(self.forensics.seeds) != self.recheck.expected_seed_count:
            blockers.append("05j seed count differs from the registered contract")
        if len(self.forensics.input_families) != self.recheck.expected_input_family_count:
            blockers.append("05j input-family count differs from the registered contract")
        if blockers:
            raise RuntimeError(f"05j provenance or methodology blockers: {blockers}")
        self.artifact_05ic_contract = contract
        payload = {
            "schema_version": "05j-repaired-representation-config-v1",
            "recheck": asdict(self.recheck),
            "artifact_05ic": contract,
            "input_families": list(self.forensics.input_families),
            "seeds": list(self.forensics.seeds),
            "pair_thresholds": {
                "rmse_mv": self.forensics.pair_rmse_mv,
                "maximum_error_mv": self.forensics.pair_max_error_mv,
                "retention_minimum": self.forensics.pair_retention_minimum,
                "retention_maximum": self.forensics.pair_retention_maximum,
            },
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "repaired_representation_recheck_config.json", payload)
        return {**base, **payload}

    def apply_verified_synaptic_domain_normalizer(self) -> Dict[str, Any]:
        """Recreate the 05i-c train-only normalizer without held-out reads."""

        if self.normalizer is None:
            raise RuntimeError("prepare_repaired_representation_recheck() must run first")
        if not self.artifact_05ic_contract:
            raise RuntimeError("the exact 05i-c artifact must be verified first")
        original = np.asarray(self.normalizer.state_scale, dtype=np.float64).copy()
        repaired, rows = self._repair_state_scales(original)
        digest = hashlib.sha256()
        digest.update(json.dumps(asdict(self.repair), sort_keys=True).encode())
        for values in (
            self.normalizer.transform_codes,
            self.normalizer.state_center,
            repaired,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
        fingerprint = digest.hexdigest()
        expected = (
            self._read_verified_05ic()[0]
            .get("coordinate_scale_repair", {})
            .get("repaired_normalizer_fingerprint")
        )
        if fingerprint != expected:
            raise RuntimeError(
                "05j recreated normalizer disagrees with the verified 05i-c artifact"
            )
        self.original_state_scale = original
        self.repaired_state_scale = repaired
        self.coordinate_rows = rows
        self.normalizer.state_scale = repaired.copy()
        report = {
            "schema_version": "05j-verified-normalizer-application-v1",
            "valid": True,
            "repaired_normalizer_fingerprint": fingerprint,
            "verified_05ic_fingerprint": expected,
            "state_width": int(len(repaired)),
            "lifted_coordinate_count": int(np.sum(repaired > original)),
            "fit_roles": ["train"],
            "development_values_used_to_fit": False,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
            "teacher_snapshot_modified": False,
        }
        _write_json(self.output_dir / "verified_synaptic_domain_normalizer.json", report)
        return report

    def _extract_recheck_role(
        self, model: Any, indices: Sequence[int]
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device
        raw = self._batch(
            indices, include_targets=True, include_event_targets=False
        )
        batch = self._torch_batch(raw, device)
        zero = dict(batch)
        for name in (
            "synaptic_features", "synaptic_conductance_us",
            "synaptic_source_na", "somatic_current_na",
        ):
            zero[name] = torch.zeros_like(batch[name])
        with torch.no_grad():
            output = model(
                batch, ablation="H2", decode_teacher=False,
                boundary_mode="no_event_jump",
            )
            zero_output = model(
                zero, ablation="H2", decode_teacher=False,
                boundary_mode="no_event_jump",
            )
        causal = torch.cat([
            batch["synaptic_features"],
            batch["synaptic_conductance_us"].unsqueeze(-1),
            batch["synaptic_source_na"].unsqueeze(-1),
            batch["somatic_current_na"].unsqueeze(-1),
        ], dim=-1)
        return {
            "indices": np.asarray(indices, dtype=np.int64),
            "base": output["voltage"].detach().cpu().double().numpy(),
            "h2_raw": output["boundary_features"].detach().cpu().double().numpy(),
            "h2_zero_causal_raw": zero_output["boundary_features"].detach().cpu().double().numpy(),
            "zero_causal_base": zero_output["voltage"].detach().cpu().double().numpy(),
            "causal_raw": causal.detach().cpu().double().numpy(),
            "voltage_t": np.asarray(raw["voltage_t"], dtype=np.float64),
            "teacher_state_normalized": np.asarray(raw["teacher_state_t"], dtype=np.float64),
            "target": np.asarray(raw["voltage_target"], dtype=np.float64),
        }

    def prepare_train_development_features(self) -> Dict[str, Any]:
        if self.repaired_state_scale is None:
            raise RuntimeError("run_coordinate_scale_repair() must run first")
        train_indices = np.asarray(
            self._pair_indices(self.audit_plan["selected_pairs"]), dtype=np.int64
        )
        development_indices = np.asarray(
            self.audit_plan["development_pair"], dtype=np.int64
        )
        train_episodes = {self._episode_identity(int(i))[1] for i in train_indices}
        development_episodes = {
            self._episode_identity(int(i))[1] for i in development_indices
        }
        overlap = sorted(train_episodes & development_episodes)
        blockers = []
        if len(train_indices) != 2 * self.recheck.required_train_pair_count:
            blockers.append("unexpected 05j train-pair count")
        if len(development_indices) != 2 * self.recheck.required_development_pair_count:
            blockers.append("unexpected 05j development-pair count")
        if overlap:
            blockers.append(f"train/development episode overlap: {overlap}")
        if blockers:
            raise RuntimeError(f"05j support blockers: {blockers}")
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = self._load_h2_checkpoint(device)
        model.eval()
        progress = Progress("05j repaired features", 2)
        self.recheck_roles = {
            "train": self._extract_recheck_role(model, train_indices),
        }
        progress.update(1, "train")
        self.recheck_roles["development"] = self._extract_recheck_role(
            model, development_indices
        )
        progress.update(2, "development")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.roles = self.recheck_roles

        surfaces = ("h2_raw", "h2_zero_causal_raw", "causal_raw")
        role_report: Dict[str, Any] = {"train": {}, "development": {}}
        feature_rows = []
        for surface in surfaces:
            train = self.roles["train"][surface]
            mean = train.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(
                train.std(axis=(0, 1), keepdims=True),
                self.forensics.feature_epsilon,
            )
            self.recheck_normalizers[surface] = (mean, scale)
            for role, values in self.roles.items():
                standardized, bounded = robust_bounded_features(
                    values[surface], mean, scale,
                    self.forensics.bounded_feature_scale,
                )
                values[f"{surface}_z"] = standardized
                values[f"{surface}_bounded"] = bounded
                role_report[role][surface] = {
                    "raw": self._surface_report(values[surface]),
                    "standardized_unclipped": self._surface_report(standardized),
                    "bounded_tanh": self._surface_report(bounded),
                    "clipping_fraction_at_registered_8": float(
                        np.mean(np.abs(standardized) > self.forensics.standardized_clip)
                    ),
                }
                feature_names = (
                    list(HINES_SYNAPTIC_FEATURE_NAMES)
                    + ["synaptic_conductance_us", "synaptic_source_na", "somatic_current_na"]
                    if surface == "causal_raw"
                    else [f"h2_hidden_{i}" for i in range(standardized.shape[-1])]
                )
                for feature, name in enumerate(feature_names):
                    feature_rows.append({
                        "role": role,
                        "surface": surface,
                        "feature_id": int(feature),
                        "feature_name": name,
                        "maximum_absolute_raw": float(
                            np.max(np.abs(values[surface][..., feature]))
                        ),
                        "maximum_absolute_standardized": float(
                            np.max(np.abs(standardized[..., feature]))
                        ),
                        "clipping_fraction_at_registered_8": float(
                            np.mean(np.abs(standardized[..., feature]) > self.forensics.standardized_clip)
                        ),
                    })
        np.savez_compressed(
            self.output_dir / "recheck_feature_normalizers.npz",
            **{
                f"{surface}_{kind}": value
                for surface, (mean, scale) in self.recheck_normalizers.items()
                for kind, value in (("mean", mean), ("scale", scale))
            },
        )
        write_parquet(self.output_dir / "recheck_feature_scale_by_feature.parquet", feature_rows)
        report = {
            "schema_version": "05j-train-development-features-v1",
            "valid": True,
            "train_pair_count": int(len(train_indices) // 2),
            "development_pair_count": int(len(development_indices) // 2),
            "train_logical_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
            "development_logical_indices_sha256": hashlib.sha256(development_indices.tobytes()).hexdigest(),
            "train_development_episode_overlap": overlap,
            "roles": role_report,
            "normalization_fit_roles": ["train"],
            "base_h2_frozen": True,
            "teacher_encoder_updated": False,
            "heldout_inputs_extracted": False,
            "heldout_boundary_targets_materialized": False,
            "heldout_event_targets_materialized": False,
        }
        _write_json(self.output_dir / "train_development_feature_recheck.json", report)
        return report

    def run_repaired_bounded_controls(self) -> Dict[str, Any]:
        report = super().run_bounded_representation_controls()
        report.update({
            "schema_version": "05j-repaired-bounded-controls-v1",
            "heldout_inputs_extracted": False,
            "heldout_frozen_h2_feature_extraction_performed": False,
            "heldout_candidate_head_inference_performed": False,
            "candidate_head_training_roles": ["train"],
            "model_selection_roles": ["development"],
            "candidate_head_evaluation_roles": ["train", "development"],
        })
        _write_json(self.output_dir / "bounded_representation_controls.json", report)
        return report

    def finalize_repaired_representation_recheck(
        self,
        feature_report: Mapping[str, Any],
        projection_report: Mapping[str, Any],
        controls_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        family_rows = summarize_robust_family_gate(
            controls_report["runs"],
            self.forensics.input_families,
            expected_seed_count=self.recheck.expected_seed_count,
            minimum_joint_passing_seed_count=(
                self.recheck.minimum_joint_passing_seed_count
            ),
        )
        robust = [row for row in family_rows if row["robust_family_passed"]]
        target_bound_valid = all(
            row["fraction_outside_residual_bound"] == 0.0
            for row in controls_report["target_residual_contract"].values()
        )
        passed = bool(feature_report["valid"] and target_bound_valid and robust)
        report = {
            "schema_version": "05j-final-report-v1",
            "valid": True,
            "decision": "TRAIN_DEVELOPMENT_REPRESENTATION_RECHECK_ONLY",
            "diagnosis": (
                "REPAIRED_REPRESENTATION_ROBUST_CANDIDATE_FOUND"
                if passed else "REPAIRED_REPRESENTATION_CONTROLS_FAIL_ROBUST_GATE"
            ),
            "representation_recheck_passed": passed,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05ic": self.artifact_05ic_contract,
            "train_development_features": dict(feature_report),
            "projection_forensics": dict(projection_report),
            "bounded_representation_controls": dict(controls_report),
            "family_gate": {
                "minimum_joint_passing_seed_count": self.recheck.minimum_joint_passing_seed_count,
                "families": family_rows,
                "passing_families": [row["family"] for row in robust],
                "target_residual_bound_valid": target_bound_valid,
            },
            "heldout_contract": {
                "inputs_extracted": False,
                "boundary_targets_materialized": False,
                "event_targets_materialized": False,
                "candidate_head_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "exact_registered_12_pair_train_support_reused": True,
                "development_pair_excluded_from_training": True,
                "feature_normalization_fit_split": "train",
                "base_h2_frozen": True,
                "teacher_encoder_updated": False,
                "candidate_head_training_roles": ["train"],
                "candidate_head_selection_roles": ["development"],
                "heldout_inputs_used": False,
                "heldout_targets_used": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": (
                    "05k_repaired_representation_micro_rollout"
                    if passed else "05j_b_repaired_representation_revision"
                ),
                "full_training_authorized": False,
            },
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append({
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05j-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
