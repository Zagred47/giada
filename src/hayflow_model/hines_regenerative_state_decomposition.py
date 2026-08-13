"""Notebook-05j-g regenerative state-target decomposition."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_region_mechanism_experts import HinesRegionMechanismExpertRevision
from .hines_repaired_representation_revision import (
    dual_ridge_path_predict,
    pair_gate_selection_score,
)
from .hines_spatial_support_revision import deterministic_pair_folds


EXPECTED_05JF_ARCHIVE_SHA256 = (
    "c16c87a113bb4e246d7a32a5491b8c6e66368a269b3d989a86d335a0534eb298"
)
EXPECTED_05JF_INDEX_SHA256 = (
    "7cd5716746e2e316f65a4077dc0eb3f33b625a6b718fdb6a89e1d52838b2f25e"
)
EXPECTED_05JF_FINAL_SHA256 = (
    "a53c90f005540efcfd10c63ddee8bdce8af97821bab30ef77d357e78455b20aa"
)

STATE_GROUPS = (
    "calcium_channel",
    "calcium_homeostasis",
    "sodium_channel",
    "potassium_channel",
    "h_current",
    "nmda_synaptic",
    "other_synaptic",
)
STATE_STATISTICS = ("signed_mean", "rms", "maximum_absolute")


@dataclass(frozen=True)
class HinesRegenerativeStateDecompositionConfig:
    state_groups: Tuple[str, ...] = STATE_GROUPS
    state_statistics: Tuple[str, ...] = STATE_STATISTICS
    ridge_lambdas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    cross_validation_folds: int = 6
    branch_fit_weight: float = 1.0
    selection_max_error_weight: float = 0.05
    selection_branch_log_weight: float = 2.0
    feature_epsilon: float = 1e-6
    spatial_shift_segments: int = 1
    material_rmse_improvement_fraction: float = 0.20
    material_max_error_improvement_fraction: float = 0.10
    oracle_specificity_improvement_fraction: float = 0.15

    def validate(self) -> None:
        if tuple(self.state_groups) != STATE_GROUPS:
            raise ValueError("05j-g state groups are preregistered")
        if tuple(self.state_statistics) != STATE_STATISTICS:
            raise ValueError("05j-g state statistics are preregistered")
        if self.cross_validation_folds < 2:
            raise ValueError("05j-g requires at least two grouped folds")
        if self.spatial_shift_segments == 0:
            raise ValueError("05j-g spatial null must be non-identity")
        positive = (
            *self.ridge_lambdas,
            self.branch_fit_weight,
            self.selection_max_error_weight,
            self.selection_branch_log_weight,
            self.feature_epsilon,
            self.material_rmse_improvement_fraction,
            self.material_max_error_improvement_fraction,
            self.oracle_specificity_improvement_fraction,
        )
        if min(positive) <= 0:
            raise ValueError("05j-g configuration values must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRegenerativeStateDecompositionConfig":
        payload = dict(values)
        for name in ("state_groups", "state_statistics", "ridge_lambdas"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def regenerative_state_group(record: Mapping[str, Any]) -> str | None:
    """Map a non-voltage state coordinate to one exclusive semantic group."""

    category = str(record.get("category", "")).lower()
    mechanism = str(record.get("mechanism", "")).lower()
    variable = str(record.get("variable", "")).lower()
    joined = f"{mechanism} {variable}"
    if category == "voltage":
        return None
    if category == "calcium_ions":
        return "calcium_homeostasis"
    if category == "synapse_states":
        # ProbAMPANMDA2 owns both AMPA and NMDA traces.  The point-process
        # class name alone is therefore insufficient; only variables whose
        # own name is NMDA-specific enter the NMDA group.
        return "nmda_synaptic" if "nmda" in variable else "other_synaptic"
    if category == "mechanism_states":
        if any(token in joined for token in ("ca_hva", "ca_lva", "calcium")):
            return "calcium_channel"
        if mechanism == "ih" or "hcn" in joined:
            return "h_current"
        if mechanism.startswith("na") or any(
            token in joined for token in ("nata", "nats", "nap_")
        ):
            return "sodium_channel"
        if mechanism.startswith("k") or mechanism.startswith("sk") or any(
            token in joined for token in ("kv", "im", "k_pst", "k_tst")
        ):
            return "potassium_channel"
        return None
    return None


def semantic_group_ids(
    records: Sequence[Mapping[str, Any]],
    groups: Sequence[str] = STATE_GROUPS,
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(groups)}
    result = np.full(len(records), -1, dtype=np.int64)
    for index, record in enumerate(records):
        group = regenerative_state_group(record)
        if group is not None:
            result[index] = lookup[group]
    return result


def aggregate_state_groups(
    values: np.ndarray,
    segment_ids: np.ndarray,
    group_ids: np.ndarray,
    *,
    segment_count: int,
    group_count: int,
) -> np.ndarray:
    """Aggregate standardized coordinates without mixing segments or groups."""

    array = np.asarray(values, dtype=np.float64)
    segments = np.asarray(segment_ids, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != len(segments) or len(groups) != len(segments):
        raise ValueError("state aggregation arrays disagree")
    result = np.zeros(
        (len(array), int(segment_count), int(group_count), len(STATE_STATISTICS)),
        dtype=np.float64,
    )
    for group in range(int(group_count)):
        group_coordinates = np.flatnonzero(groups == group)
        for segment in np.unique(segments[group_coordinates]):
            chosen = group_coordinates[segments[group_coordinates] == segment]
            if not len(chosen):
                continue
            selected = array[:, chosen]
            result[:, int(segment), group, 0] = np.mean(selected, axis=1)
            result[:, int(segment), group, 1] = np.sqrt(np.mean(selected * selected, axis=1))
            result[:, int(segment), group, 2] = np.max(np.abs(selected), axis=1)
    return result.reshape(len(array), int(segment_count), -1).astype(np.float32)


class HinesRegenerativeStateTargetDecomposition(HinesRegionMechanismExpertRevision):
    """Diagnostic causal/oracle probes around the frozen direct-tree ensemble."""

    def __init__(
        self,
        *args: Any,
        decomposition_config: HinesRegenerativeStateDecompositionConfig,
        artifact_05jf_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        decomposition_config.validate()
        self.decomposition = decomposition_config
        self.artifact_05jf_source = Path(artifact_05jf_source).resolve()
        self.artifact_05jf_contract: Dict[str, Any] = {}
        self.state_surfaces: Dict[str, Dict[str, np.ndarray]] = {}

    def _read_verified_05jf(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05jf_source
        if source.is_file():
            if sha256_file(source) != EXPECTED_05JF_ARCHIVE_SHA256:
                raise RuntimeError("05j-f archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[:-len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                index = json.loads(index_bytes)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JF_INDEX_SHA256:
                    raise RuntimeError("05j-f artifact index SHA-256 mismatch")
                for row in index["artifacts"]:
                    payload = archive.read(root + str(row["path"]).replace("\\", "/"))
                    if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05j-f indexed member mismatch: {row['path']}")
                final_bytes = archive.read(root + "final_report.json")
            kind, archive_hash = "original_zip", EXPECTED_05JF_ARCHIVE_SHA256
        elif source.is_dir():
            indices = [
                path for path in source.rglob("artifact_index.json")
                if (path.parent / "region_mechanism_expert_config.json").is_file()
            ]
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05j-f artifact index")
            index_bytes = indices[0].read_bytes()
            index = json.loads(index_bytes)
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JF_INDEX_SHA256:
                raise RuntimeError("extracted 05j-f artifact index SHA-256 mismatch")
            root = indices[0].parent
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05j-f member mismatch: {row['path']}")
            final_bytes = (root / "final_report.json").read_bytes()
            kind, archive_hash = "kaggle_extracted_directory", None
        else:
            raise RuntimeError(f"05j-f source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JF_FINAL_SHA256:
            raise RuntimeError("05j-f final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05JF_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05JF_FINAL_SHA256,
            "verified_member_count": len(index["artifacts"]),
            "all_indexed_members_verified": (
                len(index["artifacts"]) == int(index["artifact_count"])
            ),
        }

    def prepare_regenerative_state_target_decomposition(self) -> Dict[str, Any]:
        base = self.prepare_region_mechanism_expert_revision()
        report, contract = self._read_verified_05jf()
        blockers = []
        if report.get("diagnosis") != "REGION_MECHANISM_EXPERTS_DO_NOT_RESCUE_LOCALIZED_ERROR":
            blockers.append(f"unexpected 05j-f diagnosis: {report.get('diagnosis')}")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05j-f dataset fingerprint mismatch")
        if report.get("expert_canary_passed") or report.get("micro_rollout_authorized"):
            blockers.append("05j-f unexpectedly passed or authorized rollout")
        if report.get("methodology", {}).get("development_used_for_checkpoint_selection"):
            blockers.append("05j-f used development for checkpoint selection")
        if report.get("heldout_contract", {}).get("inputs_extracted"):
            blockers.append("05j-f held-out inputs were not sealed")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05j-f artifact verification is incomplete")
        if blockers:
            raise RuntimeError(f"05j-g provenance blockers: {blockers}")
        self.artifact_05jf_contract = contract
        payload = {
            "schema_version": "05j-g-state-target-decomposition-config-v1",
            "state_target_decomposition": asdict(self.decomposition),
            "artifact_05jf": contract,
            "base_prediction": "frozen_direct_tree_three_seed_ensemble",
            "causal_surface": "boundary_t_state_only",
            "oracle_surface": "non_voltage_teacher_state_delta_t_to_t_plus_1",
            "oracle_is_candidate_input": False,
            "future_voltage_coordinate_excluded": True,
            "selection_roles": ["fit_grouped_pair_cross_validation"],
            "development_used_for_model_selection": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "state_target_decomposition_config.json", payload)
        return {**base, **payload}

    def prepare_state_target_surfaces(self) -> Dict[str, Any]:
        if not self.frozen_predictions:
            raise RuntimeError("reconstruct_frozen_checkpoints() must run first")
        if self.normalizer is None:
            raise RuntimeError("verified state normalizer is unavailable")
        records = list(self._normalization_records())
        group_ids = semantic_group_ids(records, self.decomposition.state_groups)
        segment_ids = np.asarray(self.layout.core_segment_ids, dtype=np.int64)
        group_counts = {
            name: int(np.sum(group_ids == index))
            for index, name in enumerate(self.decomposition.state_groups)
        }
        blockers = []
        if len(records) != self.layout.state_width:
            blockers.append("semantic record count differs from state width")
        if int(np.sum(group_ids < 0)) != self.layout.segment_count:
            blockers.append("future voltage was not excluded exactly once per segment")
        if any(count == 0 for count in group_counts.values()):
            blockers.append("one or more preregistered state groups are empty")
        progress = Progress("05j-g state surfaces", len(self.topology_roles))
        for position, (role, state) in enumerate(self.topology_roles.items(), start=1):
            indices = np.asarray(state["indices"], dtype=np.int64)
            raw_t = self.store.read_state(indices, "t")
            raw_t1 = self.store.read_state(indices, "t_plus_1")
            semantic_t = self._state_input_view(raw_t, indices, "t")
            semantic_t1 = self._state_input_view(raw_t1, indices, "t_plus_1")
            current = self.normalizer.normalize_state(semantic_t)
            delta, _ = self.normalizer.delta_and_activity(semantic_t, semantic_t1)
            current_surface = aggregate_state_groups(
                current, segment_ids, group_ids,
                segment_count=self.layout.segment_count,
                group_count=len(self.decomposition.state_groups),
            )
            delta_surface = aggregate_state_groups(
                delta, segment_ids, group_ids,
                segment_count=self.layout.segment_count,
                group_count=len(self.decomposition.state_groups),
            )
            self.state_surfaces[role] = {
                "current": current_surface,
                "delta": delta_surface,
            }
            progress.update(position, f"{role} samples={len(indices)}")
        report = {
            "schema_version": "05j-g-state-target-surfaces-v1",
            "valid": not blockers,
            "blockers": blockers,
            "state_width": int(self.layout.state_width),
            "voltage_coordinate_count_excluded": int(np.sum(group_ids < 0)),
            "future_voltage_coordinate_excluded": True,
            "group_coordinate_counts": group_counts,
            "statistics": list(self.decomposition.state_statistics),
            "feature_width": len(self.decomposition.state_groups) * len(STATE_STATISTICS),
            "roles": {
                role: {
                    "sample_count": int(len(values["current"])),
                    "current_finite": bool(np.all(np.isfinite(values["current"]))),
                    "delta_finite": bool(np.all(np.isfinite(values["delta"]))),
                }
                for role, values in self.state_surfaces.items()
            },
            "causal_current_uses_future_state": False,
            "oracle_delta_uses_future_state": True,
            "oracle_delta_candidate_authorization": False,
            "development_used_to_define_groups": False,
            "heldout_inputs_extracted": False,
        }
        report["valid"] = bool(
            report["valid"]
            and all(
                row["current_finite"] and row["delta_finite"]
                for row in report["roles"].values()
            )
        )
        _write_json(self.output_dir / "state_target_surfaces.json", report)
        if not report["valid"]:
            raise RuntimeError(f"05j-g state surface blockers: {blockers}")
        return report

    @staticmethod
    def _improvement(baseline: float, candidate: float) -> float:
        return float((baseline - candidate) / max(abs(baseline), 1e-12))

    def _fit_probe(
        self,
        family: str,
        designs: Mapping[str, np.ndarray],
        baseline_residuals: Mapping[str, np.ndarray],
    ) -> Dict[str, Any]:
        fit_x = np.asarray(designs["fit"], dtype=np.float64)
        fit_state = self.topology_roles["fit"]
        fit_target = (
            np.asarray(fit_state["target"])
            - np.asarray(fit_state["base"])
            - np.asarray(baseline_residuals["fit"])
        )
        folds = deterministic_pair_folds(
            len(fit_x) // 2, self.decomposition.cross_validation_folds
        )
        oof = np.zeros((len(self.decomposition.ridge_lambdas), *fit_target.shape))
        for held_pairs in folds:
            held = np.sort(np.concatenate([2 * held_pairs, 2 * held_pairs + 1]))
            keep = np.ones(len(fit_x), dtype=bool)
            keep[held] = False
            path, _ = dual_ridge_path_predict(
                fit_x[keep], fit_target[keep], fit_x[held],
                self.decomposition.ridge_lambdas,
                pair_branch_weight=self.decomposition.branch_fit_weight,
            )
            oof[:, held] = path
        ladder = []
        for index, ridge in enumerate(self.decomposition.ridge_lambdas):
            residual = np.asarray(baseline_residuals["fit"]) + oof[index]
            metrics = self._role_metrics("fit", residual)
            ladder.append({
                "ridge_lambda": float(ridge),
                "cross_validation": metrics,
                "selection_score": pair_gate_selection_score(
                    metrics,
                    max_error_weight=self.decomposition.selection_max_error_weight,
                    branch_log_weight=self.decomposition.selection_branch_log_weight,
                ),
            })
        selected = min(ladder, key=lambda row: row["selection_score"])
        evaluation = np.concatenate([
            designs[role] for role in ("fit", "calibration", "development")
        ])
        path, diagnostics = dual_ridge_path_predict(
            fit_x, fit_target, evaluation, [selected["ridge_lambda"]],
            pair_branch_weight=self.decomposition.branch_fit_weight,
        )
        offset = 0
        roles = {}
        for role in ("fit", "calibration", "development"):
            count = len(designs[role])
            residual = np.asarray(baseline_residuals[role]) + path[0, offset:offset + count]
            roles[role] = self._role_metrics(role, residual)
            offset += count
        return {
            "family": family,
            "feature_width": int(fit_x.shape[-1]),
            "selected_ridge_lambda": selected["ridge_lambda"],
            "selection_score": selected["selection_score"],
            "cross_validation": selected["cross_validation"],
            "fit_diagnostics": diagnostics[0],
            "roles": roles,
            "selection_roles": ["fit_grouped_pair_cross_validation"],
            "development_used_for_selection": False,
        }

    def run_state_target_probes(self) -> Dict[str, Any]:
        if not self.state_surfaces:
            raise RuntimeError("prepare_state_target_surfaces() must run first")
        group_width = len(STATE_STATISTICS)
        normalized: Dict[str, Dict[str, np.ndarray]] = {"current": {}, "delta": {}}
        normalizers = {}
        for surface in ("current", "delta"):
            fit = np.asarray(self.state_surfaces["fit"][surface], dtype=np.float64)
            mean = fit.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(fit.std(axis=(0, 1), keepdims=True), self.decomposition.feature_epsilon)
            normalizers[surface] = (mean, scale)
            for role in self.state_surfaces:
                normalized[surface][role] = (
                    (np.asarray(self.state_surfaces[role][surface]) - mean) / scale
                ).astype(np.float32)
        np.savez_compressed(
            self.output_dir / "state_target_surface_normalizers.npz",
            **{
                f"{surface}_{kind}": value
                for surface, (mean, scale) in normalizers.items()
                for kind, value in (("mean", mean), ("scale", scale))
            },
        )
        baseline_residuals = {
            role: np.mean(np.stack([
                self.frozen_predictions[self.reassessment.audited_family][seed][role]
                for seed in self.reassessment.seeds
            ]), axis=0)
            for role in self.topology_roles
        }
        baseline_roles = {
            role: self._role_metrics(role, residual)
            for role, residual in baseline_residuals.items()
        }
        designs: Dict[str, Dict[str, np.ndarray]] = {
            "intercept_only_control": {
                role: np.zeros(
                    (*values["current"].shape[:2], 1), dtype=np.float32
                )
                for role, values in self.state_surfaces.items()
            },
            "causal_current_all": normalized["current"],
            "oracle_delta_aligned_all": normalized["delta"],
            "oracle_delta_spatial_shift_control": {
                role: np.roll(
                    values,
                    shift=self.decomposition.spatial_shift_segments,
                    axis=1,
                )
                for role, values in normalized["delta"].items()
            },
        }
        for group_index, group in enumerate(self.decomposition.state_groups):
            selected = slice(group_index * group_width, (group_index + 1) * group_width)
            designs[f"oracle_delta_{group}"] = {
                role: values[..., selected]
                for role, values in normalized["delta"].items()
            }
        runs = []
        progress = Progress("05j-g state-target probes", len(designs))
        for position, (family, family_designs) in enumerate(designs.items(), start=1):
            row = self._fit_probe(family, family_designs, baseline_residuals)
            row["improvement_vs_frozen_baseline"] = {
                role: {
                    "rmse_improvement_fraction": self._improvement(
                        baseline_roles[role]["aggregate_voltage_rmse_mv"],
                        row["roles"][role]["aggregate_voltage_rmse_mv"],
                    ),
                    "maximum_error_improvement_fraction": self._improvement(
                        baseline_roles[role]["maximum_segment_error_mv"],
                        row["roles"][role]["maximum_segment_error_mv"],
                    ),
                }
                for role in baseline_roles
            }
            runs.append(row)
            progress.update(
                position,
                f"{family} cal={row['roles']['calibration']['aggregate_voltage_rmse_mv']:.3g}",
            )
        by_family = {row["family"]: row for row in runs}
        intercept = by_family["intercept_only_control"]
        for row in runs:
            row["improvement_vs_intercept_control"] = {
                role: {
                    "rmse_improvement_fraction": self._improvement(
                        intercept["roles"][role]["aggregate_voltage_rmse_mv"],
                        row["roles"][role]["aggregate_voltage_rmse_mv"],
                    ),
                    "maximum_error_improvement_fraction": self._improvement(
                        intercept["roles"][role]["maximum_segment_error_mv"],
                        row["roles"][role]["maximum_segment_error_mv"],
                    ),
                }
                for role in baseline_roles
            }
        aligned = by_family["oracle_delta_aligned_all"]
        shifted = by_family["oracle_delta_spatial_shift_control"]
        causal = by_family["causal_current_all"]
        aligned_vs_shifted = {
            role: self._improvement(
                shifted["roles"][role]["aggregate_voltage_rmse_mv"],
                aligned["roles"][role]["aggregate_voltage_rmse_mv"],
            )
            for role in ("fit", "calibration", "development")
        }
        group_runs = [
            row for row in runs
            if row["family"].startswith("oracle_delta_")
            and row["family"] not in {
                "oracle_delta_aligned_all", "oracle_delta_spatial_shift_control"
            }
        ]
        strongest = max(
            group_runs,
            key=lambda row: row["improvement_vs_intercept_control"]["calibration"]["rmse_improvement_fraction"],
        )
        report = {
            "schema_version": "05j-g-state-target-probes-v1",
            "valid": True,
            "baseline_roles": baseline_roles,
            "runs": runs,
            "aligned_oracle_vs_spatial_shift_rmse_improvement_fraction": aligned_vs_shifted,
            "strongest_single_state_group": {
                "family": strongest["family"],
                "calibration_rmse_improvement_fraction": strongest["improvement_vs_intercept_control"]["calibration"]["rmse_improvement_fraction"],
                "development_rmse_improvement_fraction": strongest["improvement_vs_intercept_control"]["development"]["rmse_improvement_fraction"],
            },
            "causal_current_uses_future_state": False,
            "oracle_delta_uses_future_state": True,
            "future_voltage_coordinate_excluded": True,
            "oracle_or_shifted_features_used_for_candidate_training": False,
            "development_used_for_model_selection": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "state_target_probe_report.json", report)
        write_parquet(self.output_dir / "state_target_group_probe.parquet", [
            {
                "family": row["family"],
                "feature_width": row["feature_width"],
                "selected_ridge_lambda": row["selected_ridge_lambda"],
                **{
                    f"{role}_rmse_mv": row["roles"][role]["aggregate_voltage_rmse_mv"]
                    for role in ("fit", "calibration", "development")
                },
                **{
                    f"{role}_rmse_improvement_fraction": row["improvement_vs_frozen_baseline"][role]["rmse_improvement_fraction"]
                    for role in ("fit", "calibration", "development")
                },
                **{
                    f"{role}_rmse_improvement_vs_intercept_fraction": row["improvement_vs_intercept_control"][role]["rmse_improvement_fraction"]
                    for role in ("fit", "calibration", "development")
                },
            }
            for row in runs
        ])
        return report

    @staticmethod
    def _probe(report: Mapping[str, Any], family: str) -> Mapping[str, Any]:
        matches = [row for row in report["runs"] if row["family"] == family]
        if len(matches) != 1:
            raise RuntimeError(f"expected one 05j-g probe for {family}")
        return matches[0]

    def finalize_regenerative_state_target_decomposition(
        self,
        surface_report: Mapping[str, Any],
        probe_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        causal = self._probe(probe_report, "causal_current_all")
        aligned = self._probe(probe_report, "oracle_delta_aligned_all")
        shifted = self._probe(probe_report, "oracle_delta_spatial_shift_control")
        causal_gain = causal["improvement_vs_intercept_control"]
        oracle_gain = aligned["improvement_vs_intercept_control"]
        specificity = probe_report[
            "aligned_oracle_vs_spatial_shift_rmse_improvement_fraction"
        ]
        causal_material = bool(
            causal_gain["calibration"]["rmse_improvement_fraction"]
            >= self.decomposition.material_rmse_improvement_fraction
            and causal_gain["calibration"]["maximum_error_improvement_fraction"]
            >= self.decomposition.material_max_error_improvement_fraction
            and causal_gain["development"]["rmse_improvement_fraction"] > 0
        )
        oracle_material = bool(
            oracle_gain["calibration"]["rmse_improvement_fraction"]
            >= self.decomposition.material_rmse_improvement_fraction
            and oracle_gain["calibration"]["maximum_error_improvement_fraction"]
            >= self.decomposition.material_max_error_improvement_fraction
            and oracle_gain["development"]["rmse_improvement_fraction"] > 0
            and specificity["calibration"]
            >= self.decomposition.oracle_specificity_improvement_fraction
            and aligned["roles"]["development"]["aggregate_voltage_rmse_mv"]
            < shifted["roles"]["development"]["aggregate_voltage_rmse_mv"]
        )
        if causal_material:
            diagnosis = "CURRENT_REGENERATIVE_STATE_SUMMARY_IS_CAUSALLY_INFORMATIVE"
            next_experiment = "05j_h_explicit_regenerative_state_input_revision"
        elif oracle_material:
            diagnosis = "UNMODELED_REGENERATIVE_STATE_TRANSITION_DOMINATES"
            next_experiment = "05j_h_joint_regenerative_state_transition_canary"
        elif (
            oracle_gain["calibration"]["rmse_improvement_fraction"]
            >= self.decomposition.material_rmse_improvement_fraction
        ):
            diagnosis = "REGENERATIVE_STATE_LINK_NOT_CONFIRMED_ON_DEVELOPMENT"
            next_experiment = "05j_h_regenerative_support_expansion"
        else:
            diagnosis = "BOUNDARY_STATE_TARGETS_DO_NOT_EXPLAIN_LOCALIZED_VOLTAGE_ERROR"
            next_experiment = "05j_h_regime_conditioned_voltage_objective"
        report = {
            "schema_version": "05j-g-final-report-v1",
            "valid": bool(surface_report["valid"] and probe_report["valid"]),
            "decision": "DIAGNOSTIC_STATE_TARGET_DECOMPOSITION_ONLY",
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05jf": self.artifact_05jf_contract,
            "state_target_surfaces": dict(surface_report),
            "state_target_probes": dict(probe_report),
            "causal_current_material": causal_material,
            "oracle_state_delta_material": oracle_material,
            "strongest_single_state_group": probe_report["strongest_single_state_group"],
            "candidate_model_authorized": False,
            "micro_rollout_authorized": False,
            "full_training_authorized": False,
            "heldout_contract": {
                "inputs_extracted": False,
                "boundary_targets_materialized": False,
                "event_targets_materialized": False,
                "candidate_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "exact_05jf_negative_result_verified": True,
                "base_direct_tree_ensemble_frozen": True,
                "future_voltage_coordinate_excluded": True,
                "causal_probe_boundary_t_only": True,
                "future_state_delta_used_as_diagnostic_oracle_only": True,
                "spatially_shifted_capacity_control": True,
                "per_segment_intercept_control": True,
                "fit_grouped_pair_cv_selected_ridge": True,
                "development_used_for_model_selection": False,
                "heldout_inputs_extracted": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": next_experiment,
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
            "schema_version": "05j-g-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
