"""05j-k: interpretability and voltage-objective audit after 05j-j."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import (
    EXPECTED_05JI_INDEX_SHA256,
    HinesRegenerativeStateConfirmation,
    _verified_artifact_root,
)


EXPECTED_05JJ_ARCHIVE_SHA256 = (
    "3c883f7f8d6b744857e91a467228111ce73fcdc267849df8563d8fc41d9b7e5e"
)
EXPECTED_05JJ_INDEX_SHA256 = (
    "1c7591ddfcc585364160493a541519b5b4eea1e07fed8a4222f03c61aa6dc257"
)
EXPECTED_05JJ_FINAL_SHA256 = (
    "f7ba0fa5b81dfe3f8b9466ad03be4a17a5363eb0f4cb08b935805f219b5c0a7c"
)


@dataclass(frozen=True)
class HinesVoltageObjectiveReassessmentConfig:
    feature_abs_z_warning: float = 10.0
    feature_abs_z_blocker: float = 50.0
    feature_outside_fit_envelope_fraction: float = 0.05
    oracle_explosion_ratio: float = 5.0
    target_delta_shift_ratio: float = 2.0
    external_to_fit_rmse_ratio: float = 4.0
    branch_amplification_threshold: float = 2.0
    activation_voltage_threshold_mv: float = -45.0
    residual_limit_fraction_warning: float = 0.8

    def validate(self) -> None:
        positive = (
            self.feature_abs_z_warning,
            self.feature_abs_z_blocker,
            self.feature_outside_fit_envelope_fraction,
            self.oracle_explosion_ratio,
            self.target_delta_shift_ratio,
            self.external_to_fit_rmse_ratio,
            self.branch_amplification_threshold,
            self.residual_limit_fraction_warning,
        )
        if min(positive) <= 0:
            raise ValueError("05j-k thresholds must be positive")
        if self.feature_abs_z_blocker <= self.feature_abs_z_warning:
            raise ValueError("05j-k blocker must exceed warning threshold")
        if not 0 < self.feature_outside_fit_envelope_fraction < 1:
            raise ValueError("05j-k envelope fraction must lie in (0, 1)")
        if not 0 < self.residual_limit_fraction_warning <= 1:
            raise ValueError("05j-k residual-limit fraction must lie in (0, 1]")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesVoltageObjectiveReassessmentConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def _absolute_summary(values: np.ndarray) -> Dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    return {
        "median": float(np.quantile(absolute, 0.50)),
        "p90": float(np.quantile(absolute, 0.90)),
        "p95": float(np.quantile(absolute, 0.95)),
        "p99": float(np.quantile(absolute, 0.99)),
        "p999": float(np.quantile(absolute, 0.999)),
        "maximum": float(np.max(absolute)),
    }


def feature_transport_summary(
    fit: np.ndarray,
    external: np.ndarray,
    *,
    epsilon: float,
) -> Dict[str, Any]:
    fit = np.asarray(fit, dtype=np.float64)
    external = np.asarray(external, dtype=np.float64)
    if fit.ndim != 3 or external.shape[1:] != fit.shape[1:]:
        raise ValueError("05j-k feature surfaces disagree")
    mean = fit.mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(fit.std(axis=(0, 1), keepdims=True), float(epsilon))
    fit_z = (fit - mean) / scale
    external_z = (external - mean) / scale
    fit_min = fit.min(axis=0, keepdims=True)
    fit_max = fit.max(axis=0, keepdims=True)
    outside = (external < fit_min) | (external > fit_max)
    fit_summary = _absolute_summary(fit_z)
    external_summary = _absolute_summary(external_z)
    return {
        "fit_abs_z": fit_summary,
        "external_abs_z": external_summary,
        "external_to_fit_p99_ratio": float(
            external_summary["p99"] / max(fit_summary["p99"], 1e-12)
        ),
        "external_outside_fit_envelope_fraction": float(np.mean(outside)),
        "external_abs_z_gt_5_fraction": float(np.mean(np.abs(external_z) > 5.0)),
        "external_abs_z_gt_10_fraction": float(np.mean(np.abs(external_z) > 10.0)),
        "external_abs_z_gt_50_fraction": float(np.mean(np.abs(external_z) > 50.0)),
        "finite": bool(np.all(np.isfinite(fit_z)) and np.all(np.isfinite(external_z))),
    }


def voltage_error_summary(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    return {
        "rmse_mv": float(np.sqrt(np.mean(error * error))),
        "mae_mv": float(np.mean(np.abs(error))),
        "p95_absolute_error_mv": float(np.quantile(np.abs(error), 0.95)),
        "p99_absolute_error_mv": float(np.quantile(np.abs(error), 0.99)),
        "maximum_absolute_error_mv": float(np.max(np.abs(error))),
    }


class HinesVoltageDecoderObjectiveReassessment(HinesRegenerativeStateConfirmation):
    """Determine whether 05j-j is a valid rejection or a support/objective failure."""

    def __init__(
        self,
        *args: Any,
        objective_config: HinesVoltageObjectiveReassessmentConfig,
        artifact_05jj_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        objective_config.validate()
        self.objective = objective_config
        self.artifact_05jj_source = Path(artifact_05jj_source).resolve()
        self.artifact_05jj_report: Dict[str, Any] = {}
        self.artifact_05jj_contract: Dict[str, Any] = {}

    def prepare_voltage_objective_reassessment(self) -> Dict[str, Any]:
        base = self.prepare_independent_confirmation()
        cache = self.output_dir.parent / ".05j_k_artifact_cache"
        _, report, contract = _verified_artifact_root(
            self.artifact_05jj_source,
            cache,
            marker_name="independent_confirmation_config.json",
            archive_sha256=EXPECTED_05JJ_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JJ_INDEX_SHA256,
            final_sha256=EXPECTED_05JJ_FINAL_SHA256,
        )
        blockers = []
        if not report.get("valid"):
            blockers.append("05j-j artifact is not valid")
        if report.get("diagnosis") != "INDEPENDENT_SUPPORT_REJECTS_REGENERATIVE_STATE_EXPLANATION":
            blockers.append("05j-j diagnosis does not route to objective reassessment")
        if report.get("artifact_05ji", {}).get("artifact_index_sha256") != EXPECTED_05JI_INDEX_SHA256:
            blockers.append("05j-j embeds a different 05j-i artifact")
        if report.get("candidate_model_authorized") or report.get("micro_rollout_authorized"):
            blockers.append("05j-j unexpectedly authorized candidate work")
        methodology = report.get("methodology", {})
        if not methodology.get("confirmation_outcomes_not_used_for_feature_or_probe_selection"):
            blockers.append("05j-j independent support was used for selection")
        if blockers:
            raise RuntimeError(f"05j-k provenance blockers: {blockers}")
        self.artifact_05jj_report, self.artifact_05jj_contract = report, contract
        payload = {
            "schema_version": "05j-k-objective-reassessment-config-v1",
            "voltage_objective_reassessment": asdict(self.objective),
            "artifact_05jj": contract,
            "formal_05jj_decision_preserved": True,
            "interpretation_audit_is_post_result": True,
            "new_support_used_for_training": False,
            "candidate_training_performed": False,
            "feature_or_probe_selection_performed": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "voltage_objective_reassessment_config.json", payload)
        return {**base, **payload}

    def audit_oracle_transport(self) -> Dict[str, Any]:
        if not self.state_surfaces:
            raise RuntimeError("prepare_external_state_surfaces() must run first")
        surfaces = {}
        for name in ("current", "delta"):
            surfaces[name] = feature_transport_summary(
                self.state_surfaces["fit"][name],
                self.state_surfaces["calibration"][name],
                epsilon=self.decomposition.feature_epsilon,
            )
        exact = self.artifact_05jj_report["external_probes"]
        runs = {row["family"]: row for row in exact["runs"]}
        aligned = runs["oracle_delta_aligned_all"]
        shifted = runs["oracle_delta_spatial_shift_control"]
        intercept = runs["intercept_only_control"]
        aligned_rmse = float(
            aligned["roles"]["calibration"]["aggregate_voltage_rmse_mv"]
        )
        intercept_rmse = float(
            intercept["roles"]["calibration"]["aggregate_voltage_rmse_mv"]
        )
        explosion_ratio = aligned_rmse / max(intercept_rmse, 1e-12)
        aligned_diagnostics = aligned["fit_diagnostics"]
        shifted_diagnostics = shifted["fit_diagnostics"]
        delta = surfaces["delta"]
        feature_ood = bool(
            delta["external_abs_z"]["p99"] >= self.objective.feature_abs_z_warning
            or delta["external_abs_z"]["maximum"] >= self.objective.feature_abs_z_blocker
            or delta["external_outside_fit_envelope_fraction"]
            >= self.objective.feature_outside_fit_envelope_fraction
        )
        numerical_explosion = explosion_ratio >= self.objective.oracle_explosion_ratio
        regularization_asymmetry = float(
            shifted["selected_ridge_lambda"] / max(aligned["selected_ridge_lambda"], 1e-30)
        )
        report = {
            "schema_version": "05j-k-oracle-transport-audit-v1",
            "valid": all(row["finite"] for row in surfaces.values()),
            "surfaces": surfaces,
            "registered_external_metrics": {
                "intercept_rmse_mv": intercept_rmse,
                "aligned_oracle_rmse_mv": aligned_rmse,
                "spatial_shift_rmse_mv": float(
                    shifted["roles"]["calibration"]["aggregate_voltage_rmse_mv"]
                ),
                "aligned_to_intercept_rmse_ratio": explosion_ratio,
                "aligned_vs_shifted_improvement_fraction": exact[
                    "aligned_oracle_vs_spatial_shift_rmse_improvement_fraction"
                ]["calibration"],
            },
            "fit_diagnostics": {
                "aligned_regularized_condition_number_maximum": aligned_diagnostics[
                    "maximum_regularized_condition_number"
                ],
                "aligned_coefficient_l2_norm_maximum": aligned_diagnostics[
                    "maximum_segment_coefficient_l2_norm"
                ],
                "aligned_ridge_lambda": aligned["selected_ridge_lambda"],
                "shifted_ridge_lambda": shifted["selected_ridge_lambda"],
                "shifted_to_aligned_ridge_ratio": regularization_asymmetry,
                "shifted_regularized_condition_number_maximum": shifted_diagnostics[
                    "maximum_regularized_condition_number"
                ],
            },
            "feature_transport_ood": feature_ood,
            "numerical_oracle_explosion": numerical_explosion,
            "oracle_test_interpretable_as_biological_rejection": bool(
                not feature_ood and not numerical_explosion
            ),
            "formal_05jj_decision_overridden": False,
            "audit_is_post_result": True,
        }
        _write_json(self.output_dir / "oracle_transport_audit.json", report)
        return report

    def audit_voltage_objective(self) -> Dict[str, Any]:
        role_fit = self.topology_roles["fit"]
        role_external = self.topology_roles["calibration"]
        family = self.reassessment.audited_family
        residual_fit = np.mean(np.stack([
            self.frozen_predictions[family][seed]["fit"]
            for seed in self.reassessment.seeds
        ]), axis=0)
        residual_external = np.mean(np.stack([
            self.frozen_predictions[family][seed]["calibration"]
            for seed in self.reassessment.seeds
        ]), axis=0)
        predictions = {
            "persistence": np.asarray(role_external["voltage_t"]),
            "frozen_h2": np.asarray(role_external["base"]),
            "frozen_direct_tree": np.asarray(role_external["base"]) + residual_external,
        }
        target = np.asarray(role_external["target"])
        baseline_metrics = {
            name: voltage_error_summary(prediction, target)
            for name, prediction in predictions.items()
        }
        fit_prediction = np.asarray(role_fit["base"]) + residual_fit
        fit_metrics = voltage_error_summary(fit_prediction, role_fit["target"])
        external_metrics = baseline_metrics["frozen_direct_tree"]
        rmse_ratio = external_metrics["rmse_mv"] / max(fit_metrics["rmse_mv"], 1e-12)

        fit_delta = np.asarray(role_fit["target"]) - np.asarray(role_fit["voltage_t"])
        external_delta = target - np.asarray(role_external["voltage_t"])
        fit_delta_summary = _absolute_summary(fit_delta)
        external_delta_summary = _absolute_summary(external_delta)
        delta_shift = external_delta_summary["p99"] / max(
            fit_delta_summary["p99"], 1e-12
        )
        activation = target >= self.objective.activation_voltage_threshold_mv
        fit_activation = np.asarray(role_fit["target"]) >= self.objective.activation_voltage_threshold_mv
        activation_support = {
            "fit_fraction": float(np.mean(fit_activation)),
            "external_fraction": float(np.mean(activation)),
            "external_active_coordinate_count": int(np.sum(activation)),
        }
        direct_error = predictions["frozen_direct_tree"] - target
        active_error = np.abs(direct_error[activation])
        inactive_error = np.abs(direct_error[~activation])

        region_rows = []
        regions = np.asarray([str(row["region"]) for row in self.layout.segments])
        for region in sorted(set(regions.tolist())):
            selected = regions == region
            metrics = voltage_error_summary(
                predictions["frozen_direct_tree"][:, selected], target[:, selected]
            )
            region_rows.append({
                "region": region,
                "segment_count": int(np.sum(selected)),
                **metrics,
                "error_energy_fraction": float(
                    np.sum(direct_error[:, selected] ** 2)
                    / max(np.sum(direct_error ** 2), 1e-12)
                ),
            })
        write_parquet(self.output_dir / "external_region_error.parquet", region_rows)

        exact_baseline = self.artifact_05jj_report["external_probes"][
            "baseline_roles"
        ]["calibration"]
        branch_median = float(exact_baseline["median_branching_retention"])
        target_residual = target - np.asarray(role_external["base"])
        residual_limit = float(self.topology.target_residual_limit_mv)
        residual_limit_fraction = float(
            np.mean(
                np.abs(target_residual)
                >= self.objective.residual_limit_fraction_warning * residual_limit
            )
        )
        support_shift = bool(
            delta_shift >= self.objective.target_delta_shift_ratio
            or rmse_ratio >= self.objective.external_to_fit_rmse_ratio
            or (
                activation_support["external_fraction"]
                > 2.0 * max(activation_support["fit_fraction"], 1e-9)
            )
        )
        report = {
            "schema_version": "05j-k-voltage-objective-audit-v1",
            "valid": True,
            "baseline_metrics": baseline_metrics,
            "fit_frozen_direct_tree_metrics": fit_metrics,
            "external_to_fit_rmse_ratio": rmse_ratio,
            "teacher_transition_delta": {
                "fit": fit_delta_summary,
                "external": external_delta_summary,
                "external_to_fit_p99_ratio": delta_shift,
            },
            "activation_support": activation_support,
            "active_coordinate_absolute_error": {
                "count": int(len(active_error)),
                "mean_mv": float(np.mean(active_error)) if len(active_error) else None,
                "p95_mv": float(np.quantile(active_error, 0.95)) if len(active_error) else None,
            },
            "inactive_coordinate_absolute_error": {
                "count": int(len(inactive_error)),
                "mean_mv": float(np.mean(inactive_error)) if len(inactive_error) else None,
                "p95_mv": float(np.quantile(inactive_error, 0.95)) if len(inactive_error) else None,
            },
            "region_error": region_rows,
            "branching": {
                "median_predicted_to_teacher_distance_ratio": branch_median,
                "over_amplified": branch_median
                >= self.objective.branch_amplification_threshold,
            },
            "target_residual_parameterization": {
                "limit_mv": residual_limit,
                "maximum_absolute_target_residual_mv": float(
                    np.max(np.abs(target_residual))
                ),
                "fraction_above_warning_limit": residual_limit_fraction,
                "limit_is_primary_blocker": bool(residual_limit_fraction > 0),
            },
            "near_regenerative_support_shift": support_shift,
            "new_support_used_for_training": False,
            "candidate_training_performed": False,
        }
        _write_json(self.output_dir / "voltage_objective_audit.json", report)
        return report

    def finalize_voltage_objective_reassessment(
        self,
        transport_report: Mapping[str, Any],
        objective_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        interpretable = bool(
            transport_report["oracle_test_interpretable_as_biological_rejection"]
        )
        support_shift = bool(objective_report["near_regenerative_support_shift"])
        limit_blocker = bool(
            objective_report["target_residual_parameterization"][
                "limit_is_primary_blocker"
            ]
        )
        if not interpretable and support_shift:
            diagnosis = "NEAR_REGENERATIVE_SUPPORT_AND_OBJECTIVE_MISMATCH_DOMINATES"
            next_step = "05j_l_acquire_train_and_fresh_test_near_regenerative_support"
        elif limit_blocker:
            diagnosis = "VOLTAGE_TARGET_PARAMETERIZATION_LIMIT_DOMINATES"
            next_step = "05j_l_residual_target_parameterization_canary"
        elif support_shift:
            diagnosis = "NEAR_REGENERATIVE_TRAIN_SUPPORT_MISSING"
            next_step = "05j_l_acquire_train_and_fresh_test_near_regenerative_support"
        else:
            diagnosis = "VOLTAGE_OBJECTIVE_MISMATCH_WITHIN_EXISTING_SUPPORT"
            next_step = "05j_l_regime_balanced_voltage_objective_canary"
        report = {
            "schema_version": "05j-k-final-report-v1",
            "valid": bool(transport_report["valid"] and objective_report["valid"]),
            "decision": "POST_RESULT_INTERPRETABILITY_AND_VOLTAGE_OBJECTIVE_AUDIT",
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "artifact_05jj": self.artifact_05jj_contract,
            "formal_05jj_result": {
                "preserved": True,
                "diagnosis": self.artifact_05jj_report["diagnosis"],
                "regenerative_state_transition_confirmed": False,
            },
            "scientific_interpretation": {
                "oracle_rejection_interpretable": interpretable,
                "numerical_transport_failure": bool(
                    transport_report["numerical_oracle_explosion"]
                ),
                "feature_transport_ood": bool(
                    transport_report["feature_transport_ood"]
                ),
                "near_regenerative_support_shift": support_shift,
            },
            "oracle_transport_audit": dict(transport_report),
            "voltage_objective_audit": dict(objective_report),
            "candidate_model_authorized": False,
            "micro_rollout_authorized": False,
            "full_training_authorized": False,
            "methodology": {
                "formal_05jj_decision_preserved": True,
                "audit_is_post_result": True,
                "new_support_used_for_training": False,
                "candidate_training_performed": False,
                "feature_or_probe_selection_performed": False,
                "heldout_inputs_extracted": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": next_step,
                "fresh_heldout_support_required": "acquire" in next_step,
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
            "schema_version": "05j-k-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
