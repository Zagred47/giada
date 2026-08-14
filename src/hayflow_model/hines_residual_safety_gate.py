"""05j-l: fit-only safety gates for the frozen direct-tree residual decoder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .hines_repaired_representation_revision import pair_gate_selection_score
from .hines_spatial_support_revision import deterministic_pair_folds
from .hines_voltage_objective_reassessment import (
    HinesVoltageDecoderObjectiveReassessment,
)


EXPECTED_05JK_ARCHIVE_SHA256 = (
    "e18bf9ee7ee3037d209e1f36b180b86c3dadca357f3df3a4d7de21f37c71e85e"
)
EXPECTED_05JK_INDEX_SHA256 = (
    "18f02c2ccc963649dafcdf32853058ad3e78c34d57a03d312414514ccbc2ef6c"
)
EXPECTED_05JK_FINAL_SHA256 = (
    "220de838a245a38fb322a832685cf4454e2cc0b800819510464e690d046bf0d1"
)

SAFETY_FAMILIES = (
    "combined_all",
    "combined_clip_uncertainty",
    "sample_energy_scale",
    "segment_clip",
    "uncertainty_fallback",
    "identity_direct_tree",
)


@dataclass(frozen=True)
class HinesResidualSafetyGateConfig:
    quantiles: Tuple[float, ...] = (0.95, 0.99, 1.0)
    cross_validation_folds: int = 6
    near_best_cv_tolerance_fraction: float = 0.02
    external_h2_tolerance_fraction: float = 0.25
    minimum_external_improvement_vs_direct_tree_fraction: float = 0.80
    family_priority: Tuple[str, ...] = SAFETY_FAMILIES

    def validate(self) -> None:
        if tuple(self.family_priority) != SAFETY_FAMILIES:
            raise ValueError("05j-l safety-family priority is preregistered")
        if self.cross_validation_folds < 2:
            raise ValueError("05j-l needs grouped pair cross-validation")
        if not self.quantiles or any(not 0 < value <= 1 for value in self.quantiles):
            raise ValueError("05j-l quantiles must lie in (0, 1]")
        if tuple(sorted(self.quantiles)) != tuple(self.quantiles):
            raise ValueError("05j-l quantiles must be ordered")
        fractions = (
            self.near_best_cv_tolerance_fraction,
            self.external_h2_tolerance_fraction,
            self.minimum_external_improvement_vs_direct_tree_fraction,
        )
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("05j-l fractions must lie in (0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesResidualSafetyGateConfig":
        payload = dict(values)
        for name in ("quantiles", "family_priority"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def residual_gate_thresholds(
    mean_residual: np.ndarray,
    disagreement: np.ndarray,
    quantile: float,
) -> Dict[str, np.ndarray | float]:
    mean_residual = np.asarray(mean_residual, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=np.float64)
    return {
        "segment_abs": np.quantile(np.abs(mean_residual), quantile, axis=0),
        "segment_disagreement": np.quantile(disagreement, quantile, axis=0),
        "sample_energy": float(
            np.quantile(np.sqrt(np.mean(mean_residual * mean_residual, axis=1)), quantile)
        ),
    }


def apply_residual_safety_gate(
    mean_residual: np.ndarray,
    disagreement: np.ndarray,
    thresholds: Mapping[str, Any],
    family: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    residual = np.asarray(mean_residual, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=np.float64)
    if family not in SAFETY_FAMILIES:
        raise ValueError(f"unknown residual safety family {family!r}")
    output = residual.copy()
    clipped = np.zeros_like(output, dtype=bool)
    uncertain = np.zeros_like(output, dtype=bool)
    energy_scale = np.ones((len(output), 1), dtype=np.float64)
    if family in {"segment_clip", "combined_clip_uncertainty", "combined_all"}:
        limit = np.asarray(thresholds["segment_abs"], dtype=np.float64)[None, :]
        clipped = np.abs(output) > limit
        output = np.clip(output, -limit, limit)
    if family in {"uncertainty_fallback", "combined_clip_uncertainty", "combined_all"}:
        limit = np.asarray(thresholds["segment_disagreement"], dtype=np.float64)[None, :]
        uncertain = disagreement > limit
        output = np.where(uncertain, 0.0, output)
    if family in {"sample_energy_scale", "combined_all"}:
        energy = np.sqrt(np.mean(output * output, axis=1, keepdims=True))
        limit = float(thresholds["sample_energy"])
        energy_scale = np.minimum(1.0, limit / np.maximum(energy, 1e-12))
        output = output * energy_scale
    return output.astype(np.float32), {
        "clipped_coordinate_fraction": float(np.mean(clipped)),
        "uncertainty_fallback_fraction": float(np.mean(uncertain)),
        "scaled_sample_fraction": float(np.mean(energy_scale < 1.0)),
        "mean_energy_scale": float(np.mean(energy_scale)),
        "maximum_absolute_output_mv": float(np.max(np.abs(output))),
    }


class HinesResidualSafetyGateCanary(HinesVoltageDecoderObjectiveReassessment):
    """Select a conservative, non-trainable gate on original fit pairs only."""

    def __init__(
        self,
        *args: Any,
        safety_config: HinesResidualSafetyGateConfig,
        artifact_05jk_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        safety_config.validate()
        self.safety = safety_config
        self.artifact_05jk_source = Path(artifact_05jk_source).resolve()
        self.artifact_05jk_report: Dict[str, Any] = {}
        self.artifact_05jk_contract: Dict[str, Any] = {}
        self.selected_gate: Dict[str, Any] = {}

    def prepare_residual_safety_gate(self) -> Dict[str, Any]:
        base = self.prepare_voltage_objective_reassessment()
        cache = self.output_dir.parent / ".05j_l_artifact_cache"
        _, report, contract = _verified_artifact_root(
            self.artifact_05jk_source,
            cache,
            marker_name="voltage_objective_reassessment_config.json",
            archive_sha256=EXPECTED_05JK_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JK_INDEX_SHA256,
            final_sha256=EXPECTED_05JK_FINAL_SHA256,
        )
        blockers = []
        if not report.get("valid"):
            blockers.append("05j-k artifact is invalid")
        if not report.get("formal_05jj_result", {}).get("preserved"):
            blockers.append("05j-k did not preserve the formal 05j-j result")
        if not report.get("scientific_interpretation", {}).get("numerical_transport_failure"):
            blockers.append("05j-k did not establish a numerical transport failure")
        baselines = report.get("voltage_objective_audit", {}).get("baseline_metrics", {})
        if baselines.get("frozen_h2", {}).get("rmse_mv", 1e9) >= baselines.get("frozen_direct_tree", {}).get("rmse_mv", 0):
            blockers.append("05j-k does not show residual-decoder harm")
        if blockers:
            raise RuntimeError(f"05j-l provenance blockers: {blockers}")
        self.artifact_05jk_report, self.artifact_05jk_contract = report, contract
        payload = {
            "schema_version": "05j-l-residual-safety-config-v1",
            "residual_safety_gate": asdict(self.safety),
            "artifact_05jk": contract,
            "selection_support": "original_05jh_fit_grouped_pair_cross_validation",
            "05ji_support_role": "descriptive_post_result_only",
            "fresh_test_required_before_authorization": True,
            "base_h2_frozen": True,
            "direct_tree_checkpoints_frozen": True,
            "gradient_training_performed": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "residual_safety_gate_config.json", payload)
        return {**base, **payload}

    def _ensemble(self, role: str) -> Tuple[np.ndarray, np.ndarray]:
        values = np.stack([
            self.frozen_predictions[self.reassessment.audited_family][seed][role]
            for seed in self.reassessment.seeds
        ])
        return np.mean(values, axis=0), np.std(values, axis=0)

    def _metrics_for(self, role: str, residual: np.ndarray) -> Dict[str, Any]:
        state = self.topology_roles[role]
        return self._pair_set_metrics(
            (np.asarray(state["base"]) + np.asarray(residual)).reshape(
                -1, 2, self.layout.segment_count
            ),
            np.asarray(state["target"]).reshape(-1, 2, self.layout.segment_count),
        )

    def select_gate_on_fit_cross_validation(self) -> Dict[str, Any]:
        fit_mean, fit_disagreement = self._ensemble("fit")
        pair_count = len(fit_mean) // 2
        folds = deterministic_pair_folds(pair_count, self.safety.cross_validation_folds)
        rows = []
        for family in self.safety.family_priority:
            for quantile in self.safety.quantiles:
                oof = np.zeros_like(fit_mean, dtype=np.float32)
                diagnostics = []
                for held_pairs in folds:
                    held = np.sort(np.concatenate([2 * held_pairs, 2 * held_pairs + 1]))
                    keep = np.ones(len(fit_mean), dtype=bool); keep[held] = False
                    thresholds = residual_gate_thresholds(
                        fit_mean[keep], fit_disagreement[keep], float(quantile)
                    )
                    oof[held], row = apply_residual_safety_gate(
                        fit_mean[held], fit_disagreement[held], thresholds, family
                    )
                    diagnostics.append(row)
                metrics = self._metrics_for("fit", oof)
                rows.append({
                    "family": family,
                    "quantile": float(quantile),
                    "cross_validation": metrics,
                    "selection_score": pair_gate_selection_score(
                        metrics,
                        max_error_weight=self.decomposition.selection_max_error_weight,
                        branch_log_weight=self.decomposition.selection_branch_log_weight,
                    ),
                    "mean_clipped_coordinate_fraction": float(np.mean([
                        row["clipped_coordinate_fraction"] for row in diagnostics
                    ])),
                    "mean_uncertainty_fallback_fraction": float(np.mean([
                        row["uncertainty_fallback_fraction"] for row in diagnostics
                    ])),
                    "mean_scaled_sample_fraction": float(np.mean([
                        row["scaled_sample_fraction"] for row in diagnostics
                    ])),
                })
        best = min(float(row["selection_score"]) for row in rows)
        eligible = [
            row for row in rows
            if float(row["selection_score"])
            <= best * (1.0 + self.safety.near_best_cv_tolerance_fraction)
        ]
        priority = {name: index for index, name in enumerate(self.safety.family_priority)}
        selected = min(
            eligible,
            key=lambda row: (priority[row["family"]], row["quantile"]),
        )
        thresholds = residual_gate_thresholds(
            fit_mean, fit_disagreement, selected["quantile"]
        )
        self.selected_gate = {**selected, "thresholds": thresholds}
        report = {
            "schema_version": "05j-l-fit-only-gate-selection-v1",
            "valid": True,
            "fit_pair_count": pair_count,
            "cross_validation_folds": len(folds),
            "candidate_count": len(rows),
            "best_selection_score": best,
            "near_best_tolerance_fraction": self.safety.near_best_cv_tolerance_fraction,
            "eligible_candidate_count": len(eligible),
            "selected_family": selected["family"],
            "selected_quantile": selected["quantile"],
            "selected_cross_validation": selected["cross_validation"],
            "selection_support": "original_05jh_fit_only",
            "05ji_used_for_selection": False,
            "gradient_training_performed": False,
            "runs": rows,
        }
        _write_json(self.output_dir / "fit_only_gate_selection.json", report)
        write_parquet(self.output_dir / "fit_only_gate_ladder.parquet", [{
            "family": row["family"],
            "quantile": row["quantile"],
            "selection_score": row["selection_score"],
            "cv_rmse_mv": row["cross_validation"]["aggregate_voltage_rmse_mv"],
            "cv_maximum_error_mv": row["cross_validation"]["maximum_segment_error_mv"],
            "cv_median_branching_retention": row["cross_validation"]["median_branching_retention"],
        } for row in rows])
        return report

    def evaluate_frozen_gate(self) -> Dict[str, Any]:
        if not self.selected_gate:
            raise RuntimeError("select_gate_on_fit_cross_validation() must run first")
        family = str(self.selected_gate["family"])
        thresholds = self.selected_gate["thresholds"]
        roles = {}
        for role in ("fit", "development", "calibration"):
            mean, disagreement = self._ensemble(role)
            gated, diagnostics = apply_residual_safety_gate(
                mean, disagreement, thresholds, family
            )
            zero = np.zeros_like(mean)
            roles[role] = {
                "h2": self._metrics_for(role, zero),
                "direct_tree": self._metrics_for(role, mean),
                "gated": self._metrics_for(role, gated),
                "gate_diagnostics": diagnostics,
            }
        external = roles["calibration"]
        h2_rmse = float(external["h2"]["aggregate_voltage_rmse_mv"])
        tree_rmse = float(external["direct_tree"]["aggregate_voltage_rmse_mv"])
        gated_rmse = float(external["gated"]["aggregate_voltage_rmse_mv"])
        improvement = (tree_rmse - gated_rmse) / max(tree_rmse, 1e-12)
        close_to_h2 = gated_rmse <= h2_rmse * (
            1.0 + self.safety.external_h2_tolerance_fraction
        )
        descriptive_rescue = bool(
            improvement >= self.safety.minimum_external_improvement_vs_direct_tree_fraction
            and close_to_h2
        )
        report = {
            "schema_version": "05j-l-frozen-gate-evaluation-v1",
            "valid": True,
            "selected_family": family,
            "selected_quantile": self.selected_gate["quantile"],
            "roles": roles,
            "external_descriptive": {
                "direct_tree_to_gated_rmse_improvement_fraction": improvement,
                "gated_within_h2_tolerance": close_to_h2,
                "descriptive_rescue": descriptive_rescue,
                "used_for_gate_selection": False,
                "used_for_authorization": False,
            },
            "fresh_test_required_before_authorization": True,
            "gradient_training_performed": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "frozen_gate_evaluation.json", report)
        return report

    def finalize_residual_safety_gate(
        self,
        selection_report: Mapping[str, Any],
        evaluation_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        rescue = bool(evaluation_report["external_descriptive"]["descriptive_rescue"])
        if rescue:
            diagnosis = "FIT_ONLY_SAFETY_GATE_DESCRIPTIVELY_RESCUES_EXTERNAL_DECODER"
            next_step = "05j_m_acquire_fresh_near_regenerative_test_for_frozen_gate"
        else:
            diagnosis = "FIT_ONLY_SAFETY_GATE_DOES_NOT_RESCUE_EXTERNAL_DECODER"
            next_step = "05j_m_acquire_train_and_fresh_test_near_regenerative_support"
        report = {
            "schema_version": "05j-l-final-report-v1",
            "valid": bool(selection_report["valid"] and evaluation_report["valid"]),
            "decision": "FROZEN_RESIDUAL_SAFETY_GATE_CANARY",
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "artifact_05jk": self.artifact_05jk_contract,
            "selection": dict(selection_report),
            "evaluation": dict(evaluation_report),
            "candidate_authorized": False,
            "fresh_test_required_before_authorization": True,
            "micro_rollout_authorized": False,
            "full_training_authorized": False,
            "methodology": {
                "gate_thresholds_fit_on_original_fit_only": True,
                "gate_selected_by_original_fit_grouped_cv_only": True,
                "05ji_used_descriptively_after_selection": True,
                "05ji_used_for_selection": False,
                "gradient_training_performed": False,
                "base_h2_and_direct_tree_frozen": True,
                "heldout_inputs_extracted": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": next_step,
                "fresh_test_required": True,
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
            "schema_version": "05j-l-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
