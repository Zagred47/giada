"""Notebook-05e closed-form capacity probes for the HayFlow-Hines boundary path."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import CompositeFlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_conditioning_experiment import (
    HinesConditioningConfig,
    HinesResidualConditioningExperiment,
)
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import HinesIsolationConfig, sha256_file
from .hines_layer import require_torch

try:  # Keep the closed-form utilities usable in CPU-light environments.
    import torch
except ImportError:  # pragma: no cover - exercised by data-only environments.
    torch = None


EXPECTED_05D_ARCHIVE_SHA256 = (
    "61bf814a22c313093c18a7a555d76515d3d3db741eebae167b8ff479b8d7309c"
)
EXPECTED_05D_MEMBER_SHA256 = {
    "artifact_index.json": "aa177bdea706e705f8cbd1b7d96b0d66397a145390fbb2fa08719dfed6c690f1",
    "conditioning_config.json": "01ab26de245eabeb67278636b3577228aae7b8868e97711d30a78f0790d8eeca",
    "final_report.json": "df3b97191fbb93c663ded2830d53a0ea0077838a0636b499273d17696ca9bbbf",
    "free_residual_report.json": "ad43f8f3b093ee58f7242732896e04947fcd5212bb2c93f2dba92e4a59a82f2a",
    "frozen_decoder_sweep_report.json": "0002f34079ac8d977bdf9c83a2fff5bf5cbf88de8e05777558ec9749abc93670",
    "unfreezing_ladder_report.json": "9d250734b8714dc9370ed494fad66534b5de489c1e37e70c14a96ea4c1695dd3",
}


@dataclass(frozen=True)
class HinesCapacityConfig:
    rank_candidates: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 96)
    feature_epsilon: float = 1e-8
    svd_rcond: float = 1e-10
    one_transition_rmse_mv: float = 0.25
    one_transition_max_error_mv: float = 1.0
    pair_rmse_mv: float = 1.0
    pair_max_error_mv: float = 5.0
    pair_retention_minimum: float = 0.90
    pair_retention_maximum: float = 1.10

    def validate(self) -> None:
        ranks = tuple(int(value) for value in self.rank_candidates)
        if not ranks or min(ranks) <= 0:
            raise ValueError("rank candidates must be positive")
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("rank candidates must be unique and increasing")
        if self.feature_epsilon <= 0 or not 0 < self.svd_rcond < 1:
            raise ValueError("feature epsilon and SVD rcond are invalid")
        if not 0 < self.pair_retention_minimum < self.pair_retention_maximum:
            raise ValueError("invalid pair-retention interval")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesCapacityConfig":
        payload = dict(values)
        if "rank_candidates" in payload:
            payload["rank_candidates"] = tuple(payload["rank_candidates"])
        result = cls(**payload)
        result.validate()
        return result


def standardize_design(
    features: np.ndarray, epsilon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize a 2-D design without hiding constant columns."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    mean = values.mean(axis=0, keepdims=True)
    raw_std = values.std(axis=0, keepdims=True)
    scale = np.maximum(raw_std, float(epsilon))
    return (values - mean) / scale, mean.reshape(-1), raw_std.reshape(-1)


def design_spectrum(design: np.ndarray, rcond: float) -> Dict[str, Any]:
    """Return the numerical rank and singular spectrum used by a probe."""

    matrix = np.asarray(design, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = float(rcond) * leading
    nonzero = singular_values[singular_values > tolerance]
    condition = (
        float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
    )
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "numerical_rank": int(len(nonzero)),
        "rank_fraction": float(len(nonzero) / max(1, min(matrix.shape))),
        "rcond": float(rcond),
        "condition_number_nonzero": condition,
        "singular_values": [float(value) for value in singular_values],
    }


def solve_linear_probe(
    design: np.ndarray, target: np.ndarray, rcond: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve a deterministic least-squares probe in float64."""

    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(values):
        raise ValueError("design and target dimensions disagree")
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        matrix, values, rcond=float(rcond)
    )
    prediction = matrix @ coefficients
    error = prediction - values
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = float(rcond) * leading
    nonzero = singular_values[singular_values > tolerance]
    report = {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "numerical_rank": int(len(nonzero)),
        "rank_fraction": float(len(nonzero) / max(1, min(matrix.shape))),
        "rcond": float(rcond),
        "condition_number_nonzero": (
            float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
        ),
        "singular_values": [float(value) for value in singular_values],
        "lstsq_rank": int(rank),
        "residual_sum_squares": float(np.sum(error ** 2)),
        "irreducible_rmse": float(np.sqrt(np.mean(error ** 2))),
        "reported_residual_sum_squares": (
            float(residuals[0]) if len(residuals) else None
        ),
    }
    return coefficients, prediction, report


def segment_conditioned_rank_path(
    features: np.ndarray,
    target_residual: np.ndarray,
    ranks: Sequence[int],
    rcond: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fit and truncate the closed-form per-segment dynamic coefficient map.

    The mean residual of each segment is the static identity term. The centered
    residual is fit from centered frozen features independently per segment.
    An SVD of the resulting segment-by-feature coefficient matrix yields a
    deterministic low-rank family with no iterative optimizer.
    """

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target_residual, dtype=np.float64)
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("expected features [sample, segment, feature] and residual")
    sample_count, segment_count, feature_count = x.shape
    static_bias = y.mean(axis=0)
    centered_target = y - static_bias[None, :]
    centered_features = x - x.mean(axis=0, keepdims=True)
    coefficient_matrix = np.zeros((segment_count, feature_count), dtype=np.float64)
    locally_unidentifiable = []
    for segment_id in range(segment_count):
        local_x = centered_features[:, segment_id, :]
        local_y = centered_target[:, segment_id]
        if float(np.linalg.norm(local_x)) <= float(rcond):
            if float(np.max(np.abs(local_y))) > float(rcond):
                locally_unidentifiable.append(int(segment_id))
            continue
        coefficient_matrix[segment_id] = np.linalg.lstsq(
            local_x, local_y, rcond=float(rcond)
        )[0]
    left, singular_values, right = np.linalg.svd(
        coefficient_matrix, full_matrices=False
    )
    maximum_rank = min(coefficient_matrix.shape)
    rows: List[Dict[str, Any]] = []
    for requested in ranks:
        rank = min(int(requested), maximum_rank)
        approximation = (
            left[:, :rank] * singular_values[None, :rank]
        ) @ right[:rank]
        predicted = static_bias[None, :] + np.einsum(
            "nsf,sf->ns", centered_features, approximation
        )
        static_error = predicted.mean(axis=0) - static_bias
        dynamic_error = (predicted - predicted.mean(axis=0)) - centered_target
        rows.append({
            "requested_rank": int(requested),
            "effective_rank": int(rank),
            "parameter_count": int(
                segment_count + rank * (segment_count + feature_count)
            ),
            "predicted_residual": predicted,
            "static_residual_rmse_mv": float(np.sqrt(np.mean(static_error ** 2))),
            "dynamic_residual_rmse_mv": float(np.sqrt(np.mean(dynamic_error ** 2))),
        })
    tolerance = float(rcond) * (float(singular_values[0]) if len(singular_values) else 0.0)
    diagnostics = {
        "sample_count": int(sample_count),
        "segment_count": int(segment_count),
        "feature_count": int(feature_count),
        "coefficient_matrix_rank": int(np.sum(singular_values > tolerance)),
        "coefficient_singular_values": [float(value) for value in singular_values],
        "locally_unidentifiable_segment_ids": locally_unidentifiable,
        "locally_unidentifiable_segment_count": len(locally_unidentifiable),
        "static_bias_parameter_count": int(segment_count),
        "full_segment_coefficient_parameter_count": int(
            segment_count * feature_count + segment_count
        ),
    }
    return rows, diagnostics


class HinesSegmentCapacityExperiment(HinesResidualConditioningExperiment):
    """Closed-form 05e diagnostic; it has no full-training entry point."""

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        model_config: Any,
        isolation_config: HinesIsolationConfig,
        conditioning_config: HinesConditioningConfig,
        capacity_config: HinesCapacityConfig,
        checkpoint_05b_source: Path,
        artifact_05c_source: Path,
        artifact_05d_source: Path,
        code_revision: Optional[str] = None,
    ) -> None:
        super().__init__(
            bundle, output_dir, model_config, isolation_config,
            conditioning_config, checkpoint_05b_source, artifact_05c_source,
            code_revision=code_revision,
        )
        capacity_config.validate()
        self.capacity = capacity_config
        self.artifact_05d_source = Path(artifact_05d_source).resolve()
        self.artifact_05d_contract: Dict[str, Any] = {}
        self.artifact_05d_report: Dict[str, Any] = {}
        self.metric_rows: List[Dict[str, Any]] = []

    def _read_05d_source(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05d_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05D_ARCHIVE_SHA256:
                raise RuntimeError("05d archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                members: Dict[str, bytes] = {}
                resolved: Dict[str, str] = {}
                for suffix in EXPECTED_05D_MEMBER_SHA256:
                    matches = [
                        name for name in archive.namelist()
                        if name.replace("\\", "/").endswith(suffix)
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one 05d member ending in {suffix!r}, found {matches}"
                        )
                    resolved[suffix] = matches[0]
                    members[suffix] = archive.read(matches[0])
            contract: Dict[str, Any] = {
                "source_kind": "original_zip",
                "source_path": str(source),
                "archive_sha256": archive_hash,
                "final_report_member": resolved["final_report.json"],
            }
        elif source.is_dir():
            members = {}
            resolved = {}
            for suffix in EXPECTED_05D_MEMBER_SHA256:
                matches = list(source.rglob(Path(suffix).name))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one extracted 05d member ending in {suffix!r}, "
                        f"found {[str(path) for path in matches]}"
                    )
                resolved[suffix] = matches[0].relative_to(source).as_posix()
                members[suffix] = matches[0].read_bytes()
            contract = {
                "source_kind": "kaggle_extracted_directory",
                "source_path": str(source),
                "archive_sha256": None,
                "final_report_member": resolved["final_report.json"],
            }
        else:
            raise RuntimeError(f"05d artifact source does not exist: {source}")
        observed = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        }
        mismatches = {
            name: {"expected": EXPECTED_05D_MEMBER_SHA256[name], "observed": value}
            for name, value in observed.items()
            if value != EXPECTED_05D_MEMBER_SHA256[name]
        }
        if mismatches:
            raise RuntimeError(f"05d member SHA-256 mismatch: {mismatches}")
        contract["verified_member_sha256"] = observed
        return json.loads(members["final_report.json"]), contract

    def prepare_capacity_probe(self) -> Dict[str, Any]:
        base = self.prepare_conditioning()
        report, contract = self._read_05d_source()
        blockers = []
        if report.get("diagnosis") != "SHARED_REPRESENTATION_BOTTLENECK":
            blockers.append(f"unexpected 05d diagnosis: {report.get('diagnosis')}")
        if report.get("full_training_authorized") is not False:
            blockers.append("05d unexpectedly authorizes full training")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05d and mounted composite fingerprints disagree")
        if int(report.get("worst_transition", -1)) != self.worst_transition:
            blockers.append("05d worst transition disagrees with 05c")
        branch_pair = report.get("checkpoint_05b", {}).get("branch_pair")
        if tuple(branch_pair or ()) != tuple(self.branch_pair or ()):
            blockers.append("05d branch pair disagrees with 05b/05c")
        if blockers:
            raise RuntimeError(f"05e provenance blockers: {blockers}")
        self.artifact_05d_report = report
        self.artifact_05d_contract = contract
        payload = {
            "schema_version": "05e-capacity-config-v1",
            "capacity": asdict(self.capacity),
            "artifact_05d": contract,
            "code_revision": self.code_revision,
            "worst_transition": self.worst_transition,
            "branch_pair": list(self.branch_pair or ()),
            "full_training_authorized": False,
        }
        _write_json(self.output_dir / "capacity_config.json", payload)
        return {**base, **payload}

    @staticmethod
    def _numpy_voltage_metrics(
        predicted: np.ndarray, target: np.ndarray
    ) -> Dict[str, float]:
        error = predicted - target
        result = {
            "voltage_rmse_mv": float(np.sqrt(np.mean(error ** 2))),
            "maximum_segment_error_mv": float(np.max(np.abs(error))),
            "maximum_peak_error_mv": float(np.max(np.abs(
                predicted.max(axis=1) - target.max(axis=1)
            ))),
        }
        if len(predicted) == 2:
            teacher_distance = float(np.sqrt(np.mean(
                (target[0] - target[1]) ** 2
            ) + 1e-12))
            predicted_distance = float(np.sqrt(np.mean(
                (predicted[0] - predicted[1]) ** 2
            ) + 1e-12))
            result.update(
                teacher_distance_mv=teacher_distance,
                predicted_distance_mv=predicted_distance,
                branching_retention=predicted_distance / max(teacher_distance, 1e-8),
            )
        return result

    def _passes(self, metrics: Mapping[str, float], pair: bool) -> bool:
        if not pair:
            return bool(
                metrics["voltage_rmse_mv"] < self.capacity.one_transition_rmse_mv
                and metrics["maximum_segment_error_mv"]
                < self.capacity.one_transition_max_error_mv
            )
        return bool(
            metrics["voltage_rmse_mv"] < self.capacity.pair_rmse_mv
            and metrics["maximum_segment_error_mv"] < self.capacity.pair_max_error_mv
            and self.capacity.pair_retention_minimum
            <= metrics.get("branching_retention", math.nan)
            <= self.capacity.pair_retention_maximum
        )

    @staticmethod
    def _residual_decomposition(
        predicted_residual: np.ndarray, target_residual: np.ndarray
    ) -> Dict[str, Optional[float]]:
        predicted_static = predicted_residual.mean(axis=0)
        target_static = target_residual.mean(axis=0)
        static_rmse = float(np.sqrt(np.mean(
            (predicted_static - target_static) ** 2
        )))
        if len(target_residual) < 2:
            return {
                "static_residual_rmse_mv": static_rmse,
                "branch_delta_residual_rmse_mv": None,
            }
        predicted_delta = predicted_residual[0] - predicted_residual[1]
        target_delta = target_residual[0] - target_residual[1]
        return {
            "static_residual_rmse_mv": static_rmse,
            "branch_delta_residual_rmse_mv": float(np.sqrt(np.mean(
                (predicted_delta - target_delta) ** 2
            ))),
        }

    def _record_probe(
        self,
        sample: str,
        probe: str,
        base: np.ndarray,
        target: np.ndarray,
        predicted_residual: np.ndarray,
        parameter_count: int,
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        predicted = base + predicted_residual
        metrics = self._numpy_voltage_metrics(predicted, target)
        row = {
            "sample": sample,
            "probe": probe,
            "rank": rank,
            "parameter_count": int(parameter_count),
            "passed": self._passes(metrics, pair=len(target) == 2),
            **self._residual_decomposition(predicted_residual, target - base),
            **metrics,
        }
        self.metric_rows.append(dict(row))
        return row

    def _run_sample(
        self,
        sample: str,
        indices: Sequence[int],
        progress: Progress,
        progress_offset: int,
    ) -> Dict[str, Any]:
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_t, features_t, target_t, compatibility = self._fixed_base_and_features(
            indices, device
        )
        progress.update(progress_offset + 1, f"{sample}: frozen features loaded")
        base = base_t.detach().cpu().double().numpy()
        features = features_t.detach().cpu().double().numpy()
        target = target_t.detach().cpu().double().numpy()
        sample_count, segment_count, feature_count = features.shape
        target_residual = target - base
        flat_features, feature_mean, raw_std = standardize_design(
            features.reshape(-1, feature_count), self.capacity.feature_epsilon
        )
        standardized = flat_features.reshape(features.shape)
        shared_design = np.column_stack((
            np.ones(sample_count * segment_count), flat_features
        ))
        _, shared_prediction, shared_spectrum = solve_linear_probe(
            shared_design, target_residual.reshape(-1), self.capacity.svd_rcond
        )
        shared_row = self._record_probe(
            sample, "shared_linear", base, target,
            shared_prediction.reshape(sample_count, segment_count),
            shared_design.shape[1],
        )
        progress.update(progress_offset + 2, f"{sample}: shared solve complete")

        segment_ids = np.tile(np.arange(segment_count), sample_count)
        one_hot = np.eye(segment_count, dtype=np.float64)[segment_ids]
        static_prediction = np.tile(
            target_residual.mean(axis=0, keepdims=True), (sample_count, 1)
        )
        static_row = self._record_probe(
            sample, "segment_bias_only", base, target, static_prediction,
            segment_count,
        )
        conditioned_design = np.column_stack((flat_features, one_hot))
        _, conditioned_prediction, conditioned_spectrum = solve_linear_probe(
            conditioned_design, target_residual.reshape(-1),
            self.capacity.svd_rcond,
        )
        bias_shared_row = self._record_probe(
            sample, "segment_bias_plus_shared_linear", base, target,
            conditioned_prediction.reshape(sample_count, segment_count),
            conditioned_design.shape[1],
        )
        progress.update(progress_offset + 3, f"{sample}: segment-bias solve complete")

        rank_rows = []
        rank_diagnostics = None
        if sample_count >= 2:
            raw_rank_rows, rank_diagnostics = segment_conditioned_rank_path(
                standardized, target_residual, self.capacity.rank_candidates,
                self.capacity.svd_rcond,
            )
            for raw in raw_rank_rows:
                prediction = raw.pop("predicted_residual")
                row = self._record_probe(
                    sample, "segment_conditioned_low_rank", base, target,
                    prediction, raw["parameter_count"], raw["effective_rank"],
                )
                row.update({
                    "requested_rank": raw["requested_rank"],
                    "effective_rank": raw["effective_rank"],
                    "dynamic_fit_rmse_mv": raw["dynamic_residual_rmse_mv"],
                })
                self.metric_rows[-1].update({
                    "requested_rank": raw["requested_rank"],
                    "effective_rank": raw["effective_rank"],
                    "dynamic_fit_rmse_mv": raw["dynamic_residual_rmse_mv"],
                })
                rank_rows.append(row)
        progress.update(progress_offset + 4, f"{sample}: rank path complete")
        progress.update(progress_offset + 5, f"{sample}: diagnostics complete")
        return {
            "sample": sample,
            "transition_indices": [int(value) for value in indices],
            "checkpoint_compatibility": compatibility,
            "shape": {
                "sample_count": sample_count,
                "segment_count": segment_count,
                "feature_count": feature_count,
            },
            "feature_standardization": {
                "minimum_raw_std": float(raw_std.min()),
                "maximum_raw_std": float(raw_std.max()),
                "constant_column_count": int(np.sum(
                    raw_std < self.capacity.feature_epsilon
                )),
                "mean_sha256": hashlib.sha256(feature_mean.tobytes()).hexdigest(),
            },
            "shared_linear": shared_row,
            "shared_design_spectrum": shared_spectrum,
            "segment_bias_only": static_row,
            "segment_bias_plus_shared_linear": bias_shared_row,
            "segment_bias_design_spectrum": conditioned_spectrum,
            "segment_conditioned_rank_path": rank_rows,
            "segment_conditioned_diagnostics": rank_diagnostics,
        }

    def run_capacity_probes(self) -> Dict[str, Any]:
        progress = Progress("closed-form capacity probes", 10)
        one = self._run_sample(
            "one_transition", [self.worst_transition], progress, 0
        )
        pair = self._run_sample(
            "branch_pair", list(self.branch_pair or ()), progress, 5
        )
        write_parquet(self.output_dir / "capacity_probe_metrics.parquet", self.metric_rows)
        passing_ranks = [
            int(row["effective_rank"])
            for row in pair["segment_conditioned_rank_path"] if row["passed"]
        ]
        report = {
            "schema_version": "05e-capacity-probes-v1",
            "valid": bool(
                one["shape"]["segment_count"] == pair["shape"]["segment_count"]
                and pair["shape"]["sample_count"] == 2
                and len(pair["segment_conditioned_rank_path"])
                == len(self.capacity.rank_candidates)
            ),
            "one_transition": one,
            "branch_pair": pair,
            "smallest_passing_segment_conditioned_rank": (
                min(passing_ranks) if passing_ranks else None
            ),
            "full_training_authorized": False,
        }
        _write_json(self.output_dir / "capacity_probe_report.json", report)
        self._plot_diagnostics(report)
        return report

    def _plot_diagnostics(self, report: Mapping[str, Any]) -> None:
        import matplotlib.pyplot as plt

        pair = report["branch_pair"]
        ranks = pair["segment_conditioned_rank_path"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        singular = np.asarray(
            pair["shared_design_spectrum"]["singular_values"], dtype=float
        )
        axes[0].semilogy(
            np.arange(1, len(singular) + 1), np.maximum(singular, 1e-15),
            marker=".",
        )
        axes[0].set(title="Shared design singular spectrum", xlabel="index", ylabel="singular value")
        axes[0].grid(alpha=0.3)
        axes[1].plot(
            [row["effective_rank"] for row in ranks],
            [max(row["voltage_rmse_mv"], 1e-12) for row in ranks],
            marker="o", label="RMSE voltage",
        )
        axes[1].axhline(self.capacity.pair_rmse_mv, color="black", linestyle="--", label="gate RMSE")
        axes[1].set(xscale="log", yscale="log", xlabel="segment-conditioned rank", ylabel="mV", title="Closed-form capacity path")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "capacity_diagnostics.png", dpi=160)
        plt.close(figure)

    def finalize_capacity(self, probe_report: Mapping[str, Any]) -> Dict[str, Any]:
        one = probe_report["one_transition"]
        pair = probe_report["branch_pair"]
        passing_rank = probe_report.get("smallest_passing_segment_conditioned_rank")
        if one["shared_linear"]["passed"] and pair["shared_linear"]["passed"]:
            diagnosis = "SHARED_LINEAR_CAPACITY_SUFFICIENT"
        elif pair["segment_bias_plus_shared_linear"]["passed"]:
            diagnosis = "STATIC_SEGMENT_IDENTITY_MISSING"
        elif passing_rank is not None:
            diagnosis = "SEGMENT_CONDITIONED_CAPACITY_SUFFICIENT"
        elif pair["segment_conditioned_diagnostics"]["locally_unidentifiable_segment_count"]:
            diagnosis = "FROZEN_FEATURES_BRANCH_UNIDENTIFIABLE"
        else:
            diagnosis = "SEGMENT_CONDITIONING_INSUFFICIENT"
        next_experiment = (
            "05f_zero_initialized_segment_conditioned_micro_canary"
            if diagnosis in {
                "SHARED_LINEAR_CAPACITY_SUFFICIENT",
                "STATIC_SEGMENT_IDENTITY_MISSING",
                "SEGMENT_CONDITIONED_CAPACITY_SUFFICIENT",
            }
            else "05f_feature_identifiability_revision"
        )
        report = {
            "schema_version": "05e-final-report-v1",
            "valid": bool(probe_report.get("valid")),
            "decision": "DIAGNOSTIC_ONLY_NO_FULL_TRAINING",
            "full_training_authorized": False,
            "diagnosis": diagnosis,
            "selected_segment_conditioned_rank": passing_rank,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "worst_transition": self.worst_transition,
            "worst_segment": self.worst_segment,
            "branch_pair": list(self.branch_pair or ()),
            "checkpoint_05b": self.checkpoint_contract,
            "artifact_05c": self.artifact_05c_contract,
            "artifact_05d": self.artifact_05d_contract,
            "capacity_probes": probe_report,
            "methodology": {
                "iterative_optimizer_used": False,
                "float64_closed_form": True,
                "static_segment_identity_separated_from_branch_discrimination": True,
                "same_absolute_gates_as_05d": True,
                "test_split_used_for_training": False,
            },
            "next_step": {
                "experiment": next_experiment,
                "full_training_authorized": False,
                "requires_fresh_neural_micro_canary": True,
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
            "schema_version": "05e-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
