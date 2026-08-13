"""Notebook-05j-b controlled decoder/feature revision after the 05j failure."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_repaired_representation_recheck import (
    HinesRepairedRepresentationRecheck,
)


EXPECTED_05J_ARCHIVE_SHA256 = (
    "78d77d601fb85ca078fdf807c66c19014b6884ddb2596a1bcc1dbb461520d508"
)
EXPECTED_05J_INDEX_SHA256 = (
    "7f7f6dcc5d8b4e20736782648dd6d29cff71bddd9a9664b3a7ca53ae147472cb"
)
EXPECTED_05J_FINAL_SHA256 = (
    "2bc021d5a14f1bdc04d4a196a02b58ba67659baf63188e00f44e3dbc2a414423"
)


@dataclass(frozen=True)
class HinesRepairedRepresentationRevisionConfig:
    input_families: Tuple[str, ...] = ("h2", "causal", "h2_causal")
    feature_transforms: Tuple[str, ...] = ("tanh", "asinh")
    ridge_lambdas: Tuple[float, ...] = (
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0,
    )
    expected_train_pair_count: int = 12
    expected_development_pair_count: int = 1
    target_residual_limit_mv: float = 120.0
    target_atanh_margin: float = 1e-6
    asinh_reference_z: float = 8.0
    maximum_regularized_condition_number: float = 1e8
    maximum_segment_coefficient_l2_norm: float = 1e5
    branch_fit_weight: float = 1.0
    lopo_max_error_weight: float = 0.05
    lopo_branch_log_weight: float = 2.0
    minimum_passing_candidate_count: int = 1

    def validate(self) -> None:
        if tuple(self.input_families) != ("h2", "causal", "h2_causal"):
            raise ValueError("05j-b must retain h2, causal, and h2_causal")
        if tuple(self.feature_transforms) != ("tanh", "asinh"):
            raise ValueError("05j-b must compare tanh and asinh transforms")
        if not self.ridge_lambdas or min(self.ridge_lambdas) <= 0:
            raise ValueError("ridge lambdas must be positive")
        if tuple(sorted(set(self.ridge_lambdas))) != self.ridge_lambdas:
            raise ValueError("ridge lambdas must be unique and increasing")
        positive = (
            self.expected_train_pair_count,
            self.expected_development_pair_count,
            self.target_residual_limit_mv,
            self.target_atanh_margin,
            self.asinh_reference_z,
            self.maximum_regularized_condition_number,
            self.maximum_segment_coefficient_l2_norm,
            self.branch_fit_weight,
            self.minimum_passing_candidate_count,
        )
        if min(positive) <= 0:
            raise ValueError("05j-b configuration values must be positive")
        if self.target_atanh_margin >= 0.1:
            raise ValueError("target atanh margin is unexpectedly large")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRepairedRepresentationRevisionConfig":
        payload = dict(values)
        for name in ("input_families", "feature_transforms", "ridge_lambdas"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def revised_feature_transform(
    standardized: np.ndarray,
    transform: str,
    *,
    tanh_scale: float,
    asinh_reference_z: float,
) -> np.ndarray:
    """Apply either the registered saturating map or an order-preserving tail map."""

    values = np.asarray(standardized, dtype=np.float64)
    if transform == "tanh":
        result = np.tanh(values / float(tanh_scale))
    elif transform == "asinh":
        result = np.arcsinh(values) / np.arcsinh(float(asinh_reference_z))
    else:
        raise ValueError(f"unknown 05j-b feature transform: {transform}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("05j-b feature transform produced NaN/Inf")
    return result


def bounded_target_encode(
    residual_mv: np.ndarray, limit_mv: float, margin: float
) -> np.ndarray:
    ratio = np.asarray(residual_mv, dtype=np.float64) / float(limit_mv)
    bound = 1.0 - float(margin)
    if np.max(np.abs(ratio)) >= bound:
        raise RuntimeError("target residual exceeds the preregistered bounded decoder domain")
    return np.arctanh(np.clip(ratio, -bound, bound))


def bounded_target_decode(encoded: np.ndarray, limit_mv: float) -> np.ndarray:
    return float(limit_mv) * np.tanh(np.asarray(encoded, dtype=np.float64))


def dual_ridge_path_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    lambdas: Sequence[float],
    *,
    pair_branch_weight: float = 0.0,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Batched per-segment dual ridge path with an unpenalized local intercept."""

    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    evaluation = np.asarray(evaluation_x, dtype=np.float64)
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("train arrays must be [sample, segment, feature]/[sample, segment]")
    if evaluation.ndim != 3 or evaluation.shape[1:] != x.shape[1:]:
        raise ValueError("evaluation feature shape disagrees with train features")
    if len(x) < 2:
        raise ValueError("ridge fit requires at least two samples")
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    point_x = x - x_mean
    point_y = y - y_mean
    if pair_branch_weight > 0:
        if len(x) % 2:
            raise ValueError("branch-augmented ridge expects complete adjacent pairs")
        root_weight = math.sqrt(float(pair_branch_weight))
        paired_x = x.reshape(-1, 2, *x.shape[1:])
        paired_y = y.reshape(-1, 2, y.shape[1])
        branch_x = root_weight * (paired_x[:, 0] - paired_x[:, 1])
        branch_y = root_weight * (paired_y[:, 0] - paired_y[:, 1])
        point_x = np.concatenate([point_x, branch_x], axis=0)
        point_y = np.concatenate([point_y, branch_y], axis=0)
    xc = np.transpose(point_x, (1, 0, 2))
    yc = np.transpose(point_y, (1, 0))
    kernel = np.einsum("snf,smf->snm", xc, xc, optimize=True)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_target = np.einsum(
        "snm,sm->sn", np.transpose(eigenvectors, (0, 2, 1)), yc,
        optimize=True,
    )
    evaluation_centered = np.transpose(evaluation - x_mean, (1, 0, 2))
    cross_kernel = np.einsum(
        "sef,snf->sen", evaluation_centered, xc, optimize=True
    )
    predictions = []
    diagnostics: List[Dict[str, float]] = []
    for ridge in lambdas:
        denominator = eigenvalues + float(ridge)
        alpha = np.einsum(
            "snm,sm->sn",
            eigenvectors,
            projected_target / denominator,
            optimize=True,
        )
        prediction = np.einsum(
            "sen,sn->se", cross_kernel, alpha, optimize=True
        ).T + y_mean
        coefficient = np.einsum("snf,sn->sf", xc, alpha, optimize=True)
        condition = (eigenvalues[:, -1] + float(ridge)) / (
            eigenvalues[:, 0] + float(ridge)
        )
        predictions.append(prediction)
        diagnostics.append({
            "ridge_lambda": float(ridge),
            "maximum_regularized_condition_number": float(np.max(condition)),
            "median_regularized_condition_number": float(np.median(condition)),
            "maximum_segment_coefficient_l2_norm": float(
                np.max(np.linalg.norm(coefficient, axis=1))
            ),
            "median_segment_coefficient_l2_norm": float(
                np.median(np.linalg.norm(coefficient, axis=1))
            ),
            "pair_branch_weight": float(pair_branch_weight),
            "fit_row_count": int(xc.shape[1]),
        })
    result = np.stack(predictions, axis=0)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("dual ridge path produced NaN/Inf")
    return result, diagnostics


def pair_gate_selection_score(
    pair_metrics: Mapping[str, Any],
    *,
    max_error_weight: float,
    branch_log_weight: float,
) -> float:
    retention = np.asarray([
        max(float(row["branching_retention"]), 1e-12)
        for row in pair_metrics["pair_metrics"]
    ])
    return float(
        pair_metrics["aggregate_voltage_rmse_mv"]
        + float(max_error_weight) * pair_metrics["maximum_segment_error_mv"]
        + float(branch_log_weight) * np.median(np.abs(np.log(retention)))
    )


class HinesRepairedRepresentationRevision(HinesRepairedRepresentationRecheck):
    """05j-b tail-preserving features and segment-specific regularized decoders."""

    def __init__(
        self,
        *args: Any,
        revision_config: HinesRepairedRepresentationRevisionConfig,
        artifact_05j_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        revision_config.validate()
        self.revision = revision_config
        self.artifact_05j_source = Path(artifact_05j_source).resolve()
        self.artifact_05j_contract: Dict[str, Any] = {}

    def _read_verified_05j(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05j_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05J_ARCHIVE_SHA256:
                raise RuntimeError("05j archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[: -len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05J_INDEX_SHA256:
                    raise RuntimeError("05j artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                for row in index["artifacts"]:
                    member = root + str(row["path"]).replace("\\", "/")
                    if member not in names:
                        raise RuntimeError(f"missing indexed 05j member: {row['path']}")
                    payload = archive.read(member)
                    if (
                        hashlib.sha256(payload).hexdigest() != row["sha256"]
                        or len(payload) != int(row["size_bytes"])
                    ):
                        raise RuntimeError(f"05j indexed member mismatch: {row['path']}")
                final_bytes = archive.read(root + "final_report.json")
            kind = "original_zip"
        elif source.is_dir():
            indices = list(source.rglob("artifact_index.json"))
            indices = [
                path for path in indices
                if (path.parent / "repaired_representation_recheck_config.json").is_file()
            ]
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05j artifact index")
            index_bytes = indices[0].read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05J_INDEX_SHA256:
                raise RuntimeError("extracted 05j artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            root = indices[0].parent
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if (
                    not path.is_file()
                    or sha256_file(path) != row["sha256"]
                    or path.stat().st_size != int(row["size_bytes"])
                ):
                    raise RuntimeError(f"extracted 05j member mismatch: {row['path']}")
            final_bytes = (root / "final_report.json").read_bytes()
            archive_hash = None
            kind = "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05j source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05J_FINAL_SHA256:
            raise RuntimeError("05j final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05J_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05J_FINAL_SHA256,
            "verified_member_count": len(index["artifacts"]),
            "all_indexed_members_verified": (
                len(index["artifacts"]) == int(index["artifact_count"])
            ),
        }

    def prepare_repaired_representation_revision(self) -> Dict[str, Any]:
        base = self.prepare_repaired_representation_recheck()
        report, contract = self._read_verified_05j()
        blockers = []
        if report.get("diagnosis") != "REPAIRED_REPRESENTATION_CONTROLS_FAIL_ROBUST_GATE":
            blockers.append(f"unexpected 05j diagnosis: {report.get('diagnosis')}")
        if report.get("representation_recheck_passed") is not False:
            blockers.append("05j did not record the required failed recheck")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05j dataset fingerprint mismatch")
        heldout = report.get("heldout_contract", {})
        if any(bool(value) for key, value in heldout.items() if key != "reveal_authorized"):
            blockers.append("05j held-out contract was not sealed")
        if heldout.get("reveal_authorized"):
            blockers.append("05j unexpectedly authorized held-out reveal")
        if report.get("methodology", {}).get("rollout_performed"):
            blockers.append("05j unexpectedly performed rollout")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05j artifact verification is incomplete")
        if (
            self.revision.expected_train_pair_count
            != self.recheck.required_train_pair_count
            or self.revision.expected_development_pair_count
            != self.recheck.required_development_pair_count
        ):
            blockers.append("05j-b pair counts differ from the verified 05j contract")
        if not math.isclose(
            self.revision.target_residual_limit_mv,
            self.forensics.residual_limit_mv,
        ):
            blockers.append("05j-b changed the registered residual limit")
        if not math.isclose(
            self.revision.branch_fit_weight,
            self.forensics.branch_loss_weight,
        ):
            blockers.append("05j-b changed the registered branch-loss weight")
        if blockers:
            raise RuntimeError(f"05j-b provenance blockers: {blockers}")
        self.artifact_05j_contract = contract
        payload = {
            "schema_version": "05j-b-revision-config-v1",
            "revision": asdict(self.revision),
            "artifact_05j": contract,
            "diagnostic_axes": [
                "feature_tail_transform",
                "segment_specific_decoder_capacity",
                "train_only_regularization_selection",
            ],
            "ridge_selection_roles": ["train_leave_one_pair_out"],
            "candidate_evaluation_roles": ["train", "development"],
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "repaired_representation_revision_config.json", payload)
        return {**base, **payload}

    def _revision_family_features(
        self, role: Mapping[str, Any], family: str, transform: str
    ) -> np.ndarray:
        pieces = [
            np.asarray(role["voltage_t"], dtype=np.float64)[..., None] / 100.0,
            np.asarray(role["base"], dtype=np.float64)[..., None] / 100.0,
        ]
        if family in {"h2", "h2_causal"}:
            pieces.append(revised_feature_transform(
                role["h2_raw_z"], transform,
                tanh_scale=self.forensics.bounded_feature_scale,
                asinh_reference_z=self.revision.asinh_reference_z,
            ))
        if family in {"causal", "h2_causal"}:
            pieces.append(revised_feature_transform(
                role["causal_raw_z"], transform,
                tanh_scale=self.forensics.bounded_feature_scale,
                asinh_reference_z=self.revision.asinh_reference_z,
            ))
        result = np.concatenate(pieces, axis=-1)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("05j-b family feature surface contains NaN/Inf")
        return result

    def run_transform_geometry_audit(self) -> Dict[str, Any]:
        if not self.roles:
            raise RuntimeError("prepare_train_development_features() must run first")
        rows = []
        for family in self.revision.input_families:
            for transform in self.revision.feature_transforms:
                train = self._revision_family_features(self.roles["train"], family, transform)
                development = self._revision_family_features(
                    self.roles["development"], family, transform
                )
                pair_delta = train.reshape(-1, 2, *train.shape[1:])
                pair_delta = pair_delta[:, 0] - pair_delta[:, 1]
                flattened = np.transpose(train, (1, 0, 2))
                ranks = np.asarray([
                    np.linalg.matrix_rank(values, tol=1e-10) for values in flattened
                ])
                rows.append({
                    "family": family,
                    "transform": transform,
                    "feature_width": int(train.shape[-1]),
                    "maximum_absolute_train_feature": float(np.max(np.abs(train))),
                    "maximum_absolute_development_feature": float(
                        np.max(np.abs(development))
                    ),
                    "median_pair_delta_l2": float(np.median(
                        np.linalg.norm(pair_delta, axis=-1)
                    )),
                    "minimum_design_rank": int(ranks.min()),
                    "median_design_rank": float(np.median(ranks)),
                    "maximum_design_rank": int(ranks.max()),
                    "nonfinite_count": int(
                        np.sum(~np.isfinite(train)) + np.sum(~np.isfinite(development))
                    ),
                })
        report = {
            "schema_version": "05j-b-transform-geometry-v1",
            "valid": all(row["nonfinite_count"] == 0 for row in rows),
            "rows": rows,
            "normalization_fit_roles": ["train"],
            "development_values_used_to_fit": False,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
        }
        _write_json(self.output_dir / "transform_geometry_audit.json", report)
        write_parquet(self.output_dir / "transform_geometry.parquet", rows)
        return report

    def _pair_metrics_for_prediction(
        self, role: Mapping[str, Any], predicted_residual: np.ndarray
    ) -> Dict[str, Any]:
        prediction = np.asarray(role["base"]) + np.asarray(predicted_residual)
        return self._pair_set_metrics(
            prediction.reshape(-1, 2, self.layout.segment_count),
            np.asarray(role["target"]).reshape(-1, 2, self.layout.segment_count),
        )

    def _ridge_candidate(
        self,
        family: str,
        transform: str,
        progress: Progress,
        completed: int,
    ) -> Tuple[Dict[str, Any], int]:
        train = self.roles["train"]
        development = self.roles["development"]
        train_x = self._revision_family_features(train, family, transform)
        development_x = self._revision_family_features(
            development, family, transform
        )
        target_train = np.asarray(train["target"]) - np.asarray(train["base"])
        target_encoded = bounded_target_encode(
            target_train,
            self.revision.target_residual_limit_mv,
            self.revision.target_atanh_margin,
        )
        lambdas = self.revision.ridge_lambdas
        oof_encoded = np.zeros(
            (len(lambdas), *target_encoded.shape), dtype=np.float64
        )
        pair_count = len(train_x) // 2
        for pair in range(pair_count):
            held = np.asarray([2 * pair, 2 * pair + 1])
            keep = np.ones(len(train_x), dtype=bool)
            keep[held] = False
            path, _ = dual_ridge_path_predict(
                train_x[keep], target_encoded[keep], train_x[held], lambdas,
                pair_branch_weight=self.revision.branch_fit_weight,
            )
            oof_encoded[:, held] = path
            completed += 1
            progress.update(completed, f"{family}/{transform} LOPO {pair + 1}/{pair_count}")
        ladder = []
        for index, ridge in enumerate(lambdas):
            residual = bounded_target_decode(
                oof_encoded[index], self.revision.target_residual_limit_mv
            )
            metrics = self._pair_metrics_for_prediction(train, residual)
            ladder.append({
                "family": family,
                "transform": transform,
                "ridge_lambda": float(ridge),
                "selection_score": pair_gate_selection_score(
                    metrics,
                    max_error_weight=self.revision.lopo_max_error_weight,
                    branch_log_weight=self.revision.lopo_branch_log_weight,
                ),
                "lopo": metrics,
            })
        selected = min(ladder, key=lambda row: row["selection_score"])
        selected_lambda = float(selected["ridge_lambda"])
        full_path, diagnostics = dual_ridge_path_predict(
            train_x,
            target_encoded,
            np.concatenate([train_x, development_x], axis=0),
            [selected_lambda],
            pair_branch_weight=self.revision.branch_fit_weight,
        )
        predicted = bounded_target_decode(
            full_path[0], self.revision.target_residual_limit_mv
        )
        train_prediction = predicted[: len(train_x)]
        development_prediction = predicted[len(train_x):]
        train_metrics = self._pair_metrics_for_prediction(train, train_prediction)
        development_metrics = self._pair_metrics_for_prediction(
            development, development_prediction
        )
        stability = diagnostics[0]
        stable = bool(
            stability["maximum_regularized_condition_number"]
            <= self.revision.maximum_regularized_condition_number
            and stability["maximum_segment_coefficient_l2_norm"]
            <= self.revision.maximum_segment_coefficient_l2_norm
        )
        train_passed = self._pair_passes(train_metrics)
        development_passed = self._pair_passes(development_metrics)
        candidate = {
            "family": family,
            "transform": transform,
            "feature_width": int(train_x.shape[-1]),
            "selected_ridge_lambda": selected_lambda,
            "selection_roles": ["train_leave_one_pair_out"],
            "branch_fit_weight": self.revision.branch_fit_weight,
            "selection_score": float(selected["selection_score"]),
            "selection_ladder": ladder,
            "lopo": selected["lopo"],
            "train_fit": train_metrics,
            "development": development_metrics,
            "stability": stability,
            "numerically_stable": stable,
            "train_passed": train_passed,
            "development_passed": development_passed,
            "candidate_passed": bool(stable and train_passed and development_passed),
            "development_used_for_selection": False,
            "heldout_candidate_inference_performed": False,
        }
        return candidate, completed

    def run_segmentwise_regularized_revision(self) -> Dict[str, Any]:
        if not self.roles:
            raise RuntimeError("prepare_train_development_features() must run first")
        candidate_count = (
            len(self.revision.input_families) * len(self.revision.feature_transforms)
        )
        pair_count = len(self.roles["train"]["base"]) // 2
        progress = Progress("05j-b segmentwise ridge", candidate_count * pair_count)
        completed = 0
        candidates = []
        for family in self.revision.input_families:
            for transform in self.revision.feature_transforms:
                candidate, completed = self._ridge_candidate(
                    family, transform, progress, completed
                )
                candidates.append(candidate)
        report = {
            "schema_version": "05j-b-segmentwise-regularized-revision-v1",
            "valid": len(candidates) == candidate_count,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "passing_candidates": [
                f"{row['family']}:{row['transform']}"
                for row in candidates if row["candidate_passed"]
            ],
            "selection_roles": ["train_leave_one_pair_out"],
            "development_used_for_selection": False,
            "base_h2_frozen": True,
            "teacher_encoder_updated": False,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "segmentwise_regularized_revision.json", report)
        summary = [{
            "family": row["family"],
            "transform": row["transform"],
            "ridge_lambda": row["selected_ridge_lambda"],
            "lopo_rmse_mv": row["lopo"]["aggregate_voltage_rmse_mv"],
            "train_rmse_mv": row["train_fit"]["aggregate_voltage_rmse_mv"],
            "development_rmse_mv": row["development"]["aggregate_voltage_rmse_mv"],
            "development_retention": row["development"]["median_branching_retention"],
            "maximum_condition_number": row["stability"]["maximum_regularized_condition_number"],
            "maximum_coefficient_norm": row["stability"]["maximum_segment_coefficient_l2_norm"],
            "candidate_passed": row["candidate_passed"],
        } for row in candidates]
        write_parquet(self.output_dir / "segmentwise_regularized_summary.parquet", summary)
        return report

    def finalize_repaired_representation_revision(
        self,
        feature_report: Mapping[str, Any],
        geometry_report: Mapping[str, Any],
        revision_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        candidates = list(revision_report["candidates"])
        passing = [row for row in candidates if row["candidate_passed"]]
        tanh_passing = [row for row in passing if row["transform"] == "tanh"]
        asinh_passing = [row for row in passing if row["transform"] == "asinh"]
        train_fit_passing = [row for row in candidates if row["train_passed"]]
        development_passing = [row for row in candidates if row["development_passed"]]
        passed = bool(
            len(passing) >= self.revision.minimum_passing_candidate_count
            and feature_report["valid"]
            and geometry_report["valid"]
        )
        if passed and asinh_passing and not tanh_passing:
            diagnosis = "SATURATING_TANH_FEATURE_MAP_WAS_PRIMARY_BLOCKER"
        elif passed:
            diagnosis = "SHARED_COMPACT_HEAD_WAS_PRIMARY_BLOCKER"
        elif train_fit_passing and not development_passing:
            diagnosis = "TRAIN_SURFACE_FITS_BUT_FAILS_DEVELOPMENT_STABILITY"
        elif train_fit_passing:
            diagnosis = "REGULARIZED_DECODER_FAILS_JOINT_PAIR_GATE"
        else:
            diagnosis = "REPAIRED_FEATURE_SURFACE_FAILS_REGULARIZED_TRAIN_FIT"
        report = {
            "schema_version": "05j-b-final-report-v1",
            "valid": True,
            "decision": "SEGMENT_DECODER_AND_FEATURE_TRANSFORM_REVISION_ONLY",
            "diagnosis": diagnosis,
            "representation_revision_passed": passed,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05j": self.artifact_05j_contract,
            "train_development_features": dict(feature_report),
            "transform_geometry": dict(geometry_report),
            "segmentwise_regularized_revision": dict(revision_report),
            "passing_candidates": [
                {k: row[k] for k in (
                    "family", "transform", "selected_ridge_lambda",
                    "candidate_passed",
                )}
                for row in passing
            ],
            "heldout_contract": {
                "inputs_extracted": False,
                "boundary_targets_materialized": False,
                "event_targets_materialized": False,
                "candidate_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "exact_05j_failure_verified": True,
                "exact_registered_train_development_support_reused": True,
                "normalization_fit_split": "train",
                "ridge_selection": "train_leave_one_pair_out",
                "development_used_for_model_selection": False,
                "base_h2_frozen": True,
                "teacher_encoder_updated": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": (
                    "05k_repaired_representation_micro_rollout"
                    if passed else "05j_c_support_and_decoder_revision"
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
            "schema_version": "05j-b-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
