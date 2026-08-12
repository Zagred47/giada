"""Notebook-05h representation and raw-scale forensics for HayFlow-Hines."""

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
from .hines_capacity_experiment import HinesCapacityConfig
from .hines_conditioning_experiment import HinesConditioningConfig
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import HinesIsolationConfig, sha256_file
from .hines_optimization_audit import (
    HinesOptimizationAuditConfig,
    HinesSegmentOptimizationAudit,
)
from .hines_segment_canary_experiment import HinesSegmentCanaryConfig
from .hines_layer import require_torch
from .hayflow_hines import HINES_SYNAPTIC_FEATURE_NAMES

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


EXPECTED_05G_ARCHIVE_SHA256 = (
    "f369723f0d7184ea672e90fd4f530c8ad301eed088788067dae6a5d0524be66d"
)
EXPECTED_05G_MEMBER_SHA256 = {
    "artifact_index.json": "01e8f723e11133b50657f3039e4f4a8a170cfbad7a80ee3e8aa2f9ab67d61db1",
    "optimization_audit_config.json": "adb7268215a6153eca462ec50f5bac93aef96736bf2eda780dc9535a00a6fb3b",
    "optimization_support.json": "885263c3fbc1d39909be4e54c3a04e5c0e28cdc29e1d006932563322fac0f2eb",
    "feature_scale_audit.json": "cd37e60c67eb0ded52b8e4f38dca5ddefaa72df40ef222c6f257ea7786b1c188",
    "oracle_controls.json": "171a8e27b7af6071cfe041c617c2b0ad46415abae343e56a8fd07aeceaf30ec1",
    "regularized_train_audit.json": "fd04c42469296a03c78294ba135274f33ca6d1f4d888e9d7f347a5673c158225",
    "heldout_gate_report.json": "9705b5d4be6177c7e7c8c7f0102f231fca27475475d25f4f01ca3d41d78af723",
    "final_report.json": "597d20ede9c45445e5818bb92c36a1a6e531c96f911470e801c9dec699710018",
}


@dataclass(frozen=True)
class HinesRepresentationForensicsConfig:
    standardized_clip: float = 8.0
    bounded_feature_scale: float = 4.0
    feature_epsilon: float = 1e-6
    raw_heldout_norm_ratio_limit: float = 100.0
    standardized_heldout_max_limit: float = 100.0
    maximum_allowed_clipping_fraction: float = 0.001
    projection_rcond: float = 1e-10
    hidden_width: int = 64
    segment_embedding_dim: int = 8
    epochs: int = 800
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    evaluation_interval: int = 20
    patience: int = 160
    seeds: Tuple[int, ...] = (17, 29, 43)
    input_families: Tuple[str, ...] = ("h2", "causal", "h2_causal")
    residual_limit_mv: float = 120.0
    branch_loss_weight: float = 1.0
    pair_rmse_mv: float = 1.0
    pair_max_error_mv: float = 5.0
    pair_retention_minimum: float = 0.90
    pair_retention_maximum: float = 1.10

    def validate(self) -> None:
        positive = (
            self.standardized_clip, self.bounded_feature_scale,
            self.feature_epsilon, self.raw_heldout_norm_ratio_limit,
            self.standardized_heldout_max_limit, self.hidden_width,
            self.segment_embedding_dim, self.epochs, self.learning_rate,
            self.gradient_clip_norm, self.evaluation_interval, self.patience,
            self.residual_limit_mv,
        )
        if min(positive) <= 0:
            raise ValueError("05h configuration values must be positive")
        if not 0 <= self.maximum_allowed_clipping_fraction < 1:
            raise ValueError("invalid clipping fraction")
        if tuple(self.input_families) != ("h2", "causal", "h2_causal"):
            raise ValueError("05h must compare h2, causal, and h2_causal")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("05h seeds must be unique")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesRepresentationForensicsConfig":
        payload = dict(values)
        for name in ("seeds", "input_families"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def robust_bounded_features(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    bounded_feature_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    standardized = (np.asarray(values, dtype=np.float64) - mean) / scale
    bounded = np.tanh(standardized / float(bounded_feature_scale))
    return standardized, bounded


def local_linear_projection(
    features: np.ndarray, target: np.ndarray, rcond: float
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Unregularized train-only projection oracle, independently per segment."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("projection expects [sample, segment, feature] inputs")
    prediction = np.zeros_like(y)
    rows: List[Dict[str, Any]] = []
    for segment in range(x.shape[1]):
        design = np.concatenate(
            [np.ones((x.shape[0], 1), dtype=np.float64), x[:, segment]], axis=1
        )
        coefficient, _, rank, singular = np.linalg.lstsq(
            design, y[:, segment], rcond=float(rcond)
        )
        fitted = design @ coefficient
        prediction[:, segment] = fitted
        error = fitted - y[:, segment]
        nonzero = singular[singular > float(rcond) * singular[0]] if len(singular) else []
        rows.append({
            "segment_id": segment,
            "design_rank": int(rank),
            "design_condition_number": (
                float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
            ),
            "projection_rmse_mv": float(np.sqrt(np.mean(error ** 2))),
            "projection_max_error_mv": float(np.max(np.abs(error))),
            "target_residual_std_mv": float(np.std(y[:, segment])),
            "coefficient_l2_norm": float(np.linalg.norm(coefficient)),
        })
    return prediction, rows


if nn is not None:

    class BoundedLocalResidualHead(nn.Module):
        """Small shared nonlinear control; H2 remains entirely frozen."""

        def __init__(
            self, input_width: int, segment_count: int, hidden_width: int,
            segment_embedding_dim: int, residual_limit_mv: float,
        ) -> None:
            super().__init__()
            self.segment_embedding = nn.Embedding(segment_count, segment_embedding_dim)
            self.network = nn.Sequential(
                nn.Linear(input_width + segment_embedding_dim, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, 1),
            )
            self.residual_limit_mv = float(residual_limit_mv)
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            batch, segments, _ = features.shape
            ids = torch.arange(segments, device=features.device)
            embedded = self.segment_embedding(ids).unsqueeze(0).expand(batch, -1, -1)
            raw = self.network(torch.cat([features, embedded], dim=-1)).squeeze(-1)
            return self.residual_limit_mv * torch.tanh(raw)

else:

    class BoundedLocalResidualHead:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            require_torch()


class HinesRepresentationForensics(HinesSegmentOptimizationAudit):
    """05h raw-scale, projection, and bounded nonlinear train/dev audit."""

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        model_config: Any,
        isolation_config: HinesIsolationConfig,
        conditioning_config: HinesConditioningConfig,
        capacity_config: HinesCapacityConfig,
        canary_config: HinesSegmentCanaryConfig,
        audit_config: HinesOptimizationAuditConfig,
        forensics_config: HinesRepresentationForensicsConfig,
        checkpoint_05b_source: Path,
        artifact_05c_source: Path,
        artifact_05d_source: Path,
        artifact_05e_source: Path,
        artifact_05f_source: Path,
        artifact_05g_source: Path,
        code_revision: Optional[str] = None,
    ) -> None:
        super().__init__(
            bundle, output_dir, model_config, isolation_config,
            conditioning_config, capacity_config, canary_config, audit_config,
            checkpoint_05b_source, artifact_05c_source, artifact_05d_source,
            artifact_05e_source, artifact_05f_source, code_revision=code_revision,
        )
        forensics_config.validate()
        self.forensics = forensics_config
        self.artifact_05g_source = Path(artifact_05g_source).resolve()
        self.artifact_05g_report: Dict[str, Any] = {}
        self.artifact_05g_contract: Dict[str, Any] = {}
        self.roles: Dict[str, Any] = {}
        self.feature_widths: Dict[str, int] = {}

    def _read_05g_source(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05g_source
        members: Dict[str, bytes] = {}
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05G_ARCHIVE_SHA256:
                raise RuntimeError("05g archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                for suffix in EXPECTED_05G_MEMBER_SHA256:
                    matches = [n for n in archive.namelist() if n.replace("\\", "/").endswith(suffix)]
                    if len(matches) != 1:
                        raise RuntimeError(f"expected one 05g {suffix}, found {matches}")
                    members[suffix] = archive.read(matches[0])
            kind = "original_zip"
        elif source.is_dir():
            archive_hash = None
            for suffix in EXPECTED_05G_MEMBER_SHA256:
                matches = list(source.rglob(Path(suffix).name))
                if len(matches) != 1:
                    raise RuntimeError(f"expected one extracted 05g {suffix}")
                members[suffix] = matches[0].read_bytes()
            kind = "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05g source does not exist: {source}")
        observed = {name: hashlib.sha256(value).hexdigest() for name, value in members.items()}
        mismatch = {
            name: {"expected": EXPECTED_05G_MEMBER_SHA256[name], "observed": value}
            for name, value in observed.items()
            if value != EXPECTED_05G_MEMBER_SHA256[name]
        }
        if mismatch:
            raise RuntimeError(f"05g member SHA-256 mismatch: {mismatch}")
        return (
            json.loads(members["final_report.json"]),
            json.loads(members["optimization_support.json"]),
            {
                "source_kind": kind, "source_path": str(source),
                "archive_sha256": archive_hash, "verified_member_sha256": observed,
            },
        )

    def prepare_forensics(self) -> Dict[str, Any]:
        base = self.prepare_optimization_audit()
        report, support, contract = self._read_05g_source()
        blockers = []
        if report.get("diagnosis") != "REGULARIZED_FROZEN_FEATURES_CANNOT_FIT_TRAIN_SUPPORT":
            blockers.append(f"unexpected 05g diagnosis: {report.get('diagnosis')}")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05g dataset fingerprint mismatch")
        if not support.get("valid") or support.get("selected_pair_count") != 12:
            blockers.append("05g optimization support is not the registered 12-pair plan")
        if blockers:
            raise RuntimeError(f"05h provenance blockers: {blockers}")
        self.artifact_05g_report = report
        self.artifact_05g_contract = contract
        self.audit_plan = support
        payload = {
            "schema_version": "05h-representation-forensics-config-v1",
            "forensics": asdict(self.forensics),
            "artifact_05g": contract,
            "support_sha256": support["support_sha256"],
            "heldout_target_contract": "never materialized in 05h",
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "representation_forensics_config.json", payload)
        return {**base, **payload}

    def _extract_with_model(
        self, model: Any, indices: Sequence[int], include_targets: bool
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device
        raw = self._batch(
            indices,
            include_targets=include_targets,
            include_event_targets=include_targets,
        )
        batch = self._torch_batch(raw, device)
        zero = dict(batch)
        for name in (
            "synaptic_features", "synaptic_conductance_us",
            "synaptic_source_na", "somatic_current_na",
        ):
            zero[name] = torch.zeros_like(batch[name])
        with torch.no_grad():
            output = model(batch, ablation="H2", decode_teacher=False, boundary_mode="no_event_jump")
            zero_output = model(zero, ablation="H2", decode_teacher=False, boundary_mode="no_event_jump")
        causal = torch.cat([
            batch["synaptic_features"],
            batch["synaptic_conductance_us"].unsqueeze(-1),
            batch["synaptic_source_na"].unsqueeze(-1),
            batch["somatic_current_na"].unsqueeze(-1),
        ], dim=-1)
        result = {
            "indices": np.asarray(indices, dtype=np.int64),
            "base": output["voltage"].detach().cpu().double().numpy(),
            "h2_raw": output["boundary_features"].detach().cpu().double().numpy(),
            "h2_zero_causal_raw": zero_output["boundary_features"].detach().cpu().double().numpy(),
            "zero_causal_base": zero_output["voltage"].detach().cpu().double().numpy(),
            "causal_raw": causal.detach().cpu().double().numpy(),
            "voltage_t": np.asarray(raw["voltage_t"], dtype=np.float64),
            "teacher_state_normalized": np.asarray(
                raw["teacher_state_t"], dtype=np.float64
            ),
            "target": (
                np.asarray(raw["voltage_target"], dtype=np.float64)
                if include_targets else None
            ),
            "hines_diagnostics": output["hines_diagnostics"],
            "zero_causal_hines_diagnostics": zero_output["hines_diagnostics"],
        }
        return result

    @staticmethod
    def _surface_report(values: np.ndarray) -> Dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        norms = np.linalg.norm(values, axis=-1)
        return {
            "minimum": float(values.min()), "maximum": float(values.max()),
            "maximum_absolute": float(np.max(np.abs(values))),
            "median_segment_norm": float(np.median(norms)),
            "p95_segment_norm": float(np.quantile(norms, 0.95)),
            "maximum_segment_norm": float(norms.max()),
            "nonfinite_count": int(values.size - np.isfinite(values).sum()),
        }

    def run_raw_scale_forensics(self) -> Dict[str, Any]:
        train_indices = self._pair_indices(self.audit_plan["selected_pairs"])
        development_indices = list(self.audit_plan["development_pair"])
        heldout_indices = self._pair_indices(self.artifact_05f_report["pair_plan"]["heldout_pairs"])
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = self._load_h2_checkpoint(device)
        model.eval()
        print("[HayFlow 05h][features] train, development, held-out inputs...", flush=True)
        self.roles = {
            "train": self._extract_with_model(model, train_indices, True),
            "development": self._extract_with_model(model, development_indices, True),
            "heldout": self._extract_with_model(model, heldout_indices, False),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.roles["heldout"]["target"] is not None:
            raise RuntimeError("05h held-out target contract violated")

        surfaces = ("h2_raw", "h2_zero_causal_raw", "causal_raw")
        normalizers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        report: Dict[str, Any] = {"roles": {}, "top_outliers": []}
        group_rows: List[Dict[str, Any]] = []
        for surface in surfaces:
            train = self.roles["train"][surface]
            mean = train.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(train.std(axis=(0, 1), keepdims=True), self.forensics.feature_epsilon)
            normalizers[surface] = (mean, scale)
            for role, values in self.roles.items():
                z, bounded = robust_bounded_features(
                    values[surface], mean, scale, self.forensics.bounded_feature_scale
                )
                values[f"{surface}_z"] = z
                values[f"{surface}_bounded"] = bounded
                role_report = report["roles"].setdefault(role, {})
                role_report[surface] = {
                    "raw": self._surface_report(values[surface]),
                    "standardized_unclipped": self._surface_report(z),
                    "bounded_tanh": self._surface_report(bounded),
                    "clipping_fraction_at_registered_8": float(
                        np.mean(np.abs(z) > self.forensics.standardized_clip)
                    ),
                }
                if role != "train":
                    train_norm = max(
                        self._surface_report(train)["maximum_segment_norm"], 1e-12
                    )
                    role_report[surface]["raw_max_norm_to_train_ratio"] = float(
                        role_report[surface]["raw"]["maximum_segment_norm"] / train_norm
                    )
                feature_names = (
                    list(HINES_SYNAPTIC_FEATURE_NAMES)
                    + ["synaptic_conductance_us", "synaptic_source_na", "somatic_current_na"]
                    if surface == "causal_raw"
                    else [f"h2_hidden_{index}" for index in range(z.shape[2])]
                )
                for pair_position in range(z.shape[0] // 2):
                    selection = slice(2 * pair_position, 2 * pair_position + 2)
                    logical = int(values["indices"][2 * pair_position])
                    group_rows.append({
                        "role": role, "surface": surface, "group_kind": "pair",
                        "group_id": str(pair_position),
                        "protocol_family": self._protocol_family(logical),
                        "maximum_absolute_raw": float(np.max(np.abs(values[surface][selection]))),
                        "maximum_absolute_standardized": float(np.max(np.abs(z[selection]))),
                        "clipping_fraction_at_registered_8": float(np.mean(np.abs(z[selection]) > self.forensics.standardized_clip)),
                    })
                for feature in range(z.shape[2]):
                    group_rows.append({
                        "role": role, "surface": surface, "group_kind": "feature",
                        "group_id": feature_names[feature], "protocol_family": None,
                        "maximum_absolute_raw": float(np.max(np.abs(values[surface][:, :, feature]))),
                        "maximum_absolute_standardized": float(np.max(np.abs(z[:, :, feature]))),
                        "clipping_fraction_at_registered_8": float(np.mean(np.abs(z[:, :, feature]) > self.forensics.standardized_clip)),
                    })
                for segment in range(z.shape[1]):
                    group_rows.append({
                        "role": role, "surface": surface, "group_kind": "segment",
                        "group_id": str(segment), "protocol_family": None,
                        "maximum_absolute_raw": float(np.max(np.abs(values[surface][:, segment]))),
                        "maximum_absolute_standardized": float(np.max(np.abs(z[:, segment]))),
                        "clipping_fraction_at_registered_8": float(np.mean(np.abs(z[:, segment]) > self.forensics.standardized_clip)),
                    })
                flat = np.abs(z).reshape(-1)
                for flat_index in np.argpartition(flat, -min(10, len(flat)))[-10:]:
                    sample, segment, feature = np.unravel_index(flat_index, z.shape)
                    report["top_outliers"].append({
                        "role": role, "surface": surface,
                        "logical_index": int(values["indices"][sample]),
                        "sample_position": int(sample), "segment_id": int(segment),
                        "pair_position": int(sample // 2), "pair_branch": int(sample % 2),
                        "trajectory_id": str(self.store.metadata["trajectory_id"][int(values["indices"][sample])]),
                        "protocol_family": self._protocol_family(int(values["indices"][sample])),
                        "region": str(self.layout.segments[int(segment)].get("region", "unknown")),
                        "feature_id": int(feature), "feature_name": feature_names[feature],
                        "raw_value": float(values[surface][sample, segment, feature]),
                        "standardized_value": float(z[sample, segment, feature]),
                    })
        np.savez_compressed(
            self.output_dir / "forensic_feature_normalizers.npz",
            **{
                f"{surface}_{kind}": value
                for surface, (mean, scale) in normalizers.items()
                for kind, value in (("mean", mean), ("scale", scale))
            },
        )
        h2_ratio = report["roles"]["heldout"]["h2_raw"]["raw_max_norm_to_train_ratio"]
        h2_zmax = report["roles"]["heldout"]["h2_raw"]["standardized_unclipped"]["maximum_absolute"]
        h2_clip = report["roles"]["heldout"]["h2_raw"]["clipping_fraction_at_registered_8"]
        causal_ratio = report["roles"]["heldout"]["causal_raw"]["raw_max_norm_to_train_ratio"]
        zero_ratio = report["roles"]["heldout"]["h2_zero_causal_raw"]["raw_max_norm_to_train_ratio"]
        report["state_and_voltage_surfaces"] = {}
        for role, values in self.roles.items():
            report["state_and_voltage_surfaces"][role] = {
                "teacher_state_normalized_max_absolute": float(
                    np.max(np.abs(values["teacher_state_normalized"]))
                ),
                "voltage_t_max_absolute_mv": float(np.max(np.abs(values["voltage_t"]))),
                "h2_base_voltage_max_absolute_mv": float(np.max(np.abs(values["base"]))),
                "zero_causal_base_voltage_max_absolute_mv": float(
                    np.max(np.abs(values["zero_causal_base"]))
                ),
                "h2_base_delta_from_state_max_absolute_mv": float(
                    np.max(np.abs(values["base"] - values["voltage_t"]))
                ),
            }
        train_state_max = max(
            report["state_and_voltage_surfaces"]["train"][
                "teacher_state_normalized_max_absolute"
            ], 1e-12,
        )
        state_ratio = (
            report["state_and_voltage_surfaces"]["heldout"][
                "teacher_state_normalized_max_absolute"
            ] / train_state_max
        )
        raw_ood = bool(
            h2_ratio > self.forensics.raw_heldout_norm_ratio_limit
            or h2_zmax > self.forensics.standardized_heldout_max_limit
            or h2_clip > self.forensics.maximum_allowed_clipping_fraction
        )
        if state_ratio > self.forensics.raw_heldout_norm_ratio_limit:
            origin = "NORMALIZED_TEACHER_STATE_OOD"
        elif causal_ratio > self.forensics.raw_heldout_norm_ratio_limit:
            origin = "CAUSAL_FRONTEND_OOD"
        elif zero_ratio > self.forensics.raw_heldout_norm_ratio_limit:
            origin = "FROZEN_H2_STATE_PATH_AMPLIFICATION"
        else:
            origin = "CAUSAL_DRIVE_AMPLIFIED_INSIDE_FROZEN_H2"
        report.update({
            "schema_version": "05h-raw-scale-forensics-v1",
            "valid": True,
            "heldout_boundary_targets_materialized": False,
            "heldout_event_targets_materialized": False,
            "raw_heldout_ood_blocker": raw_ood,
            "inferred_anomaly_origin": origin,
            "h2_raw_heldout_to_train_max_norm_ratio": h2_ratio,
            "causal_raw_heldout_to_train_max_norm_ratio": causal_ratio,
            "zero_causal_h2_heldout_to_train_max_norm_ratio": zero_ratio,
            "normalized_teacher_state_heldout_to_train_max_ratio": state_ratio,
            "thresholds": {
                "raw_norm_ratio": self.forensics.raw_heldout_norm_ratio_limit,
                "standardized_max": self.forensics.standardized_heldout_max_limit,
                "clipping_fraction": self.forensics.maximum_allowed_clipping_fraction,
            },
        })
        report["top_outliers"] = sorted(
            report["top_outliers"], key=lambda row: abs(row["standardized_value"]), reverse=True
        )[:100]
        _write_json(self.output_dir / "raw_scale_forensics.json", report)
        write_parquet(self.output_dir / "raw_scale_outliers.parquet", report["top_outliers"])
        write_parquet(self.output_dir / "raw_scale_group_metrics.parquet", group_rows)
        return report

    def run_projection_forensics(self) -> Dict[str, Any]:
        train = self.roles["train"]
        target_residual = train["target"] - train["base"]
        prediction, rows = local_linear_projection(
            train["h2_raw_z"], target_residual, self.forensics.projection_rcond
        )
        for row in rows:
            segment = int(row["segment_id"])
            metadata = self.layout.segments[segment]
            row["region"] = str(metadata.get("region", "unknown"))
            row["section_name"] = str(metadata.get("section", metadata.get("section_name", "")))
        metrics = self._pair_set_metrics(
            (train["base"] + prediction).reshape(-1, 2, prediction.shape[-1]),
            train["target"].reshape(-1, 2, prediction.shape[-1]),
        )
        failing = sorted(rows, key=lambda row: row["projection_rmse_mv"], reverse=True)
        rank = np.asarray([row["design_rank"] for row in rows], dtype=np.float64)
        error = np.asarray([row["projection_rmse_mv"] for row in rows], dtype=np.float64)
        correlation = float(np.corrcoef(rank, error)[0, 1]) if np.std(rank) and np.std(error) else None
        region_summary = []
        for region in sorted({row["region"] for row in rows}):
            chosen = [row for row in rows if row["region"] == region]
            region_summary.append({
                "region": region, "segment_count": len(chosen),
                "median_design_rank": float(np.median([r["design_rank"] for r in chosen])),
                "median_projection_rmse_mv": float(np.median([r["projection_rmse_mv"] for r in chosen])),
                "maximum_projection_rmse_mv": float(max(r["projection_rmse_mv"] for r in chosen)),
            })
        report = {
            "schema_version": "05h-projection-forensics-v1",
            "valid": True,
            "input_surface": "unclipped train-standardized frozen H2",
            "pair_metrics": metrics,
            "minimum_design_rank": int(rank.min()), "maximum_design_rank": int(rank.max()),
            "rank_error_correlation": correlation,
            "segment_count_with_projection_rmse_above_1mv": sum(r["projection_rmse_mv"] > 1.0 for r in rows),
            "worst_segments": failing[:25], "region_summary": region_summary,
            "heldout_targets_used": False,
        }
        _write_json(self.output_dir / "projection_forensics.json", report)
        write_parquet(self.output_dir / "projection_by_segment.parquet", rows)
        return report

    def _context_features(self, role: Mapping[str, Any]) -> np.ndarray:
        sample_count = role["base"].shape[0]
        static = np.asarray(self.arrays["segment_static"], dtype=np.float64)
        static_mean = static.mean(0, keepdims=True)
        static_scale = np.maximum(static.std(0, keepdims=True), self.forensics.feature_epsilon)
        static = np.tanh((static - static_mean) / static_scale)
        static = np.broadcast_to(static[None], (sample_count, *static.shape))
        voltage = np.tanh(role["voltage_t"][..., None] / 100.0)
        base = np.tanh(role["base"][..., None] / 100.0)
        return np.concatenate([voltage, base, static], axis=-1)

    def _family_features(self, role: Mapping[str, Any], family: str) -> np.ndarray:
        parts = [self._context_features(role)]
        if family in {"h2", "h2_causal"}:
            parts.append(role["h2_raw_bounded"])
        if family in {"causal", "h2_causal"}:
            parts.append(role["causal_raw_bounded"])
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _pair_passes(self, metrics: Mapping[str, Any]) -> bool:
        rows = metrics["pair_metrics"]
        return bool(rows) and all(
            row["voltage_rmse_mv"] < self.forensics.pair_rmse_mv
            and row["maximum_segment_error_mv"] < self.forensics.pair_max_error_mv
            and self.forensics.pair_retention_minimum <= row["branching_retention"] <= self.forensics.pair_retention_maximum
            for row in rows
        )

    def run_bounded_representation_controls(self) -> Dict[str, Any]:
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train = self.roles["train"]
        development = self.roles["development"]
        target_train = torch.as_tensor(train["target"] - train["base"], dtype=torch.float32, device=device)
        target_contract = {}
        for role_name, role in (("train", train), ("development", development)):
            target_residual = role["target"] - role["base"]
            target_contract[role_name] = {
                "maximum_absolute_target_residual_mv": float(
                    np.max(np.abs(target_residual))
                ),
                "fraction_outside_residual_bound": float(
                    np.mean(np.abs(target_residual) > self.forensics.residual_limit_mv)
                ),
            }
        runs = []
        total = len(self.forensics.input_families) * len(self.forensics.seeds)
        progress = Progress("bounded representation controls", total)
        completed = 0
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        for family in self.forensics.input_families:
            train_x = torch.as_tensor(self._family_features(train, family), device=device)
            dev_x = torch.as_tensor(self._family_features(development, family), device=device)
            self.feature_widths[family] = int(train_x.shape[-1])
            for seed in self.forensics.seeds:
                torch.manual_seed(int(seed)); np.random.seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                model = BoundedLocalResidualHead(
                    train_x.shape[-1], self.layout.segment_count,
                    self.forensics.hidden_width, self.forensics.segment_embedding_dim,
                    self.forensics.residual_limit_mv,
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=self.forensics.learning_rate,
                    weight_decay=self.forensics.weight_decay,
                )
                best_state = None; best_score = math.inf; best_epoch = 0; stale = 0
                history = []
                run_progress = Progress(
                    f"{family} seed {seed}", self.forensics.epochs
                )
                for epoch in range(1, self.forensics.epochs + 1):
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    residual = model(train_x)
                    point = torch.mean((residual - target_train) ** 2)
                    paired = residual.reshape(-1, 2, residual.shape[-1])
                    target_paired = target_train.reshape_as(paired)
                    branch = torch.mean(((paired[:, 0] - paired[:, 1]) - (target_paired[:, 0] - target_paired[:, 1])) ** 2)
                    loss = point + self.forensics.branch_loss_weight * branch
                    loss.backward()
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.forensics.gradient_clip_norm))
                    optimizer.step()
                    if epoch % self.forensics.evaluation_interval == 0 or epoch == 1:
                        model.eval()
                        with torch.no_grad():
                            train_pred = (train["base"] + model(train_x).cpu().double().numpy()).reshape(-1, 2, self.layout.segment_count)
                            dev_pred = (development["base"] + model(dev_x).cpu().double().numpy()).reshape(-1, 2, self.layout.segment_count)
                        train_metrics = self._pair_set_metrics(train_pred, train["target"].reshape(-1, 2, self.layout.segment_count))
                        dev_metrics = self._pair_set_metrics(dev_pred, development["target"].reshape(-1, 2, self.layout.segment_count))
                        score = dev_metrics["aggregate_voltage_rmse_mv"] + 0.05 * dev_metrics["maximum_segment_error_mv"]
                        history.append({
                            "family": family, "seed": int(seed), "epoch": epoch,
                            "loss": float(loss.detach()), "gradient_norm": gradient_norm,
                            "train_rmse_mv": train_metrics["aggregate_voltage_rmse_mv"],
                            "development_rmse_mv": dev_metrics["aggregate_voltage_rmse_mv"],
                            "development_max_error_mv": dev_metrics["maximum_segment_error_mv"],
                        })
                        run_progress.update(
                            epoch,
                            f"train={train_metrics['aggregate_voltage_rmse_mv']:.3g} "
                            f"dev={dev_metrics['aggregate_voltage_rmse_mv']:.3g}",
                        )
                        if score < best_score:
                            best_score = score; best_epoch = epoch; stale = 0
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        else:
                            stale += self.forensics.evaluation_interval
                        if stale >= self.forensics.patience:
                            break
                if best_state is None:
                    raise RuntimeError("05h nonlinear control produced no checkpoint")
                model.load_state_dict(best_state); model.eval()
                with torch.no_grad():
                    train_residual = model(train_x).cpu().double().numpy()
                    dev_residual = model(dev_x).cpu().double().numpy()
                train_metrics = self._pair_set_metrics(
                    (train["base"] + train_residual).reshape(-1, 2, self.layout.segment_count),
                    train["target"].reshape(-1, 2, self.layout.segment_count),
                )
                dev_metrics = self._pair_set_metrics(
                    (development["base"] + dev_residual).reshape(-1, 2, self.layout.segment_count),
                    development["target"].reshape(-1, 2, self.layout.segment_count),
                )
                checkpoint = checkpoint_dir / f"{family}_seed{seed}.pt"
                torch.save({"state_dict": best_state, "family": family, "seed": seed, "best_epoch": best_epoch}, checkpoint)
                run = {
                    "family": family, "seed": int(seed), "best_epoch": best_epoch,
                    "feature_width": int(train_x.shape[-1]),
                    "parameter_count": int(sum(p.numel() for p in model.parameters())),
                    "train": train_metrics, "development": dev_metrics,
                    "train_passed": self._pair_passes(train_metrics),
                    "development_passed": self._pair_passes(dev_metrics),
                    "checkpoint": checkpoint.relative_to(self.output_dir).as_posix(),
                    "heldout_candidate_head_inference_performed": False,
                }
                runs.append(run)
                write_parquet(self.output_dir / f"history_{family}_seed{seed}.parquet", history)
                completed += 1
                progress.update(completed, f"{family} seed={seed} train={train_metrics['aggregate_voltage_rmse_mv']:.3g} dev={dev_metrics['aggregate_voltage_rmse_mv']:.3g}")
                del model
        family_summary = []
        for family in self.forensics.input_families:
            selected = [run for run in runs if run["family"] == family]
            family_summary.append({
                "family": family,
                "seed_count": len(selected),
                "train_pass_count": sum(run["train_passed"] for run in selected),
                "development_pass_count": sum(run["train_passed"] and run["development_passed"] for run in selected),
                "median_train_rmse_mv": float(np.median([run["train"]["aggregate_voltage_rmse_mv"] for run in selected])),
                "median_development_rmse_mv": float(np.median([run["development"]["aggregate_voltage_rmse_mv"] for run in selected])),
            })
        report = {
            "schema_version": "05h-bounded-representation-controls-v1",
            "valid": len(runs) == total, "runs": runs,
            "family_summary": family_summary,
            "target_residual_contract": target_contract,
            "heldout_boundary_targets_materialized": False,
            "heldout_event_targets_materialized": False,
            "heldout_frozen_h2_feature_extraction_performed": True,
            "heldout_candidate_head_inference_performed": False,
            "base_h2_frozen": True,
            "fixed_seeds": list(self.forensics.seeds),
            "cudnn_deterministic": True,
        }
        _write_json(self.output_dir / "bounded_representation_controls.json", report)
        write_parquet(self.output_dir / "bounded_representation_summary.parquet", family_summary)
        return report

    def finalize_forensics(
        self, raw: Mapping[str, Any], projection: Mapping[str, Any],
        controls: Mapping[str, Any],
    ) -> Dict[str, Any]:
        passing = [
            run for run in controls["runs"]
            if run["train_passed"] and run["development_passed"]
        ]
        target_bound_valid = all(
            row["fraction_outside_residual_bound"] == 0.0
            for row in controls["target_residual_contract"].values()
        )
        if not target_bound_valid:
            representation_diagnosis = "REGISTERED_RESIDUAL_BOUND_TOO_TIGHT"
        elif not passing:
            representation_diagnosis = "BOUNDED_REPRESENTATION_CONTROLS_FAIL_TRAIN_OR_DEVELOPMENT"
        else:
            representation_diagnosis = "BOUNDED_REPRESENTATION_CANDIDATE_PASSES_DEVELOPMENT"
        if raw["raw_heldout_ood_blocker"]:
            diagnosis = "FROZEN_H2_HELDOUT_INPUT_OOD"
        else:
            diagnosis = representation_diagnosis
        self._plot_forensics(raw, projection, controls)
        report = {
            "schema_version": "05h-final-report-v1", "valid": True,
            "decision": "DIAGNOSTIC_ONLY_NO_FULL_TRAINING",
            "full_training_authorized": False, "diagnosis": diagnosis,
            "representation_diagnosis": representation_diagnosis,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05g": self.artifact_05g_contract,
            "raw_scale_forensics": raw,
            "projection_forensics": projection,
            "bounded_representation_controls": controls,
            "heldout_contract": {
                "boundary_targets_materialized": False,
                "event_targets_materialized": False,
                "frozen_h2_feature_extraction_performed": True,
                "candidate_head_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "registered_05g_support_reused_exactly": True,
                "base_h2_frozen": True, "train_and_development_only": True,
                "pre_clipping_ood_audited": True, "rollout_performed": False,
                "full_training_path_present": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "full_training_authorized": False,
                "experiment": (
                    "05i_teacher_state_and_causal_input_scale_repair"
                    if raw["raw_heldout_ood_blocker"]
                    else "05i_bounded_representation_micro_rollout"
                    if passing else "05i_representation_redesign"
                ),
            },
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append({
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
                })
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05h-artifact-index-v1",
            "artifact_count": len(records), "artifacts": records,
        })
        return report

    def _plot_forensics(
        self, raw: Mapping[str, Any], projection: Mapping[str, Any],
        controls: Mapping[str, Any],
    ) -> None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        surfaces = ["h2_raw", "h2_zero_causal_raw", "causal_raw"]
        axes[0].bar(
            ["H2", "H2 zero U", "causal U"],
            [raw["roles"]["heldout"][name]["raw_max_norm_to_train_ratio"] for name in surfaces],
        )
        axes[0].axhline(self.forensics.raw_heldout_norm_ratio_limit, color="black", linestyle="--")
        axes[0].set_yscale("log")
        axes[0].set(title="Held-out/train raw max norm", ylabel="ratio (log)")
        region = projection["region_summary"]
        axes[1].bar(
            [row["region"] for row in region],
            [row["median_projection_rmse_mv"] for row in region],
        )
        axes[1].axhline(self.forensics.pair_rmse_mv, color="black", linestyle="--")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set(title="Linear projection residual", ylabel="median RMSE (mV)")
        family = controls["family_summary"]
        positions = np.arange(len(family)); width = 0.35
        axes[2].bar(
            positions - width / 2,
            [row["median_train_rmse_mv"] for row in family], width, label="train",
        )
        axes[2].bar(
            positions + width / 2,
            [row["median_development_rmse_mv"] for row in family], width, label="development",
        )
        axes[2].set_xticks(positions, [row["family"] for row in family], rotation=25)
        axes[2].axhline(self.forensics.pair_rmse_mv, color="black", linestyle="--")
        axes[2].set(title="Bounded nonlinear controls", ylabel="RMSE (mV)")
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        figure.savefig(self.output_dir / "representation_forensics.png", dpi=160)
        plt.close(figure)
