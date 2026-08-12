"""Notebook-05g regularized optimization audit for segment conditioning."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import CompositeFlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_capacity_experiment import HinesCapacityConfig
from .hines_conditioning_experiment import HinesConditioningConfig
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import HinesIsolationConfig, sha256_file
from .hines_segment_canary_experiment import (
    HinesSegmentCanaryConfig,
    HinesSegmentMicroCanaryExperiment,
)
from .hines_layer import require_torch

try:
    import torch
except ImportError:  # pragma: no cover - data-only environments.
    torch = None


EXPECTED_05F_ARCHIVE_SHA256 = (
    "3a641d10ede14d426c640964cf0a6491259f6297403564db0d694f544da22239"
)
EXPECTED_05F_MEMBER_SHA256 = {
    "artifact_index.json": "1e5b515ebf2e6c0d178d54ee07c1abc85b5c4ca2dc94c211bcfac75bb8f336c8",
    "micro_canary_config.json": "822cf3726e0621550a890768ed3e443fea182bc2b9b81ffdc7497b1f1ff5e7cc",
    "pair_plan.json": "d8754be4f46208e896a9c38eab37fdf95c2d3e7c66f1e6e129db82a600780a1a",
    "feature_contract.json": "a3c9328e8f1162d7b47306f22b51b01eb319a013fba947d31891734de914b7f9",
    "spectral_basis_report.json": "ea2b8aa396f3196cf104ac7e3b30f8ae09c6a33672e56fd9ffe7dc54664f1e82",
    "micro_canary_report.json": "02a880eb9389ef19abe9c82b4283214a7850277aba82c0d6b3716cfdc4b125d3",
    "rank_64_report.json": "3fc13a6c1f25ae7121f8e248ac385df65136952b6721dbb8eedbedf963bb697d",
    "rank_96_report.json": "2ed349c49611d73386f7259ad00d8490c36b12d27fd2a59a06a6ac3b1be959b3",
    "final_report.json": "7f56a0adef6dc75edd5fc1eb3328e073cf2f3e30fb4f6f5f4412873c773a463a",
}


@dataclass(frozen=True)
class HinesOptimizationAuditConfig:
    desired_train_pair_count: int = 12
    minimum_train_pair_count: int = 6
    minimum_train_protocol_family_count: int = 2
    maximum_local_steps_searched: int = 16
    maximum_candidates: int = 2048
    minimum_teacher_distance_mv: float = 0.01
    ridge_lambdas: Tuple[float, ...] = (
        1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0,
    )
    ranks: Tuple[int, ...] = (64, 96)
    feature_epsilon: float = 1e-6
    standardized_feature_clip: float = 8.0
    residual_limit_mv: float = 120.0
    maximum_heldout_to_train_feature_norm_ratio: float = 20.0
    maximum_coefficient_frobenius_norm: float = 1e5
    oracle_rmse_mv: float = 1e-5
    oracle_max_error_mv: float = 1e-4
    pair_rmse_mv: float = 1.0
    pair_max_error_mv: float = 5.0
    pair_retention_minimum: float = 0.90
    pair_retention_maximum: float = 1.10

    def validate(self) -> None:
        positive = (
            self.desired_train_pair_count, self.minimum_train_pair_count,
            self.minimum_train_protocol_family_count,
            self.maximum_local_steps_searched, self.maximum_candidates,
            self.minimum_teacher_distance_mv, self.feature_epsilon,
            self.standardized_feature_clip, self.residual_limit_mv,
            self.maximum_heldout_to_train_feature_norm_ratio,
            self.maximum_coefficient_frobenius_norm,
        )
        if min(positive) <= 0:
            raise ValueError("05g support and safety values must be positive")
        if self.desired_train_pair_count < self.minimum_train_pair_count:
            raise ValueError("desired train support must cover its minimum")
        if tuple(self.ranks) != (64, 96):
            raise ValueError("05g must audit ranks 64 and 96")
        lambdas = tuple(float(value) for value in self.ridge_lambdas)
        if not lambdas or min(lambdas) <= 0 or lambdas != tuple(sorted(set(lambdas))):
            raise ValueError("ridge lambdas must be positive, unique, and increasing")
        if not 0 < self.pair_retention_minimum < self.pair_retention_maximum:
            raise ValueError("invalid pair-retention interval")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesOptimizationAuditConfig":
        payload = dict(values)
        for name in ("ridge_lambdas", "ranks"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def dual_ridge_segment_coefficients(
    features: np.ndarray,
    target_residual: np.ndarray,
    ridge_lambda: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit per-segment coefficients through the small dual ridge system."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target_residual, dtype=np.float64)
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("expected features [sample, segment, feature] and residual")
    sample_count, segment_count, feature_count = x.shape
    target_mean = y.mean(axis=0)
    feature_mean = x.mean(axis=0)
    centered_y = y - target_mean[None, :]
    centered_x = x - feature_mean[None, :, :]
    coefficients = np.zeros((segment_count, feature_count), dtype=np.float64)
    local_ranks = []
    local_conditions = []
    identity = np.eye(sample_count, dtype=np.float64)
    for segment in range(segment_count):
        local = centered_x[:, segment, :]
        gram = local @ local.T
        singular = np.linalg.svd(local, compute_uv=False)
        tolerance = 1e-10 * (float(singular[0]) if len(singular) else 0.0)
        nonzero = singular[singular > tolerance]
        local_ranks.append(int(len(nonzero)))
        local_conditions.append(
            float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
        )
        dual = np.linalg.solve(
            gram + float(ridge_lambda) * identity, centered_y[:, segment]
        )
        coefficients[segment] = local.T @ dual
    diagnostics = {
        "ridge_lambda": float(ridge_lambda),
        "sample_count": sample_count,
        "segment_count": segment_count,
        "feature_count": feature_count,
        "minimum_local_rank": min(local_ranks),
        "maximum_local_rank": max(local_ranks),
        "median_local_condition_number": float(np.median(local_conditions)),
        "maximum_local_condition_number": float(np.max(local_conditions)),
        "coefficient_frobenius_norm": float(np.linalg.norm(coefficients)),
        "coefficient_max_absolute": float(np.max(np.abs(coefficients))),
        "target_mean_l2_norm": float(np.linalg.norm(target_mean)),
        "target_mean_max_absolute_mv": float(np.max(np.abs(target_mean))),
    }
    return target_mean, feature_mean, coefficients, diagnostics


def bounded_segment_prediction(
    features: np.ndarray,
    bias: np.ndarray,
    coefficients: np.ndarray,
    residual_limit_mv: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    raw = np.asarray(bias)[None, :] + np.einsum(
        "nsf,sf->ns", np.asarray(features), np.asarray(coefficients)
    )
    bounded = np.clip(raw, -float(residual_limit_mv), float(residual_limit_mv))
    return bounded, {
        "raw_residual_max_absolute_mv": float(np.max(np.abs(raw))),
        "bounded_residual_max_absolute_mv": float(np.max(np.abs(bounded))),
        "clipped_fraction": float(np.mean(np.abs(raw) > float(residual_limit_mv))),
    }


class HinesSegmentOptimizationAudit(HinesSegmentMicroCanaryExperiment):
    """Train-first 05g audit with sealed held-out targets."""

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
        checkpoint_05b_source: Path,
        artifact_05c_source: Path,
        artifact_05d_source: Path,
        artifact_05e_source: Path,
        artifact_05f_source: Path,
        code_revision: Optional[str] = None,
    ) -> None:
        super().__init__(
            bundle, output_dir, model_config, isolation_config,
            conditioning_config, capacity_config, canary_config,
            checkpoint_05b_source, artifact_05c_source, artifact_05d_source,
            artifact_05e_source, code_revision=code_revision,
        )
        audit_config.validate()
        self.audit = audit_config
        self.artifact_05f_source = Path(artifact_05f_source).resolve()
        self.artifact_05f_contract: Dict[str, Any] = {}
        self.artifact_05f_report: Dict[str, Any] = {}
        self.audit_plan: Dict[str, Any] = {}
        self.data: Dict[str, Any] = {}
        self.audit_rows: List[Dict[str, Any]] = []
        self.candidate_models: Dict[Tuple[float, int], Dict[str, Any]] = {}

    def _read_05f_source(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05f_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05F_ARCHIVE_SHA256:
                raise RuntimeError("05f archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                members: Dict[str, bytes] = {}
                resolved: Dict[str, str] = {}
                for suffix in EXPECTED_05F_MEMBER_SHA256:
                    matches = [
                        name for name in archive.namelist()
                        if name.replace("\\", "/").endswith(suffix)
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one 05f member ending in {suffix!r}, found {matches}"
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
            for suffix in EXPECTED_05F_MEMBER_SHA256:
                matches = list(source.rglob(Path(suffix).name))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one extracted 05f member ending in {suffix!r}, "
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
            raise RuntimeError(f"05f artifact source does not exist: {source}")
        observed = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        }
        mismatches = {
            name: {"expected": EXPECTED_05F_MEMBER_SHA256[name], "observed": value}
            for name, value in observed.items()
            if value != EXPECTED_05F_MEMBER_SHA256[name]
        }
        if mismatches:
            raise RuntimeError(f"05f member SHA-256 mismatch: {mismatches}")
        contract["verified_member_sha256"] = observed
        return json.loads(members["final_report.json"]), contract

    def prepare_optimization_audit(self) -> Dict[str, Any]:
        base = self.prepare_micro_canary()
        report, contract = self._read_05f_source()
        blockers = []
        if report.get("diagnosis") != "SEGMENT_CONDITIONED_MICRO_CANARY_OPTIMIZATION_FAILURE":
            blockers.append(f"unexpected 05f diagnosis: {report.get('diagnosis')}")
        if report.get("full_training_authorized") is not False:
            blockers.append("05f unexpectedly authorizes full training")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05f and mounted composite fingerprints disagree")
        if blockers:
            raise RuntimeError(f"05g provenance blockers: {blockers}")
        self.artifact_05f_report = report
        self.artifact_05f_contract = contract
        payload = {
            "schema_version": "05g-optimization-audit-config-v1",
            "audit": asdict(self.audit),
            "artifact_05f": contract,
            "code_revision": self.code_revision,
            "heldout_target_gate": "sealed until a train-and-development-safe candidate exists",
            "full_training_authorized": False,
        }
        _write_json(self.output_dir / "optimization_audit_config.json", payload)
        return {**base, **payload}

    def _protocol_family(self, index: int) -> str:
        trajectory = str(self.store.metadata["trajectory_id"][int(index)])
        episode = self.store.episode_by_trajectory.get(trajectory, {})
        def value(name: str, fallback: str) -> str:
            if name in episode:
                return str(episode[name])
            if name in self.store.metadata:
                return str(self.store.metadata[name][int(index)])
            return fallback

        protocol = value("protocol", "unknown_protocol")
        variant = value("protocol_variant", "unknown_variant")
        return f"{protocol}|{variant}"

    def build_optimization_support(self) -> Dict[str, Any]:
        print(
            "[HayFlow 05g][support] scanning train counterfactuals "
            f"through {self.audit.maximum_local_steps_searched} local steps...",
            flush=True,
        )
        original_canary = self.canary
        self.canary = replace(
            self.canary,
            maximum_local_steps_searched=self.audit.maximum_local_steps_searched,
            maximum_candidates_per_split=self.audit.maximum_candidates,
            minimum_teacher_distance_mv=self.audit.minimum_teacher_distance_mv,
        )
        try:
            candidates = self._pair_candidates_for_split("train")
        finally:
            self.canary = original_canary
        for row in candidates:
            row["protocol_family"] = " <> ".join(sorted({
                self._protocol_family(row["left_index"]),
                self._protocol_family(row["right_index"]),
            }))
        by_family: Dict[str, List[Dict[str, Any]]] = {}
        for row in candidates:
            by_family.setdefault(row["protocol_family"], []).append(row)
        development = list(self.branch_pair or ())
        development_episodes = {
            self._episode_identity(index)[1] for index in development
        }
        selected: List[Dict[str, Any]] = []
        used_episodes = set(development_episodes)
        while len(selected) < self.audit.desired_train_pair_count:
            added = False
            for family in sorted(by_family):
                for row in by_family[family]:
                    episodes = {row["left_episode_id"], row["right_episode_id"]}
                    if episodes & used_episodes:
                        continue
                    selected.append(dict(row))
                    used_episodes.update(episodes)
                    added = True
                    break
                if len(selected) == self.audit.desired_train_pair_count:
                    break
            if not added:
                break
        selected_families = sorted({row["protocol_family"] for row in selected})
        blockers = []
        if len(selected) < self.audit.minimum_train_pair_count:
            blockers.append(
                f"only {len(selected)} train pairs; minimum is "
                f"{self.audit.minimum_train_pair_count}"
            )
        diversity_available = len(by_family) >= self.audit.minimum_train_protocol_family_count
        diversity_selected = len(selected_families) >= self.audit.minimum_train_protocol_family_count
        plan_hash = hashlib.sha256(json.dumps(
            {"selected": selected, "development": development},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        plan = {
            "schema_version": "05g-optimization-support-v1",
            "valid": not blockers,
            "blockers": blockers,
            "candidate_pair_count": len(candidates),
            "available_protocol_family_count": len(by_family),
            "available_protocol_families": {
                family: len(rows) for family, rows in sorted(by_family.items())
            },
            "protocol_diversity_available": diversity_available,
            "selected_pair_count": len(selected),
            "selected_protocol_family_count": len(selected_families),
            "selected_protocol_families": selected_families,
            "protocol_diversity_selected": diversity_selected,
            "development_pair": development,
            "development_episode_ids": sorted(development_episodes),
            "selected_pairs": selected,
            "support_sha256": plan_hash,
        }
        _write_json(self.output_dir / "optimization_support.json", plan)
        write_parquet(
            self.output_dir / "optimization_support.parquet", selected
        )
        self.audit_plan = plan
        if blockers:
            raise RuntimeError(f"05g support blockers: {blockers}")
        print(
            "[HayFlow 05g][support] selected "
            f"{len(selected)} pairs across {len(selected_families)} protocol families",
            flush=True,
        )
        return plan

    def _frozen_base_features(
        self, indices: Sequence[int], *, include_targets: bool
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = self._load_h2_checkpoint(device)
        model.eval()
        raw = self._batch(indices, include_targets=include_targets)
        batch = self._torch_batch(raw, device)
        with torch.no_grad():
            output = model(
                batch, ablation="H2", decode_teacher=False,
                boundary_mode="no_event_jump",
            )
        base = output["voltage"].detach().cpu().double().numpy()
        features = output["boundary_features"].detach().cpu().double().numpy()
        target = (
            np.asarray(raw["voltage_target"], dtype=np.float64)
            if include_targets else None
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return base, features, target

    @staticmethod
    def _norm_report(features: np.ndarray) -> Dict[str, float]:
        norms = np.linalg.norm(features, axis=-1)
        absolute = np.abs(features)
        return {
            "minimum_segment_norm": float(norms.min()),
            "median_segment_norm": float(np.median(norms)),
            "p95_segment_norm": float(np.quantile(norms, 0.95)),
            "maximum_segment_norm": float(norms.max()),
            "maximum_absolute_feature": float(absolute.max()),
            "nonfinite_count": int(np.size(features) - np.isfinite(features).sum()),
        }

    def prepare_audit_data(self) -> Dict[str, Any]:
        if not self.audit_plan.get("valid"):
            raise RuntimeError("build_optimization_support() must succeed first")
        train_indices = self._pair_indices(self.audit_plan["selected_pairs"])
        development_indices = list(self.branch_pair or ())
        heldout_pairs = self.artifact_05f_report["pair_plan"]["heldout_pairs"]
        heldout_indices = self._pair_indices(heldout_pairs)
        print("[HayFlow 05g][features] extracting frozen H2 train features...", flush=True)
        train_base, train_features, train_target = self._frozen_base_features(
            train_indices, include_targets=True
        )
        print("[HayFlow 05g][features] extracting development features...", flush=True)
        dev_base, dev_features, dev_target = self._frozen_base_features(
            development_indices, include_targets=True
        )
        print(
            "[HayFlow 05g][features] extracting held-out inputs with boundary targets sealed...",
            flush=True,
        )
        heldout_base, heldout_features, heldout_target = self._frozen_base_features(
            heldout_indices, include_targets=False
        )
        if heldout_target is not None:
            raise RuntimeError("held-out targets were unexpectedly materialized")
        mean = train_features.mean(axis=(0, 1), keepdims=True)
        raw_std = train_features.std(axis=(0, 1), keepdims=True)
        scale = np.maximum(raw_std, self.audit.feature_epsilon)
        np.savez_compressed(
            self.output_dir / "feature_normalization.npz",
            mean=mean.astype(np.float64),
            scale=scale.astype(np.float64),
        )

        def normalize(values: np.ndarray) -> np.ndarray:
            standardized = (values - mean) / scale
            return np.clip(
                standardized, -self.audit.standardized_feature_clip,
                self.audit.standardized_feature_clip,
            )

        train_normalized = normalize(train_features)
        dev_normalized = normalize(dev_features)
        heldout_normalized = normalize(heldout_features)
        norms = {
            "train_raw": self._norm_report(train_features),
            "development_raw": self._norm_report(dev_features),
            "heldout_raw": self._norm_report(heldout_features),
            "train_standardized_clipped": self._norm_report(train_normalized),
            "development_standardized_clipped": self._norm_report(dev_normalized),
            "heldout_standardized_clipped": self._norm_report(heldout_normalized),
        }
        train_reference = max(
            norms["train_standardized_clipped"]["p95_segment_norm"], 1e-12
        )
        heldout_ratio = (
            norms["heldout_standardized_clipped"]["maximum_segment_norm"]
            / train_reference
        )
        scale_safe = bool(
            heldout_ratio <= self.audit.maximum_heldout_to_train_feature_norm_ratio
            and all(row["nonfinite_count"] == 0 for row in norms.values())
        )
        pair_counts = {
            "train": len(train_indices) // 2,
            "development": len(development_indices) // 2,
            "heldout": len(heldout_indices) // 2,
        }
        self.data = {
            "train": {
                "base": train_base.reshape(pair_counts["train"], 2, -1),
                "features": train_normalized.reshape(
                    pair_counts["train"], 2, train_features.shape[1], train_features.shape[2]
                ),
                "target": train_target.reshape(pair_counts["train"], 2, -1),
            },
            "development": {
                "base": dev_base.reshape(1, 2, -1),
                "features": dev_normalized.reshape(1, 2, dev_features.shape[1], dev_features.shape[2]),
                "target": dev_target.reshape(1, 2, -1),
            },
            "heldout": {
                "indices": heldout_indices,
                "base": heldout_base.reshape(pair_counts["heldout"], 2, -1),
                "features": heldout_normalized.reshape(
                    pair_counts["heldout"], 2, heldout_features.shape[1], heldout_features.shape[2]
                ),
                "target": None,
            },
        }
        contract = {
            "schema_version": "05g-feature-scale-audit-v1",
            "pair_counts": pair_counts,
            "feature_count": train_features.shape[2],
            "segment_count": train_features.shape[1],
            "feature_epsilon": self.audit.feature_epsilon,
            "standardized_feature_clip": self.audit.standardized_feature_clip,
            "minimum_raw_train_std": float(raw_std.min()),
            "maximum_raw_train_std": float(raw_std.max()),
            "feature_mean_sha256": hashlib.sha256(
                np.ascontiguousarray(mean).tobytes()
            ).hexdigest(),
            "feature_scale_sha256": hashlib.sha256(
                np.ascontiguousarray(scale).tobytes()
            ).hexdigest(),
            "normalization_artifact": "feature_normalization.npz",
            "norms": norms,
            "heldout_to_train_feature_norm_ratio": float(heldout_ratio),
            "maximum_allowed_ratio": self.audit.maximum_heldout_to_train_feature_norm_ratio,
            "scale_safe": scale_safe,
            "normalization_fit_roles": ["train"],
            "heldout_targets_materialized": False,
            "base_h2_frozen": True,
        }
        _write_json(self.output_dir / "feature_scale_audit.json", contract)
        print(
            "[HayFlow 05g][features] scale audit complete: "
            f"safe={scale_safe}, heldout/train norm ratio={heldout_ratio:.3g}",
            flush=True,
        )
        return contract

    def _all_pairs_pass(self, metrics: Mapping[str, Any]) -> bool:
        rows = metrics["pair_metrics"]
        return bool(rows) and all(
            row["voltage_rmse_mv"] < self.audit.pair_rmse_mv
            and row["maximum_segment_error_mv"] < self.audit.pair_max_error_mv
            and self.audit.pair_retention_minimum
            <= row["branching_retention"]
            <= self.audit.pair_retention_maximum
            for row in rows
        )

    def run_oracle_controls(self) -> Dict[str, Any]:
        train = self.data["train"]
        target_residual = train["target"] - train["base"]
        direct_prediction = train["base"] + target_residual
        direct_metrics = self._pair_set_metrics(direct_prediction, train["target"])
        bias = target_residual.reshape(-1, target_residual.shape[-1]).mean(axis=0)
        bias_residual = np.broadcast_to(bias, target_residual.shape)
        bias_metrics = self._pair_set_metrics(
            train["base"] + bias_residual, train["target"]
        )
        direct_error = direct_prediction - train["target"]
        report = {
            "schema_version": "05g-oracle-controls-v1",
            "valid": bool(
                np.sqrt(np.mean(direct_error ** 2)) < self.audit.oracle_rmse_mv
                and np.max(np.abs(direct_error)) < self.audit.oracle_max_error_mv
            ),
            "direct_per_transition_residual_oracle": direct_metrics,
            "segment_bias_only": bias_metrics,
            "direct_oracle_parameter_count": int(target_residual.size),
            "segment_bias_parameter_count": int(target_residual.shape[-1]),
            "interpretation": (
                "direct residual validates multi-pair target and metric plumbing; "
                "segment bias isolates static memorization"
            ),
        }
        _write_json(self.output_dir / "oracle_controls.json", report)
        return report

    def _candidate_metrics(
        self, role: str, bias: np.ndarray, coefficients: np.ndarray
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        values = self.data[role]
        features = values["features"].reshape(
            -1, values["features"].shape[-2], values["features"].shape[-1]
        )
        residual, safety = bounded_segment_prediction(
            features, bias, coefficients, self.audit.residual_limit_mv
        )
        residual = residual.reshape(values["base"].shape)
        metrics = self._pair_set_metrics(
            values["base"] + residual, values["target"]
        )
        return metrics, safety

    def run_regularized_train_audit(
        self, oracle_report: Mapping[str, Any], scale_report: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if not oracle_report.get("valid"):
            raise RuntimeError("multi-pair direct residual oracle failed")
        train = self.data["train"]
        flat_features = train["features"].reshape(
            -1, train["features"].shape[-2], train["features"].shape[-1]
        )
        flat_residual = (train["target"] - train["base"]).reshape(
            -1, train["target"].shape[-1]
        )
        total = len(self.audit.ridge_lambdas) * len(self.audit.ranks)
        progress = Progress("regularized train-first audit", total)
        completed = 0
        rows = []
        for ridge_lambda in self.audit.ridge_lambdas:
            target_mean, feature_mean, coefficients, fit_diagnostics = dual_ridge_segment_coefficients(
                flat_features, flat_residual, ridge_lambda
            )
            left, singular, right = np.linalg.svd(coefficients, full_matrices=False)
            for rank in self.audit.ranks:
                effective = min(int(rank), len(singular))
                low_rank = (
                    left[:, :effective] * singular[None, :effective]
                ) @ right[:effective]
                # The ridge system is fit on locally centered features. Fold
                # that center into the intercept after rank truncation.
                bias = target_mean - np.einsum(
                    "sf,sf->s", feature_mean, low_rank
                )
                train_metrics, train_safety = self._candidate_metrics(
                    "train", bias, low_rank
                )
                development_metrics, development_safety = self._candidate_metrics(
                    "development", bias, low_rank
                )
                coefficient_norm = float(np.linalg.norm(low_rank))
                coefficient_safe = bool(
                    coefficient_norm <= self.audit.maximum_coefficient_frobenius_norm
                )
                train_passed = self._all_pairs_pass(train_metrics)
                development_passed = self._all_pairs_pass(development_metrics)
                train_numeric_safe = bool(
                    coefficient_safe and scale_report["scale_safe"]
                    and train_safety["clipped_fraction"] == 0.0
                )
                development_numeric_safe = bool(
                    train_numeric_safe
                    and development_safety["clipped_fraction"] == 0.0
                )
                candidate_safe = bool(
                    train_passed and development_passed
                    and development_numeric_safe
                )
                row = {
                    "ridge_lambda": float(ridge_lambda),
                    "rank": int(rank),
                    "parameter_count": int(
                        flat_features.shape[1]
                        + effective * (flat_features.shape[1] + flat_features.shape[2])
                    ),
                    "coefficient_frobenius_norm": coefficient_norm,
                    "coefficient_max_absolute": float(np.max(np.abs(low_rank))),
                    "coefficient_safe": coefficient_safe,
                    "train_passed": train_passed,
                    "development_passed": development_passed,
                    "train_numeric_safe": train_numeric_safe,
                    "development_numeric_safe": development_numeric_safe,
                    "candidate_safe_for_heldout_reveal": candidate_safe,
                    "train_voltage_rmse_mv": train_metrics["aggregate_voltage_rmse_mv"],
                    "train_maximum_segment_error_mv": train_metrics["maximum_segment_error_mv"],
                    "train_median_branching_retention": train_metrics["median_branching_retention"],
                    "development_voltage_rmse_mv": development_metrics["aggregate_voltage_rmse_mv"],
                    "development_maximum_segment_error_mv": development_metrics["maximum_segment_error_mv"],
                    "development_median_branching_retention": development_metrics["median_branching_retention"],
                    "train_clipped_fraction": train_safety["clipped_fraction"],
                    "development_clipped_fraction": development_safety["clipped_fraction"],
                    "minimum_local_rank": fit_diagnostics["minimum_local_rank"],
                    "maximum_local_rank": fit_diagnostics["maximum_local_rank"],
                    "untruncated_coefficient_frobenius_norm": fit_diagnostics[
                        "coefficient_frobenius_norm"
                    ],
                }
                rows.append(row)
                self.audit_rows.append(dict(row))
                self.candidate_models[(float(ridge_lambda), int(rank))] = {
                    "bias": bias, "coefficients": low_rank,
                    "row": row, "train_metrics": train_metrics,
                    "development_metrics": development_metrics,
                    "train_safety": train_safety,
                    "development_safety": development_safety,
                }
                completed += 1
                progress.update(
                    completed,
                    f"lambda={ridge_lambda:g} rank={rank} "
                    f"V={row['train_voltage_rmse_mv']:.3g} "
                    f"max={row['train_maximum_segment_error_mv']:.3g}",
                )
        write_parquet(self.output_dir / "regularized_audit_metrics.parquet", rows)
        serialized: Dict[str, np.ndarray] = {
            "ridge_lambdas": np.asarray(self.audit.ridge_lambdas, dtype=np.float64),
            "ranks": np.asarray(self.audit.ranks, dtype=np.int64),
        }
        for lambda_position, ridge_lambda in enumerate(self.audit.ridge_lambdas):
            for rank in self.audit.ranks:
                model = self.candidate_models[(float(ridge_lambda), int(rank))]
                prefix = f"lambda_{lambda_position}_rank_{int(rank)}"
                serialized[f"{prefix}_bias"] = model["bias"].astype(np.float64)
                serialized[f"{prefix}_coefficients"] = model["coefficients"].astype(
                    np.float64
                )
        np.savez_compressed(
            self.output_dir / "regularized_candidate_models.npz", **serialized
        )
        safe = [row for row in rows if row["candidate_safe_for_heldout_reveal"]]
        best_train = min(rows, key=lambda row: (
            row["train_voltage_rmse_mv"], row["train_maximum_segment_error_mv"],
            row["coefficient_frobenius_norm"], row["ridge_lambda"], row["rank"],
        ))
        selected = min(safe, key=lambda row: (
            row["development_voltage_rmse_mv"],
            row["development_maximum_segment_error_mv"],
            row["coefficient_frobenius_norm"], row["ridge_lambda"], row["rank"],
        )) if safe else None
        report = {
            "schema_version": "05g-regularized-train-audit-v1",
            "valid": len(rows) == total,
            "candidate_count": len(rows),
            "train_passing_candidate_count": sum(
                row["train_passed"] for row in rows
            ),
            "numerically_safe_train_candidate_count": sum(
                row["train_passed"] and row["train_numeric_safe"] for row in rows
            ),
            "development_passing_candidate_count": sum(
                row["train_passed"] and row["development_passed"]
                for row in rows
            ),
            "safe_candidate_count": len(safe),
            "best_train_candidate": best_train,
            "selected_safe_candidate": selected,
            "heldout_reveal_authorized": selected is not None,
            "candidate_model_artifact": "regularized_candidate_models.npz",
            "rows": rows,
        }
        _write_json(self.output_dir / "regularized_train_audit.json", report)
        self._plot_audit(rows)
        return report

    def _plot_audit(self, rows: Sequence[Mapping[str, Any]]) -> None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for rank in self.audit.ranks:
            chosen = [row for row in rows if row["rank"] == rank]
            axes[0].loglog(
                [row["ridge_lambda"] for row in chosen],
                [max(row["train_voltage_rmse_mv"], 1e-12) for row in chosen],
                marker="o", label=f"rank {rank}",
            )
            axes[1].loglog(
                [row["ridge_lambda"] for row in chosen],
                [max(row["coefficient_frobenius_norm"], 1e-12) for row in chosen],
                marker="o", label=f"rank {rank}",
            )
        axes[0].axhline(self.audit.pair_rmse_mv, color="black", linestyle="--")
        axes[0].set(title="Train fit vs ridge", xlabel="ridge lambda", ylabel="RMSE (mV)")
        axes[1].axhline(
            self.audit.maximum_coefficient_frobenius_norm,
            color="black", linestyle="--",
        )
        axes[1].set(title="Coefficient scale", xlabel="ridge lambda", ylabel="Frobenius norm")
        for axis in axes:
            axis.grid(alpha=0.3)
            axis.legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "optimization_audit.png", dpi=160)
        plt.close(figure)

    def reveal_heldout_if_safe(
        self, train_report: Mapping[str, Any]
    ) -> Dict[str, Any]:
        selected = train_report.get("selected_safe_candidate")
        if selected is None:
            report = {
                "schema_version": "05g-heldout-gate-v1",
                "revealed": False,
                "reason": "no candidate passed train, development, coefficient, clipping, and feature-scale gates",
                "heldout_targets_materialized": False,
                "full_training_authorized": False,
            }
            _write_json(self.output_dir / "heldout_gate_report.json", report)
            return report
        key = (float(selected["ridge_lambda"]), int(selected["rank"]))
        candidate = self.candidate_models[key]
        heldout_indices = self.data["heldout"]["indices"]
        heldout_target = self.store.read_state(
            heldout_indices, "t_plus_1", categories=("voltage",)
        ).astype(np.float64).reshape(self.data["heldout"]["base"].shape)
        self.data["heldout"]["target"] = heldout_target
        metrics, safety = self._candidate_metrics(
            "heldout", candidate["bias"], candidate["coefficients"]
        )
        report = {
            "schema_version": "05g-heldout-gate-v1",
            "revealed": True,
            "selected_candidate": selected,
            "heldout_targets_materialized": True,
            "metrics": metrics,
            "safety": safety,
            "passed": self._all_pairs_pass(metrics) and safety["clipped_fraction"] == 0.0,
            "full_training_authorized": False,
        }
        _write_json(self.output_dir / "heldout_gate_report.json", report)
        return report

    def finalize_optimization_audit(
        self,
        scale_report: Mapping[str, Any],
        oracle_report: Mapping[str, Any],
        train_report: Mapping[str, Any],
        heldout_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not oracle_report.get("valid"):
            diagnosis = "MULTIPAIR_TARGET_OR_METRIC_PLUMBING_FAILURE"
        elif train_report.get("selected_safe_candidate") is None:
            if train_report["train_passing_candidate_count"] == 0:
                diagnosis = "REGULARIZED_FROZEN_FEATURES_CANNOT_FIT_TRAIN_SUPPORT"
            elif train_report["numerically_safe_train_candidate_count"] == 0:
                diagnosis = "REGULARIZED_TRAIN_FIT_NUMERICALLY_UNSAFE"
            else:
                diagnosis = "REGULARIZED_SEGMENT_CONDITIONING_FAILS_DEVELOPMENT"
        elif heldout_report.get("passed"):
            diagnosis = "REGULARIZED_SEGMENT_CONDITIONING_GENERALIZES_DIAGNOSTIC"
        else:
            diagnosis = "REGULARIZED_SEGMENT_CONDITIONING_FAILS_HELDOUT"
        next_experiment = {
            "REGULARIZED_SEGMENT_CONDITIONING_GENERALIZES_DIAGNOSTIC":
                "05h_regularized_segment_multistep_micro_rollout",
            "REGULARIZED_FROZEN_FEATURES_CANNOT_FIT_TRAIN_SUPPORT":
                "05h_feature_representation_or_support_revision",
            "REGULARIZED_TRAIN_FIT_NUMERICALLY_UNSAFE":
                "05h_bounded_feature_representation_revision",
            "REGULARIZED_SEGMENT_CONDITIONING_FAILS_DEVELOPMENT":
                "05h_protocol_diverse_regularization_revision",
            "REGULARIZED_SEGMENT_CONDITIONING_FAILS_HELDOUT":
                "05h_protocol_diverse_regularization_revision",
            "MULTIPAIR_TARGET_OR_METRIC_PLUMBING_FAILURE":
                "05h_data_plumbing_audit",
        }[diagnosis]
        report = {
            "schema_version": "05g-final-report-v1",
            "valid": bool(
                scale_report and oracle_report and train_report.get("valid")
                and self.audit_plan.get("valid")
            ),
            "decision": "DIAGNOSTIC_ONLY_NO_FULL_TRAINING",
            "full_training_authorized": False,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "optimization_support": self.audit_plan,
            "feature_scale_audit": scale_report,
            "oracle_controls": oracle_report,
            "regularized_train_audit": train_report,
            "heldout_gate": heldout_report,
            "checkpoint_05b": self.checkpoint_contract,
            "artifact_05c": self.artifact_05c_contract,
            "artifact_05d": self.artifact_05d_contract,
            "artifact_05e": self.artifact_05e_contract,
            "artifact_05f": self.artifact_05f_contract,
            "methodology": {
                "iterative_optimizer_used": False,
                "dual_ridge_float64": True,
                "residual_bounded": True,
                "heldout_targets_sealed_until_safety_gate": True,
                "base_h2_frozen": True,
                "teacher_encoder_updated": False,
                "rollout_performed": False,
                "full_training_path_present": False,
            },
            "next_step": {
                "experiment": next_experiment,
                "full_training_authorized": False,
                "requires_new_notebook": True,
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
            "schema_version": "05g-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
