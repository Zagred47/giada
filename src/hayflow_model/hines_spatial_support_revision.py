"""Notebook-05j-c spatial-context and expanded-support diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_repaired_representation_revision import (
    HinesRepairedRepresentationRevision,
    bounded_target_decode,
    bounded_target_encode,
    dual_ridge_path_predict,
    pair_gate_selection_score,
)
from .hines_layer import require_torch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05JB_ARCHIVE_SHA256 = (
    "b6356194fbc7cd061a8f401b0f7798effb61f10d729749993837769fd6d91bd8"
)
EXPECTED_05JB_INDEX_SHA256 = (
    "ff5811d8c2d21e08daaab754403aa8644e0ef2f0ac045f8732fc8f2b6ecedc0c"
)
EXPECTED_05JB_FINAL_SHA256 = (
    "2448943efa0264dacd04369dce98941e1791728ba0291b66b99a9f27be5fc13e"
)


@dataclass(frozen=True)
class HinesSpatialSupportRevisionConfig:
    original_pair_count: int = 12
    expanded_pair_count: int = 48
    minimum_expanded_pair_count: int = 36
    minimum_protocol_family_count: int = 6
    pca_rank_h2: int = 16
    pca_rank_causal: int = 16
    diffusion_scales: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    diffusion_self_weight: float = 0.5
    contexts: Tuple[str, ...] = ("local", "tree", "tree_global")
    support_roles: Tuple[str, ...] = ("original", "expanded")
    cross_validation_fold_count: int = 6
    ridge_lambdas: Tuple[float, ...] = (
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0,
    )
    branch_fit_weight: float = 1.0
    target_residual_limit_mv: float = 120.0
    target_atanh_margin: float = 1e-6
    input_asinh_reference_z: float = 8.0
    feature_epsilon: float = 1e-6
    maximum_regularized_condition_number: float = 1e8
    maximum_segment_coefficient_l2_norm: float = 1e5
    selection_max_error_weight: float = 0.05
    selection_branch_log_weight: float = 2.0
    material_improvement_fraction: float = 0.20

    def validate(self) -> None:
        if self.expanded_pair_count < self.minimum_expanded_pair_count:
            raise ValueError("expanded support target is below its minimum")
        if self.minimum_expanded_pair_count <= self.original_pair_count:
            raise ValueError("expanded support must be materially larger than original")
        if tuple(self.contexts) != ("local", "tree", "tree_global"):
            raise ValueError("05j-c contexts must be local, tree, and tree_global")
        if tuple(self.support_roles) != ("original", "expanded"):
            raise ValueError("05j-c must compare original and expanded support")
        if not self.diffusion_scales or self.diffusion_scales[0] != 0:
            raise ValueError("diffusion scales must start at zero")
        if tuple(sorted(set(self.diffusion_scales))) != self.diffusion_scales:
            raise ValueError("diffusion scales must be unique and increasing")
        if not 0 < self.diffusion_self_weight < 1:
            raise ValueError("diffusion self weight must lie in (0, 1)")
        if not self.ridge_lambdas or min(self.ridge_lambdas) <= 0:
            raise ValueError("ridge lambdas must be positive")
        if tuple(sorted(set(self.ridge_lambdas))) != self.ridge_lambdas:
            raise ValueError("ridge lambdas must be unique and increasing")
        positive = (
            self.original_pair_count, self.expanded_pair_count,
            self.minimum_protocol_family_count, self.pca_rank_h2,
            self.pca_rank_causal, self.cross_validation_fold_count,
            self.branch_fit_weight, self.target_residual_limit_mv,
            self.target_atanh_margin, self.input_asinh_reference_z,
            self.feature_epsilon, self.maximum_regularized_condition_number,
            self.maximum_segment_coefficient_l2_norm,
            self.selection_branch_log_weight,
        )
        if min(positive) <= 0:
            raise ValueError("05j-c configuration values must be positive")
        if not 0 < self.material_improvement_fraction < 1:
            raise ValueError("material improvement fraction must lie in (0, 1)")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesSpatialSupportRevisionConfig":
        payload = dict(values)
        for name in (
            "diffusion_scales", "contexts", "support_roles", "ridge_lambdas",
        ):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def deterministic_pca_components(values: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Train-only channel PCA with deterministic component signs."""

    matrix = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    effective = min(int(rank), eigenvectors.shape[1])
    components = eigenvectors[:, order[:effective]].T
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return mean, components


def apply_channel_pca(
    values: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    result = np.einsum(
        "nsf,kf->nsk", np.asarray(values, dtype=np.float64) - mean, components,
        optimize=True,
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("channel PCA produced NaN/Inf")
    return result


def axial_tree_diffusion(
    values: np.ndarray,
    parent_ids: Sequence[int],
    axial_conductance: Sequence[float],
    scales: Sequence[int],
    self_weight: float,
) -> np.ndarray:
    """Fixed symmetric axial-neighbour diffusion sampled at registered scales."""

    current = np.asarray(values, dtype=np.float64)
    parents = np.asarray(parent_ids, dtype=np.int64)
    axial = np.asarray(axial_conductance, dtype=np.float64)
    if current.ndim != 3 or current.shape[1] != len(parents):
        raise ValueError("tree diffusion expects [sample, segment, feature]")
    roots = np.flatnonzero(parents == np.arange(len(parents)))
    if len(roots) != 1:
        raise ValueError("tree diffusion requires exactly one self-parented root")
    nonroot = np.flatnonzero(parents != np.arange(len(parents)))
    edge_weight = np.maximum(axial[nonroot], 0.0)
    degree = np.zeros(len(parents), dtype=np.float64)
    degree[nonroot] += edge_weight
    np.add.at(degree, parents[nonroot], edge_weight)
    degree = np.maximum(degree, 1e-12)
    requested = tuple(int(value) for value in scales)
    outputs = []
    maximum = max(requested)
    for step in range(maximum + 1):
        if step in requested:
            outputs.append(current.copy())
        if step == maximum:
            break
        neighbour = np.zeros_like(current)
        weighted_parent = (
            current[:, parents[nonroot]] * edge_weight[None, :, None]
        )
        neighbour[:, nonroot] += weighted_parent
        weighted_child = current[:, nonroot] * edge_weight[None, :, None]
        np.add.at(
            neighbour.transpose(1, 0, 2),
            parents[nonroot],
            weighted_child.transpose(1, 0, 2),
        )
        mixed = neighbour / degree[None, :, None]
        current = float(self_weight) * current + (1.0 - float(self_weight)) * mixed
    result = np.concatenate(outputs, axis=-1)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("tree diffusion produced NaN/Inf")
    return result


def region_global_context(values: np.ndarray, region_ids: Sequence[int]) -> np.ndarray:
    """Broadcast train-causal summaries of every morphology region to all segments."""

    source = np.asarray(values, dtype=np.float64)
    regions = np.asarray(region_ids, dtype=np.int64)
    if source.ndim != 3 or source.shape[1] != len(regions):
        raise ValueError("region context expects [sample, segment, feature]")
    pooled = []
    for region in sorted(set(regions.tolist())):
        chosen = source[:, regions == region]
        pooled.append(chosen.sum(axis=1) / math.sqrt(chosen.shape[1]))
    global_values = np.concatenate(pooled, axis=-1)
    return np.broadcast_to(
        global_values[:, None, :],
        (source.shape[0], source.shape[1], global_values.shape[-1]),
    ).copy()


def deterministic_pair_folds(pair_count: int, fold_count: int) -> List[np.ndarray]:
    if pair_count < fold_count or fold_count < 2:
        raise ValueError("invalid pair-fold configuration")
    return [
        np.arange(fold, pair_count, fold_count, dtype=np.int64)
        for fold in range(fold_count)
    ]


class HinesSpatialSupportRevision(HinesRepairedRepresentationRevision):
    """05j-c morphology-context and deterministic expanded-support audit."""

    def __init__(
        self,
        *args: Any,
        spatial_config: HinesSpatialSupportRevisionConfig,
        artifact_05jb_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        spatial_config.validate()
        self.spatial = spatial_config
        self.artifact_05jb_source = Path(artifact_05jb_source).resolve()
        self.artifact_05jb_contract: Dict[str, Any] = {}
        self.expanded_support: Dict[str, Any] = {}
        self.expanded_roles: Dict[str, Any] = {}
        self._design_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _read_verified_05jb(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05jb_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05JB_ARCHIVE_SHA256:
                raise RuntimeError("05j-b archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[: -len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JB_INDEX_SHA256:
                    raise RuntimeError("05j-b artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                for row in index["artifacts"]:
                    member = root + str(row["path"]).replace("\\", "/")
                    if member not in names:
                        raise RuntimeError(f"missing indexed 05j-b member: {row['path']}")
                    payload = archive.read(member)
                    if (
                        hashlib.sha256(payload).hexdigest() != row["sha256"]
                        or len(payload) != int(row["size_bytes"])
                    ):
                        raise RuntimeError(f"05j-b indexed member mismatch: {row['path']}")
                final_bytes = archive.read(root + "final_report.json")
            kind = "original_zip"
        elif source.is_dir():
            indices = [
                path for path in source.rglob("artifact_index.json")
                if (path.parent / "repaired_representation_revision_config.json").is_file()
            ]
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05j-b artifact index")
            index_bytes = indices[0].read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JB_INDEX_SHA256:
                raise RuntimeError("extracted 05j-b artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            root = indices[0].parent
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if (
                    not path.is_file()
                    or sha256_file(path) != row["sha256"]
                    or path.stat().st_size != int(row["size_bytes"])
                ):
                    raise RuntimeError(f"extracted 05j-b member mismatch: {row['path']}")
            final_bytes = (root / "final_report.json").read_bytes()
            archive_hash = None
            kind = "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05j-b source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JB_FINAL_SHA256:
            raise RuntimeError("05j-b final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05JB_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05JB_FINAL_SHA256,
            "verified_member_count": len(index["artifacts"]),
            "all_indexed_members_verified": (
                len(index["artifacts"]) == int(index["artifact_count"])
            ),
        }

    def prepare_spatial_support_revision(self) -> Dict[str, Any]:
        base = self.prepare_repaired_representation_revision()
        report, contract = self._read_verified_05jb()
        blockers = []
        if report.get("diagnosis") != "REPAIRED_FEATURE_SURFACE_FAILS_REGULARIZED_TRAIN_FIT":
            blockers.append(f"unexpected 05j-b diagnosis: {report.get('diagnosis')}")
        if report.get("representation_revision_passed") is not False:
            blockers.append("05j-b did not record the required failed revision")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05j-b dataset fingerprint mismatch")
        if report.get("methodology", {}).get("development_used_for_model_selection"):
            blockers.append("05j-b used development for model selection")
        if report.get("methodology", {}).get("rollout_performed"):
            blockers.append("05j-b unexpectedly performed rollout")
        heldout = report.get("heldout_contract", {})
        if heldout.get("inputs_extracted") or heldout.get("boundary_targets_materialized"):
            blockers.append("05j-b held-out contract was not sealed")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05j-b artifact verification is incomplete")
        if self.spatial.original_pair_count != self.recheck.required_train_pair_count:
            blockers.append("05j-c original support count differs from 05j")
        if not math.isclose(
            self.spatial.branch_fit_weight, self.forensics.branch_loss_weight
        ):
            blockers.append("05j-c changed the registered branch weight")
        if not math.isclose(
            self.spatial.target_residual_limit_mv, self.forensics.residual_limit_mv
        ):
            blockers.append("05j-c changed the registered residual limit")
        if blockers:
            raise RuntimeError(f"05j-c provenance blockers: {blockers}")
        self.artifact_05jb_contract = contract
        payload = {
            "schema_version": "05j-c-spatial-support-config-v1",
            "spatial_support": asdict(self.spatial),
            "artifact_05jb": contract,
            "diagnostic_axes": ["support_size", "morphology_context"],
            "selection_roles": ["train_grouped_pair_cross_validation"],
            "development_used_for_selection": False,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "spatial_support_revision_config.json", payload)
        return {**base, **payload}

    def build_expanded_train_support(self) -> Dict[str, Any]:
        original = [dict(row) for row in self.audit_plan["selected_pairs"]]
        development = [int(value) for value in self.audit_plan["development_pair"]]
        development_episodes = {
            self._episode_identity(index)[1] for index in development
        }
        original_keys = {
            (int(row["left_index"]), int(row["right_index"])) for row in original
        }
        selected = list(original)
        used_episodes = set(development_episodes)
        for row in selected:
            used_episodes.update((row["left_episode_id"], row["right_episode_id"]))
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
        cursors = {family: 0 for family in by_family}
        while len(selected) < self.spatial.expanded_pair_count:
            added = False
            for family in sorted(by_family):
                rows = by_family[family]
                while cursors[family] < len(rows):
                    row = dict(rows[cursors[family]])
                    cursors[family] += 1
                    key = (int(row["left_index"]), int(row["right_index"]))
                    episodes = {row["left_episode_id"], row["right_episode_id"]}
                    if key in original_keys or episodes & used_episodes:
                        continue
                    selected.append(row)
                    used_episodes.update(episodes)
                    added = True
                    break
                if len(selected) == self.spatial.expanded_pair_count:
                    break
            if not added:
                break
        family_counts: Dict[str, int] = {}
        for row in selected:
            family = row.get("protocol_family") or " <> ".join(sorted({
                self._protocol_family(row["left_index"]),
                self._protocol_family(row["right_index"]),
            }))
            row["protocol_family"] = family
            family_counts[family] = family_counts.get(family, 0) + 1
        selected_episodes = {
            value for row in selected
            for value in (row["left_episode_id"], row["right_episode_id"])
        }
        blockers = []
        if len(original) != self.spatial.original_pair_count:
            blockers.append("original support does not contain 12 pairs")
        if len(selected) < self.spatial.minimum_expanded_pair_count:
            blockers.append("insufficient disjoint expanded train support")
        if len(family_counts) < self.spatial.minimum_protocol_family_count:
            blockers.append("expanded support lacks protocol-family diversity")
        if selected_episodes & development_episodes:
            blockers.append("expanded support overlaps development episodes")
        if any(row.get("split") != "train" for row in selected):
            blockers.append("non-train pair entered expanded support")
        support_hash = hashlib.sha256(json.dumps(
            {"original": original, "expanded": selected, "development": development},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        report = {
            "schema_version": "05j-c-expanded-support-v1",
            "valid": not blockers,
            "blockers": blockers,
            "candidate_pair_count": len(candidates),
            "original_pair_count": len(original),
            "expanded_pair_count": len(selected),
            "added_pair_count": len(selected) - len(original),
            "protocol_family_count": len(family_counts),
            "protocol_family_counts": dict(sorted(family_counts.items())),
            "development_pair": development,
            "development_episode_overlap": sorted(
                selected_episodes & development_episodes
            ),
            "all_pairs_from_train": all(row.get("split") == "train" for row in selected),
            "all_pairs_episode_disjoint": len(selected_episodes) == 2 * len(selected),
            "support_sha256": support_hash,
            "original_pairs": original,
            "expanded_pairs": selected,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
        }
        _write_json(self.output_dir / "expanded_train_support.json", report)
        write_parquet(self.output_dir / "expanded_train_support.parquet", [
            {
                "support_role": "original" if position < len(original) else "added",
                "pair_position": position,
                **row,
            }
            for position, row in enumerate(selected)
        ])
        self.expanded_support = report
        if blockers:
            raise RuntimeError(f"05j-c support blockers: {blockers}")
        return report

    def prepare_expanded_spatial_features(self) -> Dict[str, Any]:
        if not self.expanded_support.get("valid"):
            raise RuntimeError("build_expanded_train_support() must run first")
        require_torch()
        expanded_indices = np.asarray(
            self._pair_indices(self.expanded_support["expanded_pairs"]),
            dtype=np.int64,
        )
        development_indices = np.asarray(
            self.expanded_support["development_pair"], dtype=np.int64
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = self._load_h2_checkpoint(device)
        model.eval()
        progress = Progress("05j-c frozen features", 2)
        self.expanded_roles = {
            "expanded": self._extract_recheck_role(model, expanded_indices),
        }
        progress.update(1, "expanded train")
        self.expanded_roles["development"] = self._extract_recheck_role(
            model, development_indices
        )
        progress.update(2, "development")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        original_indices = np.asarray(
            self._pair_indices(self.expanded_support["original_pairs"]),
            dtype=np.int64,
        )
        position = {int(index): pos for pos, index in enumerate(expanded_indices)}
        original_positions = np.asarray([position[int(index)] for index in original_indices])
        report = {
            "schema_version": "05j-c-expanded-features-v1",
            "valid": True,
            "original_pair_count": int(len(original_positions) // 2),
            "expanded_pair_count": int(len(expanded_indices) // 2),
            "development_pair_count": int(len(development_indices) // 2),
            "original_is_exact_prefix": bool(np.array_equal(
                original_positions, np.arange(len(original_positions))
            )),
            "expanded_indices_sha256": hashlib.sha256(
                expanded_indices.tobytes()
            ).hexdigest(),
            "development_indices_sha256": hashlib.sha256(
                development_indices.tobytes()
            ).hexdigest(),
            "normalization_fit_roles": ["candidate_train_support"],
            "development_values_used_to_fit": False,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
        }
        self.expanded_roles["original_positions"] = original_positions
        _write_json(self.output_dir / "expanded_spatial_features.json", report)
        return report

    def _support_role(self, support: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        expanded = self.expanded_roles["expanded"]
        development = self.expanded_roles["development"]
        if support == "expanded":
            return expanded, development
        if support != "original":
            raise ValueError(f"unknown support role: {support}")
        positions = self.expanded_roles["original_positions"]
        return ({
            key: value[positions] if isinstance(value, np.ndarray) and len(value) == len(expanded["base"]) else value
            for key, value in expanded.items()
        }, development)

    def _sketch_surface(
        self,
        train_values: np.ndarray,
        development_values: np.ndarray,
        rank: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        mean = train_values.mean(axis=(0, 1), keepdims=True)
        scale = np.maximum(
            train_values.std(axis=(0, 1), keepdims=True), self.spatial.feature_epsilon
        )
        train_z = (train_values - mean) / scale
        development_z = (development_values - mean) / scale
        denominator = np.arcsinh(self.spatial.input_asinh_reference_z)
        train_transformed = np.arcsinh(train_z) / denominator
        development_transformed = np.arcsinh(development_z) / denominator
        channel_mean, components = deterministic_pca_components(
            train_transformed, rank
        )
        train_sketch = apply_channel_pca(
            train_transformed, channel_mean, components
        )
        development_sketch = apply_channel_pca(
            development_transformed, channel_mean, components
        )
        explained = np.var(train_sketch.reshape(-1, train_sketch.shape[-1]), axis=0)
        return train_sketch, development_sketch, {
            "requested_rank": int(rank),
            "effective_rank": int(components.shape[0]),
            "maximum_absolute_development_standardized": float(
                np.max(np.abs(development_z))
            ),
            "maximum_absolute_train_sketch": float(np.max(np.abs(train_sketch))),
            "maximum_absolute_development_sketch": float(
                np.max(np.abs(development_sketch))
            ),
            "retained_variance": float(explained.sum()),
        }

    def _candidate_design(self, support: str, context: str) -> Dict[str, Any]:
        key = (support, context)
        if key in self._design_cache:
            return self._design_cache[key]
        train, development = self._support_role(support)
        h2_train, h2_dev, h2_report = self._sketch_surface(
            np.asarray(train["h2_raw"]), np.asarray(development["h2_raw"]),
            self.spatial.pca_rank_h2,
        )
        causal_train, causal_dev, causal_report = self._sketch_surface(
            np.asarray(train["causal_raw"]), np.asarray(development["causal_raw"]),
            self.spatial.pca_rank_causal,
        )
        local_train = np.concatenate([h2_train, causal_train], axis=-1)
        local_dev = np.concatenate([h2_dev, causal_dev], axis=-1)
        if context == "local":
            spatial_train, spatial_dev = local_train, local_dev
        else:
            parents = np.asarray(self.arrays["parent_ids"], dtype=np.int64)
            axial = np.asarray(
                self.arrays["axial_conductance_to_parent_us"], dtype=np.float64
            )
            tree_train = axial_tree_diffusion(
                local_train, parents, axial, self.spatial.diffusion_scales,
                self.spatial.diffusion_self_weight,
            )
            tree_dev = axial_tree_diffusion(
                local_dev, parents, axial, self.spatial.diffusion_scales,
                self.spatial.diffusion_self_weight,
            )
            spatial_train, spatial_dev = tree_train, tree_dev
            if context == "tree_global":
                global_train = region_global_context(
                    causal_train, self.arrays["segment_region_ids"]
                )
                global_dev = region_global_context(
                    causal_dev, self.arrays["segment_region_ids"]
                )
                spatial_train = np.concatenate([spatial_train, global_train], axis=-1)
                spatial_dev = np.concatenate([spatial_dev, global_dev], axis=-1)
            elif context != "tree":
                raise ValueError(f"unknown spatial context: {context}")
        voltage_train = np.asarray(train["voltage_t"])[..., None] / 100.0
        voltage_dev = np.asarray(development["voltage_t"])[..., None] / 100.0
        base_train = np.asarray(train["base"])[..., None] / 100.0
        base_dev = np.asarray(development["base"])[..., None] / 100.0
        raw_train = np.concatenate(
            [voltage_train, base_train, spatial_train], axis=-1
        )
        raw_dev = np.concatenate([voltage_dev, base_dev, spatial_dev], axis=-1)
        mean = raw_train.mean(axis=(0, 1), keepdims=True)
        scale = np.maximum(
            raw_train.std(axis=(0, 1), keepdims=True), self.spatial.feature_epsilon
        )
        train_z = (raw_train - mean) / scale
        dev_z = (raw_dev - mean) / scale
        denominator = np.arcsinh(self.spatial.input_asinh_reference_z)
        train_design = np.arcsinh(train_z) / denominator
        dev_design = np.arcsinh(dev_z) / denominator
        result = {
            "train_role": train,
            "development_role": development,
            "train": train_design,
            "development": dev_design,
            "report": {
                "support": support,
                "context": context,
                "pair_count": int(len(train_design) // 2),
                "feature_width": int(train_design.shape[-1]),
                "h2_sketch": h2_report,
                "causal_sketch": causal_report,
                "maximum_absolute_train_design": float(
                    np.max(np.abs(train_design))
                ),
                "maximum_absolute_development_design": float(
                    np.max(np.abs(dev_design))
                ),
                "nonfinite_count": int(
                    np.sum(~np.isfinite(train_design))
                    + np.sum(~np.isfinite(dev_design))
                ),
                "normalization_fit_roles": [support],
                "development_values_used_to_fit": False,
            },
        }
        self._design_cache[key] = result
        return result

    def run_spatial_design_audit(self) -> Dict[str, Any]:
        if not self.expanded_roles:
            raise RuntimeError("prepare_expanded_spatial_features() must run first")
        rows = []
        total = len(self.spatial.support_roles) * len(self.spatial.contexts)
        progress = Progress("05j-c spatial designs", total)
        completed = 0
        for support in self.spatial.support_roles:
            for context in self.spatial.contexts:
                design = self._candidate_design(support, context)
                rows.append(design["report"])
                completed += 1
                progress.update(
                    completed,
                    f"{support}/{context} width={design['report']['feature_width']}",
                )
        report = {
            "schema_version": "05j-c-spatial-design-audit-v1",
            "valid": all(row["nonfinite_count"] == 0 for row in rows),
            "rows": rows,
            "development_values_used_to_fit": False,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
        }
        _write_json(self.output_dir / "spatial_design_audit.json", report)
        write_parquet(self.output_dir / "spatial_design_summary.parquet", [{
            key: value for key, value in row.items()
            if key not in {"h2_sketch", "causal_sketch"}
        } for row in rows])
        return report

    def _metrics(self, role: Mapping[str, Any], residual: np.ndarray) -> Dict[str, Any]:
        prediction = np.asarray(role["base"]) + np.asarray(residual)
        return self._pair_set_metrics(
            prediction.reshape(-1, 2, self.layout.segment_count),
            np.asarray(role["target"]).reshape(-1, 2, self.layout.segment_count),
        )

    def _cross_validated_candidate(self, support: str, context: str) -> Dict[str, Any]:
        design = self._candidate_design(support, context)
        train_x = design["train"]
        development_x = design["development"]
        train_role = design["train_role"]
        development_role = design["development_role"]
        target = np.asarray(train_role["target"]) - np.asarray(train_role["base"])
        encoded = bounded_target_encode(
            target, self.spatial.target_residual_limit_mv,
            self.spatial.target_atanh_margin,
        )
        pair_count = len(train_x) // 2
        folds = deterministic_pair_folds(
            pair_count, self.spatial.cross_validation_fold_count
        )
        lambdas = self.spatial.ridge_lambdas
        oof = np.zeros((len(lambdas), *encoded.shape), dtype=np.float64)
        fold_rows = []
        progress = Progress(f"05j-c CV {support}/{context}", len(folds))
        for fold_index, held_pairs in enumerate(folds):
            held = np.sort(np.concatenate([2 * held_pairs, 2 * held_pairs + 1]))
            keep = np.ones(len(train_x), dtype=bool)
            keep[held] = False
            path, _ = dual_ridge_path_predict(
                train_x[keep], encoded[keep], train_x[held], lambdas,
                pair_branch_weight=self.spatial.branch_fit_weight,
            )
            oof[:, held] = path
            fold_rows.append({
                "fold": fold_index,
                "held_pair_positions": held_pairs.tolist(),
                "held_sample_count": int(len(held)),
                "fit_sample_count": int(np.sum(keep)),
            })
            progress.update(fold_index + 1, f"held pairs={len(held_pairs)}")
        ladder = []
        for index, ridge in enumerate(lambdas):
            residual = bounded_target_decode(
                oof[index], self.spatial.target_residual_limit_mv
            )
            metrics = self._metrics(train_role, residual)
            ladder.append({
                "ridge_lambda": float(ridge),
                "selection_score": pair_gate_selection_score(
                    metrics,
                    max_error_weight=self.spatial.selection_max_error_weight,
                    branch_log_weight=self.spatial.selection_branch_log_weight,
                ),
                "cross_validation": metrics,
            })
        selected = min(ladder, key=lambda row: row["selection_score"])
        selected_lambda = float(selected["ridge_lambda"])
        full, diagnostics = dual_ridge_path_predict(
            train_x, encoded, np.concatenate([train_x, development_x]),
            [selected_lambda], pair_branch_weight=self.spatial.branch_fit_weight,
        )
        decoded = bounded_target_decode(
            full[0], self.spatial.target_residual_limit_mv
        )
        train_residual = decoded[: len(train_x)]
        development_residual = decoded[len(train_x):]
        cv_metrics = selected["cross_validation"]
        train_metrics = self._metrics(train_role, train_residual)
        development_metrics = self._metrics(development_role, development_residual)
        stability = diagnostics[0]
        stable = bool(
            stability["maximum_regularized_condition_number"]
            <= self.spatial.maximum_regularized_condition_number
            and stability["maximum_segment_coefficient_l2_norm"]
            <= self.spatial.maximum_segment_coefficient_l2_norm
        )
        cv_passed = self._pair_passes(cv_metrics)
        train_passed = self._pair_passes(train_metrics)
        development_passed = self._pair_passes(development_metrics)
        return {
            "support": support,
            "context": context,
            "pair_count": pair_count,
            "feature_width": int(train_x.shape[-1]),
            "fold_count": len(folds),
            "folds": fold_rows,
            "selected_ridge_lambda": selected_lambda,
            "selection_score": float(selected["selection_score"]),
            "selection_ladder": ladder,
            "cross_validation": cv_metrics,
            "train_fit": train_metrics,
            "development": development_metrics,
            "stability": stability,
            "cross_validation_passed": cv_passed,
            "train_passed": train_passed,
            "development_passed": development_passed,
            "numerically_stable": stable,
            "candidate_passed": bool(
                cv_passed and train_passed and development_passed and stable
            ),
            "development_used_for_selection": False,
            "heldout_candidate_inference_performed": False,
        }

    def run_spatial_support_controls(self) -> Dict[str, Any]:
        candidates = []
        total = len(self.spatial.support_roles) * len(self.spatial.contexts)
        progress = Progress("05j-c spatial/support controls", total)
        completed = 0
        for support in self.spatial.support_roles:
            for context in self.spatial.contexts:
                row = self._cross_validated_candidate(support, context)
                candidates.append(row)
                completed += 1
                progress.update(
                    completed,
                    f"{support}/{context} cv={row['cross_validation']['aggregate_voltage_rmse_mv']:.3g} "
                    f"dev={row['development']['aggregate_voltage_rmse_mv']:.3g}",
                )
        report = {
            "schema_version": "05j-c-spatial-support-controls-v1",
            "valid": len(candidates) == total,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "passing_candidates": [
                f"{row['support']}:{row['context']}"
                for row in candidates if row["candidate_passed"]
            ],
            "selection_roles": ["train_grouped_pair_cross_validation"],
            "development_used_for_selection": False,
            "base_h2_frozen": True,
            "teacher_encoder_updated": False,
            "heldout_inputs_extracted": False,
            "heldout_targets_materialized": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "spatial_support_controls.json", report)
        write_parquet(self.output_dir / "spatial_support_summary.parquet", [{
            "support": row["support"],
            "context": row["context"],
            "pair_count": row["pair_count"],
            "feature_width": row["feature_width"],
            "selected_ridge_lambda": row["selected_ridge_lambda"],
            "cv_rmse_mv": row["cross_validation"]["aggregate_voltage_rmse_mv"],
            "train_rmse_mv": row["train_fit"]["aggregate_voltage_rmse_mv"],
            "development_rmse_mv": row["development"]["aggregate_voltage_rmse_mv"],
            "development_retention": row["development"]["median_branching_retention"],
            "candidate_passed": row["candidate_passed"],
        } for row in candidates])
        return report

    @staticmethod
    def _candidate_by(
        rows: Sequence[Mapping[str, Any]], support: str, context: str
    ) -> Mapping[str, Any]:
        matches = [
            row for row in rows
            if row["support"] == support and row["context"] == context
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {support}/{context} candidate")
        return matches[0]

    def finalize_spatial_support_revision(
        self,
        support_report: Mapping[str, Any],
        feature_report: Mapping[str, Any],
        design_report: Mapping[str, Any],
        controls_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        rows = controls_report["candidates"]
        passing = [row for row in rows if row["candidate_passed"]]
        original_local = self._candidate_by(rows, "original", "local")
        original_tree = self._candidate_by(rows, "original", "tree_global")
        expanded_local = self._candidate_by(rows, "expanded", "local")
        expanded_tree = self._candidate_by(rows, "expanded", "tree_global")
        def improvement(baseline: float, candidate: float) -> float:
            return float((baseline - candidate) / max(abs(baseline), 1e-12))
        topology_cv_gain = improvement(
            expanded_local["cross_validation"]["aggregate_voltage_rmse_mv"],
            expanded_tree["cross_validation"]["aggregate_voltage_rmse_mv"],
        )
        topology_dev_gain = improvement(
            expanded_local["development"]["aggregate_voltage_rmse_mv"],
            expanded_tree["development"]["aggregate_voltage_rmse_mv"],
        )
        support_cv_gain = improvement(
            original_tree["cross_validation"]["aggregate_voltage_rmse_mv"],
            expanded_tree["cross_validation"]["aggregate_voltage_rmse_mv"],
        )
        support_dev_gain = improvement(
            original_tree["development"]["aggregate_voltage_rmse_mv"],
            expanded_tree["development"]["aggregate_voltage_rmse_mv"],
        )
        material = self.spatial.material_improvement_fraction
        topology_material = bool(
            topology_cv_gain >= material and topology_dev_gain >= material
        )
        support_material = bool(
            support_cv_gain >= material and support_dev_gain >= material
        )
        passed = bool(passing and support_report["valid"] and design_report["valid"])
        if passed:
            diagnosis = "SPATIAL_SUPPORT_REVISION_RESCUES_MAPPING"
        elif topology_material:
            diagnosis = (
                "NONLOCAL_CONTEXT_HELPS_BUT_MAPPING_REMAINS_BELOW_GATE"
                if support_material
                else "NONLOCAL_CONTEXT_HELPS_WITHOUT_SUPPORT_RESCUE"
            )
        elif support_material:
            diagnosis = "EXPANDED_SUPPORT_HELPS_WITHOUT_TOPOLOGY_RESCUE"
        elif any(
            value >= material for value in (
                topology_cv_gain, topology_dev_gain,
                support_cv_gain, support_dev_gain,
            )
        ):
            diagnosis = "FACTORIAL_EFFECTS_INCONSISTENT_BETWEEN_CV_AND_DEVELOPMENT"
        else:
            diagnosis = "SPATIAL_SUPPORT_REVISION_DOES_NOT_RESCUE_MAPPING"
        report = {
            "schema_version": "05j-c-final-report-v1",
            "valid": True,
            "decision": "TRAIN_ONLY_SPATIAL_SUPPORT_DIAGNOSTIC",
            "diagnosis": diagnosis,
            "spatial_support_revision_passed": passed,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05jb": self.artifact_05jb_contract,
            "expanded_support": dict(support_report),
            "expanded_features": dict(feature_report),
            "spatial_design": dict(design_report),
            "spatial_support_controls": dict(controls_report),
            "factorial_diagnosis": {
                "material_improvement_fraction": material,
                "topology_cross_validation_rmse_improvement_fraction": topology_cv_gain,
                "topology_development_rmse_improvement_fraction": topology_dev_gain,
                "expanded_support_cross_validation_rmse_improvement_fraction": support_cv_gain,
                "expanded_support_development_rmse_improvement_fraction": support_dev_gain,
                "topology_material_on_both_roles": topology_material,
                "expanded_support_material_on_both_roles": support_material,
            },
            "passing_candidates": [
                {k: row[k] for k in (
                    "support", "context", "selected_ridge_lambda",
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
                "exact_05jb_failure_verified": True,
                "original_12_pairs_preserved": True,
                "expanded_pairs_from_train_only": True,
                "development_used_for_model_selection": False,
                "candidate_selection": "train_grouped_pair_cross_validation",
                "base_h2_frozen": True,
                "teacher_encoder_updated": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": (
                    "05k_repaired_representation_micro_rollout"
                    if passed
                    else "05j_d_trainable_topology_decoder_micro_canary"
                    if diagnosis.startswith("NONLOCAL_CONTEXT_HELPS")
                    else "05j_d_representation_architecture_reassessment"
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
            "schema_version": "05j-c-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
