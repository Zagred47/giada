"""Notebook-05j-d trainable topology decoder micro-canary."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_repaired_representation_revision import (
    bounded_target_decode,
    bounded_target_encode,
    dual_ridge_path_predict,
    pair_gate_selection_score,
)
from .hines_spatial_support_revision import (
    HinesSpatialSupportRevision,
    apply_channel_pca,
    axial_tree_diffusion,
    deterministic_pair_folds,
    deterministic_pca_components,
)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


EXPECTED_05JC_ARCHIVE_SHA256 = (
    "ad42576074fd52d836184a7bee08b655e927e3995285d5a27b78e8ae76b6ef36"
)
EXPECTED_05JC_INDEX_SHA256 = (
    "0d47899e86efaaee27f885167afb79f379b702cb60881a2e1af82aef92b620f7"
)
EXPECTED_05JC_FINAL_SHA256 = (
    "b901bf72b4d2f83d55ecdca98995b0f4a5ac1aeb8a107cd6861a41944dd61c1f"
)


@dataclass(frozen=True)
class HinesTrainableTopologyCanaryConfig:
    fit_pair_count: int = 36
    calibration_pair_count: int = 12
    minimum_protocol_family_count: int = 6
    families: Tuple[str, ...] = ("direct_tree", "ridge_corrected_tree")
    seeds: Tuple[int, ...] = (17, 29, 43)
    hidden_width: int = 96
    segment_embedding_dim: int = 16
    epochs: int = 1200
    evaluation_interval: int = 20
    patience: int = 240
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    target_residual_limit_mv: float = 120.0
    target_atanh_margin: float = 1e-6
    error_scale_mv: float = 20.0
    branch_loss_weight: float = 1.0
    tail_loss_weight: float = 0.25
    tail_fraction: float = 0.05
    ridge_cross_validation_folds: int = 6
    ridge_lambdas: Tuple[float, ...] = (
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0,
    )
    selection_max_error_weight: float = 0.05
    selection_branch_log_weight: float = 2.0
    minimum_passing_seeds: int = 2
    material_improvement_fraction: float = 0.20

    def validate(self) -> None:
        if self.fit_pair_count + self.calibration_pair_count != 48:
            raise ValueError("05j-d must partition the registered 48-pair support")
        if tuple(self.families) != ("direct_tree", "ridge_corrected_tree"):
            raise ValueError("05j-d families must be direct and ridge-corrected tree")
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 3:
            raise ValueError("05j-d requires at least three distinct seeds")
        if not 1 <= self.minimum_passing_seeds <= len(self.seeds):
            raise ValueError("invalid robust seed gate")
        positive = (
            self.fit_pair_count, self.calibration_pair_count,
            self.minimum_protocol_family_count, self.hidden_width,
            self.segment_embedding_dim, self.epochs, self.evaluation_interval,
            self.patience, self.learning_rate, self.gradient_clip_norm,
            self.target_residual_limit_mv, self.target_atanh_margin,
            self.error_scale_mv, self.branch_loss_weight,
            self.ridge_cross_validation_folds,
        )
        if min(positive) <= 0:
            raise ValueError("05j-d configuration values must be positive")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail fraction must lie in (0, 1]")
        if self.tail_loss_weight < 0 or self.weight_decay < 0:
            raise ValueError("loss and decay weights cannot be negative")
        if not 0 < self.material_improvement_fraction < 1:
            raise ValueError("material improvement must lie in (0, 1)")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesTrainableTopologyCanaryConfig":
        payload = dict(values)
        for name in ("families", "seeds", "ridge_lambdas"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def deterministic_stratified_pair_split(
    rows: Sequence[Mapping[str, Any]], calibration_count: int
) -> Dict[str, Any]:
    """Deterministically reserve a family-stratified calibration subset."""

    family_positions: Dict[str, List[int]] = {}
    for position, row in enumerate(rows):
        family_positions.setdefault(str(row["protocol_family"]), []).append(position)
    if calibration_count < len(family_positions):
        raise ValueError("calibration cannot cover every protocol family")
    total = len(rows)
    exact = {
        family: len(positions) * float(calibration_count) / total
        for family, positions in family_positions.items()
    }
    quotas = {family: max(1, int(math.floor(value))) for family, value in exact.items()}
    while sum(quotas.values()) < calibration_count:
        eligible = [
            family for family, positions in family_positions.items()
            if quotas[family] < len(positions) - 1
        ]
        if not eligible:
            raise ValueError("cannot allocate the requested calibration subset")
        chosen = max(
            eligible,
            key=lambda family: (
                exact[family] - quotas[family], len(family_positions[family]), family
            ),
        )
        quotas[chosen] += 1
    while sum(quotas.values()) > calibration_count:
        eligible = [family for family in quotas if quotas[family] > 1]
        if not eligible:
            raise ValueError("cannot reduce calibration quotas")
        chosen = min(
            eligible,
            key=lambda family: (exact[family] - quotas[family], family),
        )
        quotas[chosen] -= 1
    calibration = sorted(
        position
        for family, positions in family_positions.items()
        for position in positions[-quotas[family]:]
    )
    calibration_set = set(calibration)
    fit = [position for position in range(total) if position not in calibration_set]
    return {
        "fit_pair_positions": fit,
        "calibration_pair_positions": calibration,
        "family_counts": {key: len(value) for key, value in sorted(family_positions.items())},
        "calibration_family_quotas": dict(sorted(quotas.items())),
        "fit_family_count": len({str(rows[position]["protocol_family"]) for position in fit}),
        "calibration_family_count": len({str(rows[position]["protocol_family"]) for position in calibration}),
    }


if torch is not None:
    class TrainableTopologyResidualHead(nn.Module):
        """Small shared nonlinear decoder over fixed multiscale tree features."""

        def __init__(
            self, feature_width: int, segment_count: int, hidden_width: int,
            segment_embedding_dim: int, residual_limit_mv: float,
        ) -> None:
            super().__init__()
            self.residual_limit_mv = float(residual_limit_mv)
            self.segment_embedding = nn.Embedding(segment_count, segment_embedding_dim)
            self.network = nn.Sequential(
                nn.Linear(feature_width + segment_embedding_dim, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

        def forward(self, features: Any, baseline_residual: Any = None) -> Any:
            sample_count, segment_count, _ = features.shape
            segment_ids = torch.arange(segment_count, device=features.device)
            embedding = self.segment_embedding(segment_ids)[None].expand(sample_count, -1, -1)
            correction = self.network(torch.cat([features, embedding], dim=-1)).squeeze(-1)
            if baseline_residual is None:
                logits = correction
            else:
                bound = 1.0 - 1e-6
                ratio = torch.clamp(
                    baseline_residual / self.residual_limit_mv, -bound, bound
                )
                logits = torch.atanh(ratio) + correction
            return self.residual_limit_mv * torch.tanh(logits)
else:  # pragma: no cover
    class TrainableTopologyResidualHead:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def _sample_positions(pair_positions: Sequence[int]) -> np.ndarray:
    return np.sort(np.concatenate([
        2 * np.asarray(pair_positions, dtype=np.int64),
        2 * np.asarray(pair_positions, dtype=np.int64) + 1,
    ]))


def _subset_role(role: Mapping[str, Any], positions: np.ndarray) -> Dict[str, Any]:
    sample_count = len(role["base"])
    return {
        key: value[positions]
        if isinstance(value, np.ndarray) and len(value) == sample_count else value
        for key, value in role.items()
    }


class HinesTrainableTopologyCanary(HinesSpatialSupportRevision):
    """Train-only decoder canary after the positive 05j-c topology diagnostic."""

    def __init__(
        self, *args: Any, topology_config: HinesTrainableTopologyCanaryConfig,
        artifact_05jc_source: Path, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        topology_config.validate()
        self.topology = topology_config
        self.artifact_05jc_source = Path(artifact_05jc_source).resolve()
        self.artifact_05jc_contract: Dict[str, Any] = {}
        self.topology_roles: Dict[str, Dict[str, Any]] = {}
        self.topology_designs: Dict[str, np.ndarray] = {}
        self.ridge_predictions: Dict[str, np.ndarray] = {}

    def _read_verified_05jc(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05jc_source
        if source.is_file():
            if sha256_file(source) != EXPECTED_05JC_ARCHIVE_SHA256:
                raise RuntimeError("05j-c archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[:-len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                index = json.loads(index_bytes)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JC_INDEX_SHA256:
                    raise RuntimeError("05j-c artifact index SHA-256 mismatch")
                for row in index["artifacts"]:
                    payload = archive.read(root + str(row["path"]).replace("\\", "/"))
                    if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05j-c indexed member mismatch: {row['path']}")
                final_bytes = archive.read(root + "final_report.json")
            archive_hash, kind = EXPECTED_05JC_ARCHIVE_SHA256, "original_zip"
        elif source.is_dir():
            indices = [
                path for path in source.rglob("artifact_index.json")
                if (path.parent / "spatial_support_revision_config.json").is_file()
            ]
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05j-c artifact index")
            index_bytes = indices[0].read_bytes()
            index = json.loads(index_bytes)
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JC_INDEX_SHA256:
                raise RuntimeError("extracted 05j-c artifact index SHA-256 mismatch")
            root = indices[0].parent
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05j-c member mismatch: {row['path']}")
            final_bytes = (root / "final_report.json").read_bytes()
            archive_hash, kind = None, "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05j-c source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JC_FINAL_SHA256:
            raise RuntimeError("05j-c final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind, "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05JC_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05JC_FINAL_SHA256,
            "verified_member_count": len(index["artifacts"]),
            "all_indexed_members_verified": len(index["artifacts"]) == int(index["artifact_count"]),
        }

    def prepare_trainable_topology_canary(self) -> Dict[str, Any]:
        base = self.prepare_spatial_support_revision()
        report, contract = self._read_verified_05jc()
        blockers = []
        if report.get("diagnosis") != "NONLOCAL_CONTEXT_HELPS_BUT_MAPPING_REMAINS_BELOW_GATE":
            blockers.append(f"unexpected 05j-c diagnosis: {report.get('diagnosis')}")
        if report.get("spatial_support_revision_passed") is not False:
            blockers.append("05j-c did not record the required failed gate")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05j-c dataset fingerprint mismatch")
        if not report.get("factorial_diagnosis", {}).get("topology_material_on_both_roles"):
            blockers.append("05j-c did not establish a robust topology signal")
        if report.get("methodology", {}).get("development_used_for_model_selection"):
            blockers.append("05j-c used development for model selection")
        if report.get("methodology", {}).get("rollout_performed"):
            blockers.append("05j-c unexpectedly performed rollout")
        heldout = report.get("heldout_contract", {})
        if heldout.get("inputs_extracted") or heldout.get("boundary_targets_materialized"):
            blockers.append("05j-c heldout contract was not sealed")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05j-c artifact verification is incomplete")
        if blockers:
            raise RuntimeError(f"05j-d provenance blockers: {blockers}")
        self.artifact_05jc_contract = contract
        payload = {
            "schema_version": "05j-d-trainable-topology-config-v1",
            "trainable_topology_canary": asdict(self.topology),
            "artifact_05jc": contract,
            "selection_roles": ["fit_grouped_pair_cv", "internal_calibration"],
            "development_used_for_selection": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "trainable_topology_canary_config.json", payload)
        return {**base, **payload}

    def prepare_topology_canary_designs(self) -> Dict[str, Any]:
        if not self.expanded_roles:
            raise RuntimeError("prepare_expanded_spatial_features() must run first")
        split = deterministic_stratified_pair_split(
            self.expanded_support["expanded_pairs"], self.topology.calibration_pair_count
        )
        fit_pairs = split["fit_pair_positions"]
        calibration_pairs = split["calibration_pair_positions"]
        if len(fit_pairs) != self.topology.fit_pair_count:
            raise RuntimeError("05j-d fit pair count mismatch")
        expanded = self.expanded_roles["expanded"]
        development = self.expanded_roles["development"]
        self.topology_roles = {
            "fit": _subset_role(expanded, _sample_positions(fit_pairs)),
            "calibration": _subset_role(expanded, _sample_positions(calibration_pairs)),
            "development": development,
        }
        fit = self.topology_roles["fit"]
        role_sketches: Dict[str, List[np.ndarray]] = {role: [] for role in self.topology_roles}
        sketch_report = {}
        denominator = np.arcsinh(self.spatial.input_asinh_reference_z)
        for surface, rank in (("h2_raw", self.spatial.pca_rank_h2), ("causal_raw", self.spatial.pca_rank_causal)):
            fit_values = np.asarray(fit[surface], dtype=np.float64)
            mean = fit_values.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(fit_values.std(axis=(0, 1), keepdims=True), self.spatial.feature_epsilon)
            fit_transformed = np.arcsinh((fit_values - mean) / scale) / denominator
            channel_mean, components = deterministic_pca_components(fit_transformed, rank)
            maxima = {}
            for role_name, role in self.topology_roles.items():
                transformed = np.arcsinh((np.asarray(role[surface]) - mean) / scale) / denominator
                sketch = apply_channel_pca(transformed, channel_mean, components)
                role_sketches[role_name].append(sketch)
                maxima[role_name] = float(np.max(np.abs(sketch)))
            sketch_report[surface] = {
                "rank": int(components.shape[0]), "maximum_absolute_by_role": maxima,
                "fit_values_only": True,
            }
        parents = np.asarray(self.arrays["parent_ids"], dtype=np.int64)
        axial = np.asarray(self.arrays["axial_conductance_to_parent_us"], dtype=np.float64)
        raw_designs = {}
        for role_name, role in self.topology_roles.items():
            local = np.concatenate(role_sketches[role_name], axis=-1)
            tree = axial_tree_diffusion(
                local, parents, axial, self.spatial.diffusion_scales,
                self.spatial.diffusion_self_weight,
            )
            raw_designs[role_name] = np.concatenate([
                np.asarray(role["voltage_t"])[..., None] / 100.0,
                np.asarray(role["base"])[..., None] / 100.0,
                tree,
            ], axis=-1)
        mean = raw_designs["fit"].mean(axis=(0, 1), keepdims=True)
        scale = np.maximum(raw_designs["fit"].std(axis=(0, 1), keepdims=True), self.spatial.feature_epsilon)
        self.topology_designs = {
            role: (np.arcsinh((values - mean) / scale) / denominator).astype(np.float32)
            for role, values in raw_designs.items()
        }
        fit_episode_ids = {
            value for position in fit_pairs for value in (
                self.expanded_support["expanded_pairs"][position]["left_episode_id"],
                self.expanded_support["expanded_pairs"][position]["right_episode_id"],
            )
        }
        calibration_episode_ids = {
            value for position in calibration_pairs for value in (
                self.expanded_support["expanded_pairs"][position]["left_episode_id"],
                self.expanded_support["expanded_pairs"][position]["right_episode_id"],
            )
        }
        report = {
            "schema_version": "05j-d-design-contract-v1", "valid": True,
            **split,
            "fit_pair_count": len(fit_pairs),
            "calibration_pair_count": len(calibration_pairs),
            "development_pair_count": len(development["base"]) // 2,
            "feature_width": int(self.topology_designs["fit"].shape[-1]),
            "fit_calibration_episode_overlap": sorted(fit_episode_ids & calibration_episode_ids),
            "sketches": sketch_report,
            "normalization_fit_roles": ["fit"],
            "calibration_used_for_checkpoint_selection": True,
            "development_values_used_to_fit": False,
            "development_used_for_checkpoint_selection": False,
            "heldout_inputs_extracted": False,
        }
        report["valid"] = bool(
            not report["fit_calibration_episode_overlap"]
            and split["fit_family_count"] >= self.topology.minimum_protocol_family_count
            and split["calibration_family_count"] >= self.topology.minimum_protocol_family_count
            and all(np.all(np.isfinite(value)) for value in self.topology_designs.values())
        )
        _write_json(self.output_dir / "topology_canary_design_contract.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-d design contract failed")
        return report

    def fit_fixed_tree_ridge_baseline(self) -> Dict[str, Any]:
        fit_x = self.topology_designs["fit"]
        fit_role = self.topology_roles["fit"]
        encoded = bounded_target_encode(
            np.asarray(fit_role["target"]) - np.asarray(fit_role["base"]),
            self.topology.target_residual_limit_mv, self.topology.target_atanh_margin,
        )
        folds = deterministic_pair_folds(
            self.topology.fit_pair_count, self.topology.ridge_cross_validation_folds
        )
        oof = np.zeros((len(self.topology.ridge_lambdas), *encoded.shape))
        progress = Progress("05j-d fixed ridge CV", len(folds))
        for fold_index, held_pairs in enumerate(folds):
            held = _sample_positions(held_pairs)
            keep = np.ones(len(fit_x), dtype=bool); keep[held] = False
            path, _ = dual_ridge_path_predict(
                fit_x[keep], encoded[keep], fit_x[held], self.topology.ridge_lambdas,
                pair_branch_weight=self.topology.branch_loss_weight,
            )
            oof[:, held] = path
            progress.update(fold_index + 1, f"held={len(held_pairs)} pairs")
        ladder = []
        for index, ridge in enumerate(self.topology.ridge_lambdas):
            metrics = self._metrics(
                fit_role, bounded_target_decode(oof[index], self.topology.target_residual_limit_mv)
            )
            ladder.append({
                "ridge_lambda": float(ridge), "cross_validation": metrics,
                "selection_score": pair_gate_selection_score(
                    metrics, max_error_weight=self.topology.selection_max_error_weight,
                    branch_log_weight=self.topology.selection_branch_log_weight,
                ),
            })
        selected = min(ladder, key=lambda row: row["selection_score"])
        evaluation = np.concatenate([
            self.topology_designs[role] for role in ("fit", "calibration", "development")
        ])
        path, diagnostics = dual_ridge_path_predict(
            fit_x, encoded, evaluation, [selected["ridge_lambda"]],
            pair_branch_weight=self.topology.branch_loss_weight,
        )
        decoded = bounded_target_decode(path[0], self.topology.target_residual_limit_mv)
        offset = 0
        metrics = {}
        for role in ("fit", "calibration", "development"):
            count = len(self.topology_designs[role])
            self.ridge_predictions[role] = decoded[offset:offset + count].astype(np.float32)
            metrics[role] = self._metrics(self.topology_roles[role], self.ridge_predictions[role])
            offset += count
        report = {
            "schema_version": "05j-d-fixed-tree-ridge-v1", "valid": True,
            "selected_ridge_lambda": selected["ridge_lambda"],
            "selection_score": selected["selection_score"],
            "cross_validation": selected["cross_validation"],
            "selection_ladder": ladder, "fit_diagnostics": diagnostics[0],
            "roles": metrics, "selection_roles": ["fit_grouped_pair_cv"],
            "calibration_used_for_selection": False,
            "development_used_for_selection": False,
        }
        _write_json(self.output_dir / "fixed_tree_ridge_baseline.json", report)
        return report

    def _loss(self, predicted: Any, target: Any) -> Any:
        scaled = (predicted - target) / self.topology.error_scale_mv
        point = torch.mean(scaled.square())
        paired = scaled.reshape(-1, 2, scaled.shape[-1])
        branch = torch.mean(((paired[:, 0] - paired[:, 1])).square())
        k = max(1, int(math.ceil(scaled.shape[-1] * self.topology.tail_fraction)))
        tail = torch.topk(torch.abs(scaled), k, dim=1).values.square().mean()
        return point + self.topology.branch_loss_weight * branch + self.topology.tail_loss_weight * tail

    def _role_metrics(self, role_name: str, residual: np.ndarray) -> Dict[str, Any]:
        metrics = self._metrics(self.topology_roles[role_name], residual)
        metrics["passed"] = self._pair_passes(metrics)
        return metrics

    def run_trainable_topology_canary(self) -> Dict[str, Any]:
        require_torch()
        if not self.ridge_predictions:
            raise RuntimeError("fit_fixed_tree_ridge_baseline() must run first")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = {}
        for role in ("fit", "calibration", "development"):
            state = self.topology_roles[role]
            tensors[role] = {
                "features": torch.as_tensor(self.topology_designs[role], device=device),
                "target": torch.as_tensor(np.asarray(state["target"]) - np.asarray(state["base"]), dtype=torch.float32, device=device),
                "ridge": torch.as_tensor(self.ridge_predictions[role], device=device),
            }
        checkpoint_dir = self.output_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
        runs = []
        total_runs = len(self.topology.families) * len(self.topology.seeds)
        overall = Progress("05j-d topology canary runs", total_runs)
        run_number = 0
        for family in self.topology.families:
            for seed in self.topology.seeds:
                torch.manual_seed(int(seed)); np.random.seed(int(seed))
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
                model = TrainableTopologyResidualHead(
                    self.topology_designs["fit"].shape[-1], self.layout.segment_count,
                    self.topology.hidden_width, self.topology.segment_embedding_dim,
                    self.topology.target_residual_limit_mv,
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=self.topology.learning_rate,
                    weight_decay=self.topology.weight_decay,
                )
                baseline = tensors["fit"]["ridge"] if family == "ridge_corrected_tree" else None
                best_score, best_epoch, stale, best_state = math.inf, 0, 0, None
                history = []
                progress = Progress(f"05j-d {family} seed{seed}", self.topology.epochs)
                for epoch in range(1, self.topology.epochs + 1):
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    prediction = model(tensors["fit"]["features"], baseline)
                    loss = self._loss(prediction, tensors["fit"]["target"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.topology.gradient_clip_norm)
                    optimizer.step()
                    if epoch == 1 or epoch % self.topology.evaluation_interval == 0:
                        model.eval()
                        with torch.no_grad():
                            calibration_prediction = model(
                                tensors["calibration"]["features"],
                                tensors["calibration"]["ridge"] if family == "ridge_corrected_tree" else None,
                            ).cpu().numpy()
                        calibration_metrics = self._role_metrics("calibration", calibration_prediction)
                        score = pair_gate_selection_score(
                            calibration_metrics,
                            max_error_weight=self.topology.selection_max_error_weight,
                            branch_log_weight=self.topology.selection_branch_log_weight,
                        )
                        history.append({
                            "family": family, "seed": int(seed), "epoch": epoch,
                            "training_loss": float(loss.detach().cpu()),
                            "calibration_score": score,
                            "calibration_rmse_mv": calibration_metrics["aggregate_voltage_rmse_mv"],
                            "calibration_max_error_mv": calibration_metrics["maximum_segment_error_mv"],
                            "calibration_retention": calibration_metrics["median_branching_retention"],
                        })
                        if score < best_score - 1e-9:
                            best_score, best_epoch, stale = score, epoch, 0
                            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                        else:
                            stale += self.topology.evaluation_interval
                        progress.update(epoch, f"loss={float(loss):.3g} cal={score:.3g}")
                        if stale >= self.topology.patience:
                            break
                if best_state is None:
                    raise RuntimeError("05j-d failed to select a checkpoint")
                model.load_state_dict(best_state); model.eval()
                role_metrics = {}
                with torch.no_grad():
                    for role in ("fit", "calibration", "development"):
                        residual = model(
                            tensors[role]["features"],
                            tensors[role]["ridge"] if family == "ridge_corrected_tree" else None,
                        ).cpu().numpy()
                        role_metrics[role] = self._role_metrics(role, residual)
                run_passed = all(role_metrics[role]["passed"] for role in role_metrics)
                checkpoint = checkpoint_dir / f"{family}_seed{seed}.pt"
                torch.save({
                    "state_dict": best_state, "family": family, "seed": int(seed),
                    "best_epoch": best_epoch, "feature_width": int(self.topology_designs["fit"].shape[-1]),
                    "dataset_fingerprint": self.bundle.fingerprint,
                }, checkpoint)
                runs.append({
                    "family": family, "seed": int(seed), "best_epoch": best_epoch,
                    "best_calibration_score": best_score, "roles": role_metrics,
                    "run_passed": run_passed,
                    "checkpoint": checkpoint.relative_to(self.output_dir).as_posix(),
                })
                write_parquet(self.output_dir / f"history_{family}_seed{seed}.parquet", history)
                run_number += 1
                overall.update(run_number, f"{family} seed={seed} pass={run_passed}")
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        family_summary = []
        for family in self.topology.families:
            chosen = [row for row in runs if row["family"] == family]
            passing = sum(row["run_passed"] for row in chosen)
            family_summary.append({
                "family": family, "run_count": len(chosen), "passing_seed_count": passing,
                "robust_gate_passed": passing >= self.topology.minimum_passing_seeds,
                "median_calibration_rmse_mv": float(np.median([row["roles"]["calibration"]["aggregate_voltage_rmse_mv"] for row in chosen])),
                "median_development_rmse_mv": float(np.median([row["roles"]["development"]["aggregate_voltage_rmse_mv"] for row in chosen])),
            })
        report = {
            "schema_version": "05j-d-trainable-topology-canary-v1", "valid": True,
            "device": str(device), "runs": runs, "family_summary": family_summary,
            "passing_families": [row["family"] for row in family_summary if row["robust_gate_passed"]],
            "development_used_for_checkpoint_selection": False,
            "development_inference_after_checkpoint_freeze": True,
            "heldout_inputs_extracted": False, "rollout_performed": False,
        }
        _write_json(self.output_dir / "trainable_topology_canary.json", report)
        write_parquet(self.output_dir / "trainable_topology_run_summary.parquet", [{
            "family": row["family"], "seed": row["seed"], "best_epoch": row["best_epoch"],
            "run_passed": row["run_passed"],
            **{f"{role}_rmse_mv": row["roles"][role]["aggregate_voltage_rmse_mv"] for role in ("fit", "calibration", "development")},
        } for row in runs])
        return report

    def finalize_trainable_topology_canary(
        self, design_report: Mapping[str, Any], ridge_report: Mapping[str, Any],
        canary_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        passing = list(canary_report["passing_families"])
        ridge_cal = ridge_report["roles"]["calibration"]["aggregate_voltage_rmse_mv"]
        ridge_dev = ridge_report["roles"]["development"]["aggregate_voltage_rmse_mv"]
        corrected = next(row for row in canary_report["family_summary"] if row["family"] == "ridge_corrected_tree")
        calibration_gain = (ridge_cal - corrected["median_calibration_rmse_mv"]) / max(abs(ridge_cal), 1e-12)
        development_gain = (ridge_dev - corrected["median_development_rmse_mv"]) / max(abs(ridge_dev), 1e-12)
        material = bool(
            calibration_gain >= self.topology.material_improvement_fraction
            and development_gain >= self.topology.material_improvement_fraction
        )
        passed = bool(passing)
        diagnosis = (
            "TRAINABLE_TOPOLOGY_DECODER_PASSES_ROBUST_GATE" if passed
            else "TRAINABLE_TOPOLOGY_IMPROVES_BUT_REMAINS_BELOW_GATE" if material
            else "TRAINABLE_TOPOLOGY_DECODER_FAILS_MICRO_CANARY"
        )
        report = {
            "schema_version": "05j-d-final-report-v1", "valid": True,
            "decision": "TRAIN_ONLY_TOPOLOGY_DECODER_MICRO_CANARY",
            "diagnosis": diagnosis, "trainable_topology_canary_passed": passed,
            "passing_families": passing, "full_training_authorized": False,
            "micro_rollout_authorized": passed,
            "code_revision": self.code_revision, "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05jc": self.artifact_05jc_contract,
            "design_contract": dict(design_report),
            "fixed_tree_ridge_baseline": dict(ridge_report),
            "trainable_topology_canary": dict(canary_report),
            "improvement_vs_fixed_ridge": {
                "calibration_rmse_improvement_fraction": calibration_gain,
                "development_rmse_improvement_fraction": development_gain,
                "material_on_both_roles": material,
            },
            "heldout_contract": {
                "inputs_extracted": False, "boundary_targets_materialized": False,
                "event_targets_materialized": False, "candidate_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "exact_05jc_signal_verified": True,
                "fit_calibration_split_stratified_and_episode_disjoint": True,
                "normalization_fit_roles": ["fit"],
                "calibration_used_for_checkpoint_selection": True,
                "development_used_for_checkpoint_selection": False,
                "development_inference_after_checkpoint_freeze": True,
                "base_h2_frozen": True, "teacher_encoder_updated": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": (
                    "05k_repaired_representation_micro_rollout" if passed
                    else "05j_e_topology_decoder_refinement" if material
                    else "05j_e_architecture_reassessment"
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
                    "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
                })
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05j-d-artifact-index-v1",
            "artifact_count": len(records), "artifacts": records,
        })
        return report
