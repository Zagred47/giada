"""Notebook-05j-h train-only regenerative support expansion."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_regenerative_state_decomposition import (
    HinesRegenerativeStateTargetDecomposition,
)
from .hines_spatial_support_revision import (
    apply_channel_pca,
    axial_tree_diffusion,
    deterministic_pca_components,
)
from .hines_trainable_topology_canary import (
    TrainableTopologyResidualHead,
    deterministic_stratified_pair_split,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05JG_ARCHIVE_SHA256 = (
    "6ec384d6e549878ee52bdf1562b071e87596a09acb5241f9fdbb2e0f86d0c569"
)
EXPECTED_05JG_INDEX_SHA256 = (
    "c8ba42a67859bb7ceb83630bae4b453b4967680a02c98a66374ce7633aad60a0"
)
EXPECTED_05JG_FINAL_SHA256 = (
    "8c54566d33b2cfd034d81266709997c02a1c43f3d43fc6b0926cb02a763a43b8"
)

REGENERATIVE_STRATA = ("regenerative", "near_regenerative", "subthreshold")


@dataclass(frozen=True)
class HinesRegenerativeSupportExpansionConfig:
    target_pair_count: int = 72
    minimum_pair_count: int = 36
    minimum_materially_expanded_pair_count: int = 56
    confirmation_pair_count: int = 18
    regenerative_peak_threshold_mv: float = -20.0
    near_regenerative_peak_threshold_mv: float = -45.0
    minimum_regenerative_pairs: int = 12
    minimum_near_regenerative_pairs: int = 8
    minimum_subthreshold_pairs: int = 8
    minimum_protocol_family_count: int = 6
    minimum_confirmation_pair_win_fraction: float = 2.0 / 3.0
    material_rmse_improvement_fraction: float = 0.20
    material_max_error_improvement_fraction: float = 0.10
    oracle_specificity_improvement_fraction: float = 0.15

    def validate(self) -> None:
        if self.target_pair_count <= self.minimum_materially_expanded_pair_count:
            raise ValueError("05j-h target support must exceed its minimum")
        if not 0 < self.confirmation_pair_count < self.minimum_pair_count:
            raise ValueError("05j-h confirmation size is invalid")
        if self.regenerative_peak_threshold_mv <= self.near_regenerative_peak_threshold_mv:
            raise ValueError("05j-h regenerative thresholds are reversed")
        positive = (
            self.target_pair_count,
            self.minimum_pair_count,
            self.minimum_materially_expanded_pair_count,
            self.confirmation_pair_count,
            self.minimum_regenerative_pairs,
            self.minimum_near_regenerative_pairs,
            self.minimum_subthreshold_pairs,
            self.minimum_protocol_family_count,
            self.minimum_confirmation_pair_win_fraction,
            self.material_rmse_improvement_fraction,
            self.material_max_error_improvement_fraction,
            self.oracle_specificity_improvement_fraction,
        )
        if min(positive) <= 0:
            raise ValueError("05j-h configuration values must be positive")
        if any(value > 1 for value in (
            self.minimum_confirmation_pair_win_fraction,
            self.material_rmse_improvement_fraction,
            self.material_max_error_improvement_fraction,
            self.oracle_specificity_improvement_fraction,
        )):
            raise ValueError("05j-h fractions cannot exceed one")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRegenerativeSupportExpansionConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def regenerative_stratum(
    peak_mv: float,
    *,
    regenerative_threshold_mv: float = -20.0,
    near_threshold_mv: float = -45.0,
) -> str:
    if peak_mv >= regenerative_threshold_mv:
        return "regenerative"
    if peak_mv >= near_threshold_mv:
        return "near_regenerative"
    return "subthreshold"


def select_disjoint_stratified_pairs(
    rows: Sequence[Mapping[str, Any]],
    target_count: int,
) -> List[Dict[str, Any]]:
    """Round-robin deterministic strata while forbidding episode reuse."""

    by_stratum: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in REGENERATIVE_STRATA
    }
    for original in rows:
        row = dict(original)
        identity = f"{int(row['left_index'])}:{int(row['right_index'])}"
        row["selection_sha256"] = hashlib.sha256(identity.encode()).hexdigest()
        by_stratum[str(row["regenerative_stratum"])].append(row)
    for values in by_stratum.values():
        values.sort(key=lambda row: row["selection_sha256"])
    cursors = {name: 0 for name in REGENERATIVE_STRATA}
    selected: List[Dict[str, Any]] = []
    used_episodes = set()
    while len(selected) < int(target_count):
        added = False
        for stratum in REGENERATIVE_STRATA:
            values = by_stratum[stratum]
            while cursors[stratum] < len(values):
                row = values[cursors[stratum]]
                cursors[stratum] += 1
                episodes = {
                    str(row["left_episode_id"]), str(row["right_episode_id"])
                }
                if episodes & used_episodes:
                    continue
                selected.append(row)
                used_episodes.update(episodes)
                added = True
                break
            if len(selected) == int(target_count):
                break
        if not added:
            break
    return selected


class HinesRegenerativeSupportExpansion(HinesRegenerativeStateTargetDecomposition):
    """Repeat the registered oracle test on larger independent train support."""

    def __init__(
        self,
        *args: Any,
        support_expansion_config: HinesRegenerativeSupportExpansionConfig,
        artifact_05jg_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        support_expansion_config.validate()
        self.support_expansion_config = support_expansion_config
        self.artifact_05jg_source = Path(artifact_05jg_source).resolve()
        self.artifact_05jg_contract: Dict[str, Any] = {}
        self.regenerative_support: Dict[str, Any] = {}
        # Kept so a later, strictly external confirmation shard can be encoded
        # with the exact transform fitted on the original 05j-d support.  The
        # transform is never refitted on confirmation data.
        self.expanded_topology_transform: Dict[str, Any] = {}

    def _read_verified_05jg(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05jg_source
        if source.is_file():
            if sha256_file(source) != EXPECTED_05JG_ARCHIVE_SHA256:
                raise RuntimeError("05j-g archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[:-len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                index = json.loads(index_bytes)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JG_INDEX_SHA256:
                    raise RuntimeError("05j-g artifact index SHA-256 mismatch")
                for row in index["artifacts"]:
                    payload = archive.read(root + str(row["path"]).replace("\\", "/"))
                    if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05j-g indexed member mismatch: {row['path']}")
                final_bytes = archive.read(root + "final_report.json")
            kind, archive_hash = "original_zip", EXPECTED_05JG_ARCHIVE_SHA256
        elif source.is_dir():
            indices = [
                path for path in source.rglob("artifact_index.json")
                if (path.parent / "state_target_decomposition_config.json").is_file()
            ]
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05j-g artifact index")
            index_bytes = indices[0].read_bytes()
            index = json.loads(index_bytes)
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JG_INDEX_SHA256:
                raise RuntimeError("extracted 05j-g artifact index SHA-256 mismatch")
            root = indices[0].parent
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05j-g member mismatch: {row['path']}")
            final_bytes = (root / "final_report.json").read_bytes()
            kind, archive_hash = "kaggle_extracted_directory", None
        else:
            raise RuntimeError(f"05j-g source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JG_FINAL_SHA256:
            raise RuntimeError("05j-g final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05JG_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05JG_FINAL_SHA256,
            "verified_member_count": len(index["artifacts"]),
            "all_indexed_members_verified": (
                len(index["artifacts"]) == int(index["artifact_count"])
            ),
        }

    def prepare_regenerative_support_expansion(self) -> Dict[str, Any]:
        base = self.prepare_regenerative_state_target_decomposition()
        report, contract = self._read_verified_05jg()
        blockers = []
        if report.get("diagnosis") != "REGENERATIVE_STATE_LINK_NOT_CONFIRMED_ON_DEVELOPMENT":
            blockers.append(f"unexpected 05j-g diagnosis: {report.get('diagnosis')}")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05j-g dataset fingerprint mismatch")
        if report.get("candidate_model_authorized") or report.get("micro_rollout_authorized"):
            blockers.append("05j-g unexpectedly authorized a candidate or rollout")
        if report.get("methodology", {}).get("development_used_for_model_selection"):
            blockers.append("05j-g used development for model selection")
        if report.get("heldout_contract", {}).get("inputs_extracted"):
            blockers.append("05j-g held-out inputs were not sealed")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05j-g artifact verification is incomplete")
        if blockers:
            raise RuntimeError(f"05j-h provenance blockers: {blockers}")
        self.artifact_05jg_contract = contract
        payload = {
            "schema_version": "05j-h-regenerative-support-config-v1",
            "regenerative_support_expansion": asdict(self.support_expansion_config),
            "artifact_05jg": contract,
            "support_source": "train_branch_pairs_only",
            "support_selection": "target_peak_regime_only_not_model_error",
            "confirmation_role": "reserved_train_episodes",
            "registered_state_probe_unchanged": True,
            "development_used_for_model_selection": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
            "candidate_model_training_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "regenerative_support_expansion_config.json", payload)
        return {**base, **payload}

    def build_regenerative_support(self) -> Dict[str, Any]:
        original_canary = self.canary
        self.canary = replace(
            self.canary,
            maximum_local_steps_searched=self.audit.maximum_local_steps_searched,
            maximum_candidates_per_split=self.audit.maximum_candidates,
            minimum_teacher_distance_mv=self.audit.minimum_teacher_distance_mv,
        )
        try:
            candidates = [dict(row) for row in self._pair_candidates_for_split("train")]
        finally:
            self.canary = original_canary
        logical_indices = sorted({
            int(value) for row in candidates
            for value in (row["left_index"], row["right_index"])
        })
        state_t1 = self.store.read_state(
            logical_indices, "t_plus_1", categories=("voltage",)
        )
        lookup = {value: position for position, value in enumerate(logical_indices)}
        for row in candidates:
            pair_target = state_t1[[
                lookup[int(row["left_index"])], lookup[int(row["right_index"])]
            ]]
            peak = float(np.max(pair_target))
            row["target_peak_mv"] = peak
            row["regenerative_stratum"] = regenerative_stratum(
                peak,
                regenerative_threshold_mv=self.support_expansion_config.regenerative_peak_threshold_mv,
                near_threshold_mv=self.support_expansion_config.near_regenerative_peak_threshold_mv,
            )
            row["protocol_family"] = " <> ".join(sorted({
                self._protocol_family(int(row["left_index"])),
                self._protocol_family(int(row["right_index"])),
            }))
        selected = select_disjoint_stratified_pairs(
            candidates, self.support_expansion_config.target_pair_count
        )
        split_rows = [
            {"protocol_family": row["regenerative_stratum"]} for row in selected
        ]
        split = deterministic_stratified_pair_split(
            split_rows, self.support_expansion_config.confirmation_pair_count
        )
        fit_rows = [selected[position] for position in split["fit_pair_positions"]]
        confirmation_rows = [
            selected[position] for position in split["calibration_pair_positions"]
        ]
        counts = {
            name: int(sum(row["regenerative_stratum"] == name for row in selected))
            for name in REGENERATIVE_STRATA
        }
        fit_episodes = {
            str(value) for row in fit_rows
            for value in (row["left_episode_id"], row["right_episode_id"])
        }
        confirmation_episodes = {
            str(value) for row in confirmation_rows
            for value in (row["left_episode_id"], row["right_episode_id"])
        }
        protocol_families = sorted({row["protocol_family"] for row in selected})
        blockers = []
        if len(selected) < self.support_expansion_config.minimum_pair_count:
            blockers.append("insufficient disjoint train pairs")
        required = {
            "regenerative": self.support_expansion_config.minimum_regenerative_pairs,
            "near_regenerative": self.support_expansion_config.minimum_near_regenerative_pairs,
            "subthreshold": self.support_expansion_config.minimum_subthreshold_pairs,
        }
        if fit_episodes & confirmation_episodes:
            blockers.append("fit/confirmation episode overlap")
        if any(row.get("split") != "train" for row in selected):
            blockers.append("non-train pair entered expanded support")
        support_hash = hashlib.sha256(json.dumps(
            {
                "selected": selected,
                "fit": split["fit_pair_positions"],
                "confirmation": split["calibration_pair_positions"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()).hexdigest()
        biological_support_sufficient = bool(
            len(selected)
            >= self.support_expansion_config.minimum_materially_expanded_pair_count
            and all(counts[name] >= minimum for name, minimum in required.items())
            and len(protocol_families)
            >= self.support_expansion_config.minimum_protocol_family_count
        )
        report = {
            "schema_version": "05j-h-regenerative-support-v1",
            "valid": not blockers,
            "blockers": blockers,
            "candidate_pair_count": len(candidates),
            "selected_pair_count": len(selected),
            "fit_pair_count": len(fit_rows),
            "confirmation_pair_count": len(confirmation_rows),
            "stratum_counts": counts,
            "required_stratum_counts": required,
            "biological_support_sufficient": biological_support_sufficient,
            "materially_expanded_pair_count_threshold": self.support_expansion_config.minimum_materially_expanded_pair_count,
            "protocol_family_count": len(protocol_families),
            "protocol_families": protocol_families,
            "fit_confirmation_episode_overlap": sorted(
                fit_episodes & confirmation_episodes
            ),
            "all_pairs_train_only": all(row.get("split") == "train" for row in selected),
            "all_selected_episodes_disjoint": len(fit_episodes | confirmation_episodes) == 2 * len(selected),
            "support_sha256": support_hash,
            "selection_uses_model_error": False,
            "selection_uses_target_peak_regime": True,
            "fit_pairs": fit_rows,
            "confirmation_pairs": confirmation_rows,
            "heldout_inputs_extracted": False,
        }
        self.regenerative_support = report
        _write_json(self.output_dir / "regenerative_support.json", report)
        write_parquet(self.output_dir / "regenerative_support.parquet", [
            {"support_role": role, "pair_position": position, **row}
            for role, rows in (("fit", fit_rows), ("confirmation", confirmation_rows))
            for position, row in enumerate(rows)
        ])
        if blockers:
            raise RuntimeError(f"05j-h support blockers: {blockers}")
        return report

    def _fit_registered_topology_transform(self) -> Dict[str, Any]:
        fit = self.topology_roles["fit"]
        denominator = np.arcsinh(self.spatial.input_asinh_reference_z)
        surfaces = {}
        for name, rank in (
            ("h2_raw", self.spatial.pca_rank_h2),
            ("causal_raw", self.spatial.pca_rank_causal),
        ):
            values = np.asarray(fit[name], dtype=np.float64)
            mean = values.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(
                values.std(axis=(0, 1), keepdims=True), self.spatial.feature_epsilon
            )
            transformed = np.arcsinh((values - mean) / scale) / denominator
            channel_mean, components = deterministic_pca_components(transformed, rank)
            surfaces[name] = {
                "mean": mean,
                "scale": scale,
                "channel_mean": channel_mean,
                "components": components,
            }
        transform: Dict[str, Any] = {"surfaces": surfaces}
        raw_original = {
            role: self._raw_topology_design(state, transform)
            for role, state in self.topology_roles.items()
        }
        transform["raw_mean"] = raw_original["fit"].mean(
            axis=(0, 1), keepdims=True
        )
        transform["raw_scale"] = np.maximum(
            raw_original["fit"].std(axis=(0, 1), keepdims=True),
            self.spatial.feature_epsilon,
        )
        reconstructed = {
            role: self._normalize_raw_topology(raw, transform)
            for role, raw in raw_original.items()
        }
        maximum_error = max(
            float(np.max(np.abs(reconstructed[role] - self.topology_designs[role])))
            for role in reconstructed
        )
        if maximum_error > 2e-5:
            raise RuntimeError(
                f"05j-h topology transform reconstruction mismatch: {maximum_error}"
            )
        transform["maximum_original_design_reconstruction_error"] = maximum_error
        return transform

    def _raw_topology_design(
        self, role: Mapping[str, Any], transform: Mapping[str, Any]
    ) -> np.ndarray:
        denominator = np.arcsinh(self.spatial.input_asinh_reference_z)
        sketches = []
        for name in ("h2_raw", "causal_raw"):
            row = transform["surfaces"][name]
            values = np.asarray(role[name], dtype=np.float64)
            transformed = np.arcsinh((values - row["mean"]) / row["scale"]) / denominator
            sketches.append(apply_channel_pca(
                transformed, row["channel_mean"], row["components"]
            ))
        local = np.concatenate(sketches, axis=-1)
        tree = axial_tree_diffusion(
            local,
            np.asarray(self.arrays["parent_ids"], dtype=np.int64),
            np.asarray(self.arrays["axial_conductance_to_parent_us"], dtype=np.float64),
            self.spatial.diffusion_scales,
            self.spatial.diffusion_self_weight,
        )
        return np.concatenate([
            np.asarray(role["voltage_t"])[..., None] / 100.0,
            np.asarray(role["base"])[..., None] / 100.0,
            tree,
        ], axis=-1)

    def _normalize_raw_topology(
        self, raw: np.ndarray, transform: Mapping[str, Any]
    ) -> np.ndarray:
        denominator = np.arcsinh(self.spatial.input_asinh_reference_z)
        return (
            np.arcsinh((raw - transform["raw_mean"]) / transform["raw_scale"])
            / denominator
        ).astype(np.float32)

    def prepare_expanded_regenerative_roles(self) -> Dict[str, Any]:
        if not self.regenerative_support.get("valid"):
            raise RuntimeError("build_regenerative_support() must run first")
        require_torch()
        transform = self._fit_registered_topology_transform()
        self.expanded_topology_transform = transform
        original_development_role = self.topology_roles["development"]
        original_development_design = self.topology_designs["development"].copy()
        indices = {
            "fit": np.asarray(
                self._pair_indices(self.regenerative_support["fit_pairs"]), dtype=np.int64
            ),
            "calibration": np.asarray(
                self._pair_indices(self.regenerative_support["confirmation_pairs"]), dtype=np.int64
            ),
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = self._load_h2_checkpoint(device)
        model.eval()
        roles = {}
        progress = Progress("05j-h frozen expanded features", len(indices))
        for position, (role, logical_indices) in enumerate(indices.items(), start=1):
            roles[role] = self._extract_recheck_role(model, logical_indices)
            progress.update(position, f"{role} samples={len(logical_indices)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        roles["development"] = original_development_role
        designs = {
            role: self._normalize_raw_topology(
                self._raw_topology_design(state, transform), transform
            )
            for role, state in roles.items()
        }
        development_error = float(np.max(np.abs(
            designs["development"] - original_development_design
        )))
        if development_error > 2e-5:
            raise RuntimeError("05j-h development design reconstruction mismatch")
        self.topology_roles = roles
        self.topology_designs = designs
        report = {
            "schema_version": "05j-h-expanded-regenerative-roles-v1",
            "valid": True,
            "fit_pair_count": len(indices["fit"]) // 2,
            "confirmation_pair_count": len(indices["calibration"]) // 2,
            "development_pair_count": len(roles["development"]["base"]) // 2,
            "feature_width": int(designs["fit"].shape[-1]),
            "original_design_reconstruction_max_error": transform[
                "maximum_original_design_reconstruction_error"
            ],
            "development_design_reconstruction_max_error": development_error,
            "registered_05jd_feature_transform_reused": True,
            "normalization_fit_roles": ["original_05jd_fit"],
            "confirmation_used_to_fit_features": False,
            "development_used_to_fit_features": False,
            "heldout_inputs_extracted": False,
        }
        _write_json(self.output_dir / "expanded_regenerative_roles.json", report)
        return report

    def reconstruct_expanded_direct_tree_ensemble(self) -> Dict[str, Any]:
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        previous_development = {
            seed: self.frozen_predictions[self.reassessment.audited_family][seed]["development"].copy()
            for seed in self.reassessment.seeds
        }
        registered = [
            row for row in self.artifact_05jd_report["trainable_topology_canary"]["runs"]
            if row["family"] == self.reassessment.audited_family
        ]
        predictions: Dict[int, Dict[str, np.ndarray]] = {}
        rows = []
        progress = Progress("05j-h frozen direct-tree ensemble", len(registered))
        for position, row in enumerate(registered, start=1):
            seed = int(row["seed"])
            checkpoint = torch.load(
                io.BytesIO(self._read_05jd_checkpoint_bytes(str(row["checkpoint"]))),
                map_location=device,
                weights_only=False,
            )
            if checkpoint["family"] != self.reassessment.audited_family or int(checkpoint["seed"]) != seed:
                raise RuntimeError("05j-h direct-tree checkpoint identity mismatch")
            model = TrainableTopologyResidualHead(
                self.topology_designs["fit"].shape[-1],
                self.layout.segment_count,
                self.topology.hidden_width,
                self.topology.segment_embedding_dim,
                self.topology.target_residual_limit_mv,
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            predictions[seed] = {}
            with torch.no_grad():
                for role, design in self.topology_designs.items():
                    predictions[seed][role] = model(
                        torch.as_tensor(design, device=device)
                    ).cpu().numpy()
            development_error = float(np.max(np.abs(
                predictions[seed]["development"] - previous_development[seed]
            )))
            rows.append({
                "seed": seed,
                "checkpoint": row["checkpoint"],
                "development_reconstruction_max_error": development_error,
                "valid": development_error <= 2e-4,
            })
            progress.update(position, f"seed={seed} error={development_error:.3g}")
            del model
        self.frozen_predictions = {
            self.reassessment.audited_family: predictions
        }
        report = {
            "schema_version": "05j-h-expanded-direct-tree-reconstruction-v1",
            "valid": all(row["valid"] for row in rows),
            "device": str(device),
            "runs": rows,
            "checkpoint_count": len(rows),
            "retraining_performed": False,
            "confirmation_used_for_checkpoint_selection": False,
            "development_used_for_checkpoint_selection": False,
            "heldout_inputs_extracted": False,
        }
        _write_json(self.output_dir / "expanded_direct_tree_reconstruction.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-h expanded direct-tree reconstruction mismatch")
        return report

    @staticmethod
    def _pairwise_win_report(
        candidate: Mapping[str, Any], control: Mapping[str, Any], role: str
    ) -> Dict[str, Any]:
        candidate_rows = candidate["roles"][role]["pair_metrics"]
        control_rows = control["roles"][role]["pair_metrics"]
        gains = np.asarray([
            (float(base["voltage_rmse_mv"]) - float(test["voltage_rmse_mv"]))
            / max(float(base["voltage_rmse_mv"]), 1e-12)
            for test, base in zip(candidate_rows, control_rows)
        ])
        return {
            "pair_count": len(gains),
            "win_count": int(np.sum(gains > 0)),
            "win_fraction": float(np.mean(gains > 0)),
            "median_pair_rmse_improvement_fraction": float(np.median(gains)),
            "minimum_pair_rmse_improvement_fraction": float(np.min(gains)),
            "maximum_pair_rmse_improvement_fraction": float(np.max(gains)),
        }

    def finalize_regenerative_support_expansion(
        self,
        support_report: Mapping[str, Any],
        role_report: Mapping[str, Any],
        reconstruction_report: Mapping[str, Any],
        surface_report: Mapping[str, Any],
        probe_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        aligned = self._probe(probe_report, "oracle_delta_aligned_all")
        shifted = self._probe(probe_report, "oracle_delta_spatial_shift_control")
        causal = self._probe(probe_report, "causal_current_all")
        confirmation_specificity = probe_report[
            "aligned_oracle_vs_spatial_shift_rmse_improvement_fraction"
        ]["calibration"]
        pairwise = self._pairwise_win_report(aligned, shifted, "calibration")
        oracle_gain = aligned["improvement_vs_intercept_control"]["calibration"]
        causal_gain = causal["improvement_vs_intercept_control"]["calibration"]
        oracle_confirmed = bool(
            oracle_gain["rmse_improvement_fraction"]
            >= self.support_expansion_config.material_rmse_improvement_fraction
            and oracle_gain["maximum_error_improvement_fraction"]
            >= self.support_expansion_config.material_max_error_improvement_fraction
            and confirmation_specificity
            >= self.support_expansion_config.oracle_specificity_improvement_fraction
            and pairwise["win_fraction"]
            >= self.support_expansion_config.minimum_confirmation_pair_win_fraction
        )
        causal_confirmed = bool(
            causal_gain["rmse_improvement_fraction"]
            >= self.support_expansion_config.material_rmse_improvement_fraction
            and causal_gain["maximum_error_improvement_fraction"]
            >= self.support_expansion_config.material_max_error_improvement_fraction
        )
        if not support_report["biological_support_sufficient"]:
            diagnosis = "EXISTING_DATASET_LACKS_INDEPENDENT_REGENERATIVE_CONFIRMATION_SUPPORT"
            next_experiment = "05j_i_generate_regenerative_confirmation_support"
            causal_confirmed = False
            oracle_confirmed = False
        elif causal_confirmed:
            diagnosis = "EXPANDED_SUPPORT_CONFIRMS_CAUSAL_BOUNDARY_STATE_SIGNAL"
            next_experiment = "05j_i_explicit_regenerative_state_input_revision"
        elif oracle_confirmed:
            diagnosis = "EXPANDED_SUPPORT_CONFIRMS_REGENERATIVE_STATE_TRANSITION_SIGNAL"
            next_experiment = "05j_i_joint_regenerative_state_transition_canary"
        elif (
            oracle_gain["rmse_improvement_fraction"]
            >= self.support_expansion_config.material_rmse_improvement_fraction
        ):
            diagnosis = "REGENERATIVE_STATE_SIGNAL_IS_GLOBAL_NOT_SPATIALLY_SPECIFIC"
            next_experiment = "05j_i_regime_conditioned_transition_objective"
        else:
            diagnosis = "EXPANDED_SUPPORT_REJECTS_REGENERATIVE_STATE_EXPLANATION"
            next_experiment = "05j_i_voltage_decoder_objective_reassessment"
        report = {
            "schema_version": "05j-h-final-report-v1",
            "valid": bool(
                support_report["valid"]
                and role_report["valid"]
                and reconstruction_report["valid"]
                and surface_report["valid"]
                and probe_report["valid"]
            ),
            "decision": "TRAIN_ONLY_REGENERATIVE_SUPPORT_CONFIRMATION",
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05jg": self.artifact_05jg_contract,
            "support": dict(support_report),
            "expanded_roles": dict(role_report),
            "direct_tree_reconstruction": dict(reconstruction_report),
            "state_target_surfaces": dict(surface_report),
            "state_target_probes": dict(probe_report),
            "confirmation_aligned_vs_shifted": {
                "aggregate_rmse_improvement_fraction": confirmation_specificity,
                "pairwise": pairwise,
            },
            "causal_boundary_state_confirmed": causal_confirmed,
            "regenerative_state_transition_confirmed": oracle_confirmed,
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
                "exact_05jg_conditional_result_verified": True,
                "support_train_only": True,
                "support_selected_by_biological_regime_not_model_error": True,
                "fit_confirmation_episode_disjoint": True,
                "registered_05jg_probe_unchanged": True,
                "base_direct_tree_checkpoints_frozen": True,
                "future_voltage_coordinate_excluded": True,
                "future_state_delta_used_as_diagnostic_oracle_only": True,
                "confirmation_used_for_model_selection": False,
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
            "schema_version": "05j-h-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
