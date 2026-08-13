"""Notebook-05j-e architecture reassessment after the topology canary."""

from __future__ import annotations

import hashlib
import io
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
from .hines_repaired_representation_revision import pair_gate_selection_score
from .hines_spatial_support_revision import axial_tree_diffusion
from .hines_trainable_topology_canary import (
    HinesTrainableTopologyCanary,
    TrainableTopologyResidualHead,
    deterministic_stratified_pair_split,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05JD_ARCHIVE_SHA256 = (
    "566e0d8bdf945e9abd55c7d1e74883d017b13db50dfc1ec2c143f224f935426e"
)
EXPECTED_05JD_INDEX_SHA256 = (
    "ccf988c8568fdff4e6f7f5a2e92785d1bd944576401a57852e97e69d52ee081b"
)
EXPECTED_05JD_FINAL_SHA256 = (
    "c3d97bf2df6bfd41605f34f130a41e7fcca52e474e8c7f4b5219b62e1aa01f92"
)


@dataclass(frozen=True)
class HinesArchitectureReassessmentConfig:
    audited_family: str = "direct_tree"
    seeds: Tuple[int, ...] = (17, 29, 43)
    expected_fit_pair_count: int = 36
    expected_calibration_pair_count: int = 12
    expected_development_pair_count: int = 1
    collision_quantile: float = 0.10
    large_branch_target_mv: float = 5.0
    minimum_large_target_given_collision_fraction: float = 0.05
    systematic_disagreement_ratio: float = 0.25
    material_improvement_fraction: float = 0.20
    concentrated_segment_fraction: float = 0.10
    concentrated_error_energy_fraction: float = 0.50
    high_frequency_energy_fraction: float = 0.50
    affine_ridge: float = 1e-4
    metric_atol: float = 1e-4

    def validate(self) -> None:
        if self.audited_family != "direct_tree":
            raise ValueError("05j-e must audit the qualified direct-tree signal")
        if tuple(self.seeds) != (17, 29, 43):
            raise ValueError("05j-e must reconstruct the registered three seeds")
        if (
            self.expected_fit_pair_count,
            self.expected_calibration_pair_count,
            self.expected_development_pair_count,
        ) != (36, 12, 1):
            raise ValueError("05j-e split must match the registered 05j-d split")
        fractions = (
            self.collision_quantile,
            self.minimum_large_target_given_collision_fraction,
            self.systematic_disagreement_ratio,
            self.material_improvement_fraction,
            self.concentrated_segment_fraction,
            self.concentrated_error_energy_fraction,
            self.high_frequency_energy_fraction,
        )
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("05j-e diagnostic fractions must lie in (0, 1)")
        if min(self.large_branch_target_mv, self.affine_ridge, self.metric_atol) <= 0:
            raise ValueError("05j-e thresholds must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesArchitectureReassessmentConfig":
        payload = dict(values)
        if "seeds" in payload:
            payload["seeds"] = tuple(payload["seeds"])
        result = cls(**payload)
        result.validate()
        return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def fit_segment_affine_calibrator(
    prediction: np.ndarray, target: np.ndarray, ridge: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit independent slope/intercept diagnostics on fit data only."""

    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("affine calibrator expects equal [sample, segment] arrays")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered = x - x_mean
    slope = np.sum(centered * (y - y_mean), axis=0) / (
        np.sum(centered * centered, axis=0) + float(ridge)
    )
    intercept = y_mean - slope * x_mean
    return slope, intercept


def apply_segment_affine(
    prediction: np.ndarray, slope: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    return np.asarray(prediction) * np.asarray(slope)[None] + np.asarray(intercept)[None]


def error_energy_concentration(error: np.ndarray, top_fraction: float) -> Dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)
    energy = np.sum(values * values, axis=0)
    count = max(1, int(math.ceil(len(energy) * float(top_fraction))))
    order = np.argsort(energy)[::-1]
    total = max(float(np.sum(energy)), 1e-12)
    return {
        "top_segment_count": count,
        "top_segment_ids": order[:count].astype(int).tolist(),
        "top_segment_error_energy_fraction": float(np.sum(energy[order[:count]]) / total),
    }


class HinesArchitectureReassessment(HinesTrainableTopologyCanary):
    """Read-only forensic reconstruction of the failed 05j-d canary."""

    def __init__(
        self, *args: Any, reassessment_config: HinesArchitectureReassessmentConfig,
        artifact_05jd_source: Path, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        reassessment_config.validate()
        self.reassessment = reassessment_config
        self.artifact_05jd_source = Path(artifact_05jd_source).resolve()
        self.artifact_05jd_contract: Dict[str, Any] = {}
        self.artifact_05jd_report: Dict[str, Any] = {}
        self._artifact_05jd_root: Path | None = None
        self._artifact_05jd_zip_root = ""
        self.frozen_predictions: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}

    def _read_verified_05jd(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05jd_source
        if source.is_file():
            if sha256_file(source) != EXPECTED_05JD_ARCHIVE_SHA256:
                raise RuntimeError("05j-d archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[:-len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                index = json.loads(index_bytes)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JD_INDEX_SHA256:
                    raise RuntimeError("05j-d artifact index SHA-256 mismatch")
                for row in index["artifacts"]:
                    payload = archive.read(root + str(row["path"]).replace("\\", "/"))
                    if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05j-d indexed member mismatch: {row['path']}")
                final_bytes = archive.read(root + "final_report.json")
            self._artifact_05jd_zip_root = root
            kind, archive_hash = "original_zip", EXPECTED_05JD_ARCHIVE_SHA256
        elif source.is_dir():
            indices = [
                path for path in source.rglob("artifact_index.json")
                if (path.parent / "trainable_topology_canary_config.json").is_file()
            ]
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05j-d artifact index")
            index_bytes = indices[0].read_bytes()
            index = json.loads(index_bytes)
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JD_INDEX_SHA256:
                raise RuntimeError("extracted 05j-d artifact index SHA-256 mismatch")
            root_path = indices[0].parent
            for row in index["artifacts"]:
                path = root_path / str(row["path"])
                if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05j-d member mismatch: {row['path']}")
            final_bytes = (root_path / "final_report.json").read_bytes()
            self._artifact_05jd_root = root_path
            kind, archive_hash = "kaggle_extracted_directory", None
        else:
            raise RuntimeError(f"05j-d source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JD_FINAL_SHA256:
            raise RuntimeError("05j-d final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind, "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05JD_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05JD_FINAL_SHA256,
            "verified_member_count": len(index["artifacts"]),
            "all_indexed_members_verified": len(index["artifacts"]) == int(index["artifact_count"]),
        }

    def prepare_architecture_reassessment(self) -> Dict[str, Any]:
        base = self.prepare_trainable_topology_canary()
        report, contract = self._read_verified_05jd()
        blockers = []
        if report.get("diagnosis") != "TRAINABLE_TOPOLOGY_DECODER_FAILS_MICRO_CANARY":
            blockers.append(f"unexpected 05j-d diagnosis: {report.get('diagnosis')}")
        if report.get("trainable_topology_canary_passed") is not False:
            blockers.append("05j-d did not record the required failed canary")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05j-d dataset fingerprint mismatch")
        if report.get("passing_families"):
            blockers.append("05j-d unexpectedly contains a passing family")
        if report.get("methodology", {}).get("development_used_for_checkpoint_selection"):
            blockers.append("05j-d used development for checkpoint selection")
        if report.get("methodology", {}).get("rollout_performed"):
            blockers.append("05j-d unexpectedly performed rollout")
        if report.get("heldout_contract", {}).get("inputs_extracted"):
            blockers.append("05j-d heldout inputs were not sealed")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05j-d artifact verification is incomplete")
        if blockers:
            raise RuntimeError(f"05j-e provenance blockers: {blockers}")
        self.artifact_05jd_report = report
        self.artifact_05jd_contract = contract
        payload = {
            "schema_version": "05j-e-architecture-reassessment-config-v1",
            "architecture_reassessment": asdict(self.reassessment),
            "artifact_05jd": contract,
            "mode": "frozen_checkpoint_forensics_no_retraining",
            "selection_roles": [], "development_used_for_model_selection": False,
            "heldout_inputs_extracted": False, "rollout_performed": False,
            "full_training_authorized": False, "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "architecture_reassessment_config.json", payload)
        return {**base, **payload}

    def _checkpoint_bytes(self, relative_path: str) -> bytes:
        if self.artifact_05jd_source.is_file():
            with zipfile.ZipFile(self.artifact_05jd_source) as archive:
                return archive.read(self._artifact_05jd_zip_root + relative_path)
        if self._artifact_05jd_root is None:
            raise RuntimeError("05j-d extracted root is unavailable")
        return (self._artifact_05jd_root / relative_path).read_bytes()

    def reconstruct_frozen_checkpoints(self) -> Dict[str, Any]:
        require_torch()
        if not self.topology_designs:
            raise RuntimeError("prepare_topology_canary_designs() must run first")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        registered_runs = self.artifact_05jd_report["trainable_topology_canary"]["runs"]
        reconstructed = []
        progress = Progress("05j-e frozen checkpoint reconstruction", len(registered_runs))
        for position, row in enumerate(registered_runs):
            family, seed = str(row["family"]), int(row["seed"])
            checkpoint = torch.load(
                io.BytesIO(self._checkpoint_bytes(str(row["checkpoint"]))),
                map_location=device, weights_only=False,
            )
            if checkpoint["family"] != family or int(checkpoint["seed"]) != seed:
                raise RuntimeError("05j-d checkpoint identity mismatch")
            model = TrainableTopologyResidualHead(
                self.topology_designs["fit"].shape[-1], self.layout.segment_count,
                self.topology.hidden_width, self.topology.segment_embedding_dim,
                self.topology.target_residual_limit_mv,
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"]); model.eval()
            self.frozen_predictions.setdefault(family, {}).setdefault(seed, {})
            maximum_metric_error = 0.0
            with torch.no_grad():
                for role in ("fit", "calibration", "development"):
                    features = torch.as_tensor(self.topology_designs[role], device=device)
                    baseline = (
                        torch.as_tensor(self.ridge_predictions[role], device=device)
                        if family == "ridge_corrected_tree" else None
                    )
                    residual = model(features, baseline).cpu().numpy()
                    self.frozen_predictions[family][seed][role] = residual
                    metrics = self._role_metrics(role, residual)
                    registered = row["roles"][role]
                    for name in (
                        "aggregate_voltage_rmse_mv", "maximum_segment_error_mv",
                        "minimum_branching_retention", "median_branching_retention",
                        "maximum_branching_retention",
                    ):
                        maximum_metric_error = max(
                            maximum_metric_error, abs(float(metrics[name]) - float(registered[name]))
                        )
            reconstructed.append({
                "family": family, "seed": seed,
                "checkpoint": row["checkpoint"],
                "maximum_registered_metric_error": maximum_metric_error,
                "valid": maximum_metric_error <= self.reassessment.metric_atol,
            })
            progress.update(position + 1, f"{family} seed={seed} error={maximum_metric_error:.3g}")
            del model
        report = {
            "schema_version": "05j-e-checkpoint-reconstruction-v1",
            "valid": all(row["valid"] for row in reconstructed),
            "device": str(device), "runs": reconstructed,
            "retraining_performed": False,
            "development_used_for_model_selection": False,
            "heldout_inputs_extracted": False,
        }
        _write_json(self.output_dir / "checkpoint_reconstruction.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-e checkpoint reconstruction disagrees with 05j-d")
        return report

    def _ensemble_voltage(self, role: str) -> Tuple[np.ndarray, np.ndarray]:
        residuals = np.stack([
            self.frozen_predictions[self.reassessment.audited_family][seed][role]
            for seed in self.reassessment.seeds
        ])
        voltage = np.asarray(self.topology_roles[role]["base"])[None] + residuals
        return voltage.mean(axis=0), voltage.std(axis=0)

    def run_error_anatomy(self) -> Dict[str, Any]:
        if not self.frozen_predictions:
            raise RuntimeError("reconstruct_frozen_checkpoints() must run first")
        segment_rows, region_rows, pair_rows, role_reports = [], [], [], {}
        region_names = np.asarray([
            str(row.get("region", "unknown")) for row in self.layout.segments
        ])
        parents = np.asarray(self.arrays["parent_ids"], dtype=np.int64)
        axial = np.asarray(self.arrays["axial_conductance_to_parent_us"], dtype=np.float64)
        for role in ("fit", "calibration", "development"):
            prediction, disagreement = self._ensemble_voltage(role)
            target = np.asarray(self.topology_roles[role]["target"])
            error = prediction - target
            metrics = self._pair_set_metrics(
                prediction.reshape(-1, 2, self.layout.segment_count),
                target.reshape(-1, 2, self.layout.segment_count),
            )
            concentration = error_energy_concentration(
                error, self.reassessment.concentrated_segment_fraction
            )
            smooth = axial_tree_diffusion(
                error[..., None], parents, axial, [1], self.spatial.diffusion_self_weight
            )[..., 0]
            high = error - smooth
            total_energy = max(float(np.sum(error * error)), 1e-12)
            high_fraction = float(np.sum(high * high) / total_energy)
            disagreement_rmse = float(np.sqrt(np.mean(disagreement * disagreement)))
            ensemble_rmse = float(np.sqrt(np.mean(error * error)))
            role_reports[role] = {
                "metrics": metrics,
                "ensemble_seed_disagreement_rmse_mv": disagreement_rmse,
                "disagreement_to_error_ratio": disagreement_rmse / max(ensemble_rmse, 1e-12),
                "high_frequency_error_energy_fraction": high_fraction,
                **concentration,
            }
            indices = np.asarray(self.topology_roles[role]["indices"], dtype=np.int64)
            voltage_t = np.asarray(self.topology_roles[role]["voltage_t"])
            for pair, pair_metrics in enumerate(metrics["pair_metrics"]):
                sample = slice(2 * pair, 2 * pair + 2)
                families = sorted({
                    self._protocol_family(int(indices[2 * pair])),
                    self._protocol_family(int(indices[2 * pair + 1])),
                })
                pair_error = error[sample]
                pair_rows.append({
                    "role": role, "pair_position": pair,
                    "protocol_family": " <> ".join(families),
                    "voltage_rmse_mv": float(pair_metrics["voltage_rmse_mv"]),
                    "maximum_segment_error_mv": float(pair_metrics["maximum_segment_error_mv"]),
                    "branching_retention": float(pair_metrics["branching_retention"]),
                    "pair_passed": bool(pair_metrics["passed"]),
                    "target_peak_mv": float(np.max(target[sample])),
                    "maximum_target_delta_mv": float(np.max(np.abs(target[sample] - voltage_t[sample]))),
                    "regenerative_regime": bool(np.max(target[sample]) >= -20.0),
                    "error_energy": float(np.sum(pair_error * pair_error)),
                })
            for segment in range(self.layout.segment_count):
                segment_rows.append({
                    "role": role, "segment_id": segment,
                    "region": region_names[segment],
                    "rmse_mv": float(np.sqrt(np.mean(error[:, segment] ** 2))),
                    "maximum_absolute_error_mv": float(np.max(np.abs(error[:, segment]))),
                    "mean_error_mv": float(np.mean(error[:, segment])),
                    "seed_disagreement_rmse_mv": float(np.sqrt(np.mean(disagreement[:, segment] ** 2))),
                    "error_energy": float(np.sum(error[:, segment] ** 2)),
                })
            for region in sorted(set(region_names.tolist())):
                chosen = region_names == region
                region_error = error[:, chosen]
                region_rows.append({
                    "role": role, "region": region, "segment_count": int(np.sum(chosen)),
                    "rmse_mv": float(np.sqrt(np.mean(region_error ** 2))),
                    "maximum_absolute_error_mv": float(np.max(np.abs(region_error))),
                    "error_energy_fraction": float(np.sum(region_error ** 2) / total_energy),
                })
        report = {
            "schema_version": "05j-e-error-anatomy-v1", "valid": True,
            "audited_family": self.reassessment.audited_family,
            "roles": role_reports,
            "systematic_consensus_on_calibration": role_reports["calibration"]["disagreement_to_error_ratio"] <= self.reassessment.systematic_disagreement_ratio,
            "systematic_consensus_on_development": role_reports["development"]["disagreement_to_error_ratio"] <= self.reassessment.systematic_disagreement_ratio,
            "development_high_frequency_dominant": role_reports["development"]["high_frequency_error_energy_fraction"] >= self.reassessment.high_frequency_energy_fraction,
            "development_segment_concentrated": role_reports["development"]["top_segment_error_energy_fraction"] >= self.reassessment.concentrated_error_energy_fraction,
            "retraining_performed": False, "heldout_inputs_extracted": False,
        }
        protocol_summary = []
        for role in ("fit", "calibration", "development"):
            role_rows = [row for row in pair_rows if row["role"] == role]
            for family in sorted({row["protocol_family"] for row in role_rows}):
                chosen = [row for row in role_rows if row["protocol_family"] == family]
                protocol_summary.append({
                    "role": role, "protocol_family": family,
                    "pair_count": len(chosen),
                    "median_voltage_rmse_mv": float(np.median([row["voltage_rmse_mv"] for row in chosen])),
                    "maximum_segment_error_mv": float(max(row["maximum_segment_error_mv"] for row in chosen)),
                    "median_branching_retention": float(np.median([row["branching_retention"] for row in chosen])),
                    "regenerative_pair_count": int(sum(row["regenerative_regime"] for row in chosen)),
                    "error_energy": float(sum(row["error_energy"] for row in chosen)),
                })
        _write_json(self.output_dir / "error_anatomy.json", report)
        write_parquet(self.output_dir / "error_by_segment.parquet", segment_rows)
        write_parquet(self.output_dir / "error_by_region.parquet", region_rows)
        write_parquet(self.output_dir / "error_by_pair.parquet", pair_rows)
        write_parquet(self.output_dir / "error_by_protocol.parquet", protocol_summary)
        return report

    def run_bias_and_capacity_controls(self) -> Dict[str, Any]:
        ensemble = {role: self._ensemble_voltage(role)[0] for role in ("fit", "calibration", "development")}
        targets = {role: np.asarray(self.topology_roles[role]["target"]) for role in ensemble}
        slope, intercept = fit_segment_affine_calibrator(
            ensemble["fit"], targets["fit"], self.reassessment.affine_ridge
        )
        controls = {}
        for role in ensemble:
            raw_metrics = self._pair_set_metrics(
                ensemble[role].reshape(-1, 2, self.layout.segment_count),
                targets[role].reshape(-1, 2, self.layout.segment_count),
            )
            calibrated = apply_segment_affine(ensemble[role], slope, intercept)
            affine_metrics = self._pair_set_metrics(
                calibrated.reshape(-1, 2, self.layout.segment_count),
                targets[role].reshape(-1, 2, self.layout.segment_count),
            )
            controls[role] = {
                "raw_ensemble": raw_metrics, "fit_only_segment_affine": affine_metrics,
                "rmse_improvement_fraction": float(
                    (raw_metrics["aggregate_voltage_rmse_mv"] - affine_metrics["aggregate_voltage_rmse_mv"])
                    / max(raw_metrics["aggregate_voltage_rmse_mv"], 1e-12)
                ),
                "maximum_error_improvement_fraction": float(
                    (raw_metrics["maximum_segment_error_mv"] - affine_metrics["maximum_segment_error_mv"])
                    / max(raw_metrics["maximum_segment_error_mv"], 1e-12)
                ),
            }
        material = self.reassessment.material_improvement_fraction
        calibration_repair = controls["calibration"]["rmse_improvement_fraction"] >= material
        development_repair = controls["development"]["rmse_improvement_fraction"] >= material
        report = {
            "schema_version": "05j-e-bias-capacity-controls-v1", "valid": True,
            "fit_only_segment_affine": {
                "ridge": self.reassessment.affine_ridge,
                "slope_minimum": float(np.min(slope)), "slope_median": float(np.median(slope)),
                "slope_maximum": float(np.max(slope)),
                "intercept_minimum_mv": float(np.min(intercept)),
                "intercept_median_mv": float(np.median(intercept)),
                "intercept_maximum_mv": float(np.max(intercept)),
            },
            "roles": controls,
            "static_affine_repair_material_on_calibration_and_development": bool(calibration_repair and development_repair),
            "development_used_to_fit_calibrator": False,
            "calibration_used_to_fit_calibrator": False,
            "diagnostic_oracle_only": True, "candidate_authorization": False,
            "retraining_performed": False, "heldout_inputs_extracted": False,
        }
        _write_json(self.output_dir / "bias_and_capacity_controls.json", report)
        return report

    def run_branch_identifiability_audit(self) -> Dict[str, Any]:
        fit_x = self.topology_designs["fit"].reshape(-1, 2, self.layout.segment_count, self.topology_designs["fit"].shape[-1])
        fit_distance = np.linalg.norm(fit_x[:, 0] - fit_x[:, 1], axis=-1) / math.sqrt(fit_x.shape[-1])
        threshold = float(np.quantile(fit_distance, self.reassessment.collision_quantile))
        roles, rows = {}, []
        for role in ("fit", "calibration", "development"):
            x = self.topology_designs[role].reshape(-1, 2, self.layout.segment_count, self.topology_designs[role].shape[-1])
            feature_distance = np.linalg.norm(x[:, 0] - x[:, 1], axis=-1) / math.sqrt(x.shape[-1])
            target = np.asarray(self.topology_roles[role]["target"]).reshape(-1, 2, self.layout.segment_count)
            target_branch = np.abs(target[:, 0] - target[:, 1])
            prediction, _ = self._ensemble_voltage(role)
            prediction = prediction.reshape(-1, 2, self.layout.segment_count)
            prediction_branch = np.abs(prediction[:, 0] - prediction[:, 1])
            collision = feature_distance <= threshold
            large = target_branch >= self.reassessment.large_branch_target_mv
            roles[role] = {
                "pair_count": int(len(x)),
                "feature_distance_target_branch_correlation": _correlation(feature_distance, target_branch),
                "prediction_target_branch_correlation": _correlation(prediction_branch, target_branch),
                "collision_point_count": int(np.sum(collision)),
                "large_target_collision_count": int(np.sum(collision & large)),
                "large_target_collision_fraction": float(np.mean(collision & large)),
                "large_target_given_collision_fraction": float(np.sum(collision & large) / max(np.sum(collision), 1)),
                "median_target_branch_mv_in_collision_tail": float(np.median(target_branch[collision])) if np.any(collision) else None,
                "maximum_target_branch_mv_in_collision_tail": float(np.max(target_branch[collision])) if np.any(collision) else None,
            }
            for pair in range(len(x)):
                rows.append({
                    "role": role, "pair_position": pair,
                    "median_feature_branch_distance": float(np.median(feature_distance[pair])),
                    "maximum_target_branch_mv": float(np.max(target_branch[pair])),
                    "maximum_target_branch_mv_in_collision_tail": float(np.max(target_branch[pair][collision[pair]])) if np.any(collision[pair]) else None,
                    "collision_segment_count": int(np.sum(collision[pair])),
                    "large_target_collision_segment_count": int(np.sum(collision[pair] & large[pair])),
                })
        report = {
            "schema_version": "05j-e-branch-identifiability-v1", "valid": True,
            "feature_distance_definition": "RMS difference of the 226 fit-normalized multiscale-tree features between paired futures",
            "collision_quantile_fit": self.reassessment.collision_quantile,
            "collision_distance_threshold": threshold,
            "large_branch_target_mv": self.reassessment.large_branch_target_mv,
            "minimum_large_target_given_collision_fraction": self.reassessment.minimum_large_target_given_collision_fraction,
            "roles": roles,
            "normalization_fit_roles": ["fit"],
            "retraining_performed": False, "heldout_inputs_extracted": False,
        }
        _write_json(self.output_dir / "branch_identifiability_audit.json", report)
        write_parquet(self.output_dir / "branch_identifiability_by_pair.parquet", rows)
        return report

    def finalize_architecture_reassessment(
        self, reconstruction: Mapping[str, Any], anatomy: Mapping[str, Any],
        capacity: Mapping[str, Any], identifiability: Mapping[str, Any],
    ) -> Dict[str, Any]:
        systematic = bool(
            anatomy["systematic_consensus_on_calibration"]
            and anatomy["systematic_consensus_on_development"]
        )
        affine = bool(capacity["static_affine_repair_material_on_calibration_and_development"])
        concentrated = bool(anatomy["development_segment_concentrated"])
        high_frequency = bool(anatomy["development_high_frequency_dominant"])
        collision = bool(
            identifiability["roles"]["calibration"]["large_target_given_collision_fraction"]
            >= self.reassessment.minimum_large_target_given_collision_fraction
            and identifiability["roles"]["development"]["large_target_given_collision_fraction"]
            >= self.reassessment.minimum_large_target_given_collision_fraction
        )
        if affine:
            diagnosis = "STATIC_SEGMENT_CALIBRATION_BIAS_DOMINATES"
            next_experiment = "05j_f_optimization_and_calibration_repair"
        elif collision:
            diagnosis = "TREE_FEATURE_IDENTIFIABILITY_REMAINS_INSUFFICIENT"
            next_experiment = "05j_f_state_and_input_representation_revision"
        elif concentrated or high_frequency:
            diagnosis = "LOCALIZED_MORPHOLOGY_REGIME_ERROR_DOMINATES"
            next_experiment = "05j_f_region_mechanism_expert_revision"
        elif systematic:
            diagnosis = "SYSTEMATIC_DECODER_BIAS_WITHOUT_SIMPLE_AFFINE_REPAIR"
            next_experiment = "05j_f_decoder_objective_revision"
        else:
            diagnosis = "OPTIMIZATION_VARIANCE_AND_ARCHITECTURE_REMAIN_CONFOUNDED"
            next_experiment = "05j_f_controlled_capacity_optimization_grid"
        report = {
            "schema_version": "05j-e-final-report-v1", "valid": True,
            "decision": "FROZEN_CHECKPOINT_ARCHITECTURE_REASSESSMENT",
            "diagnosis": diagnosis, "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05jd": self.artifact_05jd_contract,
            "checkpoint_reconstruction": dict(reconstruction),
            "error_anatomy": dict(anatomy),
            "bias_and_capacity_controls": dict(capacity),
            "branch_identifiability": dict(identifiability),
            "diagnostic_flags": {
                "systematic_seed_consensus": systematic,
                "static_affine_repair_material": affine,
                "large_branch_feature_collisions": collision,
                "segment_error_concentrated": concentrated,
                "high_frequency_error_dominant": high_frequency,
            },
            "canary_passed": False, "micro_rollout_authorized": False,
            "full_training_authorized": False,
            "heldout_contract": {
                "inputs_extracted": False, "boundary_targets_materialized": False,
                "event_targets_materialized": False, "candidate_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "exact_05jd_checkpoints_reconstructed": True,
                "retraining_performed": False,
                "new_checkpoint_selection_performed": False,
                "development_used_for_model_selection": False,
                "diagnostic_fit_roles": ["fit"],
                "heldout_inputs_extracted": False, "rollout_performed": False,
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
                    "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
                })
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05j-e-artifact-index-v1",
            "artifact_count": len(records), "artifacts": records,
        })
        return report
