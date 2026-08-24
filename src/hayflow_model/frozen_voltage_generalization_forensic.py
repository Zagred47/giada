"""06b-h: frozen voltage calibration and activity-regime generalization audit."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .state_scheduled_sampling_confirmation import (
    StateScheduledSamplingConfig,
    StateScheduledSamplingConfirmation,
)


EXPECTED_06BG_ARCHIVE_SHA256 = (
    "5a09397433d941f6d80adf12d6ac7936be9404189bdd2952d88ebd43ef45c4eb"
)
EXPECTED_06BG_INDEX_SHA256 = (
    "94896c60d767fe1d38cd9f848d3d4cdf6df3dcabd38a3afe7c3e1ce675dc2567"
)
EXPECTED_06BG_FINAL_SHA256 = (
    "e78531d7ec6e810e268ffb76ac18548fefeb3836d5481eefb093cac126d490b2"
)

FROZEN_MODEL_ARMS = (
    "source_scalar",
    "scalar_continue",
    "state_linear_curriculum",
    "joint_linear_curriculum",
)


def verified_06bg_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-g source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        stamp = {
            "path": str(source),
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
        marker = cache_dir / ".source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != stamp:
            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            atomic._safe_extract(source, cache_dir)
            marker.write_text(json.dumps(stamp, sort_keys=True), encoding="utf-8")
        search_root = cache_dir
    else:
        search_root = source
    matches = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06BG_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-g artifact; found {len(matches)}")
    root = matches[0]
    index = json.loads((root / "artifact_index.json").read_text(encoding="utf-8"))
    failures = []
    for row in index.get("artifacts", []):
        member = root / str(row["path"])
        if (
            not member.is_file()
            or member.stat().st_size != int(row["size_bytes"])
            or atomic._sha256_file(member) != str(row["sha256"])
        ):
            failures.append(str(row["path"]))
    if failures:
        raise RuntimeError(f"06b-g indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BG_FINAL_SHA256:
        raise RuntimeError("06b-g final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "SCALAR_SECONDARY_SIGNAL_NOT_INDEPENDENTLY_CONFIRMED"
        or final.get("coupled_06c_canary_authorized") is not False
        or final.get("next_step") != "return_to_atomic_STATE_exposure_playground"
    ):
        raise RuntimeError("06b-g result does not authorize a frozen forensic")
    if source.is_file() and archive_hash != EXPECTED_06BG_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BG_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BG_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class FrozenVoltageForensicConfig(StateScheduledSamplingConfig):
    forensic_calibration_components_per_regime: int = 1
    forensic_audit_components_per_regime: int = 1
    forensic_calibration_window_count: int = 16
    forensic_audit_window_count: int = 16
    voltage_shrinkage_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    activity_edges_mv: Tuple[float, ...] = (1.0, 5.0, 20.0)
    minimum_audit_voltage_gain_fraction: float = 0.02
    minimum_active_voltage_gain_fraction: float = 0.02
    minimum_state_gain_fraction: float = 0.0
    material_bias_mv_per_ms: float = 0.05

    def validate(self) -> None:
        super().validate()
        if tuple(sorted(set(self.voltage_shrinkage_grid))) != self.voltage_shrinkage_grid:
            raise ValueError("06b-h shrinkage grid must be sorted and unique")
        if self.voltage_shrinkage_grid[0] != 0.0 or self.voltage_shrinkage_grid[-1] != 1.0:
            raise ValueError("06b-h shrinkage grid must include persistence and raw model")
        if tuple(sorted(self.activity_edges_mv)) != self.activity_edges_mv:
            raise ValueError("06b-h activity edges must be sorted")
        if min(
            self.forensic_calibration_components_per_regime,
            self.forensic_audit_components_per_regime,
            self.forensic_calibration_window_count,
            self.forensic_audit_window_count,
        ) <= 0:
            raise ValueError("06b-h role dimensions must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FrozenVoltageForensicConfig":
        payload = dict(values)
        integer_tuples = (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
            "repair_checkpoints",
            "scheduled_checkpoints",
        )
        for name in integer_tuples:
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        for name in ("voltage_shrinkage_grid", "activity_edges_mv"):
            if name in payload:
                payload[name] = tuple(map(float, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class FrozenVoltageGeneralizationForensic(StateScheduledSamplingConfirmation):
    config: FrozenVoltageForensicConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: FrozenVoltageForensicConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
        artifact_06bd_source: Path,
        artifact_06be_source: Path,
        artifact_06bf_source: Path,
        artifact_06bg_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05t_source,
            artifact_06b_source,
            artifact_06bb_source,
            artifact_06bc_source,
            artifact_06bd_source,
            artifact_06be_source,
            artifact_06bf_source,
            code_revision=code_revision,
        )
        self.artifact_06bg_source = Path(artifact_06bg_source)
        self.frozen_forensic_models: Dict[Tuple[str, int], Tuple[Any, Any]] = {}

    def _build_forensic_roles(self) -> Dict[str, Any]:
        grouped = atomic.disjoint_episode_components_by_regime(
            self.store.episode_rows, role_seed=self.config.role_seed
        )
        upstream = (
            self.config.fit_components_per_regime
            + self.config.calibration_components_per_regime
            + self.config.development_components_per_regime
            + self.config.confirmation_components_per_regime
        )
        counts = {
            "voltage_calibration": self.config.forensic_calibration_components_per_regime,
            "voltage_audit": self.config.forensic_audit_components_per_regime,
        }
        roles = {name: [] for name in counts}
        availability = {}
        for regime, components in sorted(grouped.items()):
            availability[regime] = len(components)
            required = upstream + sum(counts.values())
            if len(components) < required:
                raise RuntimeError(
                    f"06b-h regime {regime!r} has {len(components)} components; "
                    f"{required} required"
                )
            cursor = upstream
            for role, count in counts.items():
                for component in components[cursor : cursor + count]:
                    for source in component:
                        row = dict(source)
                        row["06bh_role"] = role
                        row["06bh_regime"] = regime
                        roles[role].append(row)
                cursor += count
        previous = {
            str(row["trajectory_id"])
            for rows in self.roles.values()
            for row in rows
        }
        calibration = {str(row["trajectory_id"]) for row in roles["voltage_calibration"]}
        audit = {str(row["trajectory_id"]) for row in roles["voltage_audit"]}
        overlaps = {
            "previous:calibration": sorted(previous & calibration),
            "previous:audit": sorted(previous & audit),
            "calibration:audit": sorted(calibration & audit),
        }
        if any(overlaps.values()):
            raise RuntimeError(f"06b-h role leak: {overlaps}")
        if any(
            str(row.get("split")) != "train"
            for rows in roles.values()
            for row in rows
        ):
            raise RuntimeError("06b-h role leaked outside train")
        self.roles.update(roles)
        return {
            "valid": True,
            "available_components_by_regime": availability,
            "upstream_component_prefix_per_regime": upstream,
            "role_component_positions": {
                "voltage_calibration": upstream + 1,
                "voltage_audit": upstream + 2,
            },
            "trajectory_counts": {
                "voltage_calibration": len(calibration),
                "voltage_audit": len(audit),
            },
            "overlaps": overlaps,
            "split": "train",
        }

    def _load_06bg_models(self, root: Path, device: Any) -> Dict[str, str]:
        hashes = {}
        source_names = {
            "scalar_continue": "scalar_continue",
            "state_linear_curriculum": "state_linear_curriculum",
            "joint_linear_curriculum": "joint_linear_curriculum",
        }
        for seed in self.config.pilot_seeds:
            self.frozen_forensic_models[("source_scalar", seed)] = self.source_models[
                ("full_feedback_scalar", seed)
            ]
            for arm, checkpoint_arm in source_names.items():
                name = f"scheduled_{checkpoint_arm}_seed{seed}_step400.pt"
                path = root / name
                checkpoint = atomic.torch.load(
                    path, map_location=device, weights_only=False
                )
                if (
                    str(checkpoint.get("arm")) != checkpoint_arm
                    or int(checkpoint.get("seed", -1)) != seed
                    or int(checkpoint.get("budget", -1)) != 400
                ):
                    raise RuntimeError(f"06b-h checkpoint mismatch: {name}")
                pair = self._new_pair(seed, device)
                pair[0].load_state_dict(copy.deepcopy(checkpoint["bridge_state_dict"]))
                pair[1].load_state_dict(copy.deepcopy(checkpoint["STATE_state_dict"]))
                for model in pair:
                    model.eval()
                    for parameter in model.parameters():
                        parameter.requires_grad_(False)
                self.frozen_forensic_models[(arm, seed)] = pair
                hashes[name] = atomic._sha256_file(path)
        return hashes

    def prepare_frozen_voltage_forensic(self) -> Dict[str, Any]:
        base = self.prepare_scheduled_sampling_confirmation()
        # 06b-h performs no continuation training.  Do not propagate inherited
        # planning fields as if they described the frozen forensic itself.
        for stale_name in (
            "continuation_arms",
            "continuation_checkpoints",
            "continuation_unroll_horizon_ms",
            "continuation_training_planned",
            "training_stage_at_contract_write",
            "training_horizon_ms",
            "joint_objective_backpropagates_through_trainable_STATE_updater",
        ):
            base.pop(stale_name, None)
        source_root, source = verified_06bg_artifact_root(
            self.artifact_06bg_source,
            self.output_dir.parent / ".06bh_artifact_cache" / "06bg",
        )
        roles = self._build_forensic_roles()
        self._materialize_window_role(
            "voltage_calibration",
            self.config.forensic_calibration_window_count,
            max(self.config.rollout_horizons_ms),
        )
        self._materialize_window_role(
            "voltage_audit",
            self.config.forensic_audit_window_count,
            max(self.config.rollout_horizons_ms),
        )
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        checkpoint_hashes = self._load_06bg_models(source_root, device)
        report = {
            **base,
            "schema_version": "06b-h-frozen-voltage-contract-v1",
            "experiment": "frozen_voltage_generalization_forensic",
            "source_06bg": source,
            "frozen_checkpoint_sha256": checkpoint_hashes,
            "frozen_model_arms": list(FROZEN_MODEL_ARMS),
            "voltage_shrinkage_grid": list(self.config.voltage_shrinkage_grid),
            "global_bias_options": ["none", "pooled_calibration_residual"],
            "activity_edges_mv": list(self.config.activity_edges_mv),
            "forensic_roles": roles,
            "neural_training_performed": False,
            "candidate_selection_role": "unused_train_voltage_calibration",
            "decision_role": "unused_train_voltage_audit",
            "audit_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
        }
        atomic._write_json(self.output_dir / "frozen_voltage_contract.json", report)
        return report

    def _frozen_rollout(
        self,
        pair: Tuple[Any, Any],
        role: str,
        alpha: float,
        bias_mv: float,
        device: Any,
    ) -> Dict[str, Any]:
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        state_center = atomic.torch.as_tensor(
            self.statistics["state_center"], device=device
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], device=device
        )
        delta_scale = atomic.torch.as_tensor(
            self.statistics["delta_scale"], device=device
        )
        steps = []
        with atomic.torch.no_grad():
            for step in range(max(self.config.rollout_horizons_ms)):
                context = atomic.torch.cat(
                    (batch["drive"][:, step], batch["held_ions"]), dim=-1
                )
                normalized = (current_state - state_center) / state_scale
                raw_delta = self._bridge_forward(
                    pair[0], normalized, current_voltage, context
                )
                voltage_delta = float(alpha) * raw_delta + float(bias_mv)
                next_voltage = current_voltage + voltage_delta
                state_delta = self._state_forward(
                    pair[1], normalized, current_voltage, voltage_delta, context
                )
                next_state = current_state + state_delta * delta_scale
                steps.append(
                    {
                        "state": next_state.detach().cpu().numpy(),
                        "voltage": next_voltage.detach().cpu().numpy(),
                        "target_state": batch["state_t1"][:, step].cpu().numpy(),
                        "target_voltage": batch["voltage_t1"][:, step].cpu().numpy(),
                        "initial_state": batch["state_t"][:, 0].cpu().numpy(),
                        "initial_voltage": batch["voltage_t"][:, 0].cpu().numpy(),
                    }
                )
                current_state, current_voltage = next_state, next_voltage
        return {"steps": steps, "alpha": float(alpha), "bias_mv_per_ms": float(bias_mv)}

    def _rollout_metrics(self, rollout: Mapping[str, Any]) -> Dict[str, Any]:
        horizons = {}
        for horizon in self.config.rollout_horizons_ms:
            row = rollout["steps"][horizon - 1]
            horizons[f"{horizon}_ms"] = self._horizon_metrics(
                row["state"],
                row["target_state"],
                row["initial_state"],
                row["voltage"],
                row["target_voltage"],
                row["initial_voltage"],
            )
        return horizons

    def calibrate_frozen_voltage(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        fitted_bias = {}
        raw_metrics = {}
        prefit_total = len(FROZEN_MODEL_ARMS) * len(self.config.voltage_shrinkage_grid)
        prefit_progress = atomic._CompactProgress(
            "06b-h frozen bias prefit", prefit_total, max(1, prefit_total // 10)
        )
        prefit_completed = 0
        for arm in FROZEN_MODEL_ARMS:
            for alpha in self.config.voltage_shrinkage_grid:
                residuals = []
                for seed in self.config.pilot_seeds:
                    rollout = self._frozen_rollout(
                        self.frozen_forensic_models[(arm, seed)],
                        "voltage_calibration",
                        alpha,
                        0.0,
                        device,
                    )
                    raw_metrics[(arm, alpha, seed)] = self._rollout_metrics(rollout)[
                        "8_ms"
                    ]
                    # A single correction is added at every macro-step.  Divide
                    # endpoint residuals by elapsed steps before pooling so the
                    # fitted quantity has units of mV per ms rather than mV.
                    residuals.extend(
                        (
                            (step["target_voltage"] - step["voltage"])
                            / float(step_index)
                        ).reshape(-1)
                        for step_index, step in enumerate(rollout["steps"], start=1)
                    )
                fitted_bias[(arm, alpha)] = float(np.mean(np.concatenate(residuals)))
                prefit_completed += 1
                prefit_progress.update(
                    prefit_completed, f"arm={arm} alpha={alpha:g}"
                )
        candidates = []
        total = len(FROZEN_MODEL_ARMS) * len(self.config.voltage_shrinkage_grid) * 2
        progress = atomic._CompactProgress(
            "06b-h frozen calibration", total, max(1, total // 10)
        )
        completed = 0
        for arm in FROZEN_MODEL_ARMS:
            for alpha in self.config.voltage_shrinkage_grid:
                for bias_mode in ("none", "pooled_calibration_residual"):
                    bias = 0.0 if bias_mode == "none" else fitted_bias[(arm, alpha)]
                    seed_metrics = {}
                    for seed in self.config.pilot_seeds:
                        if bias_mode == "none":
                            seed_metrics[str(seed)] = raw_metrics[(arm, alpha, seed)]
                        else:
                            rollout = self._frozen_rollout(
                                self.frozen_forensic_models[(arm, seed)],
                                "voltage_calibration",
                                alpha,
                                bias,
                                device,
                            )
                            seed_metrics[str(seed)] = self._rollout_metrics(rollout)[
                                "8_ms"
                            ]
                    ratios = [
                        row["voltage_rmse_mv"]
                        / max(row["persistence_voltage_rmse_mv"], 1e-12)
                        for row in seed_metrics.values()
                    ]
                    state_gains = [
                        row["state_improvement_vs_persistence_fraction"]
                        for row in seed_metrics.values()
                    ]
                    candidates.append(
                        {
                            "model_arm": arm,
                            "alpha": float(alpha),
                            "bias_mode": bias_mode,
                            "bias_mv_per_ms": bias,
                            "median_voltage_error_ratio": float(np.median(ratios)),
                            "median_state_gain": float(np.median(state_gains)),
                            "all_seed_state_gain_positive": all(value > 0 for value in state_gains),
                            "seed_metrics": seed_metrics,
                        }
                    )
                    completed += 1
                    progress.update(completed, f"arm={arm} alpha={alpha:g}")
        eligible = [row for row in candidates if row["all_seed_state_gain_positive"]]
        if not eligible:
            raise RuntimeError("06b-h calibration found no STATE-safe frozen candidate")
        selected = min(
            eligible,
            key=lambda row: (
                row["median_voltage_error_ratio"],
                abs(row["bias_mv_per_ms"]),
                -row["alpha"],
                row["model_arm"],
            ),
        )
        report = {
            "schema_version": "06b-h-frozen-calibration-v1",
            "valid": True,
            "candidate_count": len(candidates),
            "eligible_candidate_count": len(eligible),
            "selection_metric": "median eight-ms voltage RMSE divided by persistence RMSE",
            "STATE_safety_constraint": "positive eight-ms STATE gain in every seed",
            "selected": selected,
            "candidates": candidates,
            "audit_accessed": False,
            "neural_training_performed": False,
        }
        atomic._write_json(self.output_dir / "frozen_voltage_calibration.json", report)
        return report

    def _activity_metrics(self, row: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        magnitude = np.abs(row["target_voltage"] - row["initial_voltage"])
        edges = self.config.activity_edges_mv
        masks = {
            "quiescent_lt_1mV": magnitude < edges[0],
            "moderate_1_to_5mV": (magnitude >= edges[0]) & (magnitude < edges[1]),
            "active_5_to_20mV": (magnitude >= edges[1]) & (magnitude < edges[2]),
            "regenerative_ge_20mV": magnitude >= edges[2],
            "active_ge_5mV": magnitude >= edges[1],
        }
        result = {}
        error = row["voltage"] - row["target_voltage"]
        persistence = row["initial_voltage"] - row["target_voltage"]
        for name, mask in masks.items():
            count = int(np.sum(mask))
            if count == 0:
                result[name] = {"count": 0, "voltage_gain_vs_persistence_fraction": None}
                continue
            rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
            baseline = float(np.sqrt(np.mean(persistence[mask] ** 2)))
            result[name] = {
                "count": count,
                "voltage_rmse_mv": rmse,
                "persistence_voltage_rmse_mv": baseline,
                "voltage_gain_vs_persistence_fraction": 1.0 - rmse / max(baseline, 1e-12),
                "mean_drift_mv": float(np.mean(error[mask])),
            }
        return result

    def _region_metrics(self, row: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        error = row["voltage"] - row["target_voltage"]
        persistence = row["initial_voltage"] - row["target_voltage"]
        result = {}
        for region_id, region_name in enumerate(self.layout.region_names):
            segment_mask = self.layout.segment_region_ids == region_id
            count = int(np.sum(segment_mask) * error.shape[0])
            if count == 0:
                continue
            rmse = float(np.sqrt(np.mean(error[:, segment_mask] ** 2)))
            baseline = float(np.sqrt(np.mean(persistence[:, segment_mask] ** 2)))
            result[str(region_name)] = {
                "count": count,
                "voltage_rmse_mv": rmse,
                "persistence_voltage_rmse_mv": baseline,
                "voltage_gain_vs_persistence_fraction": 1.0
                - rmse / max(baseline, 1e-12),
                "mean_drift_mv": float(np.mean(error[:, segment_mask])),
            }
        return result

    def audit_frozen_voltage(self, calibration: Mapping[str, Any]) -> Dict[str, Any]:
        selected = calibration["selected"]
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed = {}
        for seed in self.config.pilot_seeds:
            rollout = self._frozen_rollout(
                self.frozen_forensic_models[(selected["model_arm"], seed)],
                "voltage_audit",
                selected["alpha"],
                selected["bias_mv_per_ms"],
                device,
            )
            per_seed[str(seed)] = {
                "horizons": self._rollout_metrics(rollout),
                "activity_at_8ms": self._activity_metrics(rollout["steps"][7]),
                "region_at_8ms": self._region_metrics(rollout["steps"][7]),
            }
        report = {
            "schema_version": "06b-h-frozen-audit-v1",
            "valid": all(
                metric["nonfinite_state_count"] == 0
                and metric["nonfinite_voltage_count"] == 0
                and metric["state_domain_violation_count"] == 0
                for seed in per_seed.values()
                for metric in seed["horizons"].values()
            ),
            "selected_from_calibration": {
                key: selected[key]
                for key in ("model_arm", "alpha", "bias_mode", "bias_mv_per_ms")
            },
            "audit_role_used_for_selection": False,
            "neural_training_performed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "frozen_voltage_audit.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        if not finite:
            raise RuntimeError("06b-h cannot compute a decision median from an empty bin")
        return float(np.median(np.asarray(finite, dtype=np.float64)))

    def finalize_frozen_voltage_forensic(
        self, calibration: Mapping[str, Any], audit: Mapping[str, Any]
    ) -> Dict[str, Any]:
        endpoint = [row["horizons"]["8_ms"] for row in audit["per_seed"].values()]
        voltage_gains = [row["voltage_improvement_vs_persistence_fraction"] for row in endpoint]
        state_gains = [row["state_improvement_vs_persistence_fraction"] for row in endpoint]
        violations = [row["physical_voltage_violation_count"] for row in endpoint]
        active = [
            row["activity_at_8ms"]["active_ge_5mV"]["voltage_gain_vs_persistence_fraction"]
            for row in audit["per_seed"].values()
        ]
        quiescent = [
            row["activity_at_8ms"]["quiescent_lt_1mV"]["voltage_gain_vs_persistence_fraction"]
            for row in audit["per_seed"].values()
        ]
        overall_pass = (
            self._median(voltage_gains) >= self.config.minimum_audit_voltage_gain_fraction
            and all(value > 0 for value in voltage_gains)
            and all(value > self.config.minimum_state_gain_fraction for value in state_gains)
            and sum(violations) == 0
        )
        active_complete = all(value is not None for value in active)
        quiescent_complete = all(value is not None for value in quiescent)
        active_median = self._median(active) if active_complete else None
        quiescent_median = self._median(quiescent) if quiescent_complete else None
        active_pass = bool(
            active_complete
            and active_median >= self.config.minimum_active_voltage_gain_fraction
        )
        quiescent_failure = bool(
            quiescent_complete and quiescent_median < 0
        )
        selected = calibration["selected"]
        amplitude_material = float(selected["alpha"]) < 0.99
        bias_material = abs(float(selected["bias_mv_per_ms"])) >= self.config.material_bias_mv_per_ms
        if overall_pass and (amplitude_material or bias_material):
            diagnosis = "FROZEN_VOLTAGE_CALIBRATION_RESCUES_GENERALIZATION"
            next_step = "train_only_voltage_objective_recalibration"
        elif active_pass and quiescent_failure:
            diagnosis = "ACTIVITY_IMBALANCED_VOLTAGE_OBJECTIVE"
            next_step = "train_only_activity_stratified_voltage_objective"
        else:
            diagnosis = "VOLTAGE_REPRESENTATION_GENERALIZATION_FAILURE"
            next_step = "atomic_voltage_representation_playground"
        report = {
            "schema_version": "06b-h-final-report-v1",
            "valid": bool(calibration.get("valid") and audit.get("valid")),
            "component_decision_grade": True,
            "diagnosis": diagnosis,
            "selected_frozen_candidate": {
                key: selected[key]
                for key in ("model_arm", "alpha", "bias_mode", "bias_mv_per_ms")
            },
            "audit_medians": {
                "voltage_gain_vs_persistence_fraction": self._median(voltage_gains),
                "STATE_gain_vs_persistence_fraction": self._median(state_gains),
                "active_ge_5mV_voltage_gain_fraction": active_median,
                "quiescent_lt_1mV_voltage_gain_fraction": quiescent_median,
                "physical_voltage_violation_count": int(sum(violations)),
            },
            "gate_checks": {
                "overall_absolute_voltage_gate": overall_pass,
                "active_voltage_gate": active_pass,
                "quiescent_failure": quiescent_failure,
                "amplitude_correction_material": amplitude_material,
                "bias_correction_material": bias_material,
            },
            "calibration_role_used_for_selection": True,
            "audit_role_used_for_selection": False,
            "neural_training_performed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BG_ARCHIVE_SHA256",
    "EXPECTED_06BG_FINAL_SHA256",
    "EXPECTED_06BG_INDEX_SHA256",
    "FROZEN_MODEL_ARMS",
    "FrozenVoltageForensicConfig",
    "FrozenVoltageGeneralizationForensic",
    "verified_06bg_artifact_root",
]
