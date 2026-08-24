"""06b-l: persistence-versus-dynamic voltage error-model revision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .temporal_voltage_correction_state import (
    TemporalVoltageCorrectionConfig,
    TemporalVoltageCorrectionState,
)


EXPECTED_06BK_ARCHIVE_SHA256 = (
    "fe22d8f135257d58ae844c0e8db5c1ba52f7eab9b3f0005722ff6c1a5f1ea33c"
)
EXPECTED_06BK_INDEX_SHA256 = (
    "3536b6e9ada1ebecd725e5956fec6d8c928074ba839c4dcd88d513e94805f45f"
)
EXPECTED_06BK_FINAL_SHA256 = (
    "3df8d38762a8cdd56b84e964adc66ca111053cba81b396981d478813aa53afb4"
)

STATIC_REFERENCE = "static_dynamic_update"
CAUSAL_GATE_SCHEMES = (
    "hard_hurdle_instantaneous",
    "hard_hurdle_temporal",
    "soft_blend_instantaneous",
    "soft_blend_temporal",
)
TEACHER_REGIME_ORACLE = "teacher_regime_oracle"
TEACHER_OPTIMAL_BLEND_ORACLE = "teacher_optimal_blend_oracle"
ORACLE_SCHEMES = (TEACHER_REGIME_ORACLE, TEACHER_OPTIMAL_BLEND_ORACLE)
PRIMARY_SCHEME = "hard_hurdle_instantaneous"
FALLBACK_SCHEME = "soft_blend_instantaneous"


def verified_06bk_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-k source must be a ZIP or extracted directory")
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
        if atomic._sha256_file(path) == EXPECTED_06BK_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-k artifact; found {len(matches)}")
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
        raise RuntimeError(f"06b-k indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BK_FINAL_SHA256:
        raise RuntimeError("06b-k final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "TEMPORAL_CORRECTION_STATE_NOT_IDENTIFIED"
        or final.get("next_step") != "atomic_voltage_error_model_revision"
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-k result does not authorize 06b-l")
    if source.is_file() and archive_hash != EXPECTED_06BK_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BK_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BK_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class VoltageErrorModelRevisionConfig(TemporalVoltageCorrectionConfig):
    gate_ridge_strengths: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    hurdle_probability_thresholds: Tuple[float, ...] = (0.25, 0.5, 0.75)
    teacher_quiescent_threshold_mv: float = 1.0
    minimum_recursive_gain_over_static_fraction: float = 0.02

    def validate(self) -> None:
        super().validate()
        if tuple(sorted(set(self.gate_ridge_strengths))) != self.gate_ridge_strengths:
            raise ValueError("06b-l gate ridge strengths must be sorted and unique")
        if tuple(sorted(set(self.hurdle_probability_thresholds))) != self.hurdle_probability_thresholds:
            raise ValueError("06b-l hurdle thresholds must be sorted and unique")
        if not all(0 < value < 1 for value in self.hurdle_probability_thresholds):
            raise ValueError("06b-l hurdle thresholds must lie in (0, 1)")
        if self.teacher_quiescent_threshold_mv <= 0:
            raise ValueError("06b-l teacher quiescent threshold must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "VoltageErrorModelRevisionConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
            "repair_checkpoints",
            "scheduled_checkpoints",
            "objective_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        for name in (
            "voltage_shrinkage_grid",
            "activity_edges_mv",
            "analytic_shrinkage_strengths",
            "analytic_voltage_edges_mv",
            "temporal_ridge_strengths",
            "gate_ridge_strengths",
            "hurdle_probability_thresholds",
        ):
            if name in payload:
                payload[name] = tuple(map(float, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class VoltageErrorModelRevision(TemporalVoltageCorrectionState):
    config: VoltageErrorModelRevisionConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: VoltageErrorModelRevisionConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
        artifact_06bd_source: Path,
        artifact_06be_source: Path,
        artifact_06bf_source: Path,
        artifact_06bg_source: Path,
        artifact_06bh_source: Path,
        artifact_06bi_source: Path,
        artifact_06bj_source: Path,
        artifact_06bk_source: Path,
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
            artifact_06bg_source,
            artifact_06bh_source,
            artifact_06bi_source,
            artifact_06bj_source,
            code_revision=code_revision,
        )
        self.artifact_06bk_source = Path(artifact_06bk_source)
        self.gate_models: Dict[Tuple[int, str, float], Dict[str, Any]] = {}
        self.selected_gate: Dict[Tuple[int, str], Tuple[float, float | None]] = {}
        self.gate_calibration_valid = False

    def prepare_voltage_error_model_revision(self) -> Dict[str, Any]:
        base = self.prepare_temporal_correction_state()
        _, source = verified_06bk_artifact_root(
            self.artifact_06bk_source,
            self.output_dir.parent / ".06bl_artifact_cache" / "06bk",
        )
        report = {
            **base,
            "schema_version": "06b-l-error-model-contract-v1",
            "experiment": "voltage_error_model_revision",
            "source_06bk": source,
            "static_reference": STATIC_REFERENCE,
            "causal_gate_schemes": list(CAUSAL_GATE_SCHEMES),
            "primary_scheme": PRIMARY_SCHEME,
            "fallback_scheme": FALLBACK_SCHEME,
            "oracle_schemes": list(ORACLE_SCHEMES),
            "oracles_eligible_for_selection": False,
            "experts": ["persistence_zero_voltage_delta", "frozen_static_dynamic_update"],
            "fit_distribution": "recursive_frozen_static_lookup_exposures",
            "neural_training_performed": False,
            "optimizer_used": False,
            "new_independent_confirmation_claimed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
            "terminal_diagnostic_before_architecture_revision": True,
        }
        atomic._write_json(self.output_dir / "voltage_error_model_contract.json", report)
        return report

    @staticmethod
    def _gate_feature_names(scheme: str) -> Tuple[str, ...]:
        if scheme.endswith("instantaneous"):
            return TemporalVoltageCorrectionState._feature_names(
                "exposure_instantaneous"
            )
        if scheme.endswith("temporal"):
            return TemporalVoltageCorrectionState._feature_names(
                "causal_temporal_combined"
            )
        raise ValueError(scheme)

    def _gate_targets(
        self, observations: Mapping[str, np.ndarray], scheme: str
    ) -> np.ndarray:
        baseline = observations["baseline_delta"]
        required_delta = baseline + observations["target_correction"]
        denominator = baseline * baseline + 1e-8
        optimal_blend = np.clip(required_delta * baseline / denominator, 0.0, 1.0)
        if scheme.startswith("soft_blend"):
            return optimal_blend
        teacher_delta = (
            observations["target_correction"]
            + baseline
            - observations["teacher_current_error"]
        )
        return (
            np.abs(teacher_delta) >= self.config.teacher_quiescent_threshold_mv
        ).astype(np.float64)

    def _fit_gate_model(
        self,
        observations: Mapping[str, np.ndarray],
        scheme: str,
        ridge: float,
    ) -> Dict[str, Any]:
        names = self._gate_feature_names(scheme)
        features = np.column_stack([observations[name] for name in names])
        center = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        normalized = (features - center) / scale
        target = self._gate_targets(observations, scheme)
        region_count = len(self.layout.region_names)
        coefficients = np.zeros((region_count, len(names) + 1), dtype=np.float64)
        counts = np.zeros(region_count, dtype=np.int64)
        for region in range(region_count):
            mask = observations["region"] == region
            counts[region] = int(np.sum(mask))
            if counts[region] == 0:
                continue
            design = np.column_stack((np.ones(counts[region]), normalized[mask]))
            penalty = np.eye(design.shape[1]) * float(ridge)
            penalty[0, 0] = 0.0
            system = design.T @ design + penalty
            right = design.T @ target[mask]
            try:
                coefficients[region] = np.linalg.solve(system, right)
            except np.linalg.LinAlgError:
                coefficients[region] = np.linalg.pinv(system) @ right
        return {
            "scheme": scheme,
            "ridge": float(ridge),
            "feature_names": names,
            "feature_center": center.astype(np.float32),
            "feature_scale": scale.astype(np.float32),
            "coefficients": coefficients.astype(np.float32),
            "region_counts": counts,
            "parameter_count": int(coefficients.size),
        }

    def _predict_gate_numpy(
        self,
        model: Mapping[str, Any],
        observations: Mapping[str, np.ndarray],
        threshold: float | None,
    ) -> np.ndarray:
        features = np.column_stack(
            [observations[name] for name in model["feature_names"]]
        )
        normalized = (features - model["feature_center"]) / model["feature_scale"]
        design = np.column_stack((np.ones(len(features)), normalized))
        score = np.sum(
            design * model["coefficients"][observations["region"]], axis=1
        )
        score = np.clip(score, 0.0, 1.0)
        if threshold is not None:
            score = (score >= float(threshold)).astype(np.float64)
        return score

    def _gate_tensor(
        self,
        model: Mapping[str, Any],
        tensors: Mapping[str, Any],
        threshold: float | None,
    ) -> Any:
        shape = tensors["raw_delta"].shape
        observations = {
            name: tensors[name].detach().cpu().numpy().reshape(-1)
            for name in model["feature_names"]
        }
        observations["region"] = np.tile(
            np.asarray(self.layout.segment_region_ids), shape[0]
        )
        gate = self._predict_gate_numpy(model, observations, threshold).reshape(shape)
        return atomic.torch.as_tensor(
            gate,
            dtype=tensors["raw_delta"].dtype,
            device=tensors["raw_delta"].device,
        )

    def _recursive_gate_evaluation(
        self,
        seed: int,
        scheme: str,
        role: str,
        device: Any,
        ridge: float | None = None,
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        pair = self.source_models[("full_feedback_scalar", seed)]
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        initial_voltage = current_voltage.clone()
        state_center = atomic.torch.as_tensor(self.statistics["state_center"], device=device)
        state_scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=device)
        delta_scale = atomic.torch.as_tensor(self.statistics["delta_scale"], device=device)
        fast_signed = atomic.torch.zeros_like(current_voltage)
        fast_abs = atomic.torch.zeros_like(current_voltage)
        slow_signed = atomic.torch.zeros_like(current_voltage)
        slow_abs = atomic.torch.zeros_like(current_voltage)
        if scheme in CAUSAL_GATE_SCHEMES:
            if ridge is None:
                ridge, threshold = self.selected_gate[(seed, scheme)]
            model = self.gate_models[(seed, scheme, float(ridge))]
        else:
            model = None
        outputs = {}
        with atomic.torch.no_grad():
            for step in range(self.config.objective_unroll_horizon_ms):
                context = atomic.torch.cat((batch["drive"][:, step], batch["held_ions"]), dim=-1)
                normalized = (current_state - state_center) / state_scale
                raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
                baseline_delta = raw * self._static_gain(seed, raw, current_voltage)
                tensors = {
                    "raw_delta": raw,
                    "current_voltage": current_voltage,
                    "baseline_delta": baseline_delta,
                    "ema_fast_signed": fast_signed,
                    "ema_fast_abs": fast_abs,
                    "ema_slow_signed": slow_signed,
                    "ema_slow_abs": slow_abs,
                    "predicted_displacement": current_voltage - initial_voltage,
                }
                if scheme == STATIC_REFERENCE:
                    gate = atomic.torch.ones_like(raw)
                elif scheme in CAUSAL_GATE_SCHEMES:
                    gate = self._gate_tensor(model, tensors, threshold)
                elif scheme == TEACHER_REGIME_ORACLE:
                    teacher_delta = batch["voltage_t1"][:, step] - batch["voltage_t"][:, step]
                    gate = (
                        teacher_delta.abs() >= self.config.teacher_quiescent_threshold_mv
                    ).to(raw.dtype)
                elif scheme == TEACHER_OPTIMAL_BLEND_ORACLE:
                    required = batch["voltage_t1"][:, step] - current_voltage
                    gate = atomic.torch.clamp(
                        required * baseline_delta / (baseline_delta * baseline_delta + 1e-8),
                        0.0,
                        1.0,
                    )
                else:
                    raise ValueError(scheme)
                voltage_delta = baseline_delta * gate
                next_voltage = current_voltage + voltage_delta
                state_delta = self._state_forward(
                    pair[1], normalized, current_voltage, voltage_delta, context
                )
                next_state = current_state + state_delta * delta_scale
                if step + 1 in self.config.rollout_horizons_ms:
                    outputs[f"{step + 1}_ms"] = (next_state, next_voltage, gate)
                fast_signed = self.config.ema_fast_decay * fast_signed + (1 - self.config.ema_fast_decay) * raw
                fast_abs = self.config.ema_fast_decay * fast_abs + (1 - self.config.ema_fast_decay) * raw.abs()
                slow_signed = self.config.ema_slow_decay * slow_signed + (1 - self.config.ema_slow_decay) * raw
                slow_abs = self.config.ema_slow_decay * slow_abs + (1 - self.config.ema_slow_decay) * raw.abs()
                current_state, current_voltage = next_state, next_voltage
        horizons = {}
        for horizon, (state, voltage, gate) in outputs.items():
            step = int(horizon[:-3]) - 1
            horizons[horizon] = self._metric(state, voltage, batch, step)
            horizons[horizon]["gate_mean"] = float(gate.mean().cpu())
            horizons[horizon]["gate_zero_fraction"] = float((gate == 0).float().mean().cpu())
            horizons[horizon]["gate_one_fraction"] = float((gate == 1).float().mean().cpu())
        endpoint = outputs["8_ms"]
        metric_row = {
            "voltage": endpoint[1].cpu().numpy(),
            "target_voltage": batch["voltage_t1"][:, 7].cpu().numpy(),
            "initial_voltage": batch["voltage_t"][:, 0].cpu().numpy(),
        }
        return {
            "ridge": None if model is None else float(model["ridge"]),
            "threshold": threshold,
            "horizons": horizons,
            "activity_at_8ms": self._activity_metrics(metric_row),
            "region_at_8ms": self._region_metrics(metric_row),
        }

    def fit_and_calibrate_gate_models(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        candidate_count = sum(
            len(self.config.gate_ridge_strengths)
            * (
                len(self.config.hurdle_probability_thresholds)
                if scheme.startswith("hard_hurdle")
                else 1
            )
            for scheme in CAUSAL_GATE_SCHEMES
        )
        total = len(self.config.pilot_seeds) * candidate_count
        progress = atomic._CompactProgress(
            "06b-l mixture gate calibration", total, max(1, total // 24)
        )
        completed = 0
        per_seed = {}
        for seed in self.config.pilot_seeds:
            fit = self._recursive_exposure_observations(seed, "fit", device)
            per_seed[str(seed)] = {}
            for scheme in CAUSAL_GATE_SCHEMES:
                candidates = []
                for ridge in self.config.gate_ridge_strengths:
                    model = self._fit_gate_model(fit, scheme, ridge)
                    self.gate_models[(seed, scheme, ridge)] = model
                    thresholds: Sequence[float | None] = (
                        self.config.hurdle_probability_thresholds
                        if scheme.startswith("hard_hurdle")
                        else (None,)
                    )
                    for threshold in thresholds:
                        recursive = self._recursive_gate_evaluation(
                            seed,
                            scheme,
                            "calibration",
                            device,
                            ridge,
                            threshold,
                        )
                        candidates.append(
                            {
                                "ridge": float(ridge),
                                "threshold": threshold,
                                "parameter_count": model["parameter_count"],
                                "recursive_8ms_voltage_rmse_mv": recursive["horizons"]["8_ms"]["voltage_rmse_mv"],
                            }
                        )
                        completed += 1
                        suffix = "soft" if threshold is None else f"threshold={threshold:g}"
                        progress.update(
                            completed,
                            f"seed={seed} scheme={scheme} ridge={ridge:g} {suffix}",
                        )
                selected = min(
                    candidates,
                    key=lambda row: (row["recursive_8ms_voltage_rmse_mv"], -row["ridge"]),
                )
                self.selected_gate[(seed, scheme)] = (
                    selected["ridge"],
                    selected["threshold"],
                )
                model = self.gate_models[(seed, scheme, selected["ridge"])]
                np.savez_compressed(
                    self.output_dir / f"gate_model_{scheme}_seed{seed}.npz",
                    feature_names=np.asarray(model["feature_names"]),
                    feature_center=model["feature_center"],
                    feature_scale=model["feature_scale"],
                    coefficients=model["coefficients"],
                    region_counts=model["region_counts"],
                    ridge=np.asarray([selected["ridge"]], dtype=np.float64),
                    threshold=np.asarray(
                        [np.nan if selected["threshold"] is None else selected["threshold"]],
                        dtype=np.float64,
                    ),
                )
                per_seed[str(seed)][scheme] = {
                    "selected": selected,
                    "candidates": candidates,
                }
        valid = all(
            np.isfinite(candidate["recursive_8ms_voltage_rmse_mv"])
            for rows in per_seed.values()
            for scheme in rows.values()
            for candidate in scheme["candidates"]
        )
        report = {
            "schema_version": "06b-l-gate-calibration-v1",
            "valid": bool(valid),
            "fit_role": "historically_reused_train_fit_recursive_static_exposures",
            "selection_role": "historically_reused_train_calibration",
            "development_accessed": False,
            "oracles_used_during_selection": False,
            "neural_training_performed": False,
            "per_seed": per_seed,
        }
        self.gate_calibration_valid = bool(valid)
        atomic._write_json(self.output_dir / "gate_calibration.json", report)
        return report

    def evaluate_voltage_error_models(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        schemes = (STATIC_REFERENCE, *CAUSAL_GATE_SCHEMES, *ORACLE_SCHEMES)
        progress = atomic._CompactProgress(
            "06b-l mixture development evaluation",
            len(self.config.pilot_seeds) * len(schemes),
            1,
        )
        completed = 0
        per_seed = {}
        for seed in self.config.pilot_seeds:
            per_seed[str(seed)] = {}
            for scheme in schemes:
                per_seed[str(seed)][scheme] = self._recursive_gate_evaluation(
                    seed, scheme, "development", device
                )
                completed += 1
                progress.update(completed, f"seed={seed} scheme={scheme}")
        valid = all(
            metric["nonfinite_voltage_count"] == 0
            and np.isfinite(metric["voltage_rmse_mv"])
            for seed in per_seed.values()
            for scheme in seed.values()
            for metric in scheme["horizons"].values()
        )
        report = {
            "schema_version": "06b-l-development-v1",
            "valid": bool(valid),
            "role": "historically_reused_train_development",
            "new_independent_confirmation_claimed": False,
            "oracles_eligible_for_selection": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "voltage_error_model_development.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _gate_summary(
        self, scheme: str, evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rows = [seed[scheme] for seed in evaluation["per_seed"].values()]
        static_rows = [seed[STATIC_REFERENCE] for seed in evaluation["per_seed"].values()]
        endpoint = [row["horizons"]["8_ms"] for row in rows]
        static_endpoint = [row["horizons"]["8_ms"] for row in static_rows]
        over_static = [
            1 - row["voltage_rmse_mv"] / max(reference["voltage_rmse_mv"], 1e-12)
            for row, reference in zip(endpoint, static_endpoint)
        ]
        voltage_gains = [row["voltage_improvement_vs_persistence_fraction"] for row in endpoint]
        state_gains = [row["state_improvement_vs_persistence_fraction"] for row in endpoint]
        activity = {
            name: self._median(
                [row["activity_at_8ms"][name]["voltage_gain_vs_persistence_fraction"] for row in rows]
            )
            for name in rows[0]["activity_at_8ms"]
        }
        activity_rmse = {
            name: self._median(
                [row["activity_at_8ms"][name]["voltage_rmse_mv"] for row in rows]
            )
            for name in rows[0]["activity_at_8ms"]
        }
        region = {
            name: self._median(
                [row["region_at_8ms"][name]["voltage_gain_vs_persistence_fraction"] for row in rows]
            )
            for name in rows[0]["region_at_8ms"]
        }
        return {
            "median_recursive_gain_over_static_fraction": self._median(over_static),
            "median_voltage_gain_vs_persistence_fraction": self._median(voltage_gains),
            "median_STATE_gain_vs_persistence_fraction": self._median(state_gains),
            "minimum_seed_STATE_gain_vs_persistence_fraction": float(min(state_gains)),
            "activity_gain_vs_persistence": activity,
            "activity_rmse_mv": activity_rmse,
            "region_gain_vs_persistence": region,
            "all_seed_voltage_gain_positive": all(value > 0 for value in voltage_gains),
            "physical_voltage_violation_count": int(
                sum(row["physical_voltage_violation_count"] for row in endpoint)
            ),
        }

    def finalize_voltage_error_model_revision(
        self, evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        schemes = (STATIC_REFERENCE, *CAUSAL_GATE_SCHEMES, *ORACLE_SCHEMES)
        summaries = {scheme: self._gate_summary(scheme, evaluation) for scheme in schemes}

        def passes(scheme: str) -> bool:
            row = summaries[scheme]
            return bool(
                row["median_recursive_gain_over_static_fraction"]
                >= self.config.minimum_recursive_gain_over_static_fraction
                and row["all_seed_voltage_gain_positive"]
                and row["minimum_seed_STATE_gain_vs_persistence_fraction"]
                > self.config.minimum_STATE_gain_fraction
                and row["activity_gain_vs_persistence"]["active_ge_5mV"]
                >= self.config.minimum_active_gain_fraction
                and row["activity_gain_vs_persistence"]["moderate_1_to_5mV"]
                >= self.config.minimum_moderate_gain_fraction
                and row["activity_gain_vs_persistence"]["quiescent_lt_1mV"]
                >= self.config.minimum_quiescent_gain_fraction
                and row["region_gain_vs_persistence"]["soma"]
                >= self.config.minimum_soma_gain_fraction
                and row["physical_voltage_violation_count"] == 0
            )

        gate_results = {scheme: passes(scheme) for scheme in (*CAUSAL_GATE_SCHEMES, *ORACLE_SCHEMES)}
        gate_results[STATIC_REFERENCE] = False
        for scheme, passed in gate_results.items():
            summaries[scheme]["registered_gate_passed"] = passed
        primary_pass = gate_results[PRIMARY_SCHEME]
        fallback_pass = gate_results[FALLBACK_SCHEME]
        other_causal = [
            scheme
            for scheme in CAUSAL_GATE_SCHEMES
            if scheme not in (PRIMARY_SCHEME, FALLBACK_SCHEME) and gate_results[scheme]
        ]
        regime_oracle_pass = gate_results[TEACHER_REGIME_ORACLE]
        optimal_oracle_pass = gate_results[TEACHER_OPTIMAL_BLEND_ORACLE]
        if primary_pass or fallback_pass or other_causal:
            diagnosis = "CAUSAL_PERSISTENCE_DYNAMIC_GATE_IDENTIFIED"
            selected = (
                PRIMARY_SCHEME
                if primary_pass
                else FALLBACK_SCHEME
                if fallback_pass
                else other_causal[0]
            )
            next_step = "architecture_revision_persistence_dynamic_gate"
        elif regime_oracle_pass:
            diagnosis = "TEACHER_REGIME_GATE_WORKS_BUT_CAUSAL_GATE_FAILS"
            selected = None
            next_step = "architecture_revision_regime_state_encoder"
        elif optimal_oracle_pass:
            diagnosis = "OPTIMAL_BLEND_ORACLE_WORKS_BUT_REGIME_GATE_FAILS"
            selected = None
            next_step = "architecture_revision_continuous_mixture_state"
        else:
            diagnosis = "PERSISTENCE_DYNAMIC_EXPERT_FAMILY_INSUFFICIENT"
            selected = None
            next_step = "architecture_revision_voltage_expert_family"
        report = {
            "schema_version": "06b-l-final-report-v1",
            "valid": bool(self.gate_calibration_valid and evaluation.get("valid")),
            "component_playground_grade": True,
            "terminal_diagnostic_before_architecture_revision": True,
            "new_independent_confirmation_claimed": False,
            "diagnosis": diagnosis,
            "primary_scheme": PRIMARY_SCHEME,
            "fallback_scheme": FALLBACK_SCHEME,
            "primary_passed": primary_pass,
            "fallback_passed": fallback_pass,
            "other_causal_passing_schemes": other_causal,
            "teacher_regime_oracle_passed": regime_oracle_pass,
            "teacher_optimal_blend_oracle_passed": optimal_oracle_pass,
            "oracles_eligible_for_selection": False,
            "selected_causal_scheme": selected,
            "summaries": summaries,
            "neural_training_performed": False,
            "optimizer_used": False,
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
    "CAUSAL_GATE_SCHEMES",
    "EXPECTED_06BK_ARCHIVE_SHA256",
    "EXPECTED_06BK_FINAL_SHA256",
    "EXPECTED_06BK_INDEX_SHA256",
    "FALLBACK_SCHEME",
    "ORACLE_SCHEMES",
    "PRIMARY_SCHEME",
    "STATIC_REFERENCE",
    "TEACHER_OPTIMAL_BLEND_ORACLE",
    "TEACHER_REGIME_ORACLE",
    "VoltageErrorModelRevision",
    "VoltageErrorModelRevisionConfig",
    "verified_06bk_artifact_root",
]
