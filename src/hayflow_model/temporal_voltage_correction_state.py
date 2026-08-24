"""06b-k: analytic causal temporal correction state on recursive exposures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .analytic_causal_gain_identifiability import (
    PRIMARY_SCHEME as STATIC_LOOKUP_SCHEME,
    AnalyticCausalGainConfig,
    AnalyticCausalGainIdentifiability,
)


EXPECTED_06BJ_ARCHIVE_SHA256 = (
    "44b5d5653c7e99efc78315801f28758a5c71f06d6f492aa7c2cd34b57ba58517"
)
EXPECTED_06BJ_INDEX_SHA256 = (
    "ebf1b4e6b89020851b5baf2eca6a8ebfb09b056ca5506b77061231424fbf822c"
)
EXPECTED_06BJ_FINAL_SHA256 = (
    "68fd1d9951f56e62c9dc0504823bd4087c95e51a4e895b4ea0e5a617d0050521"
)

STATIC_REFERENCE = "static_lookup_reference"
TEMPORAL_SCHEMES = (
    "exposure_instantaneous",
    "ema_fast",
    "ema_slow",
    "predicted_displacement",
    "causal_temporal_combined",
)
ORACLE_SCHEME = "teacher_error_oracle"
PRIMARY_SCHEME = "causal_temporal_combined"
FALLBACK_SCHEME = "ema_slow"
EXPOSURE_CONTROL_SCHEME = "exposure_instantaneous"


def verified_06bj_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-j source must be a ZIP or extracted directory")
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
        if atomic._sha256_file(path) == EXPECTED_06BJ_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-j artifact; found {len(matches)}")
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
        raise RuntimeError(f"06b-j indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BJ_FINAL_SHA256:
        raise RuntimeError("06b-j final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "STATIC_GAIN_IDENTIFIED_BUT_TEMPORAL_COMPOSITION_FAILS"
        or final.get("next_step") != "atomic_temporal_voltage_correction_state"
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-j result does not authorize 06b-k")
    if source.is_file() and archive_hash != EXPECTED_06BJ_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BJ_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BJ_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class TemporalVoltageCorrectionConfig(AnalyticCausalGainConfig):
    temporal_ridge_strengths: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    temporal_correction_clip_mv: float = 5.0
    ema_fast_decay: float = 0.5
    ema_slow_decay: float = 0.8
    minimum_recursive_gain_over_static_fraction: float = 0.02
    minimum_temporal_specificity_over_exposure_control_fraction: float = 0.01

    def validate(self) -> None:
        super().validate()
        if tuple(sorted(set(self.temporal_ridge_strengths))) != self.temporal_ridge_strengths:
            raise ValueError("06b-k ridge strengths must be sorted and unique")
        if self.temporal_correction_clip_mv <= 0:
            raise ValueError("06b-k correction clip must be positive")
        if not 0 < self.ema_fast_decay < self.ema_slow_decay < 1:
            raise ValueError("06b-k EMA decays must satisfy 0 < fast < slow < 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TemporalVoltageCorrectionConfig":
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
        ):
            if name in payload:
                payload[name] = tuple(map(float, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class TemporalVoltageCorrectionState(AnalyticCausalGainIdentifiability):
    config: TemporalVoltageCorrectionConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: TemporalVoltageCorrectionConfig,
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
            code_revision=code_revision,
        )
        self.artifact_06bj_source = Path(artifact_06bj_source)
        self.static_lookups: Dict[int, Dict[str, Any]] = {}
        self.temporal_models: Dict[Tuple[int, str, float], Dict[str, Any]] = {}
        self.selected_ridge: Dict[Tuple[int, str], float] = {}
        self.temporal_calibration_valid = False

    def prepare_temporal_correction_state(self) -> Dict[str, Any]:
        base = self.prepare_analytic_identifiability()
        root, source = verified_06bj_artifact_root(
            self.artifact_06bj_source,
            self.output_dir.parent / ".06bk_artifact_cache" / "06bj",
        )
        for seed in self.config.pilot_seeds:
            values = np.load(root / f"analytic_lookup_{STATIC_LOOKUP_SCHEME}_seed{seed}.npz")
            self.static_lookups[int(seed)] = {
                "scheme": STATIC_LOOKUP_SCHEME,
                "gains": np.asarray(values["gains"], dtype=np.float32),
                "counts": np.asarray(values["counts"], dtype=np.int64),
                "shrinkage": float(values["shrinkage"][0]),
                "global_gain": float(values["global_gain"][0]),
            }
        report = {
            **base,
            "schema_version": "06b-k-temporal-contract-v1",
            "experiment": "temporal_voltage_correction_state",
            "source_06bj": source,
            "static_reference": STATIC_REFERENCE,
            "static_lookup_scheme": STATIC_LOOKUP_SCHEME,
            "temporal_schemes": list(TEMPORAL_SCHEMES),
            "primary_scheme": PRIMARY_SCHEME,
            "fallback_scheme": FALLBACK_SCHEME,
            "exposure_control_scheme": EXPOSURE_CONTROL_SCHEME,
            "oracle_scheme": ORACLE_SCHEME,
            "oracle_eligible_for_selection": False,
            "fit_distribution": "recursive_static_lookup_exposures",
            "fit_method": "closed_form_region_specific_ridge",
            "neural_training_performed": False,
            "optimizer_used": False,
            "teacher_error_used_by_causal_schemes": False,
            "teacher_error_used_by_oracle_only": True,
            "new_independent_confirmation_claimed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
        }
        atomic._write_json(self.output_dir / "temporal_correction_contract.json", report)
        return report

    @staticmethod
    def _feature_names(scheme: str) -> Tuple[str, ...]:
        instantaneous = ("raw_delta", "current_voltage", "baseline_delta")
        if scheme == "exposure_instantaneous":
            return instantaneous
        if scheme == "ema_fast":
            return instantaneous + ("ema_fast_signed", "ema_fast_abs")
        if scheme == "ema_slow":
            return instantaneous + ("ema_slow_signed", "ema_slow_abs")
        if scheme == "predicted_displacement":
            return instantaneous + ("predicted_displacement",)
        if scheme == "causal_temporal_combined":
            return instantaneous + (
                "ema_fast_signed",
                "ema_fast_abs",
                "ema_slow_signed",
                "ema_slow_abs",
                "predicted_displacement",
            )
        if scheme == ORACLE_SCHEME:
            return TemporalVoltageCorrectionState._feature_names(PRIMARY_SCHEME) + (
                "teacher_current_error",
            )
        raise ValueError(scheme)

    def _static_gain(self, seed: int, raw_delta: Any, voltage: Any) -> Any:
        lookup = self.static_lookups[int(seed)]
        zeros = atomic.torch.zeros_like(raw_delta)
        return self._lookup_gain(lookup, raw_delta, voltage, zeros)

    def _recursive_exposure_observations(
        self, seed: int, role: str, device: Any
    ) -> Dict[str, np.ndarray]:
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
        records: Dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "raw_delta",
                "current_voltage",
                "baseline_delta",
                "ema_fast_signed",
                "ema_fast_abs",
                "ema_slow_signed",
                "ema_slow_abs",
                "predicted_displacement",
                "teacher_current_error",
                "target_correction",
            )
        }
        with atomic.torch.no_grad():
            for step in range(self.config.objective_unroll_horizon_ms):
                context = atomic.torch.cat((batch["drive"][:, step], batch["held_ions"]), dim=-1)
                normalized = (current_state - state_center) / state_scale
                raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
                baseline_delta = raw * self._static_gain(seed, raw, current_voltage)
                teacher_current = batch["voltage_t"][:, step]
                target_next = batch["voltage_t1"][:, step]
                values = {
                    "raw_delta": raw,
                    "current_voltage": current_voltage,
                    "baseline_delta": baseline_delta,
                    "ema_fast_signed": fast_signed,
                    "ema_fast_abs": fast_abs,
                    "ema_slow_signed": slow_signed,
                    "ema_slow_abs": slow_abs,
                    "predicted_displacement": current_voltage - initial_voltage,
                    "teacher_current_error": teacher_current - current_voltage,
                    "target_correction": target_next - (current_voltage + baseline_delta),
                }
                for name, value in values.items():
                    records[name].append(value.cpu().numpy().reshape(-1))
                next_voltage = current_voltage + baseline_delta
                state_delta = self._state_forward(
                    pair[1], normalized, current_voltage, baseline_delta, context
                )
                next_state = current_state + state_delta * delta_scale
                fast_signed = self.config.ema_fast_decay * fast_signed + (1 - self.config.ema_fast_decay) * raw
                fast_abs = self.config.ema_fast_decay * fast_abs + (1 - self.config.ema_fast_decay) * raw.abs()
                slow_signed = self.config.ema_slow_decay * slow_signed + (1 - self.config.ema_slow_decay) * raw
                slow_abs = self.config.ema_slow_decay * slow_abs + (1 - self.config.ema_slow_decay) * raw.abs()
                current_state, current_voltage = next_state, next_voltage
        repeat = len(rows) * self.config.objective_unroll_horizon_ms
        result = {name: np.concatenate(values).astype(np.float64) for name, values in records.items()}
        result["region"] = np.tile(np.asarray(self.layout.segment_region_ids), repeat).astype(np.int64)
        return result

    def _fit_ridge_model(
        self,
        observations: Mapping[str, np.ndarray],
        scheme: str,
        ridge: float,
    ) -> Dict[str, Any]:
        names = self._feature_names(scheme)
        features = np.column_stack([observations[name] for name in names])
        center = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        normalized = (features - center) / scale
        target = np.clip(
            observations["target_correction"],
            -self.config.temporal_correction_clip_mv,
            self.config.temporal_correction_clip_mv,
        )
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

    def _predict_correction_numpy(
        self, model: Mapping[str, Any], observations: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        features = np.column_stack(
            [observations[name] for name in model["feature_names"]]
        )
        normalized = (features - model["feature_center"]) / model["feature_scale"]
        design = np.column_stack((np.ones(len(features)), normalized))
        correction = np.sum(
            design * model["coefficients"][observations["region"]], axis=1
        )
        return np.clip(
            correction,
            -self.config.temporal_correction_clip_mv,
            self.config.temporal_correction_clip_mv,
        )

    def _direct_metrics(
        self, observations: Mapping[str, np.ndarray], model: Mapping[str, Any]
    ) -> Dict[str, float]:
        correction = self._predict_correction_numpy(model, observations)
        target = observations["target_correction"]
        corrected_rmse = float(np.sqrt(np.mean((correction - target) ** 2)))
        static_rmse = float(np.sqrt(np.mean(target**2)))
        return {
            "correction_rmse_mv": corrected_rmse,
            "static_residual_rmse_mv": static_rmse,
            "improvement_over_static_fraction": 1.0
            - corrected_rmse / max(static_rmse, 1e-12),
        }

    def _correction_tensor(
        self,
        model: Mapping[str, Any],
        tensors: Mapping[str, Any],
    ) -> Any:
        shape = tensors["raw_delta"].shape
        observations = {
            name: tensors[name].detach().cpu().numpy().reshape(-1)
            for name in model["feature_names"]
        }
        observations["region"] = np.tile(
            np.asarray(self.layout.segment_region_ids), shape[0]
        )
        correction = self._predict_correction_numpy(model, observations).reshape(shape)
        return atomic.torch.as_tensor(
            correction,
            dtype=tensors["raw_delta"].dtype,
            device=tensors["raw_delta"].device,
        )

    def _recursive_evaluation(
        self,
        seed: int,
        scheme: str,
        role: str,
        device: Any,
        ridge: float | None = None,
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
        outputs = {}
        if scheme != STATIC_REFERENCE:
            chosen = self.selected_ridge[(seed, scheme)] if ridge is None else float(ridge)
            model = self.temporal_models[(seed, scheme, chosen)]
        else:
            model = None
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
                    "teacher_current_error": batch["voltage_t"][:, step] - current_voltage,
                }
                correction = (
                    atomic.torch.zeros_like(raw)
                    if model is None
                    else self._correction_tensor(model, tensors)
                )
                voltage_delta = baseline_delta + correction
                next_voltage = current_voltage + voltage_delta
                state_delta = self._state_forward(
                    pair[1], normalized, current_voltage, voltage_delta, context
                )
                next_state = current_state + state_delta * delta_scale
                if step + 1 in self.config.rollout_horizons_ms:
                    outputs[f"{step + 1}_ms"] = (next_state, next_voltage, correction)
                fast_signed = self.config.ema_fast_decay * fast_signed + (1 - self.config.ema_fast_decay) * raw
                fast_abs = self.config.ema_fast_decay * fast_abs + (1 - self.config.ema_fast_decay) * raw.abs()
                slow_signed = self.config.ema_slow_decay * slow_signed + (1 - self.config.ema_slow_decay) * raw
                slow_abs = self.config.ema_slow_decay * slow_abs + (1 - self.config.ema_slow_decay) * raw.abs()
                current_state, current_voltage = next_state, next_voltage
        horizons = {}
        for horizon, (state, voltage, correction) in outputs.items():
            step = int(horizon[:-3]) - 1
            horizons[horizon] = self._metric(state, voltage, batch, step)
            horizons[horizon]["correction_minimum_mv"] = float(correction.min().cpu())
            horizons[horizon]["correction_median_mv"] = float(correction.median().cpu())
            horizons[horizon]["correction_maximum_mv"] = float(correction.max().cpu())
        endpoint = outputs["8_ms"]
        metric_row = {
            "voltage": endpoint[1].cpu().numpy(),
            "target_voltage": batch["voltage_t1"][:, 7].cpu().numpy(),
            "initial_voltage": batch["voltage_t"][:, 0].cpu().numpy(),
        }
        return {
            "ridge": None if model is None else float(model["ridge"]),
            "horizons": horizons,
            "activity_at_8ms": self._activity_metrics(metric_row),
            "region_at_8ms": self._region_metrics(metric_row),
        }

    def fit_and_calibrate_temporal_models(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        schemes = (*TEMPORAL_SCHEMES, ORACLE_SCHEME)
        total = len(self.config.pilot_seeds) * len(schemes) * len(self.config.temporal_ridge_strengths)
        progress = atomic._CompactProgress(
            "06b-k temporal ridge calibration", total, max(1, total // 24)
        )
        completed = 0
        per_seed = {}
        for seed in self.config.pilot_seeds:
            fit = self._recursive_exposure_observations(seed, "fit", device)
            calibration = self._recursive_exposure_observations(seed, "calibration", device)
            per_seed[str(seed)] = {}
            for scheme in schemes:
                candidates = []
                for ridge in self.config.temporal_ridge_strengths:
                    model = self._fit_ridge_model(fit, scheme, ridge)
                    self.temporal_models[(seed, scheme, ridge)] = model
                    direct = self._direct_metrics(calibration, model)
                    recursive = self._recursive_evaluation(
                        seed, scheme, "calibration", device, ridge
                    )
                    candidates.append(
                        {
                            "ridge": float(ridge),
                            "parameter_count": model["parameter_count"],
                            "direct_calibration": direct,
                            "recursive_8ms_voltage_rmse_mv": recursive["horizons"]["8_ms"]["voltage_rmse_mv"],
                        }
                    )
                    completed += 1
                    progress.update(completed, f"seed={seed} scheme={scheme} ridge={ridge:g}")
                selected = min(
                    candidates,
                    key=lambda row: (row["recursive_8ms_voltage_rmse_mv"], -row["ridge"]),
                )
                self.selected_ridge[(seed, scheme)] = selected["ridge"]
                selected_model = self.temporal_models[(seed, scheme, selected["ridge"])]
                np.savez_compressed(
                    self.output_dir / f"temporal_model_{scheme}_seed{seed}.npz",
                    feature_names=np.asarray(selected_model["feature_names"]),
                    feature_center=selected_model["feature_center"],
                    feature_scale=selected_model["feature_scale"],
                    coefficients=selected_model["coefficients"],
                    region_counts=selected_model["region_counts"],
                    ridge=np.asarray([selected["ridge"]], dtype=np.float64),
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
            "schema_version": "06b-k-temporal-calibration-v1",
            "valid": bool(valid),
            "fit_role": "historically_reused_train_fit_recursive_static_exposures",
            "selection_role": "historically_reused_train_calibration",
            "development_accessed": False,
            "teacher_error_oracle_eligible_for_selection": False,
            "neural_training_performed": False,
            "per_seed": per_seed,
        }
        self.temporal_calibration_valid = bool(valid)
        atomic._write_json(self.output_dir / "temporal_calibration.json", report)
        return report

    def evaluate_temporal_models(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        schemes = (STATIC_REFERENCE, *TEMPORAL_SCHEMES, ORACLE_SCHEME)
        progress = atomic._CompactProgress(
            "06b-k temporal development evaluation",
            len(self.config.pilot_seeds) * len(schemes),
            1,
        )
        completed = 0
        per_seed = {}
        for seed in self.config.pilot_seeds:
            per_seed[str(seed)] = {}
            for scheme in schemes:
                per_seed[str(seed)][scheme] = self._recursive_evaluation(
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
            "schema_version": "06b-k-temporal-development-v1",
            "valid": bool(valid),
            "role": "historically_reused_train_development",
            "new_independent_confirmation_claimed": False,
            "oracle_eligible_for_selection": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "temporal_development.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _scheme_summary(
        self, scheme: str, evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rows = [seed[scheme] for seed in evaluation["per_seed"].values()]
        static_rows = [seed[STATIC_REFERENCE] for seed in evaluation["per_seed"].values()]
        exposure_rows = [seed[EXPOSURE_CONTROL_SCHEME] for seed in evaluation["per_seed"].values()]
        endpoint = [row["horizons"]["8_ms"] for row in rows]
        static_endpoint = [row["horizons"]["8_ms"] for row in static_rows]
        exposure_endpoint = [row["horizons"]["8_ms"] for row in exposure_rows]
        over_static = [
            1 - row["voltage_rmse_mv"] / max(reference["voltage_rmse_mv"], 1e-12)
            for row, reference in zip(endpoint, static_endpoint)
        ]
        over_exposure = [
            1 - row["voltage_rmse_mv"] / max(reference["voltage_rmse_mv"], 1e-12)
            for row, reference in zip(endpoint, exposure_endpoint)
        ]
        voltage_gains = [row["voltage_improvement_vs_persistence_fraction"] for row in endpoint]
        state_gains = [row["state_improvement_vs_persistence_fraction"] for row in endpoint]
        activity = {
            name: self._median(
                [row["activity_at_8ms"][name]["voltage_gain_vs_persistence_fraction"] for row in rows]
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
            "median_recursive_gain_over_exposure_control_fraction": self._median(over_exposure),
            "median_voltage_gain_vs_persistence_fraction": self._median(voltage_gains),
            "median_STATE_gain_vs_persistence_fraction": self._median(state_gains),
            "minimum_seed_STATE_gain_vs_persistence_fraction": float(min(state_gains)),
            "activity_gain_vs_persistence": activity,
            "region_gain_vs_persistence": region,
            "all_seed_voltage_gain_positive": all(value > 0 for value in voltage_gains),
            "physical_voltage_violation_count": int(
                sum(row["physical_voltage_violation_count"] for row in endpoint)
            ),
        }

    def finalize_temporal_correction_state(
        self, evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        schemes = (STATIC_REFERENCE, *TEMPORAL_SCHEMES, ORACLE_SCHEME)
        summaries = {scheme: self._scheme_summary(scheme, evaluation) for scheme in schemes}

        def base_gate(scheme: str) -> bool:
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

        gate_results = {scheme: base_gate(scheme) for scheme in TEMPORAL_SCHEMES}
        gate_results[ORACLE_SCHEME] = base_gate(ORACLE_SCHEME)
        gate_results[STATIC_REFERENCE] = False
        for scheme in (PRIMARY_SCHEME, FALLBACK_SCHEME):
            gate_results[scheme] = bool(
                gate_results[scheme]
                and summaries[scheme]["median_recursive_gain_over_exposure_control_fraction"]
                >= self.config.minimum_temporal_specificity_over_exposure_control_fraction
            )
        for scheme, passed in gate_results.items():
            summaries[scheme]["registered_gate_passed"] = passed

        primary_pass = gate_results[PRIMARY_SCHEME]
        fallback_pass = gate_results[FALLBACK_SCHEME]
        exposure_pass = gate_results[EXPOSURE_CONTROL_SCHEME]
        oracle_pass = gate_results[ORACLE_SCHEME]
        temporal_signal = (
            summaries[PRIMARY_SCHEME]["median_recursive_gain_over_exposure_control_fraction"]
            >= self.config.minimum_temporal_specificity_over_exposure_control_fraction
        )
        if primary_pass:
            diagnosis = "CAUSAL_TEMPORAL_CORRECTION_STATE_IDENTIFIED"
            next_step = "fresh_train_support_temporal_state_confirmation"
        elif fallback_pass:
            diagnosis = "CAUSAL_EMA_CORRECTION_STATE_IDENTIFIED"
            next_step = "fresh_train_support_ema_state_confirmation"
        elif exposure_pass:
            diagnosis = "RECURSIVE_EXPOSURE_REFIT_SUFFICIENT_WITHOUT_TEMPORAL_STATE"
            next_step = "fresh_train_support_exposure_matched_confirmation"
        elif oracle_pass:
            diagnosis = "TEACHER_ERROR_ORACLE_WORKS_BUT_CAUSAL_HISTORY_FAILS"
            next_step = "latent_error_state_observability_playground"
        elif temporal_signal:
            diagnosis = "TEMPORAL_SIGNAL_PRESENT_BUT_LOW_ACTIVITY_GATES_FAIL"
            next_step = "low_activity_temporal_residual_forensic"
        else:
            diagnosis = "TEMPORAL_CORRECTION_STATE_NOT_IDENTIFIED"
            next_step = "atomic_voltage_error_model_revision"
        report = {
            "schema_version": "06b-k-final-report-v1",
            "valid": bool(self.temporal_calibration_valid and evaluation.get("valid")),
            "component_playground_grade": True,
            "new_independent_confirmation_claimed": False,
            "diagnosis": diagnosis,
            "primary_scheme": PRIMARY_SCHEME,
            "fallback_scheme": FALLBACK_SCHEME,
            "exposure_control_scheme": EXPOSURE_CONTROL_SCHEME,
            "oracle_scheme": ORACLE_SCHEME,
            "primary_passed": primary_pass,
            "fallback_passed": fallback_pass,
            "exposure_control_passed": exposure_pass,
            "oracle_passed": oracle_pass,
            "oracle_eligible_for_selection": False,
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
    "EXPECTED_06BJ_ARCHIVE_SHA256",
    "EXPECTED_06BJ_FINAL_SHA256",
    "EXPECTED_06BJ_INDEX_SHA256",
    "EXPOSURE_CONTROL_SCHEME",
    "FALLBACK_SCHEME",
    "ORACLE_SCHEME",
    "PRIMARY_SCHEME",
    "STATIC_REFERENCE",
    "TEMPORAL_SCHEMES",
    "TemporalVoltageCorrectionConfig",
    "TemporalVoltageCorrectionState",
    "verified_06bj_artifact_root",
]
