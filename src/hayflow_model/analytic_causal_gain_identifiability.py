"""06b-j: frozen analytic identifiability of causal voltage-gain features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .voltage_objective_recalibration_playground import (
    VoltageObjectiveRecalibrationConfig,
    VoltageObjectiveRecalibrationPlayground,
)


EXPECTED_06BI_ARCHIVE_SHA256 = (
    "539005ba612468a83ea8265b7b95dcb27eda3f85df5d81c2c0ad59badebed629"
)
EXPECTED_06BI_INDEX_SHA256 = (
    "bcd49b6d08ab64baca45040731aa698e05164e6c4ac8241d1ed62ab5a05a9f63"
)
EXPECTED_06BI_FINAL_SHA256 = (
    "336eddbcd6c8a8436cf65983582fb09544b512a5af9b1ecdea9d312808be4343"
)

CAUSAL_GAIN_SCHEMES = (
    "global",
    "region",
    "raw_activity",
    "voltage_band",
    "region_raw_activity",
    "region_raw_voltage",
)
ORACLE_SCHEME = "teacher_activity_oracle"
PRIMARY_SCHEME = "region_raw_voltage"
FALLBACK_SCHEME = "region_raw_activity"


def verified_06bi_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-i source must be a ZIP or extracted directory")
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
        if atomic._sha256_file(path) == EXPECTED_06BI_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-i artifact; found {len(matches)}")
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
        raise RuntimeError(f"06b-i indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BI_FINAL_SHA256:
        raise RuntimeError("06b-i final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "VOLTAGE_OBJECTIVE_RECALIBRATION_NOT_LEARNABLE"
        or final.get("next_step") != "atomic_voltage_bridge_representation_revision"
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-i result does not authorize 06b-j")
    if source.is_file() and archive_hash != EXPECTED_06BI_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BI_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BI_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class AnalyticCausalGainConfig(VoltageObjectiveRecalibrationConfig):
    analytic_shrinkage_strengths: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    analytic_gain_minimum: float = 0.0
    analytic_gain_maximum: float = 1.5
    analytic_reference_gain: float = 0.75
    analytic_voltage_edges_mv: Tuple[float, ...] = (-70.0, -50.0, -20.0)
    minimum_direct_improvement_fraction: float = 0.05
    minimum_recursive_gain_over_global_fraction: float = 0.02
    minimum_active_gain_fraction: float = 0.10
    minimum_STATE_gain_fraction: float = 0.0

    def validate(self) -> None:
        super().validate()
        if tuple(sorted(set(self.analytic_shrinkage_strengths))) != self.analytic_shrinkage_strengths:
            raise ValueError("06b-j shrinkage strengths must be sorted and unique")
        if tuple(sorted(self.analytic_voltage_edges_mv)) != self.analytic_voltage_edges_mv:
            raise ValueError("06b-j voltage edges must be sorted")
        if not self.analytic_gain_minimum < self.analytic_reference_gain < self.analytic_gain_maximum:
            raise ValueError("06b-j reference gain is outside analytic bounds")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AnalyticCausalGainConfig":
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
        ):
            if name in payload:
                payload[name] = tuple(map(float, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class AnalyticCausalGainIdentifiability(VoltageObjectiveRecalibrationPlayground):
    config: AnalyticCausalGainConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: AnalyticCausalGainConfig,
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
            code_revision=code_revision,
        )
        self.artifact_06bi_source = Path(artifact_06bi_source)
        self.lookup_tables: Dict[Tuple[int, str, float], Dict[str, Any]] = {}
        self.selected_shrinkage: Dict[Tuple[int, str], float] = {}
        self.calibration_valid = False

    def prepare_analytic_identifiability(self) -> Dict[str, Any]:
        base = self.prepare_voltage_objective_recalibration()
        _, source = verified_06bi_artifact_root(
            self.artifact_06bi_source,
            self.output_dir.parent / ".06bj_artifact_cache" / "06bi",
        )
        report = {
            **base,
            "schema_version": "06b-j-analytic-contract-v1",
            "experiment": "analytic_causal_gain_identifiability",
            "source_06bi": source,
            "causal_gain_schemes": list(CAUSAL_GAIN_SCHEMES),
            "oracle_scheme": ORACLE_SCHEME,
            "oracle_eligible_for_selection": False,
            "primary_scheme": PRIMARY_SCHEME,
            "fallback_scheme": FALLBACK_SCHEME,
            "shrinkage_strengths": list(self.config.analytic_shrinkage_strengths),
            "fit_method": "closed_form_shrunk_least_squares",
            "neural_training_performed": False,
            "optimizer_used": False,
            "teacher_activity_used_by_causal_schemes": False,
            "teacher_activity_used_by_oracle_only": True,
            "new_independent_confirmation_claimed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
        }
        for stale in ("neural_training_planned", "trainable_objective_arms"):
            report.pop(stale, None)
        atomic._write_json(self.output_dir / "analytic_gain_contract.json", report)
        return report

    def _teacher_boundary_observations(
        self, seed: int, role: str, device: Any
    ) -> Dict[str, np.ndarray]:
        pair = self.source_models[("full_feedback_scalar", seed)]
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        state_center = atomic.torch.as_tensor(
            self.statistics["state_center"], device=device
        )
        raw_rows = []
        voltage_rows = []
        target_rows = []
        with atomic.torch.no_grad():
            for step in range(self.config.objective_unroll_horizon_ms):
                state = batch["state_t"][:, step]
                voltage = batch["voltage_t"][:, step]
                context = atomic.torch.cat(
                    (batch["drive"][:, step], batch["held_ions"]), dim=-1
                )
                normalized = (state - state_center) / atomic.torch.as_tensor(
                    self.statistics["state_scale"], device=device
                )
                raw = self._bridge_forward(pair[0], normalized, voltage, context)
                target = batch["voltage_t1"][:, step] - voltage
                raw_rows.append(raw.cpu().numpy().reshape(-1))
                voltage_rows.append(voltage.cpu().numpy().reshape(-1))
                target_rows.append(target.cpu().numpy().reshape(-1))
        repeat = len(rows) * self.config.objective_unroll_horizon_ms
        region = np.tile(np.asarray(self.layout.segment_region_ids), repeat)
        return {
            "raw_delta": np.concatenate(raw_rows).astype(np.float64),
            "voltage": np.concatenate(voltage_rows).astype(np.float64),
            "target_delta": np.concatenate(target_rows).astype(np.float64),
            "region": region.astype(np.int64),
        }

    def _cell_ids(
        self, observations: Mapping[str, np.ndarray], scheme: str
    ) -> Tuple[np.ndarray, int]:
        region = observations["region"]
        region_count = len(self.layout.region_names)
        activity_count = len(self.config.activity_edges_mv) + 1
        voltage_count = len(self.config.analytic_voltage_edges_mv) + 1
        raw_activity = np.digitize(
            np.abs(observations["raw_delta"]), self.config.activity_edges_mv
        )
        voltage_band = np.digitize(
            observations["voltage"], self.config.analytic_voltage_edges_mv
        )
        if scheme == "global":
            return np.zeros_like(region), 1
        if scheme == "region":
            return region, region_count
        if scheme == "raw_activity":
            return raw_activity, activity_count
        if scheme == "voltage_band":
            return voltage_band, voltage_count
        if scheme == "region_raw_activity":
            return region * activity_count + raw_activity, region_count * activity_count
        if scheme == "region_raw_voltage":
            cells = (region * activity_count + raw_activity) * voltage_count + voltage_band
            return cells, region_count * activity_count * voltage_count
        if scheme == ORACLE_SCHEME:
            teacher_activity = np.digitize(
                np.abs(observations["target_delta"]), self.config.activity_edges_mv
            )
            return region * activity_count + teacher_activity, region_count * activity_count
        raise ValueError(scheme)

    def _fit_lookup(
        self,
        observations: Mapping[str, np.ndarray],
        scheme: str,
        shrinkage: float,
    ) -> Dict[str, Any]:
        cells, cell_count = self._cell_ids(observations, scheme)
        raw = observations["raw_delta"]
        target = observations["target_delta"]
        sum_xx = np.bincount(cells, weights=raw * raw, minlength=cell_count)
        sum_xy = np.bincount(cells, weights=raw * target, minlength=cell_count)
        counts = np.bincount(cells, minlength=cell_count)
        global_gain = float(
            np.clip(
                np.sum(raw * target) / max(np.sum(raw * raw), 1e-12),
                self.config.analytic_gain_minimum,
                self.config.analytic_gain_maximum,
            )
        )
        nonempty = counts > 0
        mean_energy = float(np.mean(sum_xx[nonempty])) if np.any(nonempty) else 1.0
        penalty = float(shrinkage) * max(mean_energy, 1e-12)
        gains = (sum_xy + penalty * global_gain) / np.maximum(sum_xx + penalty, 1e-12)
        gains[~nonempty] = global_gain
        gains = np.clip(
            gains, self.config.analytic_gain_minimum, self.config.analytic_gain_maximum
        )
        return {
            "scheme": scheme,
            "shrinkage": float(shrinkage),
            "gains": gains.astype(np.float32),
            "counts": counts.astype(np.int64),
            "global_gain": global_gain,
            "cell_count": int(cell_count),
            "nonempty_cell_count": int(np.sum(nonempty)),
        }

    def _direct_metrics(
        self, observations: Mapping[str, np.ndarray], lookup: Mapping[str, Any]
    ) -> Dict[str, Any]:
        cells, _ = self._cell_ids(observations, str(lookup["scheme"]))
        predicted = observations["raw_delta"] * lookup["gains"][cells]
        target = observations["target_delta"]
        reference = observations["raw_delta"] * self.config.analytic_reference_gain
        rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
        reference_rmse = float(np.sqrt(np.mean((reference - target) ** 2)))
        return {
            "rmse_mv": rmse,
            "reference_alpha_075_rmse_mv": reference_rmse,
            "improvement_over_reference_fraction": 1.0
            - rmse / max(reference_rmse, 1e-12),
        }

    def fit_and_calibrate_analytic_lookups(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        schemes = (*CAUSAL_GAIN_SCHEMES, ORACLE_SCHEME)
        per_seed = {}
        progress = atomic._CompactProgress(
            "06b-j analytic fit/calibration",
            len(self.config.pilot_seeds),
            1,
        )
        for seed_index, seed in enumerate(self.config.pilot_seeds, start=1):
            fit = self._teacher_boundary_observations(seed, "fit", device)
            calibration = self._teacher_boundary_observations(seed, "calibration", device)
            per_seed[str(seed)] = {}
            for scheme in schemes:
                candidates = []
                for shrinkage in self.config.analytic_shrinkage_strengths:
                    lookup = self._fit_lookup(fit, scheme, shrinkage)
                    metrics = self._direct_metrics(calibration, lookup)
                    self.lookup_tables[(seed, scheme, shrinkage)] = lookup
                    candidates.append({
                        "shrinkage": float(shrinkage),
                        "direct_calibration": metrics,
                        "global_gain": lookup["global_gain"],
                        "cell_count": lookup["cell_count"],
                        "nonempty_cell_count": lookup["nonempty_cell_count"],
                    })
                selected = min(
                    candidates,
                    key=lambda row: (
                        row["direct_calibration"]["rmse_mv"],
                        -row["shrinkage"],
                    ),
                )
                self.selected_shrinkage[(seed, scheme)] = selected["shrinkage"]
                selected_lookup = self.lookup_tables[
                    (seed, scheme, selected["shrinkage"])
                ]
                np.savez_compressed(
                    self.output_dir / f"analytic_lookup_{scheme}_seed{seed}.npz",
                    gains=selected_lookup["gains"],
                    counts=selected_lookup["counts"],
                    shrinkage=np.asarray(
                        [selected["shrinkage"]], dtype=np.float64
                    ),
                    global_gain=np.asarray(
                        [selected_lookup["global_gain"]], dtype=np.float64
                    ),
                )
                per_seed[str(seed)][scheme] = {
                    "selected": selected,
                    "candidates": candidates,
                }
            progress.update(seed_index, f"seed={seed}")
        valid = all(
            np.isfinite(candidate["direct_calibration"]["rmse_mv"])
            and np.isfinite(candidate["global_gain"])
            for schemes in per_seed.values()
            for row in schemes.values()
            for candidate in row["candidates"]
        )
        report = {
            "schema_version": "06b-j-analytic-calibration-v1",
            "valid": bool(valid),
            "selection_role": "historically_reused_train_calibration",
            "development_accessed": False,
            "oracle_eligible_for_selection": False,
            "neural_training_performed": False,
            "per_seed": per_seed,
        }
        self.calibration_valid = bool(report["valid"])
        serializable = json.loads(json.dumps(report))
        atomic._write_json(self.output_dir / "analytic_gain_calibration.json", serializable)
        return report

    def _lookup_gain(
        self,
        lookup: Mapping[str, Any],
        raw_delta: Any,
        voltage: Any,
        teacher_delta: Any,
    ) -> Any:
        raw_np = raw_delta.detach().cpu().numpy()
        voltage_np = voltage.detach().cpu().numpy()
        teacher_np = teacher_delta.detach().cpu().numpy()
        repeat = raw_np.shape[0]
        observations = {
            "raw_delta": raw_np.reshape(-1),
            "voltage": voltage_np.reshape(-1),
            "target_delta": teacher_np.reshape(-1),
            "region": np.tile(np.asarray(self.layout.segment_region_ids), repeat),
        }
        cells, _ = self._cell_ids(observations, str(lookup["scheme"]))
        gains = lookup["gains"][cells].reshape(raw_np.shape)
        return atomic.torch.as_tensor(gains, dtype=raw_delta.dtype, device=raw_delta.device)

    def _recursive_lookup_evaluation(
        self, seed: int, scheme: str, role: str, device: Any
    ) -> Dict[str, Any]:
        pair = self.source_models[("full_feedback_scalar", seed)]
        shrinkage = self.selected_shrinkage[(seed, scheme)]
        lookup = self.lookup_tables[(seed, scheme, shrinkage)]
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        state_center = atomic.torch.as_tensor(self.statistics["state_center"], device=device)
        state_scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=device)
        delta_scale = atomic.torch.as_tensor(self.statistics["delta_scale"], device=device)
        outputs = {}
        with atomic.torch.no_grad():
            for step in range(self.config.objective_unroll_horizon_ms):
                context = atomic.torch.cat((batch["drive"][:, step], batch["held_ions"]), dim=-1)
                normalized = (current_state - state_center) / state_scale
                raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
                teacher_delta = batch["voltage_t1"][:, step] - batch["voltage_t"][:, step]
                gain = self._lookup_gain(lookup, raw, current_voltage, teacher_delta)
                voltage_delta = raw * gain
                next_voltage = current_voltage + voltage_delta
                state_delta = self._state_forward(pair[1], normalized, current_voltage, voltage_delta, context)
                next_state = current_state + state_delta * delta_scale
                if step + 1 in self.config.rollout_horizons_ms:
                    outputs[f"{step + 1}_ms"] = (next_state, next_voltage, gain)
                current_state, current_voltage = next_state, next_voltage
        horizons = {}
        for horizon, (state, voltage, gain) in outputs.items():
            step = int(horizon[:-3]) - 1
            horizons[horizon] = self._metric(state, voltage, batch, step)
            horizons[horizon]["gain_minimum"] = float(gain.min().cpu())
            horizons[horizon]["gain_median"] = float(gain.median().cpu())
            horizons[horizon]["gain_maximum"] = float(gain.max().cpu())
        endpoint = outputs["8_ms"]
        row = {
            "voltage": endpoint[1].cpu().numpy(),
            "target_voltage": batch["voltage_t1"][:, 7].cpu().numpy(),
            "initial_voltage": batch["voltage_t"][:, 0].cpu().numpy(),
        }
        return {
            "selected_shrinkage": shrinkage,
            "horizons": horizons,
            "activity_at_8ms": self._activity_metrics(row),
            "region_at_8ms": self._region_metrics(row),
        }

    def evaluate_analytic_lookups(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        schemes = (*CAUSAL_GAIN_SCHEMES, ORACLE_SCHEME)
        per_seed = {}
        progress = atomic._CompactProgress(
            "06b-j analytic recursive evaluation",
            len(self.config.pilot_seeds) * len(schemes),
            1,
        )
        completed = 0
        for seed in self.config.pilot_seeds:
            development = self._teacher_boundary_observations(seed, "development", device)
            per_seed[str(seed)] = {}
            for scheme in schemes:
                shrinkage = self.selected_shrinkage[(seed, scheme)]
                lookup = self.lookup_tables[(seed, scheme, shrinkage)]
                per_seed[str(seed)][scheme] = {
                    "direct_teacher_boundary": self._direct_metrics(development, lookup),
                    "recursive": self._recursive_lookup_evaluation(seed, scheme, "development", device),
                }
                completed += 1
                progress.update(completed, f"seed={seed} scheme={scheme}")
        valid = all(
            np.isfinite(row[scheme]["direct_teacher_boundary"]["rmse_mv"])
            and all(
                np.isfinite(metric["voltage_rmse_mv"])
                and metric["nonfinite_voltage_count"] == 0
                for metric in row[scheme]["recursive"]["horizons"].values()
            )
            for row in per_seed.values()
            for scheme in schemes
        )
        report = {
            "schema_version": "06b-j-development-evaluation-v1",
            "valid": bool(valid),
            "role": "historically_reused_train_development",
            "new_independent_confirmation_claimed": False,
            "oracle_eligible_for_selection": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "analytic_gain_development.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _scheme_summary(self, scheme: str, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
        rows = [seed[scheme] for seed in evaluation["per_seed"].values()]
        global_rows = [seed["global"] for seed in evaluation["per_seed"].values()]
        direct = [row["direct_teacher_boundary"]["improvement_over_reference_fraction"] for row in rows]
        endpoint = [row["recursive"]["horizons"]["8_ms"] for row in rows]
        global_endpoint = [row["recursive"]["horizons"]["8_ms"] for row in global_rows]
        gain_over_global = [
            1.0 - row["voltage_rmse_mv"] / max(base["voltage_rmse_mv"], 1e-12)
            for row, base in zip(endpoint, global_endpoint)
        ]
        activity = {
            name: self._median([
                row["recursive"]["activity_at_8ms"][name]["voltage_gain_vs_persistence_fraction"]
                for row in rows
            ])
            for name in rows[0]["recursive"]["activity_at_8ms"]
        }
        region = {
            name: self._median([
                row["recursive"]["region_at_8ms"][name]["voltage_gain_vs_persistence_fraction"]
                for row in rows
            ])
            for name in rows[0]["recursive"]["region_at_8ms"]
        }
        voltage_gains = [row["voltage_improvement_vs_persistence_fraction"] for row in endpoint]
        state_gains = [row["state_improvement_vs_persistence_fraction"] for row in endpoint]
        return {
            "median_direct_improvement_over_alpha075_fraction": self._median(direct),
            "median_recursive_voltage_gain_vs_persistence_fraction": self._median(voltage_gains),
            "median_recursive_STATE_gain_vs_persistence_fraction": self._median(state_gains),
            "minimum_seed_STATE_gain_vs_persistence_fraction": float(min(state_gains)),
            "median_recursive_gain_over_global_lookup_fraction": self._median(gain_over_global),
            "activity_gain_vs_persistence": activity,
            "region_gain_vs_persistence": region,
            "all_seed_voltage_gain_positive": all(value > 0 for value in voltage_gains),
            "all_seed_STATE_gain_positive": all(value > self.config.minimum_STATE_gain_fraction for value in state_gains),
            "physical_voltage_violation_count": int(sum(row["physical_voltage_violation_count"] for row in endpoint)),
        }

    def finalize_analytic_identifiability(self, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
        summaries = {
            scheme: self._scheme_summary(scheme, evaluation)
            for scheme in (*CAUSAL_GAIN_SCHEMES, ORACLE_SCHEME)
        }
        def passes(scheme: str) -> bool:
            row = summaries[scheme]
            return bool(
                row["median_direct_improvement_over_alpha075_fraction"] >= self.config.minimum_direct_improvement_fraction
                and row["median_recursive_gain_over_global_lookup_fraction"] >= self.config.minimum_recursive_gain_over_global_fraction
                and row["all_seed_voltage_gain_positive"]
                and row["minimum_seed_STATE_gain_vs_persistence_fraction"] > self.config.minimum_STATE_gain_fraction
                and row["activity_gain_vs_persistence"]["active_ge_5mV"] >= self.config.minimum_active_gain_fraction
                and row["activity_gain_vs_persistence"]["moderate_1_to_5mV"] >= self.config.minimum_moderate_gain_fraction
                and row["activity_gain_vs_persistence"]["quiescent_lt_1mV"] >= self.config.minimum_quiescent_gain_fraction
                and row["region_gain_vs_persistence"]["soma"] >= self.config.minimum_soma_gain_fraction
                and row["physical_voltage_violation_count"] == 0
            )
        gate_results = {
            scheme: passes(scheme)
            for scheme in (*CAUSAL_GAIN_SCHEMES, ORACLE_SCHEME)
        }
        for scheme, passed in gate_results.items():
            summaries[scheme]["registered_gate_passed"] = passed
        primary_pass = gate_results[PRIMARY_SCHEME]
        fallback_pass = gate_results[FALLBACK_SCHEME]
        oracle_pass = gate_results[ORACLE_SCHEME]
        direct_identified = summaries[PRIMARY_SCHEME]["median_direct_improvement_over_alpha075_fraction"] >= self.config.minimum_direct_improvement_fraction
        if primary_pass:
            diagnosis = "ANALYTIC_CAUSAL_GAIN_FEATURES_IDENTIFIED"
            next_step = "bounded_causal_lookup_confirmation_on_fresh_train_support"
        elif fallback_pass:
            diagnosis = "ANALYTIC_REGION_ACTIVITY_GAIN_IDENTIFIED"
            next_step = "bounded_region_activity_lookup_confirmation_on_fresh_train_support"
        elif direct_identified:
            diagnosis = "STATIC_GAIN_IDENTIFIED_BUT_TEMPORAL_COMPOSITION_FAILS"
            next_step = "atomic_temporal_voltage_correction_state"
        elif oracle_pass:
            diagnosis = "TEACHER_ACTIVITY_ORACLE_WORKS_BUT_CAUSAL_PARTITIONS_FAIL"
            next_step = "causal_activity_state_observability_playground"
        else:
            diagnosis = "CURRENT_CAUSAL_GAIN_PARTITIONS_INSUFFICIENT"
            next_step = "atomic_voltage_bridge_representation_revision"
        report = {
            "schema_version": "06b-j-final-report-v1",
            "valid": bool(self.calibration_valid and evaluation.get("valid")),
            "component_playground_grade": True,
            "new_independent_confirmation_claimed": False,
            "diagnosis": diagnosis,
            "primary_scheme": PRIMARY_SCHEME,
            "fallback_scheme": FALLBACK_SCHEME,
            "primary_passed": primary_pass,
            "fallback_passed": fallback_pass,
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
    "AnalyticCausalGainConfig",
    "AnalyticCausalGainIdentifiability",
    "CAUSAL_GAIN_SCHEMES",
    "EXPECTED_06BI_ARCHIVE_SHA256",
    "EXPECTED_06BI_FINAL_SHA256",
    "EXPECTED_06BI_INDEX_SHA256",
    "FALLBACK_SCHEME",
    "ORACLE_SCHEME",
    "PRIMARY_SCHEME",
    "verified_06bi_artifact_root",
]
