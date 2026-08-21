"""Frozen counterfactual audit of the interaction observed in 05p."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import CompositeFlowmapBundle

from .axial_rich_state_recurrent_canary import (
    AxialRichStateRecurrentCanary,
    AxialRichStateRecurrentCanaryConfig,
)
from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .rollout_aware_architecture_canary import torch


EXPECTED_05P_ARCHIVE_SHA256 = (
    "f133ed92009464a6bc120a13860669230605245b245d6adcb2da946c58f3020e"
)
EXPECTED_05P_INDEX_SHA256 = (
    "8e5e81a6af86fef15e07da589a4ca2d03c23c8d8caeb37b0545992e4d27b63f6"
)
EXPECTED_05P_FINAL_SHA256 = (
    "feb189283547a2143069ac8f74de54284b2f9383e7778e66fa1e8479a4bc3f3a"
)


def verified_axial_rich_state_canary_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="axial_rich_state_recurrent_canary_config.json",
        archive_sha256=EXPECTED_05P_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05P_INDEX_SHA256,
        final_sha256=EXPECTED_05P_FINAL_SHA256,
    )


@dataclass(frozen=True)
class TemporalObservabilityReassessmentConfig(
    AxialRichStateRecurrentCanaryConfig
):
    permutation_seed: int = 510073
    checkpoint_reproduction_atol_mv: float = 1.0e-4
    joint_material_gain_fraction: float = 0.05
    ablation_material_degradation_fraction: float = 0.05
    state_identity_degradation_fraction: float = 0.02
    minimum_counterfactual_seed_count: int = 2

    def validate(self) -> None:
        super().validate()
        if self.permutation_seed <= 0:
            raise ValueError("05q permutation seed must be positive")
        if not 0 < self.checkpoint_reproduction_atol_mv <= 1.0e-3:
            raise ValueError("05q checkpoint reproduction tolerance is invalid")
        for value in (
            self.joint_material_gain_fraction,
            self.ablation_material_degradation_fraction,
            self.state_identity_degradation_fraction,
        ):
            if not 0 < value < 1:
                raise ValueError("05q materiality threshold is invalid")
        if self.minimum_counterfactual_seed_count not in {2, 3}:
            raise ValueError("05q robust seed gate is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "TemporalObservabilityReassessmentConfig":
        payload = dict(values)
        for name in ("horizons_ms", "seeds"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class TemporalObservabilityContractReassessment(
    AxialRichStateRecurrentCanary
):
    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: TemporalObservabilityReassessmentConfig,
        artifact_05p_source: Path,
        artifact_05o_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05o_source,
            code_revision=code_revision,
        )
        self.artifact_05p_source = Path(artifact_05p_source).resolve()
        self.artifact_05p_root: Path | None = None
        self.artifact_05p_report: Dict[str, Any] = {}
        self.artifact_05p_contract: Dict[str, Any] = {}
        self.development_window_regimes: Tuple[str, ...] = ()

    @property
    def temporal_config(self) -> TemporalObservabilityReassessmentConfig:
        return self.config  # type: ignore[return-value]

    def prepare(self) -> Dict[str, Any]:
        support = super().prepare()
        root, final, contract = verified_axial_rich_state_canary_artifact_root(
            self.artifact_05p_source,
            self.output_dir.parent / ".05q_artifact_cache" / "05p",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05p is not decision-grade")
        if final.get("diagnosis") != "NONLINEAR_ONE_STEP_SIGNAL_DID_NOT_TRANSFER_TO_ROLLOUT":
            blockers.append("05p diagnosis does not authorize temporal reassessment")
        if final.get("next_step") != "05q_temporal_observability_contract_reassessment":
            blockers.append("05p next step is not 05q")
        if not final.get("joint_candidate_robust"):
            blockers.append("05p joint candidate is not robust")
        if final.get("recurrent_expansion_authorized"):
            blockers.append("05p unexpectedly authorized recurrent expansion")
        source_config = final.get("preflight", {})
        for name in (
            "dataset_fingerprint",
            "role_episode_counts",
            "role_window_counts",
            "state_sketch_shape",
            "parameter_counts",
        ):
            current = (
                self.bundle.fingerprint
                if name == "dataset_fingerprint"
                else support.get(name)
            )
            expected = (
                final.get(name)
                if name == "dataset_fingerprint"
                else source_config.get(name)
            )
            if current != expected:
                blockers.append(f"05q reconstructed {name} differs from 05p")
        self.artifact_05p_root = root
        self.artifact_05p_report = final
        self.artifact_05p_contract = contract
        regimes = []
        for row in self.roles["development"]:
            indices = self.store.trajectory_indices[str(row["trajectory_id"])]
            regimes.extend(
                [str(row["05l_regime"])] * len(self._episode_windows(indices))
            )
        self.development_window_regimes = tuple(regimes)
        expected_windows = len(self.materialized["development"]["initial_voltage"])
        if len(regimes) != expected_windows:
            blockers.append("05q development regime labels do not align with windows")
        regime_counts = {
            name: regimes.count(name) for name in sorted(set(regimes))
        }
        if any(count < 2 for count in regime_counts.values()):
            blockers.append("05q cannot permute state within a singleton regime")
        report = {
            "schema_version": "05q-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05p": contract,
            "artifact_05o": support["artifact_05o"],
            "dataset_fingerprint": self.bundle.fingerprint,
            "checkpoint_count": sum(
                len(rows) for rows in final.get("runs", {}).values()
            ),
            "frozen_checkpoint_evaluation_only": True,
            "retraining_performed": False,
            "fit_or_calibration_used_for_new_selection": False,
            "development_role_reconstructed": True,
            "validation_or_test_loaded": False,
            "sealed_fresh_test_loaded": False,
            "teacher_future_state_used_as_input": False,
            "counterfactual_modes": [
                "closed_loop_authentic",
                "closed_loop_zero_state",
                "closed_loop_permuted_state",
                "closed_loop_zero_axial",
                "closed_loop_zero_both",
                "teacher_voltage_authentic",
            ],
            "state_permutation_within_regime": True,
            "development_window_regime_counts": regime_counts,
            "support_reconstruction": support,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "temporal_observability_reassessment_config.json",
            {
                "schema_version": "05q-config-v1",
                "config": asdict(self.temporal_config),
                "artifact_05p": contract,
                "artifact_05o": support["artifact_05o"],
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05q preflight failed: {blockers}")
        return report

    @staticmethod
    def _metrics(
        prediction: Any, values: Mapping[str, Any], horizon: int
    ) -> Dict[str, Any]:
        target = values["target_voltage"][:, :horizon]
        endpoint_error = prediction[:, horizon - 1] - target[:, horizon - 1]
        persistence_error = (
            values["initial_voltage"] - target[:, horizon - 1]
        )
        regenerative = (
            torch.abs(target[:, horizon - 1] - values["initial_voltage"]) >= 5.0
        ) | (target[:, horizon - 1] >= -20.0)
        regenerative_error = endpoint_error[regenerative]
        return {
            "endpoint_rmse_mv": float(
                torch.sqrt(torch.mean(endpoint_error.square())).cpu()
            ),
            "regenerative_endpoint_rmse_mv": float(
                torch.sqrt(torch.mean(regenerative_error.square())).cpu()
            ) if regenerative_error.numel() else math.nan,
            "path_rmse_mv": float(
                torch.sqrt(torch.mean((prediction[:, :horizon] - target).square())).cpu()
            ),
            "endpoint_mean_drift_mv": float(torch.mean(endpoint_error).cpu()),
            "persistence_endpoint_rmse_mv": float(
                torch.sqrt(torch.mean(persistence_error.square())).cpu()
            ),
            "regenerative_coordinate_count": int(regenerative.sum().cpu()),
            "finite": bool(torch.isfinite(prediction[:, :horizon]).all().cpu()),
        }

    def _state_variant(
        self, values: Mapping[str, Any], mode: str, permutation: Any
    ) -> Any:
        state = values["initial_state"]
        if mode == "zero":
            return torch.zeros_like(state)
        if mode == "permuted":
            return state.index_select(0, permutation)
        return state

    def _closed_prediction(
        self,
        model: Any,
        values: Mapping[str, Any],
        *,
        state_mode: str,
        axial_enabled: bool,
        permutation: Any,
    ) -> Any:
        original = model.use_axial
        model.use_axial = bool(axial_enabled)
        try:
            return model(
                values["initial_voltage"],
                values["causal_drive"],
                self._state_variant(values, state_mode, permutation),
            )["voltage"]
        finally:
            model.use_axial = original

    def _teacher_voltage_prediction(
        self,
        model: Any,
        values: Mapping[str, Any],
        *,
        state_mode: str,
        axial_enabled: bool,
        permutation: Any,
    ) -> Any:
        original = model.use_axial
        model.use_axial = bool(axial_enabled)
        try:
            state = self._state_variant(values, state_mode, permutation)
            voltage = values["initial_voltage"]
            zero_drive = voltage.new_zeros(
                voltage.shape[0], model.segment_count, len(values["causal_drive"][0, 0, 0])
            )
            hidden = torch.tanh(
                model.initial_encoder(
                    model._input(voltage, zero_drive, state, initial=True)
                )
            )
            predictions = []
            for step in range(values["causal_drive"].shape[1]):
                boundary_voltage = (
                    values["initial_voltage"]
                    if step == 0
                    else values["target_voltage"][:, step - 1]
                )
                encoded = model.input_encoder(
                    model._input(
                        boundary_voltage,
                        values["causal_drive"][:, step],
                        state,
                        initial=False,
                    )
                )
                hidden = model.gru(
                    (encoded + model._mix(hidden)).reshape(-1, model.hidden_width),
                    hidden.reshape(-1, model.hidden_width),
                ).reshape_as(hidden)
                delta = model.voltage_delta_limit_mv * torch.tanh(
                    model.voltage_head(hidden).squeeze(-1)
                )
                predictions.append(boundary_voltage + delta)
            return torch.stack(predictions, dim=1)
        finally:
            model.use_axial = original

    def _evaluate_prediction(
        self, prediction: Any, values: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return {
            str(horizon): self._metrics(prediction, values, horizon)
            for horizon in range(1, max(self.config.horizons_ms) + 1)
        }

    @staticmethod
    def _comparison(
        results: Mapping[str, Any], candidate: str, baseline: str, seeds: Tuple[int, ...]
    ) -> Dict[str, Any]:
        by_seed = {}
        for seed in map(str, seeds):
            left = results[candidate][seed]["8"]
            right = results[baseline][seed]["8"]
            by_seed[seed] = {
                "rmse_gain_fraction": 1.0
                - left["endpoint_rmse_mv"] / max(right["endpoint_rmse_mv"], 1e-12),
                "regenerative_rmse_gain_fraction": 1.0
                - left["regenerative_endpoint_rmse_mv"]
                / max(right["regenerative_endpoint_rmse_mv"], 1e-12),
            }
        return {
            "by_seed": by_seed,
            "median_rmse_gain_fraction": float(np.median([
                row["rmse_gain_fraction"] for row in by_seed.values()
            ])),
            "median_regenerative_gain_fraction": float(np.median([
                row["regenerative_rmse_gain_fraction"] for row in by_seed.values()
            ])),
            "positive_win_count": sum(
                row["rmse_gain_fraction"] > 0 for row in by_seed.values()
            ),
        }

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None or self.artifact_05p_root is None:
            raise RuntimeError("05q requires PyTorch and a verified 05p artifact")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = self._tensor_role("development", device)
        rng = np.random.default_rng(self.temporal_config.permutation_seed)
        permutation_values = np.arange(
            values["initial_state"].shape[0], dtype=np.int64
        )
        labels = np.asarray(self.development_window_regimes)
        for regime in sorted(set(self.development_window_regimes)):
            positions = np.flatnonzero(labels == regime)
            permutation_values[positions] = rng.permutation(positions)
        permutation = torch.as_tensor(
            permutation_values, dtype=torch.long, device=device
        )
        modes: Dict[str, Dict[str, Any]] = {
            name: {} for name in (
                "closed_loop_authentic",
                "teacher_voltage_authentic",
                "joint_zero_state",
                "joint_permuted_state",
                "joint_zero_axial",
                "joint_zero_both",
            )
        }
        reproduction_errors = []
        families = tuple(self.CONTRACTS)
        with torch.no_grad():
            for family in families:
                for seed in self.config.seeds:
                    seed_key = str(seed)
                    model = self._model(family, device)
                    source = self.artifact_05p_report["runs"][family][seed_key]
                    checkpoint = torch.load(
                        self.artifact_05p_root / source["checkpoint"],
                        map_location=device,
                        weights_only=False,
                    )
                    model.load_state_dict(checkpoint["state_dict"])
                    model.eval()
                    axial_enabled, _ = self.CONTRACTS[family]
                    closed = self._evaluate_prediction(
                        self._closed_prediction(
                            model,
                            values,
                            state_mode="authentic",
                            axial_enabled=axial_enabled,
                            permutation=permutation,
                        ),
                        values,
                    )
                    teacher = self._evaluate_prediction(
                        self._teacher_voltage_prediction(
                            model,
                            values,
                            state_mode="authentic",
                            axial_enabled=axial_enabled,
                            permutation=permutation,
                        ),
                        values,
                    )
                    modes["closed_loop_authentic"].setdefault(family, {})[seed_key] = closed
                    modes["teacher_voltage_authentic"].setdefault(family, {})[seed_key] = teacher
                    stored = source["development"]["8"]["endpoint_rmse_mv"]
                    reproduction_errors.append(abs(closed["8"]["endpoint_rmse_mv"] - stored))
                    if family == "graphgru_axial_rich_state":
                        for mode, state_mode, use_axial in (
                            ("joint_zero_state", "zero", True),
                            ("joint_permuted_state", "permuted", True),
                            ("joint_zero_axial", "authentic", False),
                            ("joint_zero_both", "zero", False),
                        ):
                            prediction = self._closed_prediction(
                                model,
                                values,
                                state_mode=state_mode,
                                axial_enabled=use_axial,
                                permutation=permutation,
                            )
                            modes[mode].setdefault(family, {})[seed_key] = (
                                self._evaluate_prediction(prediction, values)
                            )
                    print(
                        f"[HayFlow 05q][frozen] {family} seed={seed} "
                        f"closed8={closed['8']['endpoint_rmse_mv']:.3f} "
                        f"teacher8={teacher['8']['endpoint_rmse_mv']:.3f}",
                        flush=True,
                    )
        maximum_reproduction_error = max(reproduction_errors, default=math.inf)
        if maximum_reproduction_error > self.temporal_config.checkpoint_reproduction_atol_mv:
            raise RuntimeError("05q frozen checkpoint reproduction disagrees with 05p")
        closed = modes["closed_loop_authentic"]
        teacher = modes["teacher_voltage_authentic"]
        joint = "graphgru_axial_rich_state"
        voltage = "graphgru_voltage_only"
        closed_joint_gain = self._comparison(
            {"joint": closed[joint], "voltage": closed[voltage]},
            "joint", "voltage", self.config.seeds,
        )
        teacher_joint_gain = self._comparison(
            {"joint": teacher[joint], "voltage": teacher[voltage]},
            "joint", "voltage", self.config.seeds,
        )
        ablations = {}
        for name in (
            "joint_zero_state",
            "joint_permuted_state",
            "joint_zero_axial",
            "joint_zero_both",
        ):
            ablations[name] = self._comparison(
                {"authentic": closed[joint], "ablated": modes[name][joint]},
                "authentic", "ablated", self.config.seeds,
            )

        def material(row: Mapping[str, Any], threshold: float) -> bool:
            return bool(
                row["median_rmse_gain_fraction"] >= threshold
                and row["median_regenerative_gain_fraction"]
                >= -self.config.regenerative_noninferiority_margin_fraction
                and row["positive_win_count"]
                >= self.temporal_config.minimum_counterfactual_seed_count
            )

        joint_signal = material(
            closed_joint_gain, self.temporal_config.joint_material_gain_fraction
        )
        axial_reliance = material(
            ablations["joint_zero_axial"],
            self.temporal_config.ablation_material_degradation_fraction,
        )
        state_reliance = material(
            ablations["joint_zero_state"],
            self.temporal_config.ablation_material_degradation_fraction,
        )
        state_identity = material(
            ablations["joint_permuted_state"],
            self.temporal_config.state_identity_degradation_fraction,
        )
        teacher_signal = material(
            teacher_joint_gain, self.temporal_config.joint_material_gain_fraction
        )
        exposure_limited = bool(teacher_signal and not joint_signal)
        synergy = bool(joint_signal and axial_reliance and state_reliance and state_identity)
        if synergy:
            diagnosis = "TEMPORAL_JOINT_GRAPH_STATE_SYNERGY_CONFIRMED"
            next_step = "05r_joint_graph_state_rollout_expansion"
        elif exposure_limited:
            diagnosis = "JOINT_SIGNAL_PRESENT_ONLY_AT_TEACHER_BOUNDARIES"
            next_step = "05r_exposure_aware_state_propagation_canary"
        elif joint_signal and not state_identity:
            diagnosis = "JOINT_GAIN_WITHOUT_STATE_IDENTITY_DEPENDENCE"
            next_step = "05r_state_contract_identity_reassessment"
        else:
            diagnosis = "FROZEN_COUNTERFACTUALS_DO_NOT_SUPPORT_TEMPORAL_SYNERGY"
            next_step = "05r_temporal_representation_reassessment"
        report = {
            "schema_version": "05q-final-report-v1",
            "valid": True,
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05p": self.artifact_05p_contract,
            "artifact_05o": self.prepare_report["artifact_05o"],
            "dataset_fingerprint": self.bundle.fingerprint,
            "maximum_checkpoint_reproduction_error_mv": maximum_reproduction_error,
            "counterfactual_results": modes,
            "closed_loop_joint_gain": closed_joint_gain,
            "teacher_voltage_joint_gain": teacher_joint_gain,
            "joint_ablation_degradations": ablations,
            "joint_closed_loop_signal": joint_signal,
            "teacher_voltage_joint_signal": teacher_signal,
            "joint_axial_reliance": axial_reliance,
            "joint_state_reliance": state_reliance,
            "joint_state_identity_signal": state_identity,
            "exposure_limited": exposure_limited,
            "temporal_synergy_confirmed": synergy,
            "retraining_performed": False,
            "model_or_training_authorized": False,
            "validation_or_test_loaded": False,
            "sealed_fresh_test_loaded": False,
            "bounded_rollout_expansion_authorized": synergy,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
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
        _write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "05q-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05P_ARCHIVE_SHA256",
    "EXPECTED_05P_FINAL_SHA256",
    "EXPECTED_05P_INDEX_SHA256",
    "TemporalObservabilityContractReassessment",
    "TemporalObservabilityReassessmentConfig",
    "verified_axial_rich_state_canary_artifact_root",
]
