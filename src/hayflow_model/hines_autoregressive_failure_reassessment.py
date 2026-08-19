"""Frozen causal interventions for the failed 05k autoregressive rollout."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_frozen_candidate_micro_rollout import (
    HinesFrozenCandidateMicroRollout,
    rollout_voltage_metrics,
)
from .hines_layer import require_torch
from .hines_regenerative_confirmation import _verified_artifact_root

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05K_ARCHIVE_SHA256 = (
    "0e28e16e9f4b7e14830495ae74382d2981ba6b402e95fd200b537dafaabe4ceb"
)
EXPECTED_05K_INDEX_SHA256 = (
    "a471bac2fbb643018aea760269087082408063245028a319cceb58e05ff25a95"
)
EXPECTED_05K_FINAL_SHA256 = (
    "2136c05fd459f529953ea0fe5e4e0ad7c1c428125905267f6b94beab594c39a5"
)
EXPECTED_05K_MICRO_SHA256 = (
    "e0af4b102123d4bdcae8cf6830867ca01ca61dcf66ad2e8a1a4504f091586a4b"
)


@dataclass(frozen=True)
class HinesAutoregressiveFailureReassessmentConfig:
    seeds: Tuple[int, ...] = (17, 29, 43)
    horizons_ms: Tuple[int, ...] = (2, 4, 8)
    intervention_modes: Tuple[str, ...] = (
        "teacher_boundary_reset",
        "teacher_voltage_clamp",
        "teacher_latent_reset",
        "decoder_no_feedback",
        "residual_first_step_only",
    )
    material_error_reduction_fraction: float = 0.25
    strong_error_reduction_fraction: float = 0.50
    physical_voltage_min_mv: float = -150.0
    physical_voltage_max_mv: float = 100.0
    stored_prediction_atol_mv: float = 1e-6

    def validate(self) -> None:
        expected_modes = (
            "teacher_boundary_reset",
            "teacher_voltage_clamp",
            "teacher_latent_reset",
            "decoder_no_feedback",
            "residual_first_step_only",
        )
        if self.seeds != (17, 29, 43) or self.horizons_ms != (2, 4, 8):
            raise ValueError("05k-b must retain every frozen seed and 2/4/8 ms horizon")
        if self.intervention_modes != expected_modes:
            raise ValueError("05k-b causal intervention matrix is preregistered")
        if not 0 < self.material_error_reduction_fraction < self.strong_error_reduction_fraction < 1:
            raise ValueError("05k-b attribution thresholds are invalid")
        if self.physical_voltage_min_mv >= self.physical_voltage_max_mv:
            raise ValueError("05k-b physical voltage bounds are reversed")
        if self.stored_prediction_atol_mv <= 0:
            raise ValueError("05k-b stored-prediction tolerance must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesAutoregressiveFailureReassessmentConfig":
        payload = dict(values)
        for name in ("seeds", "horizons_ms", "intervention_modes"):
            if name in payload:
                payload[name] = tuple(payload[name])
        if "seeds" in payload:
            payload["seeds"] = tuple(map(int, payload["seeds"]))
        if "horizons_ms" in payload:
            payload["horizons_ms"] = tuple(map(int, payload["horizons_ms"]))
        result = cls(**payload)
        result.validate()
        return result


def verified_micro_rollout_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """Verify every indexed member of the immutable failed 05k artifact."""

    root, report, contract = _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="frozen_candidate_micro_rollout_config.json",
        archive_sha256=EXPECTED_05K_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05K_INDEX_SHA256,
        final_sha256=EXPECTED_05K_FINAL_SHA256,
    )
    micro_path = root / "micro_rollout_report.json"
    from .hines_isolation_experiment import sha256_file

    if not micro_path.is_file() or sha256_file(micro_path) != EXPECTED_05K_MICRO_SHA256:
        raise RuntimeError("05k micro-rollout report SHA-256 mismatch")
    return root, report, contract


def classify_autoregressive_failure(
    median_reduction_by_mode: Mapping[str, float],
    *,
    material_reduction: float,
    strong_reduction: float,
) -> Dict[str, Any]:
    """Assign a descriptive mechanism without authorizing model selection."""

    values = {str(key): float(value) for key, value in median_reduction_by_mode.items()}
    if not values or not all(np.isfinite(list(values.values()))):
        raise ValueError("05k-b attribution reductions must be finite and non-empty")
    dominant = max(values, key=values.get)
    best = values[dominant]
    if best < material_reduction:
        diagnosis = "COUPLED_AUTOREGRESSIVE_INSTABILITY_NOT_ISOLATED"
    elif dominant == "teacher_boundary_reset":
        diagnosis = "CLOSED_LOOP_STATE_DISTRIBUTION_SHIFT"
    elif dominant == "teacher_voltage_clamp":
        diagnosis = "VOLTAGE_FEEDBACK_DOMINANT_INSTABILITY"
    elif dominant == "teacher_latent_reset":
        diagnosis = "LATENT_STATE_FEEDBACK_DOMINANT_INSTABILITY"
    elif dominant == "decoder_no_feedback":
        diagnosis = "DECODER_FEEDBACK_COMPOUNDING"
    else:
        diagnosis = "REPEATED_RESIDUAL_APPLICATION_INSTABILITY"
    return {
        "diagnosis": diagnosis,
        "dominant_intervention": dominant,
        "median_error_reduction_fraction": best,
        "materially_isolated": bool(best >= material_reduction),
        "strongly_isolated": bool(best >= strong_reduction),
        "all_intervention_reductions": values,
    }


class HinesAutoregressiveFailureReassessment(HinesFrozenCandidateMicroRollout):
    """Locate the feedback surface responsible for 05k without retraining."""

    def __init__(
        self,
        *args: Any,
        failure_reassessment_config: HinesAutoregressiveFailureReassessmentConfig,
        artifact_05k_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        failure_reassessment_config.validate()
        self.autoregressive_reassessment = failure_reassessment_config
        self.artifact_05k_source = Path(artifact_05k_source).resolve()
        self.artifact_05k_root = Path()
        self.artifact_05k_report: Dict[str, Any] = {}
        self.artifact_05k_contract: Dict[str, Any] = {}
        self.stored_05k_predictions: Dict[str, np.ndarray] = {}

    def prepare_autoregressive_failure_reassessment(self) -> Dict[str, Any]:
        root, report, contract = verified_micro_rollout_artifact_root(
            self.artifact_05k_source,
            self.output_dir.parent / ".05k_b_artifact_cache" / "05k",
        )
        micro = report.get("micro_rollout", {})
        blockers = []
        if not report.get("valid"):
            blockers.append("05k artifact is invalid")
        if report.get("diagnosis") != "FROZEN_CANDIDATE_FAILS_AUTOREGRESSIVE_MICRO_ROLLOUT":
            blockers.append("05k is not the registered autoregressive failure")
        if report.get("candidate_retained") or report.get(
            "limited_rollout_aware_training_canary_authorized"
        ):
            blockers.append("05k unexpectedly retained or authorized the candidate")
        if report.get("next_step") != "05k_b_autoregressive_failure_reassessment":
            blockers.append("05k did not prescribe this reassessment")
        if int(micro.get("passing_seed_count", -1)) != 0:
            blockers.append("05k failure is not shared by all frozen seeds")
        if micro.get("future_teacher_membrane_or_ion_states_injected"):
            blockers.append("05k rollout leaked future teacher state")
        if blockers:
            raise RuntimeError(f"05k-b provenance blockers: {blockers}")
        base = self.prepare_frozen_candidate_micro_rollout()
        with np.load(root / "micro_rollout_predictions.npz") as stored:
            self.stored_05k_predictions = {
                name: np.asarray(stored[name], dtype=np.float32) for name in stored.files
            }
        expected_shape = (64, 8, self.layout.segment_count)
        required = {"target", "h2", "persistence", "seed_17", "seed_29", "seed_43"}
        if not required.issubset(self.stored_05k_predictions):
            raise RuntimeError("05k-b stored rollout prediction members are incomplete")
        if any(self.stored_05k_predictions[name].shape != expected_shape for name in required):
            raise RuntimeError("05k-b stored rollout prediction shape mismatch")
        self.artifact_05k_root = root
        self.artifact_05k_report = report
        self.artifact_05k_contract = contract
        payload = {
            "schema_version": "05k-b-autoregressive-failure-reassessment-config-v1",
            "valid": True,
            "reassessment": asdict(self.autoregressive_reassessment),
            "artifact_05k": contract,
            "fixed_intervention_matrix": True,
            "retraining_performed": False,
            "checkpoint_selection_performed": False,
            "fresh_test_used_for_new_model_selection": False,
            "oracle_interventions_are_diagnostic_only": True,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "autoregressive_failure_reassessment_config.json", payload)
        return {**base, **payload}

    def _intervention_rollout(
        self,
        h2: Any,
        decoder: Any,
        transform: Mapping[str, Any],
        window: np.ndarray,
        device: Any,
        mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        recurrent = None
        trace: List[np.ndarray] = []
        residual_rms: List[float] = []
        feature_rms: List[float] = []
        with torch.no_grad():
            for step, logical_index in enumerate(window):
                raw = self._batch([int(logical_index)], include_targets=False)
                original = self._torch_batch(raw, device)
                teacher_initial = h2.initialise(original)
                if recurrent is None or mode == "teacher_boundary_reset":
                    active = None
                    state_voltage = original["voltage_t"]
                elif mode == "teacher_voltage_clamp":
                    active = {**recurrent, "voltage": original["voltage_t"]}
                    state_voltage = original["voltage_t"]
                elif mode == "teacher_latent_reset":
                    active = {
                        "voltage": recurrent["voltage"],
                        "local": teacher_initial["local"],
                        "global": teacher_initial["global"],
                        "calcium": teacher_initial["calcium"],
                        "synapse": teacher_initial["synapse"],
                    }
                    state_voltage = recurrent["voltage"]
                else:
                    active = recurrent
                    state_voltage = recurrent["voltage"]
                batch = self._hide_future_teacher_state(original, active)
                batch = self._causal_frontend_batch(
                    batch, int(logical_index), state_voltage, device
                )
                output = h2(
                    batch,
                    recurrent=active,
                    ablation="H2",
                    decode_teacher=False,
                    boundary_mode="no_event_jump",
                )
                role = {
                    "h2_raw": output["boundary_features"].detach().cpu().double().numpy(),
                    "causal_raw": self._causal_surface(batch).detach().cpu().double().numpy(),
                    "voltage_t": state_voltage.detach().cpu().double().numpy(),
                    "base": output["voltage"].detach().cpu().double().numpy(),
                }
                design = self._normalize_raw_topology(
                    self._raw_topology_design(role, transform), transform
                )
                features = torch.as_tensor(design, dtype=torch.float32, device=device)
                residual = decoder(features)
                corrected = output["voltage"] + residual
                if mode == "residual_first_step_only" and step > 0:
                    corrected = output["voltage"]
                    residual = torch.zeros_like(residual)
                raw_recurrent = {
                    key: output[key]
                    for key in ("voltage", "local", "global", "calcium", "synapse")
                }
                corrected_recurrent = {**raw_recurrent, "voltage": corrected}
                if mode == "teacher_boundary_reset":
                    recurrent = None
                elif mode == "decoder_no_feedback":
                    recurrent = raw_recurrent
                else:
                    recurrent = corrected_recurrent
                trace.append(corrected[0].detach().cpu().numpy())
                residual_rms.append(float(torch.sqrt(torch.mean(residual**2)).cpu()))
                feature_rms.append(float(np.sqrt(np.mean(np.asarray(design) ** 2))))
        return (
            np.asarray(trace, dtype=np.float32),
            np.asarray(residual_rms, dtype=np.float32),
            np.asarray(feature_rms, dtype=np.float32),
        )

    def evaluate_failure_interventions(self) -> Dict[str, Any]:
        require_torch()
        if self.fresh_store is None or not self.refit_transform or not self.stored_05k_predictions:
            raise RuntimeError("05k-b requires the verified frozen representation and 05k predictions")
        registered_transform, transform_error = self._load_registered_transform()
        if transform_error > self.fresh_config.transform_reproduction_atol:
            raise RuntimeError(f"05k-b registered transform mismatch: {transform_error}")
        max_horizon = max(self.autoregressive_reassessment.horizons_ms)
        windows = self._pair_windows(max_horizon)
        targets = np.stack(
            [
                self.fresh_store.read_state(window, "t_plus_1")[:, : self.layout.segment_count]
                for window in windows
            ]
        ).astype(np.float32)
        target_reproduction_error = float(
            np.max(np.abs(targets - self.stored_05k_predictions["target"]))
        )
        if target_reproduction_error > self.autoregressive_reassessment.stored_prediction_atol_mv:
            raise RuntimeError("05k-b fresh targets disagree with the immutable 05k artifact")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2, _ = self._load_h2_checkpoint(device)
        h2.eval()
        original_store = self.store
        self.store = self.fresh_store
        predictions: Dict[str, np.ndarray] = {}
        residuals: Dict[str, np.ndarray] = {}
        features: Dict[str, np.ndarray] = {}
        try:
            feature_width = int(self.refit_designs["fit"].shape[-1])
            total = len(self.autoregressive_reassessment.seeds) * len(
                self.autoregressive_reassessment.intervention_modes
            ) * len(windows)
            progress = Progress("05k-b causal interventions", total)
            completed = 0
            for seed in self.autoregressive_reassessment.seeds:
                decoder = self._load_frozen_decoder(seed, feature_width, device)
                for mode in self.autoregressive_reassessment.intervention_modes:
                    mode_trace, mode_residual, mode_feature = [], [], []
                    for window in windows:
                        trace, residual_rms, feature_rms = self._intervention_rollout(
                            h2, decoder, registered_transform, window, device, mode
                        )
                        mode_trace.append(trace)
                        mode_residual.append(residual_rms)
                        mode_feature.append(feature_rms)
                        completed += 1
                        progress.update(completed)
                    key = f"seed_{seed}__{mode}"
                    predictions[key] = np.asarray(mode_trace, dtype=np.float32)
                    residuals[key] = np.asarray(mode_residual, dtype=np.float32)
                    features[key] = np.asarray(mode_feature, dtype=np.float32)
                del decoder
        finally:
            self.store = original_store
            del h2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows: List[Dict[str, Any]] = []
        metrics: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        for seed in self.autoregressive_reassessment.seeds:
            seed_key = str(seed)
            metrics[seed_key] = {}
            closed = self.stored_05k_predictions[f"seed_{seed}"]
            for mode in self.autoregressive_reassessment.intervention_modes:
                key = f"seed_{seed}__{mode}"
                metrics[seed_key][mode] = {}
                for horizon in self.autoregressive_reassessment.horizons_ms:
                    result = rollout_voltage_metrics(
                        predictions[key][:, :horizon, :],
                        targets[:, :horizon, :],
                        physical_min_mv=self.autoregressive_reassessment.physical_voltage_min_mv,
                        physical_max_mv=self.autoregressive_reassessment.physical_voltage_max_mv,
                    )
                    closed_result = rollout_voltage_metrics(
                        closed[:, :horizon, :],
                        targets[:, :horizon, :],
                        physical_min_mv=self.autoregressive_reassessment.physical_voltage_min_mv,
                        physical_max_mv=self.autoregressive_reassessment.physical_voltage_max_mv,
                    )
                    reduction = 1.0 - result["endpoint_voltage_rmse_mv"] / max(
                        closed_result["endpoint_voltage_rmse_mv"], 1e-12
                    )
                    enriched = {
                        **result,
                        "error_reduction_vs_closed_loop_fraction": reduction,
                        "residual_rms_mv": float(np.mean(residuals[key][:, :horizon])),
                        "normalized_feature_rms": float(np.mean(features[key][:, :horizon])),
                    }
                    metrics[seed_key][mode][str(horizon)] = enriched
                    rows.append(
                        {
                            "seed": int(seed),
                            "mode": mode,
                            "horizon_ms": int(horizon),
                            "endpoint_rmse_mv": enriched["endpoint_voltage_rmse_mv"],
                            "endpoint_drift_mv": enriched["endpoint_mean_drift_mv"],
                            "error_reduction_vs_closed_loop_fraction": reduction,
                            "median_branching_retention": enriched["median_branching_retention"],
                            "physical_voltage_violation_count": enriched[
                                "physical_voltage_violation_count"
                            ],
                            "residual_rms_mv": enriched["residual_rms_mv"],
                            "normalized_feature_rms": enriched["normalized_feature_rms"],
                        }
                    )
        horizon_key = str(max_horizon)
        reductions = {
            mode: float(
                np.median(
                    [
                        metrics[str(seed)][mode][horizon_key][
                            "error_reduction_vs_closed_loop_fraction"
                        ]
                        for seed in self.autoregressive_reassessment.seeds
                    ]
                )
            )
            for mode in self.autoregressive_reassessment.intervention_modes
        }
        attribution = classify_autoregressive_failure(
            reductions,
            material_reduction=self.autoregressive_reassessment.material_error_reduction_fraction,
            strong_reduction=self.autoregressive_reassessment.strong_error_reduction_fraction,
        )
        finite = bool(
            all(np.all(np.isfinite(values)) for values in predictions.values())
            and all(np.all(np.isfinite(values)) for values in residuals.values())
            and all(np.all(np.isfinite(values)) for values in features.values())
        )
        np.savez_compressed(
            self.output_dir / "failure_intervention_predictions.npz",
            target=targets,
            **predictions,
        )
        write_parquet(self.output_dir / "failure_intervention_metrics.parquet", rows)
        report = {
            "schema_version": "05k-b-autoregressive-failure-reassessment-v1",
            "valid": bool(finite and target_reproduction_error <= self.autoregressive_reassessment.stored_prediction_atol_mv),
            "device": str(device),
            "pair_count": len(windows) // 2,
            "episode_count": len(windows),
            "horizons_ms": list(self.autoregressive_reassessment.horizons_ms),
            "intervention_modes": list(self.autoregressive_reassessment.intervention_modes),
            "target_reproduction_error_mv": target_reproduction_error,
            "metrics": metrics,
            "attribution": attribution,
            "retraining_performed": False,
            "checkpoint_selection_performed": False,
            "fresh_test_used_for_new_model_selection": False,
            "oracle_interventions_are_diagnostic_only": True,
            "model_or_training_authorized": False,
        }
        _write_json(self.output_dir / "autoregressive_failure_reassessment.json", report)
        if not report["valid"]:
            raise RuntimeError("05k-b causal intervention reassessment is numerically invalid")
        return report

    def finalize_failure_reassessment(
        self, reassessment_report: Mapping[str, Any]
    ) -> Dict[str, Any]:
        attribution = dict(reassessment_report.get("attribution", {}))
        report = {
            "schema_version": "05k-b-final-report-v1",
            "valid": bool(reassessment_report.get("valid")),
            "decision": "AUTOREGRESSIVE_FAILURE_CAUSAL_REASSESSMENT",
            "diagnosis": attribution.get("diagnosis"),
            "code_revision": self.code_revision,
            "artifact_05k": self.artifact_05k_contract,
            "reassessment": dict(reassessment_report),
            "candidate_reinstated": False,
            "training_authorized": False,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "fresh_test_is_now_consumed_for_diagnosis": True,
            "future_candidate_requires_new_sealed_test": True,
            "next_step": "05k_c_development_only_autoregressive_repair_design",
        }
        _write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_05K_ARCHIVE_SHA256",
    "EXPECTED_05K_FINAL_SHA256",
    "EXPECTED_05K_INDEX_SHA256",
    "EXPECTED_05K_MICRO_SHA256",
    "HinesAutoregressiveFailureReassessment",
    "HinesAutoregressiveFailureReassessmentConfig",
    "classify_autoregressive_failure",
    "verified_micro_rollout_artifact_root",
]
