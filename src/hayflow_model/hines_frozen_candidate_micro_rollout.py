"""Outcome-blind 2/4/8 ms rollout of the frozen 05j-n candidates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_data.hines_inputs import encode_realized_synaptic_drive
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_regenerative_confirmation import _verified_artifact_root
from .hines_regenerative_fresh_test import HinesRegenerativeFreshTestEvaluation
from .hines_trainable_topology_canary import TrainableTopologyResidualHead

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05JO_ARCHIVE_SHA256 = (
    "88310745dde29a17ba3073c01be127b5630c56e591e36b56cdd50e9280d1cc6d"
)
EXPECTED_05JO_INDEX_SHA256 = (
    "56143bea1bb66cd238d04cd30d3c32af0c0101b94616ee4cfcfea3e29860880d"
)
EXPECTED_05JO_FINAL_SHA256 = (
    "f6276b0448c0e24f434685e0d68baeb348fdea1e66420d66a27c21956506b9db"
)
EXPECTED_05JO_TRANSITION_SHA256 = (
    "667b7dd3a866de9d9f6fc9eaed7d61941673957e0c1676ff2f43644841aab47e"
)


@dataclass(frozen=True)
class HinesFrozenCandidateMicroRolloutConfig:
    seeds: Tuple[int, ...] = (17, 29, 43)
    horizons_ms: Tuple[int, ...] = (2, 4, 8)
    branch_step: int = 4
    pair_count: int = 32
    minimum_improvement_vs_best_baseline_fraction: float = 0.05
    minimum_branching_retention: float = 0.5
    maximum_branching_retention: float = 2.0
    maximum_segment_error_ratio_vs_h2: float = 1.0
    physical_voltage_min_mv: float = -150.0
    physical_voltage_max_mv: float = 100.0
    minimum_passing_seeds: int = 2

    def validate(self) -> None:
        if self.seeds != (17, 29, 43):
            raise ValueError("05k must evaluate all three frozen registered seeds")
        if self.horizons_ms != (2, 4, 8):
            raise ValueError("05k horizons are preregistered at 2/4/8 ms")
        if (self.branch_step, self.pair_count) != (4, 32):
            raise ValueError("05k branch boundary and pair count are preregistered")
        if not 0 < self.minimum_improvement_vs_best_baseline_fraction < 1:
            raise ValueError("05k improvement threshold is invalid")
        if not 0 < self.minimum_branching_retention < self.maximum_branching_retention:
            raise ValueError("05k branching interval is invalid")
        if self.maximum_segment_error_ratio_vs_h2 <= 0:
            raise ValueError("05k maximum-error ratio must be positive")
        if self.physical_voltage_min_mv >= self.physical_voltage_max_mv:
            raise ValueError("05k physical voltage bounds are reversed")
        if not 1 <= self.minimum_passing_seeds <= len(self.seeds):
            raise ValueError("05k robust seed gate is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesFrozenCandidateMicroRolloutConfig":
        payload = dict(values)
        for name in ("seeds", "horizons_ms"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


def verified_fresh_test_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """Verify every 05j-o member and return its immutable dataset root."""

    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="frozen_model_evaluation_config.json",
        archive_sha256=EXPECTED_05JO_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05JO_INDEX_SHA256,
        final_sha256=EXPECTED_05JO_FINAL_SHA256,
    )


def rollout_voltage_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    physical_min_mv: float,
    physical_max_mv: float,
) -> Dict[str, Any]:
    """Compute endpoint, trace and paired-future metrics for ordered pair arms."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("rollout arrays must have matching [episode,time,segment] shape")
    if prediction.shape[0] % 2:
        raise ValueError("rollout episodes must be ordered low/high pairs")
    error = prediction - target
    endpoint_error = error[:, -1, :]
    predicted_distance = np.sqrt(
        np.mean((prediction[0::2, -1, :] - prediction[1::2, -1, :]) ** 2, axis=1)
    )
    teacher_distance = np.sqrt(
        np.mean((target[0::2, -1, :] - target[1::2, -1, :]) ** 2, axis=1)
    )
    retention = predicted_distance / np.maximum(teacher_distance, 1e-12)
    finite = bool(np.all(np.isfinite(prediction)))
    below = prediction < float(physical_min_mv)
    above = prediction > float(physical_max_mv)
    return {
        "episode_count": int(prediction.shape[0]),
        "pair_count": int(prediction.shape[0] // 2),
        "horizon_ms": int(prediction.shape[1]),
        "endpoint_voltage_rmse_mv": float(np.sqrt(np.mean(endpoint_error**2))),
        "trace_voltage_rmse_mv": float(np.sqrt(np.mean(error**2))),
        "endpoint_mean_drift_mv": float(np.mean(endpoint_error)),
        "maximum_segment_error_mv": float(np.max(np.abs(endpoint_error))),
        "median_branching_retention": float(np.median(retention)),
        "minimum_branching_retention": float(np.min(retention)),
        "maximum_branching_retention": float(np.max(retention)),
        "minimum_teacher_pair_distance_mv": float(np.min(teacher_distance)),
        "nonfinite_value_count": int(np.size(prediction) - np.count_nonzero(np.isfinite(prediction))),
        "physical_voltage_violation_count": int(np.count_nonzero(below | above)),
        "minimum_predicted_voltage_mv": float(np.nanmin(prediction)),
        "maximum_predicted_voltage_mv": float(np.nanmax(prediction)),
        "numerically_finite": finite,
    }


class HinesFrozenCandidateMicroRollout(HinesRegenerativeFreshTestEvaluation):
    """Stress the frozen one-step family without tuning on rollout outcomes."""

    def __init__(
        self,
        *args: Any,
        micro_rollout_config: HinesFrozenCandidateMicroRolloutConfig,
        artifact_05jo_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        micro_rollout_config.validate()
        self.micro = micro_rollout_config
        self.artifact_05jo_source = Path(artifact_05jo_source).resolve()
        self.artifact_05jo_root = Path()
        self.artifact_05jo_report: Dict[str, Any] = {}
        self.artifact_05jo_contract: Dict[str, Any] = {}
        self._frontend_specs: List[Dict[str, Any]] = []

    def prepare_frozen_candidate_micro_rollout(self) -> Dict[str, Any]:
        # Verify the 05j-o decision before the parent opens the fresh-test store.
        root, report, contract = verified_fresh_test_artifact_root(
            self.artifact_05jo_source,
            self.output_dir.parent / ".05k_artifact_cache" / "05jo",
        )
        evaluation = report.get("frozen_model_evaluation", {})
        methodology = report.get("methodology", {})
        blockers = []
        if not report.get("valid"):
            blockers.append("05j-o artifact is invalid")
        if report.get("diagnosis") != "FRESH_TEST_CONFIRMS_REFIT_ONE_STEP_CANDIDATE":
            blockers.append("05j-o did not confirm the one-step candidate")
        if not report.get("candidate_model_authorized") or not report.get(
            "micro_rollout_authorized"
        ):
            blockers.append("05j-o did not authorize micro-rollout")
        if report.get("full_training_authorized"):
            blockers.append("05j-o unexpectedly authorized full training")
        if int(evaluation.get("passing_seed_count", -1)) < self.micro.minimum_passing_seeds:
            blockers.append("05j-o robust frozen-seed gate did not pass")
        if not evaluation.get("all_pairs_evaluated"):
            blockers.append("05j-o did not evaluate all frozen pairs")
        if any(
            methodology.get(name)
            for name in (
                "checkpoint_selection_performed",
                "retraining_performed",
                "fresh_test_used_for_model_selection",
                "rollout_performed",
            )
        ):
            blockers.append("05j-o methodology is incompatible with a clean rollout")
        teacher = report.get("teacher_fresh_test", {})
        if teacher.get("transition_store_sha256") != EXPECTED_05JO_TRANSITION_SHA256:
            blockers.append("05j-o fresh transition store SHA-256 mismatch")
        loaded_transition = self.fresh_dataset_root / "transition_dataset.h5"
        if (
            not loaded_transition.is_file()
            or sha256_file(loaded_transition) != EXPECTED_05JO_TRANSITION_SHA256
        ):
            blockers.append("loaded fresh dataset differs from the verified 05j-o shard")
        if blockers:
            raise RuntimeError(f"05k provenance blockers: {blockers}")
        base = self.prepare_fresh_test_evaluation()
        self.artifact_05jo_root = root
        self.artifact_05jo_report = report
        self.artifact_05jo_contract = contract
        payload = {
            "schema_version": "05k-frozen-candidate-micro-rollout-config-v1",
            "valid": True,
            "micro_rollout": asdict(self.micro),
            "artifact_05jo": contract,
            "checkpoint_selection_performed": False,
            "retraining_performed": False,
            "rollout_outcomes_used_for_tuning": False,
            "teacher_state_encoder_policy": "initial_boundary_only",
            "future_teacher_membrane_or_ion_states_injected": False,
            "authentic_synapse_frontend_state_used": True,
            "equal_weight_ensemble_role": "descriptive_only",
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "frozen_candidate_micro_rollout_config.json", payload)
        return {**base, **payload}

    @staticmethod
    def _hide_future_teacher_state(
        batch: Mapping[str, Any], recurrent: Mapping[str, Any] | None
    ) -> Dict[str, Any]:
        result = dict(batch)
        if recurrent is not None:
            for name in (
                "teacher_state_t",
                "voltage_t",
                "calcium_t",
                "synapse_state_t",
                "anchor_voltage_t",
            ):
                if name in result:
                    result[name] = torch.zeros_like(result[name])
        return result

    @staticmethod
    def _causal_surface(batch: Mapping[str, Any]) -> Any:
        return torch.cat(
            [
                batch["synaptic_features"],
                batch["synaptic_conductance_us"].unsqueeze(-1),
                batch["synaptic_source_na"].unsqueeze(-1),
                batch["somatic_current_na"].unsqueeze(-1),
            ],
            dim=-1,
        )

    def _synapse_frontend_specs(self) -> List[Dict[str, Any]]:
        if self._frontend_specs:
            return self._frontend_specs
        state_records: Dict[int, Dict[str, int]] = {}
        for position, record in enumerate(self.layout.core_records):
            if str(record.get("scope", "")) == "synapse":
                state_records.setdefault(int(record["owner_id"]), {})[
                    str(record.get("variable", record.get("name", "")))
                ] = position
        specs: List[Dict[str, Any]] = []
        for synapse in self.layout.synapses:
            synapse_id = int(synapse["id"])
            variables = state_records.get(synapse_id, {})
            for component in synapse.get("components", []):
                name = str(component["name"]).upper()
                suffix = "NMDA" if "NMDA" in name else "AMPA" if "AMPA" in name else ""
                a_name = f"A_{suffix}" if suffix and f"A_{suffix}" in variables else "A"
                b_name = f"B_{suffix}" if suffix and f"B_{suffix}" in variables else "B"
                if a_name not in variables or b_name not in variables:
                    continue
                increment_name = (
                    "nmda_state_increment"
                    if "NMDA" in name
                    else "inhibitory_state_increment"
                    if "GABA" in name
                    else "ampa_state_increment"
                )
                specs.append(
                    {
                        "synapse_id": synapse_id,
                        "a_index": int(variables[a_name]),
                        "b_index": int(variables[b_name]),
                        "tau_rise_ms": float(component["tau_rise_ms"]),
                        "tau_decay_ms": float(component["tau_decay_ms"]),
                        "increment_name": increment_name,
                    }
                )
        if not specs:
            raise RuntimeError("05k could not resolve authentic synaptic A/B state")
        self._frontend_specs = specs
        return specs

    def _authentic_synapse_frontend_state(self, logical_index: int) -> np.ndarray:
        """Expose only deployable authentic A/B state, never membrane state."""

        raw = self.fresh_store.read_state([int(logical_index)], "t")[0]
        isolated = np.zeros_like(raw, dtype=np.float64)
        coordinates = sorted(
            {
                int(spec[name])
                for spec in self._synapse_frontend_specs()
                for name in ("a_index", "b_index")
            }
        )
        isolated[np.asarray(coordinates, dtype=np.int64)] = raw[
            np.asarray(coordinates, dtype=np.int64)
        ]
        return isolated

    def _causal_frontend_batch(
        self,
        batch: Mapping[str, Any],
        logical_index: int,
        voltage_t: Any,
        device: Any,
    ) -> Dict[str, Any]:
        """Replace teacher-derived drive with the authentic causal front-end."""

        result = dict(batch)
        drive = encode_realized_synaptic_drive(
            self.fresh_store,
            [int(logical_index)],
            voltage_t.detach().cpu().numpy(),
            dt_ms=self.config.model.dt_ms,
            raw_state_t=self._authentic_synapse_frontend_state(logical_index)[None, :],
        )
        for name, values in drive.items():
            result[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
        return result

    def _validate_causal_frontend_contract(
        self, windows: Sequence[np.ndarray]
    ) -> Dict[str, Any]:
        state_indices = sorted(
            {
                int(spec[name])
                for spec in self._synapse_frontend_specs()
                for name in ("a_index", "b_index")
            }
        )
        shared_boundary_errors = []
        for low, high in zip(windows[0::2], windows[1::2]):
            low_state = self._authentic_synapse_frontend_state(int(low[0]))
            high_state = self._authentic_synapse_frontend_state(int(high[0]))
            shared_boundary_errors.append(
                float(np.max(np.abs(low_state[state_indices] - high_state[state_indices])))
            )
        maximum = max(shared_boundary_errors, default=math.inf)
        report = {
            "schema_version": "05k-causal-synapse-frontend-validation-v1",
            "valid": bool(maximum <= 1e-12),
            "pair_count": len(windows) // 2,
            "state_coordinate_count": len(state_indices),
            "maximum_shared_boundary_state_error": maximum,
            "tolerance": 1e-12,
            "state_scope": "authentic_synapse_A_B_only",
            "state_provider": "preserved_authentic_synapse_frontend",
            "event_input": "U_realized",
            "nmda_block_voltage_source": "autoregressive_model_voltage",
            "membrane_or_ion_state_exposed": False,
        }
        _write_json(self.output_dir / "causal_synapse_frontend_validation.json", report)
        if not report["valid"]:
            raise RuntimeError(f"05k paired synapse front-end boundary mismatch: {maximum}")
        return report

    def _pair_windows(self, horizon: int) -> List[np.ndarray]:
        if self.fresh_store is None:
            raise RuntimeError("prepare_frozen_candidate_micro_rollout() must run first")
        transition_lookup = {
            int(value): index
            for index, value in enumerate(self.fresh_store.metadata["transition_id"].tolist())
        }
        windows: List[np.ndarray] = []
        for pair in self.fresh_pair_rows:
            for key in ("low_transition_id", "high_transition_id"):
                start = transition_lookup[int(pair[key])]
                trajectory = str(self.fresh_store.metadata["trajectory_id"][start])
                episode = self.fresh_store.trajectory_indices[trajectory]
                position = int(np.flatnonzero(episode == start)[0])
                window = np.asarray(episode[position : position + int(horizon)], dtype=np.int64)
                if len(window) != int(horizon):
                    raise RuntimeError(f"05k incomplete {horizon} ms window for {trajectory}")
                steps = self.fresh_store.metadata["step_index"][window]
                if not np.array_equal(
                    steps,
                    np.arange(self.micro.branch_step, self.micro.branch_step + horizon),
                ):
                    raise RuntimeError(f"05k non-contiguous branch window for {trajectory}")
                windows.append(window)
        if len(windows) != 2 * self.micro.pair_count:
            raise RuntimeError("05k did not construct exactly 64 rollout arms")
        return windows

    def _h2_rollout(self, model: Any, window: np.ndarray, device: Any) -> np.ndarray:
        recurrent = None
        trace = []
        with torch.no_grad():
            for logical_index in window:
                raw = self._batch([int(logical_index)], include_targets=False)
                original = self._torch_batch(raw, device)
                voltage_t = (
                    original["voltage_t"] if recurrent is None else recurrent["voltage"]
                )
                batch = self._hide_future_teacher_state(original, recurrent)
                batch = self._causal_frontend_batch(
                    batch, int(logical_index), voltage_t, device
                )
                output = model(
                    batch,
                    recurrent=recurrent,
                    ablation="H2",
                    decode_teacher=False,
                    boundary_mode="no_event_jump",
                )
                recurrent = {
                    key: output[key]
                    for key in ("voltage", "local", "global", "calcium", "synapse")
                }
                trace.append(output["voltage"][0].detach().cpu().numpy())
        return np.asarray(trace, dtype=np.float32)

    def _candidate_rollout(
        self,
        h2: Any,
        decoder: Any,
        transform: Mapping[str, Any],
        window: np.ndarray,
        device: Any,
    ) -> np.ndarray:
        recurrent = None
        trace = []
        with torch.no_grad():
            for logical_index in window:
                raw = self._batch([int(logical_index)], include_targets=False)
                original = self._torch_batch(raw, device)
                state_voltage = (
                    original["voltage_t"] if recurrent is None else recurrent["voltage"]
                )
                batch = self._hide_future_teacher_state(original, recurrent)
                batch = self._causal_frontend_batch(
                    batch,
                    int(logical_index),
                    state_voltage,
                    device,
                )
                output = h2(
                    batch,
                    recurrent=recurrent,
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
                corrected_voltage = output["voltage"] + decoder(features)
                recurrent = {
                    "voltage": corrected_voltage,
                    "local": output["local"],
                    "global": output["global"],
                    "calcium": output["calcium"],
                    "synapse": output["synapse"],
                }
                trace.append(corrected_voltage[0].detach().cpu().numpy())
        return np.asarray(trace, dtype=np.float32)

    def _load_frozen_decoder(self, seed: int, feature_width: int, device: Any) -> Any:
        registered = {
            int(row["seed"]): row
            for row in self.artifact_05jn_report["decoder_refit"]["runs"]
        }[int(seed)]
        checkpoint = torch.load(
            self.artifact_05jn_root / str(registered["checkpoint"]),
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("family") != "direct_tree_refit" or int(
            checkpoint.get("seed", -1)
        ) != int(seed):
            raise RuntimeError(f"05k checkpoint identity mismatch for seed {seed}")
        model = TrainableTopologyResidualHead(
            feature_width,
            self.layout.segment_count,
            self.topology.hidden_width,
            self.topology.segment_embedding_dim,
            self.topology.target_residual_limit_mv,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def evaluate_frozen_micro_rollout(self) -> Dict[str, Any]:
        require_torch()
        if self.fresh_store is None or not self.refit_transform:
            raise RuntimeError("05k requires reconstructed frozen 05j-n representation")
        registered_transform, transform_error = self._load_registered_transform()
        if transform_error > self.fresh_config.transform_reproduction_atol:
            raise RuntimeError(f"05k registered transform mismatch: {transform_error}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2, _ = self._load_h2_checkpoint(device)
        h2.eval()
        original_store = self.store
        self.store = self.fresh_store
        try:
            max_horizon = max(self.micro.horizons_ms)
            windows = self._pair_windows(max_horizon)
            frontend_validation = self._validate_causal_frontend_contract(windows)
            targets = np.stack(
                [
                    self.fresh_store.read_state(window, "t_plus_1")[
                        :, : self.layout.segment_count
                    ]
                    for window in windows
                ]
            ).astype(np.float32)
            initial = np.stack(
                [
                    self.fresh_store.read_state([int(window[0])], "t")[
                        0, : self.layout.segment_count
                    ]
                    for window in windows
                ]
            ).astype(np.float32)
            persistence = np.repeat(initial[:, None, :], max_horizon, axis=1)
            progress = Progress("05k autoregressive H2", len(windows))
            h2_traces = []
            for position, window in enumerate(windows, start=1):
                h2_traces.append(self._h2_rollout(h2, window, device))
                progress.update(position)
            h2_traces = np.asarray(h2_traces, dtype=np.float32)
            candidate_traces: Dict[int, np.ndarray] = {}
            feature_width = int(self.refit_designs["fit"].shape[-1])
            for seed in self.micro.seeds:
                decoder = self._load_frozen_decoder(seed, feature_width, device)
                progress = Progress(f"05k frozen seed{seed}", len(windows))
                values = []
                for position, window in enumerate(windows, start=1):
                    values.append(
                        self._candidate_rollout(
                            h2, decoder, registered_transform, window, device
                        )
                    )
                    progress.update(position)
                candidate_traces[int(seed)] = np.asarray(values, dtype=np.float32)
                del decoder
        finally:
            self.store = original_store
            del h2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        baseline_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {
            "h2": {},
            "persistence": {},
        }
        run_metrics: Dict[int, Dict[str, Dict[str, Any]]] = {
            int(seed): {} for seed in self.micro.seeds
        }
        gate_rows: List[Dict[str, Any]] = []
        pair_rows: List[Dict[str, Any]] = []
        for horizon in self.micro.horizons_ms:
            key = str(horizon)
            target = targets[:, :horizon, :]
            for name, values in (("h2", h2_traces), ("persistence", persistence)):
                baseline_metrics[name][key] = rollout_voltage_metrics(
                    values[:, :horizon, :],
                    target,
                    physical_min_mv=self.micro.physical_voltage_min_mv,
                    physical_max_mv=self.micro.physical_voltage_max_mv,
                )
            best_baseline = min(
                baseline_metrics["h2"][key]["endpoint_voltage_rmse_mv"],
                baseline_metrics["persistence"][key]["endpoint_voltage_rmse_mv"],
            )
            for seed in self.micro.seeds:
                values = candidate_traces[int(seed)][:, :horizon, :]
                metrics = rollout_voltage_metrics(
                    values,
                    target,
                    physical_min_mv=self.micro.physical_voltage_min_mv,
                    physical_max_mv=self.micro.physical_voltage_max_mv,
                )
                improvement = 1.0 - metrics["endpoint_voltage_rmse_mv"] / max(
                    best_baseline, 1e-12
                )
                conditions = {
                    "minimum_improvement_passed": improvement
                    >= self.micro.minimum_improvement_vs_best_baseline_fraction,
                    "branching_retention_passed": self.micro.minimum_branching_retention
                    <= metrics["median_branching_retention"]
                    <= self.micro.maximum_branching_retention,
                    "maximum_error_passed": metrics["maximum_segment_error_mv"]
                    <= self.micro.maximum_segment_error_ratio_vs_h2
                    * baseline_metrics["h2"][key]["maximum_segment_error_mv"],
                    "physical_range_passed": metrics[
                        "physical_voltage_violation_count"
                    ]
                    == 0,
                    "numerically_finite": metrics["numerically_finite"],
                }
                passed = all(conditions.values())
                run_metrics[int(seed)][key] = {
                    **metrics,
                    "improvement_vs_best_baseline_fraction": improvement,
                    "conditions": conditions,
                    "horizon_passed": passed,
                }
                gate_rows.append(
                    {
                        "seed": int(seed),
                        "horizon_ms": int(horizon),
                        "endpoint_voltage_rmse_mv": metrics[
                            "endpoint_voltage_rmse_mv"
                        ],
                        "best_baseline_rmse_mv": best_baseline,
                        "improvement_vs_best_baseline_fraction": improvement,
                        "maximum_segment_error_mv": metrics[
                            "maximum_segment_error_mv"
                        ],
                        "h2_maximum_segment_error_mv": baseline_metrics["h2"][key][
                            "maximum_segment_error_mv"
                        ],
                        "median_branching_retention": metrics[
                            "median_branching_retention"
                        ],
                        "horizon_passed": passed,
                    }
                )
                endpoint = values[:, -1, :]
                teacher_endpoint = target[:, -1, :]
                for pair in range(self.micro.pair_count):
                    low, high = 2 * pair, 2 * pair + 1
                    pair_rows.append(
                        {
                            "seed": int(seed),
                            "horizon_ms": int(horizon),
                            "pair_position": pair,
                            "voltage_rmse_mv": float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            endpoint[[low, high]]
                                            - teacher_endpoint[[low, high]]
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            "teacher_distance_mv": float(
                                np.sqrt(
                                    np.mean(
                                        (teacher_endpoint[low] - teacher_endpoint[high])
                                        ** 2
                                    )
                                )
                            ),
                            "predicted_distance_mv": float(
                                np.sqrt(
                                    np.mean((endpoint[low] - endpoint[high]) ** 2)
                                )
                            ),
                        }
                    )

        ensemble = np.mean(
            np.stack([candidate_traces[int(seed)] for seed in self.micro.seeds]), axis=0
        )
        ensemble_metrics = {
            str(horizon): rollout_voltage_metrics(
                ensemble[:, :horizon, :],
                targets[:, :horizon, :],
                physical_min_mv=self.micro.physical_voltage_min_mv,
                physical_max_mv=self.micro.physical_voltage_max_mv,
            )
            for horizon in self.micro.horizons_ms
        }
        seed_pass = {
            str(seed): all(
                run_metrics[int(seed)][str(horizon)]["horizon_passed"]
                for horizon in self.micro.horizons_ms
            )
            for seed in self.micro.seeds
        }
        passing = sum(seed_pass.values())
        valid = bool(
            all(
                row["numerically_finite"]
                for seed in self.micro.seeds
                for row in run_metrics[int(seed)].values()
            )
            and targets.shape
            == (2 * self.micro.pair_count, max(self.micro.horizons_ms), self.layout.segment_count)
        )
        np.savez_compressed(
            self.output_dir / "micro_rollout_predictions.npz",
            target=targets,
            h2=h2_traces,
            persistence=persistence,
            ensemble=ensemble,
            **{f"seed_{seed}": candidate_traces[int(seed)] for seed in self.micro.seeds},
        )
        write_parquet(self.output_dir / "micro_rollout_gate_summary.parquet", gate_rows)
        write_parquet(self.output_dir / "micro_rollout_pair_metrics.parquet", pair_rows)
        report = {
            "schema_version": "05k-frozen-candidate-micro-rollout-v1",
            "valid": valid,
            "device": str(device),
            "pair_count": self.micro.pair_count,
            "episode_count": 2 * self.micro.pair_count,
            "horizons_ms": list(self.micro.horizons_ms),
            "baselines": baseline_metrics,
            "runs": {str(seed): run_metrics[int(seed)] for seed in self.micro.seeds},
            "equal_weight_ensemble_descriptive": ensemble_metrics,
            "seed_passed_all_horizons": seed_pass,
            "passing_seed_count": passing,
            "minimum_passing_seed_count": self.micro.minimum_passing_seeds,
            "robust_micro_rollout_gate_passed": passing
            >= self.micro.minimum_passing_seeds,
            "maximum_transform_reproduction_error": transform_error,
            "causal_synapse_frontend": frontend_validation,
            "teacher_state_encoder_used_at_initial_boundary": True,
            "future_teacher_membrane_or_ion_states_injected": False,
            "authentic_synapse_frontend_state_used": True,
            "input_view": "U_realized_plus_authentic_causal_synapse_frontend",
            "checkpoint_selection_performed": False,
            "retraining_performed": False,
            "rollout_outcomes_used_for_tuning": False,
            "ensemble_used_for_gate": False,
        }
        _write_json(self.output_dir / "micro_rollout_report.json", report)
        if not valid:
            raise RuntimeError("05k produced an invalid or non-finite micro-rollout")
        return report

    def finalize_micro_rollout(
        self, rollout_report: Mapping[str, Any]
    ) -> Dict[str, Any]:
        passed = bool(
            rollout_report.get("valid")
            and rollout_report.get("robust_micro_rollout_gate_passed")
        )
        report = {
            "schema_version": "05k-final-report-v1",
            "valid": bool(rollout_report.get("valid")),
            "decision": "FROZEN_CANDIDATE_AUTOREGRESSIVE_MICRO_ROLLOUT",
            "diagnosis": (
                "FROZEN_CANDIDATE_PASSES_2_4_8MS_MICRO_ROLLOUT"
                if passed
                else "FROZEN_CANDIDATE_FAILS_AUTOREGRESSIVE_MICRO_ROLLOUT"
            ),
            "code_revision": self.code_revision,
            "artifact_05jo": self.artifact_05jo_contract,
            "micro_rollout": dict(rollout_report),
            "candidate_retained": passed,
            "limited_rollout_aware_training_canary_authorized": passed,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "methodology": {
                "all_32_pairs_and_both_arms_evaluated": True,
                "all_three_frozen_seeds_evaluated": True,
                "checkpoint_selection_performed": False,
                "retraining_performed": False,
                "teacher_state_encoder_used_at_initial_boundary": True,
                "future_teacher_membrane_or_ion_states_injected": False,
                "authentic_synapse_frontend_state_used": True,
                "equal_weight_ensemble_role": "descriptive_only",
            },
            "next_step": (
                "05l_limited_rollout_aware_training_canary"
                if passed
                else "05k_b_autoregressive_failure_reassessment"
            ),
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append(
                    {
                        "path": path.relative_to(self.output_dir).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        _write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "05k-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05JO_ARCHIVE_SHA256",
    "EXPECTED_05JO_INDEX_SHA256",
    "EXPECTED_05JO_FINAL_SHA256",
    "EXPECTED_05JO_TRANSITION_SHA256",
    "HinesFrozenCandidateMicroRolloutConfig",
    "HinesFrozenCandidateMicroRollout",
    "verified_fresh_test_artifact_root",
    "rollout_voltage_metrics",
]
