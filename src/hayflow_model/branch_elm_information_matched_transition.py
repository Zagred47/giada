"""Information-matched one-step voltage comparison for the 06b-c sidecar.

Both arms consume the exact same numeric feature tensor and predict the
authentic NEURON one-millisecond voltage transition.  This module deliberately
does not perform an autoregressive rollout and never injects ``V_(t+1)`` into
either input arm.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .causal_voltage_state_coupling_forensic import (
    CausalVoltageBridge,
    CausalVoltageStateCouplingConfig,
    CausalVoltageStateCouplingForensic,
)


MATCHED_MODEL_NAMES = ("branch_elm_core", "hayflow_voltage_bridge")


@dataclass(frozen=True)
class InformationMatchedTransitionConfig:
    seeds: Tuple[int, ...] = (61017, 61029, 61043)
    training_steps: int = 800
    evaluation_interval: int = 100
    progress_interval: int = 50
    batch_transition_count: int = 8
    segments_per_transition: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    gradient_clip_norm: float = 1.0
    voltage_scale_mv: float = 20.0
    delta_limit_mv: float = 100.0
    active_delta_threshold_mv: float = 5.0
    active_weight: float = 4.0
    evaluation_transition_chunk: int = 16
    expected_branch_elm_parameter_count: int = 8002
    expected_hayflow_bridge_parameter_count: int = 8985
    expected_state_updater_parameter_count: int = 7212

    def validate(self) -> None:
        if tuple(self.seeds) != (61017, 61029, 61043):
            raise ValueError("information-matched comparison requires registered seeds")
        positive = (
            self.training_steps,
            self.evaluation_interval,
            self.progress_interval,
            self.batch_transition_count,
            self.segments_per_transition,
            self.evaluation_transition_chunk,
            self.expected_branch_elm_parameter_count,
            self.expected_hayflow_bridge_parameter_count,
            self.expected_state_updater_parameter_count,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("information-matched integer configuration is invalid")
        if not 0 < self.learning_rate < 1 or self.weight_decay < 0:
            raise ValueError("information-matched optimizer configuration is invalid")
        if self.voltage_scale_mv <= 0 or self.delta_limit_mv <= 0:
            raise ValueError("information-matched voltage scaling is invalid")
        if self.active_delta_threshold_mv < 0 or self.active_weight <= 0:
            raise ValueError("information-matched active objective is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "InformationMatchedTransitionConfig":
        payload = dict(values)
        if "seeds" in payload:
            payload["seeds"] = tuple(map(int, payload["seeds"]))
        result = cls(**payload)
        result.validate()
        return result


if atomic.nn is not None:

    class InformationMatchedBranchELM(atomic.nn.Module):
        """Published 8,002-parameter ELM core on the shared local tensor."""

        def __init__(
            self, input_width: int, normalized_delta_limit: float
        ) -> None:
            super().__init__()
            from ..expressive_leaky_memory_neuron import ELM

            self.normalized_delta_limit = float(normalized_delta_limit)
            self.core = ELM(
                num_input=int(input_width),
                num_output=2,
                num_memory=20,
                memory_tau_min=1.0,
                memory_tau_max=150.0,
                learn_memory_tau=False,
                num_branch=45,
                num_synapse_per_branch=100,
                input_to_synapse_routing="neuronio_routing",
            )

        def forward(self, features: Any) -> Any:
            # T=1 makes this an authentic conditional transition comparison,
            # not an autoregressive or teacher-forced sequence rollout.
            raw = self.core(features.unsqueeze(1))[:, 0, 1]
            return self.normalized_delta_limit * atomic.torch.tanh(raw)


    class InformationMatchedBridgeAdapter(atomic.nn.Module):
        """Expose the current 06b bridge through the same numeric input tensor."""

        def __init__(self, bridge: Any, slices: Mapping[str, slice]) -> None:
            super().__init__()
            self.bridge = bridge
            self.slices = dict(slices)

        def forward(self, features: Any) -> Any:
            region = features[:, self.slices["region_one_hot"]].argmax(dim=-1)
            return self.bridge(
                features[:, self.slices["axial_voltage"]],
                features[:, self.slices["mechanism_state"]],
                features[:, self.slices["mechanism_presence"]],
                features[:, self.slices["causal_context"]],
                features[:, self.slices["segment_static"]],
                region.long(),
            )

else:  # pragma: no cover

    class InformationMatchedBranchELM:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("information-matched comparison requires PyTorch")

    class InformationMatchedBridgeAdapter:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("information-matched comparison requires PyTorch")


class InformationMatchedVoltageTransitionBenchmark(
    CausalVoltageStateCouplingForensic
):
    """Paired, same-input, same-target voltage-transition benchmark."""

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        coupling_config: CausalVoltageStateCouplingConfig,
        comparison_config: InformationMatchedTransitionConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        *,
        code_revision: str,
    ) -> None:
        comparison_config.validate()
        super().__init__(
            bundle,
            output_dir,
            coupling_config,
            artifact_05t_source,
            artifact_06b_source,
            code_revision=code_revision,
        )
        self.comparison = comparison_config
        self.feature_slices: Dict[str, slice] = {}
        self.feature_width = 0
        self.selected_models: Dict[Tuple[str, int], Any] = {}

    @staticmethod
    def _parameter_count(model: Any) -> int:
        # The published 8,002-parameter Branch-ELM count includes only
        # trainable weights. ELM intentionally registers routing indices,
        # validity masks and fixed time constants as non-trainable Parameters.
        return int(
            sum(
                value.numel()
                for value in model.parameters()
                if value.requires_grad
            )
        )

    def _define_common_tensor(self) -> None:
        widths = {
            "axial_voltage": 3,
            "mechanism_state": len(self.coordinate_groups),
            "mechanism_presence": len(self.coordinate_groups),
            "causal_context": len(atomic.CAUSAL_DRIVE_FEATURES)
            + len(self.ion_feature_names),
            "segment_static": int(self.layout.segment_static.shape[1]),
            "region_one_hot": len(self.layout.region_names),
        }
        start = 0
        for name, width in widths.items():
            self.feature_slices[name] = slice(start, start + int(width))
            start += int(width)
        # The published NeuronIO routing interlocks two equal input halves.
        # A zero-information pad keeps that routing well defined if the raw
        # common feature width is odd.
        if start % 2:
            self.feature_slices["zero_padding"] = slice(start, start + 1)
            start += 1
        self.feature_width = start

    def _new_matched_models(self, device: Any) -> Dict[str, Any]:
        limit = self.comparison.delta_limit_mv / self.comparison.voltage_scale_mv
        bridge = InformationMatchedBridgeAdapter(
            self._new_bridge(device), self.feature_slices
        ).to(device)
        elm = InformationMatchedBranchELM(self.feature_width, limit).to(device)
        counts = {
            "branch_elm_core": self._parameter_count(elm),
            "hayflow_voltage_bridge": self._parameter_count(bridge),
        }
        expected = {
            "branch_elm_core": self.comparison.expected_branch_elm_parameter_count,
            "hayflow_voltage_bridge": self.comparison.expected_hayflow_bridge_parameter_count,
        }
        if counts != expected:
            raise RuntimeError(
                f"information-matched parameter contract mismatch: {counts} != {expected}"
            )
        return {"branch_elm_core": elm, "hayflow_voltage_bridge": bridge}

    def _common_numpy_batch(
        self, role: str, transition_rows: np.ndarray, segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = self.materialized[role]
        rows = np.repeat(transition_rows, segments.shape[1])
        flat_segments = segments.reshape(-1)
        region_ids = self.layout.segment_region_ids[flat_segments]
        region = np.eye(len(self.layout.region_names), dtype=np.float32)[region_ids]
        blocks: Tuple[np.ndarray, ...] = (
            self._axial_voltage_features(values["voltage_t"], rows, flat_segments),
            self.segment_state[role][rows, flat_segments],
            self.semantic_presence[flat_segments],
            values["context"][rows, flat_segments],
            self.layout.segment_static[flat_segments].astype(np.float32),
            region,
        )
        if "zero_padding" in self.feature_slices:
            blocks = blocks + (
                np.zeros((len(rows), 1), dtype=np.float32),
            )
        features = np.concatenate(blocks, axis=-1).astype(np.float32)
        target = (
            values["voltage_t1"][rows, flat_segments]
            - values["voltage_t"][rows, flat_segments]
        ) / self.comparison.voltage_scale_mv
        if features.shape[1] != self.feature_width:
            raise RuntimeError("shared voltage feature width changed")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise RuntimeError("shared voltage transition tensor is non-finite")
        return features, target.astype(np.float32)

    def prepare_information_matched_benchmark(self) -> Dict[str, Any]:
        base = self.prepare_coupling_forensic()
        self._define_common_tensor()
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        models = self._new_matched_models(device)
        state_count = self._parameter_count(
            self.frozen_state_models[("causal_start_voltage", self.comparison.seeds[0])]
        )
        if state_count != self.comparison.expected_state_updater_parameter_count:
            raise RuntimeError("current 06b STATE-updater parameter count changed")
        probe_rows = np.asarray([0], dtype=np.int64)
        probe_segments = np.asarray([[0, 1]], dtype=np.int64)
        probe, _ = self._common_numpy_batch("fit", probe_rows, probe_segments)
        tensor = atomic.torch.as_tensor(probe, dtype=atomic.torch.float32, device=device)
        with atomic.torch.no_grad():
            probe_shapes = {
                name: list(model(tensor).shape) for name, model in models.items()
            }
        report = {
            **base,
            "schema_version": "06b-c-information-matched-transition-contract-v1",
            "valid": all(shape == [2] for shape in probe_shapes.values()),
            "sidecar_only": True,
            "comparison_kind": "teacher_boundary_one_step_authentic_voltage_transition",
            "autoregressive_rollout_performed": False,
            "teacher_endpoint_used_as_input": False,
            "common_numeric_input_tensor": True,
            "common_input_fields": list(self.feature_slices),
            "common_input_width": self.feature_width,
            "common_input_slices": {
                name: [value.start, value.stop]
                for name, value in self.feature_slices.items()
            },
            "input_view": "U_realized",
            "target": "raw_NEURON_V_t_plus_1_minus_V_t",
            "target_clipping": None,
            "models": {
                "branch_elm_core": {
                    "total_parameter_count": self._parameter_count(models["branch_elm_core"]),
                    "voltage_path_parameter_count": self._parameter_count(models["branch_elm_core"]) - 21,
                    "published_core_hyperparameters_retained": True,
                    "original_event_routing_retained": False,
                    "reason": "the input adapter is replaced so both arms receive the same causal state tensor",
                },
                "hayflow_voltage_bridge": {
                    "voltage_path_parameter_count": self._parameter_count(models["hayflow_voltage_bridge"]),
                    "state_updater_parameter_count": state_count,
                    "complete_compact_transition_system_parameter_count": state_count
                    + self._parameter_count(models["hayflow_voltage_bridge"]),
                    "state_updater_affects_voltage_metric": False,
                },
            },
            "same_training_pairs": True,
            "same_sample_order": True,
            "same_optimizer_family_and_hyperparameters": True,
            "same_loss": True,
            "same_checkpoint_selection_metric": True,
            "state_and_outcome_splits_read": ["train"],
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "configuration": asdict(self.comparison),
        }
        atomic._write_json(self.output_dir / "information_matched_contract.json", report)
        if not report["valid"]:
            raise RuntimeError("information-matched transition preflight failed")
        return report

    def _predict(self, model: Any, role: str, device: Any) -> np.ndarray:
        values = self.materialized[role]
        count = len(values["indices"])
        output = np.empty_like(values["voltage_t"], dtype=np.float32)
        model.eval()
        with atomic.torch.no_grad():
            for start in range(0, count, self.comparison.evaluation_transition_chunk):
                stop = min(count, start + self.comparison.evaluation_transition_chunk)
                transition_rows = np.arange(start, stop, dtype=np.int64)
                segments = np.broadcast_to(
                    np.arange(self.layout.segment_count, dtype=np.int64),
                    (len(transition_rows), self.layout.segment_count),
                )
                features, _ = self._common_numpy_batch(role, transition_rows, segments)
                tensor = atomic.torch.as_tensor(
                    features, dtype=atomic.torch.float32, device=device
                )
                prediction = model(tensor).reshape(
                    len(transition_rows), self.layout.segment_count
                )
                output[start:stop] = (
                    prediction.cpu().numpy() * self.comparison.voltage_scale_mv
                )
        return output

    def _metrics(self, model: Any, role: str, device: Any) -> Dict[str, Any]:
        values = self.materialized[role]
        prediction = self._predict(model, role, device)
        target = values["voltage_t1"] - values["voltage_t"]
        error = prediction - target
        active = np.abs(target) >= self.comparison.active_delta_threshold_mv

        def rmse(values_: np.ndarray) -> float:
            values_64 = np.asarray(values_, dtype=np.float64)
            return float(np.sqrt(np.mean(values_64 * values_64)))

        global_rmse = rmse(error)
        persistence = rmse(target)
        soma_rmse = rmse(error[:, 0])
        soma_persistence = rmse(target[:, 0])
        active_rmse = rmse(error[active]) if np.any(active) else 0.0
        return {
            "raw_endpoint_voltage_rmse_mv": global_rmse,
            "raw_soma_endpoint_voltage_rmse_mv": soma_rmse,
            "persistence_voltage_rmse_mv": persistence,
            "persistence_soma_voltage_rmse_mv": soma_persistence,
            "improvement_vs_persistence_fraction": 1.0
            - global_rmse / max(persistence, 1e-12),
            "soma_improvement_vs_persistence_fraction": 1.0
            - soma_rmse / max(soma_persistence, 1e-12),
            "active_voltage_rmse_mv": active_rmse,
            "active_example_count": int(np.sum(active)),
            "transition_count": int(len(values["indices"])),
            "segment_count": int(self.layout.segment_count),
            "example_count": int(target.size),
            "target_is_authentic_teacher_transition": True,
            "target_clipping": None,
            "nonfinite_prediction_count": int(np.sum(~np.isfinite(prediction))),
        }

    def _train_seed(self, seed: int, device: Any) -> Dict[str, Any]:
        atomic.torch.manual_seed(seed)
        if atomic.torch.cuda.is_available():
            atomic.torch.cuda.manual_seed_all(seed)
        models = self._new_matched_models(device)
        optimizers = {
            name: atomic.torch.optim.AdamW(
                model.parameters(),
                lr=self.comparison.learning_rate,
                weight_decay=self.comparison.weight_decay,
            )
            for name, model in models.items()
        }
        best_score = {name: math.inf for name in models}
        best_state: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name in models
        }
        curves: Dict[str, List[Dict[str, Any]]] = {name: [] for name in models}
        rng = np.random.default_rng(seed)
        fit_count = len(self.materialized["fit"]["indices"])
        progress = atomic._CompactProgress(
            f"06b-c information matched seed={seed}",
            self.comparison.training_steps,
            self.comparison.progress_interval,
        )
        last_losses = {name: math.nan for name in models}
        for step in range(1, self.comparison.training_steps + 1):
            rows = rng.integers(
                0, fit_count, size=self.comparison.batch_transition_count
            )
            segments = rng.integers(
                0,
                self.layout.segment_count,
                size=(
                    self.comparison.batch_transition_count,
                    self.comparison.segments_per_transition,
                ),
            )
            features_np, target_np = self._common_numpy_batch("fit", rows, segments)
            features = atomic.torch.as_tensor(
                features_np, dtype=atomic.torch.float32, device=device
            )
            target = atomic.torch.as_tensor(
                target_np, dtype=atomic.torch.float32, device=device
            )
            threshold = (
                self.comparison.active_delta_threshold_mv
                / self.comparison.voltage_scale_mv
            )
            weight = 1.0 + self.comparison.active_weight * (
                target.abs() >= threshold
            ).float()
            for name in MATCHED_MODEL_NAMES:
                model = models[name]
                model.train()
                prediction = model(features)
                loss = atomic.torch.mean(
                    weight
                    * atomic.torch_functional.smooth_l1_loss(
                        prediction, target, reduction="none"
                    )
                )
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                atomic.torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.comparison.gradient_clip_norm
                )
                optimizers[name].step()
                last_losses[name] = float(loss.detach().cpu())
            if (
                step == 1
                or step % self.comparison.evaluation_interval == 0
                or step == self.comparison.training_steps
            ):
                for name in MATCHED_MODEL_NAMES:
                    metrics = self._metrics(models[name], "calibration", device)
                    score = float(metrics["raw_endpoint_voltage_rmse_mv"])
                    curves[name].append(
                        {
                            "step": step,
                            "train_loss": last_losses[name],
                            "calibration_raw_endpoint_voltage_rmse_mv": score,
                            "calibration_raw_soma_endpoint_voltage_rmse_mv": metrics[
                                "raw_soma_endpoint_voltage_rmse_mv"
                            ],
                        }
                    )
                    if score < best_score[name]:
                        best_score[name] = score
                        best_state[name] = copy.deepcopy(
                            {
                                key: value.detach().cpu()
                                for key, value in models[name].state_dict().items()
                            }
                        )
            progress.update(
                step,
                " ".join(
                    f"{name}={best_score[name]:.4g}mV"
                    for name in MATCHED_MODEL_NAMES
                ),
            )
        runs = {}
        for name in MATCHED_MODEL_NAMES:
            if best_state[name] is None:
                raise RuntimeError(f"{name} created no selected checkpoint")
            selected = self._new_matched_models(device)[name]
            selected.load_state_dict(best_state[name])
            selected.eval()
            self.selected_models[(name, seed)] = selected
            checkpoint = self.output_dir / f"matched_{name}_seed{seed}.pt"
            atomic.torch.save(
                {
                    "state_dict": best_state[name],
                    "seed": seed,
                    "model": name,
                    "configuration": asdict(self.comparison),
                },
                checkpoint,
            )
            runs[name] = {
                "parameter_count": self._parameter_count(selected),
                "best_calibration_raw_endpoint_voltage_rmse_mv": best_score[name],
                "development": self._metrics(selected, "development", device),
                "learning_curve": curves[name],
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": atomic._sha256_file(checkpoint),
            }
        return {"seed": seed, "models": runs}

    def run_information_matched_benchmark(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        runs = {}
        for seed in self.comparison.seeds:
            run = self._train_seed(int(seed), device)
            runs[str(seed)] = run
            atomic._write_json(
                self.output_dir / f"information_matched_seed{seed}.json", run
            )
        summary: Dict[str, Any] = {}
        for name in MATCHED_MODEL_NAMES:
            global_values = [
                float(runs[str(seed)]["models"][name]["development"]["raw_endpoint_voltage_rmse_mv"])
                for seed in self.comparison.seeds
            ]
            soma_values = [
                float(runs[str(seed)]["models"][name]["development"]["raw_soma_endpoint_voltage_rmse_mv"])
                for seed in self.comparison.seeds
            ]
            summary[name] = {
                "median_development_raw_endpoint_voltage_rmse_mv": float(
                    np.median(global_values)
                ),
                "median_development_raw_soma_endpoint_voltage_rmse_mv": float(
                    np.median(soma_values)
                ),
                "per_seed_global_rmse_mv": dict(
                    zip(map(str, self.comparison.seeds), global_values)
                ),
                "per_seed_soma_rmse_mv": dict(
                    zip(map(str, self.comparison.seeds), soma_values)
                ),
            }
        paired = {}
        for seed in self.comparison.seeds:
            elm = runs[str(seed)]["models"]["branch_elm_core"]["development"]
            hayflow = runs[str(seed)]["models"]["hayflow_voltage_bridge"]["development"]
            paired[str(seed)] = {
                "hayflow_global_error_reduction_vs_elm_fraction": 1.0
                - float(hayflow["raw_endpoint_voltage_rmse_mv"])
                / max(float(elm["raw_endpoint_voltage_rmse_mv"]), 1e-12),
                "hayflow_soma_error_reduction_vs_elm_fraction": 1.0
                - float(hayflow["raw_soma_endpoint_voltage_rmse_mv"])
                / max(float(elm["raw_soma_endpoint_voltage_rmse_mv"]), 1e-12),
            }
        report = {
            "schema_version": "06b-c-information-matched-transition-results-v1",
            "valid": True,
            "device": str(device),
            "runs": runs,
            "summary": summary,
            "paired_comparison": paired,
            "same_numeric_input_tensor": True,
            "same_target": True,
            "same_training_pairs": True,
            "same_sample_order": True,
            "autoregressive_rollout_performed": False,
            "teacher_endpoint_used_as_input": False,
        }
        atomic._write_json(
            self.output_dir / "information_matched_results.json", report
        )
        return report

    def finalize_information_matched_benchmark(
        self, contract: Mapping[str, Any], results: Mapping[str, Any]
    ) -> Dict[str, Any]:
        report = {
            "schema_version": "06b-c-information-matched-transition-final-v1",
            "valid": bool(contract.get("valid") and results.get("valid")),
            "status": "completed_information_matched_one_step_voltage_comparison",
            "sidecar_only": True,
            "primary_experiment_replaced": False,
            "comparison_complete": True,
            "scientific_voltage_ranking_authorized": True,
            "comparison_scope": "train-derived disjoint development; authentic one-step voltage only",
            "contract": {
                "same_numeric_input_tensor": True,
                "same_authentic_teacher_target": True,
                "same_training_pairs": True,
                "same_sample_order": True,
                "same_loss_and_optimizer": True,
                "target_clipping": None,
                "autoregressive_rollout_performed": False,
                "teacher_endpoint_used_as_input": False,
            },
            "parameters": contract["models"],
            "summary": results["summary"],
            "paired_comparison": results["paired_comparison"],
            "limitations": {
                "published_checkpoint_directly_comparable": False,
                "reason": "the published checkpoint uses event-only NeuronIO routing; the matched ELM core is retrained on the common causal state tensor",
                "state_updater_scored_by_voltage_metric": False,
                "fresh_test_used": False,
                "spike_metric_in_scope": False,
            },
            "retracted_05j_n_comparison_reinstated": False,
            "professor_sidecar_closed_after_this_result": True,
            "next_step": "return_to_nested_coupling_aware_local_bridge_optimization_scaling_forensic",
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_information_matched_artifact_index()
        return report

    def _write_information_matched_artifact_index(self) -> None:
        artifacts = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.name == "artifact_index.json":
                continue
            artifacts.append(
                {
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": atomic._sha256_file(path),
                }
            )
        atomic._write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "06b-c-information-matched-transition-index-v1",
                "artifacts": artifacts,
            },
        )


__all__ = [
    "MATCHED_MODEL_NAMES",
    "InformationMatchedTransitionConfig",
    "InformationMatchedBranchELM",
    "InformationMatchedBridgeAdapter",
    "InformationMatchedVoltageTransitionBenchmark",
]
