"""06b-d: synchronized factorial forensic for the causal voltage bridge.

The experiment keeps the mechanism-STATE updater frozen and continues the
same 8,985-parameter voltage bridge under paired objective and optimizer
controls.  All arms use the same initialization and minibatch stream.  Fixed
budget checkpoints provide a mini scaling law without duplicate training.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .causal_voltage_bridge_representation_forensic import (
    CausalVoltageBridgeRepresentationConfig,
    CausalVoltageBridgeRepresentationForensic,
)


EXPECTED_06BC_ARCHIVE_SHA256 = (
    "6c0df160b7f175ef2a46ff05ad88ee53d483a8eb14d893c0fa73de2190d7330d"
)
EXPECTED_06BC_INDEX_SHA256 = (
    "d828d9ad318998c6e07632b262493838dfd853cf4166edbfe034e11b6b6ec23e"
)
EXPECTED_06BC_FINAL_SHA256 = (
    "378421fff03d1e87e145334792fa1ac66e6df0627f24333b1cd256b70787af85"
)

SCALING_ARMS = (
    "voltage_constant",
    "voltage_cosine",
    "joint_constant",
    "joint_cosine",
    "joint_shuffled_cosine",
)

ARM_OBJECTIVE = {
    "voltage_constant": "voltage",
    "voltage_cosine": "voltage",
    "joint_constant": "joint",
    "joint_cosine": "joint",
    "joint_shuffled_cosine": "joint_shuffled",
}

ARM_SCHEDULE = {
    "voltage_constant": "constant",
    "voltage_cosine": "cosine",
    "joint_constant": "constant",
    "joint_cosine": "cosine",
    "joint_shuffled_cosine": "cosine",
}


def verified_06bc_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact registered 06b-c authorization artifact."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-c source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        if archive_hash != EXPECTED_06BC_ARCHIVE_SHA256:
            archive_hash = "kaggle-repacked"
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
        archive_hash = "extracted-directory"
        search_root = source
    matches = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06BC_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-c artifact; found {len(matches)}")
    root = matches[0]
    index = json.loads((root / "artifact_index.json").read_text(encoding="utf-8"))
    failures = []
    for record in index.get("artifacts", []):
        member = root / str(record["path"])
        if (
            not member.is_file()
            or member.stat().st_size != int(record["size_bytes"])
            or atomic._sha256_file(member) != str(record["sha256"])
        ):
            failures.append(str(record["path"]))
    if failures:
        raise RuntimeError(f"06b-c indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BC_FINAL_SHA256:
        raise RuntimeError("06b-c final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "LOCAL_BRIDGE_OPTIMIZATION_LIMIT_IDENTIFIED"
        or final.get("component_decision_grade") is not True
        or final.get("optimization_limit_identified") is not True
    ):
        raise RuntimeError("06b-c source does not authorize optimization scaling")
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BC_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BC_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class NestedCouplingOptimizationScalingConfig(
    CausalVoltageBridgeRepresentationConfig
):
    scaling_training_steps: int = 1500
    scaling_checkpoints: Tuple[int, ...] = (0, 250, 500, 1000, 1500)
    coupling_coordinates_per_batch: int = 1024
    cosine_minimum_learning_rate_ratio: float = 0.1
    state_gradient_scale_minimum: float = 0.01
    state_gradient_scale_maximum: float = 100.0
    minimum_scaling_gain_fraction: float = 0.01
    minimum_joint_state_gain_fraction: float = 0.01
    maximum_joint_voltage_degradation_fraction: float = 0.01
    minimum_causal_specificity_gain_fraction: float = 0.01
    minimum_recursive_gain_fraction: float = 0.02

    def validate(self) -> None:
        super().validate()
        if self.scaling_training_steps <= 0 or self.coupling_coordinates_per_batch <= 0:
            raise ValueError("06b-d training dimensions must be positive")
        checkpoints = tuple(map(int, self.scaling_checkpoints))
        if (
            not checkpoints
            or checkpoints[0] != 0
            or checkpoints[-1] != self.scaling_training_steps
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError("06b-d checkpoints must be unique, sorted, and span training")
        if not 0 < self.cosine_minimum_learning_rate_ratio <= 1:
            raise ValueError("06b-d cosine floor must lie in (0, 1]")
        if not 0 < self.state_gradient_scale_minimum <= self.state_gradient_scale_maximum:
            raise ValueError("06b-d gradient scaling bounds are invalid")
        thresholds = (
            self.minimum_scaling_gain_fraction,
            self.minimum_joint_state_gain_fraction,
            self.maximum_joint_voltage_degradation_fraction,
            self.minimum_causal_specificity_gain_fraction,
            self.minimum_recursive_gain_fraction,
        )
        if any(not 0 < value < 1 for value in thresholds):
            raise ValueError("06b-d registered thresholds must lie in (0, 1)")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "NestedCouplingOptimizationScalingConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class NestedCouplingOptimizationScalingForensic(
    CausalVoltageBridgeRepresentationForensic
):
    """Run paired objective/schedule arms on synchronized minibatches."""

    config: NestedCouplingOptimizationScalingConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: NestedCouplingOptimizationScalingConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
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
            code_revision=code_revision,
        )
        self.artifact_06bc_source = Path(artifact_06bc_source)
        self.coordinate_lookup: Optional[np.ndarray] = None
        self.scaling_models: Dict[Tuple[str, int], Any] = {}
        self.scaling_states: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self.state_gradient_scales: Dict[int, float] = {}

    def _build_coordinate_lookup(self) -> None:
        group_count = len(self.coordinate_groups)
        lookup = np.full(
            (self.layout.segment_count, group_count), -1, dtype=np.int64
        )
        for coordinate in range(len(self.mechanism_records)):
            segment = int(self.coordinate["segment"][coordinate])
            group = int(self.coordinate["semantic_group"][coordinate])
            if lookup[segment, group] >= 0:
                raise RuntimeError("06b-d duplicate semantic coordinate per segment")
            lookup[segment, group] = coordinate
        self.coordinate_lookup = lookup

    def prepare_scaling_forensic(self) -> Dict[str, Any]:
        base = self.prepare_representation_forensic()
        _, source = verified_06bc_artifact_root(
            self.artifact_06bc_source,
            self.output_dir.parent / ".06bd_artifact_cache" / "06bc",
        )
        self._build_coordinate_lookup()
        parameter_counts = {
            seed: int(sum(value.numel() for value in model.parameters()))
            for seed, model in self.frozen_bridge_models.items()
        }
        blockers = []
        if len(set(parameter_counts.values())) != 1:
            blockers.append("frozen bridge parameter counts differ across seeds")
        if set(self.config.scaling_checkpoints) != {
            0,
            250,
            500,
            1000,
            self.config.scaling_training_steps,
        }:
            blockers.append("registered mini scaling-law checkpoints changed")
        report = {
            **base,
            "schema_version": "06b-d-scaling-contract-v1",
            "valid": bool(base.get("valid")) and not blockers,
            "blockers": blockers,
            "experiment": "nested_coupling_optimization_scaling_forensic",
            "source_06bc": source,
            "arms": list(SCALING_ARMS),
            "arm_objectives": dict(ARM_OBJECTIVE),
            "arm_schedules": dict(ARM_SCHEDULE),
            "scaling_checkpoints": list(self.config.scaling_checkpoints),
            "paired_seed_count": len(self.config.pilot_seeds),
            "same_initialization_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "same_architecture_and_parameter_count": True,
            "bridge_parameter_count": next(iter(parameter_counts.values())),
            "state_updater_frozen": True,
            "state_updater_retraining_performed": False,
            "joint_objective_backpropagates_through_frozen_state_updater": True,
            "state_loss_scale": "fit-only initial gradient-norm ratio, clipped and then frozen",
            "shuffled_control": "same model and losses with predicted voltage paths rotated across coordinate examples",
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "physical_parallelism": "synchronized arms on one GPU; shared batches, deterministic sequential arm updates",
        }
        atomic._write_json(self.output_dir / "scaling_contract.json", report)
        if blockers:
            raise RuntimeError(f"06b-d preflight failed: {blockers}")
        return report

    def _new_continuation_model(self, seed: int, device: Any) -> Any:
        model = self._new_bridge(device)
        model.load_state_dict(
            copy.deepcopy(self.frozen_bridge_models[seed].state_dict())
        )
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model

    def _state_batch_plan(
        self,
        role: str,
        transition_rows: np.ndarray,
        segments: np.ndarray,
        rng: np.random.Generator,
        device: Any,
    ) -> Dict[str, Any]:
        if self.coordinate_lookup is None:
            raise RuntimeError("06b-d coordinate lookup is not prepared")
        values = self.materialized[role]
        flat_rows = np.repeat(transition_rows, segments.shape[1])
        flat_segments = segments.reshape(-1)
        lookup = self.coordinate_lookup[flat_segments]
        mask = lookup >= 0
        coordinates = lookup[mask]
        coordinate_rows = np.broadcast_to(flat_rows[:, None], lookup.shape)[mask]
        coordinate_segments = np.broadcast_to(
            flat_segments[:, None], lookup.shape
        )[mask]
        prediction_sources = np.broadcast_to(
            np.arange(len(flat_segments), dtype=np.int64)[:, None], lookup.shape
        )[mask]
        if len(coordinates) > self.config.coupling_coordinates_per_batch:
            selected = rng.choice(
                len(coordinates),
                size=self.config.coupling_coordinates_per_batch,
                replace=False,
            )
            coordinates = coordinates[selected]
            coordinate_rows = coordinate_rows[selected]
            coordinate_segments = coordinate_segments[selected]
            prediction_sources = prediction_sources[selected]
        tensor = lambda value, dtype=None: atomic.torch.as_tensor(
            value, dtype=dtype, device=device
        )
        return {
            "prediction_sources": tensor(prediction_sources, atomic.torch.long),
            "state": tensor(
                values["state"][coordinate_rows, coordinates], atomic.torch.float32
            ),
            "voltage_t": tensor(
                values["voltage_t"][coordinate_rows, coordinate_segments],
                atomic.torch.float32,
            ),
            "context": tensor(
                values["context"][coordinate_rows, coordinate_segments],
                atomic.torch.float32,
            ),
            "static": tensor(
                self.layout.segment_static[coordinate_segments], atomic.torch.float32
            ),
            "mechanism": tensor(
                self.coordinate["mechanism"][coordinates], atomic.torch.long
            ),
            "variable": tensor(
                self.coordinate["variable"][coordinates], atomic.torch.long
            ),
            "kind": tensor(self.coordinate["kind"][coordinates], atomic.torch.long),
            "region": tensor(
                self.coordinate["region"][coordinates], atomic.torch.long
            ),
            "target": tensor(
                values["delta"][coordinate_rows, coordinates], atomic.torch.float32
            ),
        }

    def _differentiable_state_loss(
        self,
        seed: int,
        prediction: Any,
        plan: Mapping[str, Any],
        *,
        shuffled: bool,
    ) -> Any:
        predicted = prediction.reshape(-1)[plan["prediction_sources"]]
        if shuffled:
            predicted = atomic.torch.roll(predicted, shifts=1, dims=0)
        fractions = atomic.torch.as_tensor(
            np.asarray(self.config.voltage_path_sample_indices, dtype=np.float32)
            / float(self.config.expected_microtrace_sample_count - 1),
            dtype=atomic.torch.float32,
            device=predicted.device,
        )
        path = (
            predicted[:, None]
            * self.config.bridge_voltage_scale_mv
            * fractions[None, :]
        )
        endpoint_model = self.frozen_state_models[("linear_endpoint_path", seed)]
        state_prediction = endpoint_model(
            plan["state"],
            plan["voltage_t"],
            path,
            plan["context"],
            plan["static"],
            plan["mechanism"],
            plan["variable"],
            plan["kind"],
            plan["region"],
        )
        weight = 1.0 + self.config.active_delta_weight * (
            plan["target"].abs() >= self.config.active_delta_threshold
        ).float()
        return atomic.torch.mean(
            weight
            * atomic.torch_functional.smooth_l1_loss(
                state_prediction, plan["target"], reduction="none"
            )
        )

    def _voltage_loss(self, prediction: Any, target: Any) -> Any:
        active_threshold = (
            self.config.bridge_active_delta_threshold_mv
            / self.config.bridge_voltage_scale_mv
        )
        weight = 1.0 + self.config.bridge_active_weight * (
            target.abs() >= active_threshold
        ).float()
        return atomic.torch.mean(
            weight
            * atomic.torch_functional.smooth_l1_loss(
                prediction, target, reduction="none"
            )
        )

    @staticmethod
    def _gradient_vector(loss: Any, model: Any, *, retain_graph: bool) -> Any:
        gradients = atomic.torch.autograd.grad(
            loss,
            tuple(model.parameters()),
            retain_graph=retain_graph,
            allow_unused=True,
        )
        values = []
        for parameter, gradient in zip(model.parameters(), gradients):
            values.append(
                atomic.torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
            )
        return atomic.torch.cat(values)

    def _gradient_probe(
        self,
        seed: int,
        model: Any,
        inputs: Tuple[Any, ...],
        target: Any,
        plan: Mapping[str, Any],
        *,
        shuffled: bool,
    ) -> Dict[str, float]:
        prediction = model(*inputs)
        voltage_loss = self._voltage_loss(prediction, target)
        state_loss = self._differentiable_state_loss(
            seed, prediction, plan, shuffled=shuffled
        )
        voltage_gradient = self._gradient_vector(
            voltage_loss, model, retain_graph=True
        )
        state_gradient = self._gradient_vector(state_loss, model, retain_graph=False)
        voltage_norm = float(atomic.torch.linalg.vector_norm(voltage_gradient).detach().cpu())
        state_norm = float(atomic.torch.linalg.vector_norm(state_gradient).detach().cpu())
        denominator = max(voltage_norm * state_norm, 1e-20)
        cosine = float(
            atomic.torch.dot(voltage_gradient, state_gradient).detach().cpu()
        ) / denominator
        return {
            "voltage_gradient_norm": voltage_norm,
            "state_gradient_norm": state_norm,
            "gradient_cosine": cosine,
            "voltage_loss": float(voltage_loss.detach().cpu()),
            "state_loss": float(state_loss.detach().cpu()),
        }

    def _learning_rate(self, arm: str, step: int) -> float:
        if ARM_SCHEDULE[arm] == "constant":
            return self.config.bridge_learning_rate
        phase = float(step) / float(self.config.scaling_training_steps)
        multiplier = self.config.cosine_minimum_learning_rate_ratio + (
            1.0 - self.config.cosine_minimum_learning_rate_ratio
        ) * 0.5 * (1.0 + math.cos(math.pi * phase))
        return self.config.bridge_learning_rate * multiplier

    def _checkpoint(
        self,
        seed: int,
        arm: str,
        budget: int,
        model: Any,
        device: Any,
        probe: Tuple[Tuple[Any, ...], Any, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        self.scaling_states[(arm, seed, budget)] = state
        checkpoint = self.output_dir / f"scaling_{arm}_seed{seed}_step{budget}.pt"
        atomic.torch.save(
            {
                "state_dict": state,
                "arm": arm,
                "seed": seed,
                "budget": budget,
                "configuration": asdict(self.config),
            },
            checkpoint,
        )
        inputs, target, plan = probe
        probe_report = self._gradient_probe(
            seed,
            model,
            inputs,
            target,
            plan,
            shuffled=arm == "joint_shuffled_cosine",
        )
        model.eval()
        calibration = self._evaluate_bridge(model, "calibration", device)
        predicted = self._predict_bridge(
            model,
            self.materialized["calibration"]["state"],
            self.materialized["calibration"]["voltage_t"],
            self.materialized["calibration"]["context"],
            device,
        )
        state_metrics = self._state_metrics_from_path(
            self.frozen_state_models[("linear_endpoint_path", seed)],
            "calibration",
            predicted,
            device,
        )
        model.train()
        return {
            "budget": budget,
            "learning_rate": self._learning_rate(arm, budget),
            "calibration_voltage": calibration,
            "calibration_state": state_metrics,
            "gradient_probe": probe_report,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": atomic._sha256_file(checkpoint),
        }

    def train_synchronized_scaling_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        reports: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            models = {
                arm: self._new_continuation_model(seed, device) for arm in SCALING_ARMS
            }
            optimizers = {
                arm: atomic.torch.optim.AdamW(
                    models[arm].parameters(),
                    lr=self.config.bridge_learning_rate,
                    weight_decay=self.config.bridge_weight_decay,
                )
                for arm in SCALING_ARMS
            }
            rng = np.random.default_rng(seed + 640000)
            probe_rng = np.random.default_rng(seed + 640001)
            fit = self.materialized["fit"]
            probe_rows = probe_rng.integers(
                0,
                len(fit["indices"]),
                size=self.config.bridge_batch_transition_count,
            )
            probe_segments = probe_rng.integers(
                0,
                self.layout.segment_count,
                size=(
                    self.config.bridge_batch_transition_count,
                    self.config.bridge_segments_per_transition,
                ),
            )
            probe_inputs, probe_target = self._bridge_batch(
                "fit", probe_rows, probe_segments, device
            )
            probe_plan = self._state_batch_plan(
                "fit", probe_rows, probe_segments, probe_rng, device
            )
            base_probe = self._gradient_probe(
                seed,
                models["joint_cosine"],
                probe_inputs,
                probe_target,
                probe_plan,
                shuffled=False,
            )
            scale = np.clip(
                base_probe["voltage_gradient_norm"]
                / max(base_probe["state_gradient_norm"], 1e-20),
                self.config.state_gradient_scale_minimum,
                self.config.state_gradient_scale_maximum,
            )
            self.state_gradient_scales[seed] = float(scale)
            seed_report = {
                "state_gradient_scale": float(scale),
                "initial_gradient_probe": base_probe,
                "arms": {arm: [] for arm in SCALING_ARMS},
            }
            probe = (probe_inputs, probe_target, probe_plan)
            for arm in SCALING_ARMS:
                seed_report["arms"][arm].append(
                    self._checkpoint(seed, arm, 0, models[arm], device, probe)
                )
            progress = atomic._CompactProgress(
                f"06b-d synchronized seed={seed}",
                self.config.scaling_training_steps,
                self.config.bridge_progress_interval,
            )
            for step in range(1, self.config.scaling_training_steps + 1):
                rows = rng.integers(
                    0,
                    len(fit["indices"]),
                    size=self.config.bridge_batch_transition_count,
                )
                segments = rng.integers(
                    0,
                    self.layout.segment_count,
                    size=(
                        self.config.bridge_batch_transition_count,
                        self.config.bridge_segments_per_transition,
                    ),
                )
                inputs, target = self._bridge_batch("fit", rows, segments, device)
                plan = self._state_batch_plan("fit", rows, segments, rng, device)
                step_losses = []
                for arm in SCALING_ARMS:
                    model = models[arm]
                    optimizer = optimizers[arm]
                    learning_rate = self._learning_rate(arm, step)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    prediction = model(*inputs)
                    voltage_loss = self._voltage_loss(prediction, target)
                    objective = ARM_OBJECTIVE[arm]
                    if objective == "voltage":
                        loss = voltage_loss
                    else:
                        state_loss = self._differentiable_state_loss(
                            seed,
                            prediction,
                            plan,
                            shuffled=objective == "joint_shuffled",
                        )
                        loss = voltage_loss + float(scale) * state_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.bridge_gradient_clip_norm
                    )
                    optimizer.step()
                    step_losses.append(float(loss.detach().cpu()))
                if step in self.config.scaling_checkpoints:
                    for arm in SCALING_ARMS:
                        seed_report["arms"][arm].append(
                            self._checkpoint(
                                seed, arm, step, models[arm], device, probe
                            )
                        )
                progress.update(
                    step,
                    f"median_loss={float(np.median(step_losses)):.4g}",
                )
            for arm, model in models.items():
                model.eval()
                self.scaling_models[(arm, seed)] = model
            reports[str(seed)] = seed_report
            atomic._write_json(
                self.output_dir / f"scaling_seed{seed}.json", seed_report
            )
        report = {
            "schema_version": "06b-d-synchronized-training-v1",
            "valid": all(
                len(row["arms"][arm]) == len(self.config.scaling_checkpoints)
                for row in reports.values()
                for arm in SCALING_ARMS
            ),
            "device": str(device),
            "arms": list(SCALING_ARMS),
            "reports": reports,
            "same_minibatch_stream_within_seed": True,
            "same_initialization_within_seed": True,
            "state_updater_retraining_performed": False,
        }
        atomic._write_json(self.output_dir / "synchronized_training.json", report)
        return report

    def _model_at_budget(self, arm: str, seed: int, budget: int, device: Any) -> Any:
        model = self._new_bridge(device)
        model.load_state_dict(self.scaling_states[(arm, seed, budget)])
        model.eval()
        return model

    def evaluate_fixed_budget_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed: Dict[str, Any] = {}
        role = "development"
        values = self.materialized[role]
        for seed in self.config.pilot_seeds:
            seed_rows: Dict[str, Any] = {}
            endpoint = self.frozen_state_models[("linear_endpoint_path", seed)]
            for arm in SCALING_ARMS:
                budget_rows = {}
                for budget in self.config.scaling_checkpoints:
                    model = self._model_at_budget(arm, seed, budget, device)
                    voltage = self._evaluate_bridge(model, role, device)
                    predicted = self._predict_bridge(
                        model,
                        values["state"],
                        values["voltage_t"],
                        values["context"],
                        device,
                    )
                    state = self._state_metrics_from_path(
                        endpoint, role, predicted, device
                    )
                    budget_rows[str(budget)] = {
                        "voltage": voltage,
                        "state": state,
                    }
                seed_rows[arm] = budget_rows
            per_seed[str(seed)] = seed_rows
        report = {
            "schema_version": "06b-d-fixed-budget-development-v1",
            "valid": all(
                row[arm][str(budget)]["voltage"]["nonfinite_count"] == 0
                and row[arm][str(budget)]["state"]["nonfinite_count"] == 0
                for row in per_seed.values()
                for arm in SCALING_ARMS
                for budget in self.config.scaling_checkpoints
            ),
            "development_used_for_selection": False,
            "all_budgets_preregistered": True,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "fixed_budget_development.json", report)
        return report

    def evaluate_final_nested_rollouts(self) -> Dict[str, Any]:
        windows = self._nested_development_windows()
        if not windows:
            raise RuntimeError("06b-d found no nested development windows")
        digest = hashlib.sha256(
            json.dumps(
                [list(map(int, row)) for row in windows], separators=(",", ":")
            ).encode()
        ).hexdigest()
        first = np.asarray([row[0] for row in windows], dtype=np.int64)
        initial = atomic.mechanism_logit(
            self.store.read_state(first, "t", categories=("mechanism_states",))
        ).astype(np.float32)
        fractions = np.asarray(
            self.config.voltage_path_sample_indices, dtype=np.float32
        ) / float(self.config.expected_microtrace_sample_count - 1)
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed: Dict[str, Any] = {}
        maximum_horizon = max(self.config.rollout_horizons_ms)
        for seed in self.config.pilot_seeds:
            seed_rows = {}
            endpoint = self.frozen_state_models[("linear_endpoint_path", seed)]
            for arm in SCALING_ARMS:
                transformed = initial.copy()
                model = self.scaling_models[(arm, seed)]
                horizons = {}
                for step in range(maximum_horizon):
                    indices = np.asarray([row[step] for row in windows], dtype=np.int64)
                    voltage_t = self.store.read_state(
                        indices, "t", categories=("voltage",)
                    ).astype(np.float32)
                    context = np.concatenate(
                        (
                            atomic.encode_causal_realized_drive(self.store, indices),
                            self._ion_context(indices),
                        ),
                        axis=-1,
                    )
                    normalized = (
                        transformed - self.statistics["state_center"]
                    ) / self.statistics["state_scale"]
                    voltage_delta = self._predict_bridge(
                        model,
                        normalized.astype(np.float32),
                        voltage_t,
                        context,
                        device,
                    )
                    path = voltage_delta[:, :, None] * fractions[None, None, :]
                    predicted_delta = self._predict_full_delta_path(
                        endpoint,
                        normalized.astype(np.float32),
                        voltage_t,
                        path,
                        context,
                        device,
                    )
                    transformed += predicted_delta * self.statistics["delta_scale"]
                    horizon = step + 1
                    if horizon in self.config.rollout_horizons_ms:
                        target = atomic.mechanism_logit(
                            self.store.read_state(
                                indices,
                                "t_plus_1",
                                categories=("mechanism_states",),
                            )
                        ).astype(np.float32)
                        error = (
                            transformed - target
                        ) / self.statistics["state_scale"]
                        persistence_error = (
                            initial - target
                        ) / self.statistics["state_scale"]
                        rmse = float(np.sqrt(np.mean(error * error)))
                        persistence = float(
                            np.sqrt(np.mean(persistence_error * persistence_error))
                        )
                        raw = atomic.inverse_mechanism_logit(transformed)
                        horizons[f"{horizon}_ms"] = {
                            "normalized_state_rmse": rmse,
                            "persistence_normalized_state_rmse": persistence,
                            "improvement_vs_persistence_fraction": 1.0
                            - rmse / max(persistence, 1e-12),
                            "nonfinite_count": int(np.sum(~np.isfinite(raw))),
                            "domain_violation_count": int(
                                np.sum((raw < 0.0) | (raw > 1.0))
                            ),
                        }
                seed_rows[arm] = horizons
            per_seed[str(seed)] = seed_rows
        report = {
            "schema_version": "06b-d-final-nested-rollouts-v1",
            "valid": all(
                metric["nonfinite_count"] == 0
                and metric["domain_violation_count"] == 0
                for seed in per_seed.values()
                for arm in seed.values()
                for metric in arm.values()
            ),
            "common_window_count": len(windows),
            "common_window_set_sha256": digest,
            "all_horizons_are_prefixes_of_same_windows": True,
            "teacher_voltage_boundary_used_each_ms": True,
            "autonomous_voltage_rollout_claimed": False,
            "state_updater_retraining_performed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "final_nested_rollouts.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _plot_summary(
        self, development: Mapping[str, Any], rollout: Mapping[str, Any]
    ) -> List[str]:
        import matplotlib.pyplot as plt

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        budgets = list(self.config.scaling_checkpoints)
        for arm in SCALING_ARMS:
            voltage = [
                self._median(
                    [
                        development["per_seed"][str(seed)][arm][str(budget)][
                            "voltage"
                        ]["improvement_vs_persistence_fraction"]
                        for seed in self.config.pilot_seeds
                    ]
                )
                for budget in budgets
            ]
            state = [
                self._median(
                    [
                        development["per_seed"][str(seed)][arm][str(budget)][
                            "state"
                        ]["improvement_vs_persistence_fraction"]
                        for seed in self.config.pilot_seeds
                    ]
                )
                for budget in budgets
            ]
            axes[0].plot(budgets, voltage, marker="o", label=arm)
            axes[1].plot(budgets, state, marker="o", label=arm)
            axes[2].plot(
                self.config.rollout_horizons_ms,
                [
                    self._median(
                        [
                            rollout["per_seed"][str(seed)][arm][f"{horizon}_ms"][
                                "improvement_vs_persistence_fraction"
                            ]
                            for seed in self.config.pilot_seeds
                        ]
                    )
                    for horizon in self.config.rollout_horizons_ms
                ],
                marker="o",
                label=arm,
            )
        axes[0].set(xlabel="optimizer steps", ylabel="median voltage gain")
        axes[1].set(xlabel="optimizer steps", ylabel="median one-step STATE gain")
        axes[2].set(
            xlabel="nested horizon (ms)", ylabel="median recursive STATE gain"
        )
        for axis in axes:
            axis.axhline(0.0, color="black", linewidth=1)
            axis.grid(alpha=0.25)
        axes[2].legend(fontsize=6, loc="best")
        figure.tight_layout()
        path = figure_dir / "nested_coupling_scaling_matrix.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        return [str(path.relative_to(self.output_dir))]

    def finalize_scaling_forensic(
        self,
        training: Mapping[str, Any],
        development: Mapping[str, Any],
        rollout: Mapping[str, Any],
    ) -> Dict[str, Any]:
        maximum = str(self.config.scaling_training_steps)
        reference = "500"
        per_seed = {}
        for seed in self.config.pilot_seeds:
            rows = development["per_seed"][str(seed)]
            voltage_gain = lambda arm, budget=maximum: float(
                rows[arm][budget]["voltage"]["improvement_vs_persistence_fraction"]
            )
            state_gain = lambda arm, budget=maximum: float(
                rows[arm][budget]["state"]["improvement_vs_persistence_fraction"]
            )
            rollout_gain = lambda arm: float(
                rollout["per_seed"][str(seed)][arm]["8_ms"][
                    "improvement_vs_persistence_fraction"
                ]
            )
            per_seed[str(seed)] = {
                "scaling_voltage_gain": voltage_gain("joint_cosine")
                - voltage_gain("joint_cosine", reference),
                "scaling_state_gain": state_gain("joint_cosine")
                - state_gain("joint_cosine", reference),
                "joint_voltage_effect": voltage_gain("joint_cosine")
                - voltage_gain("voltage_cosine"),
                "joint_state_effect": state_gain("joint_cosine")
                - state_gain("voltage_cosine"),
                "joint_recursive_effect": rollout_gain("joint_cosine")
                - rollout_gain("voltage_cosine"),
                "causal_specificity_voltage_effect": voltage_gain("joint_cosine")
                - voltage_gain("joint_shuffled_cosine"),
                "causal_specificity_state_effect": state_gain("joint_cosine")
                - state_gain("joint_shuffled_cosine"),
                "cosine_voltage_objective_effect": voltage_gain("voltage_cosine")
                - voltage_gain("voltage_constant"),
                "cosine_joint_objective_effect": state_gain("joint_cosine")
                - state_gain("joint_constant"),
            }
        median = {
            key: self._median([row[key] for row in per_seed.values()])
            for key in next(iter(per_seed.values()))
        }
        scaling_identified = max(
            median["scaling_voltage_gain"], median["scaling_state_gain"]
        ) >= self.config.minimum_scaling_gain_fraction
        joint_identified = (
            median["joint_state_effect"]
            >= self.config.minimum_joint_state_gain_fraction
            and median["joint_voltage_effect"]
            >= -self.config.maximum_joint_voltage_degradation_fraction
        )
        causal_specificity = (
            median["causal_specificity_state_effect"]
            >= self.config.minimum_causal_specificity_gain_fraction
        )
        recursive_identified = (
            median["joint_recursive_effect"]
            >= self.config.minimum_recursive_gain_fraction
        )
        schedule_identified = max(
            median["cosine_voltage_objective_effect"],
            median["cosine_joint_objective_effect"],
        ) >= self.config.minimum_scaling_gain_fraction
        if joint_identified and causal_specificity and recursive_identified:
            diagnosis = "NESTED_COUPLING_OBJECTIVE_IDENTIFIED"
            next_step = "06c_coupled_voltage_state_micro_canary"
        elif joint_identified and causal_specificity:
            diagnosis = "ONE_STEP_COUPLING_OBJECTIVE_ONLY"
            next_step = "redesign_recursive_voltage_state_contract"
        elif scaling_identified or schedule_identified:
            diagnosis = "LOCAL_BRIDGE_OPTIMIZATION_SCALING_IDENTIFIED"
            next_step = "bounded_bridge_optimization_lead_refinement"
        else:
            diagnosis = "LOCAL_BRIDGE_REPRESENTATION_REDESIGN_REQUIRED"
            next_step = "return_to_atomic_voltage_representation_playground"
        figures = self._plot_summary(development, rollout)
        report = {
            "schema_version": "06b-d-final-report-v1",
            "valid": bool(
                training.get("valid")
                and development.get("valid")
                and rollout.get("valid")
            ),
            "component_decision_grade": True,
            "diagnosis": diagnosis,
            "gate_checks": {
                "optimizer_budget_scaling_identified": scaling_identified,
                "joint_objective_identified": joint_identified,
                "causal_specificity_identified": causal_specificity,
                "recursive_benefit_identified": recursive_identified,
                "cosine_schedule_identified": schedule_identified,
            },
            "median_contrasts": median,
            "per_seed_contrasts": per_seed,
            "registered_thresholds": {
                "minimum_scaling_gain_fraction": self.config.minimum_scaling_gain_fraction,
                "minimum_joint_state_gain_fraction": self.config.minimum_joint_state_gain_fraction,
                "maximum_joint_voltage_degradation_fraction": self.config.maximum_joint_voltage_degradation_fraction,
                "minimum_causal_specificity_gain_fraction": self.config.minimum_causal_specificity_gain_fraction,
                "minimum_recursive_gain_fraction": self.config.minimum_recursive_gain_fraction,
            },
            "multiple_questions_answered_in_one_matrix": [
                "optimizer_budget_scaling",
                "constant_vs_cosine_schedule",
                "voltage_only_vs_joint_state_aware_objective",
                "causal_vs_shuffled_coupling",
                "one_step_vs_nested_eight_ms_transfer",
                "voltage_and_internal_STATE_gradient_alignment",
            ],
            "same_minibatch_stream_within_seed": True,
            "same_initialization_within_seed": True,
            "state_updater_retraining_performed": False,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "autonomous_voltage_rollout_claimed": False,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "figures": figures,
            "next_step": next_step,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BC_ARCHIVE_SHA256",
    "EXPECTED_06BC_INDEX_SHA256",
    "EXPECTED_06BC_FINAL_SHA256",
    "SCALING_ARMS",
    "NestedCouplingOptimizationScalingConfig",
    "NestedCouplingOptimizationScalingForensic",
    "verified_06bc_artifact_root",
]
