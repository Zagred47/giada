"""06b-i: train-only causal voltage-objective recalibration playground."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .frozen_voltage_generalization_forensic import (
    FrozenVoltageForensicConfig,
    FrozenVoltageGeneralizationForensic,
)


EXPECTED_06BH_ARCHIVE_SHA256 = (
    "8c2c255a234c7fb25673bafcec0e7521ca084fd98d7d9c0c9cc86211bab1b948"
)
EXPECTED_06BH_INDEX_SHA256 = (
    "4ab21e154a1972cdbd9e059c7bcc7094814810eb4b0b0cac8a8a2c49f69a1c2d"
)
EXPECTED_06BH_FINAL_SHA256 = (
    "bfc1a664d3549a2aecd28a6c2df200e188e0a05498c90c2afde353f75ea2f813"
)

TRAINABLE_OBJECTIVE_ARMS = {
    "raw_bridge_control": "original_active_weight_bridge_update",
    "activity_balanced_bridge": "activity_balanced_bridge_update",
    "activity_region_balanced_bridge": "activity_region_balanced_bridge_update",
    "global_gain_balanced": "activity_balanced_global_gain_only",
    "causal_gain_balanced": "activity_balanced_causal_gain_only",
}
FROZEN_REFERENCE_ARM = "frozen_alpha_075"
PRIMARY_ARM = "causal_gain_balanced"
FALLBACK_ARM = "activity_region_balanced_bridge"


def verified_06bh_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-h source must be a ZIP or extracted directory")
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
        if atomic._sha256_file(path) == EXPECTED_06BH_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-h artifact; found {len(matches)}")
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
        raise RuntimeError(f"06b-h indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BH_FINAL_SHA256:
        raise RuntimeError("06b-h final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "FROZEN_VOLTAGE_CALIBRATION_RESCUES_GENERALIZATION"
        or final.get("next_step") != "train_only_voltage_objective_recalibration"
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-h result does not authorize 06b-i")
    if source.is_file() and archive_hash != EXPECTED_06BH_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BH_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BH_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
        "selected_frozen_candidate": final["selected_frozen_candidate"],
    }


@dataclass(frozen=True)
class VoltageObjectiveRecalibrationConfig(FrozenVoltageForensicConfig):
    objective_training_steps: int = 400
    objective_checkpoints: Tuple[int, ...] = (0, 100, 200, 400)
    objective_unroll_horizon_ms: int = 8
    objective_batch_window_count: int = 4
    objective_learning_rate: float = 0.00015
    objective_weight_decay: float = 0.00001
    objective_gradient_clip_norm: float = 1.0
    objective_fit_window_count: int = 64
    gain_hidden_width: int = 16
    gain_region_embedding_width: int = 4
    gain_minimum: float = 0.0
    gain_maximum: float = 1.5
    gain_initial: float = 0.75
    inverse_frequency_weight_cap: float = 20.0
    minimum_training_effect_fraction: float = 0.02
    minimum_gain_over_frozen_alpha_fraction: float = 0.02
    minimum_active_gain_fraction: float = 0.10
    minimum_quiescent_gain_fraction: float = 0.0
    minimum_moderate_gain_fraction: float = 0.0
    minimum_soma_gain_fraction: float = 0.0
    minimum_STATE_gain_fraction: float = 0.0
    maximum_gain_parameter_count: int = 1000

    def validate(self) -> None:
        super().validate()
        checkpoints = tuple(map(int, self.objective_checkpoints))
        if (
            checkpoints[0] != 0
            or checkpoints[-1] != self.objective_training_steps
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError("06b-i checkpoints must span the fixed training budget")
        if self.objective_unroll_horizon_ms != max(self.rollout_horizons_ms):
            raise ValueError("06b-i must optimize the registered maximum horizon")
        if not self.gain_minimum < self.gain_initial < self.gain_maximum:
            raise ValueError("06b-i gain initialization is outside its bounds")
        if min(
            self.objective_training_steps,
            self.objective_batch_window_count,
            self.objective_fit_window_count,
            self.gain_hidden_width,
            self.gain_region_embedding_width,
            self.maximum_gain_parameter_count,
        ) <= 0:
            raise ValueError("06b-i positive dimensions are invalid")
        if not 0 < self.objective_learning_rate < 1:
            raise ValueError("06b-i learning rate is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "VoltageObjectiveRecalibrationConfig":
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
        for name in ("voltage_shrinkage_grid", "activity_edges_mv"):
            if name in payload:
                payload[name] = tuple(map(float, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


if atomic.nn is not None:

    class GlobalVoltageGain(atomic.nn.Module):
        def __init__(self, minimum: float, maximum: float, initial: float) -> None:
            super().__init__()
            self.minimum = float(minimum)
            self.maximum = float(maximum)
            fraction = (float(initial) - self.minimum) / (self.maximum - self.minimum)
            logit = np.log(fraction / (1.0 - fraction))
            self.logit = atomic.nn.Parameter(atomic.torch.tensor(float(logit)))

        def forward(self, raw_delta: Any, voltage: Any, region_ids: Any) -> Any:
            del voltage, region_ids
            gain = self.minimum + (self.maximum - self.minimum) * atomic.torch.sigmoid(
                self.logit
            )
            return gain.expand_as(raw_delta)


    class CausalVoltageGain(atomic.nn.Module):
        def __init__(
            self,
            *,
            region_count: int,
            region_width: int,
            hidden_width: int,
            minimum: float,
            maximum: float,
            initial: float,
        ) -> None:
            super().__init__()
            self.minimum = float(minimum)
            self.maximum = float(maximum)
            self.region = atomic.nn.Embedding(region_count, region_width)
            self.network = atomic.nn.Sequential(
                atomic.nn.Linear(3 + region_width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, 1),
            )
            atomic.nn.init.zeros_(self.region.weight)
            atomic.nn.init.zeros_(self.network[-1].weight)
            fraction = (float(initial) - self.minimum) / (self.maximum - self.minimum)
            atomic.nn.init.constant_(
                self.network[-1].bias, float(np.log(fraction / (1.0 - fraction)))
            )

        def forward(self, raw_delta: Any, voltage: Any, region_ids: Any) -> Any:
            batch, segments = raw_delta.shape
            region = self.region(region_ids)[None, :, :].expand(batch, -1, -1)
            features = atomic.torch.cat(
                (
                    raw_delta[:, :, None] / 20.0,
                    raw_delta.abs()[:, :, None] / 20.0,
                    voltage[:, :, None] / 100.0,
                    region,
                ),
                dim=-1,
            )
            gain = self.minimum + (self.maximum - self.minimum) * atomic.torch.sigmoid(
                self.network(features).squeeze(-1)
            )
            return gain

else:  # pragma: no cover
    GlobalVoltageGain = None
    CausalVoltageGain = None


class VoltageObjectiveRecalibrationPlayground(FrozenVoltageGeneralizationForensic):
    config: VoltageObjectiveRecalibrationConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: VoltageObjectiveRecalibrationConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
        artifact_06bd_source: Path,
        artifact_06be_source: Path,
        artifact_06bf_source: Path,
        artifact_06bg_source: Path,
        artifact_06bh_source: Path,
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
            code_revision=code_revision,
        )
        self.artifact_06bh_source = Path(artifact_06bh_source)
        self.objective_models: Dict[Tuple[str, int], Tuple[Any, Any, Any]] = {}
        self.objective_states: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self.weight_contract: Dict[str, Any] = {}

    def _fit_weight_contract(self) -> Dict[str, Any]:
        target_delta = np.abs(
            self.window_data["fit"]["voltage_t1"]
            - self.window_data["fit"]["voltage_t"]
        )
        activity = np.digitize(target_delta, self.config.activity_edges_mv)
        region_ids = np.asarray(self.layout.segment_region_ids, dtype=np.int64)
        region_count = len(self.layout.region_names)
        activity_count = len(self.config.activity_edges_mv) + 1
        activity_counts = np.bincount(activity.reshape(-1), minlength=activity_count)
        activity_weights = activity_counts.sum() / np.maximum(
            activity_count * activity_counts, 1
        )
        activity_weights = np.minimum(
            activity_weights, self.config.inverse_frequency_weight_cap
        )
        activity_weights /= np.average(activity_weights, weights=activity_counts)
        joint_counts = np.zeros((activity_count, region_count), dtype=np.int64)
        for activity_id in range(activity_count):
            per_segment = np.sum(activity == activity_id, axis=(0, 1))
            for region_id in range(region_count):
                joint_counts[activity_id, region_id] = int(
                    np.sum(per_segment[region_ids == region_id])
                )
        nonzero = joint_counts > 0
        joint_weights = np.zeros_like(joint_counts, dtype=np.float64)
        joint_weights[nonzero] = joint_counts.sum() / (
            np.sum(nonzero) * joint_counts[nonzero]
        )
        joint_weights = np.minimum(
            joint_weights, self.config.inverse_frequency_weight_cap
        )
        joint_weights /= np.sum(joint_weights * joint_counts) / joint_counts.sum()
        self.activity_weights = activity_weights.astype(np.float32)
        self.joint_weights = joint_weights.astype(np.float32)
        report = {
            "valid": bool(np.all(np.isfinite(activity_weights)) and np.all(np.isfinite(joint_weights))),
            "fit_role_only": True,
            "activity_edges_mv": list(self.config.activity_edges_mv),
            "activity_counts": activity_counts.tolist(),
            "activity_weights": activity_weights.tolist(),
            "region_names": list(self.layout.region_names),
            "activity_region_counts": joint_counts.tolist(),
            "activity_region_weights": joint_weights.tolist(),
            "weight_cap": self.config.inverse_frequency_weight_cap,
        }
        self.weight_contract = report
        return report

    def prepare_voltage_objective_recalibration(self) -> Dict[str, Any]:
        base = self.prepare_frozen_voltage_forensic()
        _, source = verified_06bh_artifact_root(
            self.artifact_06bh_source,
            self.output_dir.parent / ".06bi_artifact_cache" / "06bh",
        )
        self._materialize_window_role(
            "fit",
            self.config.objective_fit_window_count,
            self.config.objective_unroll_horizon_ms,
        )
        weights = self._fit_weight_contract()
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        probe = CausalVoltageGain(
            region_count=len(self.layout.region_names),
            region_width=self.config.gain_region_embedding_width,
            hidden_width=self.config.gain_hidden_width,
            minimum=self.config.gain_minimum,
            maximum=self.config.gain_maximum,
            initial=self.config.gain_initial,
        ).to(device)
        gain_parameters = int(sum(parameter.numel() for parameter in probe.parameters()))
        if gain_parameters > self.config.maximum_gain_parameter_count:
            raise RuntimeError("06b-i causal gain exceeds its parameter ceiling")
        report = {
            **base,
            "schema_version": "06b-i-objective-contract-v1",
            "experiment": "voltage_objective_recalibration_playground",
            "source_06bh": source,
            "frozen_reference_arm": FROZEN_REFERENCE_ARM,
            "trainable_objective_arms": dict(TRAINABLE_OBJECTIVE_ARMS),
            "primary_arm": PRIMARY_ARM,
            "fallback_arm": FALLBACK_ARM,
            "fixed_endpoint_step": self.config.objective_training_steps,
            "checkpoints": list(self.config.objective_checkpoints),
            "same_source_checkpoint_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "mechanism_STATE_updater_frozen": True,
            "causal_gain_parameter_count": gain_parameters,
            "causal_gain_parameter_ceiling": self.config.maximum_gain_parameter_count,
            "future_targets_used_only_in_training_loss": True,
            "inference_gain_inputs": ["raw_predicted_delta", "current_voltage", "region_id"],
            "weight_contract": weights,
            "role_reuse": {
                "fit_components": "historically used train fit components 1-3",
                "calibration_component": "historically used train component 4",
                "development_component": "historically used train component 5",
                "new_independent_confirmation_claimed": False,
            },
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
        }
        for stale in (
            "candidate_selection_role",
            "decision_role",
            "audit_used_for_selection",
            "neural_training_performed",
        ):
            report.pop(stale, None)
        report["neural_training_planned"] = True
        atomic._write_json(self.output_dir / "voltage_objective_contract.json", report)
        return report

    def _new_objective_arm(self, arm: str, seed: int, device: Any) -> Tuple[Any, Any, Any]:
        source = self.source_models[("full_feedback_scalar", seed)]
        bridge = copy.deepcopy(source[0]).to(device)
        state = copy.deepcopy(source[1]).to(device)
        state.eval()
        for parameter in state.parameters():
            parameter.requires_grad_(False)
        if arm in (
            "raw_bridge_control",
            "activity_balanced_bridge",
            "activity_region_balanced_bridge",
        ):
            bridge.train()
            for parameter in bridge.parameters():
                parameter.requires_grad_(True)
            calibrator = None
        else:
            bridge.eval()
            for parameter in bridge.parameters():
                parameter.requires_grad_(False)
            if arm == "global_gain_balanced":
                calibrator = GlobalVoltageGain(
                    self.config.gain_minimum,
                    self.config.gain_maximum,
                    self.config.gain_initial,
                ).to(device)
            elif arm == "causal_gain_balanced":
                calibrator = CausalVoltageGain(
                    region_count=len(self.layout.region_names),
                    region_width=self.config.gain_region_embedding_width,
                    hidden_width=self.config.gain_hidden_width,
                    minimum=self.config.gain_minimum,
                    maximum=self.config.gain_maximum,
                    initial=self.config.gain_initial,
                ).to(device)
            else:
                raise ValueError(arm)
            calibrator.train()
        return bridge, state, calibrator

    def _gain(
        self, arm: str, calibrator: Any, raw_delta: Any, voltage: Any
    ) -> Any:
        if arm == FROZEN_REFERENCE_ARM:
            return atomic.torch.full_like(raw_delta, self.config.gain_initial)
        if calibrator is None:
            return atomic.torch.ones_like(raw_delta)
        region_ids = atomic.torch.as_tensor(
            self.layout.segment_region_ids,
            dtype=atomic.torch.long,
            device=raw_delta.device,
        )
        return calibrator(raw_delta, voltage, region_ids)

    def _loss_weight(self, arm: str, target_delta: Any) -> Any:
        if arm == "raw_bridge_control":
            return 1.0 + self.config.bridge_active_weight * (
                target_delta.abs() >= self.config.bridge_active_delta_threshold_mv
            ).float()
        edges = atomic.torch.as_tensor(
            self.config.activity_edges_mv,
            dtype=target_delta.dtype,
            device=target_delta.device,
        )
        activity = atomic.torch.bucketize(target_delta.abs(), edges)
        if arm == "activity_region_balanced_bridge":
            region_ids = atomic.torch.as_tensor(
                self.layout.segment_region_ids,
                dtype=atomic.torch.long,
                device=target_delta.device,
            )
            weights = atomic.torch.as_tensor(
                self.joint_weights,
                dtype=target_delta.dtype,
                device=target_delta.device,
            )
            return weights[activity, region_ids[None, :]]
        weights = atomic.torch.as_tensor(
            self.activity_weights,
            dtype=target_delta.dtype,
            device=target_delta.device,
        )
        return weights[activity]

    def _objective_unroll(
        self,
        model: Tuple[Any, Any, Any],
        arm: str,
        batch: Mapping[str, Any],
        *,
        collect: bool,
    ) -> Tuple[Any, Dict[str, Any]]:
        bridge, state_model, calibrator = model
        state_center = atomic.torch.as_tensor(
            self.statistics["state_center"], device=batch["state_t"].device
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], device=batch["state_t"].device
        )
        delta_scale = atomic.torch.as_tensor(
            self.statistics["delta_scale"], device=batch["state_t"].device
        )
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        losses = []
        outputs = {}
        for step in range(self.config.objective_unroll_horizon_ms):
            context = atomic.torch.cat(
                (batch["drive"][:, step], batch["held_ions"]), dim=-1
            )
            normalized = (current_state - state_center) / state_scale
            raw_delta = self._bridge_forward(
                bridge, normalized, current_voltage, context
            )
            gain = self._gain(arm, calibrator, raw_delta, current_voltage)
            voltage_delta = raw_delta * gain
            next_voltage = current_voltage + voltage_delta
            state_delta = self._state_forward(
                state_model, normalized, current_voltage, voltage_delta, context
            )
            next_state = current_state + state_delta * delta_scale
            target_delta = batch["voltage_t1"][:, step] - batch["voltage_t"][:, step]
            voltage_error = (
                next_voltage - batch["voltage_t1"][:, step]
            ) / self.config.bridge_voltage_scale_mv
            point = atomic.torch_functional.smooth_l1_loss(
                voltage_error,
                atomic.torch.zeros_like(voltage_error),
                reduction="none",
            )
            weighted = atomic.torch.mean(self._loss_weight(arm, target_delta) * point)
            high = atomic.torch.relu(
                (next_voltage - self.config.physical_voltage_maximum_mv)
                / self.config.bridge_voltage_scale_mv
            )
            low = atomic.torch.relu(
                (self.config.physical_voltage_minimum_mv - next_voltage)
                / self.config.bridge_voltage_scale_mv
            )
            physical = atomic.torch.mean(high * high + low * low)
            drift = atomic.torch.mean(atomic.torch.mean(voltage_error, dim=1) ** 2)
            losses.append(
                weighted
                + self.config.repair_physical_penalty_weight * physical
                + self.config.repair_drift_penalty_weight * drift
            )
            if collect and step + 1 in self.config.rollout_horizons_ms:
                outputs[f"{step + 1}_ms"] = {
                    "state": next_state,
                    "voltage": next_voltage,
                    "gain": gain,
                }
            current_state, current_voltage = next_state, next_voltage
        return atomic.torch.stack(losses).mean(), outputs

    def _evaluate_objective_arm(
        self, model: Tuple[Any, Any, Any], arm: str, role: str, device: Any
    ) -> Dict[str, Any]:
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        previous_modes = []
        for component in model:
            if component is not None:
                previous_modes.append((component, bool(component.training)))
                component.eval()
        with atomic.torch.no_grad():
            _, outputs = self._objective_unroll(model, arm, batch, collect=True)
        result = {"horizons": {}, "gain": {}}
        for horizon, output in outputs.items():
            step = int(horizon[:-3]) - 1
            result["horizons"][horizon] = self._metric(
                output["state"], output["voltage"], batch, step
            )
            gain = output["gain"].detach().cpu().numpy()
            result["gain"][horizon] = {
                "minimum": float(np.min(gain)),
                "median": float(np.median(gain)),
                "maximum": float(np.max(gain)),
                "standard_deviation": float(np.std(gain)),
            }
        endpoint = outputs["8_ms"]
        row = {
            "voltage": endpoint["voltage"].detach().cpu().numpy(),
            "target_voltage": batch["voltage_t1"][:, 7].cpu().numpy(),
            "initial_voltage": batch["voltage_t"][:, 0].cpu().numpy(),
        }
        result["activity_at_8ms"] = self._activity_metrics(row)
        result["region_at_8ms"] = self._region_metrics(row)
        for component, was_training in previous_modes:
            component.train(was_training)
        return result

    def _save_objective_checkpoint(
        self,
        arm: str,
        seed: int,
        budget: int,
        model: Tuple[Any, Any, Any],
        device: Any,
    ) -> Dict[str, Any]:
        bridge, _, calibrator = model
        payload = {
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "source_arm": "full_feedback_scalar_step600",
            "STATE_updater_source": "full_feedback_scalar_step600_frozen",
            "bridge_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in bridge.state_dict().items()
            },
            "calibrator_state_dict": None
            if calibrator is None
            else {
                name: value.detach().cpu().clone()
                for name, value in calibrator.state_dict().items()
            },
            "configuration": asdict(self.config),
        }
        self.objective_states[(arm, seed, budget)] = payload
        path = self.output_dir / f"objective_{arm}_seed{seed}_step{budget}.pt"
        atomic.torch.save(payload, path)
        calibration = self._evaluate_objective_arm(model, arm, "calibration", device)
        return {
            "budget": budget,
            "checkpoint": path.name,
            "checkpoint_sha256": atomic._sha256_file(path),
            "calibration": calibration,
        }

    def train_synchronized_objective_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        reports = {}
        for seed in self.config.pilot_seeds:
            models = {
                arm: self._new_objective_arm(arm, seed, device)
                for arm in TRAINABLE_OBJECTIVE_ARMS
            }
            optimizers = {}
            for arm, (bridge, _, calibrator) in models.items():
                parameters = (
                    [parameter for parameter in bridge.parameters() if parameter.requires_grad]
                    + ([] if calibrator is None else list(calibrator.parameters()))
                )
                optimizers[arm] = atomic.torch.optim.AdamW(
                    parameters,
                    lr=self.config.objective_learning_rate,
                    weight_decay=self.config.objective_weight_decay,
                )
            seed_report = {arm: [] for arm in TRAINABLE_OBJECTIVE_ARMS}
            for arm, model in models.items():
                seed_report[arm].append(
                    self._save_objective_checkpoint(arm, seed, 0, model, device)
                )
            rng = np.random.default_rng(seed + 680000)
            progress = atomic._CompactProgress(
                f"06b-i synchronized objective seed={seed}",
                self.config.objective_training_steps,
                max(1, self.config.objective_training_steps // 20),
            )
            for step in range(1, self.config.objective_training_steps + 1):
                rows = rng.choice(
                    len(self.window_data["fit"]["indices"]),
                    size=self.config.objective_batch_window_count,
                    replace=False,
                )
                batch = self._batch_tensors("fit", rows, device)
                losses = []
                for arm, model in models.items():
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._objective_unroll(model, arm, batch, collect=False)
                    loss.backward()
                    parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ]
                    atomic.torch.nn.utils.clip_grad_norm_(
                        parameters, self.config.objective_gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                if step in self.config.objective_checkpoints:
                    for arm, model in models.items():
                        seed_report[arm].append(
                            self._save_objective_checkpoint(
                                arm, seed, step, model, device
                            )
                        )
                progress.update(step, f"median_loss={float(np.median(losses)):.4g}")
            for arm, model in models.items():
                for component in model:
                    if component is not None:
                        component.eval()
                self.objective_models[(arm, seed)] = model
            reports[str(seed)] = seed_report
            atomic._write_json(
                self.output_dir / f"objective_training_seed{seed}.json", seed_report
            )
        report = {
            "schema_version": "06b-i-training-v1",
            "valid": all(
                len(rows[arm]) == len(self.config.objective_checkpoints)
                for rows in reports.values()
                for arm in TRAINABLE_OBJECTIVE_ARMS
            ),
            "device": str(device),
            "same_source_checkpoint_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "mechanism_STATE_updater_frozen": True,
            "training_roles": ["fit"],
            "calibration_used_for_monitoring_only": True,
            "development_used_during_training": False,
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "objective_training_report.json", report)
        return report

    def _frozen_reference(self, seed: int, device: Any) -> Tuple[Any, Any, Any]:
        source = self.source_models[("full_feedback_scalar", seed)]
        bridge = copy.deepcopy(source[0]).to(device).eval()
        state = copy.deepcopy(source[1]).to(device).eval()
        for component in (bridge, state):
            for parameter in component.parameters():
                parameter.requires_grad_(False)
        return bridge, state, None

    def evaluate_objective_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed = {}
        for seed in self.config.pilot_seeds:
            per_seed[str(seed)] = {
                FROZEN_REFERENCE_ARM: self._evaluate_objective_arm(
                    self._frozen_reference(seed, device),
                    FROZEN_REFERENCE_ARM,
                    "development",
                    device,
                )
            }
            for arm in TRAINABLE_OBJECTIVE_ARMS:
                per_seed[str(seed)][arm] = self._evaluate_objective_arm(
                    self.objective_models[(arm, seed)], arm, "development", device
                )
        report = {
            "schema_version": "06b-i-development-evaluation-v1",
            "valid": all(
                row[arm]["horizons"]["8_ms"]["nonfinite_voltage_count"] == 0
                for row in per_seed.values()
                for arm in (FROZEN_REFERENCE_ARM, *TRAINABLE_OBJECTIVE_ARMS)
            ),
            "role": "historically_reused_train_development",
            "new_independent_confirmation_claimed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "objective_development_evaluation.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _arm_summary(
        self,
        arm: str,
        evaluation: Mapping[str, Any],
        training: Mapping[str, Any],
    ) -> Dict[str, Any]:
        rows = [seed[arm] for seed in evaluation["per_seed"].values()]
        endpoint = [row["horizons"]["8_ms"] for row in rows]
        baseline = [
            seed[FROZEN_REFERENCE_ARM]["horizons"]["8_ms"]
            for seed in evaluation["per_seed"].values()
        ]
        voltage_gains = [row["voltage_improvement_vs_persistence_fraction"] for row in endpoint]
        state_gains = [row["state_improvement_vs_persistence_fraction"] for row in endpoint]
        gain_over_frozen = [
            1.0 - row["voltage_rmse_mv"] / max(reference["voltage_rmse_mv"], 1e-12)
            for row, reference in zip(endpoint, baseline)
        ]
        activity = {}
        for name in rows[0]["activity_at_8ms"]:
            activity[name] = self._median(
                [row["activity_at_8ms"][name]["voltage_gain_vs_persistence_fraction"] for row in rows]
            )
        region = {}
        for name in rows[0]["region_at_8ms"]:
            region[name] = self._median(
                [row["region_at_8ms"][name]["voltage_gain_vs_persistence_fraction"] for row in rows]
            )
        if arm == FROZEN_REFERENCE_ARM:
            training_effect = 0.0
        else:
            start = []
            final = []
            for seed_rows in training["reports"].values():
                records = seed_rows[arm]
                start.append(records[0]["calibration"]["horizons"]["8_ms"]["voltage_rmse_mv"])
                final.append(records[-1]["calibration"]["horizons"]["8_ms"]["voltage_rmse_mv"])
            training_effect = self._median(
                [1.0 - end / max(begin, 1e-12) for begin, end in zip(start, final)]
            )
        passes = bool(
            self._median(gain_over_frozen) >= self.config.minimum_gain_over_frozen_alpha_fraction
            and all(value > 0 for value in voltage_gains)
            and all(value > self.config.minimum_STATE_gain_fraction for value in state_gains)
            and activity["active_ge_5mV"] >= self.config.minimum_active_gain_fraction
            and activity["moderate_1_to_5mV"] >= self.config.minimum_moderate_gain_fraction
            and activity["quiescent_lt_1mV"] >= self.config.minimum_quiescent_gain_fraction
            and region["soma"] >= self.config.minimum_soma_gain_fraction
            and sum(row["physical_voltage_violation_count"] for row in endpoint) == 0
            and training_effect >= self.config.minimum_training_effect_fraction
        )
        return {
            "median_voltage_gain_vs_persistence_fraction": self._median(voltage_gains),
            "median_STATE_gain_vs_persistence_fraction": self._median(state_gains),
            "median_voltage_gain_over_frozen_alpha_fraction": self._median(gain_over_frozen),
            "median_training_effect_fraction": training_effect,
            "activity_gain_vs_persistence": activity,
            "region_gain_vs_persistence": region,
            "physical_voltage_violation_count": int(
                sum(row["physical_voltage_violation_count"] for row in endpoint)
            ),
            "all_seed_voltage_gain_positive": all(value > 0 for value in voltage_gains),
            "all_seed_STATE_gain_positive": all(value > 0 for value in state_gains),
            "registered_gate_passed": passes,
        }

    def finalize_voltage_objective_recalibration(
        self, training: Mapping[str, Any], evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        summaries = {
            arm: self._arm_summary(arm, evaluation, training)
            for arm in (FROZEN_REFERENCE_ARM, *TRAINABLE_OBJECTIVE_ARMS)
        }
        primary_pass = summaries[PRIMARY_ARM]["registered_gate_passed"]
        fallback_pass = summaries[FALLBACK_ARM]["registered_gate_passed"]
        diagnostic_passes = [
            arm
            for arm in TRAINABLE_OBJECTIVE_ARMS
            if arm not in (PRIMARY_ARM, FALLBACK_ARM)
            and summaries[arm]["registered_gate_passed"]
        ]
        if primary_pass:
            diagnosis = "CAUSAL_GAIN_OBJECTIVE_LEARNABLE_ON_REUSED_TRAIN_ROLES"
            selected = PRIMARY_ARM
            next_step = "fresh_train_support_causal_gain_confirmation"
        elif fallback_pass:
            diagnosis = "ACTIVITY_REGION_OBJECTIVE_LEARNABLE_ON_REUSED_TRAIN_ROLES"
            selected = FALLBACK_ARM
            next_step = "fresh_train_support_activity_region_confirmation"
        elif diagnostic_passes:
            diagnosis = "SECONDARY_VOLTAGE_OBJECTIVE_SIGNAL_REQUIRES_PREREGISTERED_CONFIRMATION"
            selected = None
            next_step = "preregister_secondary_objective_confirmation"
        else:
            diagnosis = "VOLTAGE_OBJECTIVE_RECALIBRATION_NOT_LEARNABLE"
            selected = None
            next_step = "atomic_voltage_bridge_representation_revision"
        report = {
            "schema_version": "06b-i-final-report-v1",
            "valid": bool(training.get("valid") and evaluation.get("valid")),
            "component_playground_grade": True,
            "new_independent_confirmation_claimed": False,
            "diagnosis": diagnosis,
            "primary_arm": PRIMARY_ARM,
            "fallback_arm": FALLBACK_ARM,
            "primary_passed": primary_pass,
            "fallback_passed": fallback_pass,
            "diagnostic_passing_arms": diagnostic_passes,
            "selected_candidate": selected,
            "summaries": summaries,
            "mechanism_STATE_updater_frozen": True,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
            "fresh_train_support_confirmation_authorized": bool(primary_pass or fallback_pass),
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
    "CausalVoltageGain",
    "EXPECTED_06BH_ARCHIVE_SHA256",
    "EXPECTED_06BH_FINAL_SHA256",
    "EXPECTED_06BH_INDEX_SHA256",
    "FALLBACK_ARM",
    "FROZEN_REFERENCE_ARM",
    "GlobalVoltageGain",
    "PRIMARY_ARM",
    "TRAINABLE_OBJECTIVE_ARMS",
    "VoltageObjectiveRecalibrationConfig",
    "VoltageObjectiveRecalibrationPlayground",
    "verified_06bh_artifact_root",
]
