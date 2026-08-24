"""06b-f: synchronized train-only repair of recurrent STATE and voltage.

All arms restore the same exact 06b-d bridge and 06b mechanism-STATE updater.
They cross teacher/predicted feedback during a four-ms differentiable unroll,
plus a shuffled causal control and a voltage-protected gradient-routing arm.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .recursive_voltage_state_contract_forensic import (
    RecursiveVoltageStateContractConfig,
    RecursiveVoltageStateContractForensic,
)


ACCEPTED_06BE_ARTIFACTS = {
    "acc8b29f4eacd9c209e6ca4e622da5fa5527d500b30ccaefdfc2580f721fa2ad": {
        "archive_sha256": "9239a7ae329571903a165226c6cb7098fbfb8daf5ca3578262b0bf945baa9cdc",
        "final_report_sha256": "37a38f215caa1f996fdd81eac2901f6860b9744e01a43e2042ef438bb594f928",
        "role": "canonical",
    },
    "dc43155bb0065768f473b41cfd0f7fc3bbe40d00052953a2786e27bf5da0a3ac": {
        "archive_sha256": "820aba669e730030a3408fa0ad3dc3b4e738e1c013d120290d8a04e1492d1954",
        "final_report_sha256": "ec58bbbe14aba877abaa1ac43b67cc1675e7017e32a65b40c265b6702b84d905",
        "role": "confirmatory_exact_replication",
    },
}
EXPECTED_06BE_INDEX_SHA256 = tuple(ACCEPTED_06BE_ARTIFACTS)

REPAIR_ARMS: Dict[str, Tuple[bool, bool, str]] = {
    "teacherV_teacherS": (True, True, "scalar"),
    "teacherV_predictedS": (True, False, "scalar"),
    "predictedV_teacherS": (False, True, "scalar"),
    "full_feedback_scalar": (False, False, "scalar"),
    "full_feedback_voltage_protected": (False, False, "voltage_protected"),
    "full_feedback_shuffled": (False, False, "shuffled"),
}


def verified_06be_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Accept only either independently registered exact 06b-e execution."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-e source must be a ZIP or extracted directory")
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
        archive_hash = "extracted-directory"
        search_root = source
    matches = []
    for path in search_root.rglob("artifact_index.json"):
        digest = atomic._sha256_file(path)
        if digest in ACCEPTED_06BE_ARTIFACTS:
            matches.append((path.parent, digest))
    if len(matches) != 1:
        raise RuntimeError(f"expected one registered 06b-e artifact; found {len(matches)}")
    root, index_hash = matches[0]
    expected = ACCEPTED_06BE_ARTIFACTS[index_hash]
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
        raise RuntimeError(f"06b-e indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != expected["final_report_sha256"]:
        raise RuntimeError("06b-e final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "MECHANISM_STATE_EXPOSURE_PRIMARY_LIMIT"
        or final.get("bounded_train_only_repair_matrix_authorized") is not True
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-e source does not authorize bounded repair")
    if source.is_file() and archive_hash != expected["archive_sha256"]:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "source_role": expected["role"],
        "archive_sha256": archive_hash,
        "artifact_index_sha256": index_hash,
        "final_report_sha256": expected["final_report_sha256"],
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class RecursiveJointRepairConfig(RecursiveVoltageStateContractConfig):
    repair_training_steps: int = 600
    repair_checkpoints: Tuple[int, ...] = (0, 200, 400, 600)
    repair_unroll_horizon_ms: int = 4
    repair_batch_window_count: int = 4
    repair_fit_window_count: int = 64
    repair_calibration_window_count: int = 16
    repair_development_window_count: int = 16
    repair_learning_rate: float = 0.0003
    repair_weight_decay: float = 0.00001
    repair_gradient_clip_norm: float = 1.0
    repair_voltage_loss_weight: float = 1.0
    repair_state_loss_weight: float = 1.0
    repair_physical_penalty_weight: float = 1.0
    repair_drift_penalty_weight: float = 0.1
    minimum_state_error_reduction_fraction: float = 0.02
    minimum_voltage_error_reduction_fraction: float = 0.10
    minimum_scaling_error_reduction_fraction: float = 0.02
    maximum_one_step_degradation_fraction: float = 0.02
    minimum_shuffled_specificity_fraction: float = 0.01

    def validate(self) -> None:
        super().validate()
        if self.repair_training_steps <= 0 or self.repair_unroll_horizon_ms <= 1:
            raise ValueError("06b-f repair dimensions are invalid")
        checkpoints = tuple(map(int, self.repair_checkpoints))
        if (
            checkpoints[0] != 0
            or checkpoints[-1] != self.repair_training_steps
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError("06b-f checkpoints must span one unique trajectory")
        if self.repair_unroll_horizon_ms > max(self.rollout_horizons_ms):
            raise ValueError("06b-f training horizon exceeds evaluation horizon")
        if min(
            self.repair_batch_window_count,
            self.repair_fit_window_count,
            self.repair_calibration_window_count,
            self.repair_development_window_count,
        ) <= 0:
            raise ValueError("06b-f window counts must be positive")
        if self.repair_batch_window_count > self.repair_fit_window_count:
            raise ValueError("06b-f batch cannot exceed the unique fit-window pool")
        positive = (
            self.repair_learning_rate,
            self.repair_gradient_clip_norm,
            self.repair_voltage_loss_weight,
            self.repair_state_loss_weight,
            self.repair_physical_penalty_weight,
            self.minimum_state_error_reduction_fraction,
            self.minimum_voltage_error_reduction_fraction,
            self.minimum_scaling_error_reduction_fraction,
            self.maximum_one_step_degradation_fraction,
            self.minimum_shuffled_specificity_fraction,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("06b-f positive configuration value is invalid")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RecursiveJointRepairConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
            "repair_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class RecursiveJointRepairMatrix(RecursiveVoltageStateContractForensic):
    """Train synchronized recurrent-feedback repairs on train-only windows."""

    config: RecursiveJointRepairConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: RecursiveJointRepairConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
        artifact_06bd_source: Path,
        artifact_06be_source: Path,
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
            code_revision=code_revision,
        )
        self.artifact_06be_source = Path(artifact_06be_source)
        self.repair_models: Dict[Tuple[str, int], Tuple[Any, Any]] = {}
        self.repair_states: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self.window_data: Dict[str, Dict[str, np.ndarray]] = {}

    def _windows_for_role(self, role: str, count: int, horizon: int) -> List[np.ndarray]:
        allowed = {str(row["trajectory_id"]) for row in self.roles[role]}
        windows: List[np.ndarray] = []
        for trajectory in sorted(allowed):
            indices = self.store.trajectory_indices[trajectory]
            for start in range(max(0, len(indices) - horizon + 1)):
                candidate = indices[start : start + horizon]
                steps = self.store.metadata["step_index"][candidate]
                if np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                    windows.append(candidate)
        windows.sort(
            key=lambda row: hashlib.sha256(
                f"{self.config.role_seed}|06bf|{role}|{','.join(map(str,row))}".encode()
            ).hexdigest()
        )
        return windows[:count]

    def _materialize_window_role(self, role: str, count: int, horizon: int) -> Dict[str, np.ndarray]:
        windows = self._windows_for_role(role, count, horizon)
        if len(windows) < count:
            raise RuntimeError(f"06b-f found only {len(windows)} {role} windows")
        index = np.asarray(windows, dtype=np.int64)
        flat = index.reshape(-1)
        shape_state = (len(windows), horizon, -1)
        shape_voltage = (len(windows), horizon, -1)
        state_t = atomic.mechanism_logit(
            self.store.read_state(flat, "t", categories=("mechanism_states",))
        ).astype(np.float32).reshape(shape_state)
        state_t1 = atomic.mechanism_logit(
            self.store.read_state(flat, "t_plus_1", categories=("mechanism_states",))
        ).astype(np.float32).reshape(shape_state)
        voltage_t = self.store.read_state(flat, "t", categories=("voltage",)).astype(
            np.float32
        ).reshape(shape_voltage)
        voltage_t1 = self.store.read_state(
            flat, "t_plus_1", categories=("voltage",)
        ).astype(np.float32).reshape(shape_voltage)
        drive = atomic.encode_causal_realized_drive(self.store, flat).astype(np.float32)
        drive = drive.reshape(len(windows), horizon, self.layout.segment_count, -1)
        ions = self._ion_context(index[:, 0]).astype(np.float32)
        payload = {
            "indices": index,
            "state_t": state_t,
            "state_t1": state_t1,
            "voltage_t": voltage_t,
            "voltage_t1": voltage_t1,
            "drive": drive,
            "held_ions": ions,
        }
        self.window_data[role] = payload
        return payload

    def prepare_recursive_joint_repair(self) -> Dict[str, Any]:
        base = self.prepare_recursive_contract_forensic()
        _, source = verified_06be_artifact_root(
            self.artifact_06be_source,
            self.output_dir.parent / ".06bf_artifact_cache" / "06be",
        )
        self._materialize_window_role(
            "fit", self.config.repair_fit_window_count, self.config.repair_unroll_horizon_ms
        )
        self._materialize_window_role(
            "calibration",
            self.config.repair_calibration_window_count,
            max(self.config.rollout_horizons_ms),
        )
        self._materialize_window_role(
            "development",
            self.config.repair_development_window_count,
            max(self.config.rollout_horizons_ms),
        )
        bridge_count = int(
            sum(
                value.numel()
                for value in self.contract_models[("joint_cosine", self.config.pilot_seeds[0])].parameters()
            )
        )
        state_count = int(
            sum(
                value.numel()
                for value in self.frozen_state_models[("linear_endpoint_path", self.config.pilot_seeds[0])].parameters()
            )
        )
        report = {
            **base,
            "schema_version": "06b-f-joint-repair-contract-v1",
            "experiment": "recursive_joint_repair_matrix",
            "source_06be": source,
            "repair_arms": {
                name: {
                    "teacher_voltage_feedback_during_training": flags[0],
                    "teacher_STATE_feedback_during_training": flags[1],
                    "gradient_routing": flags[2],
                }
                for name, flags in REPAIR_ARMS.items()
            },
            "trainable_bridge_parameter_count": bridge_count,
            "trainable_STATE_parameter_count": state_count,
            "trainable_parameter_count_per_arm": bridge_count + state_count,
            "common_initial_bridge_arm": "joint_cosine_step1500",
            "common_initial_STATE_arm": "linear_endpoint_path",
            "training_horizon_ms": self.config.repair_unroll_horizon_ms,
            "evaluation_horizons_ms": list(self.config.rollout_horizons_ms),
            "repair_checkpoints": list(self.config.repair_checkpoints),
            "same_initialization_within_seed": True,
            "same_window_stream_within_seed": True,
            "same_targets_and_optimizer": True,
            "held_initial_ions_are_causal": True,
            "realized_external_input_is_step_specific": True,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "full_training_authorized": False,
        }
        atomic._write_json(self.output_dir / "joint_repair_contract.json", report)
        return report

    def _new_pair(self, seed: int, device: Any) -> Tuple[Any, Any]:
        bridge = self._new_bridge(device)
        bridge.load_state_dict(
            copy.deepcopy(self.contract_models[("joint_cosine", seed)].state_dict())
        )
        state = self._new_capacity_capped_model(device)
        state.load_state_dict(
            copy.deepcopy(
                self.frozen_state_models[("linear_endpoint_path", seed)].state_dict()
            )
        )
        bridge.train()
        state.train()
        for model in (bridge, state):
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        return bridge, state

    def _batch_tensors(
        self, role: str, rows: np.ndarray, device: Any
    ) -> Dict[str, Any]:
        source = self.window_data[role]
        return {
            name: atomic.torch.as_tensor(value[rows], device=device)
            for name, value in source.items()
            if name != "indices"
        }

    def _segment_state_tensor(self, normalized: Any) -> Any:
        lookup = atomic.torch.as_tensor(
            self.coordinate_lookup, dtype=atomic.torch.long, device=normalized.device
        )
        mask = lookup >= 0
        gathered = normalized[:, lookup.clamp_min(0)]
        return gathered * mask[None, :, :]

    def _axial_tensor(self, voltage: Any) -> Any:
        parent = atomic.torch.as_tensor(
            self.layout.parent_ids, dtype=atomic.torch.long, device=voltage.device
        )
        children = atomic.torch.as_tensor(
            self.child_ids, dtype=atomic.torch.long, device=voltage.device
        )
        mask = atomic.torch.as_tensor(
            self.child_mask, dtype=voltage.dtype, device=voltage.device
        )
        parent_delta = voltage[:, parent] - voltage
        child_delta = ((voltage[:, children] - voltage[:, :, None]) * mask).sum(-1)
        child_delta = child_delta / mask.sum(-1).clamp_min(1.0)[None, :]
        return atomic.torch.stack(
            (voltage / 100.0, parent_delta / 100.0, child_delta / 100.0), dim=-1
        )

    def _bridge_forward(self, model: Any, normalized_state: Any, voltage: Any, context: Any) -> Any:
        batch = voltage.shape[0]
        segment_count = self.layout.segment_count
        device = voltage.device
        state = self._segment_state_tensor(normalized_state)
        presence = atomic.torch.as_tensor(
            self.semantic_presence, dtype=voltage.dtype, device=device
        )[None, :, :].expand(batch, -1, -1)
        static = atomic.torch.as_tensor(
            self.layout.segment_static, dtype=voltage.dtype, device=device
        )[None, :, :].expand(batch, -1, -1)
        region = atomic.torch.as_tensor(
            self.layout.segment_region_ids, dtype=atomic.torch.long, device=device
        )[None, :].expand(batch, -1)
        prediction = model(
            self._axial_tensor(voltage).reshape(batch * segment_count, -1),
            state.reshape(batch * segment_count, -1),
            presence.reshape(batch * segment_count, -1),
            context.reshape(batch * segment_count, -1),
            static.reshape(batch * segment_count, -1),
            region.reshape(-1),
        )
        return prediction.reshape(batch, segment_count) * self.config.bridge_voltage_scale_mv

    def _state_forward(
        self, model: Any, normalized_state: Any, voltage: Any, voltage_delta: Any, context: Any
    ) -> Any:
        batch, coordinate_count = normalized_state.shape
        device = normalized_state.device
        segments = atomic.torch.as_tensor(
            self.coordinate["segment"], dtype=atomic.torch.long, device=device
        )
        fractions = atomic.torch.as_tensor(
            np.asarray(self.config.voltage_path_sample_indices, dtype=np.float32)
            / float(self.config.expected_microtrace_sample_count - 1),
            device=device,
        )
        path = voltage_delta[:, segments, None] * fractions[None, None, :]
        static = atomic.torch.as_tensor(
            self.layout.segment_static[self.coordinate["segment"]],
            dtype=normalized_state.dtype,
            device=device,
        )[None, :, :].expand(batch, -1, -1)
        ids = lambda name: atomic.torch.as_tensor(
            self.coordinate[name], dtype=atomic.torch.long, device=device
        )[None, :].expand(batch, -1).reshape(-1)
        prediction = model(
            normalized_state.reshape(-1),
            voltage[:, segments].reshape(-1),
            path.reshape(batch * coordinate_count, -1),
            context[:, segments].reshape(batch * coordinate_count, -1),
            static.reshape(batch * coordinate_count, -1),
            ids("mechanism"),
            ids("variable"),
            ids("kind"),
            ids("region"),
        )
        return prediction.reshape(batch, coordinate_count)

    def _unroll_objectives(
        self,
        bridge: Any,
        state_model: Any,
        batch: Mapping[str, Any],
        arm: str,
        *,
        horizon: int,
        collect: bool,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        teacher_voltage, teacher_state, routing = REPAIR_ARMS[arm]
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        voltage_losses = []
        state_losses = []
        outputs: Dict[str, Any] = {}
        state_center = atomic.torch.as_tensor(
            self.statistics["state_center"], device=current_state.device
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], device=current_state.device
        )
        delta_scale = atomic.torch.as_tensor(
            self.statistics["delta_scale"], device=current_state.device
        )
        for step in range(horizon):
            state_input = batch["state_t"][:, step] if teacher_state else current_state
            voltage_input = (
                batch["voltage_t"][:, step] if teacher_voltage else current_voltage
            )
            context = atomic.torch.cat(
                (batch["drive"][:, step], batch["held_ions"]), dim=-1
            )
            normalized = (state_input - state_center) / state_scale
            voltage_delta = self._bridge_forward(
                bridge, normalized, voltage_input, context
            )
            next_voltage = voltage_input + voltage_delta
            state_path = (
                atomic.torch.roll(voltage_delta, shifts=1, dims=0)
                if routing == "shuffled"
                else voltage_delta
            )
            state_delta = self._state_forward(
                state_model, normalized, voltage_input, state_path, context
            )
            next_state = state_input + state_delta * delta_scale
            voltage_error = (
                next_voltage - batch["voltage_t1"][:, step]
            ) / self.config.bridge_voltage_scale_mv
            voltage_active = (
                (batch["voltage_t1"][:, step] - batch["voltage_t"][:, step]).abs()
                >= self.config.bridge_active_delta_threshold_mv
            ).float()
            voltage_loss = atomic.torch.mean(
                (1.0 + self.config.bridge_active_weight * voltage_active)
                * atomic.torch_functional.smooth_l1_loss(
                    voltage_error, atomic.torch.zeros_like(voltage_error), reduction="none"
                )
            )
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
            voltage_losses.append(
                self.config.repair_voltage_loss_weight * voltage_loss
                + self.config.repair_physical_penalty_weight * physical
                + self.config.repair_drift_penalty_weight * drift
            )
            state_error = (next_state - batch["state_t1"][:, step]) / state_scale
            target_delta = (
                batch["state_t1"][:, step] - batch["state_t"][:, step]
            ) / delta_scale
            state_active = (
                target_delta.abs() >= self.config.active_delta_threshold
            ).float()
            state_losses.append(
                self.config.repair_state_loss_weight
                * atomic.torch.mean(
                    (1.0 + self.config.active_delta_weight * state_active)
                    * atomic.torch_functional.smooth_l1_loss(
                        state_error,
                        atomic.torch.zeros_like(state_error),
                        reduction="none",
                    )
                )
            )
            current_voltage = next_voltage
            current_state = next_state
            if collect and step + 1 in self.config.rollout_horizons_ms:
                outputs[f"{step + 1}_ms"] = (next_state, next_voltage)
        return (
            atomic.torch.stack(voltage_losses).mean(),
            atomic.torch.stack(state_losses).mean(),
            outputs,
        )

    def _metric(self, predicted_state: Any, predicted_voltage: Any, batch: Mapping[str, Any], step: int) -> Dict[str, Any]:
        state = predicted_state.detach().cpu().numpy()
        voltage = predicted_voltage.detach().cpu().numpy()
        target_state = batch["state_t1"][:, step].detach().cpu().numpy()
        target_voltage = batch["voltage_t1"][:, step].detach().cpu().numpy()
        initial_state = batch["state_t"][:, 0].detach().cpu().numpy()
        initial_voltage = batch["voltage_t"][:, 0].detach().cpu().numpy()
        return self._horizon_metrics(
            state, target_state, initial_state, voltage, target_voltage, initial_voltage
        )

    def _evaluate_pair(self, bridge: Any, state_model: Any, role: str, device: Any) -> Dict[str, Any]:
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        bridge.eval()
        state_model.eval()
        with atomic.torch.no_grad():
            _, _, full = self._unroll_objectives(
                bridge,
                state_model,
                batch,
                "full_feedback_scalar",
                horizon=max(self.config.rollout_horizons_ms),
                collect=True,
            )
            _, _, teacher = self._unroll_objectives(
                bridge,
                state_model,
                batch,
                "teacherV_teacherS",
                horizon=max(self.config.rollout_horizons_ms),
                collect=True,
            )
        bridge.train()
        state_model.train()
        return {
            "full_feedback": {
                horizon: self._metric(pair[0], pair[1], batch, int(horizon[:-3]) - 1)
                for horizon, pair in full.items()
            },
            "teacher_boundary": {
                horizon: self._metric(pair[0], pair[1], batch, int(horizon[:-3]) - 1)
                for horizon, pair in teacher.items()
            },
        }

    def _checkpoint(self, arm: str, seed: int, budget: int, pair: Tuple[Any, Any], device: Any) -> Dict[str, Any]:
        bridge, state = pair
        payload = {
            "bridge_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in bridge.state_dict().items()
            },
            "STATE_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in state.state_dict().items()
            },
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "configuration": asdict(self.config),
        }
        self.repair_states[(arm, seed, budget)] = payload
        path = self.output_dir / f"repair_{arm}_seed{seed}_step{budget}.pt"
        atomic.torch.save(payload, path)
        calibration = self._evaluate_pair(bridge, state, "calibration", device)
        metric = calibration["full_feedback"][
            f"{max(self.config.rollout_horizons_ms)}_ms"
        ]
        return {
            "budget": budget,
            "checkpoint": path.name,
            "checkpoint_sha256": atomic._sha256_file(path),
            "calibration_full_feedback_eight_ms": metric,
        }

    def train_synchronized_repair_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        reports = {}
        for seed in self.config.pilot_seeds:
            pairs = {arm: self._new_pair(seed, device) for arm in REPAIR_ARMS}
            optimizers = {
                arm: atomic.torch.optim.AdamW(
                    list(pair[0].parameters()) + list(pair[1].parameters()),
                    lr=self.config.repair_learning_rate,
                    weight_decay=self.config.repair_weight_decay,
                )
                for arm, pair in pairs.items()
            }
            rng = np.random.default_rng(seed + 660000)
            seed_report = {arm: [] for arm in REPAIR_ARMS}
            for arm, pair in pairs.items():
                seed_report[arm].append(self._checkpoint(arm, seed, 0, pair, device))
            print(
                f"[HayFlow 06b-f][checkpoint seed={seed}] 0/"
                f"{self.config.repair_training_steps}: {len(REPAIR_ARMS)} arms",
                flush=True,
            )
            progress = atomic._CompactProgress(
                f"06b-f synchronized seed={seed}",
                self.config.repair_training_steps,
                max(1, self.config.repair_training_steps // 20),
            )
            for step in range(1, self.config.repair_training_steps + 1):
                rows = rng.choice(
                    len(self.window_data["fit"]["indices"]),
                    size=self.config.repair_batch_window_count,
                    replace=False,
                )
                batch = self._batch_tensors("fit", rows, device)
                losses = []
                for arm, pair in pairs.items():
                    bridge, state = pair
                    optimizer = optimizers[arm]
                    voltage_loss, state_loss, _ = self._unroll_objectives(
                        bridge,
                        state,
                        batch,
                        arm,
                        horizon=self.config.repair_unroll_horizon_ms,
                        collect=False,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if REPAIR_ARMS[arm][2] == "voltage_protected":
                        state_loss.backward(retain_graph=True)
                        for parameter in bridge.parameters():
                            parameter.grad = None
                        voltage_loss.backward()
                    else:
                        (voltage_loss + state_loss).backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        list(bridge.parameters()) + list(state.parameters()),
                        self.config.repair_gradient_clip_norm,
                    )
                    optimizer.step()
                    losses.append(float((voltage_loss + state_loss).detach().cpu()))
                if step in self.config.repair_checkpoints:
                    for arm, pair in pairs.items():
                        seed_report[arm].append(
                            self._checkpoint(arm, seed, step, pair, device)
                        )
                    print(
                        f"[HayFlow 06b-f][checkpoint seed={seed}] {step}/"
                        f"{self.config.repair_training_steps}: {len(REPAIR_ARMS)} arms",
                        flush=True,
                    )
                progress.update(step, f"median_loss={float(np.median(losses)):.4g}")
            for arm, pair in pairs.items():
                pair[0].eval()
                pair[1].eval()
                self.repair_models[(arm, seed)] = pair
            reports[str(seed)] = seed_report
            atomic._write_json(self.output_dir / f"repair_seed{seed}.json", seed_report)
        report = {
            "schema_version": "06b-f-synchronized-training-v1",
            "valid": all(
                len(seed[arm]) == len(self.config.repair_checkpoints)
                for seed in reports.values()
                for arm in REPAIR_ARMS
            ),
            "device": str(device),
            "same_initialization_within_seed": True,
            "same_window_stream_within_seed": True,
            "single_trajectory_per_arm_supplies_all_budgets": True,
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "synchronized_repair_training.json", report)
        return report

    def evaluate_final_repair_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed = {}
        for seed in self.config.pilot_seeds:
            per_seed[str(seed)] = {
                arm: self._evaluate_pair(pair[0], pair[1], "development", device)
                for arm, pair in (
                    (arm, self.repair_models[(arm, seed)]) for arm in REPAIR_ARMS
                )
            }
        report = {
            "schema_version": "06b-f-final-development-v1",
            "valid": all(
                metric["nonfinite_state_count"] == 0
                and metric["nonfinite_voltage_count"] == 0
                and metric["state_domain_violation_count"] == 0
                for seed in per_seed.values()
                for arm in seed.values()
                for boundary in arm.values()
                for metric in boundary.values()
            ),
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "final_repair_matrix.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def finalize_recursive_joint_repair(
        self, training: Mapping[str, Any], evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        horizon = f"{max(self.config.rollout_horizons_ms)}_ms"
        one = "1_ms"
        baseline = "teacherV_teacherS"
        primary = "full_feedback_voltage_protected"
        scalar = "full_feedback_scalar"
        shuffled = "full_feedback_shuffled"

        def row(seed: str, arm: str, boundary: str, h: str) -> Mapping[str, Any]:
            return evaluation["per_seed"][seed][arm][boundary][h]

        per_seed = {}
        for seed in map(str, self.config.pilot_seeds):
            base = row(seed, baseline, "full_feedback", horizon)
            protected = row(seed, primary, "full_feedback", horizon)
            scalar_row = row(seed, scalar, "full_feedback", horizon)
            shuffled_row = row(seed, shuffled, "full_feedback", horizon)
            base_one = row(seed, baseline, "full_feedback", one)
            protected_one = row(seed, primary, "full_feedback", one)
            per_seed[seed] = {
                "protected_state_error_reduction": 1.0
                - protected["normalized_state_rmse"]
                / max(base["normalized_state_rmse"], 1e-12),
                "protected_voltage_error_reduction": 1.0
                - protected["voltage_rmse_mv"] / max(base["voltage_rmse_mv"], 1e-12),
                "protected_state_gain": protected[
                    "state_improvement_vs_persistence_fraction"
                ],
                "protected_voltage_gain": protected[
                    "voltage_improvement_vs_persistence_fraction"
                ],
                "protected_physical_voltage_violations": protected[
                    "physical_voltage_violation_count"
                ],
                "voltage_protection_state_effect": protected[
                    "state_improvement_vs_persistence_fraction"
                ]
                - scalar_row["state_improvement_vs_persistence_fraction"],
                "voltage_protection_voltage_effect": protected[
                    "voltage_improvement_vs_persistence_fraction"
                ]
                - scalar_row["voltage_improvement_vs_persistence_fraction"],
                "scalar_causal_specificity": scalar_row[
                    "state_improvement_vs_persistence_fraction"
                ]
                - shuffled_row["state_improvement_vs_persistence_fraction"],
                "one_step_state_degradation": (
                    protected_one["normalized_state_rmse"]
                    / max(base_one["normalized_state_rmse"], 1e-12)
                    - 1.0
                ),
                "one_step_voltage_degradation": (
                    protected_one["voltage_rmse_mv"]
                    / max(base_one["voltage_rmse_mv"], 1e-12)
                    - 1.0
                ),
            }
        median = {
            name: self._median([value[name] for value in per_seed.values()])
            for name in next(iter(per_seed.values()))
        }
        scaling_by_seed = {}
        for seed in map(str, self.config.pilot_seeds):
            checkpoints = {
                int(checkpoint["budget"]): checkpoint[
                    "calibration_full_feedback_eight_ms"
                ]
                for checkpoint in training["reports"][seed][primary]
            }
            early_budget = self.config.repair_checkpoints[1]
            final_budget = self.config.repair_checkpoints[-1]
            early = checkpoints[early_budget]
            final = checkpoints[final_budget]
            early_joint_error = 0.5 * (
                early["normalized_state_rmse"]
                / max(early["persistence_normalized_state_rmse"], 1e-12)
                + early["voltage_rmse_mv"]
                / max(early["persistence_voltage_rmse_mv"], 1e-12)
            )
            final_joint_error = 0.5 * (
                final["normalized_state_rmse"]
                / max(final["persistence_normalized_state_rmse"], 1e-12)
                + final["voltage_rmse_mv"]
                / max(final["persistence_voltage_rmse_mv"], 1e-12)
            )
            scaling_by_seed[seed] = {
                "early_budget": early_budget,
                "final_budget": final_budget,
                "early_joint_normalized_error": early_joint_error,
                "final_joint_normalized_error": final_joint_error,
                "joint_error_reduction_fraction": 1.0
                - final_joint_error / max(early_joint_error, 1e-12),
            }
        median_scaling_reduction = self._median(
            [
                value["joint_error_reduction_fraction"]
                for value in scaling_by_seed.values()
            ]
        )
        all_seed_safe = all(
            value["protected_voltage_gain"] > 0
            and value["protected_state_gain"] > 0
            and value["protected_physical_voltage_violations"] == 0
            for value in per_seed.values()
        )
        state_repaired = (
            median["protected_state_error_reduction"]
            >= self.config.minimum_state_error_reduction_fraction
        )
        voltage_repaired = (
            median["protected_voltage_error_reduction"]
            >= self.config.minimum_voltage_error_reduction_fraction
        )
        one_step_retained = max(
            median["one_step_state_degradation"],
            median["one_step_voltage_degradation"],
        ) <= self.config.maximum_one_step_degradation_fraction
        causal_specificity = (
            median["scalar_causal_specificity"]
            >= self.config.minimum_shuffled_specificity_fraction
        )
        scaling_continues = (
            median_scaling_reduction
            >= self.config.minimum_scaling_error_reduction_fraction
        )
        if (
            state_repaired
            and voltage_repaired
            and all_seed_safe
            and one_step_retained
            and causal_specificity
            and scaling_continues
        ):
            diagnosis = "JOINT_RECURSIVE_REPAIR_IDENTIFIED"
            next_step = "06c_coupled_voltage_state_micro_canary"
            authorize_06c = True
        elif voltage_repaired and not state_repaired:
            diagnosis = "VOLTAGE_REPAIRED_STATE_EXPOSURE_REMAINS"
            next_step = "state_scheduled_sampling_refinement"
            authorize_06c = False
        elif state_repaired and not voltage_repaired:
            diagnosis = "STATE_EXPOSURE_REPAIRED_VOLTAGE_UNSTABLE"
            next_step = "voltage_stability_refinement"
            authorize_06c = False
        else:
            diagnosis = "JOINT_RECURSIVE_REPAIR_NOT_IDENTIFIED"
            next_step = "return_to_atomic_coupled_boundary_playground"
            authorize_06c = False
        report = {
            "schema_version": "06b-f-final-report-v1",
            "valid": bool(training.get("valid") and evaluation.get("valid")),
            "component_decision_grade": True,
            "diagnosis": diagnosis,
            "gate_checks": {
                "state_exposure_repaired": state_repaired,
                "voltage_recurrence_repaired": voltage_repaired,
                "all_seed_physical_safety": all_seed_safe,
                "one_step_performance_retained": one_step_retained,
                "causal_specificity_retained": causal_specificity,
                "fixed_budget_scaling_continues": scaling_continues,
            },
            "median_contrasts": median,
            "per_seed_contrasts": per_seed,
            "fixed_budget_scaling": {
                "per_seed": scaling_by_seed,
                "median_joint_error_reduction_fraction": median_scaling_reduction,
                "joint_error_definition": (
                    "mean of STATE RMSE/persistence and voltage RMSE/persistence"
                ),
            },
            "registered_thresholds": {
                "minimum_state_error_reduction_fraction": self.config.minimum_state_error_reduction_fraction,
                "minimum_voltage_error_reduction_fraction": self.config.minimum_voltage_error_reduction_fraction,
                "minimum_scaling_error_reduction_fraction": self.config.minimum_scaling_error_reduction_fraction,
                "maximum_one_step_degradation_fraction": self.config.maximum_one_step_degradation_fraction,
                "minimum_shuffled_specificity_fraction": self.config.minimum_shuffled_specificity_fraction,
            },
            "primary_arm": primary,
            "primary_arm_selected_before_execution": True,
            "multiple_questions_answered_in_one_matrix": [
                "STATE_exposure_repair",
                "voltage_feedback_repair",
                "joint_feedback_repair",
                "voltage_protected_gradient_routing",
                "causal_shuffled_specificity",
                "fixed_budget_scaling",
            ],
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": authorize_06c,
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
    "ACCEPTED_06BE_ARTIFACTS",
    "EXPECTED_06BE_INDEX_SHA256",
    "REPAIR_ARMS",
    "RecursiveJointRepairConfig",
    "RecursiveJointRepairMatrix",
    "verified_06be_artifact_root",
]
