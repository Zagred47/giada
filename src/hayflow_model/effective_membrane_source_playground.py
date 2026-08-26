"""06b-o: effective membrane-source versus direct-voltage playground.

The train-only experiment is the first architecture revision after 06b-n.  It
does not add another gate to the retired voltage-mixture branch.  Instead it
asks whether a local neural operator is easier to compose when it predicts the
effective right-hand-side source of the one-millisecond cable equation and a
fixed differentiable Hines solve performs spatial propagation.

One synchronized 2x2x2 matrix crosses output parameterization, causal STATE
feedback and temporal memory.  Exact reconstruction oracles, topology
counterfactuals and spatial shuffles share the same run and require no extra
training.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .atomic_state_dynamics_playground import (
    AtomicStateDynamicsConfig,
    AtomicStateDynamicsPlayground,
    SemanticMechanismStateUpdater,
)
from .hayflow_hines import hayflow_hines_arrays
from .hines_layer import DifferentiableHinesSolve
from .topology_controlled_recurrence_expansion import (
    topology_relabelled_parent_ids,
)


EXPECTED_06BN_ARCHIVE_SHA256 = (
    "5bbdc11d747b5219b6a7713544b4c594b967d08e733fa40d2299ce160cacb54c"
)
EXPECTED_06BN_INDEX_SHA256 = (
    "332a68f9028999eb135e64be2df82e37f7d8ae207e304e57aa205c94a71ea097"
)
EXPECTED_06BN_FINAL_SHA256 = (
    "fdd4109bd9ff9da5ce0f3c512d06924da7b1f6981408f7e536d5d5fa66d12465"
)

DIRECT_VOLTAGE = "direct_voltage"
HINES_SOURCE = "hines_effective_source"
FROZEN_BOUNDARY_STATE = "frozen_boundary_state"
PREDICTED_DYNAMIC_STATE = "predicted_dynamic_state"
INSTANTANEOUS = "instantaneous"
LOCAL_RECURRENT = "local_recurrent"

OUTPUT_PARAMETERIZATIONS = (DIRECT_VOLTAGE, HINES_SOURCE)
STATE_FEEDBACK_CONTRACTS = (FROZEN_BOUNDARY_STATE, PREDICTED_DYNAMIC_STATE)
TEMPORAL_CONTRACTS = (INSTANTANEOUS, LOCAL_RECURRENT)


def verified_06bn_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact registered 06b-n result that authorizes this revision."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-n source must be a ZIP or extracted directory")
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
    roots = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06BN_INDEX_SHA256
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one exact 06b-n artifact; found {len(roots)}")
    root = roots[0]
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
        raise RuntimeError(f"06b-n indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BN_FINAL_SHA256:
        raise RuntimeError("06b-n final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis")
        != "OBJECTIVE_COUPLING_AND_RELAXATION_DO_NOT_CLOSE_ROLLOUT_GAP"
        or final.get("next_step") != "revise_voltage_expert_family_or_state_contract"
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-n result does not authorize 06b-o")
    if source.is_file() and archive_hash != EXPECTED_06BN_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BN_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BN_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
        "next_step": final["next_step"],
    }


@dataclass(frozen=True)
class EffectiveMembraneSourceConfig(AtomicStateDynamicsConfig):
    pilot_seeds: Tuple[int, ...] = (61017, 61029, 61043)
    output_parameterizations: Tuple[str, ...] = OUTPUT_PARAMETERIZATIONS
    state_feedback_contracts: Tuple[str, ...] = STATE_FEEDBACK_CONTRACTS
    temporal_contracts: Tuple[str, ...] = TEMPORAL_CONTRACTS
    matrix_hidden_width: int = 32
    matrix_region_embedding_width: int = 4
    matrix_training_steps: int = 400
    matrix_checkpoints: Tuple[int, ...] = (0, 100, 200, 400)
    matrix_training_horizon_ms: int = 4
    matrix_batch_window_count: int = 4
    matrix_fit_window_count: int = 64
    matrix_calibration_window_count: int = 16
    matrix_development_window_count: int = 16
    matrix_learning_rate: float = 3e-4
    matrix_weight_decay: float = 1e-5
    matrix_gradient_clip_norm: float = 1.0
    macro_step_ms: float = 1.0
    voltage_scale_mv: float = 20.0
    normalized_output_limit: float = 8.0
    native_target_loss_weight: float = 0.25
    state_loss_weight: float = 0.25
    physical_penalty_weight: float = 1.0
    drift_penalty_weight: float = 0.1
    moderate_activity_weight: float = 2.0
    active_activity_weight: float = 4.0
    physical_voltage_minimum_mv: float = -120.0
    physical_voltage_maximum_mv: float = 80.0
    maximum_source_model_parameter_count: int = 20000
    maximum_state_model_parameter_count: int = 12000
    topology_relabel_seed_offset: int = 606500
    source_materiality_fraction: float = 0.02
    state_materiality_fraction: float = 0.02
    memory_materiality_fraction: float = 0.02
    topology_materiality_fraction: float = 0.02
    minimum_global_gain_vs_persistence_fraction: float = 0.10
    minimum_active_gain_vs_persistence_fraction: float = 0.10
    exact_reconstruction_tolerance_mv: float = 1e-4

    def validate(self) -> None:
        super().validate()
        if tuple(self.output_parameterizations) != OUTPUT_PARAMETERIZATIONS:
            raise ValueError("06b-o output-parameterization axis changed")
        if tuple(self.state_feedback_contracts) != STATE_FEEDBACK_CONTRACTS:
            raise ValueError("06b-o STATE-feedback axis changed")
        if tuple(self.temporal_contracts) != TEMPORAL_CONTRACTS:
            raise ValueError("06b-o temporal axis changed")
        if len(set(self.pilot_seeds)) < 3 or any(seed <= 0 for seed in self.pilot_seeds):
            raise ValueError("06b-o requires three positive independent pilot seeds")
        if (
            self.matrix_checkpoints[0] != 0
            or self.matrix_checkpoints[-1] != self.matrix_training_steps
            or tuple(sorted(set(self.matrix_checkpoints))) != self.matrix_checkpoints
        ):
            raise ValueError("06b-o checkpoints must span a sorted training trajectory")
        if max(self.rollout_horizons_ms) < self.matrix_training_horizon_ms:
            raise ValueError("06b-o evaluation horizon cannot be shorter than training")
        positive = (
            self.matrix_hidden_width,
            self.matrix_region_embedding_width,
            self.matrix_training_steps,
            self.matrix_training_horizon_ms,
            self.matrix_batch_window_count,
            self.matrix_fit_window_count,
            self.matrix_calibration_window_count,
            self.matrix_development_window_count,
            self.matrix_learning_rate,
            self.matrix_gradient_clip_norm,
            self.macro_step_ms,
            self.voltage_scale_mv,
            self.normalized_output_limit,
            self.maximum_source_model_parameter_count,
            self.maximum_state_model_parameter_count,
            self.exact_reconstruction_tolerance_mv,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("06b-o positive configuration value is invalid")
        if self.matrix_weight_decay < 0:
            raise ValueError("06b-o weight decay cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EffectiveMembraneSourceConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "pilot_seeds",
            "matrix_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        for name in (
            "output_parameterizations",
            "state_feedback_contracts",
            "temporal_contracts",
        ):
            if name in payload:
                payload[name] = tuple(map(str, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


if atomic.nn is not None:

    class CausalMembraneSourceCell(atomic.nn.Module):
        """Parameter-matched local operator used by every matrix arm."""

        def __init__(
            self,
            feature_width: int,
            region_count: int,
            region_width: int,
            hidden_width: int,
            output_limit: float,
        ) -> None:
            super().__init__()
            self.hidden_width = int(hidden_width)
            self.output_limit = float(output_limit)
            self.region = atomic.nn.Embedding(region_count, region_width)
            self.encoder = atomic.nn.Sequential(
                atomic.nn.Linear(feature_width + region_width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, hidden_width),
                atomic.nn.SiLU(),
            )
            self.recurrent = atomic.nn.GRUCell(hidden_width, hidden_width)
            self.readout = atomic.nn.Linear(hidden_width, 1)
            atomic.nn.init.zeros_(self.region.weight)
            atomic.nn.init.zeros_(self.readout.weight)
            atomic.nn.init.zeros_(self.readout.bias)

        def forward(
            self,
            features: Any,
            region_ids: Any,
            hidden: Any,
            *,
            recurrent: bool,
        ) -> Tuple[Any, Any]:
            batch, segments, _ = features.shape
            if not recurrent:
                hidden = atomic.torch.zeros_like(hidden)
            region = self.region(region_ids)[None, :, :].expand(batch, -1, -1)
            encoded = self.encoder(atomic.torch.cat((features, region), dim=-1))
            next_hidden = self.recurrent(
                encoded.reshape(batch * segments, -1),
                hidden.reshape(batch * segments, -1),
            ).reshape(batch, segments, -1)
            output = self.output_limit * atomic.torch.tanh(
                self.readout(next_hidden).squeeze(-1)
            )
            return output, next_hidden


else:  # pragma: no cover

    class CausalMembraneSourceCell:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06b-o requires PyTorch")


class EffectiveMembraneSourcePlayground(AtomicStateDynamicsPlayground):
    """Run the aligned physical-source architecture playground."""

    config: EffectiveMembraneSourceConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: EffectiveMembraneSourceConfig,
        artifact_05t_source: Path,
        artifact_06bn_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05t_source,
            code_revision=code_revision,
        )
        self.artifact_06bn_source = Path(artifact_06bn_source)
        self.window_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.matrix_models: Dict[Tuple[str, int], Tuple[Any, Any]] = {}
        self.training_valid = False
        self.physical: Dict[str, np.ndarray] = {}
        self.topology: Dict[str, Dict[str, Any]] = {}

    def _specs(self) -> Tuple[Tuple[str, str, str], ...]:
        return tuple(
            itertools.product(
                self.config.output_parameterizations,
                self.config.state_feedback_contracts,
                self.config.temporal_contracts,
            )
        )

    @staticmethod
    def _spec_key(spec: Tuple[str, str, str]) -> str:
        return "|".join(spec)

    @staticmethod
    def _children(parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values: List[List[int]] = [[] for _ in range(len(parent))]
        for child, parent_id in enumerate(parent):
            if child != int(parent_id):
                values[int(parent_id)].append(child)
        width = max(1, max(map(len, values)))
        ids = np.zeros((len(parent), width), dtype=np.int64)
        mask = np.zeros((len(parent), width), dtype=np.float32)
        for index, children in enumerate(values):
            if children:
                ids[index, : len(children)] = children
                mask[index, : len(children)] = 1.0
            else:
                ids[index, 0] = index
        return ids, mask

    @staticmethod
    def _axial_total(parent: np.ndarray, coupling: np.ndarray) -> np.ndarray:
        total = np.asarray(coupling, dtype=np.float64).copy()
        for child, parent_id in enumerate(parent):
            if child != int(parent_id):
                total[int(parent_id)] += float(coupling[child])
        return total

    def _build_topologies(self) -> None:
        authentic = np.asarray(self.physical["parent_ids"], dtype=np.int64)
        relabelled = topology_relabelled_parent_ids(
            authentic,
            seed=self.config.topology_relabel_seed_offset,
        )
        coupling = np.asarray(
            self.physical["axial_conductance_to_parent_us"], dtype=np.float64
        )
        for name, parent, edge in (
            ("authentic", authentic, coupling),
            ("relabelled", relabelled, coupling),
            ("no_axial", authentic, np.zeros_like(coupling)),
        ):
            children, child_mask = self._children(parent)
            self.topology[name] = {
                "parent": parent,
                "children": children,
                "child_mask": child_mask,
                "coupling": edge,
                "axial_total": self._axial_total(parent, edge),
                "solver": DifferentiableHinesSolve(parent),
            }

    def _coordinate_layout(self) -> None:
        group_count = len(self.coordinate_groups)
        lookup = np.full(
            (self.layout.segment_count, group_count), -1, dtype=np.int64
        )
        for coordinate, (segment, group) in enumerate(
            zip(self.coordinate["segment"], self.coordinate["semantic_group"])
        ):
            if lookup[int(segment), int(group)] >= 0:
                raise RuntimeError("duplicate mechanism STATE semantic coordinate")
            lookup[int(segment), int(group)] = int(coordinate)
        self.coordinate_lookup = lookup
        self.semantic_presence = (lookup >= 0).astype(np.float32)

    def _windows_for_role(
        self, role: str, count: int, horizon: int
    ) -> List[np.ndarray]:
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
                f"{self.config.role_seed}|06bo|{role}|{','.join(map(str,row))}".encode()
            ).hexdigest()
        )
        return windows[:count]

    def _materialize_window_role(
        self, role: str, count: int, horizon: int
    ) -> Dict[str, np.ndarray]:
        windows = self._windows_for_role(role, count, horizon)
        if len(windows) < count:
            raise RuntimeError(f"06b-o found only {len(windows)} {role} windows")
        index = np.asarray(windows, dtype=np.int64)
        flat = index.reshape(-1)
        state_shape = (len(windows), horizon, -1)
        voltage_shape = (len(windows), horizon, -1)
        state_t = atomic.mechanism_logit(
            self.store.read_state(flat, "t", categories=("mechanism_states",))
        ).astype(np.float32).reshape(state_shape)
        state_t1 = atomic.mechanism_logit(
            self.store.read_state(flat, "t_plus_1", categories=("mechanism_states",))
        ).astype(np.float32).reshape(state_shape)
        voltage_t = self.store.read_state(
            flat, "t", categories=("voltage",)
        ).astype(np.float32).reshape(voltage_shape)
        voltage_t1 = self.store.read_state(
            flat, "t_plus_1", categories=("voltage",)
        ).astype(np.float32).reshape(voltage_shape)
        drive = atomic.encode_causal_realized_drive(self.store, flat).astype(np.float32)
        drive = drive.reshape(len(windows), horizon, self.layout.segment_count, -1)
        payload = {
            "indices": index,
            "state_t": state_t,
            "state_t1": state_t1,
            "voltage_t": voltage_t,
            "voltage_t1": voltage_t1,
            "drive": drive,
            "held_ions": self._ion_context(index[:, 0]).astype(np.float32),
        }
        self.window_data[role] = payload
        return payload

    def _feature_width(self) -> int:
        return int(
            4
            + 2 * len(self.coordinate_groups)
            + len(atomic.CAUSAL_DRIVE_FEATURES)
            + len(self.ion_feature_names)
            + self.layout.segment_static.shape[1]
        )

    def _new_source_model(self, seed: int, device: Any) -> Any:
        atomic.torch.manual_seed(seed + 670000)
        model = CausalMembraneSourceCell(
            self._feature_width(),
            len(self.layout.region_names),
            self.config.matrix_region_embedding_width,
            self.config.matrix_hidden_width,
            self.config.normalized_output_limit,
        ).to(device)
        count = sum(value.numel() for value in model.parameters())
        if count > self.config.maximum_source_model_parameter_count:
            raise RuntimeError(f"06b-o source model has {count} parameters")
        return model

    def _new_state_model(self, seed: int, device: Any) -> Any:
        atomic.torch.manual_seed(seed + 671000)
        model = SemanticMechanismStateUpdater(
            mechanism_count=len(self.layout.mechanism_names),
            variable_count=len(self.layout.variable_names),
            kind_count=len(self.layout.kind_names),
            region_count=len(self.layout.region_names),
            static_width=self.layout.segment_static.shape[1],
            drive_width=len(atomic.CAUSAL_DRIVE_FEATURES) + len(self.ion_feature_names),
            hidden_width=self.config.hidden_width,
            embedding_width=self.config.embedding_width,
            normalized_delta_limit=self.config.normalized_delta_limit,
        ).to(device)
        count = sum(value.numel() for value in model.parameters())
        if count > self.config.maximum_state_model_parameter_count:
            raise RuntimeError(f"06b-o STATE model has {count} parameters")
        return model

    def _batch_tensors(
        self, role: str, rows: np.ndarray, device: Any
    ) -> Dict[str, Any]:
        return {
            name: atomic.torch.as_tensor(value[rows], device=device)
            for name, value in self.window_data[role].items()
            if name != "indices"
        }

    def prepare_effective_membrane_source_playground(self) -> Dict[str, Any]:
        base = self.prepare_playground()
        _, source = verified_06bn_artifact_root(
            self.artifact_06bn_source,
            self.output_dir.parent / ".06bo_artifact_cache" / "06bn",
        )
        self._coordinate_layout()
        self.physical = hayflow_hines_arrays(self.layout)
        self._build_topologies()
        horizon = max(self.config.rollout_horizons_ms)
        self._materialize_window_role(
            "fit", self.config.matrix_fit_window_count, self.config.matrix_training_horizon_ms
        )
        self._materialize_window_role(
            "calibration", self.config.matrix_calibration_window_count, horizon
        )
        self._materialize_window_role(
            "development", self.config.matrix_development_window_count, horizon
        )
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        source_count = sum(
            value.numel() for value in self._new_source_model(self.config.pilot_seeds[0], device).parameters()
        )
        state_count = sum(
            value.numel() for value in self._new_state_model(self.config.pilot_seeds[0], device).parameters()
        )
        report = {
            **base,
            "schema_version": "06b-o-effective-source-contract-v1",
            "experiment": "effective_membrane_source_playground",
            "source_06bn": source,
            "scientific_question": (
                "Does predicting an effective local cable-equation RHS source and "
                "delegating spatial propagation to Hines improve recursive learnability?"
            ),
            "factorial_axes": {
                "output_parameterization": list(self.config.output_parameterizations),
                "state_feedback": list(self.config.state_feedback_contracts),
                "temporal_contract": list(self.config.temporal_contracts),
            },
            "factor_arm_count": len(self._specs()),
            "trainable_parameter_count_per_arm": source_count + state_count,
            "same_numeric_input_tensor": True,
            "same_initialization_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "same_native_and_endpoint_objective_weights": True,
            "teacher_endpoint_used_as_input": False,
            "teacher_source_used_only_as_supervision_or_oracle": True,
            "teacher_state_refresh_selection_eligible": False,
            "topology_counterfactuals_require_retraining": False,
            "state_and_outcome_splits_read": ["train"],
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "configuration": asdict(self.config),
        }
        atomic._write_json(self.output_dir / "effective_source_contract.json", report)
        return report

    def _topology_tensors(
        self, name: str, dtype: Any, device: Any
    ) -> Dict[str, Any]:
        row = self.topology[name]
        return {
            "parent": atomic.torch.as_tensor(
                row["parent"], dtype=atomic.torch.long, device=device
            ),
            "children": atomic.torch.as_tensor(
                row["children"], dtype=atomic.torch.long, device=device
            ),
            "child_mask": atomic.torch.as_tensor(
                row["child_mask"], dtype=dtype, device=device
            ),
            "coupling": atomic.torch.as_tensor(
                row["coupling"], dtype=dtype, device=device
            ),
            "axial_total": atomic.torch.as_tensor(
                row["axial_total"], dtype=dtype, device=device
            ),
            "solver": row["solver"].to(device=device, dtype=dtype),
        }

    def _physical_terms(
        self, voltage: Any, topology_name: str
    ) -> Tuple[Any, Any, Any, Any]:
        topology = self._topology_tensors(
            topology_name, voltage.dtype, voltage.device
        )
        mass = (1000.0 / self.config.macro_step_ms) * atomic.torch.as_tensor(
            self.physical["capacitance_uf"],
            dtype=voltage.dtype,
            device=voltage.device,
        )
        leak = atomic.torch.as_tensor(
            self.physical["leak_conductance_us"],
            dtype=voltage.dtype,
            device=voltage.device,
        )
        reversal = atomic.torch.as_tensor(
            self.physical["leak_reversal_mv"],
            dtype=voltage.dtype,
            device=voltage.device,
        )
        diagonal = mass + leak + topology["axial_total"]
        diagonal = diagonal[None, :].expand_as(voltage)
        coupling = topology["coupling"][None, :].expand_as(voltage)
        base_rhs = mass[None, :] * voltage + leak[None, :] * reversal[None, :]
        source_scale = diagonal * self.config.voltage_scale_mv
        return diagonal, coupling, base_rhs, source_scale

    def _matrix_apply(self, voltage: Any, topology_name: str) -> Any:
        diagonal, coupling, _, _ = self._physical_terms(voltage, topology_name)
        topology = self._topology_tensors(
            topology_name, voltage.dtype, voltage.device
        )
        parent = topology["parent"]
        result = diagonal * voltage
        non_root = atomic.torch.arange(
            voltage.shape[1], dtype=atomic.torch.long, device=voltage.device
        ) != int(topology["solver"].root)
        nodes = atomic.torch.arange(
            voltage.shape[1], dtype=atomic.torch.long, device=voltage.device
        )[non_root]
        parents = parent[nodes]
        edge = coupling[:, nodes]
        result = result.scatter_add(1, nodes[None, :].expand(voltage.shape[0], -1), -edge * voltage[:, parents])
        result = result.scatter_add(1, parents[None, :].expand(voltage.shape[0], -1), -edge * voltage[:, nodes])
        return result

    def _normalized_source_target(
        self, current_voltage: Any, target_voltage: Any, topology_name: str
    ) -> Any:
        _, _, base_rhs, source_scale = self._physical_terms(
            current_voltage, topology_name
        )
        return (self._matrix_apply(target_voltage, topology_name) - base_rhs) / source_scale

    def _apply_output(
        self,
        parameterization: str,
        normalized_output: Any,
        current_voltage: Any,
        topology_name: str,
    ) -> Any:
        if parameterization == DIRECT_VOLTAGE:
            return current_voltage + normalized_output * self.config.voltage_scale_mv
        if parameterization != HINES_SOURCE:
            raise ValueError(parameterization)
        diagonal, coupling, base_rhs, source_scale = self._physical_terms(
            current_voltage, topology_name
        )
        source = normalized_output * source_scale
        solver = self._topology_tensors(
            topology_name, current_voltage.dtype, current_voltage.device
        )["solver"]
        return solver(diagonal, coupling, base_rhs + source)

    def _segment_state_tensor(self, normalized_state: Any) -> Any:
        lookup = atomic.torch.as_tensor(
            self.coordinate_lookup,
            dtype=atomic.torch.long,
            device=normalized_state.device,
        )
        mask = lookup >= 0
        gathered = normalized_state[:, lookup.clamp_min(0)]
        return gathered * mask[None, :, :]

    def _features(
        self,
        normalized_state: Any,
        voltage: Any,
        context: Any,
        topology_name: str,
    ) -> Any:
        topology = self._topology_tensors(
            topology_name, voltage.dtype, voltage.device
        )
        parent_delta = voltage[:, topology["parent"]] - voltage
        child_delta = (
            (voltage[:, topology["children"]] - voltage[:, :, None])
            * topology["child_mask"][None, :, :]
        ).sum(-1)
        child_delta = child_delta / topology["child_mask"].sum(-1).clamp_min(1.0)[
            None, :
        ]
        if topology_name == "no_axial":
            parent_delta = atomic.torch.zeros_like(parent_delta)
            child_delta = atomic.torch.zeros_like(child_delta)
        state = self._segment_state_tensor(normalized_state)
        presence = atomic.torch.as_tensor(
            self.semantic_presence,
            dtype=voltage.dtype,
            device=voltage.device,
        )[None, :, :].expand(voltage.shape[0], -1, -1)
        static = atomic.torch.as_tensor(
            self.layout.segment_static,
            dtype=voltage.dtype,
            device=voltage.device,
        )[None, :, :].expand(voltage.shape[0], -1, -1)
        basic = atomic.torch.stack(
            (
                voltage / 100.0,
                parent_delta / 100.0,
                child_delta / 100.0,
                (parent_delta - child_delta) / 100.0,
            ),
            dim=-1,
        )
        return atomic.torch.cat((basic, state, presence, context, static), dim=-1)

    def _state_forward(
        self,
        model: Any,
        normalized_state: Any,
        voltage: Any,
        voltage_delta: Any,
        context: Any,
    ) -> Any:
        batch, coordinate_count = normalized_state.shape
        device = normalized_state.device
        segments = atomic.torch.as_tensor(
            self.coordinate["segment"], dtype=atomic.torch.long, device=device
        )
        static = atomic.torch.as_tensor(
            self.layout.segment_static[self.coordinate["segment"]],
            dtype=normalized_state.dtype,
            device=device,
        )[None, :, :].expand(batch, -1, -1)

        def ids(name: str) -> Any:
            return atomic.torch.as_tensor(
                self.coordinate[name], dtype=atomic.torch.long, device=device
            )[None, :].expand(batch, -1).reshape(-1)

        prediction = model(
            normalized_state.reshape(-1),
            voltage[:, segments].reshape(-1),
            voltage_delta[:, segments].reshape(-1),
            context[:, segments].reshape(batch * coordinate_count, -1),
            static.reshape(batch * coordinate_count, -1),
            ids("mechanism"),
            ids("variable"),
            ids("kind"),
            ids("region"),
        )
        return prediction.reshape(batch, coordinate_count)

    def _activity_weight(self, target_delta: Any) -> Any:
        absolute = target_delta.abs()
        return atomic.torch.where(
            absolute < 1.0,
            atomic.torch.ones_like(absolute),
            atomic.torch.where(
                absolute < 5.0,
                atomic.torch.full_like(absolute, self.config.moderate_activity_weight),
                atomic.torch.full_like(absolute, self.config.active_activity_weight),
            ),
        )

    def _unroll(
        self,
        source_model: Any,
        state_model: Any,
        spec: Tuple[str, str, str],
        batch: Mapping[str, Any],
        *,
        horizon: int,
        collect: bool,
        topology_name: str = "authentic",
        teacher_state_refresh: bool = False,
        spatial_shuffle: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        parameterization, state_contract, temporal_contract = spec
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        hidden = atomic.torch.zeros(
            current_voltage.shape[0],
            self.layout.segment_count,
            self.config.matrix_hidden_width,
            dtype=current_voltage.dtype,
            device=current_voltage.device,
        )
        center = atomic.torch.as_tensor(
            self.statistics["state_center"], device=current_state.device
        )
        scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], device=current_state.device
        )
        delta_scale = atomic.torch.as_tensor(
            self.statistics["delta_scale"], device=current_state.device
        )
        region_ids = atomic.torch.as_tensor(
            self.layout.segment_region_ids,
            dtype=atomic.torch.long,
            device=current_voltage.device,
        )
        permutation = atomic.torch.as_tensor(
            np.random.default_rng(672001).permutation(self.layout.segment_count),
            dtype=atomic.torch.long,
            device=current_voltage.device,
        )
        losses = []
        outputs: Dict[str, Any] = {}
        for step in range(horizon):
            state_input = (
                batch["state_t"][:, step] if teacher_state_refresh else current_state
            )
            normalized = (state_input - center) / scale
            context = atomic.torch.cat(
                (batch["drive"][:, step], batch["held_ions"]), dim=-1
            )
            features = self._features(
                normalized, current_voltage, context, topology_name
            )
            normalized_output, next_hidden = source_model(
                features,
                region_ids,
                hidden,
                recurrent=temporal_contract == LOCAL_RECURRENT,
            )
            if spatial_shuffle:
                normalized_output = normalized_output[:, permutation]
            next_voltage = self._apply_output(
                parameterization,
                normalized_output,
                current_voltage,
                topology_name,
            )
            voltage_delta = next_voltage - current_voltage
            state_delta = self._state_forward(
                state_model, normalized, current_voltage, voltage_delta, context
            )
            predicted_next_state = state_input + state_delta * delta_scale
            target_voltage = batch["voltage_t1"][:, step]
            target_state = batch["state_t1"][:, step]
            target_delta = target_voltage - batch["voltage_t"][:, step]
            endpoint_error = (
                next_voltage - target_voltage
            ) / self.config.voltage_scale_mv
            endpoint_loss = atomic.torch.mean(
                self._activity_weight(target_delta)
                * atomic.torch_functional.smooth_l1_loss(
                    endpoint_error,
                    atomic.torch.zeros_like(endpoint_error),
                    reduction="none",
                )
            )
            native_target = (
                (target_voltage - current_voltage) / self.config.voltage_scale_mv
                if parameterization == DIRECT_VOLTAGE
                else self._normalized_source_target(
                    current_voltage, target_voltage, topology_name
                )
            )
            native_loss = atomic.torch_functional.smooth_l1_loss(
                normalized_output, native_target
            )
            state_error = (predicted_next_state - target_state) / scale
            state_loss = atomic.torch_functional.smooth_l1_loss(
                state_error, atomic.torch.zeros_like(state_error)
            )
            high = atomic.torch.relu(
                (next_voltage - self.config.physical_voltage_maximum_mv)
                / self.config.voltage_scale_mv
            )
            low = atomic.torch.relu(
                (self.config.physical_voltage_minimum_mv - next_voltage)
                / self.config.voltage_scale_mv
            )
            physical_loss = atomic.torch.mean(high * high + low * low)
            drift_loss = atomic.torch.mean(
                atomic.torch.mean(endpoint_error, dim=1) ** 2
            )
            losses.append(
                endpoint_loss
                + self.config.native_target_loss_weight * native_loss
                + self.config.state_loss_weight * state_loss
                + self.config.physical_penalty_weight * physical_loss
                + self.config.drift_penalty_weight * drift_loss
            )
            if collect and step + 1 in self.config.rollout_horizons_ms:
                outputs[f"{step + 1}_ms"] = {
                    "voltage": next_voltage,
                    "state": predicted_next_state,
                    "normalized_output": normalized_output,
                }
            if teacher_state_refresh:
                current_state = predicted_next_state
            elif state_contract == PREDICTED_DYNAMIC_STATE:
                current_state = predicted_next_state
            hidden = (
                next_hidden
                if temporal_contract == LOCAL_RECURRENT
                else atomic.torch.zeros_like(next_hidden)
            )
            current_voltage = next_voltage
        return atomic.torch.stack(losses).mean(), outputs

    @staticmethod
    def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
        return float(
            np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))
        )

    def _masked_voltage_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        initial: np.ndarray,
        masks: Mapping[str, np.ndarray],
    ) -> Dict[str, Any]:
        rows: Dict[str, Any] = {}
        for name, mask in masks.items():
            expanded = np.broadcast_to(np.asarray(mask, dtype=bool), target.shape)
            count = int(expanded.sum())
            if not count:
                rows[name] = {
                    "coordinate_count": 0,
                    "voltage_rmse_mv": None,
                    "persistence_rmse_mv": None,
                    "voltage_gain_vs_persistence_fraction": None,
                }
                continue
            model_rmse = self._rmse(prediction[expanded], target[expanded])
            persistence_rmse = self._rmse(initial[expanded], target[expanded])
            rows[name] = {
                "coordinate_count": count,
                "voltage_rmse_mv": model_rmse,
                "persistence_rmse_mv": persistence_rmse,
                "voltage_gain_vs_persistence_fraction": (
                    1.0 - model_rmse / max(persistence_rmse, 1e-12)
                ),
            }
        return rows

    def _metrics(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        prediction = output["voltage"].detach().cpu().numpy()
        predicted_state = output["state"].detach().cpu().numpy()
        target = batch["voltage_t1"][:, step].detach().cpu().numpy()
        target_state = batch["state_t1"][:, step].detach().cpu().numpy()
        initial = batch["voltage_t"][:, 0].detach().cpu().numpy()
        state_initial = batch["state_t"][:, 0].detach().cpu().numpy()
        target_activity = np.abs(target - initial)
        activity_masks = {
            "quiescent_lt_1mV": target_activity < 1.0,
            "moderate_1_to_5mV": (target_activity >= 1.0) & (target_activity < 5.0),
            "active_ge_5mV": target_activity >= 5.0,
            "regenerative_ge_20mV": target_activity >= 20.0,
        }
        region_masks = {
            str(name): self.layout.segment_region_ids == index
            for index, name in enumerate(self.layout.region_names)
        }
        voltage_rmse = self._rmse(prediction, target)
        persistence_rmse = self._rmse(initial, target)
        state_scale = self.statistics["state_scale"][None, :]
        state_rmse = self._rmse(
            (predicted_state - target_state) / state_scale,
            np.zeros_like(target_state),
        )
        state_persistence = self._rmse(
            (state_initial - target_state) / state_scale,
            np.zeros_like(target_state),
        )
        return {
            "voltage_rmse_mv": voltage_rmse,
            "persistence_rmse_mv": persistence_rmse,
            "voltage_gain_vs_persistence_fraction": (
                1.0 - voltage_rmse / max(persistence_rmse, 1e-12)
            ),
            "endpoint_mean_drift_mv": float(np.mean(prediction - target)),
            "normalized_state_rmse": state_rmse,
            "state_improvement_vs_persistence_fraction": (
                1.0 - state_rmse / max(state_persistence, 1e-12)
            ),
            "nonfinite_voltage_count": int((~np.isfinite(prediction)).sum()),
            "nonfinite_state_count": int((~np.isfinite(predicted_state)).sum()),
            "physical_voltage_violation_count": int(
                (
                    (prediction < self.config.physical_voltage_minimum_mv)
                    | (prediction > self.config.physical_voltage_maximum_mv)
                ).sum()
            ),
            "activity": self._masked_voltage_metrics(
                prediction, target, initial, activity_masks
            ),
            "region": self._masked_voltage_metrics(
                prediction, target, initial, region_masks
            ),
        }

    def _evaluate_pair(
        self,
        source_model: Any,
        state_model: Any,
        spec: Tuple[str, str, str],
        role: str,
        device: Any,
        *,
        topology_name: str = "authentic",
        teacher_state_refresh: bool = False,
        spatial_shuffle: bool = False,
    ) -> Dict[str, Any]:
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        source_training = bool(source_model.training)
        state_training = bool(state_model.training)
        source_model.eval()
        state_model.eval()
        with atomic.torch.no_grad():
            _, outputs = self._unroll(
                source_model,
                state_model,
                spec,
                batch,
                horizon=max(self.config.rollout_horizons_ms),
                collect=True,
                topology_name=topology_name,
                teacher_state_refresh=teacher_state_refresh,
                spatial_shuffle=spatial_shuffle,
            )
        result = {
            "horizons": {
                horizon: self._metrics(output, batch, int(horizon[:-3]) - 1)
                for horizon, output in outputs.items()
            }
        }
        source_model.train(source_training)
        state_model.train(state_training)
        return result

    def run_exact_source_reconstruction_audit(self) -> Dict[str, Any]:
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        rows = np.arange(
            min(8, len(self.window_data["development"]["indices"])), dtype=np.int64
        )
        batch = self._batch_tensors("development", rows, device)
        errors: Dict[str, List[float]] = {
            "authentic": [],
            "relabelled_with_authentic_source": [],
            "no_axial_with_authentic_source": [],
            "passive_only": [],
        }
        native_magnitudes = []
        source_targets = []
        direct_targets = []
        float32_authentic_errors = []
        with atomic.torch.no_grad():
            for step in range(max(self.config.rollout_horizons_ms)):
                current_float32 = batch["voltage_t"][:, step]
                target_float32 = batch["voltage_t1"][:, step]
                normalized_float32 = self._normalized_source_target(
                    current_float32, target_float32, "authentic"
                )
                reconstructed_float32 = self._apply_output(
                    HINES_SOURCE,
                    normalized_float32,
                    current_float32,
                    "authentic",
                )
                float32_authentic_errors.append(
                    float((reconstructed_float32 - target_float32).abs().max().cpu())
                )

                # This stage verifies an algebraic identity, not the numerical
                # precision used by the trainable model.  In float32 the cable
                # RHS subtracts large mass/axial terms to recover a small
                # one-millisecond voltage increment, so cancellation can exceed
                # the preregistered identity tolerance even when A^-1(Ax) is
                # implemented correctly.  Keep that operational error above,
                # but perform the blocking identity check in float64.
                current = current_float32.to(dtype=atomic.torch.float64)
                target = target_float32.to(dtype=atomic.torch.float64)
                normalized = self._normalized_source_target(
                    current, target, "authentic"
                )
                native_magnitudes.append(float(normalized.abs().median().cpu()))
                source_targets.append(normalized.abs().cpu().numpy().reshape(-1))
                direct_targets.append(
                    ((target - current) / self.config.voltage_scale_mv)
                    .abs()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                reconstructed = self._apply_output(
                    HINES_SOURCE, normalized, current, "authentic"
                )
                relabelled = self._apply_output(
                    HINES_SOURCE, normalized, current, "relabelled"
                )
                no_axial = self._apply_output(
                    HINES_SOURCE, normalized, current, "no_axial"
                )
                passive = self._apply_output(
                    HINES_SOURCE, atomic.torch.zeros_like(normalized), current, "authentic"
                )
                errors["authentic"].append(
                    float((reconstructed - target).abs().max().cpu())
                )
                errors["relabelled_with_authentic_source"].append(
                    self._rmse(relabelled.cpu().numpy(), target.cpu().numpy())
                )
                errors["no_axial_with_authentic_source"].append(
                    self._rmse(no_axial.cpu().numpy(), target.cpu().numpy())
                )
                errors["passive_only"].append(
                    self._rmse(passive.cpu().numpy(), target.cpu().numpy())
                )
        source_absolute = np.concatenate(source_targets)
        direct_absolute = np.concatenate(direct_targets)
        report = {
            "schema_version": "06b-o-source-reconstruction-v1",
            "valid": max(errors["authentic"]) <= self.config.exact_reconstruction_tolerance_mv,
            "teacher_source_is_selection_eligible": False,
            "identity_audit_dtype": "float64",
            "operational_training_dtype": "float32",
            "maximum_authentic_reconstruction_error_mv": max(errors["authentic"]),
            "maximum_float32_authentic_reconstruction_error_mv": max(
                float32_authentic_errors
            ),
            "median_relabelled_rmse_mv": float(
                np.median(errors["relabelled_with_authentic_source"])
            ),
            "median_no_axial_rmse_mv": float(
                np.median(errors["no_axial_with_authentic_source"])
            ),
            "median_passive_only_rmse_mv": float(np.median(errors["passive_only"])),
            "median_absolute_normalized_teacher_source": float(
                np.median(native_magnitudes)
            ),
            "normalized_target_support": {
                "source_p99": float(np.quantile(source_absolute, 0.99)),
                "source_maximum": float(source_absolute.max()),
                "source_beyond_model_limit_fraction": float(
                    np.mean(source_absolute > self.config.normalized_output_limit)
                ),
                "direct_voltage_p99": float(np.quantile(direct_absolute, 0.99)),
                "direct_voltage_maximum": float(direct_absolute.max()),
                "direct_voltage_beyond_model_limit_fraction": float(
                    np.mean(direct_absolute > self.config.normalized_output_limit)
                ),
            },
            "equation": (
                "A(V_t) V_t+1 = mass*V_t + leak*E_leak + I_effective"
            ),
        }
        atomic._write_json(self.output_dir / "source_reconstruction_audit.json", report)
        if not report["valid"]:
            raise RuntimeError("06b-o effective-source reconstruction identity failed")
        return report

    def _new_pair(self, seed: int, device: Any) -> Tuple[Any, Any]:
        return self._new_source_model(seed, device), self._new_state_model(seed, device)

    @staticmethod
    def _copy_state_dict(model: Any) -> Dict[str, Any]:
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    def train_synchronized_source_matrix(self) -> Dict[str, Any]:
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        specs = self._specs()
        reports: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            pairs = {spec: self._new_pair(seed, device) for spec in specs}
            optimizers = {
                spec: atomic.torch.optim.AdamW(
                    list(pair[0].parameters()) + list(pair[1].parameters()),
                    lr=self.config.matrix_learning_rate,
                    weight_decay=self.config.matrix_weight_decay,
                )
                for spec, pair in pairs.items()
            }
            trajectories: Dict[str, List[Dict[str, Any]]] = {
                self._spec_key(spec): [] for spec in specs
            }
            best: Dict[Tuple[str, str, str], Tuple[float, int, Dict[str, Any], Dict[str, Any]]] = {}
            rng = np.random.default_rng(seed + 673000)
            digest = hashlib.sha256()
            progress = atomic._CompactProgress(
                f"06b-o 2x2x2 seed={seed}",
                self.config.matrix_training_steps,
                max(1, self.config.matrix_training_steps // 25),
            )
            for step in range(self.config.matrix_training_steps + 1):
                if step in self.config.matrix_checkpoints:
                    for spec, pair in pairs.items():
                        calibration = self._evaluate_pair(
                            pair[0], pair[1], spec, "calibration", device
                        )
                        score = calibration["horizons"]["8_ms"]["voltage_rmse_mv"]
                        trajectories[self._spec_key(spec)].append(
                            {"step": step, "calibration": calibration}
                        )
                        if spec not in best or score < best[spec][0]:
                            best[spec] = (
                                score,
                                step,
                                self._copy_state_dict(pair[0]),
                                self._copy_state_dict(pair[1]),
                            )
                if step == self.config.matrix_training_steps:
                    break
                rows = rng.choice(
                    len(self.window_data["fit"]["indices"]),
                    size=self.config.matrix_batch_window_count,
                    replace=False,
                )
                digest.update(np.asarray(rows, dtype=np.int64).tobytes())
                batch = self._batch_tensors("fit", rows, device)
                losses = []
                for spec, pair in pairs.items():
                    optimizer = optimizers[spec]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._unroll(
                        pair[0],
                        pair[1],
                        spec,
                        batch,
                        horizon=self.config.matrix_training_horizon_ms,
                        collect=False,
                    )
                    if not bool(atomic.torch.isfinite(loss)):
                        raise RuntimeError(
                            f"non-finite 06b-o loss seed={seed} step={step} "
                            f"arm={self._spec_key(spec)}"
                        )
                    loss.backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        list(pair[0].parameters()) + list(pair[1].parameters()),
                        self.config.matrix_gradient_clip_norm,
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(
                    step + 1,
                    "loss[min/median/max]="
                    f"{min(losses):.3g}/{float(np.median(losses)):.3g}/{max(losses):.3g}",
                )
            selected = {}
            for spec, pair in pairs.items():
                score, selected_step, source_state, state_state = best[spec]
                pair[0].load_state_dict(source_state)
                pair[1].load_state_dict(state_state)
                pair[0].eval()
                pair[1].eval()
                key = self._spec_key(spec)
                self.matrix_models[(key, seed)] = pair
                path = self.output_dir / f"source_matrix_{key.replace('|','__')}_seed{seed}.pt"
                atomic.torch.save(
                    {
                        "spec": spec,
                        "seed": seed,
                        "selected_step": selected_step,
                        "source_state_dict": source_state,
                        "state_state_dict": state_state,
                    },
                    path,
                )
                selected[key] = {
                    "selected_step": selected_step,
                    "selected_calibration_rmse_mv": score,
                    "checkpoint": path.name,
                    "checkpoint_sha256": atomic._sha256_file(path),
                }
            reports[str(seed)] = {
                "batch_stream_sha256": digest.hexdigest(),
                "trajectories": trajectories,
                "selected": selected,
            }
        report = {
            "schema_version": "06b-o-source-matrix-training-v1",
            "valid": all(
                len(rows) == len(self.config.matrix_checkpoints)
                for seed in reports.values()
                for rows in seed["trajectories"].values()
            ),
            "factor_arm_count": len(specs),
            "same_minibatch_stream_within_seed": True,
            "same_initialization_within_seed": True,
            "calibration_selected_checkpoint": True,
            "development_used_during_training": False,
            "reports": reports,
        }
        self.training_valid = bool(report["valid"])
        atomic._write_json(self.output_dir / "source_matrix_training.json", report)
        return report

    def evaluate_source_matrix(self) -> Dict[str, Any]:
        if not self.training_valid:
            raise RuntimeError("06b-o source matrix training is incomplete")
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        specs = self._specs()
        per_seed: Dict[str, Any] = {}
        total = len(specs) * len(self.config.pilot_seeds)
        progress = atomic._CompactProgress(
            "06b-o development and frozen controls",
            total,
            max(1, total // 12),
        )
        completed = 0
        for seed in self.config.pilot_seeds:
            rows: Dict[str, Any] = {}
            for spec in specs:
                key = self._spec_key(spec)
                pair = self.matrix_models[(key, seed)]
                authentic = self._evaluate_pair(
                    pair[0], pair[1], spec, "development", device
                )
                teacher_state = self._evaluate_pair(
                    pair[0],
                    pair[1],
                    spec,
                    "development",
                    device,
                    teacher_state_refresh=True,
                )
                relabelled = self._evaluate_pair(
                    pair[0],
                    pair[1],
                    spec,
                    "development",
                    device,
                    topology_name="relabelled",
                )
                no_axial = self._evaluate_pair(
                    pair[0],
                    pair[1],
                    spec,
                    "development",
                    device,
                    topology_name="no_axial",
                )
                spatial_shuffle = self._evaluate_pair(
                    pair[0],
                    pair[1],
                    spec,
                    "development",
                    device,
                    spatial_shuffle=True,
                )
                rows[key] = {
                    "authentic": authentic,
                    "teacher_state_refresh_upper_bound": teacher_state,
                    "relabelled_topology": relabelled,
                    "no_axial": no_axial,
                    "spatial_output_shuffle": spatial_shuffle,
                }
                completed += 1
                progress.update(completed, f"seed={seed} {key}")
            per_seed[str(seed)] = rows
        report = {
            "schema_version": "06b-o-source-matrix-development-v1",
            "valid": all(
                control["authentic"]["horizons"]["8_ms"]["nonfinite_voltage_count"] == 0
                and control["authentic"]["horizons"]["8_ms"]["nonfinite_state_count"] == 0
                for rows in per_seed.values()
                for control in rows.values()
            ),
            "role": "historically_reused_train_development",
            "teacher_state_refresh_selection_eligible": False,
            "counterfactuals_retrained": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "source_matrix_development.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    @staticmethod
    def _available_median(values: Sequence[Optional[float]]) -> Optional[float]:
        finite = [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ]
        return float(np.median(np.asarray(finite, dtype=np.float64))) if finite else None

    def _summary(self, key: str, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
        rows = [seed[key] for seed in evaluation["per_seed"].values()]
        authentic = [row["authentic"]["horizons"]["8_ms"] for row in rows]
        teacher = [
            row["teacher_state_refresh_upper_bound"]["horizons"]["8_ms"]
            for row in rows
        ]
        relabelled = [
            row["relabelled_topology"]["horizons"]["8_ms"] for row in rows
        ]
        no_axial = [row["no_axial"]["horizons"]["8_ms"] for row in rows]
        shuffled = [
            row["spatial_output_shuffle"]["horizons"]["8_ms"] for row in rows
        ]
        activity_names = authentic[0]["activity"]
        region_names = authentic[0]["region"]
        return {
            "median_voltage_rmse_mv": self._median(
                [row["voltage_rmse_mv"] for row in authentic]
            ),
            "median_gain_vs_persistence_fraction": self._median(
                [row["voltage_gain_vs_persistence_fraction"] for row in authentic]
            ),
            "median_STATE_gain_vs_persistence_fraction": self._median(
                [row["state_improvement_vs_persistence_fraction"] for row in authentic]
            ),
            "maximum_absolute_drift_mv": max(
                abs(row["endpoint_mean_drift_mv"]) for row in authentic
            ),
            "physical_voltage_violation_count": int(
                sum(row["physical_voltage_violation_count"] for row in authentic)
            ),
            "teacher_state_refresh_gain_fraction": self._median(
                [
                    1.0 - teacher_row["voltage_rmse_mv"] / max(row["voltage_rmse_mv"], 1e-12)
                    for row, teacher_row in zip(authentic, teacher)
                ]
            ),
            "relabelled_topology_degradation_fraction": self._median(
                [
                    relabelled_row["voltage_rmse_mv"] / max(row["voltage_rmse_mv"], 1e-12) - 1.0
                    for row, relabelled_row in zip(authentic, relabelled)
                ]
            ),
            "no_axial_degradation_fraction": self._median(
                [
                    no_axial_row["voltage_rmse_mv"] / max(row["voltage_rmse_mv"], 1e-12) - 1.0
                    for row, no_axial_row in zip(authentic, no_axial)
                ]
            ),
            "spatial_shuffle_degradation_fraction": self._median(
                [
                    shuffled_row["voltage_rmse_mv"] / max(row["voltage_rmse_mv"], 1e-12) - 1.0
                    for row, shuffled_row in zip(authentic, shuffled)
                ]
            ),
            "activity_gain_vs_persistence": {
                name: self._available_median(
                    [
                        row["activity"][name]["voltage_gain_vs_persistence_fraction"]
                        for row in authentic
                    ]
                )
                for name in activity_names
            },
            "region_gain_vs_persistence": {
                name: self._available_median(
                    [
                        row["region"][name]["voltage_gain_vs_persistence_fraction"]
                        for row in authentic
                    ]
                )
                for name in region_names
            },
            "per_horizon_voltage_rmse_mv": {
                horizon: self._median(
                    [
                        row["authentic"]["horizons"][horizon]["voltage_rmse_mv"]
                        for row in rows
                    ]
                )
                for horizon in rows[0]["authentic"]["horizons"]
            },
        }

    def _paired_main_effect(
        self,
        evaluation: Mapping[str, Any],
        axis: int,
        positive: str,
        negative: str,
    ) -> float:
        contrasts = []
        for spec in self._specs():
            if spec[axis] != positive:
                continue
            reference = list(spec)
            reference[axis] = negative
            positive_key = self._spec_key(spec)
            negative_key = self._spec_key(tuple(reference))
            for seed in evaluation["per_seed"].values():
                positive_rmse = seed[positive_key]["authentic"]["horizons"][
                    "8_ms"
                ]["voltage_rmse_mv"]
                negative_rmse = seed[negative_key]["authentic"]["horizons"][
                    "8_ms"
                ]["voltage_rmse_mv"]
                contrasts.append(
                    1.0 - positive_rmse / max(negative_rmse, 1e-12)
                )
        return self._median(contrasts)

    def _checkpoint_scaling(
        self, training: Mapping[str, Any]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for spec in self._specs():
            key = self._spec_key(spec)
            by_step: Dict[int, List[float]] = {
                int(step): [] for step in self.config.matrix_checkpoints
            }
            selected_steps = []
            for seed in training["reports"].values():
                selected_steps.append(int(seed["selected"][key]["selected_step"]))
                for row in seed["trajectories"][key]:
                    by_step[int(row["step"])].append(
                        float(
                            row["calibration"]["horizons"]["8_ms"][
                                "voltage_rmse_mv"
                            ]
                        )
                    )
            medians = {
                str(step): self._median(values) for step, values in by_step.items()
            }
            ordered = [medians[str(step)] for step in self.config.matrix_checkpoints]
            result[key] = {
                "median_calibration_rmse_by_step": medians,
                "relative_gain_first_to_last": 1.0
                - ordered[-1] / max(ordered[0], 1e-12),
                "nonincreasing_interval_fraction": float(
                    np.mean(
                        [right <= left for left, right in zip(ordered[:-1], ordered[1:])]
                    )
                ),
                "selected_steps": selected_steps,
            }
        return result

    def _paired_interaction(
        self,
        evaluation: Mapping[str, Any],
        first_axis: int,
        first_positive: str,
        first_negative: str,
        second_axis: int,
        second_positive: str,
        second_negative: str,
    ) -> float:
        """Median difference-in-differences on negative log 8 ms RMSE."""

        interactions = []
        remaining_axis = ({0, 1, 2} - {first_axis, second_axis}).pop()
        remaining_values = sorted({spec[remaining_axis] for spec in self._specs()})
        for remaining in remaining_values:
            for seed in evaluation["per_seed"].values():
                scores = {}
                for first in (first_negative, first_positive):
                    for second in (second_negative, second_positive):
                        spec = [None, None, None]
                        spec[first_axis] = first
                        spec[second_axis] = second
                        spec[remaining_axis] = remaining
                        rmse = seed[self._spec_key(tuple(spec))]["authentic"][
                            "horizons"
                        ]["8_ms"]["voltage_rmse_mv"]
                        scores[(first, second)] = -float(np.log(max(rmse, 1e-12)))
                interactions.append(
                    (scores[(first_positive, second_positive)] - scores[(first_positive, second_negative)])
                    - (scores[(first_negative, second_positive)] - scores[(first_negative, second_negative)])
                )
        return self._median(interactions)

    def finalize_effective_source_playground(
        self,
        audit: Mapping[str, Any],
        training: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summaries = {
            self._spec_key(spec): self._summary(self._spec_key(spec), evaluation)
            for spec in self._specs()
        }
        effects = {
            "hines_source_over_direct_voltage": self._paired_main_effect(
                evaluation, 0, HINES_SOURCE, DIRECT_VOLTAGE
            ),
            "predicted_STATE_feedback_over_frozen_boundary": self._paired_main_effect(
                evaluation, 1, PREDICTED_DYNAMIC_STATE, FROZEN_BOUNDARY_STATE
            ),
            "local_recurrence_over_instantaneous": self._paired_main_effect(
                evaluation, 2, LOCAL_RECURRENT, INSTANTANEOUS
            ),
        }
        interactions = {
            "source_x_STATE_feedback": self._paired_interaction(
                evaluation,
                0,
                HINES_SOURCE,
                DIRECT_VOLTAGE,
                1,
                PREDICTED_DYNAMIC_STATE,
                FROZEN_BOUNDARY_STATE,
            ),
            "source_x_temporal_memory": self._paired_interaction(
                evaluation,
                0,
                HINES_SOURCE,
                DIRECT_VOLTAGE,
                2,
                LOCAL_RECURRENT,
                INSTANTANEOUS,
            ),
            "STATE_feedback_x_temporal_memory": self._paired_interaction(
                evaluation,
                1,
                PREDICTED_DYNAMIC_STATE,
                FROZEN_BOUNDARY_STATE,
                2,
                LOCAL_RECURRENT,
                INSTANTANEOUS,
            ),
        }
        best_key = min(
            summaries, key=lambda name: summaries[name]["median_voltage_rmse_mv"]
        )
        best = summaries[best_key]
        best_spec = tuple(best_key.split("|"))
        critical = [
            value
            for name, value in best["region_gain_vs_persistence"].items()
            if value is not None
            and any(token in name.lower() for token in ("soma", "ais", "axon"))
        ]
        activity_thresholds = {
            "quiescent_lt_1mV": 0.0,
            "moderate_1_to_5mV": 0.0,
            "active_ge_5mV": self.config.minimum_active_gain_vs_persistence_fraction,
        }
        activity_safe = all(
            best["activity_gain_vs_persistence"].get(name) is not None
            and best["activity_gain_vs_persistence"][name] >= threshold
            for name, threshold in activity_thresholds.items()
        )
        safety = bool(
            best["median_gain_vs_persistence_fraction"]
            >= self.config.minimum_global_gain_vs_persistence_fraction
            and activity_safe
            and critical
            and all(value >= 0 for value in critical)
            and best["physical_voltage_violation_count"] == 0
        )
        source_effect = effects["hines_source_over_direct_voltage"]
        state_effect = effects["predicted_STATE_feedback_over_frozen_boundary"]
        memory_effect = effects["local_recurrence_over_instantaneous"]
        topology_signal = bool(
            best_spec[0] == HINES_SOURCE
            and best["relabelled_topology_degradation_fraction"]
            >= self.config.topology_materiality_fraction
            and best["spatial_shuffle_degradation_fraction"] > 0
        )
        if (
            best_spec[0] == HINES_SOURCE
            and source_effect >= self.config.source_materiality_fraction
            and safety
        ):
            diagnosis = "EFFECTIVE_MEMBRANE_SOURCE_CONTRACT_IDENTIFIED"
            next_step = "independent_train_support_source_operator_confirmation"
        elif source_effect >= self.config.source_materiality_fraction:
            diagnosis = "SOURCE_PARAMETERIZATION_SIGNAL_WITHOUT_ABSOLUTE_SAFETY"
            next_step = "atomic_source_regime_and_boundary_reassessment"
        elif max(summary["teacher_state_refresh_gain_fraction"] for summary in summaries.values()) >= self.config.state_materiality_fraction:
            diagnosis = "CAUSAL_STATE_CONTRACT_REMAINS_PRIMARY_LIMIT"
            next_step = "revise_causal_state_observability_before_voltage_operator"
        else:
            diagnosis = "NO_EFFECTIVE_SOURCE_OR_STATE_CONTRACT_SIGNAL"
            next_step = "reassess_supervised_physical_quantity_and_time_discretization"
        report = {
            "schema_version": "06b-o-final-report-v1",
            "valid": bool(audit.get("valid") and training.get("valid") and evaluation.get("valid")),
            "component_playground_grade": True,
            "diagnosis": diagnosis,
            "best_observed_arm": best_key,
            "selected_candidate": best_key if diagnosis == "EFFECTIVE_MEMBRANE_SOURCE_CONTRACT_IDENTIFIED" else None,
            "best_observed_arm_metrics": best,
            "factor_main_effects": effects,
            "paired_log_rmse_interactions": interactions,
            "checkpoint_scaling": self._checkpoint_scaling(training),
            "source_parameterization_signal": source_effect >= self.config.source_materiality_fraction,
            "predicted_STATE_feedback_signal": state_effect >= self.config.state_materiality_fraction,
            "temporal_memory_signal": memory_effect >= self.config.memory_materiality_fraction,
            "authentic_topology_signal_for_best_arm": topology_signal,
            "exact_source_reconstruction_valid": bool(audit.get("valid")),
            "teacher_source_oracle_selectable": False,
            "teacher_state_refresh_selectable": False,
            "summaries": summaries,
            "new_independent_confirmation_claimed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "DIRECT_VOLTAGE",
    "HINES_SOURCE",
    "FROZEN_BOUNDARY_STATE",
    "PREDICTED_DYNAMIC_STATE",
    "INSTANTANEOUS",
    "LOCAL_RECURRENT",
    "EXPECTED_06BN_ARCHIVE_SHA256",
    "EXPECTED_06BN_INDEX_SHA256",
    "EXPECTED_06BN_FINAL_SHA256",
    "CausalMembraneSourceCell",
    "EffectiveMembraneSourceConfig",
    "EffectiveMembraneSourcePlayground",
    "verified_06bn_artifact_root",
]
