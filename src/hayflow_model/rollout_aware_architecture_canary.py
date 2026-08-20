"""Rollout-aware GraphGRU versus ordered ConvGRU architecture canary.

This experiment is deliberately narrower than a HayFlow training run.  It
tests whether a recurrent model trained through its own predicted voltages can
remain useful for eight one-millisecond steps, and whether the authentic
morphology graph adds signal beyond an equally sized fixed-order convolution.
Only original train episodes are partitioned into fit, calibration and
development roles.  Existing validation/test splits and the sealed 05j-o
fresh test are never read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import (
    CompositeFlowmapBundle,
    CompositeTransitionStore,
    classify_regime,
)

from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root


EXPECTED_05KD_ARCHIVE_SHA256 = (
    "f6d036156f9fbec1344388759632f060dc7d6126a552274f6c97d290f4de8685"
)
EXPECTED_05KD_INDEX_SHA256 = (
    "23001abc8f49b4efd3379eb8ae5351343fc9082bb5293bbb35002562d3b5a2e8"
)
EXPECTED_05KD_FINAL_SHA256 = (
    "3d401bbf5b43d8ee705f1733b849c511445c642074813722b734020c0559cc70"
)

CAUSAL_DRIVE_FEATURES = (
    "ampa_state_increment",
    "nmda_state_increment",
    "inhibitory_state_increment",
    "released_quantity",
    "excitatory_event_count_scaled",
    "inhibitory_event_count_scaled",
    "release_success_count_scaled",
    "offset_mean_ms",
    "offset_second_moment_ms2",
    "somatic_current_na",
    "somatic_charge_na_ms",
    "weight_multiplier_sum_scaled",
)


@dataclass(frozen=True)
class RolloutAwareArchitectureCanaryConfig:
    horizons_ms: Tuple[int, ...] = (2, 4, 8)
    seeds: Tuple[int, ...] = (17, 29, 43)
    fit_groups_per_regime: int = 3
    calibration_groups_per_regime: int = 1
    development_groups_per_regime: int = 1
    windows_per_episode: int = 3
    hidden_width: int = 32
    epochs: int = 60
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    evaluation_interval: int = 5
    progress_interval: int = 10
    voltage_delta_limit_mv: float = 120.0
    minimum_improvement_vs_persistence_fraction: float = 0.10
    minimum_passing_seed_count: int = 2
    maximum_parameter_ratio: float = 1.05
    topology_materiality_fraction: float = 0.05
    physical_voltage_floor_mv: float = -150.0
    physical_voltage_ceiling_mv: float = 100.0
    role_seed: int = 51057

    def validate(self) -> None:
        if self.horizons_ms != (2, 4, 8):
            raise ValueError("05l horizons are preregistered at 2/4/8 ms")
        if self.seeds != (17, 29, 43):
            raise ValueError("05l uses the three registered architecture seeds")
        if min(
            self.fit_groups_per_regime,
            self.calibration_groups_per_regime,
            self.development_groups_per_regime,
            self.windows_per_episode,
            self.hidden_width,
            self.epochs,
            self.batch_size,
        ) <= 0:
            raise ValueError("05l positive integer configuration is invalid")
        if not 0 < self.learning_rate < 1 or self.weight_decay < 0:
            raise ValueError("05l optimizer configuration is invalid")
        if self.maximum_parameter_ratio < 1:
            raise ValueError("parameter ratio cannot be below one")
        if not 0 < self.minimum_improvement_vs_persistence_fraction < 1:
            raise ValueError("persistence improvement threshold is invalid")
        if self.minimum_passing_seed_count not in {2, 3}:
            raise ValueError("05l robust seed gate is invalid")
        if self.physical_voltage_floor_mv >= self.physical_voltage_ceiling_mv:
            raise ValueError("physical voltage interval is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "RolloutAwareArchitectureCanaryConfig":
        payload = dict(values)
        for name in ("horizons_ms", "seeds"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


def verified_architecture_failure_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="architecture_failure_reassessment_config.json",
        archive_sha256=EXPECTED_05KD_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05KD_INDEX_SHA256,
        final_sha256=EXPECTED_05KD_FINAL_SHA256,
    )


try:  # PyTorch is optional for data-contract-only imports.
    import torch
    from torch import nn
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    torch_functional = None


if nn is not None:

    class _RolloutAwareSegmentGRU(nn.Module):
        def __init__(
            self,
            segment_static: np.ndarray,
            parent_ids: np.ndarray,
            *,
            hidden_width: int,
            voltage_delta_limit_mv: float,
        ) -> None:
            super().__init__()
            static = np.asarray(segment_static, dtype=np.float32)
            parent = np.asarray(parent_ids, dtype=np.int64)
            self.segment_count = int(static.shape[0])
            self.hidden_width = int(hidden_width)
            self.voltage_delta_limit_mv = float(voltage_delta_limit_mv)
            self.register_buffer("segment_static", torch.as_tensor(static))
            self.register_buffer("parent_ids", torch.as_tensor(parent))
            child_ids = np.flatnonzero(parent != np.arange(len(parent))).astype(np.int64)
            self.register_buffer("child_ids", torch.as_tensor(child_ids))
            self.register_buffer("child_parent_ids", torch.as_tensor(parent[child_ids]))
            input_width = 1 + len(CAUSAL_DRIVE_FEATURES) + static.shape[1]
            self.input_encoder = nn.Sequential(
                nn.Linear(input_width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, hidden_width),
            )
            self.initial_encoder = nn.Linear(input_width, hidden_width)
            self.gru = nn.GRUCell(hidden_width, hidden_width)
            self.voltage_head = nn.Sequential(
                nn.Linear(hidden_width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, 1),
            )

        def _input(self, voltage: Any, drive: Any) -> Any:
            static = self.segment_static.unsqueeze(0).expand(voltage.shape[0], -1, -1)
            return torch.cat([(voltage / 100.0).unsqueeze(-1), drive, static], dim=-1)

        def initialise(self, voltage: Any) -> Any:
            drive = voltage.new_zeros(
                voltage.shape[0], self.segment_count, len(CAUSAL_DRIVE_FEATURES)
            )
            return torch.tanh(self.initial_encoder(self._input(voltage, drive)))

        def _mix(self, hidden: Any) -> Any:
            raise NotImplementedError

        def step(self, voltage: Any, hidden: Any, drive: Any) -> Tuple[Any, Any]:
            encoded = self.input_encoder(self._input(voltage, drive))
            mixed = self._mix(hidden)
            next_hidden = self.gru(
                (encoded + mixed).reshape(-1, self.hidden_width),
                hidden.reshape(-1, self.hidden_width),
            ).reshape_as(hidden)
            delta = self.voltage_delta_limit_mv * torch.tanh(
                self.voltage_head(next_hidden).squeeze(-1)
            )
            return voltage + delta, next_hidden

        def forward(self, voltage_t: Any, causal_drive: Any) -> Dict[str, Any]:
            voltage = voltage_t
            hidden = self.initialise(voltage_t)
            values = []
            for step in range(causal_drive.shape[1]):
                voltage, hidden = self.step(voltage, hidden, causal_drive[:, step])
                values.append(voltage)
            return {"voltage": torch.stack(values, dim=1), "hidden": hidden}


    class MorphologyGraphGRU(_RolloutAwareSegmentGRU):
        """Parent/children message passing with no dependency on segment order."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Three H-wide inputs make this parameter matched to a width-H,
            # kernel-3 Conv1D mixer.
            self.mixer = nn.Linear(3 * self.hidden_width, self.hidden_width)

        def _mix(self, hidden: Any) -> Any:
            parent = hidden[:, self.parent_ids]
            child_sum = torch.zeros_like(hidden)
            child_count = hidden.new_zeros(self.segment_count)
            if self.child_ids.numel():
                child_sum.index_add_(1, self.child_parent_ids, hidden[:, self.child_ids])
                child_count.index_add_(
                    0,
                    self.child_parent_ids,
                    hidden.new_ones(self.child_parent_ids.shape[0]),
                )
            child_mean = child_sum / child_count.clamp_min(1.0).view(1, -1, 1)
            return torch_functional.silu(
                self.mixer(torch.cat([hidden, parent, child_mean], dim=-1))
            )


    class OrderedConvGRUControl(_RolloutAwareSegmentGRU):
        """Parameter-matched kernel-3 control on the arbitrary segment order."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mixer = nn.Conv1d(
                self.hidden_width,
                self.hidden_width,
                kernel_size=3,
                padding=1,
            )

        def _mix(self, hidden: Any) -> Any:
            return torch_functional.silu(
                self.mixer(hidden.transpose(1, 2)).transpose(1, 2)
            )

else:  # pragma: no cover

    class MorphologyGraphGRU:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("05l requires PyTorch")

    class OrderedConvGRUControl:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("05l requires PyTorch")


def model_parameter_count(model: Any) -> int:
    return int(sum(value.numel() for value in model.parameters()))


def encode_causal_realized_drive(
    store: Any, indices: Sequence[int]
) -> np.ndarray:
    """Encode only pre-membrane realized events, never teacher boundary state."""

    segments = int(store.layout.segment_count)
    output = np.zeros(
        (len(indices), segments, len(CAUSAL_DRIVE_FEATURES)), dtype=np.float32
    )
    for row, logical_index in enumerate(indices):
        offset_sum = np.zeros(segments, dtype=np.float64)
        offset_square = np.zeros(segments, dtype=np.float64)
        offset_weight = np.zeros(segments, dtype=np.float64)
        for action in store.actions(int(logical_index), "U_realized"):
            if action.get("kind") == "somatic_current":
                offset = float(action.get("offset_ms", 0.0) or 0.0)
                duration = max(
                    0.0,
                    min(1.0 - offset, float(action.get("duration_ms", 0.0) or 0.0)),
                )
                amplitude = float(action.get("amplitude_na", 0.0) or 0.0)
                output[row, 0, 9] += amplitude
                output[row, 0, 10] += amplitude * duration
                continue
            segment = int(action["segment_id"])
            ampa = float(action.get("ampa_state_increment", 0.0) or 0.0)
            nmda = float(action.get("nmda_state_increment", 0.0) or 0.0)
            inhibitory = float(
                action.get("inhibitory_state_increment", 0.0) or 0.0
            )
            released = float(action.get("released_quantity", 0.0) or 0.0)
            excitatory = bool(ampa or nmda or action.get("synapse_type") == "ProbAMPANMDA2")
            output[row, segment, 0] += ampa
            output[row, segment, 1] += nmda
            output[row, segment, 2] += inhibitory
            output[row, segment, 3] += released
            output[row, segment, 4 if excitatory else 5] += 0.05
            output[row, segment, 6] += 0.05 * float(
                bool(action.get("release_success", False))
            )
            output[row, segment, 11] += 0.05 * float(
                action.get("weight_multiplier", 1.0) or 1.0
            )
            offset = float(action.get("offset_ms", 0.0) or 0.0)
            weight = max(abs(ampa) + abs(nmda) + abs(inhibitory), released, 1e-6)
            offset_sum[segment] += weight * offset
            offset_square[segment] += weight * offset * offset
            offset_weight[segment] += weight
        active = offset_weight > 0
        output[row, active, 7] = (
            offset_sum[active] / offset_weight[active]
        ).astype(np.float32)
        output[row, active, 8] = (
            offset_square[active] / offset_weight[active]
        ).astype(np.float32)
    return output


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _clean_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def disjoint_episode_roles(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    config: RolloutAwareArchitectureCanaryConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """Partition original train episodes while keeping seed/snapshot components intact."""

    rows = [dict(row) for row in episode_rows if str(row.get("split")) == "train"]
    if not rows:
        raise RuntimeError("05l found no original train episodes")
    union = _DisjointSet(len(rows))
    owners: Dict[Tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for name in ("seed", "snapshot_id", "snapshot_source"):
            value = _clean_key(row.get(name))
            if not value:
                continue
            key = (name, value)
            if key in owners:
                union.union(index, owners[key])
            else:
                owners[key] = index
    components: Dict[int, List[Dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        components.setdefault(union.find(index), []).append(row)
    priority = (
        "nmda_plateau",
        "nmda_spike",
        "calcium_spike",
        "backpropagating_ap",
        "somatic_spike",
        "axonal_spike",
    )
    grouped: Dict[str, List[List[Dict[str, Any]]]] = {}
    for component in components.values():
        labels = " ".join(_clean_key(row.get("event_labels")) for row in component)
        regime = next((name for name in priority if name in labels), None)
        if regime is None:
            first = component[0]
            regime = classify_regime(first, first.get("category", ""))
        grouped.setdefault(regime, []).append(component)
    roles: Dict[str, List[Dict[str, Any]]] = {
        "fit": [], "calibration": [], "development": []
    }
    requested = {
        "fit": config.fit_groups_per_regime,
        "calibration": config.calibration_groups_per_regime,
        "development": config.development_groups_per_regime,
    }
    for regime in sorted(grouped):
        components_for_regime = sorted(
            grouped[regime],
            key=lambda component: hashlib.sha256(
                (
                    str(config.role_seed)
                    + "|"
                    + "|".join(sorted(str(row["trajectory_id"]) for row in component))
                ).encode()
            ).hexdigest(),
        )
        cursor = 0
        for role in ("fit", "calibration", "development"):
            for component in components_for_regime[
                cursor : cursor + requested[role]
            ]:
                for row in component:
                    row = dict(row)
                    row["05l_regime"] = regime
                    roles[role].append(row)
            cursor += requested[role]
    if any(not rows_for_role for rows_for_role in roles.values()):
        raise RuntimeError("05l could not construct all three disjoint roles")
    return roles


class RolloutAwareArchitectureCanary:
    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: RolloutAwareArchitectureCanaryConfig,
        artifact_05kd_source: Path,
        *,
        code_revision: str,
    ) -> None:
        config.validate()
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir()
        self.config = config
        self.artifact_05kd_source = Path(artifact_05kd_source).resolve()
        self.code_revision = str(code_revision)
        self.store = CompositeTransitionStore(bundle)
        self.roles: Dict[str, List[Dict[str, Any]]] = {}
        self.windows: Dict[str, List[np.ndarray]] = {}
        self.materialized: Dict[str, Dict[str, np.ndarray]] = {}
        self.artifact_contract: Dict[str, Any] = {}
        self.prepare_report: Dict[str, Any] = {}

    def close(self) -> None:
        self.store.close()

    def _parent_ids(self) -> np.ndarray:
        return np.asarray(self.store.layout.parent_ids, dtype=np.int64)

    def _models(self, device: Any) -> Dict[str, Any]:
        kwargs = {
            "segment_static": self.store.layout.segment_static,
            "parent_ids": self._parent_ids(),
            "hidden_width": self.config.hidden_width,
            "voltage_delta_limit_mv": self.config.voltage_delta_limit_mv,
        }
        return {
            "morphology_graph_gru": MorphologyGraphGRU(**kwargs).to(device),
            "ordered_convgru_control": OrderedConvGRUControl(**kwargs).to(device),
        }

    def _episode_windows(self, indices: np.ndarray) -> List[np.ndarray]:
        horizon = max(self.config.horizons_ms)
        if len(indices) < horizon:
            return []
        starts = []
        for start in range(len(indices) - horizon + 1):
            candidate = indices[start : start + horizon]
            steps = self.store.metadata["step_index"][candidate]
            if np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                starts.append(start)
        if not starts:
            return []
        scored = []
        for start in starts:
            window = indices[start : start + horizon]
            score = sum(len(self.store.events(int(value))) for value in window)
            scored.append((score, start))
        selected = [max(scored, key=lambda row: (row[0], -row[1]))[1]]
        evenly = np.linspace(0, len(starts) - 1, self.config.windows_per_episode)
        selected.extend(starts[int(round(value))] for value in evenly)
        selected = sorted(set(selected))[: self.config.windows_per_episode]
        return [indices[start : start + horizon] for start in selected]

    def _build_windows(self) -> Dict[str, List[np.ndarray]]:
        windows: Dict[str, List[np.ndarray]] = {}
        for role, rows in self.roles.items():
            values: List[np.ndarray] = []
            for row in rows:
                indices = self.store.trajectory_indices[str(row["trajectory_id"])]
                values.extend(self._episode_windows(indices))
            if not values:
                raise RuntimeError(f"05l role {role} has no contiguous 8 ms windows")
            windows[role] = values
        return windows

    def _materialize_role(self, windows: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        initial, target, drive = [], [], []
        for window in windows:
            initial.append(
                self.store.read_state(
                    [int(window[0])], "t", categories=("voltage",)
                )[0]
            )
            target.append(
                self.store.read_state(
                    window, "t_plus_1", categories=("voltage",)
                )
            )
            drive.append(encode_causal_realized_drive(self.store, window))
        return {
            "initial_voltage": np.asarray(initial, dtype=np.float32),
            "target_voltage": np.asarray(target, dtype=np.float32),
            "causal_drive": np.asarray(drive, dtype=np.float32),
        }

    @staticmethod
    def _role_overlap(roles: Mapping[str, Sequence[Mapping[str, Any]]], name: str) -> Dict[str, List[str]]:
        values = {
            role: {
                _clean_key(row.get(name))
                for row in rows
                if _clean_key(row.get(name))
            }
            for role, rows in roles.items()
        }
        result = {}
        keys = sorted(values)
        for left_position, left in enumerate(keys):
            for right in keys[left_position + 1 :]:
                overlap = sorted(values[left] & values[right])
                if overlap:
                    result[f"{left}__{right}"] = overlap
        return result

    def prepare(self) -> Dict[str, Any]:
        _, final, contract = verified_architecture_failure_artifact_root(
            self.artifact_05kd_source,
            self.output_dir.parent / ".05l_artifact_cache" / "05kd",
        )
        blockers = []
        if not final.get("valid"):
            blockers.append("05k-d artifact is invalid")
        if final.get("diagnosis") != "RETIRE_FREE_RUNNING_H2_LATENT_RECURRENCE":
            blockers.append("05k-d did not retire H2 recurrence")
        if not final.get("rollout_aware_architecture_canary_authorized"):
            blockers.append("05k-d did not authorize 05l")
        if final.get("next_step") != "05l_rollout_aware_graphgru_vs_convgru_canary":
            blockers.append("05k-d next step is not 05l")
        self.artifact_contract = contract
        self.roles = disjoint_episode_roles(
            self.store.episode_rows, config=self.config
        )
        seed_overlap = self._role_overlap(self.roles, "seed")
        snapshot_overlap = self._role_overlap(self.roles, "snapshot_id")
        trajectory_overlap = self._role_overlap(self.roles, "trajectory_id")
        if seed_overlap or snapshot_overlap or trajectory_overlap:
            blockers.append("05l role isolation failed")
        self.windows = self._build_windows()
        self.materialized = {
            role: self._materialize_role(values)
            for role, values in self.windows.items()
        }
        if torch is None:
            blockers.append("PyTorch is unavailable")
            counts = {}
            ratio = math.inf
        else:
            models = self._models(torch.device("cpu"))
            counts = {name: model_parameter_count(model) for name, model in models.items()}
            ratio = max(counts.values()) / min(counts.values())
            if ratio > self.config.maximum_parameter_ratio:
                blockers.append("GraphGRU and ConvGRU are not parameter matched")
        report = {
            "schema_version": "05l-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05kd": contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "dataset_transition_count": self.bundle.transition_count,
            "source_split_used": "train_only",
            "excluded_splits": sorted(
                set(self.store.split_indices) - {"train"}
            ),
            "fresh_05jo_loaded": False,
            "teacher_future_state_used_as_model_input": False,
            "causal_drive_features": list(CAUSAL_DRIVE_FEATURES),
            "role_episode_counts": {
                role: len(rows) for role, rows in self.roles.items()
            },
            "role_window_counts": {
                role: len(values) for role, values in self.windows.items()
            },
            "role_regime_counts": {
                role: {
                    regime: sum(row["05l_regime"] == regime for row in rows)
                    for regime in sorted({row["05l_regime"] for row in rows})
                }
                for role, rows in self.roles.items()
            },
            "role_overlap": {
                "seed": seed_overlap,
                "snapshot": snapshot_overlap,
                "trajectory": trajectory_overlap,
            },
            "parameter_counts": counts,
            "parameter_ratio": ratio,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "rollout_aware_architecture_canary_config.json",
            {
                "schema_version": "05l-config-v1",
                "config": asdict(self.config),
                "artifact_05kd": contract,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05l preflight failed: {blockers}")
        return report

    def _tensor_role(self, role: str, device: Any) -> Dict[str, Any]:
        values = self.materialized[role]
        return {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in values.items()
        }

    @staticmethod
    def _loss(prediction: Any, target: Any) -> Any:
        path = torch_functional.smooth_l1_loss(
            (prediction - target) / 10.0,
            torch.zeros_like(prediction),
        )
        endpoint = torch_functional.smooth_l1_loss(
            (prediction[:, -1] - target[:, -1]) / 10.0,
            torch.zeros_like(prediction[:, -1]),
        )
        peak = torch_functional.smooth_l1_loss(
            prediction.amax(dim=2) / 10.0,
            target.amax(dim=2) / 10.0,
        )
        drift = torch.mean(torch.abs(torch.mean(prediction - target, dim=2))) / 10.0
        return 0.5 * path + endpoint + 0.25 * peak + 0.1 * drift

    def _evaluate_arrays(
        self, model: Any, values: Mapping[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        model.eval()
        result: Dict[str, Dict[str, float]] = {}
        with torch.no_grad():
            for horizon in self.config.horizons_ms:
                prediction = model(
                    values["initial_voltage"], values["causal_drive"][:, :horizon]
                )["voltage"]
                target = values["target_voltage"][:, :horizon]
                endpoint_error = prediction[:, -1] - target[:, -1]
                persistence_error = values["initial_voltage"] - target[:, -1]
                violations = (
                    (prediction < self.config.physical_voltage_floor_mv)
                    | (prediction > self.config.physical_voltage_ceiling_mv)
                )
                result[str(horizon)] = {
                    "endpoint_rmse_mv": float(
                        torch.sqrt(torch.mean(endpoint_error.square())).cpu()
                    ),
                    "path_rmse_mv": float(
                        torch.sqrt(torch.mean((prediction - target).square())).cpu()
                    ),
                    "endpoint_mean_drift_mv": float(torch.mean(endpoint_error).cpu()),
                    "persistence_endpoint_rmse_mv": float(
                        torch.sqrt(torch.mean(persistence_error.square())).cpu()
                    ),
                    "physical_voltage_violation_count": int(violations.sum().cpu()),
                    "finite": bool(torch.isfinite(prediction).all().cpu()),
                }
        return result

    def _train_one(
        self, family: str, seed: int, device: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self._models(device)[family]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        fit = self._tensor_role("fit", device)
        calibration = self._tensor_role("calibration", device)
        rng = np.random.default_rng(seed + 7001)
        best_score = math.inf
        best_state = None
        best_epoch = -1
        history: List[Dict[str, Any]] = []
        started = time.monotonic()
        window_count = fit["initial_voltage"].shape[0]
        for epoch in range(self.config.epochs):
            if epoch < self.config.epochs // 3:
                horizon = 2
            elif epoch < 2 * self.config.epochs // 3:
                horizon = 4
            else:
                horizon = 8
            order = rng.permutation(window_count)
            losses = []
            gradients = []
            model.train()
            for start in range(0, window_count, self.config.batch_size):
                positions = torch.as_tensor(
                    order[start : start + self.config.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    fit["initial_voltage"].index_select(0, positions),
                    fit["causal_drive"].index_select(0, positions)[:, :horizon],
                )["voltage"]
                target = fit["target_voltage"].index_select(0, positions)[:, :horizon]
                loss = self._loss(prediction, target)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.gradient_clip_norm
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradients.append(float(gradient.detach().cpu()))
            evaluate = (
                epoch == 0
                or (epoch + 1) % self.config.evaluation_interval == 0
                or epoch + 1 == self.config.epochs
            )
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "curriculum_horizon_ms": horizon,
                "fit_loss": float(np.mean(losses)),
                "gradient_norm_pre_clip": float(np.mean(gradients)),
            }
            if evaluate:
                calibration_metrics = self._evaluate_arrays(model, calibration)
                score = calibration_metrics["8"]["endpoint_rmse_mv"]
                row["calibration_endpoint_rmse_8ms_mv"] = score
                row["calibration_drift_8ms_mv"] = calibration_metrics["8"][
                    "endpoint_mean_drift_mv"
                ]
                if score < best_score:
                    best_score = score
                    best_epoch = epoch + 1
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            history.append(row)
            if (
                epoch == 0
                or (epoch + 1) % self.config.progress_interval == 0
                or epoch + 1 == self.config.epochs
            ):
                elapsed = time.monotonic() - started
                eta = elapsed / (epoch + 1) * (self.config.epochs - epoch - 1)
                metric = row.get("calibration_endpoint_rmse_8ms_mv", math.nan)
                print(
                    f"[HayFlow 05l][{family} seed={seed}] {epoch + 1}/{self.config.epochs} "
                    f"ETA {eta / 60:.1f} min loss={row['fit_loss']:.4g} "
                    f"cal8={metric:.4g}",
                    flush=True,
                )
        if best_state is None:
            raise RuntimeError("05l produced no calibration checkpoint")
        model.load_state_dict(best_state)
        checkpoint = self.checkpoint_dir / f"{family}-seed{seed}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "family": family,
                "seed": seed,
                "best_epoch": best_epoch,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
            checkpoint,
        )
        report = {
            "family": family,
            "seed": seed,
            "parameter_count": model_parameter_count(model),
            "best_epoch": best_epoch,
            "calibration_selection_rmse_8ms_mv": best_score,
            "checkpoint": checkpoint.relative_to(self.output_dir).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "history": history,
        }
        return model, report

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None:
            raise RuntimeError("05l requires PyTorch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        development = self._tensor_role("development", device)
        families = ("morphology_graph_gru", "ordered_convgru_control")
        runs: Dict[str, Dict[str, Any]] = {family: {} for family in families}
        for family in families:
            for seed in self.config.seeds:
                model, report = self._train_one(family, seed, device)
                metrics = self._evaluate_arrays(model, development)
                eight = metrics["8"]
                improvement = 1.0 - eight["endpoint_rmse_mv"] / max(
                    eight["persistence_endpoint_rmse_mv"], 1e-12
                )
                passed = bool(
                    all(row["finite"] for row in metrics.values())
                    and sum(
                        row["physical_voltage_violation_count"]
                        for row in metrics.values()
                    )
                    == 0
                    and improvement
                    >= self.config.minimum_improvement_vs_persistence_fraction
                )
                report.update(
                    development=metrics,
                    development_improvement_vs_persistence_fraction=improvement,
                    passed=passed,
                )
                runs[family][str(seed)] = report
        passing = {
            family: sum(bool(row["passed"]) for row in values.values())
            for family, values in runs.items()
        }
        median_rmse = {
            family: float(
                np.median(
                    [row["development"]["8"]["endpoint_rmse_mv"] for row in values.values()]
                )
            )
            for family, values in runs.items()
        }
        graph_robust = passing["morphology_graph_gru"] >= self.config.minimum_passing_seed_count
        conv_robust = passing["ordered_convgru_control"] >= self.config.minimum_passing_seed_count
        graph_gain = 1.0 - median_rmse["morphology_graph_gru"] / max(
            median_rmse["ordered_convgru_control"], 1e-12
        )
        if graph_robust and graph_gain >= self.config.topology_materiality_fraction:
            diagnosis = "ROLLOUT_AWARE_GRAPH_RECURRENCE_SIGNAL"
            next_step = "05m_expanded_graphgru_development_canary"
        elif graph_robust and conv_robust:
            diagnosis = "ROLLOUT_AWARE_RECURRENCE_SIGNAL_TOPOLOGY_UNRESOLVED"
            next_step = "05m_topology_controlled_recurrence_expansion"
        elif conv_robust and not graph_robust:
            diagnosis = "ORDERED_RECURRENCE_SIGNAL_GRAPH_NOT_SUPPORTED"
            next_step = "05m_graph_message_passing_reassessment"
        else:
            diagnosis = "NO_ROLLOUT_AWARE_RECURRENCE_SIGNAL"
            next_step = "05m_recurrent_state_architecture_reassessment"
        report = {
            "schema_version": "05l-final-report-v1",
            "valid": True,
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05kd": self.artifact_contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "preflight": self.prepare_report,
            "training_contract": {
                "rollout_aware_closed_loop_voltage": True,
                "initial_teacher_voltage_only": True,
                "teacher_forcing_inside_window": False,
                "primary_input": "U_realized_strictly_causal",
                "teacher_future_state_used_as_input": False,
                "fit_role_selects_gradients": True,
                "calibration_role_selects_checkpoint": True,
                "development_role_used_once_after_freeze": True,
                "validation_or_test_split_loaded": False,
                "fresh_05jo_loaded": False,
            },
            "runs": runs,
            "passing_seed_count": passing,
            "median_development_rmse_8ms_mv": median_rmse,
            "graph_improvement_vs_convgru_fraction": graph_gain,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
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
                "schema_version": "05l-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "CAUSAL_DRIVE_FEATURES",
    "EXPECTED_05KD_ARCHIVE_SHA256",
    "EXPECTED_05KD_FINAL_SHA256",
    "EXPECTED_05KD_INDEX_SHA256",
    "MorphologyGraphGRU",
    "OrderedConvGRUControl",
    "RolloutAwareArchitectureCanary",
    "RolloutAwareArchitectureCanaryConfig",
    "disjoint_episode_roles",
    "encode_causal_realized_drive",
    "model_parameter_count",
    "verified_architecture_failure_artifact_root",
]
