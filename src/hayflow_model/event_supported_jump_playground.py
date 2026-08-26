"""06b-q: event-supported jump operators and passive-default safety.

This train-only playground follows the registered 06b-p failure.  It replaces
the misleading ``exact_events`` label (which denoted 42 pooled statistics) by
an actual ordered, receptor-resolved per-segment event tensor.  A paired 3x2
matrix crosses event representation with sparse-feature normalization.  A
second, adaptive 2-arm stage tests whether a passive-default residual gate
protects quiet and moderate regimes.  Mechanism-current factorization is only
audited: it is never trained unless the stored dataset contains the causal
time-integrated teacher currents required by that target.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .atomic_effective_source_learnability import (
    GLOBAL_P99,
    HYBRID,
    NET_EFFECTIVE_SOURCE,
    REGION_P99,
    AtomicEffectiveSourceConfig,
    AtomicEffectiveSourceLearnability,
)
from .effective_membrane_source_playground import CausalMembraneSourceCell


EXPECTED_06BP_ARCHIVE_SHA256 = (
    "0a2d53f9ffe8ed64ac8823cac1b95cd632576b501791879df2ff665cf5cc190e"
)
EXPECTED_06BP_INDEX_SHA256 = (
    "30b492ad00018b9e5b2d907c698ee9156a99b76dbf636fc67d2430507d02ca42"
)
EXPECTED_06BP_FINAL_SHA256 = (
    "731856becd6ff13aa2cc9a8875991497d37bffa9c3af6e66bbe2c80017d00dde"
)

MOMENT_POOL = "moment_pool"
DEEPSET_EVENTS = "deepset_events"
CHRONOLOGICAL_JUMP = "chronological_jump"
EVENT_REPRESENTATIONS = (MOMENT_POOL, DEEPSET_EVENTS, CHRONOLOGICAL_JUMP)

LEGACY_ALL_ENTRY_P99 = "legacy_all_entry_p99"
NONZERO_ROBUST_LOG = "nonzero_robust_log"
EVENT_NORMALIZATIONS = (LEGACY_ALL_ENTRY_P99, NONZERO_ROBUST_LOG)

UNGATED_RESIDUAL = "ungated_residual"
PASSIVE_DEFAULT_GATE = "passive_default_gate"
SAFETY_GATES = (UNGATED_RESIDUAL, PASSIVE_DEFAULT_GATE)

EVENT_FEATURE_NAMES = (
    "offset_ms",
    "released_quantity",
    "weight_multiplier",
    "gmax_us",
    "ampa_state_increment",
    "nmda_state_increment",
    "gabaa_state_increment",
    "gabab_state_increment",
    "has_ampa",
    "has_nmda",
    "has_gabaa",
    "has_gabab",
    "somatic_amplitude_na",
    "somatic_duration_ms",
    "is_somatic_current",
    "order_fraction",
)


def verified_06bp_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """Verify the exact registered 06b-p result, including every member."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-p source must be a ZIP or extracted directory")
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
        if atomic._sha256_file(path) == EXPECTED_06BP_INDEX_SHA256
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one exact 06b-p artifact; found {len(roots)}")
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
        raise RuntimeError(f"06b-p indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BP_FINAL_SHA256:
        raise RuntimeError("06b-p final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("valid") is not True
        or final.get("selected_candidate") is not None
        or final.get("validation_state_accessed") is not False
        or final.get("test_state_accessed") is not False
    ):
        raise RuntimeError("06b-p result does not authorize 06b-q")
    if source.is_file() and archive_hash != EXPECTED_06BP_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, final, {
        "valid": True,
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BP_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BP_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "reported_diagnosis": final.get("diagnosis"),
    }


def _component_key(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _train_components(episode_rows: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Connected components under seed/snapshot identity, without labels."""

    rows = [dict(row) for row in episode_rows if str(row.get("split")) == "train"]
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for name in ("seed", "snapshot_id", "snapshot_source"):
        seen: Dict[str, int] = {}
        for index, row in enumerate(rows):
            value = _component_key(row, name)
            if not value:
                continue
            if value in seen:
                union(index, seen[value])
            else:
                seen[value] = index
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(find(index), []).append(row)
    return list(grouped.values())


def has_realized_event(store: Any, logical_index: int) -> bool:
    """Causal event support derived only from U_realized, never outcomes.

    A failed stochastic release still belongs to the causal event contract:
    its timestamp, receptor identity and realized zero release are known before
    the membrane update.  It must therefore count as event support.
    """

    for action in store.actions(int(logical_index), "U_realized"):
        if action.get("kind") == "somatic_current":
            if abs(float(action.get("amplitude_na") or 0.0)) > 0.0:
                return True
        elif action.get("kind") == "synaptic_event":
            return True
    return False


def build_event_supported_roles(
    store: Any,
    *,
    role_seed: int,
    component_targets: Mapping[str, Tuple[int, int]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Create seed/snapshot-disjoint roles stratified only by causal input.

    Each target is ``(event-positive components, no-event components)``.  The
    component graph is formed before support labels are evaluated, preventing
    leakage through shared seeds or snapshots.
    """

    components = _train_components(store.episode_rows)
    supported: Dict[bool, List[List[Dict[str, Any]]]] = {True: [], False: []}
    summaries = []
    for component in components:
        indices = np.concatenate(
            [store.trajectory_indices[str(row["trajectory_id"])] for row in component]
        )
        positive = any(has_realized_event(store, int(index)) for index in indices)
        digest = hashlib.sha256(
            f"{role_seed}|06bq|{'|'.join(sorted(str(r['trajectory_id']) for r in component))}".encode()
        ).hexdigest()
        component.sort(key=lambda row: str(row["trajectory_id"]))
        supported[positive].append(component)
        summaries.append(
            {
                "digest": digest,
                "positive": positive,
                "episode_count": len(component),
                "transition_count": int(len(indices)),
            }
        )
    for positive in (False, True):
        supported[positive].sort(
            key=lambda component: hashlib.sha256(
                f"{role_seed}|06bq-order|{'|'.join(str(r['trajectory_id']) for r in component)}".encode()
            ).hexdigest()
        )
    roles = {role: [] for role in component_targets}
    cursors = {False: 0, True: 0}
    for role, (positive_count, negative_count) in component_targets.items():
        for positive, count in ((True, positive_count), (False, negative_count)):
            chosen = supported[positive][cursors[positive] : cursors[positive] + count]
            if len(chosen) != count:
                raise RuntimeError(
                    f"06b-q lacks {'event' if positive else 'no-event'} components for {role}"
                )
            cursors[positive] += count
            for component in chosen:
                for source in component:
                    row = dict(source)
                    row["06bq_role"] = role
                    row["06bq_causal_support"] = "event" if positive else "no_event"
                    roles[role].append(row)
    return roles, {
        "valid": all(roles.values()),
        "source_split": "train_only",
        "stratification_source": "U_realized_only",
        "outcome_labels_used": False,
        "available_component_counts": {
            "event": len(supported[True]), "no_event": len(supported[False])
        },
        "selected_episode_counts": {role: len(rows) for role, rows in roles.items()},
        "components": summaries,
    }


def ordered_event_tensor(
    store: Any,
    indices: Sequence[int],
    *,
    max_events_per_segment: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode authentic ordered U_realized as a padded segment-local sequence."""

    batch = len(indices)
    segments = store.layout.segment_count
    width = len(EVENT_FEATURE_NAMES)
    values = np.zeros((batch, segments, max_events_per_segment, width), dtype=np.float32)
    mask = np.zeros((batch, segments, max_events_per_segment), dtype=bool)
    counts = np.zeros((batch, segments), dtype=np.int16)
    synapses = {int(row["id"]): row for row in store.layout.synapses}
    maximum = 0
    for batch_index, logical_index in enumerate(indices):
        actions = store.actions(int(logical_index), "U_realized")
        denominator = max(1, len(actions) - 1)
        for order, action in enumerate(actions):
            if action.get("kind") == "somatic_current":
                segment = 0
                vector = [
                    float(action.get("offset_ms") or 0.0), 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    float(action.get("amplitude_na") or 0.0),
                    float(action.get("duration_ms") or 0.0), 1.0,
                    float(order) / denominator,
                ]
            else:
                synapse_id = int(action["synapse_id"])
                row = synapses[synapse_id]
                segment = int(row["segment_id"])
                components = {str(item["name"]).upper() for item in row.get("components", [])}
                inhibitory = float(action.get("inhibitory_state_increment") or 0.0)
                gabaa = inhibitory if "GABAA" in components else 0.0
                gabab = inhibitory if "GABAB" in components else 0.0
                vector = [
                    float(action.get("offset_ms") or 0.0),
                    float(action.get("released_quantity") or 0.0),
                    float(action.get("weight_multiplier") or 1.0),
                    float(action.get("gmax") or row.get("parameters", {}).get("gmax", 0.0) or 0.0),
                    float(action.get("ampa_state_increment") or 0.0),
                    float(action.get("nmda_state_increment") or 0.0), gabaa, gabab,
                    float("AMPA" in components), float("NMDA" in components),
                    float("GABAA" in components), float("GABAB" in components),
                    0.0, 0.0, 0.0, float(order) / denominator,
                ]
            position = int(counts[batch_index, segment])
            maximum = max(maximum, position + 1)
            if position >= max_events_per_segment:
                raise RuntimeError(
                    f"ordered event tensor truncation: need {position + 1}, configured {max_events_per_segment}"
                )
            values[batch_index, segment, position] = vector
            mask[batch_index, segment, position] = True
            counts[batch_index, segment] += 1
    return values, mask, counts


def fit_event_normalizers(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    quantile: float,
    floor: float,
) -> Dict[str, np.ndarray]:
    """Fit legacy and sparse-aware scales on fit data only."""

    absolute = np.abs(values.astype(np.float64))
    legacy = np.maximum(np.quantile(absolute, quantile, axis=(0, 1, 2)), floor)
    robust = np.ones(values.shape[-1], dtype=np.float64)
    for feature in range(values.shape[-1]):
        observed = absolute[..., feature][mask]
        observed = observed[observed > 0.0]
        if len(observed):
            robust[feature] = max(float(np.quantile(observed, 0.5)), floor)
    for name in (
        "has_ampa", "has_nmda", "has_gabaa", "has_gabab", "is_somatic_current"
    ):
        robust[EVENT_FEATURE_NAMES.index(name)] = 1.0
    return {
        LEGACY_ALL_ENTRY_P99: legacy.astype(np.float32),
        NONZERO_ROBUST_LOG: robust.astype(np.float32),
    }


def normalize_event_tensor(
    values: Any,
    mask: Any,
    scale: Any,
    mode: str,
    *,
    clip: float,
) -> Any:
    normalized = values / scale[None, None, None, :]
    if mode == NONZERO_ROBUST_LOG:
        normalized = atomic.torch.sign(normalized) * atomic.torch.log1p(normalized.abs())
    elif mode != LEGACY_ALL_ENTRY_P99:
        raise ValueError(mode)
    return normalized.clamp(-clip, clip) * mask[..., None]


@dataclass(frozen=True)
class EventSupportedJumpConfig(AtomicEffectiveSourceConfig):
    event_representations: Tuple[str, ...] = EVENT_REPRESENTATIONS
    event_normalizations: Tuple[str, ...] = EVENT_NORMALIZATIONS
    safety_gates: Tuple[str, ...] = SAFETY_GATES
    support_fit_components: Tuple[int, int] = (3, 3)
    support_calibration_components: Tuple[int, int] = (1, 1)
    support_development_components: Tuple[int, int] = (1, 1)
    event_max_per_segment: int = 16
    event_embedding_width: int = 12
    event_scale_quantile: float = 0.99
    event_scale_floor: float = 1e-6
    event_normalized_clip: float = 12.0
    jump_training_steps: int = 300
    jump_checkpoints: Tuple[int, ...] = (0, 25, 75, 150, 300)
    jump_batch_transition_count: int = 32
    gradient_probe_steps: Tuple[int, ...] = (25, 150)
    safety_training_steps: int = 200
    safety_checkpoints: Tuple[int, ...] = (0, 50, 100, 200)
    synthetic_training_steps: int = 120
    synthetic_checkpoints: Tuple[int, ...] = (0, 20, 60, 120)
    minimum_event_support_fraction: float = 0.01
    minimum_recursive_gain_fraction: float = 0.02
    passive_gate_initial_bias: float = -4.0

    def validate(self) -> None:
        super().validate()
        if tuple(self.event_representations) != EVENT_REPRESENTATIONS:
            raise ValueError("06b-q representation matrix changed")
        if tuple(self.event_normalizations) != EVENT_NORMALIZATIONS:
            raise ValueError("06b-q normalization matrix changed")
        if tuple(self.safety_gates) != SAFETY_GATES:
            raise ValueError("06b-q safety gate changed")
        if self.jump_checkpoints[0] != 0 or self.jump_checkpoints[-1] != self.jump_training_steps:
            raise ValueError("jump checkpoints must span the budget")
        if self.safety_checkpoints[0] != 0 or self.safety_checkpoints[-1] != self.safety_training_steps:
            raise ValueError("safety checkpoints must span the budget")
        if self.event_max_per_segment <= 0 or self.event_embedding_width <= 0:
            raise ValueError("event dimensions must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EventSupportedJumpConfig":
        payload = dict(values)
        tuple_names = {
            "pilot_seeds", "output_parameterizations", "state_feedback_contracts",
            "temporal_contracts", "matrix_checkpoints", "rollout_horizons_ms",
            "scaling_modes", "objectives", "atomic_checkpoints", "input_contracts",
            "physical_targets", "fragment_checkpoints", "substep_audit_dt_ms",
            "voltage_sensitivity_perturbations_mv", "event_representations",
            "event_normalizations", "safety_gates", "support_fit_components",
            "support_calibration_components", "support_development_components",
            "jump_checkpoints", "gradient_probe_steps", "safety_checkpoints",
            "synthetic_checkpoints",
        }
        for name in tuple_names:
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


if atomic.nn is not None:

    class EventEncoderBank(atomic.nn.Module):
        """All event encoders are always instantiated for parameter matching."""

        def __init__(self, feature_width: int, embedding_width: int) -> None:
            super().__init__()
            self.feature_width = int(feature_width)
            self.embedding_width = int(embedding_width)
            self.moment = atomic.nn.Sequential(
                atomic.nn.Linear(2 * feature_width, embedding_width), atomic.nn.SiLU(),
                atomic.nn.Linear(embedding_width, embedding_width),
            )
            self.phi = atomic.nn.Sequential(
                atomic.nn.Linear(feature_width, embedding_width), atomic.nn.SiLU(),
                atomic.nn.Linear(embedding_width, embedding_width),
            )
            self.rho = atomic.nn.Sequential(
                atomic.nn.Linear(embedding_width, embedding_width), atomic.nn.SiLU(),
                atomic.nn.Linear(embedding_width, embedding_width),
            )
            self.jump_input = atomic.nn.Linear(feature_width, embedding_width)
            self.jump = atomic.nn.GRUCell(embedding_width, embedding_width)

        def forward(self, values: Any, mask: Any, mode: str) -> Any:
            weights = mask[..., None]
            count = weights.sum(dim=2).clamp_min(1.0)
            if mode == MOMENT_POOL:
                summed = (values * weights).sum(dim=2)
                mean = summed / count
                return self.moment(atomic.torch.cat((summed, mean), dim=-1))
            if mode == DEEPSET_EVENTS:
                pooled = (self.phi(values) * weights).sum(dim=2)
                return self.rho(pooled)
            if mode != CHRONOLOGICAL_JUMP:
                raise ValueError(mode)
            batch, segments, events, _ = values.shape
            hidden = atomic.torch.zeros(
                batch * segments, self.embedding_width,
                dtype=values.dtype, device=values.device,
            )
            flattened = values.reshape(batch * segments, events, -1)
            flat_mask = mask.reshape(batch * segments, events)
            for event in range(events):
                candidate = self.jump(self.jump_input(flattened[:, event]), hidden)
                hidden = atomic.torch.where(flat_mask[:, event, None], candidate, hidden)
            return hidden.reshape(batch, segments, -1)


    class EventConditionedSourceCell(atomic.nn.Module):
        """Matched source model with a selectable event encoder and safety gate."""

        def __init__(
            self,
            *,
            base_feature_width: int,
            event_feature_width: int,
            event_embedding_width: int,
            region_count: int,
            region_width: int,
            hidden_width: int,
            output_limit: float,
            passive_gate_initial_bias: float,
        ) -> None:
            super().__init__()
            self.events = EventEncoderBank(event_feature_width, event_embedding_width)
            self.source = CausalMembraneSourceCell(
                base_feature_width + event_embedding_width,
                region_count, region_width, hidden_width, output_limit,
            )
            self.gate = atomic.nn.Linear(hidden_width, 1)
            atomic.nn.init.zeros_(self.gate.weight)
            atomic.nn.init.constant_(self.gate.bias, passive_gate_initial_bias)

        def forward(
            self,
            base: Any,
            event_values: Any,
            event_mask: Any,
            region_ids: Any,
            *,
            representation: str,
            gate_mode: str,
        ) -> Tuple[Any, Any]:
            event_embedding = self.events(event_values, event_mask, representation)
            hidden = atomic.torch.zeros(
                base.shape[0], base.shape[1], self.source.hidden_width,
                dtype=base.dtype, device=base.device,
            )
            output, next_hidden = self.source(
                atomic.torch.cat((base, event_embedding), dim=-1),
                region_ids, hidden, recurrent=False,
            )
            if gate_mode == PASSIVE_DEFAULT_GATE:
                gate = atomic.torch.sigmoid(self.gate(next_hidden).squeeze(-1))
                output = gate * output
            elif gate_mode != UNGATED_RESIDUAL:
                raise ValueError(gate_mode)
            return output, next_hidden


else:  # pragma: no cover

    class EventEncoderBank:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06b-q requires PyTorch")

    class EventConditionedSourceCell(EventEncoderBank):
        pass


class EventSupportedJumpPlayground(AtomicEffectiveSourceLearnability):
    """Run the paired support/representation/normalization experiment."""

    config: EventSupportedJumpConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: EventSupportedJumpConfig,
        artifact_05t_source: Path,
        artifact_06bn_source: Path,
        artifact_06bo_source: Path,
        artifact_06bp_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle, output_dir, config, artifact_05t_source, artifact_06bn_source,
            artifact_06bo_source, code_revision=code_revision,
        )
        self.artifact_06bp_source = Path(artifact_06bp_source)
        self.event_scales: Dict[str, np.ndarray] = {}
        self.jump_models: Dict[Tuple[str, int], Any] = {}
        self.selected_jump_arm: Optional[str] = None
        self.support_report: Dict[str, Any] = {}

    @staticmethod
    def _arm_key(spec: Tuple[str, str]) -> str:
        return "|".join(spec)

    def _jump_specs(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (representation, normalization)
            for representation in self.config.event_representations
            for normalization in self.config.event_normalizations
        )

    def _event_rich_windows(self, role: str, count: int, horizon: int) -> List[np.ndarray]:
        allowed = {str(row["trajectory_id"]) for row in self.roles[role]}
        positive: List[np.ndarray] = []
        negative: List[np.ndarray] = []
        for trajectory in sorted(allowed):
            indices = self.store.trajectory_indices[trajectory]
            for start in range(max(0, len(indices) - horizon + 1)):
                candidate = indices[start : start + horizon]
                steps = self.store.metadata["step_index"][candidate]
                if not np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                    continue
                supported = any(has_realized_event(self.store, int(index)) for index in candidate)
                (positive if supported else negative).append(candidate)
        key = lambda row: hashlib.sha256(
            f"{self.config.role_seed}|06bq-window|{role}|{','.join(map(str, row))}".encode()
        ).hexdigest()
        positive.sort(key=key)
        negative.sort(key=key)
        positive_count = count // 2
        negative_count = count - positive_count
        if len(positive) < positive_count or len(negative) < negative_count:
            raise RuntimeError(
                f"06b-q {role} window support insufficient: "
                f"event={len(positive)}, no_event={len(negative)}"
            )
        return positive[:positive_count] + negative[:negative_count]

    def _materialize_event_role(self, role: str, count: int, horizon: int) -> Dict[str, Any]:
        windows = self._event_rich_windows(role, count, horizon)
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
        voltage_t = self.store.read_state(flat, "t", categories=("voltage",)).astype(np.float32).reshape(voltage_shape)
        voltage_t1 = self.store.read_state(flat, "t_plus_1", categories=("voltage",)).astype(np.float32).reshape(voltage_shape)
        drive = atomic.encode_causal_realized_drive(self.store, flat).astype(np.float32)
        drive = drive.reshape(len(windows), horizon, self.layout.segment_count, -1)
        events, event_mask, event_counts = ordered_event_tensor(
            self.store, flat, max_events_per_segment=self.config.event_max_per_segment
        )
        payload = {
            "indices": index,
            "state_t": state_t,
            "state_t1": state_t1,
            "voltage_t": voltage_t,
            "voltage_t1": voltage_t1,
            "drive": drive,
            "held_ions": self._ion_context(index[:, 0]).astype(np.float32),
            "ordered_events": events.reshape(len(windows), horizon, *events.shape[1:]),
            "ordered_event_mask": event_mask.reshape(len(windows), horizon, *event_mask.shape[1:]),
            "ordered_event_counts": event_counts.reshape(len(windows), horizon, -1),
        }
        self.window_data[role] = payload
        return payload

    def _event_flat_tensors(self, role: str, rows: np.ndarray, device: Any) -> Dict[str, Any]:
        values = self.window_data[role]
        horizon = values["voltage_t"].shape[1]
        rows = np.asarray(rows, dtype=np.int64)
        window, step = rows // horizon, rows % horizon
        payload = {
            "state": values["state_t"][window, step],
            "voltage": values["voltage_t"][window, step],
            "target_voltage": values["voltage_t1"][window, step],
            "drive": values["drive"][window, step],
            "held_ions": values["held_ions"][window],
            "ordered_events": values["ordered_events"][window, step],
            "ordered_event_mask": values["ordered_event_mask"][window, step],
        }
        return {
            name: atomic.torch.as_tensor(value, device=device)
            for name, value in payload.items()
        }

    def _new_jump_model(self, seed: int, device: Any) -> Any:
        atomic.torch.manual_seed(seed + 691000)
        return EventConditionedSourceCell(
            base_feature_width=self._feature_width(),
            event_feature_width=len(EVENT_FEATURE_NAMES),
            event_embedding_width=self.config.event_embedding_width,
            region_count=len(self.layout.region_names),
            region_width=self.config.matrix_region_embedding_width,
            hidden_width=self.config.matrix_hidden_width,
            output_limit=self.config.normalized_output_limit,
            passive_gate_initial_bias=self.config.passive_gate_initial_bias,
        ).to(device)

    def _normalized_events(self, batch: Mapping[str, Any], normalization: str) -> Any:
        scale = atomic.torch.as_tensor(
            self.event_scales[normalization],
            dtype=batch["ordered_events"].dtype,
            device=batch["ordered_events"].device,
        )
        return normalize_event_tensor(
            batch["ordered_events"], batch["ordered_event_mask"], scale,
            normalization, clip=self.config.event_normalized_clip,
        )

    def _jump_forward(
        self,
        model: Any,
        spec: Tuple[str, str],
        batch: Mapping[str, Any],
        *,
        gate_mode: str = UNGATED_RESIDUAL,
    ) -> Tuple[Any, Any, Any]:
        base, _, target_source = self._teacher_features(batch)
        region_ids = atomic.torch.as_tensor(
            self.layout.segment_region_ids, dtype=atomic.torch.long,
            device=batch["voltage"].device,
        )
        standardized, _ = model(
            base, self._normalized_events(batch, spec[1]),
            batch["ordered_event_mask"], region_ids,
            representation=spec[0], gate_mode=gate_mode,
        )
        scale_mode = getattr(self, "selected_source_scaling", REGION_P99)
        decoded = standardized * self._scale_tensor(scale_mode, standardized)
        prediction = self._apply_physical_target(
            decoded, batch["voltage"], NET_EFFECTIVE_SOURCE, batch
        )
        return decoded, target_source, prediction

    def _jump_loss(
        self,
        model: Any,
        spec: Tuple[str, str],
        batch: Mapping[str, Any],
        *,
        gate_mode: str = UNGATED_RESIDUAL,
    ) -> Any:
        decoded, target_source, prediction = self._jump_forward(
            model, spec, batch, gate_mode=gate_mode
        )
        scale = self._scale_tensor(self.selected_source_scaling, decoded)
        native = atomic.torch_functional.smooth_l1_loss(decoded / scale, target_source / scale)
        endpoint = atomic.torch.mean(
            self._activity_weight(batch["target_voltage"] - batch["voltage"])
            * atomic.torch_functional.smooth_l1_loss(
                (prediction - batch["target_voltage"]) / self.config.voltage_scale_mv,
                atomic.torch.zeros_like(prediction), reduction="none",
            )
        )
        if self.selected_objective == HYBRID:
            return endpoint + self.config.hybrid_native_weight * native
        return endpoint

    @staticmethod
    def _metric_payload(
        prediction: np.ndarray,
        target: np.ndarray,
        passive: np.ndarray,
        event_mask: np.ndarray,
        current: np.ndarray,
    ) -> Dict[str, Any]:
        error = prediction - target
        baseline = passive - target
        event_rows = event_mask.reshape(len(event_mask), -1).any(axis=1)
        result = {
            "endpoint_rmse_mv": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
            "passive_endpoint_rmse_mv": float(np.sqrt(np.mean(baseline.astype(np.float64) ** 2))),
            "mean_drift_mv": float(np.mean(error)),
            "physical_voltage_violation_count": int(((prediction < -120.0) | (prediction > 80.0)).sum()),
            "event_transition_fraction": float(event_rows.mean()),
        }
        for label, selected in (("event", event_rows), ("no_event", ~event_rows)):
            result[f"{label}_rmse_mv"] = (
                float(np.sqrt(np.mean(error[selected].astype(np.float64) ** 2)))
                if selected.any() else None
            )
        activity = np.abs(target - current)
        for label, selected in (
            ("quiescent_lt_1mV", activity < 1.0),
            ("moderate_1_to_5mV", (activity >= 1.0) & (activity < 5.0)),
            ("active_ge_5mV", activity >= 5.0),
        ):
            result[f"{label}_rmse_mv"] = (
                float(np.sqrt(np.mean(error[selected].astype(np.float64) ** 2)))
                if selected.any() else None
            )
        return result

    def _evaluate_rows(
        self, model: Any, spec: Tuple[str, str], role: str, rows: np.ndarray,
        device: Any, *, gate_mode: str = UNGATED_RESIDUAL,
        control: Optional[str] = None,
    ) -> Dict[str, Any]:
        batch = self._event_flat_tensors(role, rows, device)
        if control == "event_deletion":
            batch["ordered_events"] = atomic.torch.zeros_like(batch["ordered_events"])
            batch["ordered_event_mask"] = atomic.torch.zeros_like(batch["ordered_event_mask"])
        elif control == "timestamp_reversal":
            batch["ordered_events"] = batch["ordered_events"].clone()
            offset = EVENT_FEATURE_NAMES.index("offset_ms")
            batch["ordered_events"][..., offset] = atomic.torch.where(
                batch["ordered_event_mask"],
                1.0 - batch["ordered_events"][..., offset],
                batch["ordered_events"][..., offset],
            )
        elif control == "receptor_permutation":
            batch["ordered_events"] = batch["ordered_events"].clone()
            columns = [EVENT_FEATURE_NAMES.index(name) for name in (
                "ampa_state_increment", "nmda_state_increment",
                "gabaa_state_increment", "gabab_state_increment",
                "has_ampa", "has_nmda", "has_gabaa", "has_gabab",
            )]
            batch["ordered_events"][..., columns] = batch["ordered_events"][..., columns].roll(1, dims=-1)
        model.eval()
        with atomic.torch.no_grad():
            _, _, prediction = self._jump_forward(model, spec, batch, gate_mode=gate_mode)
            passive = self._apply_physical_target(
                atomic.torch.zeros_like(prediction), batch["voltage"],
                NET_EFFECTIVE_SOURCE, batch,
            )
        return self._metric_payload(
            prediction.cpu().numpy(), batch["target_voltage"].cpu().numpy(),
            passive.cpu().numpy(), batch["ordered_event_mask"].cpu().numpy(),
            batch["voltage"].cpu().numpy(),
        )

    @staticmethod
    def _gradient_vector(model: Any) -> Any:
        pieces = [
            parameter.grad.detach().reshape(-1)
            for parameter in model.parameters() if parameter.grad is not None
        ]
        return atomic.torch.cat(pieces) if pieces else atomic.torch.zeros(1)

    def _gradient_contrast(self, model: Any, spec: Tuple[str, str], device: Any) -> Dict[str, float]:
        fit = self.window_data["fit"]
        flat_mask = fit["ordered_event_mask"].reshape(-1, *fit["ordered_event_mask"].shape[2:])
        positive = np.flatnonzero(flat_mask.reshape(len(flat_mask), -1).any(axis=1))[:8]
        negative = np.flatnonzero(~flat_mask.reshape(len(flat_mask), -1).any(axis=1))[:8]
        vectors = []
        norms = []
        for rows in (positive, negative):
            model.zero_grad(set_to_none=True)
            batch = self._event_flat_tensors("fit", rows, device)
            self._jump_loss(model, spec, batch).backward()
            vector = self._gradient_vector(model)
            vectors.append(vector)
            norms.append(float(vector.norm().cpu()))
        cosine = float(atomic.torch_functional.cosine_similarity(vectors[0], vectors[1], dim=0).cpu())
        model.zero_grad(set_to_none=True)
        return {"event_gradient_norm": norms[0], "no_event_gradient_norm": norms[1], "cosine": cosine}

    def prepare_event_supported_jump_playground(self) -> Dict[str, Any]:
        base = self.prepare_atomic_source_learnability()
        _, prior, source = verified_06bp_artifact_root(
            self.artifact_06bp_source,
            self.output_dir.parent / ".06bq_artifact_cache" / "06bp",
        )
        adaptive = prior.get("adaptive_choice", {})
        self.selected_source_scaling = str(adaptive.get("scaling", REGION_P99))
        if self.selected_source_scaling not in (GLOBAL_P99, REGION_P99):
            self.selected_source_scaling = REGION_P99
        self.selected_objective = str(adaptive.get("objective", HYBRID))
        targets = {
            "fit": tuple(self.config.support_fit_components),
            "calibration": tuple(self.config.support_calibration_components),
            "development": tuple(self.config.support_development_components),
        }
        self.roles, self.support_report = build_event_supported_roles(
            self.store, role_seed=self.config.role_seed, component_targets=targets
        )
        counts = {
            "fit": self.config.matrix_fit_window_count,
            "calibration": self.config.matrix_calibration_window_count,
            "development": self.config.matrix_development_window_count,
        }
        self.window_data = {}
        for role, count in counts.items():
            self._materialize_event_role(role, count, self.config.matrix_training_horizon_ms)
        fit = self.window_data["fit"]
        fit_events = fit["ordered_events"].reshape(-1, *fit["ordered_events"].shape[2:])
        fit_mask = fit["ordered_event_mask"].reshape(-1, *fit["ordered_event_mask"].shape[2:])
        self.event_scales = fit_event_normalizers(
            fit_events, fit_mask, quantile=self.config.event_scale_quantile,
            floor=self.config.event_scale_floor,
        )
        parameter_counts = {
            self._arm_key(spec): sum(p.numel() for p in self._new_jump_model(self.config.pilot_seeds[0], atomic.torch.device("cpu")).parameters())
            for spec in self._jump_specs()
        }
        mechanism_audit = self._mechanism_factorization_eligibility_audit()
        rate_audit = self._rate_form_state_sidecar()
        role_support = {}
        for role, values in self.window_data.items():
            mask = values["ordered_event_mask"].reshape(-1, *values["ordered_event_mask"].shape[2:])
            role_support[role] = {
                "transition_count": int(len(mask)),
                "event_transition_count": int(mask.reshape(len(mask), -1).any(axis=1).sum()),
                "event_transition_fraction": float(mask.reshape(len(mask), -1).any(axis=1).mean()),
                "maximum_events_per_segment": int(values["ordered_event_counts"].max()),
            }
        blockers = []
        if len(set(parameter_counts.values())) != 1:
            blockers.append("representation arms are not parameter matched")
        for role, row in role_support.items():
            if row["event_transition_fraction"] < self.config.minimum_event_support_fraction:
                blockers.append(f"{role} lacks realized-event support")
        report = {
            **base,
            "schema_version": "06b-q-contract-v1",
            "experiment": "event_supported_jump_and_mechanism_playground",
            "valid": not blockers,
            "blockers": blockers,
            "source_06bp": source,
            "prior_formal_diagnosis": prior.get("diagnosis"),
            "prior_selected_scaling": self.selected_source_scaling,
            "prior_selected_objective": self.selected_objective,
            "causal_support_roles": self.support_report,
            "role_support": role_support,
            "raw_event_contract": {
                "feature_names": list(EVENT_FEATURE_NAMES),
                "ordered": True,
                "shared_presynaptic_event_preserved": True,
                "receptor_resolved": True,
                "maximum_events_per_segment": self.config.event_max_per_segment,
                "truncation_allowed": False,
            },
            "factorial_axes": {
                "event_representation": list(self.config.event_representations),
                "normalization": list(self.config.event_normalizations),
            },
            "parameter_counts": parameter_counts,
            "event_scales": {
                mode: {
                    "minimum": float(scale.min()), "median": float(np.median(scale)),
                    "maximum": float(scale.max()),
                } for mode, scale in self.event_scales.items()
            },
            "mechanism_factorization_eligibility": mechanism_audit,
            "rate_form_state_sidecar": rate_audit,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "teacher_endpoint_used_as_model_input": False,
            "configuration": asdict(self.config),
        }
        atomic._write_json(self.output_dir / "event_supported_jump_contract.json", report)
        if blockers:
            raise RuntimeError(f"06b-q preflight failed: {blockers}")
        return report

    def _mechanism_factorization_eligibility_audit(self) -> Dict[str, Any]:
        records = list(getattr(self.layout, "core_records", []))
        current_like = [
            row for row in records
            if any(token in str(row.get("variable", row.get("name", ""))).lower() for token in ("current", "_i", "ica", "ina", "ik"))
        ]
        integrated = [
            row for row in current_like
            if any(token in str(row.get("variable", row.get("name", ""))).lower() for token in ("integral", "charge", "time_average"))
        ]
        eligible = bool(integrated)
        return {
            "valid": True,
            "boundary_current_coordinate_count": len(current_like),
            "time_integrated_current_coordinate_count": len(integrated),
            "teacher_grounded_exact_factorization_available": eligible,
            "factorized_model_trained": False,
            "reason": (
                "eligible exact integrated targets detected"
                if eligible else
                "boundary currents cannot be summed as a causal 1 ms integrated mechanism target"
            ),
            "required_followup_if_selected": (
                None if eligible else "teacher logger must store per-mechanism integrated current over each macro-step"
            ),
        }

    def _rate_form_state_sidecar(self) -> Dict[str, Any]:
        indices = np.asarray(self.window_data["fit"]["indices"]).reshape(-1)[:256]
        start = self.store.read_state(indices, "t", categories=("mechanism_states",)).astype(np.float64)
        end = self.store.read_state(indices, "t_plus_1", categories=("mechanism_states",)).astype(np.float64)
        bounded = np.isfinite(start).all(axis=0) & np.isfinite(end).all(axis=0)
        bounded &= (start.min(axis=0) >= -1e-6) & (start.max(axis=0) <= 1.0 + 1e-6)
        bounded &= (end.min(axis=0) >= -1e-6) & (end.max(axis=0) <= 1.0 + 1e-6)
        stable = []
        for coordinate in np.flatnonzero(bounded):
            x, y = start[:, coordinate], end[:, coordinate]
            variance = float(np.var(x))
            if variance <= 1e-12:
                continue
            slope = float(np.cov(x, y, bias=True)[0, 1] / variance)
            stable.append(0.0 <= slope <= 1.0)
        return {
            "valid": True,
            "selection_eligible": False,
            "coordinate_count": int(start.shape[1]),
            "bounded_gate_like_coordinate_count": int(bounded.sum()),
            "identifiable_affine_coordinate_count": len(stable),
            "stable_relaxation_compatible_fraction": float(np.mean(stable)) if stable else 0.0,
            "interpretation": "data-only eligibility probe; no candidate or teacher outcome selected",
        }

    def run_sparse_event_synthetic_preflight(self) -> Dict[str, Any]:
        """Known-answer sparse-event task; validates optimization before biology."""

        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(self.config.role_seed + 692000)
        sample_count, segments, events = 384, 8, min(6, self.config.event_max_per_segment)
        raw = np.zeros((sample_count, segments, events, len(EVENT_FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros((sample_count, segments, events), dtype=bool)
        multiplicity = rng.integers(0, events + 1, size=(sample_count, segments))
        for row in range(sample_count):
            for segment in range(segments):
                count = int(multiplicity[row, segment])
                if not count:
                    continue
                offsets = np.sort(rng.uniform(0.0, 1.0, size=count))
                raw[row, segment, :count, 0] = offsets
                raw[row, segment, :count, 1] = rng.uniform(0.2, 1.0, size=count)
                receptor = rng.integers(0, 4, size=count)
                raw[row, segment, np.arange(count), 4 + receptor] = raw[row, segment, :count, 1]
                raw[row, segment, np.arange(count), 8 + receptor] = 1.0
                raw[row, segment, :count, 15] = np.arange(count) / max(1, count - 1)
                mask[row, segment, :count] = True
        target = (
            1.7 * raw[..., 4].sum(2) + 2.3 * raw[..., 5].sum(2)
            - 1.1 * raw[..., 6].sum(2) - 0.7 * raw[..., 7].sum(2)
            + 0.4 * (raw[..., 0] * raw[..., 5]).sum(2)
        ).astype(np.float32)
        scales = fit_event_normalizers(raw[:256], mask[:256], quantile=self.config.event_scale_quantile, floor=self.config.event_scale_floor)
        specs = self._jump_specs()
        results = {}
        progress = atomic._CompactProgress(
            "06b-q synthetic 3x2", len(specs) * self.config.synthetic_training_steps,
            max(1, self.config.synthetic_training_steps // 4),
        )
        completed = 0
        for spec in specs:
            atomic.torch.manual_seed(self.config.pilot_seeds[0] + 693000)
            encoder = EventEncoderBank(len(EVENT_FEATURE_NAMES), self.config.event_embedding_width).to(device)
            readout = atomic.nn.Linear(self.config.event_embedding_width, 1).to(device)
            optimizer = atomic.torch.optim.AdamW(list(encoder.parameters()) + list(readout.parameters()), lr=1e-3)
            x = atomic.torch.as_tensor(raw, device=device)
            m = atomic.torch.as_tensor(mask, device=device)
            y = atomic.torch.as_tensor(target, device=device)
            scale = atomic.torch.as_tensor(scales[spec[1]], device=device)
            history = []
            gen = np.random.default_rng(self.config.role_seed + 694000)
            for step in range(self.config.synthetic_training_steps + 1):
                if step in self.config.synthetic_checkpoints:
                    encoder.eval(); readout.eval()
                    with atomic.torch.no_grad():
                        normalized = normalize_event_tensor(x[320:], m[320:], scale, spec[1], clip=self.config.event_normalized_clip)
                        prediction = readout(encoder(normalized, m[320:], spec[0])).squeeze(-1)
                        rmse = float(atomic.torch.sqrt(atomic.torch.mean((prediction - y[320:]) ** 2)).cpu())
                    history.append({"step": step, "heldout_rmse": rmse})
                if step == self.config.synthetic_training_steps:
                    break
                rows = atomic.torch.as_tensor(gen.choice(256, 32, replace=False), dtype=atomic.torch.long, device=device)
                normalized = normalize_event_tensor(x[rows], m[rows], scale, spec[1], clip=self.config.event_normalized_clip)
                prediction = readout(encoder(normalized, m[rows], spec[0])).squeeze(-1)
                loss = atomic.torch.mean((prediction - y[rows]) ** 2)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                completed += 1
                progress.update(completed, self._arm_key(spec))
            results[self._arm_key(spec)] = history
        final = {key: rows[-1]["heldout_rmse"] for key, rows in results.items()}
        report = {
            "schema_version": "06b-q-synthetic-preflight-v1",
            "valid": all(np.isfinite(list(final.values()))),
            "known_target": True,
            "sparsity_controlled": True,
            "multiplicity_controlled": True,
            "same_sample_stream": True,
            "same_initialization_within_representation": True,
            "checkpoints": list(self.config.synthetic_checkpoints),
            "histories": results,
            "final_rmse": final,
        }
        atomic._write_json(self.output_dir / "sparse_event_synthetic_preflight.json", report)
        return report

    def train_event_representation_matrix(self) -> Dict[str, Any]:
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        specs = self._jump_specs()
        reports: Dict[str, Any] = {}
        cal_count = self._flat_count("calibration")
        calibration_a = np.arange(cal_count, dtype=np.int64)[::2]
        calibration_b = np.arange(cal_count, dtype=np.int64)[1::2]
        fit_mask = self.window_data["fit"]["ordered_event_mask"].reshape(-1, self.layout.segment_count, self.config.event_max_per_segment)
        positive = np.flatnonzero(fit_mask.reshape(len(fit_mask), -1).any(axis=1))
        negative = np.flatnonzero(~fit_mask.reshape(len(fit_mask), -1).any(axis=1))
        if min(len(positive), len(negative)) < self.config.jump_batch_transition_count // 2:
            raise RuntimeError("06b-q balanced minibatch support is insufficient")
        for seed in self.config.pilot_seeds:
            models = {spec: self._new_jump_model(seed, device) for spec in specs}
            optimizers = {spec: atomic.torch.optim.AdamW(models[spec].parameters(), lr=self.config.matrix_learning_rate, weight_decay=self.config.matrix_weight_decay) for spec in specs}
            best: Dict[Tuple[str, str], Tuple[float, int, Dict[str, Any]]] = {}
            histories = {self._arm_key(spec): [] for spec in specs}
            probes = {self._arm_key(spec): {} for spec in specs}
            rng = np.random.default_rng(seed + 695000)
            digest = hashlib.sha256()
            progress = atomic._CompactProgress(
                f"06b-q 3x2 seed={seed}", self.config.jump_training_steps,
                max(1, self.config.jump_training_steps // 20),
            )
            for step in range(self.config.jump_training_steps + 1):
                if step in self.config.jump_checkpoints:
                    for spec, model in models.items():
                        metrics = self._evaluate_rows(model, spec, "calibration", calibration_a, device)
                        key = self._arm_key(spec)
                        histories[key].append({"step": step, "calibration_half_A": metrics})
                        score = metrics["endpoint_rmse_mv"]
                        if spec not in best or score < best[spec][0]:
                            best[spec] = (score, step, self._copy_state_dict(model))
                if step in self.config.gradient_probe_steps and step > 0:
                    for spec, model in models.items():
                        probes[self._arm_key(spec)][str(step)] = self._gradient_contrast(model, spec, device)
                if step == self.config.jump_training_steps:
                    break
                half = self.config.jump_batch_transition_count // 2
                rows = np.concatenate((rng.choice(positive, half, replace=False), rng.choice(negative, self.config.jump_batch_transition_count - half, replace=False)))
                rng.shuffle(rows); digest.update(rows.astype(np.int64).tobytes())
                batch = self._event_flat_tensors("fit", rows, device)
                losses = []
                for spec, model in models.items():
                    optimizer = optimizers[spec]; optimizer.zero_grad(set_to_none=True)
                    loss = self._jump_loss(model, spec, batch)
                    if not bool(atomic.torch.isfinite(loss)):
                        raise RuntimeError(f"nonfinite 06b-q loss: {self._arm_key(spec)}")
                    loss.backward(); atomic.torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.matrix_gradient_clip_norm); optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(step + 1, f"loss={float(np.median(losses)):.3g}")
            selected = {}
            for spec, model in models.items():
                score, selected_step, state = best[spec]
                model.load_state_dict(state); model.eval()
                key = self._arm_key(spec); self.jump_models[(key, seed)] = model
                path = self.output_dir / f"jump_{key.replace('|','__')}_seed{seed}.pt"
                atomic.torch.save({"spec": spec, "seed": seed, "selected_step": selected_step, "state_dict": state}, path)
                selected[key] = {
                    "selected_step": selected_step,
                    "calibration_half_A_rmse_mv": score,
                    "calibration_half_B": self._evaluate_rows(model, spec, "calibration", calibration_b, device),
                    "checkpoint": path.name,
                    "checkpoint_sha256": atomic._sha256_file(path),
                }
            reports[str(seed)] = {
                "batch_stream_sha256": digest.hexdigest(),
                "histories": histories,
                "postwarmup_gradient_probes": probes,
                "selected": selected,
            }
        scores = {
            self._arm_key(spec): float(np.median([
                report["selected"][self._arm_key(spec)]["calibration_half_B"]["endpoint_rmse_mv"]
                for report in reports.values()
            ])) for spec in specs
        }
        self.selected_jump_arm = min(scores, key=scores.get)
        report = {
            "schema_version": "06b-q-training-v1",
            "valid": True,
            "factorial_arm_count": len(specs),
            "same_numeric_input_tensor": True,
            "same_parameter_count": True,
            "same_minibatch_stream_within_seed": True,
            "balanced_event_no_event_minibatches": True,
            "calibration_half_A_selects_checkpoints": True,
            "calibration_half_B_selects_arm": True,
            "development_used_during_training": False,
            "selected_arm": self.selected_jump_arm,
            "median_calibration_half_B_rmse_mv": scores,
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "event_representation_training.json", report)
        return report

    def _recursive_metrics(self, model: Any, spec: Tuple[str, str], device: Any, *, gate_mode: str) -> Dict[str, Any]:
        values = self.window_data["development"]
        batch_count, horizon = values["voltage_t"].shape[:2]
        current = atomic.torch.as_tensor(values["voltage_t"][:, 0], device=device)
        target = atomic.torch.as_tensor(values["voltage_t1"], device=device)
        predictions = []
        passive_current = current.clone(); passive_predictions = []
        region_ids = atomic.torch.as_tensor(self.layout.segment_region_ids, dtype=atomic.torch.long, device=device)
        center = atomic.torch.as_tensor(self.statistics["state_center"], device=device)
        state_scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=device)
        model.eval()
        with atomic.torch.no_grad():
            for step in range(horizon):
                state = atomic.torch.as_tensor(values["state_t"][:, step], device=device)
                drive = atomic.torch.as_tensor(values["drive"][:, step], device=device)
                ions = atomic.torch.as_tensor(values["held_ions"], device=device)
                context = atomic.torch.cat((drive, ions), dim=-1)
                base = self._features((state - center) / state_scale, current, context, "authentic")
                event_values = atomic.torch.as_tensor(values["ordered_events"][:, step], device=device)
                event_mask = atomic.torch.as_tensor(values["ordered_event_mask"][:, step], device=device)
                scale = atomic.torch.as_tensor(self.event_scales[spec[1]], device=device)
                normalized = normalize_event_tensor(event_values, event_mask, scale, spec[1], clip=self.config.event_normalized_clip)
                standardized, _ = model(base, normalized, event_mask, region_ids, representation=spec[0], gate_mode=gate_mode)
                decoded = standardized * self._scale_tensor(self.selected_source_scaling, standardized)
                current = self._apply_physical_target(decoded, current, NET_EFFECTIVE_SOURCE, {})
                passive_current = self._apply_physical_target(atomic.torch.zeros_like(passive_current), passive_current, NET_EFFECTIVE_SOURCE, {})
                predictions.append(current); passive_predictions.append(passive_current)
        prediction = atomic.torch.stack(predictions, dim=1).cpu().numpy()
        passive = atomic.torch.stack(passive_predictions, dim=1).cpu().numpy()
        target_np = target.cpu().numpy()
        result = {}
        for step in (1, 2, 4, 8):
            if step > horizon:
                continue
            error = prediction[:, step - 1] - target_np[:, step - 1]
            baseline = passive[:, step - 1] - target_np[:, step - 1]
            rmse = float(np.sqrt(np.mean(error.astype(np.float64) ** 2)))
            passive_rmse = float(np.sqrt(np.mean(baseline.astype(np.float64) ** 2)))
            result[str(step)] = {
                "endpoint_rmse_mv": rmse,
                "passive_endpoint_rmse_mv": passive_rmse,
                "gain_over_passive_fraction": 1.0 - rmse / max(passive_rmse, 1e-12),
                "mean_drift_mv": float(np.mean(error)),
                "physical_voltage_violation_count": int(((prediction[:, :step] < -120.0) | (prediction[:, :step] > 80.0)).sum()),
            }
        return result

    def evaluate_event_representation_matrix(self, training: Mapping[str, Any]) -> Dict[str, Any]:
        if self.selected_jump_arm != training.get("selected_arm"):
            raise RuntimeError("06b-q selected arm mismatch")
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        rows = np.arange(self._flat_count("development"), dtype=np.int64)
        reports = {}
        for spec in self._jump_specs():
            key = self._arm_key(spec)
            per_seed = []
            for seed in self.config.pilot_seeds:
                model = self.jump_models[(key, seed)]
                per_seed.append({
                    "one_step": self._evaluate_rows(model, spec, "development", rows, device),
                    "controls": {
                        control: self._evaluate_rows(model, spec, "development", rows, device, control=control)
                        for control in ("event_deletion", "timestamp_reversal", "receptor_permutation")
                    },
                    "recursive": self._recursive_metrics(model, spec, device, gate_mode=UNGATED_RESIDUAL),
                })
            reports[key] = per_seed
        report = {
            "schema_version": "06b-q-evaluation-v1",
            "valid": True,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "causal_controls": ["event_deletion", "timestamp_reversal", "receptor_permutation"],
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "event_representation_evaluation.json", report)
        return report

    def run_passive_default_safety_gate(self, training: Mapping[str, Any]) -> Dict[str, Any]:
        """Adaptive 2-arm safety test using only the already selected 3x2 arm."""

        selected = str(training["selected_arm"])
        spec = tuple(selected.split("|"))
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        fit_mask = self.window_data["fit"]["ordered_event_mask"].reshape(
            -1, self.layout.segment_count, self.config.event_max_per_segment
        )
        positive = np.flatnonzero(fit_mask.reshape(len(fit_mask), -1).any(axis=1))
        negative = np.flatnonzero(~fit_mask.reshape(len(fit_mask), -1).any(axis=1))
        cal_rows = np.arange(self._flat_count("calibration"), dtype=np.int64)[1::2]
        reports = {}
        for seed in self.config.pilot_seeds:
            models = {gate: self._new_jump_model(seed + 11, device) for gate in self.config.safety_gates}
            optimizers = {gate: atomic.torch.optim.AdamW(model.parameters(), lr=self.config.matrix_learning_rate, weight_decay=self.config.matrix_weight_decay) for gate, model in models.items()}
            best = {}; histories = {gate: [] for gate in self.config.safety_gates}
            rng = np.random.default_rng(seed + 696000)
            progress = atomic._CompactProgress(f"06b-q safety seed={seed}", self.config.safety_training_steps, max(1, self.config.safety_training_steps // 10))
            for step in range(self.config.safety_training_steps + 1):
                if step in self.config.safety_checkpoints:
                    for gate, model in models.items():
                        metrics = self._evaluate_rows(model, spec, "calibration", cal_rows, device, gate_mode=gate)
                        histories[gate].append({"step": step, "calibration": metrics})
                        score = metrics["endpoint_rmse_mv"]
                        if gate not in best or score < best[gate][0]:
                            best[gate] = (score, step, self._copy_state_dict(model))
                if step == self.config.safety_training_steps:
                    break
                half = self.config.jump_batch_transition_count // 2
                rows = np.concatenate(
                    (
                        rng.choice(positive, half, replace=False),
                        rng.choice(
                            negative,
                            self.config.jump_batch_transition_count - half,
                            replace=False,
                        ),
                    )
                )
                rng.shuffle(rows)
                batch = self._event_flat_tensors("fit", rows, device)
                losses = []
                for gate, model in models.items():
                    optimizer = optimizers[gate]; optimizer.zero_grad(set_to_none=True)
                    loss = self._jump_loss(model, spec, batch, gate_mode=gate)
                    loss.backward(); atomic.torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.matrix_gradient_clip_norm); optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(step + 1, f"loss={float(np.median(losses)):.3g}")
            seed_report = {}
            for gate, model in models.items():
                score, selected_step, state = best[gate]
                model.load_state_dict(state); model.eval()
                seed_report[gate] = {
                    "selected_step": selected_step,
                    "calibration_rmse_mv": score,
                    "development_one_step": self._evaluate_rows(model, spec, "development", np.arange(self._flat_count("development")), device, gate_mode=gate),
                    "development_recursive": self._recursive_metrics(model, spec, device, gate_mode=gate),
                }
            reports[str(seed)] = seed_report
        report = {
            "schema_version": "06b-q-safety-gate-v1",
            "valid": True,
            "adaptive_dependency": selected,
            "gate_arms": list(self.config.safety_gates),
            "same_architecture_and_parameter_count": True,
            "development_used_for_checkpoint_selection": False,
            "selected_gate": min(
                self.config.safety_gates,
                key=lambda gate: float(
                    np.median(
                        [row[gate]["calibration_rmse_mv"] for row in reports.values()]
                    )
                ),
            ),
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "passive_default_safety_gate.json", report)
        return report

    @staticmethod
    def _median(values: Iterable[float]) -> float:
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        return float(np.median(finite)) if finite else float("nan")

    def finalize_event_supported_jump_playground(
        self,
        contract: Mapping[str, Any],
        synthetic: Mapping[str, Any],
        training: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        safety: Mapping[str, Any],
    ) -> Dict[str, Any]:
        selected = str(training["selected_arm"])
        selected_rows = evaluation["reports"][selected]
        one_step = self._median(row["one_step"]["endpoint_rmse_mv"] for row in selected_rows)
        passive = self._median(row["one_step"]["passive_endpoint_rmse_mv"] for row in selected_rows)
        control_effects = {
            control: self._median(
                row["controls"][control]["endpoint_rmse_mv"] - row["one_step"]["endpoint_rmse_mv"]
                for row in selected_rows
            ) for control in ("event_deletion", "timestamp_reversal", "receptor_permutation")
        }
        gate_summary = {}
        for gate in self.config.safety_gates:
            gate_summary[gate] = {
                "median_recursive_8ms_rmse_mv": self._median(
                    row[gate]["development_recursive"]["8"]["endpoint_rmse_mv"]
                    for row in safety["reports"].values()
                ),
                "median_no_event_one_step_rmse_mv": self._median(
                    row[gate]["development_one_step"]["no_event_rmse_mv"]
                    for row in safety["reports"].values()
                ),
                "median_quiescent_one_step_rmse_mv": self._median(
                    row[gate]["development_one_step"]["quiescent_lt_1mV_rmse_mv"]
                    for row in safety["reports"].values()
                ),
                "median_moderate_one_step_rmse_mv": self._median(
                    row[gate]["development_one_step"]["moderate_1_to_5mV_rmse_mv"]
                    for row in safety["reports"].values()
                ),
            }
        selected_gate = str(safety["selected_gate"])
        recursive = self._median(
            row[selected_gate]["development_recursive"]["8"]["endpoint_rmse_mv"]
            for row in safety["reports"].values()
        )
        recursive_passive = self._median(
            row[selected_gate]["development_recursive"]["8"]["passive_endpoint_rmse_mv"]
            for row in safety["reports"].values()
        )
        physical_violations = int(
            sum(
                row[selected_gate]["development_recursive"]["8"][
                    "physical_voltage_violation_count"
                ]
                for row in safety["reports"].values()
            )
        )
        safety_preserves_quiet_moderate = bool(
            gate_summary[selected_gate]["median_quiescent_one_step_rmse_mv"]
            <= gate_summary[UNGATED_RESIDUAL]["median_quiescent_one_step_rmse_mv"] + 1e-9
            and gate_summary[selected_gate]["median_moderate_one_step_rmse_mv"]
            <= gate_summary[UNGATED_RESIDUAL]["median_moderate_one_step_rmse_mv"] + 1e-9
        )
        atomic_gain = 1.0 - one_step / max(passive, 1e-12)
        recursive_gain = 1.0 - recursive / max(recursive_passive, 1e-12)
        causal = control_effects["event_deletion"] > 0.0 and control_effects["receptor_permutation"] > 0.0
        candidate = (
            f"{selected}|{selected_gate}" if atomic_gain >= self.config.minimum_atomic_gain_over_passive_fraction
            and recursive_gain >= self.config.minimum_recursive_gain_fraction
            and causal and safety_preserves_quiet_moderate
            and physical_violations == 0 else None
        )
        if candidate:
            diagnosis = "ORDERED_EVENT_OPERATOR_SURVIVES_CAUSAL_AND_RECURSIVE_GATES"
            next_step = "independent_train_support_confirmation_before_any_test"
        elif atomic_gain > 0.0 and recursive_gain <= 0.0:
            diagnosis = "EVENT_OPERATOR_LEARNS_TEACHER_BOUNDARY_BUT_FAILS_RECURSIVE_BOUNDARY"
            next_step = "collect_intermediate_state_or_train_recursive_boundary_exposure"
        elif not causal:
            diagnosis = "EVENT_REPRESENTATION_GAIN_IS_NOT_CAUSALLY_ATTRIBUTABLE_TO_EVENT_CONTENT"
            next_step = "mechanism_integral_logging_contract_before_architecture_expansion"
        else:
            diagnosis = "EVENT_SUPPORT_AND_NORMALIZATION_DO_NOT_RESCUE_EFFECTIVE_SOURCE"
            next_step = "teacher_grounded_mechanism_integral_dataset_extension"
        report = {
            "schema_version": "06b-q-final-report-v1",
            "valid": bool(contract.get("valid") and synthetic.get("valid") and training.get("valid") and evaluation.get("valid") and safety.get("valid")),
            "component_playground_grade": True,
            "diagnosis": diagnosis,
            "selected_calibration_arm": selected,
            "selected_safety_gate": selected_gate,
            "selected_candidate": candidate,
            "median_one_step_rmse_mv": one_step,
            "median_one_step_gain_over_passive_fraction": atomic_gain,
            "median_recursive_8ms_rmse_mv": recursive,
            "median_recursive_8ms_gain_over_passive_fraction": recursive_gain,
            "causal_control_rmse_increases_mv": control_effects,
            "causal_event_content_material": causal,
            "safety_gate_summary": gate_summary,
            "safety_preserves_quiet_and_moderate": safety_preserves_quiet_moderate,
            "physical_voltage_violation_count": physical_violations,
            "mechanism_factorization_eligibility": contract["mechanism_factorization_eligibility"],
            "rate_form_state_sidecar": contract["rate_form_state_sidecar"],
            "development_used_for_checkpoint_or_arm_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "fresh_test_accessed": False,
            "teacher_endpoint_used_as_model_input": False,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BP_ARCHIVE_SHA256", "EXPECTED_06BP_INDEX_SHA256",
    "EXPECTED_06BP_FINAL_SHA256", "MOMENT_POOL", "DEEPSET_EVENTS",
    "CHRONOLOGICAL_JUMP", "EVENT_REPRESENTATIONS", "LEGACY_ALL_ENTRY_P99",
    "NONZERO_ROBUST_LOG", "EVENT_NORMALIZATIONS", "UNGATED_RESIDUAL",
    "PASSIVE_DEFAULT_GATE", "SAFETY_GATES", "EVENT_FEATURE_NAMES",
    "verified_06bp_artifact_root", "has_realized_event",
    "build_event_supported_roles", "ordered_event_tensor",
    "fit_event_normalizers", "normalize_event_tensor",
    "EventSupportedJumpConfig", "EventEncoderBank",
    "EventConditionedSourceCell", "EventSupportedJumpPlayground",
]
