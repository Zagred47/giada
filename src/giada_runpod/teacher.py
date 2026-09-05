"""Lean, causal NEURON generation used only by the RunPod scaling track."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from src.hayflow_data import BurnInCriteria, build_input_views
from src.hayflow_model.rollout_aware_architecture_canary import CAUSAL_DRIVE_FEATURES
from src.hayflow_teacher import (
    DiagnosticDatasetSession,
    TargetedDiagnosticDatasetSession,
    expected_audit_hashes,
)
from src.hayflow_teacher.audit import git_commit

from .config import ScaleConfig
from .neuronio_inputs import (
    NeuronIOInputConfig,
    build_dendritic_synapse_map,
    sample_neuronio_actions,
)
from .planning import ShardPlan
from .store import LeanShardWriter, validate_lean_shard


CANONICAL_MECHANISM_GROUP_COUNT = 18
CANONICAL_ION_COUNT = 6
CANONICAL_MATCHED_REGION_COUNT = 11


class RunPodCausalTeacherSession(DiagnosticDatasetSession):
    """Minimal teacher session with the validated causal release driver.

    The paper-scale NeuronIO generator does not consume the calibrated
    dendritic protocol catalog.  Inheriting the v1/v1.1 dataset session would
    therefore introduce two irrelevant artifact dependencies.  This session
    retains the audited base teacher and delegates only the already-validated
    causal one-millisecond driver from v1.1.
    """

    def __init__(
        self,
        elm_repo: Path,
        teacher_repo: Path,
        *,
        output_dir: Path,
        seed: int,
    ) -> None:
        super().__init__(
            elm_repo,
            teacher_repo,
            output_dir=output_dir,
            seed=seed,
            expected_teacher_hashes=expected_audit_hashes(),
        )
        self.active_random123_seed = int(seed)
        self._active_transition_id = -1
        self._last_release_outcomes: List[Any] = []
        self._last_release_verification: Dict[str, Any] = {}

    def _configure_rngs(self, seed: int, sequences: Sequence[float]) -> None:
        super()._configure_rngs(seed, sequences)
        self.active_random123_seed = int(seed)

    def _drive_one_ms(self, *args: Any, **kwargs: Any) -> Any:
        return TargetedDiagnosticDatasetSession._drive_one_ms(self, *args, **kwargs)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _release_seed(seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"giada-release|{seed}".encode()).digest()[:4], "big")


class BoundaryProjector:
    """Read only the segment-local fields required by the matched models."""

    def __init__(self, session: RunPodCausalTeacherSession, segment_ids: Sequence[int]) -> None:
        self.session = session
        self.segment_ids = np.asarray(segment_ids, dtype=np.int64)
        self.segment_count = len(session.audit.live_segments)
        if len(self.segment_ids) == 0 or np.any(self.segment_ids < 0) or np.any(self.segment_ids >= self.segment_count):
            raise ValueError("projected segment ids are invalid")
        variables = session.state_schema["variables"]
        mechanism_rows = [row for row in variables if row["category"] == "mechanism_states" and row["scope"] == "segment"]
        ion_rows = [row for row in variables if row["category"] == "calcium_ions" and row["scope"] == "segment"]
        self.mechanism_group_names = sorted(
            {f"{row['mechanism']}|{row['variable']}|{row['kind']}" for row in mechanism_rows}
        )
        self.ion_names = sorted({str(row["variable"]) for row in ion_rows})
        if len(self.mechanism_group_names) != CANONICAL_MECHANISM_GROUP_COUNT:
            raise RuntimeError(
                f"canonical matched contract requires {CANONICAL_MECHANISM_GROUP_COUNT} "
                f"mechanism groups, got {len(self.mechanism_group_names)}"
            )
        if len(self.ion_names) != CANONICAL_ION_COUNT:
            raise RuntimeError(
                f"canonical matched contract requires {CANONICAL_ION_COUNT} ions, "
                f"got {len(self.ion_names)}"
            )
        group_index = {name: index for index, name in enumerate(self.mechanism_group_names)}
        ion_index = {name: index for index, name in enumerate(self.ion_names)}
        self.mechanism_lookup: Dict[tuple[int, int], Any] = {}
        self.ion_lookup: Dict[tuple[int, int], Any] = {}
        mechanism_variables = session.state_variables["mechanism_states"]
        ion_variables = session.state_variables["calcium_ions"]
        for variable in mechanism_variables:
            if variable.scope.value != "segment":
                continue
            group = group_index[f"{variable.mechanism}|{variable.name}|{variable.kind.value}"]
            key = (int(variable.owner_id), int(group))
            if key in self.mechanism_lookup:
                raise RuntimeError(f"duplicate semantic mechanism state {key}")
            self.mechanism_lookup[key] = variable
        for variable in ion_variables:
            if variable.scope.value != "segment":
                continue
            key = (int(variable.owner_id), int(ion_index[variable.name]))
            if key in self.ion_lookup:
                raise RuntimeError(f"duplicate ion state {key}")
            self.ion_lookup[key] = variable
        segments = session.audit.segment_df.set_index("segment_id")
        self.parent_ids = np.arange(self.segment_count, dtype=np.int64)
        children: List[List[int]] = [[] for _ in range(self.segment_count)]
        for segment in range(self.segment_count):
            parent = segments.loc[segment, "parent_segment_id"]
            if parent == parent:  # NaN-safe
                self.parent_ids[segment] = int(parent)
                children[int(parent)].append(segment)
        self.children = children
        self.presence = np.zeros((self.segment_count, len(self.mechanism_group_names)), dtype=np.uint8)
        for segment, group in self.mechanism_lookup:
            self.presence[segment, group] = 1
        manifest_segments = sorted(session.audit.manifest.segments, key=lambda row: int(row.id))
        raw_static = np.asarray(
            [
                [
                    math.log1p(float(row.area_um2)),
                    math.log1p(float(row.length_um)),
                    math.log1p(float(row.diameter_um)),
                    math.log1p(max(0.0, float(row.axial_conductance_to_parent_us))),
                    math.log1p(max(0.0, float(row.membrane_capacitance_uf))),
                    math.log1p(max(0.0, float(row.passive_leak_conductance_us))),
                    float(row.passive_reversal_mv) / 100.0,
                ]
                for row in manifest_segments
            ],
            dtype=np.float32,
        )
        self.segment_static = (raw_static - raw_static.mean(axis=0, keepdims=True)) / np.maximum(
            raw_static.std(axis=0, keepdims=True), 1e-6
        )
        definition_regions = {str(row.region) for row in session.event_definitions}
        segment_regions = [str(row.region.value) for row in manifest_segments]
        self.region_names = sorted(set(segment_regions) | definition_regions)
        if len(self.region_names) > CANONICAL_MATCHED_REGION_COUNT:
            raise RuntimeError("teacher region vocabulary exceeds the matched contract")
        # The registered 06b-c tensor has 11 region coordinates. Only the soma
        # coordinate is active in this paper-scale profile; reserved zero-use
        # coordinates retain the exact tensor width and parameter comparison.
        while len(self.region_names) < CANONICAL_MATCHED_REGION_COUNT:
            self.region_names.append(f"reserved_matched_region_{len(self.region_names):02d}")
        region_index = {name: index for index, name in enumerate(self.region_names)}
        self.segment_region_ids = np.asarray(
            [region_index[name] for name in segment_regions], dtype=np.int64
        )

    def _value(self, variable: Any) -> float:
        return float(
            self.session.audit._read_variable(
                variable, self.session._owner_for(variable)
            )
        )

    def capture(self) -> Dict[str, np.ndarray]:
        voltage_all = np.asarray(
            [float(segment.v) for segment in self.session.audit.live_segments],
            dtype=np.float32,
        )
        selected = self.segment_ids
        parent = self.parent_ids[selected]
        parent_delta = voltage_all[parent] - voltage_all[selected]
        child_delta = np.zeros(len(selected), dtype=np.float32)
        for row, segment in enumerate(selected):
            children = self.children[int(segment)]
            if children:
                child_delta[row] = float(np.mean(voltage_all[children]) - voltage_all[segment])
        mechanism = np.zeros((len(selected), len(self.mechanism_group_names)), dtype=np.float32)
        ions = np.zeros((len(selected), len(self.ion_names)), dtype=np.float32)
        for row, segment in enumerate(selected):
            for group in np.flatnonzero(self.presence[int(segment)]):
                mechanism[row, group] = self._value(self.mechanism_lookup[(int(segment), int(group))])
            for ion in range(len(self.ion_names)):
                variable = self.ion_lookup.get((int(segment), ion))
                if variable is not None:
                    ions[row, ion] = self._value(variable)
        result = {
            "segment_id": selected.astype(np.int32),
            "voltage_t_mv": voltage_all[selected],
            "parent_delta_t_mv": parent_delta.astype(np.float32),
            "mean_child_delta_t_mv": child_delta,
            "mechanism_state_t": mechanism,
            "ion_state_t": ions,
        }
        if not all(np.isfinite(value).all() for value in result.values()):
            raise RuntimeError("non-finite compact boundary state")
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            "mechanism_group_names": self.mechanism_group_names,
            "ion_names": self.ion_names,
            "causal_drive_features": list(CAUSAL_DRIVE_FEATURES),
            "segment_ids": self.segment_ids.tolist(),
            "mechanism_presence": self.presence[self.segment_ids].tolist(),
            "segment_static": self.segment_static[self.segment_ids].tolist(),
            "region_names": self.region_names,
            "segment_region_ids": self.segment_region_ids[self.segment_ids].tolist(),
            "boundary_interval_ms": 1.0,
            "target": "raw_authentic_NEURON_V_t_plus_1_minus_V_t",
            "future_state_used_as_input": False,
        }


def encode_realized_drive(
    actions: Sequence[Mapping[str, Any]],
    *,
    selected_segments: Sequence[int],
) -> np.ndarray:
    """Standalone form of the registered 12-feature causal drive encoder."""

    selected = {int(segment): row for row, segment in enumerate(selected_segments)}
    output = np.zeros((len(selected), len(CAUSAL_DRIVE_FEATURES)), dtype=np.float32)
    offset_sum = np.zeros(len(selected), dtype=np.float64)
    offset_square = np.zeros(len(selected), dtype=np.float64)
    offset_weight = np.zeros(len(selected), dtype=np.float64)
    for action in actions:
        if action.get("kind") == "somatic_current":
            row = selected.get(0)
            if row is not None:
                offset = float(action.get("offset_ms", 0.0) or 0.0)
                duration = max(0.0, min(1.0 - offset, float(action.get("duration_ms", 0.0) or 0.0)))
                amplitude = float(action.get("amplitude_na", 0.0) or 0.0)
                output[row, 9] += amplitude
                output[row, 10] += amplitude * duration
            continue
        row = selected.get(int(action["segment_id"]))
        if row is None:
            continue
        ampa = float(action.get("ampa_state_increment", 0.0) or 0.0)
        nmda = float(action.get("nmda_state_increment", 0.0) or 0.0)
        inhibitory = float(action.get("inhibitory_state_increment", 0.0) or 0.0)
        released = float(action.get("released_quantity", 0.0) or 0.0)
        excitatory = bool(ampa or nmda or action.get("synapse_type") == "ProbAMPANMDA2")
        output[row, 0] += ampa
        output[row, 1] += nmda
        output[row, 2] += inhibitory
        output[row, 3] += released
        output[row, 4 if excitatory else 5] += 0.05
        output[row, 6] += 0.05 * float(bool(action.get("release_success", False)))
        output[row, 11] += 0.05 * float(action.get("weight_multiplier", 1.0) or 1.0)
        event_offset = float(action.get("offset_ms", 0.0) or 0.0)
        weight = max(abs(ampa) + abs(nmda) + abs(inhibitory), released, 1e-6)
        offset_sum[row] += weight * event_offset
        offset_square[row] += weight * event_offset * event_offset
        offset_weight[row] += weight
    active = offset_weight > 0
    output[active, 7] = (offset_sum[active] / offset_weight[active]).astype(np.float32)
    output[active, 8] = (offset_square[active] / offset_weight[active]).astype(np.float32)
    return output


class ScaleTeacherGenerator:
    """One long-lived NEURON process that writes one or more independent shards."""

    def __init__(self, elm_repo: Path, teacher_repo: Path, work_dir: Path, *, seed: int) -> None:
        self.elm_repo = Path(elm_repo).resolve()
        self.teacher_repo = Path(teacher_repo).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.session = RunPodCausalTeacherSession(
            self.elm_repo,
            self.teacher_repo,
            output_dir=self.work_dir / "teacher_runtime",
            seed=int(seed),
        )
        self.prepared = False

    def prepare(self, burnin: BurnInCriteria | None = None) -> Dict[str, Any]:
        teacher = self.session.prepare_teacher()
        burnin_report = self.session.run_burn_in(burnin or BurnInCriteria())
        self.mapping = build_dendritic_synapse_map(self.session)
        self.equilibrium_rng = json.loads(
            self.session.equilibrium_rng_path.read_text(encoding="utf-8")
        )
        self.prepared = True
        report = {
            "teacher": teacher,
            "burnin_duration_ms": burnin_report["burnin_duration_ms"],
            "dendritic_synapse_pairs": len(self.mapping.segment_ids),
            "worker_pid": os.getpid(),
        }
        _atomic_json(self.work_dir / "worker_prepare_report.json", report)
        return report

    def _segments_for(self, config: ScaleConfig, shard: ShardPlan) -> np.ndarray:
        soma = int(self.session.audit.representatives["soma"])
        if config.storage_profile == "soma_paper":
            return np.asarray([soma], dtype=np.int64)
        required = list(dict.fromkeys(int(value) for value in self.session.audit.representatives.values()))
        rng = np.random.default_rng(config.root_seed + shard.shard_index)
        available = np.asarray([value for value in range(642) if value not in required], dtype=np.int64)
        extra_count = max(0, config.sampled_segments_per_transition - len(required))
        extra = rng.choice(available, size=extra_count, replace=False).tolist()
        return np.asarray((required + extra)[: config.sampled_segments_per_transition], dtype=np.int64)

    def generate_shard(self, config: ScaleConfig, shard: ShardPlan, output_root: Path) -> Dict[str, Any]:
        if not self.prepared:
            raise RuntimeError("prepare() must run before generation")
        output_root = Path(output_root).resolve()
        shard_path = output_root / "shards" / f"{shard.shard_id}.h5"
        done_path = output_root / "status" / f"{shard.shard_id}.done.json"
        if done_path.is_file() and shard_path.is_file():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            validation = validate_lean_shard(
                shard_path, expected_transition_count=shard.expected_transition_count
            )
            if done.get("plan_sha256") == shard.plan_sha256 and validation["valid"] and done.get("sha256") == validation["sha256"]:
                return {**done, "resumed": True}
            raise RuntimeError(f"existing {shard.shard_id} does not match its immutable plan")
        partial = shard_path.with_suffix(shard_path.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        segments = self._segments_for(config, shard)
        projector = BoundaryProjector(self.session, segments)
        metadata = {
            **projector.metadata(),
            "stage": config.stage,
            "storage_profile": config.storage_profile,
            "plan_sha256": shard.plan_sha256,
            "teacher_commit": git_commit(self.teacher_repo),
            "input_methodology": "NeuronIO_NMDA_ranges_temporal_smoothing_spatial_length_weighting",
        }
        started = time.perf_counter()
        writer = LeanShardWriter(
            shard_path,
            segment_count_per_transition=len(segments),
            mechanism_group_count=len(projector.mechanism_group_names),
            ion_count=len(projector.ion_names),
            schema_metadata=metadata,
            compression=config.compression,
            chunk_transitions=config.chunk_transitions,
        )
        try:
            local_row = 0
            last_progress = started
            for trajectory in shard.trajectories:
                actions_by_step, input_metadata = sample_neuronio_actions(
                    trajectory.duration_ms,
                    self.mapping,
                    seed=trajectory.seed,
                    config=NeuronIOInputConfig(),
                )
                self.session._restore_native_snapshot(
                    self.session.equilibrium_snapshot_path,
                    self.equilibrium_rng["sequences"],
                    _release_seed(trajectory.seed),
                )
                for step in range(trajectory.duration_ms):
                    state_t = projector.capture()
                    actions = actions_by_step.get(step, ())
                    self.session._active_transition_id = local_row
                    self.session._last_release_outcomes = []
                    self.session._last_release_verification = {}
                    _, scheduled, _ = self.session._drive_one_ms(
                        float(self.session.h.t), actions, lambda: 0.0, sample_interval_ms=1.0
                    )
                    state_t1 = projector.capture()
                    views = build_input_views(scheduled, self.session._last_release_outcomes)
                    realized = []
                    for action in views["U_realized"]:
                        item = dict(action)
                        if item.get("kind") == "synaptic_event":
                            record = self.session.audit.synapse_records[int(item["synapse_id"])]
                            item["segment_id"] = int(record["segment_id"])
                            item["synapse_type"] = str(record["class_name"])
                        realized.append(item)
                    scheduled_synaptic = [
                        item for item in scheduled if item.get("kind") == "synaptic_event"
                    ]
                    outcome_events = []
                    for outcome in self.session._last_release_outcomes:
                        item = outcome.to_dict()
                        scheduled_item = scheduled_synaptic[int(item["event_index"])]
                        record = self.session.audit.synapse_records[int(item["synapse_id"])]
                        item.update(
                            {
                                "kind": "synaptic_event",
                                "segment_id": int(record["segment_id"]),
                                "offset_ms": float(scheduled_item.get("offset_ms", 0.0)),
                            }
                        )
                        outcome_events.append(item)
                    row = {
                        **state_t,
                        "voltage_t_plus_1_mv": state_t1["voltage_t_mv"],
                        "causal_drive": encode_realized_drive(realized, selected_segments=segments),
                        "trajectory_index": trajectory.trajectory_index,
                        "step_index": step,
                        "seed": trajectory.seed,
                        "split_code": 1 if trajectory.split == "validation" else 0,
                        "scheduled_event_count": len(scheduled_synaptic),
                        "realized_event_count": len([a for a in realized if a.get("kind") == "synaptic_event"]),
                    }
                    writer.append(row, outcome_events)
                    local_row += 1
                    now = time.perf_counter()
                    if (
                        now - last_progress >= config.progress_interval_s
                        or local_row == shard.expected_transition_count
                    ):
                        elapsed = max(now - started, 1e-9)
                        rate = local_row / elapsed
                        remaining = shard.expected_transition_count - local_row
                        print(
                            f"[GIADA RunPod][{shard.shard_id}] "
                            f"{local_row:,}/{shard.expected_transition_count:,} "
                            f"({100.0 * local_row / shard.expected_transition_count:.1f}%) "
                            f"{rate:.2f} transition/s; ETA {remaining / max(rate, 1e-9) / 60.0:.1f} min",
                            flush=True,
                        )
                        last_progress = now
            storage = writer.close(expected_transition_count=shard.expected_transition_count)
        except Exception:
            writer.abort()
            raise
        report = {
            **storage,
            "schema_version": "giada-runpod-shard-completion-v1",
            "shard_id": shard.shard_id,
            "plan_sha256": shard.plan_sha256,
            "trajectory_count": len(shard.trajectories),
            "elapsed_seconds": time.perf_counter() - started,
            "transitions_per_second": shard.expected_transition_count / max(time.perf_counter() - started, 1e-9),
            "resumed": False,
        }
        _atomic_json(done_path, report)
        return report
