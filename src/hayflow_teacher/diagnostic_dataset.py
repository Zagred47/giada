"""Burn-in, snapshot, and 1 ms transition generation for the Hay teacher.

This module deliberately builds on :class:`TeacherAuditSession`: the audit and
the dataset generator therefore instantiate the same morphology, mechanisms,
synapses, Random123 bindings, and manifest.  NEURON and storage dependencies
remain lazy so training-only imports stay lightweight.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..hayflow_data import (
    DATASET_SCHEMA_VERSION,
    BurnInCriteria,
    InputAction,
    ProtocolTrajectory,
    TransitionH5Writer,
    estimate_dataset_size_bytes,
    schema_record,
    validate_hdf5_store,
    validate_input_actions,
    validate_split_isolation,
    write_json,
)
from ..hayflow_schema import VariableKind
from .audit import git_commit, sha256_file
from .audit_runtime import PINNED_TEACHER_COMMIT, TeacherAuditSession
from .event_extractor import (
    EventDefinition,
    default_event_definitions,
    event_ids_by_transition,
    extract_events,
)


DEFAULT_MICROTRACE_STEP_MS = 0.025


class DiagnosticDatasetSession:
    """Stateful runtime used by notebook 01, with explicit phase boundaries."""

    def __init__(
        self,
        elm_repo: Path,
        teacher_repo: Path,
        output_dir: Optional[Path] = None,
        seed: int = 271828,
        expected_teacher_hashes: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.elm_repo = Path(elm_repo).resolve()
        self.teacher_repo = Path(teacher_repo).resolve()
        self.output_dir = Path(
            output_dir
            or self.elm_repo / "artifacts" / "transition_dataset_diagnostic"
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.branching_dir = self.output_dir / "branching"
        self.figures_dir = self.output_dir / "figures"
        for directory in (
            self.snapshots_dir,
            self.branching_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.seed = int(seed)
        self.expected_teacher_hashes = dict(expected_teacher_hashes or {})
        self.audit = TeacherAuditSession(
            self.elm_repo,
            self.teacher_repo,
            artifact_dir=self.output_dir / "_teacher_contract",
            seed=self.seed,
        )
        self.np = None
        self.pd = None
        self.h = None
        self.cvode = None
        self.somatic_clamp = None
        self.callback_refs: List[Any] = []
        self.state_variables: Dict[str, List[Any]] = {}
        self.state_schema: Dict[str, Any] = {}
        self.micro_variables: List[Any] = []
        self.micro_variable_ids: List[str] = []
        self.event_definitions: List[EventDefinition] = []
        self.equilibrium_snapshot_path = (
            self.snapshots_dir / "equilibrium_snapshot.neuron.bin"
        )
        self.equilibrium_rng_path = (
            self.snapshots_dir / "equilibrium_snapshot.rng.json"
        )
        self.burnin_report: Dict[str, Any] = {}
        self.dataset_manifest: Dict[str, Any] = {}
        self.transition_path = self.output_dir / "transition_dataset.h5"

    def prepare_teacher(self) -> Dict[str, Any]:
        """Instantiate the exact audited teacher and write stable tables."""

        environment = self.audit.audit_environment()
        teacher = self.audit.load_canonical_teacher()
        _, morphology = self.audit.audit_morphology()
        self.audit.audit_mechanisms_and_synapses()
        self.np = self.audit.np
        self.pd = self.audit.pd
        self.h = self.audit.h
        self.cvode = self.audit.cvode

        if morphology["segment_count"] != 642:
            raise RuntimeError(
                f"expected 642 canonical segments, got {morphology['segment_count']}"
            )
        if git_commit(self.teacher_repo) != PINNED_TEACHER_COMMIT:
            raise RuntimeError("teacher commit differs from the audited commit")
        observed_hashes = {
            str(record["path"]): str(record["sha256"])
            for record in environment["source_files"]
        }
        mismatches = {
            path: {"expected": expected, "observed": observed_hashes.get(path)}
            for path, expected in self.expected_teacher_hashes.items()
            if observed_hashes.get(path) != expected
        }
        if mismatches:
            raise RuntimeError(f"teacher source hashes differ from audit: {mismatches}")

        # The clamp is instrumentation.  It is created once before every
        # SaveState and stays present with zero amplitude unless a protocol
        # explicitly schedules a current input.
        soma_segment = self.audit.live_segments[
            self.audit.representatives["soma"]
        ]
        self.somatic_clamp = self.h.IClamp(soma_segment)
        self.somatic_clamp.delay = 0.0
        self.somatic_clamp.dur = 1e9
        self.somatic_clamp.amp = 0.0

        self._build_state_schema()
        self.event_definitions = default_event_definitions(
            self.audit.representatives
        )
        write_json(
            self.output_dir / "event_definition_config.json",
            {
                "status": "diagnostic_not_final",
                "visual_review_required": True,
                "interpretation_note": (
                    "The initial calcium and NMDA candidate definitions use "
                    "regional voltage proxies. They are hypotheses for visual "
                    "review, not accepted biological ground truth."
                ),
                "definitions": [item.to_dict() for item in self.event_definitions],
            },
        )
        self.audit.manifest.write_json(self.output_dir / "manifest.json")
        self.audit.segment_df.to_parquet(
            self.output_dir / "segments.parquet", index=False
        )
        self.audit.synapse_df.to_parquet(
            self.output_dir / "synapses.parquet", index=False
        )
        return {
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "teacher_hashes_match_audit": not mismatches,
            "segment_count": morphology["segment_count"],
            "synapse_count": len(self.audit.synapse_records),
            "state_widths": {
                key: len(value) for key, value in self.state_variables.items()
            },
            "microtrace_variable_count": len(self.micro_variables),
            "representatives": dict(self.audit.representatives),
            "teacher_variant": teacher["variant"],
        }

    def _build_state_schema(self) -> None:
        accessible = self.audit.global_accessible_state_ids
        categories: Dict[str, List[Any]] = defaultdict(list)
        for variable in self.audit.manifest.variables:
            if variable.id not in accessible:
                continue
            if (
                variable.scope.value == "segment"
                and variable.mechanism == "neuron"
                and variable.name == "v"
            ):
                category = "voltage"
            elif variable.scope.value == "synapse":
                category = "synapse_states"
            elif variable.kind == VariableKind.CONCENTRATION:
                category = "calcium_ions"
            else:
                category = "mechanism_states"
            categories[category].append(variable)

        observable_kinds = {
            VariableKind.ION_CURRENT,
            VariableKind.AXIAL_CURRENT,
            VariableKind.SYNAPTIC_CONDUCTANCE,
        }
        for variable in self.audit.manifest.variables:
            if variable.id in accessible or variable.kind not in observable_kinds:
                continue
            owner = self._owner_for(variable)
            if self.audit._variable_is_accessible(variable, owner):
                categories["currents_conductances"].append(variable)

        required = {
            "voltage",
            "mechanism_states",
            "calcium_ions",
            "synapse_states",
            "currents_conductances",
        }
        missing = required - set(categories)
        if missing:
            raise RuntimeError(f"empty required state categories: {sorted(missing)}")
        self.state_variables = {
            category: list(values) for category, values in categories.items()
        }

        rows = []
        for category, variables in self.state_variables.items():
            for index, variable in enumerate(variables):
                rows.append(
                    schema_record(
                        variable_id=variable.id,
                        category=category,
                        index=index,
                        scope=variable.scope.value,
                        owner_id=variable.owner_id,
                        mechanism=variable.mechanism,
                        variable=variable.name,
                        kind=variable.kind.value,
                        unit=variable.unit,
                    )
                )

        representative_ids = set(self.audit.representatives.values())
        representative_synapse_ids = {
            int(record["synapse_id"])
            for record in self.audit.synapse_records
            if int(record["segment_id"]) in representative_ids
        }
        self.micro_variables = [
            variable
            for variables in self.state_variables.values()
            for variable in variables
            if (
                variable.scope.value == "segment"
                and int(variable.owner_id) in representative_ids
            )
            or (
                variable.scope.value == "synapse"
                and int(variable.owner_id) in representative_synapse_ids
            )
        ]
        self.micro_variable_ids = [item.id for item in self.micro_variables]
        self.state_schema = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "boundary_interval_ms": 1.0,
            "index_contract": "(scope, owner_id, mechanism, variable)",
            "categories": {
                category: {
                    "width": len(variables),
                    "variable_ids": [item.id for item in variables],
                }
                for category, variables in self.state_variables.items()
            },
            "rng_state": {
                "width": len(self.audit.synapse_rngs),
                "generator": "Random123",
                "index": "synapse_id",
            },
            "variables": rows,
            "microtrace_variable_ids": self.micro_variable_ids,
            "probe_order": list(self.audit.representatives),
            "all_segment_voltage_order": list(range(len(self.audit.live_segments))),
        }
        write_json(self.output_dir / "state_schema.json", self.state_schema)

    def _owner_for(self, variable: Any) -> Any:
        if variable.scope.value == "segment":
            return self.audit.live_segments[int(variable.owner_id)]
        if variable.scope.value == "synapse":
            record = self.audit.synapse_records[int(variable.owner_id)]
            return self.audit._synapse_variable_owner(record, variable)
        raise ValueError(f"unsupported variable scope {variable.scope.value}")

    def _read_variables(self, variables: Iterable[Any]) -> Any:
        return self.np.asarray(
            [
                self.audit._read_variable(variable, self._owner_for(variable))
                for variable in variables
            ],
            dtype=float,
        )

    def capture_boundary_state(self) -> Dict[str, Any]:
        return {
            category: self._read_variables(variables)
            for category, variables in self.state_variables.items()
        }

    def run_burn_in(
        self, criteria: Optional[BurnInCriteria] = None
    ) -> Dict[str, Any]:
        """Advance until measured state changes satisfy all criteria."""

        if self.h is None:
            raise RuntimeError("prepare_teacher() must run before burn-in")
        criteria = criteria or BurnInCriteria()
        criteria.validate()
        self.audit._seed_neuron()
        self.audit._reset_owned_rngs()
        self.somatic_clamp.amp = 0.0
        self.h.finitialize(self.audit.v_init_mv)

        voltage_variables = self.state_variables["voltage"]
        calcium_variables = self.state_variables["calcium_ions"]
        slow_variables = [
            variable
            for variable in self.state_variables["mechanism_states"]
            if variable.mechanism in set(criteria.slow_mechanisms)
        ]
        if not slow_variables:
            raise RuntimeError("burn-in slow-state selection is empty")
        previous_voltage = self._read_variables(voltage_variables)
        previous_calcium = self._read_variables(calcium_variables)
        previous_slow = self._read_variables(slow_variables)
        history = []
        stable_run = 0
        converged = False

        for elapsed_ms in range(1, criteria.maximum_duration_ms + 1):
            self.audit._advance_exact(float(elapsed_ms))
            voltage = self._read_variables(voltage_variables)
            calcium = self._read_variables(calcium_variables)
            slow = self._read_variables(slow_variables)
            voltage_delta = float(
                self.np.max(self.np.abs(voltage - previous_voltage))
            )
            denominator = self.np.maximum(
                self.np.abs(previous_calcium), criteria.calcium_floor
            )
            calcium_delta = float(
                self.np.max(self.np.abs(calcium - previous_calcium) / denominator)
            )
            slow_delta = float(self.np.max(self.np.abs(slow - previous_slow)))
            passed = (
                voltage_delta <= criteria.voltage_delta_mv
                and calcium_delta <= criteria.calcium_relative_delta
                and slow_delta <= criteria.slow_state_delta
            )
            stable_run = stable_run + 1 if passed else 0
            history.append(
                {
                    "time_ms": elapsed_ms,
                    "max_abs_voltage_delta_mv": voltage_delta,
                    "max_relative_calcium_delta": calcium_delta,
                    "max_abs_slow_state_delta": slow_delta,
                    "criteria_passed": bool(passed),
                    "consecutive_passes": stable_run,
                }
            )
            previous_voltage, previous_calcium, previous_slow = (
                voltage,
                calcium,
                slow,
            )
            if stable_run >= criteria.consecutive_ms:
                converged = True
                break

        self.burnin_report = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "v_init_mv": self.audit.v_init_mv,
            "converged": converged,
            "burnin_duration_ms": float(self.h.t),
            "criteria": {
                **criteria.__dict__,
                "slow_mechanisms": list(criteria.slow_mechanisms),
            },
            "slow_state_variable_count": len(slow_variables),
            "calcium_variable_count": len(calcium_variables),
            "voltage_variable_count": len(voltage_variables),
            "metrics": history,
        }
        write_json(self.output_dir / "burnin_report.json", self.burnin_report)
        if not converged:
            raise RuntimeError(
                "burn-in did not converge before the configured safety limit; "
                "inspect burnin_report.json instead of accepting an arbitrary time"
            )

        rng_state = self.audit._snapshot_rng_sequences()
        self._write_native_snapshot(self.equilibrium_snapshot_path)
        write_json(
            self.equilibrium_rng_path,
            {
                "time_ms": float(self.h.t),
                "rng_mode": self.audit.rng_mode,
                "sequences": rng_state,
            },
        )
        boundary = self.capture_boundary_state()
        self.np.savez_compressed(
            self.snapshots_dir / "equilibrium_snapshot.named_state.npz",
            **boundary,
            rng_state=self.np.asarray(rng_state, dtype=float),
        )
        write_json(
            self.snapshots_dir / "equilibrium_snapshot.metadata.json",
            {
                "time_ms": float(self.h.t),
                "teacher_commit": PINNED_TEACHER_COMMIT,
                "native_snapshot": self.equilibrium_snapshot_path.name,
                "rng_snapshot": self.equilibrium_rng_path.name,
                "named_state": "equilibrium_snapshot.named_state.npz",
                "compatibility_warning": (
                    "NEURON SaveState files require identical mechanisms, section "
                    "creation order, point processes, and NetCons."
                ),
            },
        )
        return self.burnin_report

    def _write_native_snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = self.h.SaveState()
        saved.save()
        file_object = self.h.File(str(path))
        saved.fwrite(file_object)

    def _restore_native_snapshot(
        self, path: Path, rng_sequences: Sequence[float]
    ) -> None:
        self.somatic_clamp.amp = 0.0
        saved = self.h.SaveState()
        file_object = self.h.File(str(path))
        saved.fread(file_object)
        saved.restore()
        self.audit._restore_rng_sequences(rng_sequences)

    def _rekey_rngs(self, seed: int) -> None:
        for stream_id, rng in enumerate(self.audit.synapse_rngs):
            rng.Random123(int(seed), int(stream_id), 0)
            rng.negexp(1.0)
            rng.seq(0)

    def build_default_protocols(self) -> List[ProtocolTrajectory]:
        """Create 36 short trajectories spanning the four requested classes."""

        self._require_equilibrium()

        def synapses(region: str, inhibitory: bool = False, limit: int = 8) -> List[int]:
            class_name = "ProbUDFsyn2" if inhibitory else "ProbAMPANMDA2"
            selected = [
                int(record["synapse_id"])
                for record in self.audit.synapse_records
                if record["class_name"] == class_name
                and self.audit._region_for_segment(int(record["segment_id"])) == region
            ]
            if not selected:
                raise RuntimeError(f"no {class_name} synapses found in {region}")
            return selected[:limit]

        basal = synapses("basal")
        trunk = synapses("apical_trunk")
        nexus = synapses("nexus")
        tuft = synapses("tuft")
        basal_inh = synapses("basal", inhibitory=True)
        nexus_inh = synapses("nexus", inhibitory=True)

        def event(synapse_id: int, offset: float) -> InputAction:
            return InputAction("synaptic_event", offset, synapse_id=synapse_id)

        def current(amplitude: float, offset: float = 0.05, duration: float = 0.9) -> InputAction:
            return InputAction(
                "somatic_current",
                offset,
                duration_ms=duration,
                amplitude_na=amplitude,
            )

        templates: List[Tuple[str, str, Dict[int, Tuple[InputAction, ...]], bool]] = [
            ("rest_subthreshold", "rest_no_input", {}, False),
            (
                "rest_subthreshold",
                "weak_somatic_current",
                {3: (current(0.05),), 4: (current(0.05),)},
                False,
            ),
            (
                "rest_subthreshold",
                "sparse_balanced_synaptic",
                {
                    3: (event(basal[0], 0.25),),
                    7: (event(basal_inh[0], 0.55),),
                },
                False,
            ),
            (
                "local_synaptic",
                "basal_single_and_cluster",
                {
                    2: (event(basal[0], 0.5),),
                    6: tuple(event(item, 0.15 + 0.1 * i) for i, item in enumerate(basal[:6])),
                },
                True,
            ),
            (
                "local_synaptic",
                "trunk_nexus_tuft_cluster",
                {
                    2: tuple(event(item, 0.2 + 0.1 * i) for i, item in enumerate(trunk[:5])),
                    5: tuple(event(item, 0.2 + 0.1 * i) for i, item in enumerate(nexus[:5])),
                    8: tuple(event(item, 0.2 + 0.1 * i) for i, item in enumerate(tuft[:5])),
                },
                True,
            ),
            (
                "local_synaptic",
                "inhibitory_local",
                {
                    3: tuple(event(item, 0.2 + 0.1 * i) for i, item in enumerate(basal_inh[:5])),
                    7: tuple(event(item, 0.2 + 0.1 * i) for i, item in enumerate(nexus_inh[:5])),
                },
                True,
            ),
            (
                "somatic_events",
                "somatic_single_pulse",
                {4: (current(1.5),), 5: (current(1.5),)},
                True,
            ),
            (
                "somatic_events",
                "somatic_double_pulse",
                {3: (current(1.5),), 4: (current(1.5),), 7: (current(1.5),), 8: (current(1.5),)},
                True,
            ),
            (
                "somatic_events",
                "somatic_rapid_firing_candidate",
                {step: (current(1.8),) for step in range(2, 9)},
                True,
            ),
            (
                "dendritic_events",
                "nexus_hot_zone_candidate",
                {step: tuple(event(item, 0.1 + 0.1 * i) for i, item in enumerate(nexus[:8])) for step in (3, 4, 5)},
                True,
            ),
            (
                "dendritic_events",
                "distal_tuft_cluster",
                {step: tuple(event(item, 0.1 + 0.1 * i) for i, item in enumerate(tuft[:8])) for step in (4, 5, 6)},
                True,
            ),
            (
                "dendritic_events",
                "mixed_bap_ca_nmda_candidate",
                {
                    3: (current(1.5),),
                    4: tuple(event(item, 0.1 + 0.1 * i) for i, item in enumerate(nexus[:8])),
                    5: tuple(event(item, 0.1 + 0.1 * i) for i, item in enumerate(tuft[:8])),
                },
                True,
            ),
        ]
        plans = []
        split_index = {"train": 0, "validation": 1, "test": 2}
        for template_index, (category, protocol, actions, enriched) in enumerate(templates):
            for split in ("train", "validation", "test"):
                seed = 100_000 + template_index * 10 + split_index[split]
                plan = ProtocolTrajectory(
                    trajectory_id=f"{split}-{protocol}-seed{seed}",
                    category=category,
                    protocol=protocol,
                    seed=seed,
                    duration_ms=12,
                    split=split,
                    actions_by_step=actions,
                    event_enriched=enriched,
                )
                plan.validate()
                plans.append(plan)
        validate_split_isolation(
            {
                "trajectory_id": item.trajectory_id,
                "protocol": item.protocol,
                "seed": item.seed,
                "split": item.split,
            }
            for item in plans
        )
        return plans

    def generate_dataset(
        self, protocols: Optional[Sequence[ProtocolTrajectory]] = None
    ) -> Dict[str, Any]:
        """Write boundary states, inputs, microtraces, events, and snapshots."""

        self._require_equilibrium()
        protocols = list(protocols or self.build_default_protocols())
        if not protocols:
            raise ValueError("at least one protocol trajectory is required")
        for protocol in protocols:
            protocol.validate()
        widths = {
            category: len(variables)
            for category, variables in self.state_variables.items()
        }
        widths["rng_state"] = len(self.audit.synapse_rngs)
        micro_grid = self.np.linspace(0.0, 1.0, 41)
        transition_rows = []
        event_counts = Counter()
        total_events = 0

        with TransitionH5Writer(
            self.transition_path,
            widths,
            len(micro_grid),
            len(self.micro_variables),
            len(self.audit.live_segments),
            len(self.audit.representatives),
        ) as writer:
            writer.set_microtrace_grid(micro_grid)
            for trajectory in protocols:
                equilibrium_rng = json.loads(
                    self.equilibrium_rng_path.read_text(encoding="utf-8")
                )["sequences"]
                self._restore_native_snapshot(
                    self.equilibrium_snapshot_path, equilibrium_rng
                )
                self._rekey_rngs(trajectory.seed)
                trajectory_indices = []
                trajectory_times = []
                trajectory_traces = {
                    label: [] for label in self.audit.representatives
                }
                for step_index in range(trajectory.duration_ms):
                    actions = list(trajectory.actions_by_step.get(step_index, ()))
                    validate_input_actions(actions)
                    transition_id = writer.count
                    snapshot_path = (
                        self.snapshots_dir / f"transition_{transition_id:06d}.neuron.bin"
                    )
                    row = self._run_transition(
                        transition_id,
                        trajectory,
                        step_index,
                        actions,
                        snapshot_path,
                    )
                    index = writer.append(row)
                    trajectory_indices.append(index)
                    local_times = row["absolute_time_ms"]
                    if step_index:
                        local_times = local_times[1:]
                    trajectory_times.extend(local_times.tolist())
                    for probe_index, label in enumerate(self.audit.representatives):
                        values = row["micro_probe_voltage"][:, probe_index]
                        if step_index:
                            values = values[1:]
                        trajectory_traces[label].extend(values.tolist())
                    transition_rows.append(
                        {
                            **row["metadata"],
                            "action_count": len(actions),
                        }
                    )

                events = extract_events(
                    trajectory_times,
                    trajectory_traces,
                    self.event_definitions,
                )
                starts = [
                    float(transition_rows[index]["start_time_ms"])
                    for index in trajectory_indices
                ]
                assigned = event_ids_by_transition(events, starts)
                for local_index, event_ids in enumerate(assigned):
                    transition_index = trajectory_indices[local_index]
                    start = starts[local_index]
                    labels = []
                    for event_id in event_ids:
                        event = dict(events[event_id])
                        event["trajectory_event_id"] = event_id
                        event["transition_onset_offset_ms"] = (
                            event["onset_ms"] - start
                        )
                        labels.append(event)
                        event_counts[event["kind"]] += 1
                    writer.update_events(transition_index, labels)
                total_events += len(events)

        category_counts = Counter(item.category for item in protocols)
        split_counts = Counter(item.split for item in protocols)
        state_width = sum(len(items) for items in self.state_variables.values())
        size_estimate = estimate_dataset_size_bytes(
            len(transition_rows),
            state_width + len(self.audit.synapse_rngs),
            len(self.micro_variables),
            41,
            len(self.audit.live_segments),
        )
        native_snapshot_sizes = [
            path.stat().st_size
            for path in self.snapshots_dir.glob("transition_*.neuron.bin")
        ]
        mean_snapshot_bytes = int(
            sum(native_snapshot_sizes) / len(native_snapshot_sizes)
        )
        size_estimate.update(
            {
                "observed_mean_native_snapshot_bytes": mean_snapshot_bytes,
                "estimated_total_bytes_per_transition_including_snapshot": (
                    size_estimate[
                        "estimated_uncompressed_bytes_per_transition"
                    ]
                    + mean_snapshot_bytes
                ),
                "estimated_total_bytes_per_million_including_snapshots": (
                    size_estimate[
                        "estimated_uncompressed_bytes_per_million_transitions"
                    ]
                    + 1_000_000 * mean_snapshot_bytes
                ),
            }
        )
        self.dataset_manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "teacher_source_hashes": {
                str(record["path"]): str(record["sha256"])
                for record in self.audit.environment["source_files"]
            },
            "teacher_manifest": "manifest.json",
            "state_schema": "state_schema.json",
            "transition_store": self.transition_path.name,
            "snapshot_directory": "snapshots",
            "boundary_interval_ms": 1.0,
            "microtrace_interval_ms": DEFAULT_MICROTRACE_STEP_MS,
            "float_policy": {
                "boundary_state": "float64",
                "microtraces": "float32 after finite-value validation",
            },
            "trajectory_count": len(protocols),
            "transition_count": len(transition_rows),
            "trajectory_counts_by_category": dict(category_counts),
            "trajectory_counts_by_split": dict(split_counts),
            "event_counts": dict(event_counts),
            "event_count": total_events,
            "rng": {
                "mode": self.audit.rng_mode,
                "stream_count": len(self.audit.synapse_rngs),
                "split_isolation": "distinct trajectory seeds; no window split",
            },
            "solver_boundary_policy": (
                "CVODE re_init at every 1 ms boundary for symmetric, exactly "
                "replayable flow-map transitions; equations and tolerances unchanged"
            ),
            "full_segment_microtrace_policy": (
                "enabled for every transition only in this small diagnostic dataset"
            ),
            "size_estimate": size_estimate,
        }
        write_json(
            self.output_dir / "dataset_manifest.json", self.dataset_manifest
        )
        write_json(
            self.output_dir / "transition_index.json",
            {"transitions": transition_rows},
        )
        return self.dataset_manifest

    def _run_transition(
        self,
        transition_id: int,
        trajectory: ProtocolTrajectory,
        step_index: int,
        actions: Sequence[InputAction],
        snapshot_path: Optional[Path],
    ) -> Dict[str, Any]:
        start_time = float(self.h.t)
        state_t = self.capture_boundary_state()
        rng_t = self.np.asarray(
            self.audit._snapshot_rng_sequences(), dtype=float
        )
        if snapshot_path is not None:
            self._write_native_snapshot(snapshot_path)
        # SaveState cannot preserve adaptive solver history.  Both dataset
        # generation and replay therefore start the macro-step after re_init.
        self.cvode.re_init()
        public_actions = self._schedule_actions(start_time, actions)
        micro = self._sample_transition_microtrace(start_time)
        state_t_plus_1 = self.capture_boundary_state()
        rng_t_plus_1 = self.np.asarray(
            self.audit._snapshot_rng_sequences(), dtype=float
        )
        relative_snapshot = (
            snapshot_path.relative_to(self.output_dir).as_posix()
            if snapshot_path is not None
            else ""
        )
        return {
            "state_t": state_t,
            "state_t_plus_1": state_t_plus_1,
            "rng_t": rng_t,
            "rng_t_plus_1": rng_t_plus_1,
            "micro_selected": micro["selected"],
            "micro_probe_voltage": micro["probe_voltage"],
            "micro_all_voltage": micro["all_voltage"],
            "absolute_time_ms": micro["time_ms"],
            "inputs": public_actions,
            "events": [],
            "metadata": {
                "transition_id": int(transition_id),
                "trajectory_id": trajectory.trajectory_id,
                "category": trajectory.category,
                "protocol": trajectory.protocol,
                "split": trajectory.split,
                "seed": trajectory.seed,
                "step_index": int(step_index),
                "start_time_ms": start_time,
                "native_snapshot_ref": relative_snapshot,
            },
        }

    def _schedule_actions(
        self, start_time: float, actions: Sequence[InputAction]
    ) -> List[Dict[str, Any]]:
        self.callback_refs = []
        public = []
        for action in actions:
            action.validate()
            item = action.to_dict()
            if action.kind == "synaptic_event":
                record = self.audit.synapse_records[int(action.synapse_id)]
                if abs(action.weight_multiplier - 1.0) > 1e-12:
                    raise NotImplementedError(
                        "diagnostic-v0 supports explicit multiplier metadata but "
                        "keeps canonical NetCon weights at 1.0"
                    )
                item["rng_sequence_before"] = float(record["rng"].seq())
                item["release_observed"] = None
                item["release_observability"] = (
                    "not directly exposed without changing the teacher; infer later "
                    "from recorded conductance/state discontinuities"
                )
                record["netcon"].event(start_time + action.offset_ms)
            else:
                amplitude = float(action.amplitude_na)

                def set_amplitude(value: float = amplitude) -> None:
                    self.somatic_clamp.amp = value

                def clear_amplitude() -> None:
                    self.somatic_clamp.amp = 0.0

                self.callback_refs.extend([set_amplitude, clear_amplitude])
                self.cvode.event(start_time + action.offset_ms, set_amplitude)
                self.cvode.event(
                    start_time + action.offset_ms + float(action.duration_ms),
                    clear_amplitude,
                )
            public.append(item)
        return public

    def _sample_transition_microtrace(self, start_time: float) -> Dict[str, Any]:
        times = start_time + self.np.linspace(0.0, 1.0, 41)
        selected = []
        probes = []
        all_voltage = []
        representative_ids = list(self.audit.representatives.values())
        segment_ids = list(range(len(self.audit.live_segments)))
        for sample_time in times:
            self.audit._advance_exact(float(sample_time))
            selected.append(self._read_variables(self.micro_variables))
            probes.append(
                [
                    float(self.audit.live_segments[segment_id].v)
                    for segment_id in representative_ids
                ]
            )
            all_voltage.append(
                [
                    float(self.audit.live_segments[segment_id].v)
                    for segment_id in segment_ids
                ]
            )
        self.callback_refs = []
        return {
            "time_ms": self.np.asarray(times, dtype=float),
            "selected": self.np.asarray(selected, dtype=float),
            "probe_voltage": self.np.asarray(probes, dtype=float),
            "all_voltage": self.np.asarray(all_voltage, dtype=float),
        }

    def run_branching_diagnostic(self) -> Dict[str, Any]:
        """Prove same-future identity and different-future divergence."""

        self._require_equilibrium()
        equilibrium_rng = json.loads(
            self.equilibrium_rng_path.read_text(encoding="utf-8")
        )["sequences"]
        basal_id = int(
            self.audit._excitatory_synapse_for(
                self.audit.representatives["basal"]
            )["synapse_id"]
        )
        synaptic_future = [
            InputAction("synaptic_event", 0.1 + index * 0.1, synapse_id=basal_id)
            for index in range(8)
        ]
        # The deterministic current makes different-future divergence robust
        # even if every stochastic vesicle-release attempt happens to fail.
        synaptic_future.insert(
            0,
            InputAction(
                "somatic_current",
                0.05,
                duration_ms=0.9,
                amplitude_na=0.1,
            ),
        )
        empty_future: List[InputAction] = []

        def branch(actions: Sequence[InputAction]) -> Dict[str, Any]:
            self._restore_native_snapshot(
                self.equilibrium_snapshot_path, equilibrium_rng
            )
            trajectory = ProtocolTrajectory(
                "branch", "local_synaptic", "branching", self.seed, 1, "test"
            )
            return self._run_transition(
                -1, trajectory, 0, actions, snapshot_path=None
            )

        first = branch(synaptic_future)
        second = branch(synaptic_future)
        control = branch(empty_future)
        same_errors = {
            category: float(
                self.np.max(
                    self.np.abs(
                        first["state_t_plus_1"][category]
                        - second["state_t_plus_1"][category]
                    )
                )
            )
            for category in self.state_variables
        }
        different_voltage = float(
            self.np.max(
                self.np.abs(
                    first["state_t_plus_1"]["voltage"]
                    - control["state_t_plus_1"]["voltage"]
                )
            )
        )
        report = {
            "same_snapshot_same_input_max_error_by_category": same_errors,
            "same_input_numerically_identical": max(same_errors.values()) <= 1e-12,
            "different_input_max_voltage_difference_mv": different_voltage,
            "different_inputs_diverge": different_voltage > 1e-9,
            "synaptic_future": [item.to_dict() for item in synaptic_future],
            "control_future": [],
        }
        write_json(self.branching_dir / "branching_report.json", report)
        return report

    def validate_dataset(
        self, replay_count: int = 3
    ) -> Dict[str, Any]:
        """Run structural checks and replay sampled transitions and a test path."""

        base = validate_hdf5_store(self.transition_path)
        branching = self.run_branching_diagnostic()
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("validation requires h5py") from error

        replays = []
        with h5py.File(self.transition_path, "r") as handle:
            count = int(handle.attrs["transition_count"])
            voltage_ids = [
                int(variable.owner_id)
                for variable in self.state_variables["voltage"]
            ]
            segment_mapping_stable = voltage_ids == list(
                range(len(self.audit.live_segments))
            )
            start_voltage = handle[
                "microtraces/all_segment_voltage"
            ][:, 0, voltage_ids]
            stop_voltage = handle[
                "microtraces/all_segment_voltage"
            ][:, -1, voltage_ids]
            boundary_voltage_error = max(
                float(
                    self.np.max(
                        self.np.abs(
                            start_voltage - handle["states/voltage/t"][...]
                        )
                    )
                ),
                float(
                    self.np.max(
                        self.np.abs(
                            stop_voltage
                            - handle["states/voltage/t_plus_1"][...]
                        )
                    )
                ),
            )
            decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)
            missing_snapshot_refs = [
                decode(handle["metadata/native_snapshot_ref"][index])
                for index in range(count)
                if not (
                    self.output_dir
                    / decode(handle["metadata/native_snapshot_ref"][index])
                ).is_file()
            ]
            generator = random.Random(self.seed)
            selected = generator.sample(range(count), min(replay_count, count))
            for index in selected:
                replays.append(self._replay_hdf5_transition(handle, index))

            test_event_counts: Dict[str, int] = defaultdict(int)
            for index in range(count):
                if decode(handle["metadata/split"][index]) != "test":
                    continue
                trajectory_id = decode(
                    handle["metadata/trajectory_id"][index]
                )
                test_event_counts[trajectory_id] += len(
                    json.loads(handle["events/labels_json"][index])
                )
            test_trajectory = max(
                sorted(test_event_counts), key=test_event_counts.get
            )
            trajectory_replay = self._replay_hdf5_trajectory(
                handle, test_trajectory
            )

        source_hashes = {
            str(record["path"]): str(record["sha256"])
            for record in self.audit.environment["source_files"]
        }
        hashes_match = all(
            source_hashes.get(path) == expected
            for path, expected in self.expected_teacher_hashes.items()
        )
        blockers = list(base["issues"])
        if not hashes_match:
            blockers.append("teacher source hash differs from the audited teacher")
        if not all(item["reproduced"] for item in replays):
            blockers.append("one or more sampled transitions failed replay")
        if not trajectory_replay["reproduced"]:
            blockers.append("test trajectory or its event labels failed replay")
        if int(self.dataset_manifest.get("event_count", 0)) == 0:
            blockers.append("no candidate event was detected in the dataset")
        if not segment_mapping_stable:
            blockers.append("voltage state does not preserve segment_id order")
        if boundary_voltage_error > 1e-5:
            blockers.append("microtrace voltage does not match boundary voltage")
        if missing_snapshot_refs:
            blockers.append("one or more native transition snapshots are missing")
        if not branching["same_input_numerically_identical"]:
            blockers.append("same-input branching is not deterministic")
        if not branching["different_inputs_diverge"]:
            blockers.append("different-input branching did not diverge")

        report = {
            "valid": not blockers,
            "blockers": blockers,
            "hdf5": base,
            "teacher_commit_matches_audit": (
                git_commit(self.teacher_repo) == PINNED_TEACHER_COMMIT
            ),
            "teacher_source_hashes_match_audit": hashes_match,
            "sampled_transition_replays": replays,
            "test_trajectory_replay": trajectory_replay,
            "branching": branching,
            "segment_mapping_stable": segment_mapping_stable,
            "missing_native_snapshot_count": len(missing_snapshot_refs),
            "all_transition_boundary_voltage_error_mv": boundary_voltage_error,
            "microtrace_boundary_voltage_consistent": (
                boundary_voltage_error <= 1e-5
            ),
        }
        write_json(self.output_dir / "validation_report.json", report)
        if blockers:
            raise RuntimeError(f"diagnostic dataset validation failed: {blockers}")
        self._plot_examples()
        self._write_artifact_index()
        return report

    def _replay_hdf5_transition(
        self, handle: Any, index: int, *, include_arrays: bool = False
    ) -> Dict[str, Any]:
        decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)
        snapshot = self.output_dir / decode(
            handle["metadata/native_snapshot_ref"][index]
        )
        rng = handle["rng_state/t"][index, :]
        self._restore_native_snapshot(snapshot, rng)
        actions = [
            InputAction(
                kind=row["kind"],
                offset_ms=row["offset_ms"],
                synapse_id=row.get("synapse_id"),
                weight_multiplier=row.get("weight_multiplier", 1.0),
                duration_ms=row.get("duration_ms"),
                amplitude_na=row.get("amplitude_na"),
                metadata=row.get("metadata", {}),
            )
            for row in json.loads(handle["inputs/ordered_actions_json"][index])
        ]
        trajectory = ProtocolTrajectory(
            decode(handle["metadata/trajectory_id"][index]),
            decode(handle["metadata/category"][index]),
            decode(handle["metadata/protocol"][index]),
            int(handle["metadata/seed"][index]),
            1,
            decode(handle["metadata/split"][index]),
        )
        replay = self._run_transition(
            index,
            trajectory,
            int(handle["metadata/step_index"][index]),
            actions,
            snapshot_path=None,
        )
        errors = {
            category: float(
                self.np.max(
                    self.np.abs(
                        replay["state_t_plus_1"][category]
                        - handle[f"states/{category}/t_plus_1"][index, :]
                    )
                )
            )
            for category in self.state_variables
        }
        rng_error = float(
            self.np.max(
                self.np.abs(
                    replay["rng_t_plus_1"] - handle["rng_state/t_plus_1"][index, :]
                )
            )
        )
        micro_error = float(
            self.np.max(
                self.np.abs(
                    replay["micro_probe_voltage"]
                    - handle["microtraces/probe_voltage"][index, :, :]
                )
            )
        )
        voltage_ids = [
            int(variable.owner_id)
            for variable in self.state_variables["voltage"]
        ]
        all_voltage = handle["microtraces/all_segment_voltage"][index, :, :]
        boundary_voltage = handle["states/voltage/t"][index, :]
        boundary_voltage_1 = handle["states/voltage/t_plus_1"][index, :]
        boundary_error = max(
            float(self.np.max(self.np.abs(all_voltage[0, voltage_ids] - boundary_voltage))),
            float(self.np.max(self.np.abs(all_voltage[-1, voltage_ids] - boundary_voltage_1))),
        )
        maximum = max([*errors.values(), rng_error, micro_error])
        report = {
            "transition_id": int(index),
            "max_state_error_by_category": errors,
            "max_rng_sequence_error": rng_error,
            "max_probe_microtrace_error_mv": micro_error,
            "microtrace_boundary_voltage_error_mv": boundary_error,
            "reproduced": maximum <= 1e-5 and boundary_error <= 1e-5,
        }
        if include_arrays:
            report["replayed_probe_microtrace"] = replay[
                "micro_probe_voltage"
            ]
        return report

    def _replay_hdf5_trajectory(
        self, handle: Any, trajectory_id: str
    ) -> Dict[str, Any]:
        decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)
        indices = [
            index
            for index in range(int(handle.attrs["transition_count"]))
            if decode(handle["metadata/trajectory_id"][index]) == trajectory_id
        ]
        indices.sort(key=lambda index: int(handle["metadata/step_index"][index]))
        state_errors = []
        times = []
        traces = {label: [] for label in self.audit.representatives}
        stored_events = []
        for position, index in enumerate(indices):
            replay = self._replay_hdf5_transition(
                handle, index, include_arrays=True
            )
            state_errors.append(
                max(replay["max_state_error_by_category"].values())
            )
            local_times = (
                float(handle["metadata/start_time_ms"][index])
                + handle["microtraces/time_offsets_ms"][...]
            )
            if position:
                local_times = local_times[1:]
            times.extend(local_times.tolist())
            values = replay["replayed_probe_microtrace"]
            for probe_index, label in enumerate(self.audit.representatives):
                trace = values[:, probe_index]
                if position:
                    trace = trace[1:]
                traces[label].extend(trace.tolist())
            stored_events.extend(
                json.loads(handle["events/labels_json"][index])
            )
        replay_events = extract_events(times, traces, self.event_definitions)

        def canonical(events: Sequence[Mapping[str, Any]]) -> List[Tuple[Any, ...]]:
            return [
                (
                    event["kind"],
                    event["segment_id"],
                    round(float(event["onset_ms"]), 9),
                    round(float(event["peak_ms"]), 9),
                    round(float(event["offset_ms"]), 9),
                )
                for event in events
            ]

        event_match = canonical(stored_events) == canonical(replay_events)
        return {
            "trajectory_id": trajectory_id,
            "transition_count": len(indices),
            "max_state_error": max(state_errors) if state_errors else None,
            "stored_event_count": len(stored_events),
            "replayed_event_count": len(replay_events),
            "events_match": event_match,
            "reproduced": bool(state_errors)
            and max(state_errors) <= 1e-5
            and event_match,
        }

    def _plot_examples(self) -> None:
        import h5py

        with h5py.File(self.transition_path, "r") as handle:
            decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)
            categories = {}
            for index in range(int(handle.attrs["transition_count"])):
                category = decode(handle["metadata/category"][index])
                categories.setdefault(
                    category, decode(handle["metadata/trajectory_id"][index])
                )
            for category, trajectory_id in categories.items():
                indices = [
                    index
                    for index in range(int(handle.attrs["transition_count"]))
                    if decode(handle["metadata/trajectory_id"][index])
                    == trajectory_id
                ]
                indices.sort(
                    key=lambda index: int(handle["metadata/step_index"][index])
                )
                time = []
                traces = {
                    label: [] for label in self.audit.representatives
                }
                events = []
                for position, index in enumerate(indices):
                    local_time = (
                        float(handle["metadata/start_time_ms"][index])
                        + handle["microtraces/time_offsets_ms"][...]
                    )
                    values = handle["microtraces/probe_voltage"][index, :, :]
                    if position:
                        local_time = local_time[1:]
                        values = values[1:, :]
                    time.extend(local_time.tolist())
                    for probe_index, label in enumerate(
                        self.audit.representatives
                    ):
                        traces[label].extend(values[:, probe_index].tolist())
                    events.extend(
                        json.loads(handle["events/labels_json"][index])
                    )
                figure, axis = self.audit.plt.subplots(figsize=(10, 5))
                for label in self.audit.representatives:
                    axis.plot(time, traces[label], label=label)
                for event in events:
                    axis.axvline(
                        float(event["onset_ms"]), color="black", alpha=0.15
                    )
                axis.set(
                    title=(
                        f"Diagnostic trajectory: {category} "
                        f"({len(events)} detected events)"
                    ),
                    xlabel="absolute teacher time (ms)",
                    ylabel="voltage (mV)",
                )
                axis.grid(alpha=0.2)
                axis.legend(ncol=3)
                figure.tight_layout()
                figure.savefig(
                    self.figures_dir / f"example_{category}.png", dpi=160
                )
                self.audit.plt.close(figure)

    def _write_artifact_index(self) -> None:
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
        write_json(
            self.output_dir / "artifact_index.json",
            {"schema_version": DATASET_SCHEMA_VERSION, "artifacts": records},
        )

    def package_artifacts(self, archive_base: Optional[Path] = None) -> Path:
        base = Path(archive_base or self.output_dir.parent / self.output_dir.name)
        archive = shutil.make_archive(
            str(base), "zip", root_dir=self.output_dir.parent, base_dir=self.output_dir.name
        )
        return Path(archive)

    def _require_equilibrium(self) -> None:
        if self.h is None:
            raise RuntimeError("prepare_teacher() must run first")
        if not self.equilibrium_snapshot_path.is_file():
            raise RuntimeError("run_burn_in() must produce the equilibrium snapshot")


def expected_audit_hashes() -> Dict[str, str]:
    """Critical hashes accepted by the successful canonical teacher audit."""

    return {
        "simulate_L5PC_and_create_dataset.py": (
            "fa5751daa7197276a6596d53d7adcee401ba15a479503d0c0a48044a12782c51"
        ),
        "L5PC_NEURON_simulation/L5PCbiophys5b.hoc": (
            "dec2a62342faaeeca6436110472f9485e069a434ef993649c6f2820c56cc28f3"
        ),
        "L5PC_NEURON_simulation/L5PCtemplate_2.hoc": (
            "8020f4a3fd4b821ad84ba2b1aa4b057c7018227cde3b6a00ed421e5545800e1c"
        ),
        "L5PC_NEURON_simulation/morphologies/cell1.asc": (
            "a8afc0925afec1dd9241528ee6ccfdcb1c321904d876835990592c2573567383"
        ),
        "L5PC_NEURON_simulation/mods/CaDynamics_E2.mod": (
            "fee7fa2ad830cc009dcce02a6247096fb1431e29705819cdec6b2ccd6cabe0d6"
        ),
        "L5PC_NEURON_simulation/mods/Ca_HVA.mod": (
            "db310c0746fc0f86e27101cd406feab92d75a1dddf2f2d7259480cd50e5de6df"
        ),
        "L5PC_NEURON_simulation/mods/Ca_LVAst.mod": (
            "94e1dae140da644b71fb7c64cb549f76851c8f27860978f4474a16e5efdf99dc"
        ),
        "L5PC_NEURON_simulation/mods/Ih.mod": (
            "c8fce16998f0915f4ea4e7ee17db3143c28ca5b668d0446d81db052546a60d66"
        ),
        "L5PC_NEURON_simulation/mods/Im.mod": (
            "93074ca6f0480cddd130b5120205d4b06dc689c0dd03e1da84b096f5bc3b47cb"
        ),
        "L5PC_NEURON_simulation/mods/K_Pst.mod": (
            "41ef49ec8f490210f10c647753e29ec6461e8dffcae8b593a110546a9e9af546"
        ),
        "L5PC_NEURON_simulation/mods/K_Tst.mod": (
            "b8adba9bad5c4f5bc7e2231b42fc4d649e5bbaa42286bf6666a8c94c93ff595e"
        ),
        "L5PC_NEURON_simulation/mods/NaTa_t.mod": (
            "91f6c02eabee637cadd7014d83782388ede064f06c60ea370047f2a19e5e7476"
        ),
        "L5PC_NEURON_simulation/mods/NaTs2_t.mod": (
            "ac334e8e8fa08a88941b9497385cfeab53439971fc0dd27bcfb1e575d54204ea"
        ),
        "L5PC_NEURON_simulation/mods/Nap_Et2.mod": (
            "d2b8a386a1e6ac87aa701e065a0d1a998fb039082469b5336739596de9d579ee"
        ),
        "L5PC_NEURON_simulation/mods/ProbAMPANMDA2.mod": (
            "e1f45094bb8bcb8583aa3decbc4605d6a8c68b0ea3ece07ba278b342642ea20a"
        ),
        "L5PC_NEURON_simulation/mods/ProbAMPANMDA_3.mod": (
            "866447904e2b7221d9fc80081ffa33733f863bdb87af03cceb41b4e4a7d31f09"
        ),
        "L5PC_NEURON_simulation/mods/ProbAMPANMDA_EMS.mod": (
            "148780464aec6b25d70fe16f7ab6b5342c26dc3351a02e86de461a60ebc14c1d"
        ),
        "L5PC_NEURON_simulation/mods/ProbGABAAB_EMS.mod": (
            "7e433b43dc41db6d5a824206a43c8b0df1e7caa55e9ea7d5ce58fc7ecbc592e3"
        ),
        "L5PC_NEURON_simulation/mods/ProbUDFsyn2.mod": (
            "60a0ab09ea4d25f51f4bd8d188d8f7dc2da822812f701df12113db182e3f439d"
        ),
        "L5PC_NEURON_simulation/mods/SK_E2.mod": (
            "c1cbebfdf892a8d86f782b369c5a975c571302236dcbc444c433b4e044e49949"
        ),
        "L5PC_NEURON_simulation/mods/SKv3_1.mod": (
            "dbff870df17bea4ff119d964fb330756b087e069ca754a9ebfe74287a7d53e41"
        ),
        "L5PC_NEURON_simulation/mods/epsp.mod": (
            "7e9a2beafabff999bf269f4a500d8ee80241daed31b151c9471afe59ded1319b"
        ),
    }
