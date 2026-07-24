"""Versioned full-state diagnostic dataset with confirmed dendritic events.

This module extends the audited 1 ms transition generator without changing the
Hay teacher.  The two positive dendritic recipes are loaded from the validated
01b artifact bundle; nearby controls are derived explicitly and never replace
the confirmed positive schedules.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..hayflow_data import (
    DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION,
    InputAction,
    ProtocolTrajectory,
    validate_split_isolation,
    write_json,
)
from .audit import git_commit, sha256_file
from .audit_runtime import PINNED_TEACHER_COMMIT
from .diagnostic_dataset import DiagnosticDatasetSession, _ConsoleProgress
from .dendritic_calibration import (
    DendriticProtocolCalibrator,
    build_candidate_actions,
    candidate_from_selected_protocol,
)
from .event_extractor import default_event_definitions, extract_events


PLATEAU_PROTOCOL_ID = (
    "tuft_nmda_plateau-tuft-n12-b2-r1-unpaired-branch-w400"
)
CALCIUM_PROTOCOL_ID = (
    "paired_bap_calcium_spike-hot_zone-n12-b3-r1-paired-nearest-w800"
)
CONFIRMED_PROTOCOL_IDS = (PLATEAU_PROTOCOL_ID, CALCIUM_PROTOCOL_ID)
CONFIRMED_SEEDS = (310001, 310002, 310003)
REQUIRED_SPLITS = {
    "train",
    "validation",
    "deterministic_test",
    "event_boundary_test",
    "branching_test",
}


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_artifact_root(path: Path) -> Path:
    path = Path(path).resolve()
    if (path / "artifact_index.json").is_file():
        return path
    matches = list(path.rglob("artifact_index.json"))
    if len(matches) != 1:
        raise RuntimeError(
            "calibration source must contain exactly one artifact_index.json"
        )
    return matches[0].parent


def validate_calibration_artifacts(root: Path) -> Dict[str, Any]:
    """Verify every artifact recorded by notebook 01b."""

    root = _find_artifact_root(root)
    index = json.loads((root / "artifact_index.json").read_text(encoding="utf-8"))
    missing = []
    size_mismatches = []
    hash_mismatches = []
    for record in index["artifacts"]:
        relative = str(record["path"])
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            size_mismatches.append(relative)
        if sha256_file(path) != str(record["sha256"]):
            hash_mismatches.append(relative)
    valid = not (missing or size_mismatches or hash_mismatches)
    return {
        "valid": valid,
        "artifact_count": len(index["artifacts"]),
        "expected_artifact_count": 88,
        "artifact_count_matches_reference": len(index["artifacts"]) == 88,
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "root": str(root),
    }


def _input_action(row: Mapping[str, Any]) -> InputAction:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return InputAction(
        kind=str(row["kind"]),
        offset_ms=float(row["offset_ms"]),
        synapse_id=(
            None if row.get("synapse_id") is None else int(row["synapse_id"])
        ),
        weight_multiplier=float(row.get("weight_multiplier", 1.0)),
        duration_ms=(
            None if row.get("duration_ms") is None else float(row["duration_ms"])
        ),
        amplitude_na=(
            None if row.get("amplitude_na") is None else float(row["amplitude_na"])
        ),
        metadata=dict(metadata),
    )


def actions_from_selected_protocol(
    protocol: Mapping[str, Any],
) -> Dict[int, Tuple[InputAction, ...]]:
    return {
        int(step): tuple(_input_action(row) for row in rows)
        for step, rows in protocol["input_schedule_template"].items()
    }


def filter_synaptic_actions(
    actions_by_step: Mapping[int, Sequence[InputAction]],
    retained_synapse_ids: Iterable[int],
    *,
    keep_somatic_current: bool = True,
) -> Dict[int, Tuple[InputAction, ...]]:
    retained = set(map(int, retained_synapse_ids))
    result = {}
    for step, actions in actions_by_step.items():
        selected = [
            action
            for action in actions
            if (
                action.kind == "somatic_current" and keep_somatic_current
            )
            or (
                action.kind == "synaptic_event"
                and int(action.synapse_id) in retained
            )
        ]
        if selected:
            result[int(step)] = tuple(selected)
    return result


class DiagnosticDatasetV1Session(DiagnosticDatasetSession):
    """Build and validate ``artifacts/diagnostic_dataset_v1``."""

    def __init__(
        self,
        elm_repo: Path,
        teacher_repo: Path,
        *,
        calibration_source: Path,
        dataset_config_path: Path,
        output_dir: Optional[Path] = None,
        seed: int = 271828,
        expected_teacher_hashes: Optional[Mapping[str, str]] = None,
        native_snapshot_stride: int = 10,
    ) -> None:
        destination = Path(
            output_dir
            or Path(elm_repo) / "artifacts" / "diagnostic_dataset_v1"
        )
        super().__init__(
            elm_repo,
            teacher_repo,
            output_dir=destination,
            seed=seed,
            expected_teacher_hashes=expected_teacher_hashes,
        )
        if int(native_snapshot_stride) <= 0:
            raise ValueError("native_snapshot_stride must be positive")
        self.native_snapshot_stride = int(native_snapshot_stride)
        self.calibration_source = Path(calibration_source).resolve()
        self.dataset_config_path = Path(dataset_config_path).resolve()
        self.calibration_root: Optional[Path] = None
        self.calibration_integrity: Dict[str, Any] = {}
        self.selected_document: Dict[str, Any] = {}
        self.confirmation_document: Dict[str, Any] = {}
        self.selected_protocols: Dict[str, Mapping[str, Any]] = {}
        self.protocol_registry: Dict[str, Dict[str, Any]] = {}
        self.protocol_rows: List[Dict[str, Any]] = []
        self.prefix_overlap_rows: List[Dict[str, Any]] = []
        self.preflight_report: Dict[str, Any] = {}
        self.reference_trace_by_protocol_seed: Dict[Tuple[str, int], Path] = {}
        self.runtime_reference_trace_by_protocol_seed: Dict[
            Tuple[str, int], Path
        ] = {}
        self.preflight_prefix_duration_ms = 0
        self.alternate_tuft_selection: Dict[str, Any] = {}
        self.micro_observable_ids = [
            "cai_event_probe_mM",
            "ica_event_probe_mA_per_cm2",
            "ica_hva_event_probe_mA_per_cm2",
            "ica_lva_event_probe_mA_per_cm2",
            "sum_g_NMDA_uS",
            "sum_i_NMDA_nA",
            "sum_g_AMPA_uS",
            "sum_i_AMPA_nA",
        ]

    def _prepare_calibration_source(self) -> Path:
        source = self.calibration_source
        if source.is_file() and source.suffix.lower() == ".zip":
            extracted = self.output_dir.parent / "_calibration_reference_01b"
            if extracted.exists():
                shutil.rmtree(extracted)
            shutil.unpack_archive(str(source), str(extracted), "zip")
            source = extracted
        root = _find_artifact_root(source)
        integrity = validate_calibration_artifacts(root)
        if not integrity["valid"] or not integrity[
            "artifact_count_matches_reference"
        ]:
            raise RuntimeError(
                f"01b calibration artifact validation failed: {integrity}"
            )
        self.calibration_integrity = integrity
        self.calibration_root = root
        return root

    @staticmethod
    def _protocol_plan_sha256(
        protocols: Sequence[ProtocolTrajectory],
    ) -> str:
        payload = {
            "trajectories": [
                {
                    "trajectory_id": row.trajectory_id,
                    "category": row.category,
                    "protocol": row.protocol,
                    "protocol_id": row.protocol_id,
                    "protocol_variant": row.protocol_variant,
                    "seed": row.seed,
                    "duration_ms": row.duration_ms,
                    "split": row.split,
                    "stimulus_onset_step": row.stimulus_onset_step,
                    "actions": {
                        str(step): [action.to_dict() for action in actions]
                        for step, actions in sorted(row.actions_by_step.items())
                    },
                }
                for row in sorted(protocols, key=lambda item: item.trajectory_id)
            ]
        }
        return canonical_json_sha256(payload)

    def prepare_v1_contract(self) -> Dict[str, Any]:
        """Bind the exact 01b protocols and extend the static probe schema."""

        if self.h is None:
            raise RuntimeError("prepare_teacher() must run first")
        root = self._prepare_calibration_source()
        self.selected_document = json.loads(
            (root / "selected_dendritic_protocols.json").read_text(
                encoding="utf-8"
            )
        )
        self.confirmation_document = json.loads(
            (root / "confirmation_report.json").read_text(encoding="utf-8")
        )
        if self.selected_document["teacher_commit"] != PINNED_TEACHER_COMMIT:
            raise RuntimeError("selected protocols use a different teacher commit")
        if not self.confirmation_document.get("valid"):
            raise RuntimeError("long-horizon dendritic confirmation is not valid")
        self.selected_protocols = dict(
            self.selected_document["selected_protocols"]
        )
        observed_ids = {
            str(row["candidate_id"]) for row in self.selected_protocols.values()
        }
        if observed_ids != set(CONFIRMED_PROTOCOL_IDS):
            raise RuntimeError(
                f"confirmed protocol ids differ from contract: {observed_ids}"
            )

        plateau = self.selected_protocols["tuft_nmda_plateau"]
        center_id = int(plateau["event_probe_segment_id"])
        if center_id != 460:
            raise RuntimeError("confirmed tuft cluster center changed")
        self.audit.representatives["tuft_cluster_center"] = center_id
        parents = {}
        distances = {}
        regions = {}
        for row in self.audit.segment_df.itertuples(index=False):
            segment_id = int(row.segment_id)
            parent = row.parent_segment_id
            parents[segment_id] = (
                None if self.audit.pd.isna(parent) else int(parent)
            )
            distances[segment_id] = float(row.distance_from_soma_um)
            regions[segment_id] = str(row.region)
        center_ancestors = set(
            DendriticProtocolCalibrator._ancestors(center_id, parents)
        )

        def same_confirmed_path(segment_id: int) -> bool:
            return segment_id in center_ancestors or center_id in set(
                DendriticProtocolCalibrator._ancestors(segment_id, parents)
            )

        alternate_candidates = [
            (int(record["segment_id"]), int(record["synapse_id"]))
            for record in self.audit.synapse_records
            if record["class_name"] == "ProbAMPANMDA2"
            and regions[int(record["segment_id"])] == "tuft"
            and not same_confirmed_path(int(record["segment_id"]))
        ]
        alternate_center, alternate_rows = (
            DendriticProtocolCalibrator.compact_synapse_cluster(
                alternate_candidates,
                8,
                parents,
                distances,
                target_segment_id=int(self.audit.representatives["tuft"]),
                maximum_center_distance_um=220.0,
            )
        )
        self.alternate_tuft_selection = {
            "center_segment_id": int(alternate_center),
            "synapse_ids": [int(row[2]) for row in alternate_rows],
            "segment_ids": [int(row[1]) for row in alternate_rows],
            "center_distances_um": [float(row[0]) for row in alternate_rows],
        }
        self.audit.representatives["tuft_alternate_cluster_center"] = int(
            alternate_center
        )
        self._build_state_schema()
        base_definitions = default_event_definitions(
            self.audit.representatives
        )
        self.event_definitions = [
            replace(
                definition,
                signal="tuft_cluster_center",
                segment_id=center_id,
                region="tuft_stimulus_cluster",
            )
            if definition.kind in {"nmda_spike", "nmda_plateau"}
            else definition
            for definition in base_definitions
        ]
        self.event_definitions.extend(
            replace(
                definition,
                signal="tuft_alternate_cluster_center",
                segment_id=int(alternate_center),
                region="tuft_alternate_branch_cluster",
            )
            for definition in base_definitions
            if definition.kind in {"nmda_spike", "nmda_plateau"}
        )
        self.state_schema["schema_version"] = (
            DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION
        )
        self.state_schema["core_state_width"] = sum(
            int(self.state_schema["categories"][name]["width"])
            for name in (
                "voltage",
                "mechanism_states",
                "calcium_ions",
                "synapse_states",
            )
        )
        self.state_schema["privileged_state_width"] = int(
            self.state_schema["categories"]["currents_conductances"]["width"]
        )
        self.state_schema["protocol_microtrace_observable_ids"] = list(
            self.micro_observable_ids
        )
        write_json(self.output_dir / "state_schema.json", self.state_schema)
        write_json(
            self.output_dir / "event_definition_config.json",
            {
                "schema_version": DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION,
                "status": "confirmed_protocol_labels_with_configurable_thresholds",
                "definitions": [row.to_dict() for row in self.event_definitions],
                "right_censoring_policy": (
                    "retain censored labels but exclude duration/offset losses"
                ),
            },
        )
        for trial in self.confirmation_document["trials"]:
            key = (str(trial["candidate_id"]), int(trial["seed"]))
            self.reference_trace_by_protocol_seed[key] = root / str(
                trial["trace_path"]
            )
        return {
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "repository_commit": git_commit(self.elm_repo),
            "calibration_artifacts": {
                key: value
                for key, value in self.calibration_integrity.items()
                if key != "root"
            },
            "confirmed_protocol_ids": list(CONFIRMED_PROTOCOL_IDS),
            "confirmed_seeds": list(CONFIRMED_SEEDS),
            "probe_order": list(self.audit.representatives),
            "alternate_tuft_control": self.alternate_tuft_selection,
            "core_state_width": self.state_schema["core_state_width"],
            "privileged_state_width": self.state_schema[
                "privileged_state_width"
            ],
        }

    @staticmethod
    def _synaptic_ids(actions: Mapping[int, Sequence[InputAction]]) -> List[int]:
        return sorted(
            {
                int(action.synapse_id)
                for rows in actions.values()
                for action in rows
                if action.kind == "synaptic_event"
            }
        )

    def _register(
        self,
        plans: List[ProtocolTrajectory],
        *,
        trajectory_id: str,
        category: str,
        protocol: str,
        protocol_id: str,
        protocol_variant: str,
        seed: int,
        duration_ms: int,
        split: str,
        actions_by_step: Mapping[int, Sequence[InputAction]],
        stimulus_onset_step: int,
        required_event_kinds: Sequence[str] = (),
        negative_control: bool = False,
        event_probe_label: Optional[str] = None,
        event_probe_segment_id: Optional[int] = None,
        control_of: Optional[str] = None,
        branch_future: Optional[str] = None,
    ) -> None:
        normalized = {
            int(step): tuple(actions)
            for step, actions in actions_by_step.items()
        }
        selected_synapses = self._synaptic_ids(normalized)
        metadata = {
            "event_probe_segment_id": event_probe_segment_id,
            "selected_synapse_ids": selected_synapses,
            "control_of": control_of,
            "branch_future": branch_future,
            "expected_event_present": bool(required_event_kinds)
            and not negative_control,
        }
        plan = ProtocolTrajectory(
            trajectory_id=trajectory_id,
            category=category,
            protocol=protocol,
            seed=int(seed),
            duration_ms=int(duration_ms),
            split=split,
            actions_by_step=normalized,
            event_enriched=bool(required_event_kinds),
            protocol_id=protocol_id,
            protocol_variant=protocol_variant,
            stimulus_onset_step=int(stimulus_onset_step),
            required_event_kinds=tuple(required_event_kinds),
            negative_control=bool(negative_control),
            event_probe_label=event_probe_label,
            snapshot_source="equilibrium_snapshot",
            metadata=metadata,
        )
        plan.validate()
        plans.append(plan)
        row = {
            "trajectory_id": trajectory_id,
            "category": category,
            "protocol": protocol,
            "protocol_id": protocol_id,
            "protocol_variant": protocol_variant,
            "seed": int(seed),
            "duration_ms": int(duration_ms),
            "split": split,
            "stimulus_onset_step": int(stimulus_onset_step),
            "required_event_kinds": list(required_event_kinds),
            "negative_control": bool(negative_control),
            "event_probe_label": event_probe_label,
            "event_probe_segment_id": event_probe_segment_id,
            "selected_synapse_ids": selected_synapses,
            "control_of": control_of,
            "branch_future": branch_future,
            "snapshot_source": "equilibrium_snapshot",
            "configuration_sha256": canonical_json_sha256(
                {
                    "protocol": protocol,
                    "protocol_id": protocol_id,
                    "variant": protocol_variant,
                    "seed": int(seed),
                    "duration_ms": int(duration_ms),
                    "actions": {
                        str(step): [action.to_dict() for action in actions]
                        for step, actions in sorted(normalized.items())
                    },
                }
            ),
        }
        self.protocol_rows.append(row)
        self.protocol_registry[trajectory_id] = row

    def build_v1_protocols(
        self, *, dendritic_duration_ms: int = 80
    ) -> List[ProtocolTrajectory]:
        """Create the five required groups with whole-trajectory splits."""

        self._require_equilibrium()
        if not self.selected_protocols:
            raise RuntimeError("prepare_v1_contract() must run first")
        if self.calibrated_somatic_current_na is None:
            self.calibrate_somatic_spike_current()
        if self.calibrated_somatic_single_spike_current_na is None:
            self.calibrate_somatic_single_spike_current()
        if int(dendritic_duration_ms) < 70:
            raise ValueError(
                "confirmed dendritic trajectories need at least 70 ms"
            )
        self.protocol_registry = {}
        self.protocol_rows = []
        plans: List[ProtocolTrajectory] = []

        paired_spike_current = float(self.calibrated_somatic_current_na)
        single_spike_current = float(
            self.calibrated_somatic_single_spike_current_na
        )
        selected_ca_current = float(
            self.selected_protocols["paired_bap_calcium_spike"]
            ["somatic_current_na"]
        )
        if abs(paired_spike_current - selected_ca_current) > 1e-12:
            raise RuntimeError(
                "somatic calibration no longer matches the confirmed Ca protocol"
            )

        def current(amplitude: float) -> InputAction:
            return InputAction(
                "somatic_current",
                0.05,
                duration_ms=0.9,
                amplitude_na=float(amplitude),
            )

        def event(synapse_id: int, offset: float) -> InputAction:
            return InputAction(
                "synaptic_event", float(offset), synapse_id=int(synapse_id)
            )

        basal_exc = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbAMPANMDA2"
            and self.audit._region_for_segment(int(row["segment_id"]))
            == "basal"
        ][:8]
        basal_inh = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbUDFsyn2"
            and self.audit._region_for_segment(int(row["segment_id"]))
            == "basal"
        ][:4]
        if len(basal_exc) < 8 or len(basal_inh) < 1:
            raise RuntimeError("canonical basal synapse pool is incomplete")

        # A. Rest and subthreshold controls.
        self._register(
            plans,
            trajectory_id="train-rest-no-input-seed110001",
            category="rest_subthreshold",
            protocol="rest_no_input",
            protocol_id="rest_no_input",
            protocol_variant="no_input",
            seed=110001,
            duration_ms=12,
            split="train",
            actions_by_step={},
            stimulus_onset_step=0,
        )
        self._register(
            plans,
            trajectory_id="validation-sparse-subthreshold-seed120001",
            category="rest_subthreshold",
            protocol="sparse_subthreshold_synaptic",
            protocol_id="sparse_subthreshold_synaptic",
            protocol_variant="sparse_balanced",
            seed=120001,
            duration_ms=12,
            split="validation",
            actions_by_step={
                3: (event(basal_exc[0], 0.25),),
                7: (event(basal_inh[0], 0.55),),
            },
            stimulus_onset_step=3,
        )
        self._register(
            plans,
            trajectory_id="deterministic-weak-current-seed130001",
            category="rest_subthreshold",
            protocol="weak_somatic_current",
            protocol_id="weak_somatic_current",
            protocol_variant="subthreshold",
            seed=130001,
            duration_ms=12,
            split="deterministic_test",
            actions_by_step={3: (current(0.05),), 4: (current(0.05),)},
            stimulus_onset_step=3,
        )

        # B. Somatic single, double, rapid and recovery protocols.
        somatic_specs = [
            (
                "somatic_single_spike",
                210001,
                "train",
                16,
                {3: (current(single_spike_current),)},
            ),
            (
                "somatic_double_pulse",
                210002,
                "validation",
                16,
                {
                    3: (current(single_spike_current),),
                    7: (current(single_spike_current),),
                },
            ),
            (
                "somatic_rapid_firing",
                210003,
                "deterministic_test",
                16,
                {
                    step: (current(single_spike_current),)
                    for step in range(2, 9)
                },
            ),
            (
                "somatic_recovery",
                210004,
                "event_boundary_test",
                20,
                {
                    2: (current(single_spike_current),),
                    10: (current(single_spike_current),),
                },
            ),
        ]
        for name, seed, split, duration, actions in somatic_specs:
            self._register(
                plans,
                trajectory_id=f"{split}-{name}-seed{seed}",
                category="somatic_events",
                protocol=name,
                protocol_id=name,
                protocol_variant="canonical_current",
                seed=seed,
                duration_ms=duration,
                split=split,
                actions_by_step=actions,
                stimulus_onset_step=min(actions),
                required_event_kinds=("somatic_spike", "axonal_spike"),
                event_probe_label="soma",
                event_probe_segment_id=int(self.audit.representatives["soma"]),
            )

        plateau = self.selected_protocols["tuft_nmda_plateau"]
        calcium = self.selected_protocols["paired_bap_calcium_spike"]
        plateau_actions = actions_from_selected_protocol(plateau)
        calcium_actions = actions_from_selected_protocol(calcium)
        positive_splits = {
            310002: "train",
            310003: "validation",
        }

        # C. Exact confirmed plateau schedules on all three validated seeds.
        for seed in sorted(positive_splits):
            split = positive_splits[seed]
            self._register(
                plans,
                trajectory_id=f"{split}-{PLATEAU_PROTOCOL_ID}-seed{seed}",
                category="dendritic_events",
                protocol=PLATEAU_PROTOCOL_ID,
                protocol_id=PLATEAU_PROTOCOL_ID,
                protocol_variant="confirmed_positive",
                seed=seed,
                duration_ms=dendritic_duration_ms,
                split=split,
                actions_by_step=plateau_actions,
                stimulus_onset_step=int(plateau["burst_start_ms"]),
                required_event_kinds=("nmda_plateau",),
                event_probe_label="tuft_cluster_center",
                event_probe_segment_id=int(plateau["event_probe_segment_id"]),
            )
        plateau_ids = list(map(int, plateau["selected_synapse_ids"]))
        self._register(
            plans,
            trajectory_id="event-boundary-plateau-fewer-n8-seed311001",
            category="dendritic_events",
            protocol="plateau_negative_fewer_synapses_n8",
            protocol_id="plateau_negative_fewer_synapses_n8",
            protocol_variant="near_threshold_fewer_synapses",
            seed=311001,
            duration_ms=dendritic_duration_ms,
            split="event_boundary_test",
            actions_by_step=filter_synaptic_actions(
                plateau_actions, plateau_ids[:8]
            ),
            stimulus_onset_step=int(plateau["burst_start_ms"]),
            negative_control=True,
            event_probe_label="tuft_cluster_center",
            event_probe_segment_id=int(plateau["event_probe_segment_id"]),
            control_of=PLATEAU_PROTOCOL_ID,
        )
        wider_candidate = replace(
            candidate_from_selected_protocol(plateau), event_window_ms=0.8
        )
        wider_actions = build_candidate_actions(
            wider_candidate,
            plateau_ids,
            duration_ms=dendritic_duration_ms,
        )
        self._register(
            plans,
            trajectory_id="validation-plateau-wider-window-seed311002",
            category="dendritic_events",
            protocol="plateau_negative_wider_window_w800",
            protocol_id="plateau_negative_wider_window_w800",
            protocol_variant="near_threshold_timing",
            seed=311002,
            duration_ms=dendritic_duration_ms,
            split="validation",
            actions_by_step=wider_actions,
            stimulus_onset_step=int(plateau["burst_start_ms"]),
            negative_control=True,
            event_probe_label="tuft_cluster_center",
            event_probe_segment_id=int(plateau["event_probe_segment_id"]),
            control_of=PLATEAU_PROTOCOL_ID,
        )
        alternate_candidate = replace(
            candidate_from_selected_protocol(plateau), synapse_count=8
        )
        alternate_actions = build_candidate_actions(
            alternate_candidate,
            self.alternate_tuft_selection["synapse_ids"],
            duration_ms=dendritic_duration_ms,
        )
        self._register(
            plans,
            trajectory_id="event-boundary-plateau-alternate-branch-seed311003",
            category="dendritic_events",
            protocol="plateau_negative_alternate_branch_n8",
            protocol_id="plateau_negative_alternate_branch_n8",
            protocol_variant="near_threshold_alternate_branch",
            seed=311003,
            duration_ms=dendritic_duration_ms,
            split="event_boundary_test",
            actions_by_step=alternate_actions,
            stimulus_onset_step=int(plateau["burst_start_ms"]),
            negative_control=True,
            event_probe_label="tuft_alternate_cluster_center",
            event_probe_segment_id=int(
                self.alternate_tuft_selection["center_segment_id"]
            ),
            control_of=PLATEAU_PROTOCOL_ID,
        )

        # D. Exact BAC-like Ca schedules plus unpaired/fewer-synapse controls.
        for seed in sorted(positive_splits):
            split = positive_splits[seed]
            self._register(
                plans,
                trajectory_id=f"{split}-{CALCIUM_PROTOCOL_ID}-seed{seed}",
                category="dendritic_events",
                protocol=CALCIUM_PROTOCOL_ID,
                protocol_id=CALCIUM_PROTOCOL_ID,
                protocol_variant="confirmed_positive",
                seed=seed,
                duration_ms=dendritic_duration_ms,
                split=split,
                actions_by_step=calcium_actions,
                stimulus_onset_step=int(calcium["burst_start_ms"]),
                required_event_kinds=("calcium_spike",),
                event_probe_label="hot_zone",
                event_probe_segment_id=int(calcium["event_probe_segment_id"]),
            )
        calcium_ids = list(map(int, calcium["selected_synapse_ids"]))
        self._register(
            plans,
            trajectory_id="event-boundary-ca-unpaired-seed312001",
            category="dendritic_events",
            protocol="calcium_negative_unpaired",
            protocol_id="calcium_negative_unpaired",
            protocol_variant="near_threshold_unpaired",
            seed=312001,
            duration_ms=dendritic_duration_ms,
            split="event_boundary_test",
            actions_by_step=filter_synaptic_actions(
                calcium_actions, calcium_ids, keep_somatic_current=False
            ),
            stimulus_onset_step=int(calcium["burst_start_ms"]),
            negative_control=True,
            event_probe_label="hot_zone",
            event_probe_segment_id=int(calcium["event_probe_segment_id"]),
            control_of=CALCIUM_PROTOCOL_ID,
        )
        self._register(
            plans,
            trajectory_id="validation-ca-fewer-n8-seed312002",
            category="dendritic_events",
            protocol="calcium_negative_fewer_synapses_n8",
            protocol_id="calcium_negative_fewer_synapses_n8",
            protocol_variant="near_threshold_fewer_synapses",
            seed=312002,
            duration_ms=dendritic_duration_ms,
            split="validation",
            actions_by_step=filter_synaptic_actions(
                calcium_actions, calcium_ids[:8], keep_somatic_current=True
            ),
            stimulus_onset_step=int(calcium["burst_start_ms"]),
            negative_control=True,
            event_probe_label="hot_zone",
            event_probe_segment_id=int(calcium["event_probe_segment_id"]),
            control_of=CALCIUM_PROTOCOL_ID,
        )

        # E. Five futures share equilibrium state and Random123 key.
        branch_seed = 310001
        branch_specs = [
            ("no_input", dendritic_duration_ms, {}, (), None, None),
            (
                "subthreshold",
                dendritic_duration_ms,
                {3: (current(0.05),)},
                (),
                "soma",
                int(self.audit.representatives["soma"]),
            ),
            (
                "somatic_spike",
                dendritic_duration_ms,
                {3: (current(single_spike_current),)},
                ("somatic_spike",),
                "soma",
                int(self.audit.representatives["soma"]),
            ),
            (
                "nmda_plateau",
                dendritic_duration_ms,
                plateau_actions,
                ("nmda_plateau",),
                "tuft_cluster_center",
                int(plateau["event_probe_segment_id"]),
            ),
            (
                "calcium_spike",
                dendritic_duration_ms,
                calcium_actions,
                ("calcium_spike",),
                "hot_zone",
                int(calcium["event_probe_segment_id"]),
            ),
        ]
        for future, duration, actions, required, label, segment_id in branch_specs:
            protocol_id = {
                "nmda_plateau": PLATEAU_PROTOCOL_ID,
                "calcium_spike": CALCIUM_PROTOCOL_ID,
            }.get(future, f"branch_{future}")
            self._register(
                plans,
                trajectory_id=f"branching-{future}-seed{branch_seed}",
                category="branching",
                protocol=f"branch_future_{future}",
                protocol_id=protocol_id,
                protocol_variant="shared_snapshot_future",
                seed=branch_seed,
                duration_ms=duration,
                split="branching_test",
                actions_by_step=actions,
                stimulus_onset_step=(3 if actions else 0),
                required_event_kinds=required,
                event_probe_label=label,
                event_probe_segment_id=segment_id,
                branch_future=future,
            )

        validate_split_isolation(self.protocol_rows)
        return plans

    @staticmethod
    def _safe_read(owner: Any, name: str) -> float:
        return float(getattr(owner, name)) if hasattr(owner, name) else 0.0

    def _read_protocol_micro_observables(self) -> Sequence[float]:
        trajectory = self._active_trajectory
        contract = (
            self.protocol_registry.get(trajectory.trajectory_id, {})
            if trajectory is not None
            else {}
        )
        segment_id = contract.get("event_probe_segment_id")
        if segment_id is None:
            segment_id = int(self.audit.representatives["hot_zone"])
        segment = self.audit.live_segments[int(segment_id)]
        synapse_ids = list(map(int, contract.get("selected_synapse_ids", ())))

        def synapse_sum(name: str) -> float:
            return float(
                sum(
                    self._safe_read(
                        self.audit.synapse_records[synapse_id]["point_process"],
                        name,
                    )
                    for synapse_id in synapse_ids
                )
            )

        return [
            self._safe_read(segment, "cai"),
            self._safe_read(segment, "ica"),
            self._safe_read(segment, "ica_Ca_HVA"),
            self._safe_read(segment, "ica_Ca_LVAst"),
            synapse_sum("g_NMDA"),
            synapse_sum("i_NMDA"),
            synapse_sum("g_AMPA"),
            synapse_sum("i_AMPA"),
        ]

    def _run_trajectory_prefix_in_memory(
        self,
        trajectory: ProtocolTrajectory,
        duration_ms: int,
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        """Run the real storage transition path without writing the HDF5 file."""

        duration = min(int(duration_ms), int(trajectory.duration_ms))
        if duration <= 0:
            raise ValueError("preflight duration must be positive")
        equilibrium_rng = json.loads(
            self.equilibrium_rng_path.read_text(encoding="utf-8")
        )
        self._restore_native_snapshot(
            self.equilibrium_snapshot_path,
            equilibrium_rng["sequences"],
            equilibrium_rng.get("random123_seed", self.seed),
        )
        self._rekey_rngs(trajectory.seed)
        times: List[float] = []
        traces = {label: [] for label in self.audit.representatives}
        traces["voltage_event_probe_mv"] = []
        traces.update({label: [] for label in self.micro_observable_ids})
        event_probe_segment_id = trajectory.metadata.get(
            "event_probe_segment_id"
        )
        self._active_trajectory = trajectory
        preflight_snapshot = self.output_dir / "_preflight_checkpoint.neuron.bin"
        try:
            for step_index in range(duration):
                transition = self._run_transition(
                    -1,
                    trajectory,
                    step_index,
                    list(trajectory.actions_by_step.get(step_index, ())),
                    snapshot_path=(
                        preflight_snapshot if step_index == 0 else None
                    ),
                )
                keep = slice(None) if step_index == 0 else slice(1, None)
                times.extend(transition["absolute_time_ms"][keep].tolist())
                for probe_index, label in enumerate(self.audit.representatives):
                    traces[label].extend(
                        transition["micro_probe_voltage"][keep, probe_index].tolist()
                    )
                if event_probe_segment_id is not None:
                    traces["voltage_event_probe_mv"].extend(
                        transition["micro_all_voltage"][
                            keep, int(event_probe_segment_id)
                        ].tolist()
                    )
                for observable_index, label in enumerate(
                    self.micro_observable_ids
                ):
                    traces[label].extend(
                        transition["micro_protocol_observables"][
                            keep, observable_index
                        ].tolist()
                    )
        finally:
            self._active_trajectory = None
            preflight_snapshot.unlink(missing_ok=True)
        return times, traces

    def _compare_reference_prefix(
        self,
        trajectory: ProtocolTrajectory,
        time_ms: Sequence[float],
        traces: Mapping[str, Sequence[float]],
        duration_ms: float,
        *,
        reference_path: Optional[Path] = None,
        reference_kind: str = "historical_01b",
        tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        key = (str(trajectory.protocol_id), int(trajectory.seed))
        reference_path = reference_path or self.reference_trace_by_protocol_seed.get(
            key
        )
        if reference_path is None:
            raise KeyError(f"no 01b reference trace for {key}")
        with self.np.load(reference_path) as reference:
            sample_count = int(round(float(duration_ms) / 0.025)) + 1
            errors = {}
            new_time = self.np.asarray(time_ms[:sample_count], dtype=float)
            old_time = self.np.asarray(
                reference["time_ms"][:sample_count], dtype=float
            )
            errors["time_ms"] = (
                1.0e300
                if new_time.shape != old_time.shape
                else float(self.np.max(self.np.abs(new_time - old_time)))
            )
            for label in self.audit.representatives:
                reference_key = f"voltage_{label}_mv"
                if reference_key not in reference:
                    continue
                current = self.np.asarray(
                    traces[label][:sample_count], dtype=float
                )
                expected = self.np.asarray(
                    reference[reference_key][:sample_count], dtype=float
                )
                errors[reference_key] = (
                    1.0e300
                    if current.shape != expected.shape
                    else float(self.np.max(self.np.abs(current - expected)))
                )
            if (
                "voltage_event_probe_mv" in reference
                and "voltage_event_probe_mv" in traces
            ):
                current = self.np.asarray(
                    traces["voltage_event_probe_mv"][:sample_count],
                    dtype=float,
                )
                expected = self.np.asarray(
                    reference["voltage_event_probe_mv"][:sample_count],
                    dtype=float,
                )
                errors["voltage_event_probe_mv"] = (
                    1.0e300
                    if current.shape != expected.shape
                    else float(self.np.max(self.np.abs(current - expected)))
                )
            observable_reference_keys = {
                "cai_event_probe_mM": "cai_event_probe_mM",
                "ica_event_probe_mA_per_cm2": "ica_event_probe_mA_per_cm2",
                "ica_hva_event_probe_mA_per_cm2": (
                    "ica_hva_event_probe_mA_per_cm2"
                ),
                "ica_lva_event_probe_mA_per_cm2": (
                    "ica_lva_event_probe_mA_per_cm2"
                ),
                "sum_g_NMDA_uS": "sum_g_NMDA",
                "sum_i_NMDA_nA": "sum_i_NMDA",
                "sum_g_AMPA_uS": "sum_g_AMPA",
                "sum_i_AMPA_nA": "sum_i_AMPA",
            }
            for label, reference_key in observable_reference_keys.items():
                if reference_key not in reference:
                    continue
                current = self.np.asarray(
                    traces[label][:sample_count], dtype=float
                )
                expected = self.np.asarray(
                    reference[reference_key][:sample_count], dtype=float
                )
                errors[reference_key] = (
                    1.0e300
                    if current.shape != expected.shape
                    else float(self.np.max(self.np.abs(current - expected)))
                )
        maximum = max(errors.values(), default=1.0e300)
        return {
            "trajectory_id": trajectory.trajectory_id,
            "protocol_id": trajectory.protocol_id,
            "seed": trajectory.seed,
            "duration_ms": float(duration_ms),
            "reference_kind": str(reference_kind),
            "tolerance": float(tolerance),
            "maximum_absolute_error": maximum,
            "per_variable_max_absolute_error": errors,
            "valid": maximum <= float(tolerance),
            "reference_trace": (
                str(reference_path.relative_to(self.calibration_root))
                if self.calibration_root is not None
                and reference_path.is_relative_to(self.calibration_root)
                else str(reference_path.relative_to(self.output_dir))
            ),
        }

    def run_v1_preflight(
        self,
        protocols: Sequence[ProtocolTrajectory],
        *,
        prefix_duration_ms: int = 6,
    ) -> Dict[str, Any]:
        """Fail fast on driver, branching, and single-spike invariants."""

        self._require_equilibrium()
        protocols = list(protocols)
        prefix_protocols = [
            row
            for row in protocols
            if (str(row.protocol_id), int(row.seed))
            in self.reference_trace_by_protocol_seed
        ]
        if len(prefix_protocols) != 6:
            raise RuntimeError(
                f"preflight expected 6 confirmed prefixes, found {len(prefix_protocols)}"
            )
        runtime_root = self.output_dir / "preflight_runtime_references"
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        calibrator = DendriticProtocolCalibrator(
            self,
            output_dir=runtime_root,
            sample_interval_ms=0.025,
        )
        self.runtime_reference_trace_by_protocol_seed = {}
        self.preflight_prefix_duration_ms = int(prefix_duration_ms)
        prefix_rows = []
        historical_rows = []
        for trajectory in prefix_protocols:
            selected = next(
                row
                for row in self.selected_protocols.values()
                if str(row["candidate_id"]) == str(trajectory.protocol_id)
            )
            trial = calibrator.run_trial(
                candidate_from_selected_protocol(selected),
                int(trajectory.seed),
                int(prefix_duration_ms),
                trace_directory=calibrator.traces_dir,
            )
            runtime_reference = runtime_root / str(trial["trace_path"])
            key = (str(trajectory.protocol_id), int(trajectory.seed))
            self.runtime_reference_trace_by_protocol_seed[key] = runtime_reference
            times, traces = self._run_trajectory_prefix_in_memory(
                trajectory, prefix_duration_ms
            )
            runtime_comparison = self._compare_reference_prefix(
                trajectory,
                times,
                traces,
                float(prefix_duration_ms),
                reference_path=runtime_reference,
                reference_kind="corrected_runtime_01b_vs_storage",
                tolerance=0.0,
            )
            expected_schedule = {
                str(step): [action.to_dict() for action in actions]
                for step, actions in sorted(trajectory.actions_by_step.items())
            }
            schedule_match = (
                canonical_json_sha256({"schedule": expected_schedule})
                == canonical_json_sha256({"schedule": trial["input_schedule"]})
            )
            runtime_comparison["input_schedule_exact"] = schedule_match
            runtime_comparison["valid"] = bool(
                runtime_comparison["valid"] and schedule_match
            )
            prefix_rows.append(runtime_comparison)
            historical_rows.append(
                self._compare_reference_prefix(
                    trajectory,
                    times,
                    traces,
                    float(prefix_duration_ms),
                    reference_kind="historical_01b_before_assigned_state_fix",
                    tolerance=0.0,
                )
            )

        single = next(
            row for row in protocols if row.protocol_id == "somatic_single_spike"
        )
        single_times, single_traces = self._run_trajectory_prefix_in_memory(
            single, single.duration_ms
        )
        single_events = extract_events(
            single_times,
            {
                label: single_traces[label]
                for label in self.audit.representatives
            },
            self.event_definitions,
        )
        single_kinds = sorted({str(row["kind"]) for row in single_events})
        single_valid = {"somatic_spike", "axonal_spike"}.issubset(single_kinds)

        branch_protocols = [
            row for row in protocols if row.split == "branching_test"
        ]
        equilibrium_rng = json.loads(
            self.equilibrium_rng_path.read_text(encoding="utf-8")
        )
        branch_states = []
        for trajectory in branch_protocols:
            self._restore_native_snapshot(
                self.equilibrium_snapshot_path,
                equilibrium_rng["sequences"],
                equilibrium_rng.get("random123_seed", self.seed),
            )
            self._rekey_rngs(trajectory.seed)
            state = self.capture_boundary_state()
            branch_states.append(
                self.np.concatenate(
                    [
                        *(state[category] for category in self.state_variables),
                        self.np.asarray(
                            self.audit._snapshot_rng_sequences(), dtype=float
                        ),
                    ]
                )
            )
        branch_error = max(
            (
                float(self.np.max(self.np.abs(row - branch_states[0])))
                for row in branch_states[1:]
            ),
            default=0.0,
        )
        report = {
            "schema_version": DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION,
            "protocol_plan_sha256": self._protocol_plan_sha256(protocols),
            "valid": (
                all(row["valid"] for row in prefix_rows)
                and single_valid
                and branch_error == 0.0
            ),
            "prefix_duration_ms": int(prefix_duration_ms),
            "confirmed_prefixes": {
                "valid": all(row["valid"] for row in prefix_rows),
                "comparison_count": len(prefix_rows),
                "maximum_error": max(
                    (row["maximum_absolute_error"] for row in prefix_rows),
                    default=1.0e300,
                ),
                "comparisons": prefix_rows,
            },
            "historical_01b_trace_drift": {
                "gating": False,
                "reason": (
                    "The archived 01b traces predate the post-restore fcurrent() "
                    "fix. They remain hash-verified provenance, but exact runtime "
                    "identity is tested against freshly regenerated 01b traces "
                    "using the corrected canonical driver."
                ),
                "all_historical_traces_still_exact": all(
                    row["valid"] for row in historical_rows
                ),
                "maximum_error": max(
                    (
                        row["maximum_absolute_error"]
                        for row in historical_rows
                    ),
                    default=1.0e300,
                ),
                "comparisons": historical_rows,
            },
            "single_spike": {
                "valid": single_valid,
                "trajectory_id": single.trajectory_id,
                "calibrated_amplitude_na": (
                    self.calibrated_somatic_single_spike_current_na
                ),
                "observed_event_kinds": single_kinds,
            },
            "branching_initial_state": {
                "valid": branch_error == 0.0,
                "future_count": len(branch_states),
                "maximum_error": branch_error,
            },
        }
        self.preflight_report = report
        write_json(self.output_dir / "preflight_report.json", report)
        return report

    def _on_trajectory_complete(
        self,
        trajectory: ProtocolTrajectory,
        time_ms: Sequence[float],
        traces: Mapping[str, Sequence[float]],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        if (
            str(trajectory.protocol_id), int(trajectory.seed)
        ) not in self.reference_trace_by_protocol_seed:
            return
        self.prefix_overlap_rows.append(
            self._compare_reference_prefix(
                trajectory,
                time_ms,
                traces,
                float(self.preflight_prefix_duration_ms),
                reference_path=self.runtime_reference_trace_by_protocol_seed.get(
                    (str(trajectory.protocol_id), int(trajectory.seed))
                ),
                reference_kind="preflight_corrected_runtime",
                tolerance=0.0,
            )
        )

    @staticmethod
    def _decode(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    @staticmethod
    def _parquet_safe_rows(
        rows: Iterable[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        safe = []
        for row in rows:
            converted = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    converted[key] = json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    )
                else:
                    converted[key] = value
            safe.append(converted)
        return safe

    def _write_v1_tables(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("diagnostic v1 tables require h5py") from error

        transition_rows = []
        event_rows = []
        with h5py.File(self.transition_path, "r") as handle:
            count = int(handle.attrs["transition_count"])
            for index in range(count):
                trajectory_id = self._decode(
                    handle["metadata/trajectory_id"][index]
                )
                contract = self.protocol_registry[trajectory_id]
                labels = json.loads(handle["events/labels_json"][index])
                event_flags = sorted({str(row["kind"]) for row in labels})
                actions = json.loads(
                    handle["inputs/ordered_actions_json"][index]
                )
                transition_rows.append(
                    {
                        "transition_id": int(
                            handle["metadata/transition_id"][index]
                        ),
                        "trajectory_id": trajectory_id,
                        "protocol_id": self._decode(
                            handle["metadata/protocol_id"][index]
                        ),
                        "protocol_variant": self._decode(
                            handle["metadata/protocol_variant"][index]
                        ),
                        "seed": int(handle["metadata/seed"][index]),
                        "absolute_time_ms": float(
                            handle["metadata/start_time_ms"][index]
                        ),
                        "stimulus_relative_time_ms": float(
                            handle["metadata/stimulus_relative_time_ms"][index]
                        ),
                        "split": self._decode(
                            handle["metadata/split"][index]
                        ),
                        "teacher_commit": PINNED_TEACHER_COMMIT,
                        "teacher_manifest_sha256": sha256_file(
                            self.output_dir / "manifest.json"
                        ),
                        "snapshot_source": self._decode(
                            handle["metadata/snapshot_source"][index]
                        ),
                        "native_snapshot_ref": self._decode(
                            handle["metadata/native_snapshot_ref"][index]
                        ),
                        "snapshot_step_index": int(
                            handle["metadata/snapshot_step_index"][index]
                        ),
                        "event_flags": event_flags,
                        "action_count": len(actions),
                        "has_full_microtrace": True,
                        "microtrace_mode": self._decode(
                            handle["metadata/microtrace_mode"][index]
                        ),
                        "negative_control": bool(
                            handle["metadata/negative_control"][index]
                        ),
                        "event_probe_segment_id": contract[
                            "event_probe_segment_id"
                        ],
                    }
                )
                for event_index, event in enumerate(labels):
                    event_rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "transition_id": index,
                            "trajectory_event_id": event.get(
                                "trajectory_event_id", event_index
                            ),
                            "protocol_id": contract["protocol_id"],
                            "seed": contract["seed"],
                            "split": contract["split"],
                            **event,
                        }
                    )

        self.pd.DataFrame(
            self._parquet_safe_rows(transition_rows)
        ).to_parquet(self.output_dir / "transition_index.parquet", index=False)
        self.pd.DataFrame(
            self._parquet_safe_rows(self.protocol_rows)
        ).to_parquet(self.output_dir / "protocols.parquet", index=False)
        self.pd.DataFrame(self._parquet_safe_rows(event_rows)).to_parquet(
            self.output_dir / "events.parquet", index=False
        )
        branching_rows = [
            row for row in self.protocol_rows if row["split"] == "branching_test"
        ]
        self.pd.DataFrame(
            self._parquet_safe_rows(branching_rows)
        ).to_parquet(self.output_dir / "branching_index.parquet", index=False)

        split_payload = {
            split: sorted(
                row["trajectory_id"]
                for row in self.protocol_rows
                if row["split"] == split
            )
            for split in sorted(REQUIRED_SPLITS)
        }
        write_json(self.output_dir / "splits.json", split_payload)

        variable_table = []
        for name, category in self.state_schema["categories"].items():
            width = int(category["width"])
            variable_table.append(
                {
                    "group": name,
                    "dimension": width,
                    "dtype": "float64",
                    "storage_frequency": "S_t and S_t_plus_1 every 1 ms",
                    "bytes_per_transition": 2 * width * 8,
                    "role": (
                        "privileged"
                        if name == "currents_conductances"
                        else "core_state"
                    ),
                }
            )
        rng_width = int(self.state_schema["rng_state"]["width"])
        variable_table.append(
            {
                "group": "rng_state",
                "dimension": rng_width,
                "dtype": "float64",
                "storage_frequency": "t and t_plus_1 every 1 ms",
                "bytes_per_transition": 2 * rng_width * 8,
                "role": "core_state",
            }
        )
        storage_report = {
            "schema_version": DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION,
            "variable_storage_table": variable_table,
            "core_state_width": self.state_schema["core_state_width"],
            "privileged_state_width": self.state_schema[
                "privileged_state_width"
            ],
            "microtrace_sample_interval_ms": 0.025,
            "microtrace_sample_count_per_transition": 41,
            "all_segment_voltage_microtrace": True,
            "protocol_microtrace_observables": list(
                self.micro_observable_ids
            ),
            "size_estimate": self.dataset_manifest["size_estimate"],
        }
        write_json(self.output_dir / "storage_report.json", storage_report)
        write_json(
            self.output_dir / "prefix_overlap_report.json",
            {
                "required_error": 0.0,
                "expected_comparison_count": 6,
                "comparisons": self.prefix_overlap_rows,
                "valid": len(self.prefix_overlap_rows) == 6
                and all(row["valid"] for row in self.prefix_overlap_rows),
            },
        )
        return {
            "transition_count": len(transition_rows),
            "event_count": len(event_rows),
            "protocol_count": len(protocols),
            "branching_future_count": len(branching_rows),
        }

    def generate_dataset(
        self, protocols: Optional[Sequence[ProtocolTrajectory]] = None
    ) -> Dict[str, Any]:
        protocols = list(protocols or self.build_v1_protocols())
        if not self.preflight_report.get("valid"):
            raise RuntimeError(
                "run_v1_preflight(protocols) must pass before the expensive "
                "dataset generation phase"
            )
        if self.preflight_report.get(
            "protocol_plan_sha256"
        ) != self._protocol_plan_sha256(protocols):
            raise RuntimeError(
                "protocol plan changed after preflight; rebuild the plan and "
                "rerun run_v1_preflight before generation"
            )
        self.prefix_overlap_rows = []
        provenance_dir = self.output_dir / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.dataset_config_path,
            provenance_dir / "diagnostic_dataset_v1.yml",
        )
        for name in (
            "selected_dendritic_protocols.json",
            "confirmation_report.json",
            "artifact_index.json",
        ):
            shutil.copy2(self.calibration_root / name, provenance_dir / name)
        manifest = super().generate_dataset(protocols)
        table_report = self._write_v1_tables(protocols)
        manifest.update(
            {
                "schema_version": DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION,
                "repository_commit": git_commit(self.elm_repo),
                "teacher_manifest_sha256": sha256_file(
                    self.output_dir / "manifest.json"
                ),
                "dataset_configuration": (
                    "provenance/diagnostic_dataset_v1.yml"
                ),
                "dataset_configuration_sha256": sha256_file(
                    self.dataset_config_path
                ),
                "calibration_source_notebook": (
                    "notebooks/01b_dendritic_protocol_calibration.ipynb"
                ),
                "calibration_provenance": {
                    "selected_protocols": (
                        "provenance/selected_dendritic_protocols.json"
                    ),
                    "confirmation_report": (
                        "provenance/confirmation_report.json"
                    ),
                    "artifact_index": "provenance/artifact_index.json",
                },
                "calibration_selected_protocols_sha256": sha256_file(
                    self.calibration_root / "selected_dendritic_protocols.json"
                ),
                "calibration_artifact_validation": {
                    key: value
                    for key, value in self.calibration_integrity.items()
                    if key != "root"
                },
                "confirmed_protocol_ids": list(CONFIRMED_PROTOCOL_IDS),
                "confirmed_seeds": list(CONFIRMED_SEEDS),
                "indices": {
                    "transitions": "transition_index.parquet",
                    "protocols": "protocols.parquet",
                    "events": "events.parquet",
                    "branching": "branching_index.parquet",
                    "splits": "splits.json",
                },
                "storage_report": "storage_report.json",
                "preflight_report": "preflight_report.json",
                "preflight_protocol_plan_sha256": self.preflight_report[
                    "protocol_plan_sha256"
                ],
                "somatic_single_spike_calibration": {
                    "report": "somatic_single_spike_current_calibration.json",
                    "selected_amplitude_na": (
                        self.calibrated_somatic_single_spike_current_na
                    ),
                    "selection_rule": (
                        self.somatic_single_spike_calibration_report.get(
                            "selection_rule"
                        )
                    ),
                },
                "prefix_overlap_report": "prefix_overlap_report.json",
                "table_report": table_report,
                "right_censoring_policy": {
                    "retain_transition": True,
                    "duration_and_offset_loss_eligible": False,
                    "eligibility_condition": "right_censored == false",
                },
            }
        )
        self.dataset_manifest = manifest
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        return manifest

    def _exhaustive_sequential_replay(self) -> Dict[str, Any]:
        """Replay every trajectory once, covering every transition exactly once."""

        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("exhaustive replay requires h5py") from error

        failures = []
        maximum_error = 0.0
        replayed = 0
        with h5py.File(self.transition_path, "r") as handle:
            transition_count = int(handle.attrs["transition_count"])
            replay_progress = _ConsoleProgress(
                "replay esaustivo", transition_count
            )
            trajectory_indices: Dict[str, List[int]] = defaultdict(list)
            for index in range(transition_count):
                trajectory_indices[
                    self._decode(handle["metadata/trajectory_id"][index])
                ].append(index)
            for trajectory_id, indices in sorted(trajectory_indices.items()):
                indices.sort(
                    key=lambda index: int(
                        handle["metadata/step_index"][index]
                    )
                )
                first = indices[0]
                snapshot = self.output_dir / self._decode(
                    handle["metadata/native_snapshot_ref"][first]
                )
                seed = int(handle["metadata/seed"][first])
                self._restore_native_snapshot(
                    snapshot, handle["rng_state/t"][first, :], seed
                )
                contract = self.protocol_registry[trajectory_id]
                trajectory = ProtocolTrajectory(
                    trajectory_id=trajectory_id,
                    category=self._decode(
                        handle["metadata/category"][first]
                    ),
                    protocol=self._decode(
                        handle["metadata/protocol"][first]
                    ),
                    seed=seed,
                    duration_ms=len(indices),
                    split=self._decode(handle["metadata/split"][first]),
                    protocol_id=contract["protocol_id"],
                    protocol_variant=contract["protocol_variant"],
                    stimulus_onset_step=int(
                        contract["stimulus_onset_step"]
                    ),
                    required_event_kinds=tuple(
                        contract["required_event_kinds"]
                    ),
                    negative_control=bool(contract["negative_control"]),
                    event_probe_label=contract["event_probe_label"],
                )
                self._active_trajectory = trajectory
                for index in indices:
                    actions = [
                        _input_action(row)
                        for row in json.loads(
                            handle["inputs/ordered_actions_json"][index]
                        )
                    ]
                    replay = self._run_transition(
                        index,
                        trajectory,
                        int(handle["metadata/step_index"][index]),
                        actions,
                        snapshot_path=None,
                    )
                    errors: Dict[str, float] = {}
                    for category in self.state_variables:
                        for replay_key, boundary in (
                            ("state_t", "t"),
                            ("state_t_plus_1", "t_plus_1"),
                        ):
                            errors[f"{category}.{replay_key}"] = float(
                                self.np.max(
                                    self.np.abs(
                                        replay[replay_key][category]
                                        - handle[
                                            f"states/{category}/{boundary}"
                                        ][index, :]
                                    )
                                )
                            )
                    for replay_key, boundary in (
                        ("rng_t", "t"),
                        ("rng_t_plus_1", "t_plus_1"),
                    ):
                        errors[f"rng.{replay_key}"] = float(
                            self.np.max(
                                self.np.abs(
                                    replay[replay_key]
                                    - handle[f"rng_state/{boundary}"][index, :]
                                )
                            )
                        )
                    errors["microtrace.probe_voltage"] = float(
                        self.np.max(
                            self.np.abs(
                                replay["micro_probe_voltage"]
                                - handle["microtraces/probe_voltage"][
                                    index, :, :
                                ]
                            )
                        )
                    )
                    if "protocol_observables" in handle["microtraces"]:
                        errors["microtrace.protocol_observables"] = float(
                            self.np.max(
                                self.np.abs(
                                    replay["micro_protocol_observables"]
                                    - handle[
                                        "microtraces/protocol_observables"
                                    ][index, :, :]
                                )
                            )
                        )
                    row_error = max(errors.values(), default=0.0)
                    maximum_error = max(maximum_error, row_error)
                    replayed += 1
                    replay_progress.update(
                        replayed,
                        detail=(
                            f"traiettoria {trajectory_id}; "
                            f"step {int(handle['metadata/step_index'][index]) + 1}/"
                            f"{len(indices)}; errore max={maximum_error:.3g}"
                        ),
                    )
                    if row_error > 1e-5:
                        failures.append(
                            {
                                "transition_id": index,
                                "trajectory_id": trajectory_id,
                                "maximum_error": row_error,
                                "errors_by_component": dict(
                                    sorted(errors.items())
                                ),
                            }
                        )
                replay_progress.update(
                    replayed,
                    detail=(
                        f"completata traiettoria {trajectory_id}; "
                        f"fallimenti={len(failures)}"
                    ),
                    force=True,
                )
                self._active_trajectory = None
        return {
            "valid": not failures,
            "replayed_transition_count": replayed,
            "failure_count": len(failures),
            "maximum_error": maximum_error,
            "tolerance": 1e-5,
            "failures": failures,
        }

    def _v1_contract_checks(self) -> Dict[str, Any]:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("diagnostic v1 checks require h5py") from error

        events_by_trajectory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        event_bounds_valid = True
        input_offsets_valid = True
        branching_initial_states = []
        branching_voltage_traces = {}
        with h5py.File(self.transition_path, "r") as handle:
            count = int(handle.attrs["transition_count"])
            trajectory_indices: Dict[str, List[int]] = defaultdict(list)
            for index in range(count):
                trajectory_id = self._decode(
                    handle["metadata/trajectory_id"][index]
                )
                trajectory_indices[trajectory_id].append(index)
                actions = json.loads(
                    handle["inputs/ordered_actions_json"][index]
                )
                if any(
                    not 0.0 <= float(action["offset_ms"]) < 1.0
                    for action in actions
                ):
                    input_offsets_valid = False
                events_by_trajectory[trajectory_id].extend(
                    json.loads(handle["events/labels_json"][index])
                )
            for trajectory_id, indices in trajectory_indices.items():
                indices.sort(
                    key=lambda index: int(
                        handle["metadata/step_index"][index]
                    )
                )
                start = float(handle["metadata/start_time_ms"][indices[0]])
                stop = start + len(indices)
                for event in events_by_trajectory[trajectory_id]:
                    if not (
                        start <= float(event["onset_ms"])
                        <= float(event["peak_ms"])
                        <= float(event["offset_ms"])
                        <= stop + 1e-9
                    ):
                        event_bounds_valid = False
                contract = self.protocol_registry[trajectory_id]
                if contract["split"] == "branching_test":
                    branching_initial_states.append(
                        {
                            **{
                                category: handle[f"states/{category}/t"][
                                    indices[0], :
                                ]
                                for category in self.state_variables
                            },
                            "rng_state": handle["rng_state/t"][indices[0], :],
                        }
                    )
                    trace_parts = []
                    for position, index in enumerate(indices):
                        values = handle[
                            "microtraces/all_segment_voltage"
                        ][index, :, :]
                        trace_parts.append(
                            values if position == 0 else values[1:, :]
                        )
                    branching_voltage_traces[
                        contract["branch_future"]
                    ] = self.np.concatenate(trace_parts, axis=0)

        required_failures = []
        required_censored = []
        for row in self.protocol_rows:
            if row["negative_control"] or not row["required_event_kinds"]:
                continue
            observed = events_by_trajectory[row["trajectory_id"]]
            for kind in row["required_event_kinds"]:
                matches = [
                    event
                    for event in observed
                    if event["kind"] == kind
                    and (
                        kind not in {"nmda_spike", "nmda_plateau", "calcium_spike"}
                        or int(event["segment_id"])
                        == int(row["event_probe_segment_id"])
                    )
                ]
                if not matches:
                    required_failures.append(
                        {"trajectory_id": row["trajectory_id"], "kind": kind}
                    )
                for event in matches:
                    if event.get("right_censored"):
                        required_censored.append(
                            {
                                "trajectory_id": row["trajectory_id"],
                                "kind": kind,
                            }
                        )

        negative_outcomes = []
        for row in self.protocol_rows:
            if not row["negative_control"]:
                continue
            control_kind = (
                "nmda_plateau"
                if row["control_of"] == PLATEAU_PROTOCOL_ID
                else "calcium_spike"
            )
            present = any(
                event["kind"] == control_kind
                and int(event["segment_id"])
                == int(row["event_probe_segment_id"])
                for event in events_by_trajectory[row["trajectory_id"]]
            )
            negative_outcomes.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "control_of": row["control_of"],
                    "target_event_kind": control_kind,
                    "target_event_present": present,
                }
            )
        negative_family_coverage = {
            protocol_id: any(
                row["control_of"] == protocol_id
                and not row["target_event_present"]
                for row in negative_outcomes
            )
            for protocol_id in CONFIRMED_PROTOCOL_IDS
        }

        initial_error = 0.0
        if branching_initial_states:
            reference = branching_initial_states[0]
            initial_error = max(
                float(
                    self.np.max(
                        self.np.abs(values[category] - reference[category])
                    )
                )
                for values in branching_initial_states[1:]
                for category in reference
            )
        control_trace = branching_voltage_traces.get("no_input")
        branching_divergence = {
            name: float(
                self.np.max(self.np.abs(values - control_trace))
            )
            for name, values in branching_voltage_traces.items()
            if name != "no_input"
        }
        observed_splits = {row["split"] for row in self.protocol_rows}
        seed_splits: Dict[int, set] = defaultdict(set)
        for row in self.protocol_rows:
            seed_splits[int(row["seed"])].add(str(row["split"]))
        leaking_seeds = {
            seed: sorted(splits)
            for seed, splits in seed_splits.items()
            if len(splits) > 1
        }
        core_width = int(self.state_schema["core_state_width"])
        privileged_width = int(self.state_schema["privileged_state_width"])
        overlap_valid = len(self.prefix_overlap_rows) == 6 and all(
            row["valid"] for row in self.prefix_overlap_rows
        )
        if not self.preflight_report and (
            self.output_dir / "preflight_report.json"
        ).is_file():
            self.preflight_report = json.loads(
                (self.output_dir / "preflight_report.json").read_text(
                    encoding="utf-8"
                )
            )
        preflight_valid = bool(self.preflight_report.get("valid")) and (
            self.preflight_report.get("protocol_plan_sha256")
            == self.dataset_manifest.get("preflight_protocol_plan_sha256")
        )
        return {
            "preflight_valid": preflight_valid,
            "required_events_present": not required_failures,
            "required_event_failures": required_failures,
            "required_events_not_right_censored": not required_censored,
            "right_censored_required_events": required_censored,
            "negative_control_outcomes": negative_outcomes,
            "negative_family_coverage": negative_family_coverage,
            "negative_controls_valid": all(
                negative_family_coverage.values()
            ),
            "event_bounds_valid": event_bounds_valid,
            "input_offsets_half_open_valid": input_offsets_valid,
            "required_splits": sorted(REQUIRED_SPLITS),
            "observed_splits": sorted(observed_splits),
            "split_set_valid": observed_splits == REQUIRED_SPLITS,
            "seed_split_isolation_valid": not leaking_seeds,
            "leaking_seeds": leaking_seeds,
            "segment_count": len(self.audit.live_segments),
            "segment_mapping_stable": len(self.audit.live_segments) == 642,
            "core_state_width": core_width,
            "core_state_width_stable": core_width == 17220,
            "privileged_state_width": privileged_width,
            "privileged_state_width_stable": privileged_width == 9182,
            "calibration_artifacts": {
                key: value
                for key, value in self.calibration_integrity.items()
                if key != "root"
            },
            "prefix_overlap": {
                "valid": overlap_valid,
                "comparison_count": len(self.prefix_overlap_rows),
                "expected_comparison_count": 6,
                "maximum_error": max(
                    (
                        row["maximum_absolute_error"]
                        for row in self.prefix_overlap_rows
                    ),
                    default=1.0e300,
                ),
            },
            "branching": {
                "future_count": len(branching_initial_states),
                "same_initial_state_max_error": initial_error,
                "same_initial_state": initial_error == 0.0,
                "maximum_voltage_divergence_vs_no_input_mv": (
                    branching_divergence
                ),
                "different_futures_diverge": bool(branching_divergence)
                and all(value > 1e-9 for value in branching_divergence.values()),
            },
        }

    def validate_dataset_v1(self, replay_count: int = 5) -> Dict[str, Any]:
        """Run legacy checks plus exhaustive replay and the v1 acceptance gate."""

        print(
            "[HayFlow][validazione] fase 1/4: controlli strutturali e replay "
            "campionati",
            flush=True,
        )
        try:
            base = super().validate_dataset(replay_count=replay_count)
        except RuntimeError:
            base = json.loads(
                (self.output_dir / "validation_report.json").read_text(
                    encoding="utf-8"
                )
            )
        print(
            f"[HayFlow][validazione] fase 1/4 completata: valid={base['valid']}",
            flush=True,
        )
        print(
            "[HayFlow][validazione] fase 2/4: replay esaustivo di tutte le "
            "transizioni",
            flush=True,
        )
        exhaustive = self._exhaustive_sequential_replay()
        print(
            "[HayFlow][validazione] fase 2/4 completata: "
            f"valid={exhaustive['valid']}, "
            f"errore max={exhaustive['maximum_error']:.3g}",
            flush=True,
        )
        print(
            "[HayFlow][validazione] fase 3/4: contratto v1, eventi, split, "
            "branching e hash 01b",
            flush=True,
        )
        contract = self._v1_contract_checks()
        blockers = []
        if not base["valid"]:
            blockers.append("base diagnostic validation failed")
        if not exhaustive["valid"]:
            blockers.append("one or more transitions failed exhaustive replay")
        for key in (
            "preflight_valid",
            "required_events_present",
            "required_events_not_right_censored",
            "negative_controls_valid",
            "event_bounds_valid",
            "input_offsets_half_open_valid",
            "split_set_valid",
            "seed_split_isolation_valid",
            "segment_mapping_stable",
            "core_state_width_stable",
            "privileged_state_width_stable",
        ):
            if not contract[key]:
                blockers.append(f"v1 contract failed: {key}")
        if not contract["calibration_artifacts"]["valid"]:
            blockers.append("01b artifact hashes are invalid")
        if not contract["prefix_overlap"]["valid"]:
            blockers.append("confirmed first-35-ms overlap is not exact")
        if not contract["branching"]["same_initial_state"]:
            blockers.append("branching futures do not share the same initial state")
        if not contract["branching"]["different_futures_diverge"]:
            blockers.append("branching futures do not diverge")
        report = {
            "schema_version": DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION,
            "valid": not blockers,
            "blockers": blockers,
            "base_validation": base,
            "exhaustive_transition_replay": exhaustive,
            "contract": contract,
        }
        write_json(self.output_dir / "validation_report.json", report)
        print(
            "[HayFlow][validazione] fase 4/4: figure diagnostiche e indice "
            "degli artefatti",
            flush=True,
        )
        self._plot_v1_figures()
        self._write_artifact_index()
        if blockers:
            raise RuntimeError(
                f"diagnostic dataset v1 validation failed: {blockers}"
            )
        print(
            "[HayFlow][validazione] completata: valid=True",
            flush=True,
        )
        return report

    def _plot_v1_figures(self) -> None:
        """Write unit-separated, stimulus-relative diagnostic figures."""

        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("diagnostic figures require h5py") from error

        probe_order = list(self.state_schema["probe_order"])
        observable_index = {
            name: index for index, name in enumerate(self.micro_observable_ids)
        }
        with h5py.File(self.transition_path, "r") as handle:
            count = int(handle.attrs["transition_count"])
            indices_by_trajectory: Dict[str, List[int]] = defaultdict(list)
            for index in range(count):
                indices_by_trajectory[
                    self._decode(handle["metadata/trajectory_id"][index])
                ].append(index)
            for protocol_id, filename in (
                (PLATEAU_PROTOCOL_ID, "confirmed_nmda_plateau.png"),
                (CALCIUM_PROTOCOL_ID, "confirmed_paired_calcium_spike.png"),
            ):
                rows = [
                    row
                    for row in self.protocol_rows
                    if row["protocol_id"] == protocol_id
                    and row["protocol_variant"] == "confirmed_positive"
                ]
                figure, axes = self.audit.plt.subplots(
                    5, 1, figsize=(12, 13), sharex=True
                )
                for row in sorted(rows, key=lambda item: item["seed"]):
                    indices = sorted(
                        indices_by_trajectory[row["trajectory_id"]],
                        key=lambda index: int(
                            handle["metadata/step_index"][index]
                        ),
                    )
                    time_parts = []
                    voltage_parts = []
                    observable_parts = []
                    probe_index = probe_order.index(row["event_probe_label"])
                    offsets = handle["microtraces/time_offsets_ms"][...]
                    for local_index, index in enumerate(indices):
                        keep = slice(None) if local_index == 0 else slice(1, None)
                        relative_boundary = float(
                            handle["metadata/stimulus_relative_time_ms"][index]
                        )
                        time_parts.append(relative_boundary + offsets[keep])
                        voltage_parts.append(
                            handle["microtraces/probe_voltage"][
                                index, keep, probe_index
                            ]
                        )
                        observable_parts.append(
                            handle["microtraces/protocol_observables"][
                                index, keep, :
                            ]
                        )
                    time = self.np.concatenate(time_parts)
                    voltage = self.np.concatenate(voltage_parts)
                    observable = self.np.concatenate(observable_parts, axis=0)
                    label = f"seed {row['seed']}"
                    axes[0].plot(time, voltage, label=label)
                    cai = observable[
                        :, observable_index["cai_event_probe_mM"]
                    ]
                    axes[1].plot(time, 1000.0 * (cai - cai[0]), label=label)
                    axes[2].plot(
                        time,
                        observable[
                            :, observable_index["ica_event_probe_mA_per_cm2"]
                        ],
                        label=label,
                    )
                    axes[3].plot(
                        time,
                        1000.0
                        * observable[:, observable_index["sum_g_NMDA_uS"]],
                        label=label,
                    )
                    axes[4].plot(
                        time,
                        observable[:, observable_index["sum_i_NMDA_nA"]],
                        label=label,
                    )
                    trajectory_events = []
                    for index in indices:
                        trajectory_events.extend(
                            json.loads(handle["events/labels_json"][index])
                        )
                    for event in trajectory_events:
                        if event["kind"] not in row["required_event_kinds"]:
                            continue
                        if int(event["segment_id"]) != int(
                            row["event_probe_segment_id"]
                        ):
                            continue
                        color = axes[0].lines[-1].get_color()
                        axes[0].axvline(
                            event["stimulus_relative_onset_ms"],
                            color=color,
                            alpha=0.25,
                            linestyle="--",
                        )
                        axes[0].axvline(
                            event["stimulus_relative_peak_ms"],
                            color=color,
                            alpha=0.25,
                            linestyle=":",
                        )
                        axes[0].axvline(
                            event["stimulus_relative_offset_ms"],
                            color=color,
                            alpha=0.25,
                            linestyle="-.",
                        )
                stimulus_left = -1.0 if protocol_id == CALCIUM_PROTOCOL_ID else 0.0
                stimulus_right = 3.0
                for axis in axes:
                    axis.axvspan(
                        stimulus_left,
                        stimulus_right,
                        color="grey",
                        alpha=0.10,
                        label="stimulus window" if axis is axes[0] else None,
                    )
                    axis.grid(alpha=0.2)
                axes[0].set_ylabel("probe voltage (mV)")
                axes[1].set_ylabel("delta Ca (uM)")
                axes[2].set_ylabel("Ca current (mA/cm2)")
                axes[3].set_ylabel("NMDA conductance (nS)")
                axes[4].set_ylabel("NMDA current (nA)")
                axes[4].set_xlabel("time relative to dendritic stimulus (ms)")
                axes[0].set_title(
                    f"{protocol_id}\nprobe segment {rows[0]['event_probe_segment_id']}"
                )
                for axis in axes:
                    axis.legend(ncol=4, fontsize=8)
                figure.tight_layout()
                figure.savefig(self.figures_dir / filename, dpi=180)
                self.audit.plt.close(figure)
