"""Read-only recoverability audit for sealed GIADA paper-scale corpora.

This module answers a narrower question than the production validator: which
signals needed by a surrogate-validity study are physically present in the
HDF5 shards?  It never changes the corpus and deliberately distinguishes
presynaptic input events from postsynaptic somatic spike labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


EVENT_FIELDS = (
    "transition_row",
    "synapse_id",
    "segment_id",
    "offset_ms",
    "release_success",
    "released_quantity",
    "ampa_state_increment",
    "nmda_state_increment",
    "inhibitory_state_increment",
)


def _component_roots(corpus: Path) -> Dict[str, Path]:
    root = Path(corpus).resolve()
    manifest_path = root / "composite_manifest.json"
    if not manifest_path.exists() and (root / "composite" / "composite_manifest.json").exists():
        manifest_path = root / "composite" / "composite_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            str(row["component_id"]): (manifest_path.parent / row["root"]).resolve()
            for row in manifest["components"]
        }
    if (root / "shards").is_dir():
        return {root.name: root}
    components = {
        name: root / name for name in ("background", "targeted")
        if (root / name / "shards").is_dir()
    }
    if not components:
        raise FileNotFoundError(f"no HDF5 component shards found below {root}")
    return components


def _select_evenly(paths: Sequence[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=np.int64)
    return [paths[int(index)] for index in sorted(set(indices.tolist()))]


def _decode_metadata(handle: Any) -> Mapping[str, Any]:
    raw = handle.attrs.get("schema_metadata_json", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _event_type_counts(events: Mapping[str, np.ndarray]) -> Dict[str, int]:
    tolerance = 1.0e-12
    ampa = np.abs(events["ampa_state_increment"]) > tolerance
    nmda = np.abs(events["nmda_state_increment"]) > tolerance
    inhibitory = np.abs(events["inhibitory_state_increment"]) > tolerance
    return {
        "ampa_nonzero": int(ampa.sum()),
        "nmda_nonzero": int(nmda.sum()),
        "inhibitory_nonzero": int(inhibitory.sum()),
        "all_increments_zero": int((~(ampa | nmda | inhibitory)).sum()),
    }


def audit_shard(path: Path) -> Dict[str, Any]:
    """Inspect one shard without modifying or hashing it."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("surrogate-validity audit requires h5py") from error

    blockers: list[str] = []
    warnings: list[str] = []
    source = Path(path).resolve()
    with h5py.File(source, "r") as handle:
        transition_count = int(handle.attrs.get("transition_count", -1))
        metadata = _decode_metadata(handle)
        required_rows = (
            "voltage_t_mv", "voltage_t_plus_1_mv", "voltage_min_mv",
            "voltage_max_mv", "trajectory_index", "step_index",
            "scheduled_event_count", "realized_event_count", "causal_drive",
            "mechanism_state_t", "ion_state_t", "segment_id",
            "high_resolution_sample_count",
        )
        for name in required_rows:
            if name not in handle:
                blockers.append(f"missing dataset {name}")
            elif handle[name].shape[0] != transition_count:
                blockers.append(f"row count mismatch in {name}")
        if blockers:
            return {
                "path": str(source), "valid": False, "blockers": blockers,
                "warnings": warnings, "transition_count": transition_count,
            }

        event_missing = [name for name in EVENT_FIELDS if f"events/{name}" not in handle]
        if event_missing:
            blockers.append(f"missing event datasets: {event_missing}")
            event_count = 0
            events = {name: np.empty(0) for name in EVENT_FIELDS}
        else:
            events = {name: np.asarray(handle[f"events/{name}"][...]) for name in EVENT_FIELDS}
            event_lengths = {name: len(values) for name, values in events.items()}
            if len(set(event_lengths.values())) != 1:
                blockers.append(f"event dataset length mismatch: {event_lengths}")
            event_count = min(event_lengths.values(), default=0)

        voltage_t = np.asarray(handle["voltage_t_mv"][:, 0], dtype=np.float64)
        voltage_t1 = np.asarray(handle["voltage_t_plus_1_mv"][:, 0], dtype=np.float64)
        voltage_max = np.asarray(handle["voltage_max_mv"][:, 0], dtype=np.float64)
        scheduled = np.asarray(handle["scheduled_event_count"][...], dtype=np.int64)
        realized = np.asarray(handle["realized_event_count"][...], dtype=np.int64)
        high_res = np.asarray(handle["high_resolution_sample_count"][...], dtype=np.int64)
        causal = np.asarray(handle["causal_drive"][:, 0, :], dtype=np.float64)
        trajectories = np.asarray(handle["trajectory_index"][...], dtype=np.int64)
        steps = np.asarray(handle["step_index"][...], dtype=np.int64)

        scheduled_total = int(scheduled.sum())
        if event_count != scheduled_total:
            blockers.append(
                f"event table has {event_count} rows but scheduled_event_count sums to "
                f"{scheduled_total}"
            )
        if event_count:
            rows = events["transition_row"].astype(np.int64, copy=False)
            offsets = events["offset_ms"].astype(np.float64, copy=False)
            success = events["release_success"].astype(bool, copy=False)
            released = events["released_quantity"].astype(np.float64, copy=False)
            if rows.min() < 0 or rows.max() >= transition_count:
                blockers.append("event transition reference out of range")
            if not np.all((offsets >= 0.0) & (offsets < 1.0)):
                blockers.append("event offset outside [0, 1 ms)")
            if np.any(success & (released <= 0.0)):
                blockers.append("successful release with non-positive released quantity")
            if np.any((~success) & (np.abs(released) > 1.0e-12)):
                blockers.append("failed release with non-zero released quantity")
            release_success_count = int(success.sum())
            release_failure_count = int((~success).sum())
            unique_synapses = int(np.unique(events["synapse_id"]).size)
            unique_event_segments = int(np.unique(events["segment_id"]).size)
        else:
            release_success_count = release_failure_count = 0
            unique_synapses = unique_event_segments = 0
            warnings.append("sampled shard contains no synaptic event rows")

        continuity_gaps = 0
        voltage_discontinuities = 0
        for trajectory in np.unique(trajectories):
            indices = np.flatnonzero(trajectories == trajectory)
            order = indices[np.argsort(steps[indices], kind="stable")]
            if len(order) < 2:
                continue
            continuity_gaps += int(np.count_nonzero(np.diff(steps[order]) != 1))
            voltage_discontinuities += int(np.count_nonzero(
                ~np.isclose(voltage_t1[order[:-1]], voltage_t[order[1:]], atol=1.0e-5, rtol=0.0)
            ))

        endpoint_upcross = (voltage_t < -55.0) & (voltage_t1 >= -55.0)
        hidden_peak = (voltage_t < -25.0) & (voltage_t1 < -25.0) & (voltage_max >= -25.0)
        selected_segments = np.unique(np.asarray(handle["segment_id"][...], dtype=np.int64))
        exact_output_spike_labels = (
            "output_spikes/offset_ms" in handle
            or "output_spike_time_ms" in handle
            or "events/output_spike_time_ms" in handle
        )
        if not exact_output_spike_labels:
            warnings.append("no explicit postsynaptic somatic spike-time dataset")
        output_spike_rows = 0
        output_spike_detector = None
        if "output_spikes/offset_ms" in handle:
            spike_fields = (
                "output_spikes/transition_row",
                "output_spikes/offset_ms",
                "output_spikes/peak_voltage_mv",
            )
            missing = [name for name in spike_fields if name not in handle]
            if missing:
                blockers.append(f"incomplete output spike table: {missing}")
            else:
                spike_lengths = {name: len(handle[name]) for name in spike_fields}
                if len(set(spike_lengths.values())) != 1:
                    blockers.append(f"output spike dataset length mismatch: {spike_lengths}")
                output_spike_rows = min(spike_lengths.values(), default=0)
                spike_transition_rows = np.asarray(
                    handle["output_spikes/transition_row"][...], dtype=np.int64
                )
                spike_offsets = np.asarray(
                    handle["output_spikes/offset_ms"][...], dtype=np.float64
                )
                if output_spike_rows and (
                    spike_transition_rows.min() < 0
                    or spike_transition_rows.max() >= transition_count
                ):
                    blockers.append("output spike transition reference out of range")
                if not np.all((spike_offsets >= 0.0) & (spike_offsets < 1.0)):
                    blockers.append("output spike offset outside [0, 1 ms)")
                if "output_spike_count" not in handle:
                    blockers.append("missing per-row output spike count")
                elif int(np.asarray(handle["output_spike_count"][...]).sum()) != output_spike_rows:
                    blockers.append("per-row output spike counts do not match spike table")
            output_spike_detector = str(handle.attrs.get("output_spike_detector", ""))

        return {
            "path": str(source),
            "valid": not blockers,
            "blockers": blockers,
            "warnings": warnings,
            "transition_count": transition_count,
            "storage_profile": metadata.get("storage_profile"),
            "selected_segment_ids": selected_segments.astype(int).tolist(),
            "mechanism_state_width": int(handle["mechanism_state_t"].shape[-1]),
            "ion_state_width": int(handle["ion_state_t"].shape[-1]),
            "causal_drive_width": int(causal.shape[-1]),
            "causal_drive_nonzero_rows": int(np.any(np.abs(causal) > 1.0e-12, axis=1).sum()),
            "scheduled_synaptic_events": scheduled_total,
            "realized_successful_events": int(realized.sum()),
            "event_table_rows": event_count,
            "release_success_count": release_success_count,
            "release_failure_count": release_failure_count,
            "unique_synapse_ids": unique_synapses,
            "unique_event_segment_ids": unique_event_segments,
            "event_type_counts": _event_type_counts(events),
            "trajectory_count": int(np.unique(trajectories).size),
            "step_continuity_gaps": continuity_gaps,
            "voltage_boundary_discontinuities": voltage_discontinuities,
            "high_resolution_sample_count_histogram": {
                str(int(value)): int(count)
                for value, count in zip(*np.unique(high_res, return_counts=True))
            },
            "endpoint_minus55_upcrossings": int(endpoint_upcross.sum()),
            "hidden_above_minus25_excursions_detectable_from_max": int(hidden_peak.sum()),
            "explicit_output_spike_times_present": exact_output_spike_labels,
            "explicit_output_spike_rows": output_spike_rows,
            "output_spike_detector": output_spike_detector,
        }


def audit_corpus(
    corpus: Path,
    *,
    sample_shards_per_component: int = 4,
    progress=None,
) -> Dict[str, Any]:
    """Audit deterministic, evenly spaced shard samples from each component."""

    components: Dict[str, Any] = {}
    overall_blockers: list[str] = []
    roots = _component_roots(Path(corpus))
    for component, root in sorted(roots.items()):
        available = sorted((root / "shards").glob("*.h5"))
        selected = _select_evenly(available, int(sample_shards_per_component))
        if not available:
            overall_blockers.append(f"{component}: no shards found")
        reports = []
        for index, path in enumerate(selected, 1):
            if progress is not None:
                progress(component, index, len(selected), path)
            report = audit_shard(path)
            reports.append(report)
            overall_blockers.extend(
                f"{component}/{path.name}: {message}" for message in report["blockers"]
            )
        totals = Counter()
        for report in reports:
            for key in (
                "transition_count", "scheduled_synaptic_events",
                "realized_successful_events", "event_table_rows",
                "release_success_count", "release_failure_count",
                "causal_drive_nonzero_rows", "endpoint_minus55_upcrossings",
                "hidden_above_minus25_excursions_detectable_from_max",
                "explicit_output_spike_rows",
                "step_continuity_gaps", "voltage_boundary_discontinuities",
            ):
                totals[key] += int(report.get(key, 0))
        components[component] = {
            "root": str(root),
            "available_shards": len(available),
            "sampled_shards": len(selected),
            "sampled_shard_names": [path.name for path in selected],
            "sample_totals": dict(totals),
            "explicit_output_spike_times_present_in_all_sampled_shards": bool(reports) and all(
                row["explicit_output_spike_times_present"] for row in reports
            ),
            "shards": reports,
        }
    return {
        "schema_version": "giada-surrogate-validity-recoverability-audit-v1",
        "valid": not overall_blockers,
        "read_only": True,
        "corpus": str(Path(corpus).resolve()),
        "sample_shards_per_component": int(sample_shards_per_component),
        "blockers": overall_blockers,
        "components": components,
        "interpretation": {
            "events_are_presynaptic_inputs": True,
            "events_are_not_postsynaptic_spike_labels": True,
            "minus55_upcrossings_are_a_derived_1ms_definition": True,
            "original_neuronio_spike_times_recoverable_exactly": bool(components) and all(
                row["explicit_output_spike_times_present_in_all_sampled_shards"]
                for row in components.values()
            ),
        },
    }
