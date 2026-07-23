"""Chunked HDF5 storage for the small full-state diagnostic dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


class TransitionH5Writer:
    """Append one transition at a time without holding the dataset in RAM."""

    def __init__(
        self,
        path: Path,
        state_widths: Mapping[str, int],
        microtrace_samples: int,
        microtrace_variable_count: int,
        segment_count: int,
        probe_count: int,
        *,
        micro_observable_names: Sequence[str] = (),
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError as error:
            raise RuntimeError("HDF5 output requires h5py and numpy") from error

        self.h5py = h5py
        self.np = np
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")
        self.file.attrs["format"] = "hayflow_diagnostic_transitions"
        self.file.attrs["transition_count"] = 0
        self.file.attrs["micro_observable_names_json"] = json.dumps(
            list(micro_observable_names), separators=(",", ":")
        )
        self.count = 0
        options = {
            "compression": compression,
            "compression_opts": compression_level,
            "shuffle": True,
        }

        states = self.file.create_group("states")
        self.state_datasets: Dict[str, Dict[str, Any]] = {}
        for category, width in state_widths.items():
            if category == "rng_state":
                continue
            group = states.create_group(category)
            self.state_datasets[category] = {
                boundary: group.create_dataset(
                    boundary,
                    shape=(0, int(width)),
                    maxshape=(None, int(width)),
                    chunks=(1, max(1, int(width))),
                    dtype="f8",
                    **options,
                )
                for boundary in ("t", "t_plus_1")
            }

        rng = self.file.create_group("rng_state")
        rng_width = int(state_widths.get("rng_state", 0))
        self.rng_datasets = {
            boundary: rng.create_dataset(
                boundary,
                shape=(0, rng_width),
                maxshape=(None, rng_width),
                chunks=(1, max(1, rng_width)),
                dtype="f8",
                **options,
            )
            for boundary in ("t", "t_plus_1")
        }
        micro = self.file.create_group("microtraces")
        self.micro_time = micro.create_dataset(
            "time_offsets_ms", shape=(microtrace_samples,), dtype="f8"
        )
        self.micro_selected = micro.create_dataset(
            "selected_variables",
            shape=(0, microtrace_samples, microtrace_variable_count),
            maxshape=(None, microtrace_samples, microtrace_variable_count),
            chunks=(1, microtrace_samples, max(1, microtrace_variable_count)),
            dtype="f4",
            **options,
        )
        self.micro_probe_voltage = micro.create_dataset(
            "probe_voltage",
            shape=(0, microtrace_samples, probe_count),
            maxshape=(None, microtrace_samples, probe_count),
            chunks=(1, microtrace_samples, max(1, probe_count)),
            dtype="f4",
            **options,
        )
        self.micro_all_voltage = micro.create_dataset(
            "all_segment_voltage",
            shape=(0, microtrace_samples, segment_count),
            maxshape=(None, microtrace_samples, segment_count),
            chunks=(1, microtrace_samples, segment_count),
            dtype="f4",
            **options,
        )
        self.micro_all_voltage.attrs["policy"] = (
            "stored for this small diagnostic dataset; reconsider before scale-up"
        )
        self.micro_somatic_current = micro.create_dataset(
            "somatic_current_na",
            shape=(0, microtrace_samples),
            maxshape=(None, microtrace_samples),
            chunks=(1, microtrace_samples),
            dtype="f4",
            **options,
        )
        self.micro_somatic_current.attrs["meaning"] = (
            "current actually delivered by the diagnostic soma IClamp"
        )
        self.micro_observables = None
        if micro_observable_names:
            observable_count = len(micro_observable_names)
            self.micro_observables = micro.create_dataset(
                "protocol_observables",
                shape=(0, microtrace_samples, observable_count),
                maxshape=(None, microtrace_samples, observable_count),
                chunks=(1, microtrace_samples, observable_count),
                dtype="f4",
                **options,
            )
            self.micro_observables.attrs["variable_names_json"] = json.dumps(
                list(micro_observable_names), separators=(",", ":")
            )
        summaries = micro.create_group("all_segment_voltage_summary")
        self.micro_voltage_summary = {
            name: summaries.create_dataset(
                name,
                shape=(0, segment_count),
                maxshape=(None, segment_count),
                chunks=(1, segment_count),
                dtype="f4",
                **options,
            )
            for name in (
                "minimum_mv",
                "maximum_mv",
                "integral_mv_ms",
                "minimum_time_offset_ms",
                "maximum_time_offset_ms",
            )
        }

        strings = h5py.string_dtype(encoding="utf-8")
        metadata = self.file.create_group("metadata")
        self.meta = {
            name: metadata.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dtype
            )
            for name, dtype in {
                "transition_id": "i8",
                "trajectory_id": strings,
                "category": strings,
                "protocol": strings,
                "split": strings,
                "seed": "i8",
                "step_index": "i8",
                "start_time_ms": "f8",
                "native_snapshot_ref": strings,
                "snapshot_step_index": "i8",
                "protocol_id": strings,
                "protocol_variant": strings,
                "stimulus_relative_time_ms": "f8",
                "snapshot_source": strings,
                "microtrace_mode": strings,
                "negative_control": "i1",
            }.items()
        }
        inputs = self.file.create_group("inputs")
        self.input_json = inputs.create_dataset(
            "ordered_actions_json", shape=(0,), maxshape=(None,), dtype=strings
        )
        events = self.file.create_group("events")
        self.event_json = events.create_dataset(
            "labels_json", shape=(0,), maxshape=(None,), dtype=strings
        )

    def set_microtrace_grid(self, time_offsets_ms: Sequence[float]) -> None:
        values = self.np.asarray(time_offsets_ms, dtype=float)
        if values.shape != self.micro_time.shape:
            raise ValueError("microtrace time grid has the wrong shape")
        self.micro_time[...] = values

    @staticmethod
    def _append(dataset: Any, value: Any) -> None:
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def append(self, row: Mapping[str, Any]) -> int:
        index = self.count
        for category, datasets in self.state_datasets.items():
            self._append(datasets["t"], row["state_t"][category])
            self._append(
                datasets["t_plus_1"], row["state_t_plus_1"][category]
            )
        self._append(self.rng_datasets["t"], row["rng_t"])
        self._append(self.rng_datasets["t_plus_1"], row["rng_t_plus_1"])
        self._append(self.micro_selected, row["micro_selected"])
        self._append(self.micro_probe_voltage, row["micro_probe_voltage"])
        self._append(self.micro_all_voltage, row["micro_all_voltage"])
        self._append(
            self.micro_somatic_current, row["micro_somatic_current"]
        )
        if self.micro_observables is not None:
            self._append(
                self.micro_observables, row["micro_protocol_observables"]
            )
        all_voltage = self.np.asarray(row["micro_all_voltage"], dtype=float)
        # Index NumPy, not the h5py dataset.  Per-segment extrema naturally
        # produce repeated, non-monotonic indices, which h5py rejects for
        # fancy indexing even though the equivalent NumPy operation is valid.
        micro_time = self.micro_time[...]
        self._append(
            self.micro_voltage_summary["minimum_mv"],
            self.np.min(all_voltage, axis=0),
        )
        self._append(
            self.micro_voltage_summary["maximum_mv"],
            self.np.max(all_voltage, axis=0),
        )
        self._append(
            self.micro_voltage_summary["integral_mv_ms"],
            self.np.trapz(all_voltage, micro_time, axis=0),
        )
        self._append(
            self.micro_voltage_summary["minimum_time_offset_ms"],
            micro_time[self.np.argmin(all_voltage, axis=0)],
        )
        self._append(
            self.micro_voltage_summary["maximum_time_offset_ms"],
            micro_time[self.np.argmax(all_voltage, axis=0)],
        )

        metadata = row["metadata"]
        defaults = {
            "snapshot_step_index": metadata.get("step_index", 0),
            "protocol_id": metadata.get("protocol", ""),
            "protocol_variant": "canonical",
            "stimulus_relative_time_ms": float(metadata.get("step_index", 0)),
            "snapshot_source": "equilibrium_snapshot",
            "microtrace_mode": "full_all_segment_voltage",
            "negative_control": 0,
        }
        for name, dataset in self.meta.items():
            self._append(dataset, metadata.get(name, defaults.get(name)))
        self._append(
            self.input_json,
            json.dumps(row["inputs"], sort_keys=True, separators=(",", ":")),
        )
        self._append(
            self.event_json,
            json.dumps(row.get("events", []), sort_keys=True, separators=(",", ":")),
        )
        self.count += 1
        self.file.attrs.modify("transition_count", self.count)
        self.file.flush()
        return index

    def update_events(self, index: int, events: Sequence[Mapping[str, Any]]) -> None:
        if not 0 <= int(index) < self.count:
            raise IndexError("transition index outside written rows")
        self.event_json[int(index)] = json.dumps(
            list(events), sort_keys=True, separators=(",", ":")
        )

    def close(self) -> None:
        if self.file:
            self.file.attrs.modify("transition_count", self.count)
            self.file.close()
            self.file = None

    def __enter__(self) -> "TransitionH5Writer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def validate_hdf5_store(path: Path) -> Dict[str, Any]:
    """Validate dimensions, finite values, time grids, and split isolation."""

    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError("HDF5 validation requires h5py and numpy") from error

    from .diagnostic_contract import validate_split_isolation

    issues = []
    rows = []
    with h5py.File(path, "r") as handle:
        count = int(handle.attrs["transition_count"])
        for group_name in ("states", "rng_state", "microtraces", "metadata", "inputs", "events"):
            if group_name not in handle:
                issues.append(f"missing group {group_name}")
        for category in handle["states"]:
            for boundary in ("t", "t_plus_1"):
                values = handle[f"states/{category}/{boundary}"]
                if values.shape[0] != count:
                    issues.append(f"{values.name} row count mismatch")
                if not np.isfinite(values[...]).all():
                    issues.append(f"{values.name} contains NaN/Inf")
        for name in ("t", "t_plus_1"):
            values = handle[f"rng_state/{name}"]
            if values.shape[0] != count or not np.isfinite(values[...]).all():
                issues.append(f"{values.name} is invalid")
        grid = handle["microtraces/time_offsets_ms"][...]
        if grid[0] != 0.0 or abs(float(grid[-1]) - 1.0) > 1e-12:
            issues.append("microtrace grid must span [0, 1] ms")
        if not np.allclose(np.diff(grid), 0.025, atol=1e-12):
            issues.append("microtrace grid is not uniformly 0.025 ms")
        for name in (
            "microtraces/selected_variables",
            "microtraces/probe_voltage",
            "microtraces/all_segment_voltage",
            "microtraces/somatic_current_na",
            "microtraces/all_segment_voltage_summary/minimum_mv",
            "microtraces/all_segment_voltage_summary/maximum_mv",
            "microtraces/all_segment_voltage_summary/integral_mv_ms",
            "microtraces/all_segment_voltage_summary/minimum_time_offset_ms",
            "microtraces/all_segment_voltage_summary/maximum_time_offset_ms",
        ):
            values = handle[name]
            if values.shape[0] != count or not np.isfinite(values[...]).all():
                issues.append(f"/{name} is invalid")
        if "protocol_observables" in handle["microtraces"]:
            values = handle["microtraces/protocol_observables"]
            if values.shape[0] != count or not np.isfinite(values[...]).all():
                issues.append("/microtraces/protocol_observables is invalid")
        def text(value: Any) -> str:
            return value.decode() if isinstance(value, bytes) else str(value)

        for index in range(count):
            rows.append(
                {
                    "trajectory_id": text(handle["metadata/trajectory_id"][index]),
                    "protocol": text(handle["metadata/protocol"][index]),
                    "split": text(handle["metadata/split"][index]),
                    "seed": int(handle["metadata/seed"][index]),
                }
            )
            actions = json.loads(handle["inputs/ordered_actions_json"][index])
            offsets = [float(action["offset_ms"]) for action in actions]
            if offsets != sorted(offsets) or any(
                offset < 0.0 or offset >= 1.0 for offset in offsets
            ):
                issues.append(f"transition {index} has invalid input timestamps")
        try:
            validate_split_isolation(rows)
        except ValueError as error:
            issues.append(str(error))
    return {
        "valid": not issues,
        "transition_count": count,
        "issues": issues,
    }
