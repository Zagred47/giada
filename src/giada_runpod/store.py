"""Compact append-only HDF5 shards for paper-scale one-step evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "giada-paper-scale-shard-v1"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class LeanShardWriter:
    """Write local raw state and causal inputs without diagnostic microtraces.

    The file is written to ``*.partial`` and becomes visible at its final name
    only after a successful close.  This is the basis of safe interruption and
    resume on preemptible RunPod machines.
    """

    def __init__(
        self,
        path: Path,
        *,
        segment_count_per_transition: int,
        mechanism_group_count: int,
        ion_count: int,
        schema_metadata: Mapping[str, Any],
        compression: str = "lzf",
        chunk_transitions: int = 256,
    ) -> None:
        try:
            import h5py
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("lean RunPod storage requires h5py") from error
        self.h5py = h5py
        self.path = Path(path)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite shard {self.path}")
        self.k = int(segment_count_per_transition)
        self.g = int(mechanism_group_count)
        self.i = int(ion_count)
        if min(self.k, self.g, self.i) < 0 or self.k == 0:
            raise ValueError("invalid compact shard dimensions")
        compression_value = None if compression == "none" else compression
        self.handle = h5py.File(self.partial_path, "w")
        self.handle.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "complete": False,
                "transition_count": 0,
                "schema_metadata_json": json.dumps(dict(schema_metadata), sort_keys=True),
            }
        )
        self.datasets: Dict[str, Any] = {}
        chunk = max(1, int(chunk_transitions))

        def dataset(name: str, tail: tuple[int, ...], dtype: Any) -> Any:
            ds = self.handle.create_dataset(
                name,
                shape=(0, *tail),
                maxshape=(None, *tail),
                chunks=(chunk, *tail),
                dtype=dtype,
                compression=compression_value,
                shuffle=bool(compression_value),
            )
            self.datasets[name] = ds
            return ds

        dataset("segment_id", (self.k,), "i4")
        for name in (
            "voltage_t_mv",
            "voltage_t_plus_1_mv",
            "parent_delta_t_mv",
            "mean_child_delta_t_mv",
        ):
            dataset(name, (self.k,), "f4")
        dataset("mechanism_state_t", (self.k, self.g), "f4")
        dataset("ion_state_t", (self.k, self.i), "f4")
        dataset("causal_drive", (self.k, 12), "f4")
        dataset("trajectory_index", (), "i8")
        dataset("step_index", (), "i4")
        dataset("seed", (), "i8")
        dataset("split_code", (), "u1")
        dataset("scheduled_event_count", (), "i4")
        dataset("realized_event_count", (), "i4")
        self.event_datasets: Dict[str, Any] = {}
        event_chunk = max(256, chunk * 8)

        def event_dataset(name: str, dtype: Any) -> None:
            self.event_datasets[name] = self.handle.create_dataset(
                f"events/{name}",
                shape=(0,),
                maxshape=(None,),
                chunks=(event_chunk,),
                dtype=dtype,
                compression=compression_value,
                shuffle=bool(compression_value),
            )

        for name, dtype in (
            ("transition_row", "i8"),
            ("synapse_id", "i4"),
            ("segment_id", "i4"),
            ("offset_ms", "f4"),
            ("release_success", "u1"),
            ("released_quantity", "f4"),
            ("ampa_state_increment", "f4"),
            ("nmda_state_increment", "f4"),
            ("inhibitory_state_increment", "f4"),
        ):
            event_dataset(name, dtype)
        self.count = 0
        self.event_count = 0

    def append(self, row: Mapping[str, Any], realized_events: Sequence[Mapping[str, Any]]) -> None:
        index = self.count
        required = {
            "segment_id",
            "voltage_t_mv",
            "voltage_t_plus_1_mv",
            "parent_delta_t_mv",
            "mean_child_delta_t_mv",
            "mechanism_state_t",
            "ion_state_t",
            "causal_drive",
            "trajectory_index",
            "step_index",
            "seed",
            "split_code",
            "scheduled_event_count",
            "realized_event_count",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"compact transition missing fields: {sorted(missing)}")
        for name, dataset in self.datasets.items():
            dataset.resize(index + 1, axis=0)
            dataset[index] = row[name]
        events = [item for item in realized_events if item.get("kind") == "synaptic_event"]
        if events:
            start, stop = self.event_count, self.event_count + len(events)
            for dataset in self.event_datasets.values():
                dataset.resize(stop, axis=0)
            for offset, event in enumerate(events):
                target = start + offset
                values = {
                    "transition_row": index,
                    "synapse_id": int(event["synapse_id"]),
                    "segment_id": int(event["segment_id"]),
                    "offset_ms": float(event.get("offset_ms", 0.0)),
                    "release_success": int(bool(event.get("release_success", False))),
                    "released_quantity": float(event.get("released_quantity", 0.0)),
                    "ampa_state_increment": float(event.get("ampa_state_increment", 0.0)),
                    "nmda_state_increment": float(event.get("nmda_state_increment", 0.0)),
                    "inhibitory_state_increment": float(event.get("inhibitory_state_increment", 0.0)),
                }
                for name, dataset in self.event_datasets.items():
                    dataset[target] = values[name]
            self.event_count = stop
        self.count += 1

    def close(self, *, expected_transition_count: int) -> Dict[str, Any]:
        if self.count != int(expected_transition_count):
            self.handle.close()
            raise RuntimeError(
                f"shard contains {self.count:,} transitions, expected "
                f"{expected_transition_count:,}"
            )
        self.handle.attrs["transition_count"] = self.count
        self.handle.attrs["causal_release_outcome_count"] = self.event_count
        self.handle.attrs["complete"] = True
        self.handle.flush()
        self.handle.close()
        self.partial_path.replace(self.path)
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(self.path),
            "transition_count": self.count,
            "causal_release_outcome_count": self.event_count,
            "size_bytes": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
        }

    def abort(self) -> None:
        if getattr(self, "handle", None):
            self.handle.close()

    def __enter__(self) -> "LeanShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            self.abort()


def validate_lean_shard(path: Path, *, expected_transition_count: int | None = None) -> Dict[str, Any]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("lean RunPod storage requires h5py") from error
    source = Path(path)
    blockers = []
    with h5py.File(source, "r") as handle:
        count = int(handle.attrs.get("transition_count", -1))
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            blockers.append("schema version mismatch")
        if not bool(handle.attrs.get("complete", False)):
            blockers.append("shard is not marked complete")
        if expected_transition_count is not None and count != int(expected_transition_count):
            blockers.append("transition count mismatch")
        required = (
            "segment_id", "voltage_t_mv", "voltage_t_plus_1_mv",
            "mechanism_state_t", "ion_state_t", "causal_drive",
        )
        for name in required:
            if name not in handle:
                blockers.append(f"missing dataset {name}")
            elif handle[name].shape[0] != count:
                blockers.append(f"row count mismatch in {name}")
            elif not np.isfinite(handle[name][...]).all():
                blockers.append(f"non-finite values in {name}")
        if "events/transition_row" in handle:
            event_rows = handle["events/transition_row"][...]
            if len(event_rows) and (event_rows.min() < 0 or event_rows.max() >= count):
                blockers.append("event transition reference out of range")
    return {
        "valid": not blockers,
        "blockers": blockers,
        "path": str(source),
        "transition_count": count,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
