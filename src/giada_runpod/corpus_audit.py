"""Compact distribution audit for a validated GIADA soma corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


ProgressCallback = Callable[[int, int], None]


def _split_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    delta = np.concatenate(state["delta"])
    voltage_t = np.concatenate(state["voltage_t"])
    voltage_t1 = np.concatenate(state["voltage_t1"])
    absolute = np.abs(delta)
    quantiles = np.quantile(absolute, [0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
    return {
        "transition_count": int(len(delta)),
        "absolute_delta_quantiles_mv": {
            label: float(value)
            for label, value in zip(
                ("p50", "p90", "p95", "p99", "p99_9", "maximum"),
                quantiles,
            )
        },
        "absolute_delta_ge_1mv_count": int(np.count_nonzero(absolute >= 1.0)),
        "absolute_delta_ge_2mv_count": int(np.count_nonzero(absolute >= 2.0)),
        "absolute_delta_ge_5mv_count": int(np.count_nonzero(absolute >= 5.0)),
        "somatic_upcrossings_minus55mv": int(
            np.count_nonzero((voltage_t < -55.0) & (voltage_t1 >= -55.0))
        ),
        "voltage_minimum_mv": float(min(voltage_t.min(), voltage_t1.min())),
        "voltage_maximum_mv": float(max(voltage_t.max(), voltage_t1.max())),
        "scheduled_event_count": int(state["scheduled"]),
        "realized_event_count": int(state["realized"]),
    }


def audit_soma_corpus(
    corpus_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Read every shard and summarize target/activity support without mutation."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("soma corpus audit requires h5py") from error

    root = Path(corpus_root)
    paths = sorted((root / "shards").glob("shard-*.h5"))
    if not paths:
        raise FileNotFoundError(f"no soma shards under {root}")
    states: Dict[int, Dict[str, Any]] = {
        code: {
            "delta": [],
            "voltage_t": [],
            "voltage_t1": [],
            "scheduled": 0,
            "realized": 0,
        }
        for code in (0, 1)
    }
    for index, path in enumerate(paths, 1):
        with h5py.File(path, "r") as handle:
            split = np.asarray(handle["split_code"][:], dtype=np.uint8)
            voltage_t = np.asarray(handle["voltage_t_mv"][:, 0], dtype=np.float64)
            voltage_t1 = np.asarray(
                handle["voltage_t_plus_1_mv"][:, 0], dtype=np.float64
            )
            scheduled = np.asarray(handle["scheduled_event_count"][:])
            realized = np.asarray(handle["realized_event_count"][:])
            for code in (0, 1):
                mask = split == code
                if not np.any(mask):
                    continue
                state = states[code]
                state["delta"].append(voltage_t1[mask] - voltage_t[mask])
                state["voltage_t"].append(voltage_t[mask])
                state["voltage_t1"].append(voltage_t1[mask])
                state["scheduled"] += int(scheduled[mask].sum())
                state["realized"] += int(realized[mask].sum())
        if progress is not None:
            progress(index, len(paths))

    missing = [code for code, state in states.items() if not state["delta"]]
    report = {
        "schema_version": "giada-runpod-soma-corpus-audit-v1",
        "valid": not missing,
        "corpus_root": str(root.resolve()),
        "shard_count": len(paths),
        "missing_split_codes": missing,
        "splits": {
            name: _split_summary(states[code])
            for code, name in ((0, "train"), (1, "validation"))
            if states[code]["delta"]
        },
    }
    validation_path = root / "validation_report.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        report["source_validation"] = {
            "valid": bool(validation.get("valid")),
            "validated_shard_count": validation.get("validated_shard_count"),
            "validated_transition_count": validation.get(
                "validated_transition_count"
            ),
        }
        report["valid"] = bool(report["valid"] and validation.get("valid"))
    return report
