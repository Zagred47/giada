"""Compact distribution audit for a validated GIADA soma corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


ProgressCallback = Callable[[int, int], None]


def _empty_state() -> Dict[str, Any]:
    return {
        "delta": [],
        "voltage_t": [],
        "voltage_t1": [],
        "scheduled": 0,
        "realized": 0,
    }


def _append_state(
    state: Dict[str, Any],
    voltage_t: np.ndarray,
    voltage_t1: np.ndarray,
    scheduled: np.ndarray,
    realized: np.ndarray,
    mask: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    state["delta"].append(voltage_t1[mask] - voltage_t[mask])
    state["voltage_t"].append(voltage_t[mask])
    state["voltage_t1"].append(voltage_t1[mask])
    state["scheduled"] += int(scheduled[mask].sum())
    state["realized"] += int(realized[mask].sum())


def _plan_lookup(plan_path: Path | None) -> Dict[int, Dict[str, Any]]:
    if plan_path is None:
        return {}
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    result: Dict[int, Dict[str, Any]] = {}
    for shard in payload["shards"]:
        for row in shard["trajectories"]:
            result[int(row["trajectory_index"])] = {
                "protocol": str(row.get("protocol", "neuronio_nmda_ergodic_v1")),
                "split": str(row["split"]),
            }
    return result


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
    plan_path: Path | None = None,
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
    states: Dict[int, Dict[str, Any]] = {code: _empty_state() for code in (0, 1)}
    plan = _plan_lookup(plan_path)
    protocol_states: Dict[str, Dict[int, Dict[str, Any]]] = {}
    blockers = []
    for index, path in enumerate(paths, 1):
        with h5py.File(path, "r") as handle:
            split = np.asarray(handle["split_code"][:], dtype=np.uint8)
            voltage_t = np.asarray(handle["voltage_t_mv"][:, 0], dtype=np.float64)
            voltage_t1 = np.asarray(
                handle["voltage_t_plus_1_mv"][:, 0], dtype=np.float64
            )
            scheduled = np.asarray(handle["scheduled_event_count"][:])
            realized = np.asarray(handle["realized_event_count"][:])
            trajectory = np.asarray(handle["trajectory_index"][:], dtype=np.int64)
            for code in (0, 1):
                mask = split == code
                _append_state(
                    states[code], voltage_t, voltage_t1, scheduled, realized, mask
                )
            if plan:
                for trajectory_index in np.unique(trajectory):
                    planned = plan.get(int(trajectory_index))
                    if planned is None:
                        blockers.append(
                            f"trajectory {int(trajectory_index)} is absent from plan"
                        )
                        continue
                    expected_code = 1 if planned["split"] == "validation" else 0
                    row_mask = trajectory == trajectory_index
                    if np.any(split[row_mask] != expected_code):
                        blockers.append(
                            f"trajectory {int(trajectory_index)} split disagrees with plan"
                        )
                    by_split = protocol_states.setdefault(
                        planned["protocol"],
                        {code: _empty_state() for code in (0, 1)},
                    )
                    _append_state(
                        by_split[expected_code],
                        voltage_t,
                        voltage_t1,
                        scheduled,
                        realized,
                        row_mask,
                    )
        if progress is not None:
            progress(index, len(paths))

    missing = [code for code, state in states.items() if not state["delta"]]
    report = {
        "schema_version": "giada-runpod-soma-corpus-audit-v1",
        "valid": not missing and not blockers,
        "corpus_root": str(root.resolve()),
        "shard_count": len(paths),
        "missing_split_codes": missing,
        "blockers": sorted(set(blockers)),
        "splits": {
            name: _split_summary(states[code])
            for code, name in ((0, "train"), (1, "validation"))
            if states[code]["delta"]
        },
    }
    if protocol_states:
        report["protocol_splits"] = {
            protocol: {
                name: _split_summary(by_split[code])
                for code, name in ((0, "train"), (1, "validation"))
                if by_split[code]["delta"]
            }
            for protocol, by_split in sorted(protocol_states.items())
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
