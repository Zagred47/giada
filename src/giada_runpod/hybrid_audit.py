"""Causal-contract and outcome audit for the GIADA hybrid pilot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from .hybrid_inputs import HYBRID_PROTOCOL_SPECS


def audit_hybrid_corpus(
    corpus_root: Path,
    *,
    plan_path: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Dict[str, Any]:
    """Verify design identity and summarize each causal arm without selection."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("hybrid corpus audit requires h5py") from error
    root = Path(corpus_root)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    trajectories = {
        int(row["trajectory_index"]): row
        for shard in plan["shards"]
        for row in shard["trajectories"]
    }
    expected_protocols = {row.protocol for row in HYBRID_PROTOCOL_SPECS}
    expected_arms = defaultdict(set)
    target_segments = {}
    for row in HYBRID_PROTOCOL_SPECS:
        expected_arms[row.family].add(row.arm)
        target_segments[row.protocol] = row.target_segment_id
    blockers = []
    pair_rows: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    summaries: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    split_seeds: Dict[str, set[int]] = defaultdict(set)
    paths = sorted((root / "shards").glob("shard-*.h5"))
    if not paths:
        raise FileNotFoundError(f"no hybrid shards under {root}")
    for index, path in enumerate(paths, 1):
        with h5py.File(path, "r") as handle:
            metadata = json.loads(handle.attrs["schema_metadata_json"])
            if metadata.get("input_methodology") != "GIADA_hybrid_stochastic_plus_causal_paired_v1":
                blockers.append(f"{path.name}: wrong input methodology")
            if np.any(np.asarray(handle["high_resolution_sample_count"][:]) != 41):
                blockers.append(f"{path.name}: missing 0.025 ms micro-sampling")
            trajectory_ids = np.asarray(handle["trajectory_index"][:], dtype=np.int64)
            segment_ids = np.asarray(handle["segment_id"][:], dtype=np.int64)
            maxima = np.asarray(handle["voltage_max_mv"][:], dtype=np.float64)
            minima = np.asarray(handle["voltage_min_mv"][:], dtype=np.float64)
            for trajectory_index in np.unique(trajectory_ids):
                planned = trajectories.get(int(trajectory_index))
                if planned is None:
                    blockers.append(f"unplanned trajectory {int(trajectory_index)}")
                    continue
                mask = trajectory_ids == trajectory_index
                split = str(planned["split"])
                protocol = str(planned["protocol"])
                split_seeds[split].add(int(planned["seed"]))
                selected = segment_ids[mask][0]
                if not np.all(segment_ids[mask] == selected):
                    blockers.append(f"trajectory {trajectory_index}: unstable segment mapping")
                target = int(target_segments[protocol])
                if target not in selected:
                    blockers.append(f"{protocol}: target segment {target} not stored")
                    target_peak = float("nan")
                else:
                    target_column = int(np.flatnonzero(selected == target)[0])
                    target_peak = float(maxima[mask, target_column].max())
                soma_column = int(np.flatnonzero(selected == 0)[0])
                row = {
                    "trajectory_index": int(trajectory_index),
                    "protocol": protocol,
                    "family": str(planned["protocol_family"]),
                    "arm": str(planned["protocol_arm"]),
                    "pair_id": str(planned["pair_id"]),
                    "seed": int(planned["seed"]),
                    "split": split,
                    "soma_peak_mv": float(maxima[mask, soma_column].max()),
                    "target_peak_mv": target_peak,
                    "minimum_voltage_mv": float(minima[mask].min()),
                }
                pair_rows[row["pair_id"]].append(row)
                state = summaries[protocol].setdefault(
                    split,
                    {"count": 0, "soma_peaks": [], "target_peaks": []},
                )
                state["count"] += 1
                state["soma_peaks"].append(row["soma_peak_mv"])
                state["target_peaks"].append(row["target_peak_mv"])
        if progress is not None:
            progress(index, len(paths))
    observed_protocols = set(summaries)
    if observed_protocols != expected_protocols:
        blockers.append(
            f"protocol coverage mismatch: missing={sorted(expected_protocols-observed_protocols)} "
            f"extra={sorted(observed_protocols-expected_protocols)}"
        )
    pair_contrasts = []
    for pair_id, rows in sorted(pair_rows.items()):
        families = {row["family"] for row in rows}
        if len(families) != 1:
            blockers.append(f"{pair_id}: mixed families")
            continue
        family = next(iter(families))
        if {row["arm"] for row in rows} != expected_arms[family]:
            blockers.append(f"{pair_id}: incomplete causal arms")
        if len({row["seed"] for row in rows}) != 1 or len({row["split"] for row in rows}) != 1:
            blockers.append(f"{pair_id}: arms do not share seed and split")
        ordered = sorted(rows, key=lambda row: row["arm"])
        pair_contrasts.append(
            {
                "pair_id": pair_id,
                "family": family,
                "split": ordered[0]["split"],
                "seed": ordered[0]["seed"],
                "arms": [row["arm"] for row in ordered],
                "soma_peak_range_mv": float(
                    max(row["soma_peak_mv"] for row in ordered)
                    - min(row["soma_peak_mv"] for row in ordered)
                ),
                "target_peak_range_mv": float(
                    max(row["target_peak_mv"] for row in ordered)
                    - min(row["target_peak_mv"] for row in ordered)
                ),
            }
        )
    seed_leakage = sorted(split_seeds["train"] & split_seeds["validation"])
    if seed_leakage:
        blockers.append(f"seed leakage across splits: {seed_leakage}")
    compact = {
        protocol: {
            split: {
                "trajectory_count": int(state["count"]),
                "median_soma_peak_mv": float(np.median(state["soma_peaks"])),
                "median_target_peak_mv": float(np.median(state["target_peaks"])),
                "soma_peak_ge_minus20_count": int(
                    np.count_nonzero(np.asarray(state["soma_peaks"]) >= -20.0)
                ),
                "target_peak_ge_minus45_count": int(
                    np.count_nonzero(np.asarray(state["target_peaks"]) >= -45.0)
                ),
            }
            for split, state in sorted(by_split.items())
        }
        for protocol, by_split in sorted(summaries.items())
    }
    return {
        "schema_version": "giada-runpod-hybrid-corpus-audit-v1",
        "valid": not blockers,
        "blockers": blockers,
        "methodology": "GIADA hybrid: stochastic background plus causal targeted pairs",
        "corpus_root": str(root.resolve()),
        "shard_count": len(paths),
        "trajectory_count": len(pair_rows) and sum(len(rows) for rows in pair_rows.values()),
        "seed_split_leakage": seed_leakage,
        "protocol_summaries": compact,
        "pair_contrasts": pair_contrasts,
        "outcomes_used_for_plan_selection": False,
        "paper_test_claimed": False,
    }

