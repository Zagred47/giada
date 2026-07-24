"""Metrics and decision logic for diagnostic full-state flow-map experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from ..hayflow_data.flowmap_dataset import DYNAMIC_CATEGORIES, EVENT_KINDS


def _errors(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    absolute = np.abs(error).reshape(-1)
    maximum = float(np.max(absolute)) if len(absolute) else 0.0
    if math.isfinite(maximum) and maximum > 0.0:
        rmse = maximum * float(np.sqrt(np.mean((error / maximum) ** 2)))
    else:
        rmse = maximum
    return {
        "rmse": rmse,
        "mae": float(np.mean(absolute)),
        "absolute_error_p50": float(np.percentile(absolute, 50.0)),
        "absolute_error_p95": float(np.percentile(absolute, 95.0)),
        "absolute_error_p99": float(np.percentile(absolute, 99.0)),
    }


def state_metric_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    layout: Any,
    normalizer: Optional[Any] = None,
    model_name: str,
    split: str,
    regimes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return global, block, region, mechanism, and physical-domain metrics."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    regimes = list(regimes or ["all"] * len(prediction))
    rows: List[Dict[str, Any]] = []

    transformed_prediction = None
    transformed_target = None
    if normalizer is not None:
        transformed_prediction = normalizer.normalize_state(prediction)
        transformed_target = normalizer.normalize_state(target)

    def add(
        scope: str,
        name: str,
        pred: Any,
        true: Any,
        *,
        normalized_pred: Any = None,
        normalized_true: Any = None,
        **extra: Any,
    ) -> None:
        metrics = _errors(pred, true)
        if normalized_pred is not None and normalized_true is not None:
            normalized = _errors(normalized_pred, normalized_true)
            metrics.update(
                {
                    "normalized_rmse": normalized["rmse"],
                    "normalized_mae": normalized["mae"],
                }
            )
        rows.append(
            {
                "model": model_name,
                "split": split,
                "scope": scope,
                "name": name,
                **metrics,
                **extra,
            }
        )

    add(
        "state",
        "all",
        prediction,
        target,
        normalized_pred=transformed_prediction,
        normalized_true=transformed_target,
    )
    for category in DYNAMIC_CATEGORIES:
        state_slice = layout.category_slices[category]
        add(
            "category",
            category,
            prediction[:, state_slice],
            target[:, state_slice],
            normalized_pred=(
                transformed_prediction[:, state_slice]
                if transformed_prediction is not None
                else None
            ),
            normalized_true=(
                transformed_target[:, state_slice]
                if transformed_target is not None
                else None
            ),
        )

    voltage = layout.category_slices["voltage"]
    for region_id, region in enumerate(layout.region_names):
        segment_mask = layout.segment_region_ids == region_id
        if not segment_mask.any():
            continue
        add(
            "voltage_region",
            region,
            prediction[:, voltage][:, segment_mask],
            target[:, voltage][:, segment_mask],
        )
    for regime in sorted(set(regimes)):
        row_mask = np.asarray([value == regime for value in regimes])
        if row_mask.any():
            add(
                "voltage_regime",
                regime,
                prediction[row_mask, voltage],
                target[row_mask, voltage],
                sample_count=int(row_mask.sum()),
            )

    by_mechanism: Dict[str, List[int]] = defaultdict(list)
    by_category: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(layout.core_records):
        by_mechanism[str(record["mechanism"])].append(index)
        by_category[str(record["category"])].append(index)
    for mechanism, indices in sorted(by_mechanism.items()):
        if mechanism == "neuron":
            continue
        add(
            "mechanism",
            mechanism,
            prediction[:, indices],
            target[:, indices],
            normalized_pred=(
                transformed_prediction[:, indices]
                if transformed_prediction is not None
                else None
            ),
            normalized_true=(
                transformed_target[:, indices]
                if transformed_target is not None
                else None
            ),
        )

    eps = 1e-12
    calcium_indices = np.asarray(by_category["calcium_ions"], dtype=np.int64)
    calcium_segments = layout.core_segment_ids[calcium_indices]
    for region_id, region in enumerate(layout.region_names):
        selected = calcium_indices[calcium_segments == region_id]
        if not len(selected):
            continue
        add(
            "calcium_region",
            region,
            prediction[:, selected],
            target[:, selected],
        )
    calcium_error = np.abs(
        prediction[:, calcium_indices] - target[:, calcium_indices]
    )
    rows.append(
        {
            "model": model_name,
            "split": split,
            "scope": "calcium",
            "name": "all",
            "mae": float(calcium_error.mean()),
            "relative_mae": float(
                np.mean(calcium_error / np.maximum(np.abs(target[:, calcium_indices]), eps))
            ),
        }
    )

    bounded = [
        i
        for i, record in enumerate(layout.core_records)
        if record["category"] == "mechanism_states"
        or (
            record["category"] == "synapse_states"
            and record["mechanism"] == "NetCon"
            and record["variable"] in {"Pv", "Pr", "u"}
        )
    ]
    positive = [
        i
        for i, record in enumerate(layout.core_records)
        if record["category"] == "calcium_ions"
        or (
            record["category"] == "synapse_states"
            and record["variable"] != "tsyn"
            and i not in bounded
        )
    ]
    outside = 0
    denominator = 0
    if bounded:
        values = prediction[:, bounded]
        outside += int(((values < 0.0) | (values > 1.0)).sum())
        denominator += values.size
    if positive:
        values = prediction[:, positive]
        outside += int((values < 0.0).sum())
        denominator += values.size
    rows.append(
        {
            "model": model_name,
            "split": split,
            "scope": "physical_domain",
            "name": "bounded_and_positive",
            "outside_domain_count": outside,
            "outside_domain_fraction": float(outside / max(1, denominator)),
        }
    )
    return rows


def binary_event_metric_rows(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    timing_prediction: Optional[np.ndarray],
    timing_target: Optional[np.ndarray],
    timing_mask: Optional[np.ndarray],
    model_name: str,
    split: str,
    region_prediction: Optional[np.ndarray] = None,
    region_target: Optional[np.ndarray] = None,
    region_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    predicted = np.asarray(probabilities) >= float(threshold)
    targets = np.asarray(targets).astype(bool)
    rows = []
    for column, kind in enumerate(EVENT_KINDS):
        true_positive = int((predicted[:, column] & targets[:, column]).sum())
        false_positive = int((predicted[:, column] & ~targets[:, column]).sum())
        false_negative = int((~predicted[:, column] & targets[:, column]).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        row: Dict[str, Any] = {
            "model": model_name,
            "split": split,
            "event_kind": kind,
            "threshold": float(threshold),
            "support": int(targets[:, column].sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if (
            timing_prediction is not None
            and timing_target is not None
            and timing_mask is not None
        ):
            mask = np.asarray(timing_mask)[:, column, :].astype(bool)
            absolute = np.abs(
                np.asarray(timing_prediction)[:, column, :]
                - np.asarray(timing_target)[:, column, :]
            )
            names = ("onset", "peak", "offset", "duration")
            for timing_column, name in enumerate(names):
                selected = mask[:, timing_column]
                row[f"{name}_mae_ms"] = (
                    float(absolute[selected, timing_column].mean())
                    if selected.any()
                    else math.nan
                )
        if (
            region_prediction is not None
            and region_target is not None
            and region_mask is not None
        ):
            selected = np.asarray(region_mask)[:, column].astype(bool)
            row["region_support"] = int(selected.sum())
            row["region_accuracy"] = (
                float(
                    np.mean(
                        np.asarray(region_prediction)[selected, column]
                        == np.asarray(region_target)[selected, column]
                    )
                )
                if selected.any()
                else math.nan
            )
        rows.append(row)
    return rows


def rollout_metric_row(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    model_name: str,
    split: str,
    horizon_ms: int,
    voltage_width: int,
    layout: Optional[Any] = None,
) -> Dict[str, Any]:
    row = {
        "model": model_name,
        "split": split,
        "horizon_ms": int(horizon_ms),
        **_errors(prediction, target),
    }
    voltage_prediction = prediction[..., :voltage_width]
    voltage_target = target[..., :voltage_width]
    row["voltage_rmse_mv"] = float(
        np.sqrt(np.mean((voltage_prediction - voltage_target) ** 2))
    )
    row["voltage_rest_drift_mv"] = float(
        np.mean(voltage_prediction - voltage_target)
    )
    row["numerically_finite"] = bool(np.isfinite(prediction).all())
    row["maximum_absolute_state_value"] = float(np.max(np.abs(prediction)))
    if layout is not None:
        bounded = [
            i
            for i, record in enumerate(layout.core_records)
            if record["category"] == "mechanism_states"
        ]
        positive = [
            i
            for i, record in enumerate(layout.core_records)
            if record["category"] == "calcium_ions"
            or (
                record["category"] == "synapse_states"
                and record["variable"] != "tsyn"
            )
        ]
        outside = 0
        denominator = 0
        if bounded:
            values = prediction[:, bounded]
            outside += int(((values < 0.0) | (values > 1.0)).sum())
            denominator += values.size
        if positive:
            values = prediction[:, positive]
            outside += int((values < 0.0).sum())
            denominator += values.size
        row["outside_domain_fraction"] = float(outside / max(1, denominator))
    return row


def decide_go_no_go(
    comparisons: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply explicit, conservative diagnostic decision rules."""

    b3_test = float(comparisons.get("b3_test_voltage_rmse", math.inf))
    persistence = float(comparisons.get("persistence_test_voltage_rmse", math.inf))
    affine = float(comparisons.get("affine_test_voltage_rmse", math.inf))
    rollout_bounded = bool(comparisons.get("b3_rollout_16ms_bounded", False))
    event_f1 = float(comparisons.get("b3_macro_event_f1", 0.0))
    branching_distinguished = bool(
        comparisons.get("branching_futures_distinguished", False)
    )
    full_gain = float(comparisons.get("full_state_gain_fraction", 0.0))
    privileged_gain = float(comparisons.get("privileged_gain_fraction", 0.0))
    privileged_rollout_gain = float(
        comparisons.get("privileged_rollout_gain_fraction", 0.0)
    )
    privileged_event_gain = float(comparisons.get("privileged_event_f1_gain", 0.0))
    generalizes = b3_test < 0.95 * min(persistence, affine)
    state_helped = full_gain > 0.02
    privileged_candidates = (
        privileged_gain,
        privileged_rollout_gain,
        privileged_event_gain,
    )
    privileged_helped = any(
        math.isfinite(value) and value > 0.0 for value in privileged_candidates
    )
    event_usable = event_f1 >= 0.5
    if (
        generalizes
        and rollout_bounded
        and event_usable
        and state_helped
        and privileged_helped
        and branching_distinguished
    ):
        decision = "GO"
        blockers: List[str] = []
    elif generalizes and event_usable:
        decision = "CONDITIONAL_GO"
        blockers = [] if rollout_bounded else ["16 ms rollout is not bounded"]
        if not state_helped:
            blockers.append("full state has not shown a material gain")
        if not privileged_helped:
            blockers.append("privileged supervision has not improved fidelity")
        if not branching_distinguished:
            blockers.append("branching futures were not reliably distinguished")
    else:
        decision = "NO_GO"
        blockers = []
        if not generalizes:
            blockers.append("B3 does not clearly beat persistence and affine on tests")
        if not event_usable:
            blockers.append("event macro-F1 remains below 0.5")
    return {
        "decision": decision,
        "blockers": blockers,
        "criteria": {
            "b3_generalizes_beyond_b0_b1": generalizes,
            "rollout_16ms_bounded": rollout_bounded,
            "event_macro_f1_at_least_0_5": event_usable,
            "branching_futures_distinguished": branching_distinguished,
            "full_state_material_gain": state_helped,
            "privileged_supervision_gain": privileged_helped,
        },
        "inputs": dict(comparisons),
    }


def write_parquet(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a required Parquet table and fail rather than silently downgrade."""

    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False)
