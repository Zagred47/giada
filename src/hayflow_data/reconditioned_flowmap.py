"""Train-only distribution audit and zero-inflated normalization for 02b."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .flowmap_dataset import DYNAMIC_CATEGORIES, FlowmapLayout


@dataclass(frozen=True)
class ReconditioningConfig:
    activity_epsilon: float = 1e-9
    sparse_update_fraction: float = 0.10
    minimum_scale: float = 1e-8
    gate_transform: str = "logit"

    def validate(self) -> None:
        if self.activity_epsilon <= 0.0:
            raise ValueError("activity_epsilon must be positive")
        if not 0.0 < self.sparse_update_fraction < 1.0:
            raise ValueError("sparse_update_fraction must be in (0, 1)")
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        if self.gate_transform not in {"identity", "logit"}:
            raise ValueError("gate_transform must be identity or logit")


def _mad(values: np.ndarray, center: Optional[np.ndarray] = None) -> np.ndarray:
    center = np.median(values, axis=0) if center is None else center
    return 1.4826 * np.median(np.abs(values - center), axis=0)


def _safe_scale(
    values: np.ndarray,
    *,
    minimum: float,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Hybrid robust/RMS scale that cannot amplify sparse outliers."""

    center = np.median(values, axis=0) if center is None else center
    mad = _mad(values, center)
    standard = np.std(values, axis=0)
    rms = np.sqrt(np.mean((values - center) ** 2, axis=0))
    return np.maximum.reduce(
        [mad, standard, rms, np.full_like(mad, float(minimum))]
    )


def _nonzero_statistics(
    values: np.ndarray,
    active: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return zero-centered scales estimated only from active updates."""

    width = values.shape[1]
    center = np.zeros(width, dtype=np.float64)
    scale = np.ones(width, dtype=np.float64)
    for column in range(width):
        selected = values[active[:, column], column]
        if len(selected):
            location = float(np.median(selected))
            scale[column] = float(
                max(
                    1.4826 * np.median(np.abs(selected - location)),
                    np.std(selected),
                    np.sqrt(np.mean(selected ** 2)),
                )
            )
    return center, scale


def _physical_domain(record: Mapping[str, Any]) -> str:
    category = str(record["category"])
    mechanism = str(record["mechanism"])
    variable = str(record["variable"])
    if category == "mechanism_states":
        return "bounded_0_1"
    if category == "calcium_ions":
        return "nonnegative"
    if category == "synapse_states":
        if mechanism == "NetCon" and variable in {"Pv", "Pr", "u"}:
            return "bounded_0_1"
        if variable != "tsyn":
            return "nonnegative"
    return "unbounded"


class ReconditionedStateNormalizer:
    """Semantic state transform with a separate sparse-value contract."""

    IDENTITY = 0
    LOG1P = 1
    LOGIT = 2

    def __init__(
        self,
        layout: FlowmapLayout,
        config: ReconditioningConfig,
    ) -> None:
        config.validate()
        self.layout = layout
        self.config = config
        self.transform_codes = self._transform_codes()
        self.state_center: Optional[np.ndarray] = None
        self.state_scale: Optional[np.ndarray] = None
        self.delta_center: Optional[np.ndarray] = None
        self.delta_scale: Optional[np.ndarray] = None
        self.update_fraction: Optional[np.ndarray] = None
        self.sparse_mask: Optional[np.ndarray] = None
        self.activity_positive_weight: Optional[np.ndarray] = None
        self.audit_rows: List[Dict[str, Any]] = []

    def _transform_codes(self) -> np.ndarray:
        codes = np.zeros(self.layout.state_width, dtype=np.int8)
        for index, row in enumerate(self.layout.core_records):
            category = str(row["category"])
            mechanism = str(row["mechanism"])
            variable = str(row["variable"])
            if category == "mechanism_states" and self.config.gate_transform == "logit":
                codes[index] = self.LOGIT
            elif category == "calcium_ions":
                codes[index] = self.LOG1P
            elif category == "synapse_states":
                if (
                    mechanism == "NetCon"
                    and variable in {"Pv", "Pr", "u"}
                    and self.config.gate_transform == "logit"
                ):
                    codes[index] = self.LOGIT
                elif variable != "tsyn":
                    codes[index] = self.LOG1P
        return codes

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        log_mask = self.transform_codes == self.LOG1P
        logit_mask = self.transform_codes == self.LOGIT
        if log_mask.any():
            result[..., log_mask] = np.log1p(
                np.maximum(result[..., log_mask], 0.0)
            )
        if logit_mask.any():
            clipped = np.clip(result[..., logit_mask], 1e-6, 1.0 - 1e-6)
            result[..., logit_mask] = np.log(clipped / (1.0 - clipped))
        return result

    def inverse(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        log_mask = self.transform_codes == self.LOG1P
        logit_mask = self.transform_codes == self.LOGIT
        if log_mask.any():
            transformed = np.clip(result[..., log_mask], -30.0, 30.0)
            result[..., log_mask] = np.maximum(np.expm1(transformed), 0.0)
        if logit_mask.any():
            transformed = np.clip(result[..., logit_mask], -30.0, 30.0)
            result[..., logit_mask] = 1.0 / (1.0 + np.exp(-transformed))
        return result

    def fit(
        self,
        state_t: np.ndarray,
        state_t_plus_1: np.ndarray,
    ) -> "ReconditionedStateNormalizer":
        z_t = self.transform(state_t)
        z_t1 = self.transform(state_t_plus_1)
        delta = z_t1 - z_t
        activity = np.abs(delta) > self.config.activity_epsilon
        self.update_fraction = activity.mean(axis=0)
        self.sparse_mask = self.update_fraction < self.config.sparse_update_fraction

        self.state_center = np.median(z_t, axis=0)
        self.state_scale = _safe_scale(
            z_t,
            center=self.state_center,
            minimum=self.config.minimum_scale,
        )

        dense_center = np.median(delta, axis=0)
        dense_scale = _safe_scale(
            delta,
            center=dense_center,
            minimum=self.config.minimum_scale,
        )
        active_center, active_scale = _nonzero_statistics(delta, activity)
        active_scale = np.maximum(active_scale, self.config.minimum_scale)
        self.delta_center = np.where(
            self.sparse_mask, active_center, dense_center
        )
        self.delta_scale = np.where(
            self.sparse_mask, active_scale, dense_scale
        )
        positives = activity.sum(axis=0).astype(np.float64)
        negatives = len(activity) - positives
        self.activity_positive_weight = np.clip(
            negatives / np.maximum(positives, 1.0), 1.0, 100.0
        )
        self.audit_rows = self._build_audit(delta, activity)
        return self

    def _build_audit(
        self,
        delta: np.ndarray,
        activity: np.ndarray,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for column, record in enumerate(self.layout.core_records):
            values = delta[:, column]
            nonzero = values[activity[:, column]]
            quantiles = (
                np.percentile(nonzero, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
                if len(nonzero)
                else np.full(7, np.nan)
            )
            rows.append(
                {
                    "index": int(column),
                    "category": str(record["category"]),
                    "scope": str(record["scope"]),
                    "owner_id": int(record["owner_id"]),
                    "mechanism": str(record["mechanism"]),
                    "variable": str(record["variable"]),
                    "kind": str(record["kind"]),
                    "physical_domain": _physical_domain(record),
                    "sample_count": int(len(values)),
                    "nonzero_count": int(len(nonzero)),
                    "zero_fraction": float(1.0 - activity[:, column].mean()),
                    "update_fraction": float(activity[:, column].mean()),
                    "sparse": bool(self.sparse_mask[column]),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "median": float(np.median(values)),
                    "mad": float(1.4826 * np.median(np.abs(values - np.median(values)))),
                    "nonzero_p01": float(quantiles[0]),
                    "nonzero_p05": float(quantiles[1]),
                    "nonzero_p25": float(quantiles[2]),
                    "nonzero_p50": float(quantiles[3]),
                    "nonzero_p75": float(quantiles[4]),
                    "nonzero_p95": float(quantiles[5]),
                    "nonzero_p99": float(quantiles[6]),
                    "normalization_center": float(self.delta_center[column]),
                    "normalization_scale": float(self.delta_scale[column]),
                }
            )
        return rows

    def normalize_state(self, raw: np.ndarray) -> np.ndarray:
        return (self.transform(raw) - self.state_center) / self.state_scale

    def delta_and_activity(
        self,
        raw_t: np.ndarray,
        raw_t1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        delta = self.transform(raw_t1) - self.transform(raw_t)
        activity = np.abs(delta) > self.config.activity_epsilon
        normalized = (delta - self.delta_center) / self.delta_scale
        return normalized.astype(np.float32), activity

    def reconstruct(
        self,
        raw_t: np.ndarray,
        normalized_delta: np.ndarray,
        *,
        activity_probability: Optional[np.ndarray] = None,
        apply_hurdle: bool = True,
        synapse_mode: str = "hurdle",
        activity_threshold: float = 0.5,
    ) -> np.ndarray:
        delta = normalized_delta * self.delta_scale + self.delta_center
        if apply_hurdle and activity_probability is not None:
            active = activity_probability >= float(activity_threshold)
            delta = np.where(self.sparse_mask & ~active, 0.0, delta)
        synapse_slice = self.layout.category_slices["synapse_states"]
        if synapse_mode == "exclude":
            delta[..., synapse_slice] = 0.0
        return self.inverse(self.transform(raw_t) + delta)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.config.__dict__, sort_keys=True).encode())
        for values in (
            self.transform_codes,
            self.state_center,
            self.state_scale,
            self.delta_center,
            self.delta_scale,
            self.sparse_mask,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "02b-zero-inflated-v1",
            "fit_split": "train",
            "config": self.config.__dict__,
            "fingerprint": self.fingerprint(),
            "transform_codes": self.transform_codes.tolist(),
            "state_center": self.state_center.tolist(),
            "state_scale": self.state_scale.tolist(),
            "delta_center": self.delta_center.tolist(),
            "delta_scale": self.delta_scale.tolist(),
            "update_fraction": self.update_fraction.tolist(),
            "sparse_mask": self.sparse_mask.astype(int).tolist(),
            "activity_positive_weight": self.activity_positive_weight.tolist(),
            "inverse_numeric_guard": [-30.0, 30.0],
        }


class ReconditionedAuxiliaryNormalizer:
    """Per-variable privileged normalization with explicit applicability."""

    def __init__(self, minimum_scale: float = 1e-8) -> None:
        self.minimum_scale = float(minimum_scale)
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.applicable: Optional[np.ndarray] = None
        self.layout: Dict[str, slice] = {}
        self.audit_rows: List[Dict[str, Any]] = []

    def fit(
        self,
        values: np.ndarray,
        layout: Mapping[str, slice],
        *,
        privileged_records: Sequence[Mapping[str, Any]],
    ) -> "ReconditionedAuxiliaryNormalizer":
        values = np.asarray(values, dtype=np.float64)
        self.layout = dict(layout)
        self.applicable = np.isfinite(values)
        self.center = np.zeros(values.shape[1], dtype=np.float64)
        self.scale = np.full(
            values.shape[1], self.minimum_scale, dtype=np.float64
        )
        for column in range(values.shape[1]):
            selected = values[self.applicable[:, column], column]
            if not len(selected):
                continue
            center = float(np.median(selected))
            self.center[column] = center
            self.scale[column] = float(
                _safe_scale(
                    selected[:, None],
                    center=np.asarray([center]),
                    minimum=self.minimum_scale,
                )[0]
            )
        current_slice = self.layout["currents_conductances_t_plus_1"]
        rows: List[Dict[str, Any]] = []
        for column in range(values.shape[1]):
            selected = values[self.applicable[:, column], column]
            nonzero = selected[np.abs(selected) > 1e-12]
            quantiles = (
                np.percentile(nonzero, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
                if len(nonzero)
                else np.full(7, np.nan)
            )
            block = next(
                name for name, bounds in self.layout.items()
                if bounds.start <= column < bounds.stop
            )
            record = (
                privileged_records[column]
                if column < current_slice.stop
                else None
            )
            rows.append(
                {
                    "index": int(column),
                    "block": block,
                    "mechanism": str(record["mechanism"]) if record else "dense_auxiliary",
                    "variable": str(record["variable"]) if record else block,
                    "physical_domain": (
                        "nonnegative"
                        if record and "conductance" in str(record["kind"])
                        else "unbounded"
                    ),
                    "sample_count": int(len(selected)),
                    "applicable_count": int(len(selected)),
                    "nonzero_count": int(len(nonzero)),
                    "zero_fraction": float(1.0 - len(nonzero) / max(1, len(selected))),
                    "minimum": float(np.min(selected)) if len(selected) else np.nan,
                    "maximum": float(np.max(selected)) if len(selected) else np.nan,
                    "median": float(np.median(selected)) if len(selected) else np.nan,
                    "mad": float(_mad(selected[:, None])[0]) if len(selected) else np.nan,
                    "nonzero_p01": float(quantiles[0]),
                    "nonzero_p05": float(quantiles[1]),
                    "nonzero_p25": float(quantiles[2]),
                    "nonzero_p50": float(quantiles[3]),
                    "nonzero_p75": float(quantiles[4]),
                    "nonzero_p95": float(quantiles[5]),
                    "nonzero_p99": float(quantiles[6]),
                    "normalization_center": float(self.center[column]),
                    "normalization_scale": float(self.scale[column]),
                }
            )
        self.audit_rows = rows
        return self

    def transform(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(values)
        normalized = (np.where(mask, values, self.center) - self.center) / self.scale
        return normalized.astype(np.float32), mask

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.minimum_scale).encode())
        digest.update(np.ascontiguousarray(self.center).tobytes())
        digest.update(np.ascontiguousarray(self.scale).tobytes())
        digest.update(
            json.dumps(
                {key: [value.start, value.stop] for key, value in self.layout.items()},
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "02b-privileged-v1",
            "fit_split": "train",
            "minimum_scale": self.minimum_scale,
            "fingerprint": self.fingerprint(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "layout": {
                key: [value.start, value.stop] for key, value in self.layout.items()
            },
        }


def distribution_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compact category/block summary kept next to the variable-level table."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = str(row.get("category", row.get("block", "unknown")))
        grouped.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        result.append(
            {
                "family": key,
                "variable_count": len(values),
                "median_zero_fraction": float(
                    np.median([float(row["zero_fraction"]) for row in values])
                ),
                "completely_static_count": int(
                    sum(float(row["zero_fraction"]) == 1.0 for row in values)
                ),
                "sparse_count": int(sum(bool(row.get("sparse", False)) for row in values)),
                "minimum_normalization_scale": float(
                    min(float(row["normalization_scale"]) for row in values)
                ),
                "maximum_normalization_scale": float(
                    max(float(row["normalization_scale"]) for row in values)
                ),
            }
        )
    return result
