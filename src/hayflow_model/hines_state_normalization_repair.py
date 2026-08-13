"""Notebook-05i train-only semantic repair of HayFlow teacher-state scales."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_representation_forensics import HinesRepresentationForensics
from .hines_layer import require_torch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05H_ARCHIVE_SHA256 = (
    "a471b49740154239821de9f9f71096e78bfcd1a9866026500615bae6b6c9524d"
)
EXPECTED_05H_INDEX_SHA256 = (
    "ceefd6faf219ea8d09d77bce3e596f5c23975e1a1165e9c95beb2dd966e3e538"
)
EXPECTED_05H_FINAL_SHA256 = (
    "b41c32a3836152054bba1ab45bc59e642af029653f841aee7630663dffa75c2d"
)


@dataclass(frozen=True)
class HinesStateNormalizationRepairConfig:
    """Pre-registered train-only scale repair and input-support gates."""

    pooling_quantile: float = 0.25
    exact_group_multiplier: float = 0.25
    mechanism_group_multiplier: float = 0.15
    category_group_multiplier: float = 0.10
    transform_group_multiplier: float = 0.05
    identity_absolute_floor: float = 1e-3
    voltage_absolute_floor_mv: float = 1.0
    synapse_time_absolute_floor_ms: float = 1.0
    log1p_absolute_floor: float = 1e-4
    logit_absolute_floor: float = 3e-1
    baseline_minimum_scale: float = 1e-8
    clipping_threshold: float = 8.0
    standardized_maximum: float = 100.0
    maximum_clipping_fraction: float = 0.01
    h2_raw_norm_ratio_maximum: float = 100.0
    h2_standardized_maximum: float = 100.0
    h2_maximum_clipping_fraction: float = 0.01
    feature_epsilon: float = 1e-6
    physical_voltage_absolute_maximum_mv: float = 200.0
    top_coordinate_count: int = 200

    def validate(self) -> None:
        positive = (
            self.exact_group_multiplier,
            self.mechanism_group_multiplier,
            self.category_group_multiplier,
            self.transform_group_multiplier,
            self.identity_absolute_floor,
            self.voltage_absolute_floor_mv,
            self.synapse_time_absolute_floor_ms,
            self.log1p_absolute_floor,
            self.logit_absolute_floor,
            self.baseline_minimum_scale,
            self.clipping_threshold,
            self.standardized_maximum,
            self.h2_raw_norm_ratio_maximum,
            self.h2_standardized_maximum,
            self.feature_epsilon,
            self.physical_voltage_absolute_maximum_mv,
            self.top_coordinate_count,
        )
        if min(positive) <= 0:
            raise ValueError("05i scale-repair configuration values must be positive")
        if not 0.0 < self.pooling_quantile <= 1.0:
            raise ValueError("pooling_quantile must be in (0, 1]")
        for name in ("maximum_clipping_fraction", "h2_maximum_clipping_fraction"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesStateNormalizationRepairConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def _record_key(record: Mapping[str, Any], transform_code: int, tier: str) -> Tuple[Any, ...]:
    category = str(record["category"])
    mechanism = str(record["mechanism"])
    variable = str(record["variable"])
    if tier == "exact":
        return category, mechanism, variable, int(transform_code)
    if tier == "mechanism":
        return category, mechanism, int(transform_code)
    if tier == "category":
        return category, int(transform_code)
    if tier == "transform":
        return (int(transform_code),)
    raise ValueError(f"unknown pooling tier: {tier}")


def semantic_state_scale_repair(
    records: Sequence[Mapping[str, Any]],
    transform_codes: np.ndarray,
    original_scale: np.ndarray,
    config: HinesStateNormalizationRepairConfig,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Lift degenerate scales using only semantically related train-fit scales.

    The routine never examines development or held-out values.  A coordinate
    receives the strongest available floor in this order: exact variable,
    mechanism, category, transform family, and finally a registered absolute
    transform-aware floor.
    """

    config.validate()
    codes = np.asarray(transform_codes, dtype=np.int8)
    scale = np.asarray(original_scale, dtype=np.float64)
    if len(records) != len(codes) or scale.shape != codes.shape:
        raise ValueError("records, transform codes, and state scale must align")
    finite_positive = np.isfinite(scale) & (scale > config.baseline_minimum_scale * 1.000001)
    tiers = (
        ("exact", config.exact_group_multiplier),
        ("mechanism", config.mechanism_group_multiplier),
        ("category", config.category_group_multiplier),
        ("transform", config.transform_group_multiplier),
    )
    pools: Dict[str, Dict[Tuple[Any, ...], List[float]]] = {
        tier: {} for tier, _ in tiers
    }
    for index, record in enumerate(records):
        if not finite_positive[index]:
            continue
        for tier, _ in tiers:
            pools[tier].setdefault(
                _record_key(record, int(codes[index]), tier), []
            ).append(float(scale[index]))

    def absolute_floor(record: Mapping[str, Any], code: int) -> float:
        if code == 2:
            # The registered logit clips raw gates to [1e-6, 1-1e-6], whose
            # complete transformed span is about 27.63.  A 0.3 scale floor
            # therefore bounds even an endpoint-to-endpoint excursion below
            # the pre-registered |z|=100 input-support limit.
            return float(config.logit_absolute_floor)
        if code == 1:
            return float(config.log1p_absolute_floor)
        if str(record["category"]) == "voltage":
            return float(config.voltage_absolute_floor_mv)
        if (
            str(record["category"]) == "synapse_states"
            and str(record["variable"]) == "tsyn"
        ):
            return float(config.synapse_time_absolute_floor_ms)
        return float(config.identity_absolute_floor)
    repaired = scale.copy()
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        candidates: List[Tuple[str, float, int]] = [
            (
                "absolute_semantic_floor",
                absolute_floor(record, int(codes[index])),
                0,
            )
        ]
        for tier, multiplier in tiers:
            values = pools[tier].get(
                _record_key(record, int(codes[index]), tier), []
            )
            if values:
                pooled = float(np.quantile(values, config.pooling_quantile))
                candidates.append((f"{tier}_train_pool", multiplier * pooled, len(values)))
        source, semantic_floor, support_count = max(candidates, key=lambda item: item[1])
        repaired[index] = max(float(scale[index]), semantic_floor)
        rows.append({
            "state_index": int(index),
            "category": str(record["category"]),
            "scope": str(record["scope"]),
            "owner_id": int(record["owner_id"]),
            "mechanism": str(record["mechanism"]),
            "variable": str(record["variable"]),
            "kind": str(record["kind"]),
            "transform_code": int(codes[index]),
            "transform_name": {
                0: "identity",
                1: "log1p_nonnegative",
                2: "logit_clip_1e-6",
            }[int(codes[index])],
            "original_scale": float(scale[index]),
            "semantic_floor": float(semantic_floor),
            "repaired_scale": float(repaired[index]),
            "scale_lifted": bool(repaired[index] > scale[index]),
            "floor_source": source,
            "floor_support_coordinate_count": int(support_count),
        })
    if not np.all(np.isfinite(repaired)) or np.any(repaired <= 0.0):
        raise RuntimeError("semantic scale repair produced an invalid scale")
    return repaired, rows


class HinesStateNormalizationRepair(HinesRepresentationForensics):
    """05i state-coordinate forensics and frozen-H2 input-contract repair."""

    def __init__(self, *args: Any, repair_config: HinesStateNormalizationRepairConfig,
                 artifact_05h_source: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        repair_config.validate()
        self.repair = repair_config
        self.artifact_05h_source = Path(artifact_05h_source).resolve()
        self.artifact_05h_contract: Dict[str, Any] = {}
        self.original_state_scale: Optional[np.ndarray] = None
        self.repaired_state_scale: Optional[np.ndarray] = None
        self.coordinate_rows: List[Dict[str, Any]] = []
        self.repair_roles: Dict[str, np.ndarray] = {}

    @staticmethod
    def _one_suffix(names: Sequence[str], suffix: str) -> str:
        matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one 05h {suffix}, found {matches}")
        return matches[0]

    def _read_verified_05h(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05h_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05H_ARCHIVE_SHA256:
                raise RuntimeError("05h archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                index_bytes = archive.read(index_name)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05H_INDEX_SHA256:
                    raise RuntimeError("05h artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                verified: Dict[str, str] = {}
                for row in index["artifacts"]:
                    member = self._one_suffix(names, str(row["path"]))
                    payload = archive.read(member)
                    observed = hashlib.sha256(payload).hexdigest()
                    if observed != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05h indexed member mismatch: {row['path']}")
                    verified[str(row["path"])] = observed
                final_name = self._one_suffix(names, "final_report.json")
                final_bytes = archive.read(final_name)
            kind = "original_zip"
        elif source.is_dir():
            indices = list(source.rglob("artifact_index.json"))
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05h artifact_index.json")
            index_bytes = indices[0].read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05H_INDEX_SHA256:
                raise RuntimeError("extracted 05h artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            root = indices[0].parent
            verified = {}
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file():
                    raise RuntimeError(f"missing extracted 05h member: {row['path']}")
                observed = sha256_file(path)
                if observed != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05h member mismatch: {row['path']}")
                verified[str(row["path"])] = observed
            final_bytes = (root / "final_report.json").read_bytes()
            archive_hash = None
            kind = "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05h source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05H_FINAL_SHA256:
            raise RuntimeError("05h final report SHA-256 mismatch")
        report = json.loads(final_bytes)
        contract = {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05H_INDEX_SHA256,
            "verified_member_count": len(verified),
            "all_indexed_members_verified": len(verified) == int(index["artifact_count"]),
            "final_report_sha256": EXPECTED_05H_FINAL_SHA256,
        }
        return report, contract

    def prepare_scale_repair(self) -> Dict[str, Any]:
        base = self.prepare_forensics()
        report, contract = self._read_verified_05h()
        blockers = []
        if report.get("diagnosis") != "FROZEN_H2_HELDOUT_INPUT_OOD":
            blockers.append(f"unexpected 05h diagnosis: {report.get('diagnosis')}")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05h dataset fingerprint mismatch")
        heldout = report.get("heldout_contract", {})
        if heldout.get("boundary_targets_materialized") or heldout.get("event_targets_materialized"):
            blockers.append("05h held-out target contract was not sealed")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05h artifact verification is incomplete")
        if blockers:
            raise RuntimeError(f"05i provenance blockers: {blockers}")
        self.artifact_05h_contract = contract
        payload = {
            "schema_version": "05i-state-normalization-repair-config-v1",
            "repair": asdict(self.repair),
            "artifact_05h": contract,
            "fit_split": "train",
            "development_or_heldout_used_to_fit": False,
            "teacher_state_modified": False,
            "heldout_targets_materialized": False,
            "candidate_head_inference_performed": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "state_normalization_repair_config.json", payload)
        return {**base, **payload}

    def _role_indices(self) -> Dict[str, np.ndarray]:
        return {
            "fit_train": self._stratified_indices(
                "train", seed=501, per_episode=4,
                limit=self.config.normalization_transitions,
            ),
            "audit_train": np.asarray(
                self._pair_indices(self.audit_plan["selected_pairs"]), dtype=np.int64
            ),
            "development": np.asarray(
                self.audit_plan["development_pair"], dtype=np.int64
            ),
            "heldout_input": np.asarray(
                self._pair_indices(self.artifact_05f_report["pair_plan"]["heldout_pairs"]),
                dtype=np.int64,
            ),
        }

    def run_coordinate_scale_repair(self) -> Dict[str, Any]:
        if self.normalizer is None:
            raise RuntimeError("prepare_scale_repair() must run first")
        self.repair_roles = self._role_indices()
        original = np.asarray(self.normalizer.state_scale, dtype=np.float64).copy()
        repaired, rows = semantic_state_scale_repair(
            self.layout.core_records,
            self.normalizer.transform_codes,
            original,
            self.repair,
        )
        self.original_state_scale = original
        self.repaired_state_scale = repaired
        repair_digest = hashlib.sha256()
        repair_digest.update(json.dumps(asdict(self.repair), sort_keys=True).encode())
        for values in (
            self.normalizer.transform_codes,
            self.normalizer.state_center,
            repaired,
        ):
            repair_digest.update(np.ascontiguousarray(values).tobytes())
        repaired_fingerprint = repair_digest.hexdigest()
        role_reports: Dict[str, Any] = {}
        role_coordinate: Dict[str, Dict[str, np.ndarray]] = {}
        progress = Progress("state support audit", len(self.repair_roles))
        for position, (role, indices) in enumerate(self.repair_roles.items(), start=1):
            raw = self.store.read_state(indices, "t")
            transformed = self.normalizer.transform(raw)
            original_z = (transformed - self.normalizer.state_center) / original
            repaired_z = (transformed - self.normalizer.state_center) / repaired
            role_coordinate[role] = {
                "raw_min": np.min(raw, axis=0),
                "raw_max": np.max(raw, axis=0),
                "transformed_min": np.min(transformed, axis=0),
                "transformed_max": np.max(transformed, axis=0),
                "original_max_abs_z": np.max(np.abs(original_z), axis=0),
                "repaired_max_abs_z": np.max(np.abs(repaired_z), axis=0),
                "repaired_clip_fraction": np.mean(
                    np.abs(repaired_z) > self.repair.clipping_threshold, axis=0
                ),
            }
            role_reports[role] = {
                "transition_count": int(len(indices)),
                "logical_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "raw_nonfinite_count": int(raw.size - np.isfinite(raw).sum()),
                "transformed_nonfinite_count": int(
                    transformed.size - np.isfinite(transformed).sum()
                ),
                "original_standardized_nonfinite_count": int(
                    original_z.size - np.isfinite(original_z).sum()
                ),
                "repaired_standardized_nonfinite_count": int(
                    repaired_z.size - np.isfinite(repaired_z).sum()
                ),
                "original_maximum_absolute_standardized": float(
                    np.max(np.abs(original_z))
                ),
                "repaired_maximum_absolute_standardized": float(
                    np.max(np.abs(repaired_z))
                ),
                "repaired_clipping_fraction_at_threshold": float(
                    np.mean(np.abs(repaired_z) > self.repair.clipping_threshold)
                ),
            }
            progress.update(position, f"{role} max|z|={role_reports[role]['repaired_maximum_absolute_standardized']:.4g}")

        for index, row in enumerate(rows):
            segment_id = int(self.layout.core_segment_ids[index])
            segment = self.layout.segments[segment_id]
            row.update({
                "segment_id": segment_id,
                "region": str(segment.get("region", "unknown")),
                "section_name": str(
                    segment.get("section", segment.get("section_name", ""))
                ),
                "synapse_id": (
                    int(row["owner_id"]) if row["scope"] == "synapse" else None
                ),
            })
            for role, arrays in role_coordinate.items():
                for name, values in arrays.items():
                    row[f"{role}_{name}"] = float(values[index])
        self.coordinate_rows = rows
        write_parquet(self.output_dir / "state_coordinate_scale_audit.parquet", rows)

        group_rows: List[Dict[str, Any]] = []
        group_keys = ("category", "mechanism", "variable", "transform_code", "floor_source")
        for key in group_keys:
            for value in sorted({str(row[key]) for row in rows}):
                selected = [row for row in rows if str(row[key]) == value]
                group_rows.append({
                    "group_kind": key,
                    "group_value": value,
                    "coordinate_count": len(selected),
                    "lifted_coordinate_count": sum(row["scale_lifted"] for row in selected),
                    "minimum_repaired_scale": float(min(row["repaired_scale"] for row in selected)),
                    "heldout_original_maximum_absolute_standardized": float(max(row["heldout_input_original_max_abs_z"] for row in selected)),
                    "heldout_repaired_maximum_absolute_standardized": float(max(row["heldout_input_repaired_max_abs_z"] for row in selected)),
                    "heldout_repaired_clipping_fraction_maximum": float(max(row["heldout_input_repaired_clip_fraction"] for row in selected)),
                })
        write_parquet(self.output_dir / "state_scale_group_audit.parquet", group_rows)
        top = sorted(
            rows,
            key=lambda row: row["heldout_input_original_max_abs_z"],
            reverse=True,
        )[: self.repair.top_coordinate_count]
        write_parquet(self.output_dir / "state_scale_top_outliers.parquet", top)

        heldout_report = role_reports["heldout_input"]
        state_contract = bool(
            all(
                row["raw_nonfinite_count"] == 0
                and row["transformed_nonfinite_count"] == 0
                and row["repaired_standardized_nonfinite_count"] == 0
                and row["repaired_maximum_absolute_standardized"]
                <= self.repair.standardized_maximum
                and row["repaired_clipping_fraction_at_threshold"]
                <= self.repair.maximum_clipping_fraction
                for row in role_reports.values()
            )
        )
        report = {
            "schema_version": "05i-coordinate-scale-repair-v1",
            "valid": True,
            "state_input_contract_passed": state_contract,
            "fit_split": "train",
            "development_or_heldout_used_to_fit": False,
            "teacher_state_modified": False,
            "repaired_normalizer_fingerprint": repaired_fingerprint,
            "state_width": int(len(rows)),
            "original_minimum_scale_count": int(
                np.sum(original <= self.repair.baseline_minimum_scale * 1.000001)
            ),
            "lifted_coordinate_count": int(np.sum(repaired > original)),
            "minimum_original_scale": float(original.min()),
            "minimum_repaired_scale": float(repaired.min()),
            "roles": role_reports,
            "thresholds": {
                "standardized_maximum": self.repair.standardized_maximum,
                "clipping_threshold": self.repair.clipping_threshold,
                "maximum_clipping_fraction": self.repair.maximum_clipping_fraction,
            },
            "heldout_improvement_factor": float(
                heldout_report["original_maximum_absolute_standardized"]
                / max(heldout_report["repaired_maximum_absolute_standardized"], 1e-12)
            ),
            "top_original_outliers": top,
            "heldout_boundary_targets_materialized": False,
            "heldout_event_targets_materialized": False,
        }
        self.normalizer.state_scale = repaired.copy()
        _write_json(self.output_dir / "coordinate_scale_repair.json", report)
        _write_json(self.output_dir / "repaired_normalization_schema.json", {
            "schema_version": "05i-repaired-normalization-v1",
            "fit_split": "train",
            "teacher_transform_unchanged": True,
            "teacher_state_modified": False,
            "repaired_normalizer_fingerprint": repaired_fingerprint,
            "transform_codes": self.normalizer.transform_codes.tolist(),
            "state_center": self.normalizer.state_center.tolist(),
            "original_state_scale": original.tolist(),
            "repaired_state_scale": repaired.tolist(),
            "delta_center_unchanged": self.normalizer.delta_center.tolist(),
            "delta_scale_unchanged": self.normalizer.delta_scale.tolist(),
            "repair_config": asdict(self.repair),
        })
        return report

    @staticmethod
    def _surface(values: np.ndarray) -> Dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        norms = np.linalg.norm(values, axis=-1)
        return {
            "maximum_absolute": float(np.max(np.abs(values))),
            "maximum_segment_norm": float(np.max(norms)),
            "median_segment_norm": float(np.median(norms)),
            "nonfinite_count": int(values.size - np.isfinite(values).sum()),
        }

    def run_repaired_frozen_h2_audit(self) -> Dict[str, Any]:
        if self.repaired_state_scale is None:
            raise RuntimeError("run_coordinate_scale_repair() must run first")
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = self._load_h2_checkpoint(device)
        model.eval()
        role_map = {
            "train": self.repair_roles["audit_train"],
            "development": self.repair_roles["development"],
            "heldout_input": self.repair_roles["heldout_input"],
        }
        extracted: Dict[str, Any] = {}
        progress = Progress("repaired frozen H2", len(role_map))
        for position, (role, indices) in enumerate(role_map.items(), start=1):
            extracted[role] = self._extract_with_model(model, indices, False)
            if extracted[role]["target"] is not None:
                raise RuntimeError("05i target-sealing contract violated")
            progress.update(position, role)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        surfaces = ("h2_raw", "h2_zero_causal_raw")
        report_roles: Dict[str, Any] = {role: {} for role in extracted}
        for surface in surfaces:
            train = extracted["train"][surface]
            mean = train.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(
                train.std(axis=(0, 1), keepdims=True), self.repair.feature_epsilon
            )
            train_max_norm = max(self._surface(train)["maximum_segment_norm"], 1e-12)
            for role, values in extracted.items():
                raw = values[surface]
                standardized = (raw - mean) / scale
                surface_report = {
                    "raw": self._surface(raw),
                    "standardized_unclipped": self._surface(standardized),
                    "clipping_fraction_at_threshold": float(
                        np.mean(np.abs(standardized) > self.repair.clipping_threshold)
                    ),
                    "raw_max_norm_to_train_ratio": float(
                        self._surface(raw)["maximum_segment_norm"] / train_max_norm
                    ),
                }
                report_roles[role][surface] = surface_report
        for role, values in extracted.items():
            report_roles[role]["teacher_state_normalized"] = self._surface(
                values["teacher_state_normalized"]
            )
            report_roles[role]["physical_voltage"] = {
                "state_maximum_absolute_mv": float(np.max(np.abs(values["voltage_t"]))),
                "h2_boundary_maximum_absolute_mv": float(np.max(np.abs(values["base"]))),
                "zero_causal_boundary_maximum_absolute_mv": float(
                    np.max(np.abs(values["zero_causal_base"]))
                ),
            }
        heldout = report_roles["heldout_input"]
        h2_contract = bool(
            all(
                heldout[surface]["raw"]["nonfinite_count"] == 0
                and heldout[surface]["raw_max_norm_to_train_ratio"]
                <= self.repair.h2_raw_norm_ratio_maximum
                and heldout[surface]["standardized_unclipped"]["maximum_absolute"]
                <= self.repair.h2_standardized_maximum
                and heldout[surface]["clipping_fraction_at_threshold"]
                <= self.repair.h2_maximum_clipping_fraction
                for surface in surfaces
            )
            and all(
                value <= self.repair.physical_voltage_absolute_maximum_mv
                for value in heldout["physical_voltage"].values()
            )
        )
        report = {
            "schema_version": "05i-repaired-frozen-h2-audit-v1",
            "valid": True,
            "repaired_h2_input_contract_passed": h2_contract,
            "roles": report_roles,
            "thresholds": {
                "raw_norm_ratio_maximum": self.repair.h2_raw_norm_ratio_maximum,
                "standardized_maximum": self.repair.h2_standardized_maximum,
                "clipping_threshold": self.repair.clipping_threshold,
                "maximum_clipping_fraction": self.repair.h2_maximum_clipping_fraction,
                "physical_voltage_absolute_maximum_mv": self.repair.physical_voltage_absolute_maximum_mv,
            },
            "base_h2_frozen": True,
            "zero_causal_control_performed": True,
            "heldout_boundary_targets_materialized": False,
            "heldout_event_targets_materialized": False,
            "heldout_candidate_head_inference_performed": False,
        }
        _write_json(self.output_dir / "repaired_frozen_h2_audit.json", report)
        return report

    def finalize_scale_repair(
        self, coordinate_report: Mapping[str, Any], h2_report: Mapping[str, Any]
    ) -> Dict[str, Any]:
        passed = bool(
            coordinate_report["state_input_contract_passed"]
            and h2_report["repaired_h2_input_contract_passed"]
        )
        diagnosis = (
            "STATE_NORMALIZATION_INPUT_CONTRACT_REPAIRED"
            if passed else "STATE_NORMALIZATION_REPAIR_INSUFFICIENT"
        )
        self._plot_scale_repair(coordinate_report, h2_report)
        report = {
            "schema_version": "05i-final-report-v1",
            "valid": True,
            "decision": "INPUT_CONTRACT_ONLY_NO_HEAD_TRAINING",
            "diagnosis": diagnosis,
            "input_contract_passed": passed,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05h": self.artifact_05h_contract,
            "coordinate_scale_repair": dict(coordinate_report),
            "repaired_frozen_h2_audit": dict(h2_report),
            "heldout_contract": {
                "boundary_targets_materialized": False,
                "event_targets_materialized": False,
                "frozen_h2_feature_extraction_performed": True,
                "candidate_head_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "normalization_fit_split": "train",
                "development_or_heldout_used_to_fit": False,
                "teacher_transform_unchanged": True,
                "teacher_state_modified": False,
                "semantic_hierarchical_scale_floors": True,
                "pre_clipping_support_audited": True,
                "base_h2_frozen": True,
                "zero_causal_control_performed": True,
                "candidate_head_training_performed": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": (
                    "05j_repaired_representation_train_development_recheck"
                    if passed else "05i_revision_semantic_scale_policy"
                ),
                "full_training_authorized": False,
            },
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append({
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05i-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report

    def _plot_scale_repair(
        self, coordinate_report: Mapping[str, Any], h2_report: Mapping[str, Any]
    ) -> None:
        import matplotlib.pyplot as plt

        top = coordinate_report["top_original_outliers"][:20]
        labels = [f"{row['mechanism']}:{row['variable']}\n#{row['state_index']}" for row in top]
        original = [row["heldout_input_original_max_abs_z"] for row in top]
        repaired = [row["heldout_input_repaired_max_abs_z"] for row in top]
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
        positions = np.arange(len(top))
        axes[0].bar(positions - 0.2, original, 0.4, label="original")
        axes[0].bar(positions + 0.2, repaired, 0.4, label="repaired")
        axes[0].set_yscale("log")
        axes[0].set_xticks(positions, labels, rotation=90, fontsize=7)
        axes[0].set(title="Top held-out state coordinates", ylabel="max |standardized| (log)")
        axes[0].legend()
        role_names = list(coordinate_report["roles"])
        axes[1].bar(
            role_names,
            [coordinate_report["roles"][role]["repaired_maximum_absolute_standardized"] for role in role_names],
        )
        axes[1].axhline(self.repair.standardized_maximum, color="black", linestyle="--")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set(title="Repaired state support", ylabel="max |standardized|")
        surfaces = ("h2_raw", "h2_zero_causal_raw")
        axes[2].bar(
            ["H2", "H2 zero U"],
            [h2_report["roles"]["heldout_input"][surface]["raw_max_norm_to_train_ratio"] for surface in surfaces],
        )
        axes[2].axhline(self.repair.h2_raw_norm_ratio_maximum, color="black", linestyle="--")
        axes[2].set_yscale("log")
        axes[2].set(title="Held-out/train frozen-H2 norm", ylabel="ratio (log)")
        for axis in axes:
            axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        figure.savefig(self.output_dir / "state_normalization_repair.png", dpi=160)
        plt.close(figure)
