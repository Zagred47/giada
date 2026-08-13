"""Notebook-05i-c bounded recency and synaptic-domain input repair."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_netcon_semantic_repair import (
    HinesNetConSemanticRepair,
    HinesNetConSemanticRepairConfig,
    netcon_semantic_records,
)
from .hines_state_normalization_repair import HinesStateNormalizationRepair


EXPECTED_05IB_ARCHIVE_SHA256 = (
    "7b0c1f97d325a1edf8bdf9d57b5c85188a1cd0561be2e959956fb1eeea152b62"
)
EXPECTED_05IB_INDEX_SHA256 = (
    "a1084f12e85d51bab2079af1c4995e523f2766c92e7e2a2c358a5137ade2fb20"
)
EXPECTED_05IB_FINAL_SHA256 = (
    "dd97b7336b1dd23d38c5d39046acb600dc04fdf5926336fc1974aea5feb9b72b"
)


_DYNAMIC_TRACE_NAMES = {
    "ProbAMPANMDA2": frozenset({"A_AMPA", "B_AMPA", "A_NMDA", "B_NMDA"}),
    "ProbUDFsyn2": frozenset({"A", "B"}),
}


@dataclass(frozen=True)
class HinesSynapticDomainRepairConfig:
    """Preregistered 05i-c representation and domain-scale contract."""

    tsyn_representation: str = "bounded_rational_recency"
    minimum_recency_time_ms: float = 1.0
    bounded_recency_target_standardized_span: float = 50.0
    trace_reference_raw_increment: float = 1.0
    trace_reference_target_standardized_span: float = 35.0
    roundtrip_atol: float = 1e-9
    recency_domain_atol: float = 1e-12
    expected_tsyn_coordinate_count: int = 1278
    expected_dynamic_trace_coordinate_count: int = 3834

    @property
    def bounded_recency_scale_floor(self) -> float:
        return 1.0 / self.bounded_recency_target_standardized_span

    @property
    def synaptic_trace_log1p_scale_floor(self) -> float:
        return (
            math.log1p(self.trace_reference_raw_increment)
            / self.trace_reference_target_standardized_span
        )

    def validate(self) -> None:
        if self.tsyn_representation != "bounded_rational_recency":
            raise ValueError("05i-c requires bounded_rational_recency")
        positive = (
            self.minimum_recency_time_ms,
            self.bounded_recency_target_standardized_span,
            self.trace_reference_raw_increment,
            self.trace_reference_target_standardized_span,
            self.roundtrip_atol,
            self.recency_domain_atol,
            self.expected_tsyn_coordinate_count,
            self.expected_dynamic_trace_coordinate_count,
        )
        if min(positive) <= 0:
            raise ValueError("05i-c domain-repair values must be positive")
        if self.bounded_recency_target_standardized_span >= 100.0:
            raise ValueError("bounded recency must retain margin below the |z|=100 gate")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesSynapticDomainRepairConfig":
        result = cls(**dict(values))
        result.validate()
        return result


class BoundedSynapticStateEncoder:
    """Causal reversible state view with bounded last-event recency."""

    PROBABILITY_NAMES = frozenset({"Pv", "Pr", "u"})
    AMPLITUDE_NAMES = frozenset({"weight_AMPA", "weight_NMDA"})

    def __init__(
        self,
        layout: Any,
        netcon_config: HinesNetConSemanticRepairConfig,
        domain_config: HinesSynapticDomainRepairConfig,
    ) -> None:
        netcon_config.validate()
        domain_config.validate()
        self.layout = layout
        self.netcon_config = netcon_config
        self.config = domain_config
        records, mapping = netcon_semantic_records(layout)
        self.records: List[Dict[str, Any]] = []
        for original in records:
            row = dict(original)
            if (
                str(row.get("stored_mechanism")) == "NetCon"
                and str(row.get("variable")) == "tsyn"
            ):
                row["source_semantic_variable"] = "tsyn"
                row["variable"] = "last_event_recency"
                row["unit"] = "dimensionless"
            self.records.append(row)
        self.mapping_report = mapping
        self.recency_indices = np.asarray(
            [
                index for index, row in enumerate(self.records)
                if str(row.get("variable")) == "last_event_recency"
            ],
            dtype=np.int64,
        )
        self.probability_indices = np.asarray(
            [
                index for index, row in enumerate(self.records)
                if str(row.get("stored_mechanism")) == "NetCon"
                and str(row.get("variable")) in self.PROBABILITY_NAMES
            ],
            dtype=np.int64,
        )
        self.amplitude_indices = np.asarray(
            [
                index for index, row in enumerate(self.records)
                if str(row.get("variable")) in self.AMPLITUDE_NAMES
            ],
            dtype=np.int64,
        )
        self.trace_indices = np.asarray(
            [
                index for index, row in enumerate(self.records)
                if str(row.get("mechanism")) in _DYNAMIC_TRACE_NAMES
                and str(row.get("variable"))
                in _DYNAMIC_TRACE_NAMES[str(row.get("mechanism"))]
            ],
            dtype=np.int64,
        )
        synapses = {int(row["id"]): row for row in getattr(layout, "synapses", [])}
        recency_time = []
        for state_index in self.recency_indices:
            synapse_id = int(self.records[int(state_index)]["owner_id"])
            parameters = dict(synapses.get(synapse_id, {}).get("parameters", {}))
            candidates = [float(domain_config.minimum_recency_time_ms)]
            for name in ("Dep", "Fac"):
                value = float(parameters.get(name, 0.0) or 0.0)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"invalid {name} for synapse {synapse_id}: {value}")
                if value > 0.0:
                    candidates.append(value)
            recency_time.append(max(candidates))
        self.recency_time_ms = np.asarray(recency_time, dtype=np.float64)
        if len(self.recency_time_ms) and (
            not np.all(np.isfinite(self.recency_time_ms))
            or np.min(self.recency_time_ms) <= 0.0
        ):
            raise ValueError("recency reference times must be finite and positive")

    @staticmethod
    def _times(values: Sequence[float], rows: int) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (rows,) or not np.all(np.isfinite(result)):
            raise ValueError("boundary times must be finite and align with state rows")
        return result

    def encode(self, raw_state: np.ndarray, boundary_time_ms: Sequence[float]) -> np.ndarray:
        raw = np.asarray(raw_state, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.records):
            raise ValueError("raw teacher state has the wrong shape")
        times = self._times(boundary_time_ms, len(raw))
        result = raw.copy()
        if len(self.recency_indices):
            age = times[:, None] - raw[:, self.recency_indices]
            if np.min(age) < -self.netcon_config.negative_age_atol:
                raise ValueError("NetCon tsyn lies after the causal boundary")
            age = np.maximum(age, 0.0)
            tau = self.recency_time_ms[None, :]
            result[:, self.recency_indices] = tau / (tau + age)
        return result

    def decode(
        self, semantic_state: np.ndarray, boundary_time_ms: Sequence[float]
    ) -> np.ndarray:
        semantic = np.asarray(semantic_state, dtype=np.float64)
        if semantic.ndim != 2 or semantic.shape[1] != len(self.records):
            raise ValueError("semantic teacher state has the wrong shape")
        times = self._times(boundary_time_ms, len(semantic))
        result = semantic.copy()
        if len(self.recency_indices):
            recency = semantic[:, self.recency_indices]
            tolerance = self.config.recency_domain_atol
            if np.min(recency) <= 0.0 or np.max(recency) > 1.0 + tolerance:
                raise ValueError("bounded recency must lie in (0, 1]")
            recency = np.minimum(recency, 1.0)
            tau = self.recency_time_ms[None, :]
            age = tau * (1.0 / recency - 1.0)
            result[:, self.recency_indices] = times[:, None] - age
        return result

    def configure_transform_codes(self, normalizer: Any) -> None:
        codes = np.asarray(normalizer.transform_codes, dtype=np.int8).copy()
        codes[self.probability_indices] = int(normalizer.LOGIT)
        codes[self.amplitude_indices] = int(normalizer.LOG1P)
        codes[self.trace_indices] = int(normalizer.LOG1P)
        codes[self.recency_indices] = int(normalizer.IDENTITY)
        normalizer.transform_codes = codes


class HinesSynapticDomainRepair(HinesNetConSemanticRepair):
    """05i-c bounded recency and domain-calibrated synaptic input audit."""

    def __init__(
        self,
        *args: Any,
        domain_config: HinesSynapticDomainRepairConfig,
        artifact_05ib_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        domain_config.validate()
        self.domain = domain_config
        self.artifact_05ib_source = Path(artifact_05ib_source).resolve()
        self.netcon_encoder = BoundedSynapticStateEncoder(
            self.layout, self.netcon, domain_config
        )
        self.artifact_05ib_contract: Dict[str, Any] = {}

    def _repair_state_scales(
        self, original_scale: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        repaired, rows = super()._repair_state_scales(original_scale)
        domain_floors = {
            "bounded_recency_domain_floor": (
                self.netcon_encoder.recency_indices,
                self.domain.bounded_recency_scale_floor,
            ),
            "synaptic_trace_reference_floor": (
                self.netcon_encoder.trace_indices,
                self.domain.synaptic_trace_log1p_scale_floor,
            ),
        }
        for source, (indices, floor) in domain_floors.items():
            for raw_index in indices:
                index = int(raw_index)
                previous_floor = float(rows[index]["semantic_floor"])
                previous_scale = float(repaired[index])
                repaired[index] = max(previous_scale, float(floor))
                rows[index]["registered_domain_floor"] = float(floor)
                if float(floor) > previous_floor:
                    rows[index]["semantic_floor"] = float(floor)
                    rows[index]["floor_source"] = source
                    rows[index]["floor_support_coordinate_count"] = 0
                rows[index]["repaired_scale"] = float(repaired[index])
                rows[index]["scale_lifted"] = bool(
                    repaired[index] > float(original_scale[index])
                )
        return repaired, rows

    @staticmethod
    def _one_suffix(names: Sequence[str], suffix: str) -> str:
        matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one 05i-b {suffix}, found {matches}")
        return matches[0]

    def _read_verified_05ib(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05ib_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05IB_ARCHIVE_SHA256:
                raise RuntimeError("05i-b archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                archive_root = index_name[: -len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05IB_INDEX_SHA256:
                    raise RuntimeError("05i-b artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                verified = {}
                for row in index["artifacts"]:
                    member = archive_root + str(row["path"]).replace("\\", "/")
                    if member not in names:
                        raise RuntimeError(f"missing indexed 05i-b member: {row['path']}")
                    payload = archive.read(member)
                    observed = hashlib.sha256(payload).hexdigest()
                    if observed != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05i-b indexed member mismatch: {row['path']}")
                    verified[str(row["path"])] = observed
                final_bytes = archive.read(archive_root + "final_report.json")
            kind = "original_zip"
        elif source.is_dir():
            indices = list(source.rglob("artifact_index.json"))
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05i-b artifact_index.json")
            index_bytes = indices[0].read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05IB_INDEX_SHA256:
                raise RuntimeError("extracted 05i-b artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            root = indices[0].parent
            verified = {}
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file():
                    raise RuntimeError(f"missing extracted 05i-b member: {row['path']}")
                observed = sha256_file(path)
                if observed != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05i-b member mismatch: {row['path']}")
                verified[str(row["path"])] = observed
            final_bytes = (root / "final_report.json").read_bytes()
            archive_hash = None
            kind = "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05i-b source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05IB_FINAL_SHA256:
            raise RuntimeError("05i-b final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05IB_INDEX_SHA256,
            "verified_member_count": len(verified),
            "all_indexed_members_verified": len(verified) == int(index["artifact_count"]),
            "final_report_sha256": EXPECTED_05IB_FINAL_SHA256,
        }

    def prepare_synaptic_domain_repair(self) -> Dict[str, Any]:
        base = HinesStateNormalizationRepair.prepare_scale_repair(self)
        report, contract = self._read_verified_05ib()
        source_thresholds = {
            name: report.get(name, {}).get("thresholds", {})
            for name in ("coordinate_scale_repair", "repaired_frozen_h2_audit")
        }
        current_thresholds = {
            "coordinate_scale_repair": {
                "standardized_maximum": self.repair.standardized_maximum,
                "clipping_threshold": self.repair.clipping_threshold,
                "maximum_clipping_fraction": self.repair.maximum_clipping_fraction,
            },
            "repaired_frozen_h2_audit": {
                "standardized_maximum": self.repair.h2_standardized_maximum,
                "clipping_threshold": self.repair.clipping_threshold,
                "maximum_clipping_fraction": self.repair.h2_maximum_clipping_fraction,
                "physical_voltage_absolute_maximum_mv": self.repair.physical_voltage_absolute_maximum_mv,
                "raw_norm_ratio_maximum": self.repair.h2_raw_norm_ratio_maximum,
            },
        }
        thresholds_unchanged = source_thresholds == current_thresholds
        blockers = []
        if report.get("diagnosis") != "NETCON_SEMANTIC_INPUT_REPAIR_INSUFFICIENT":
            blockers.append(f"unexpected 05i-b diagnosis: {report.get('diagnosis')}")
        if report.get("input_contract_passed") is not False:
            blockers.append("05i-b did not register the expected failed input contract")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05i-b dataset fingerprint mismatch")
        if not report.get("netcon_semantic_roundtrip", {}).get("mapping_contract_passed"):
            blockers.append("05i-b semantic mapping did not pass")
        if not report.get("repaired_frozen_h2_audit", {}).get("repaired_h2_input_contract_passed"):
            blockers.append("05i-b frozen-H2 contract did not pass")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05i-b artifact verification is incomplete")
        if not thresholds_unchanged:
            blockers.append("05i-c thresholds differ from verified 05i-b")
        if len(self.netcon_encoder.recency_indices) != self.domain.expected_tsyn_coordinate_count:
            blockers.append("unexpected bounded-recency coordinate count")
        if len(self.netcon_encoder.trace_indices) != self.domain.expected_dynamic_trace_coordinate_count:
            blockers.append("unexpected dynamic synaptic-trace coordinate count")
        heldout = report.get("heldout_contract", {})
        if heldout.get("boundary_targets_materialized") or heldout.get("event_targets_materialized"):
            blockers.append("05i-b held-out future-target contract was not sealed")
        if blockers:
            raise RuntimeError(f"05i-c provenance or representation blockers: {blockers}")
        self.artifact_05ib_contract = contract
        payload = {
            "schema_version": "05i-c-synaptic-domain-config-v1",
            "domain_repair": {
                **asdict(self.domain),
                "bounded_recency_scale_floor": self.domain.bounded_recency_scale_floor,
                "synaptic_trace_log1p_scale_floor": self.domain.synaptic_trace_log1p_scale_floor,
            },
            "artifact_05ib": contract,
            "threshold_contract": {
                "source_05ib": source_thresholds,
                "current_05ic": current_thresholds,
                "unchanged": thresholds_unchanged,
            },
            "coordinate_counts": {
                "bounded_recency": int(len(self.netcon_encoder.recency_indices)),
                "dynamic_synaptic_trace": int(len(self.netcon_encoder.trace_indices)),
            },
            "recency_time_ms": {
                "minimum": float(self.netcon_encoder.recency_time_ms.min()),
                "maximum": float(self.netcon_encoder.recency_time_ms.max()),
            },
            "fit_split": "train",
            "development_or_heldout_used_to_fit": False,
            "teacher_snapshot_modified": False,
            "heldout_targets_materialized": False,
            "candidate_head_inference_performed": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "synaptic_domain_repair_config.json", payload)
        return {**base, **payload}

    def run_bounded_recency_roundtrip_audit(self) -> Dict[str, Any]:
        roles = self._role_indices()
        role_reports = {}
        maximum_roundtrip = 0.0
        recency_minimum = np.inf
        recency_maximum = -np.inf
        progress = Progress("bounded recency roundtrip", len(roles))
        for position, (role, indices) in enumerate(roles.items(), start=1):
            raw = self.store.read_state(indices, "t")
            times = self._boundary_times(indices, "t")
            semantic = self.netcon_encoder.encode(raw, times)
            restored = self.netcon_encoder.decode(semantic, times)
            error = float(np.max(np.abs(restored - raw)))
            recency = semantic[:, self.netcon_encoder.recency_indices]
            age = times[:, None] - raw[:, self.netcon_encoder.recency_indices]
            maximum_roundtrip = max(maximum_roundtrip, error)
            recency_minimum = min(recency_minimum, float(recency.min()))
            recency_maximum = max(recency_maximum, float(recency.max()))
            role_reports[role] = {
                "transition_count": int(len(indices)),
                "roundtrip_maximum_absolute_error": error,
                "age_minimum_ms": float(age.min()),
                "age_maximum_ms": float(age.max()),
                "recency_minimum": float(recency.min()),
                "recency_maximum": float(recency.max()),
                "recency_median": float(np.median(recency)),
            }
            progress.update(position, f"{role} roundtrip={error:.3g}")
        passed = bool(
            maximum_roundtrip <= self.domain.roundtrip_atol
            and recency_minimum > 0.0
            and recency_maximum <= 1.0 + self.domain.recency_domain_atol
        )
        report = {
            "schema_version": "05i-c-bounded-recency-roundtrip-v1",
            "valid": True,
            "representation_contract_passed": passed,
            "representation": "tau_ms / (tau_ms + boundary_time_ms - tsyn_ms)",
            "inverse": "tsyn_ms = boundary_time_ms - tau_ms * (1 / recency - 1)",
            "tau_source": "max(minimum_recency_time_ms, positive teacher Dep, positive teacher Fac)",
            "maximum_roundtrip_absolute_error": maximum_roundtrip,
            "recency_domain": [recency_minimum, recency_maximum],
            "roles": role_reports,
            "teacher_snapshot_modified": False,
            "heldout_boundary_targets_materialized": False,
            "heldout_event_targets_materialized": False,
        }
        _write_json(self.output_dir / "bounded_recency_roundtrip_audit.json", report)
        return report

    def run_synaptic_domain_floor_audit(self) -> Dict[str, Any]:
        if not self.coordinate_rows:
            raise RuntimeError("run_coordinate_scale_repair() must run first")
        families = {
            "bounded_recency": (
                self.netcon_encoder.recency_indices,
                self.domain.bounded_recency_scale_floor,
            ),
            "dynamic_synaptic_trace": (
                self.netcon_encoder.trace_indices,
                self.domain.synaptic_trace_log1p_scale_floor,
            ),
        }
        reports = {}
        for name, (indices, floor) in families.items():
            selected = [self.coordinate_rows[int(index)] for index in indices]
            reports[name] = {
                "coordinate_count": len(selected),
                "registered_scale_floor": float(floor),
                "minimum_repaired_scale": float(min(row["repaired_scale"] for row in selected)),
                "heldout_maximum_absolute_standardized": float(
                    max(row["heldout_input_repaired_max_abs_z"] for row in selected)
                ),
                "coordinate_count_above_global_limit": int(
                    sum(
                        row["heldout_input_repaired_max_abs_z"]
                        > self.repair.standardized_maximum
                        for row in selected
                    )
                ),
            }
        passed = bool(
            reports["bounded_recency"]["coordinate_count"]
            == self.domain.expected_tsyn_coordinate_count
            and reports["dynamic_synaptic_trace"]["coordinate_count"]
            == self.domain.expected_dynamic_trace_coordinate_count
            and all(
                row["minimum_repaired_scale"] + 1e-15
                >= row["registered_scale_floor"]
                and row["coordinate_count_above_global_limit"] == 0
                for row in reports.values()
            )
        )
        report = {
            "schema_version": "05i-c-synaptic-domain-floor-audit-v1",
            "valid": True,
            "domain_floor_contract_passed": passed,
            "families": reports,
            "fit_split": "train",
            "development_or_heldout_used_to_fit": False,
            "thresholds_relaxed": False,
            "teacher_snapshot_modified": False,
            "heldout_future_targets_materialized": False,
        }
        _write_json(self.output_dir / "synaptic_domain_floor_audit.json", report)
        return report

    def finalize_synaptic_domain_repair(
        self,
        recency_report: Mapping[str, Any],
        coordinate_report: Mapping[str, Any],
        domain_floor_report: Mapping[str, Any],
        h2_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base = HinesStateNormalizationRepair.finalize_scale_repair(
            self, coordinate_report, h2_report
        )
        passed = bool(
            recency_report["representation_contract_passed"]
            and coordinate_report["state_input_contract_passed"]
            and domain_floor_report["domain_floor_contract_passed"]
            and h2_report["repaired_h2_input_contract_passed"]
        )
        report = {
            **base,
            "schema_version": "05i-c-final-report-v1",
            "decision": "SYNAPTIC_DOMAIN_INPUT_CONTRACT_ONLY_NO_HEAD_TRAINING",
            "diagnosis": (
                "SYNAPTIC_DOMAIN_INPUT_CONTRACT_REPAIRED"
                if passed else "SYNAPTIC_DOMAIN_INPUT_REPAIR_INSUFFICIENT"
            ),
            "input_contract_passed": passed,
            "full_training_authorized": False,
            "artifact_05ib": self.artifact_05ib_contract,
            "bounded_recency_roundtrip": dict(recency_report),
            "synaptic_domain_floor_audit": dict(domain_floor_report),
            "heldout_contract": {
                "boundary_targets_materialized": False,
                "event_targets_materialized": False,
                "frozen_h2_feature_extraction_performed": True,
                "candidate_head_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                **base["methodology"],
                "bounded_causal_recency": True,
                "recency_mapping_reversible": True,
                "recency_time_from_teacher_parameters": True,
                "dynamic_trace_domain_floor": True,
                "domain_floors_use_heldout_values": False,
                "thresholds_relaxed": False,
                "candidate_head_training_performed": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": (
                    "05j_repaired_representation_train_development_recheck"
                    if passed else "05i_d_synaptic_input_representation_revision"
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
            "schema_version": "05i-c-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
