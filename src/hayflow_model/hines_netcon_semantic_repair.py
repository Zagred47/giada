"""Notebook-05i-b class-aware NetCon state and relative-time repair."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_teacher.neuron_manifest import KNOWN_NET_RECEIVE_STATE_LAYOUT
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_state_normalization_repair import HinesStateNormalizationRepair


EXPECTED_05I_ARCHIVE_SHA256 = (
    "76f94225937e8946c3142753604b0d6eb6c30771dba5d64970099dff83952943"
)
EXPECTED_05I_INDEX_SHA256 = (
    "d0c3023f734b219302571b34bbb31e687677b94a8ddccc21db67b13789623780"
)
EXPECTED_05I_FINAL_SHA256 = (
    "15912aa1e503f63bfc4d073d8f6c7e6deb63f68130af8298b9cd09bc9feec792"
)


@dataclass(frozen=True)
class HinesNetConSemanticRepairConfig:
    tsyn_representation: str = "boundary_relative_age_ms"
    roundtrip_atol: float = 1e-9
    negative_age_atol: float = 1e-9
    probability_domain_atol: float = 1e-9
    nonnegative_domain_atol: float = 1e-12
    expected_synapse_count_per_class: int = 639
    expected_netcon_coordinate_count: int = 6390
    expected_tsyn_coordinate_count: int = 1278
    expected_probability_coordinate_count: int = 3834
    expected_amplitude_coordinate_count: int = 1278

    def validate(self) -> None:
        if self.tsyn_representation != "boundary_relative_age_ms":
            raise ValueError("05i-b requires boundary_relative_age_ms")
        if min(
            self.roundtrip_atol,
            self.negative_age_atol,
            self.probability_domain_atol,
            self.nonnegative_domain_atol,
            self.expected_synapse_count_per_class,
            self.expected_netcon_coordinate_count,
            self.expected_tsyn_coordinate_count,
            self.expected_probability_coordinate_count,
            self.expected_amplitude_coordinate_count,
        ) <= 0:
            raise ValueError("05i-b semantic-contract values must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesNetConSemanticRepairConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def netcon_semantic_records(layout: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Decode raw NetCon weight slots using each synapse point-process class."""

    patterns: Dict[str, Dict[str, str]] = {}
    for point_process, slots in KNOWN_NET_RECEIVE_STATE_LAYOUT.items():
        patterns[point_process] = {
            f"weight[{int(slot)}]": str(name) for name, slot in slots
        }
    records: List[Dict[str, Any]] = []
    mapped = []
    unmapped = []
    class_counts: Dict[str, int] = {}
    semantic_counts: Dict[str, int] = {}
    for state_index, original in enumerate(layout.core_records):
        row = dict(original)
        row["stored_mechanism"] = str(original["mechanism"])
        row["stored_variable"] = str(original["variable"])
        if str(original["mechanism"]) == "NetCon":
            synapse_id = int(original["owner_id"])
            point_process = str(layout.synapse_type[synapse_id])
            semantic = patterns.get(point_process, {}).get(str(original["variable"]))
            if semantic is None:
                unmapped.append({
                    "state_index": state_index,
                    "synapse_id": synapse_id,
                    "point_process_class": point_process,
                    "stored_variable": str(original["variable"]),
                })
            else:
                row["point_process_class"] = point_process
                row["mechanism"] = f"NetCon[{point_process}]"
                row["variable"] = semantic
                class_counts[point_process] = class_counts.get(point_process, 0) + 1
                semantic_counts[semantic] = semantic_counts.get(semantic, 0) + 1
                mapped.append({
                    "state_index": state_index,
                    "synapse_id": synapse_id,
                    "point_process_class": point_process,
                    "stored_variable": str(original["variable"]),
                    "semantic_variable": semantic,
                })
        records.append(row)
    report = {
        "mapped_coordinate_count": len(mapped),
        "unmapped_coordinate_count": len(unmapped),
        "class_coordinate_counts": class_counts,
        "semantic_coordinate_counts": semantic_counts,
        "mappings": mapped,
        "unmapped": unmapped,
    }
    return records, report


class NetConSemanticStateEncoder:
    """Causal reversible view of raw teacher state at a known boundary time."""

    PROBABILITY_NAMES = frozenset({"Pv", "Pr", "u"})
    AMPLITUDE_NAMES = frozenset({"weight_AMPA", "weight_NMDA"})

    def __init__(self, layout: Any, config: HinesNetConSemanticRepairConfig) -> None:
        config.validate()
        self.layout = layout
        self.config = config
        self.records, self.mapping_report = netcon_semantic_records(layout)
        self.tsyn_indices = np.asarray(
            [i for i, row in enumerate(self.records) if row.get("variable") == "tsyn"],
            dtype=np.int64,
        )
        self.probability_indices = np.asarray(
            [
                i for i, row in enumerate(self.records)
                if row.get("variable") in self.PROBABILITY_NAMES
                and str(row.get("stored_mechanism")) == "NetCon"
            ],
            dtype=np.int64,
        )
        self.amplitude_indices = np.asarray(
            [
                i for i, row in enumerate(self.records)
                if row.get("variable") in self.AMPLITUDE_NAMES
            ],
            dtype=np.int64,
        )

    @staticmethod
    def _times(values: Sequence[float], rows: int) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (rows,):
            raise ValueError("boundary times must align with state rows")
        if not np.all(np.isfinite(result)):
            raise ValueError("boundary times must be finite")
        return result

    def encode(self, raw_state: np.ndarray, boundary_time_ms: Sequence[float]) -> np.ndarray:
        raw = np.asarray(raw_state, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.records):
            raise ValueError("raw teacher state has the wrong shape")
        times = self._times(boundary_time_ms, len(raw))
        result = raw.copy()
        if len(self.tsyn_indices):
            age = times[:, None] - raw[:, self.tsyn_indices]
            if np.min(age) < -self.config.negative_age_atol:
                raise ValueError("NetCon tsyn lies after the causal boundary")
            result[:, self.tsyn_indices] = np.maximum(age, 0.0)
        return result

    def decode(
        self, semantic_state: np.ndarray, boundary_time_ms: Sequence[float]
    ) -> np.ndarray:
        semantic = np.asarray(semantic_state, dtype=np.float64)
        if semantic.ndim != 2 or semantic.shape[1] != len(self.records):
            raise ValueError("semantic teacher state has the wrong shape")
        times = self._times(boundary_time_ms, len(semantic))
        result = semantic.copy()
        if len(self.tsyn_indices):
            result[:, self.tsyn_indices] = (
                times[:, None] - semantic[:, self.tsyn_indices]
            )
        return result

    def configure_transform_codes(self, normalizer: Any) -> None:
        codes = np.asarray(normalizer.transform_codes, dtype=np.int8).copy()
        codes[self.probability_indices] = int(normalizer.LOGIT)
        codes[self.amplitude_indices] = int(normalizer.LOG1P)
        codes[self.tsyn_indices] = int(normalizer.LOG1P)
        normalizer.transform_codes = codes


class HinesNetConSemanticRepair(HinesStateNormalizationRepair):
    """05i-b class-aware NetCon semantic and relative-time input audit."""

    def __init__(
        self,
        *args: Any,
        netcon_config: HinesNetConSemanticRepairConfig,
        artifact_05i_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        netcon_config.validate()
        self.netcon = netcon_config
        self.artifact_05i_source = Path(artifact_05i_source).resolve()
        self.netcon_encoder = NetConSemanticStateEncoder(self.layout, netcon_config)
        self.artifact_05i_contract: Dict[str, Any] = {}

    def _normalization_records(self) -> Sequence[Mapping[str, Any]]:
        return self.netcon_encoder.records

    def _configure_state_normalizer(self, normalizer: Any) -> None:
        self.netcon_encoder.configure_transform_codes(normalizer)

    def _boundary_times(
        self, indices: Sequence[int], boundary: str
    ) -> np.ndarray:
        if boundary not in {"t", "t_plus_1"}:
            raise ValueError("boundary must be t or t_plus_1")
        ordered = np.asarray(indices, dtype=np.int64)
        times = np.asarray(
            self.store.metadata["start_time_ms"][ordered], dtype=np.float64
        )
        if boundary == "t_plus_1":
            times = times + float(self.config.model.dt_ms)
        return times

    def _state_input_view(
        self,
        raw_state: np.ndarray,
        indices: Sequence[int],
        boundary: str,
    ) -> np.ndarray:
        return self.netcon_encoder.encode(
            raw_state, self._boundary_times(indices, boundary)
        )

    @staticmethod
    def _one_suffix(names: Sequence[str], suffix: str) -> str:
        matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one 05i {suffix}, found {matches}")
        return matches[0]

    def _read_verified_05i(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05i_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05I_ARCHIVE_SHA256:
                raise RuntimeError("05i archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                archive_root = index_name[: -len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05I_INDEX_SHA256:
                    raise RuntimeError("05i artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                verified = {}
                for row in index["artifacts"]:
                    member = archive_root + str(row["path"]).replace("\\", "/")
                    if member not in names:
                        raise RuntimeError(f"missing indexed 05i member: {row['path']}")
                    payload = archive.read(member)
                    observed = hashlib.sha256(payload).hexdigest()
                    if observed != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05i indexed member mismatch: {row['path']}")
                    verified[str(row["path"])] = observed
                final_bytes = archive.read(archive_root + "final_report.json")
            kind = "original_zip"
        elif source.is_dir():
            indices = list(source.rglob("artifact_index.json"))
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05i artifact_index.json")
            index_bytes = indices[0].read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05I_INDEX_SHA256:
                raise RuntimeError("extracted 05i artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            root = indices[0].parent
            verified = {}
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file():
                    raise RuntimeError(f"missing extracted 05i member: {row['path']}")
                observed = sha256_file(path)
                if observed != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05i member mismatch: {row['path']}")
                verified[str(row["path"])] = observed
            final_bytes = (root / "final_report.json").read_bytes()
            archive_hash = None
            kind = "kaggle_extracted_directory"
        else:
            raise RuntimeError(f"05i source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05I_FINAL_SHA256:
            raise RuntimeError("05i final report SHA-256 mismatch")
        report = json.loads(final_bytes)
        contract = {
            "source_kind": kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05I_INDEX_SHA256,
            "verified_member_count": len(verified),
            "all_indexed_members_verified": len(verified) == int(index["artifact_count"]),
            "final_report_sha256": EXPECTED_05I_FINAL_SHA256,
        }
        return report, contract

    def prepare_netcon_semantic_repair(self) -> Dict[str, Any]:
        base = self.prepare_scale_repair()
        report, contract = self._read_verified_05i()
        mapping = self.netcon_encoder.mapping_report
        class_synapse_counts = {
            name: sum(1 for value in self.layout.synapse_type.values() if value == name)
            for name in KNOWN_NET_RECEIVE_STATE_LAYOUT
        }
        inherited_thresholds = {
            "coordinate_scale_repair": {
                "standardized_maximum": self.repair.standardized_maximum,
                "clipping_threshold": self.repair.clipping_threshold,
                "maximum_clipping_fraction": self.repair.maximum_clipping_fraction,
            },
            "repaired_frozen_h2_audit": {
                "standardized_maximum": self.repair.h2_standardized_maximum,
                "clipping_threshold": self.repair.clipping_threshold,
                "maximum_clipping_fraction": self.repair.h2_maximum_clipping_fraction,
                "physical_voltage_absolute_maximum_mv": (
                    self.repair.physical_voltage_absolute_maximum_mv
                ),
                "raw_norm_ratio_maximum": self.repair.h2_raw_norm_ratio_maximum,
            },
        }
        source_thresholds = {
            name: report.get(name, {}).get("thresholds", {})
            for name in inherited_thresholds
        }
        thresholds_unchanged = all(
            source_thresholds[name] == values
            for name, values in inherited_thresholds.items()
        )
        blockers = []
        if report.get("diagnosis") != "STATE_NORMALIZATION_REPAIR_INSUFFICIENT":
            blockers.append(f"unexpected 05i diagnosis: {report.get('diagnosis')}")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05i dataset fingerprint mismatch")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05i artifact verification is incomplete")
        if not thresholds_unchanged:
            blockers.append(
                "05i-b decision thresholds differ from the verified 05i artifact"
            )
        if mapping["unmapped_coordinate_count"]:
            blockers.append("one or more NetCon slots have no known semantic mapping")
        if mapping["mapped_coordinate_count"] != self.netcon.expected_netcon_coordinate_count:
            blockers.append("unexpected NetCon coordinate count")
        if len(self.netcon_encoder.tsyn_indices) != self.netcon.expected_tsyn_coordinate_count:
            blockers.append("unexpected tsyn coordinate count")
        if len(self.netcon_encoder.probability_indices) != self.netcon.expected_probability_coordinate_count:
            blockers.append("unexpected NetCon probability coordinate count")
        if len(self.netcon_encoder.amplitude_indices) != self.netcon.expected_amplitude_coordinate_count:
            blockers.append("unexpected NetCon amplitude coordinate count")
        if any(
            count != self.netcon.expected_synapse_count_per_class
            for count in class_synapse_counts.values()
        ):
            blockers.append(f"unexpected synapse class counts: {class_synapse_counts}")
        if blockers:
            raise RuntimeError(f"05i-b provenance or mapping blockers: {blockers}")
        self.artifact_05i_contract = contract
        payload = {
            "schema_version": "05i-b-netcon-semantic-config-v1",
            "netcon": asdict(self.netcon),
            "artifact_05i": contract,
            "class_synapse_counts": class_synapse_counts,
            "mapping_counts": {
                key: mapping[key]
                for key in (
                    "mapped_coordinate_count", "unmapped_coordinate_count",
                    "class_coordinate_counts", "semantic_coordinate_counts",
                )
            },
            "threshold_contract": {
                "source_05i": source_thresholds,
                "current_05i_b": inherited_thresholds,
                "unchanged": thresholds_unchanged,
            },
            "thresholds_inherited_unchanged": thresholds_unchanged,
            "teacher_snapshot_modified": False,
            "heldout_targets_materialized": False,
            "candidate_head_inference_performed": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "netcon_semantic_repair_config.json", payload)
        return {**base, **payload}

    def run_netcon_semantic_roundtrip_audit(self) -> Dict[str, Any]:
        roles = self._role_indices()
        role_reports = {}
        progress = Progress("NetCon semantic roundtrip", len(roles))
        probability_min = np.inf
        probability_max = -np.inf
        amplitude_min = np.inf
        maximum_roundtrip = 0.0
        minimum_age = np.inf
        for position, (role, indices) in enumerate(roles.items(), start=1):
            raw = self.store.read_state(indices, "t")
            times = self._boundary_times(indices, "t")
            semantic = self.netcon_encoder.encode(raw, times)
            restored = self.netcon_encoder.decode(semantic, times)
            error = float(np.max(np.abs(restored - raw)))
            age = semantic[:, self.netcon_encoder.tsyn_indices]
            probability = raw[:, self.netcon_encoder.probability_indices]
            amplitude = raw[:, self.netcon_encoder.amplitude_indices]
            maximum_roundtrip = max(maximum_roundtrip, error)
            minimum_age = min(minimum_age, float(age.min()))
            probability_min = min(probability_min, float(probability.min()))
            probability_max = max(probability_max, float(probability.max()))
            amplitude_min = min(amplitude_min, float(amplitude.min()))
            role_reports[role] = {
                "transition_count": int(len(indices)),
                "roundtrip_maximum_absolute_error": error,
                "boundary_time_minimum_ms": float(times.min()),
                "boundary_time_maximum_ms": float(times.max()),
                "tsyn_raw_minimum_ms": float(
                    raw[:, self.netcon_encoder.tsyn_indices].min()
                ),
                "tsyn_raw_maximum_ms": float(
                    raw[:, self.netcon_encoder.tsyn_indices].max()
                ),
                "age_minimum_ms": float(age.min()),
                "age_maximum_ms": float(age.max()),
                "age_median_ms": float(np.median(age)),
                "recent_age_fraction_le_1ms": float(np.mean(age <= 1.0)),
                "recent_age_fraction_le_80ms": float(np.mean(age <= 80.0)),
                "probability_minimum": float(probability.min()),
                "probability_maximum": float(probability.max()),
                "amplitude_minimum": float(amplitude.min()),
            }
            progress.update(position, f"{role} roundtrip={error:.3g}")
        mapping_passed = bool(
            self.netcon_encoder.mapping_report["unmapped_coordinate_count"] == 0
            and maximum_roundtrip <= self.netcon.roundtrip_atol
            and minimum_age >= -self.netcon.negative_age_atol
            and probability_min >= -self.netcon.probability_domain_atol
            and probability_max <= 1.0 + self.netcon.probability_domain_atol
            and amplitude_min >= -self.netcon.nonnegative_domain_atol
        )
        report = {
            "schema_version": "05i-b-netcon-semantic-roundtrip-v1",
            "valid": True,
            "mapping_contract_passed": mapping_passed,
            "tsyn_representation": self.netcon.tsyn_representation,
            "causal_boundary_time_source": "transition metadata start_time_ms",
            "inverse": "tsyn_ms = boundary_time_ms - age_ms",
            "maximum_roundtrip_absolute_error": maximum_roundtrip,
            "minimum_age_ms": minimum_age,
            "probability_domain": [probability_min, probability_max],
            "minimum_amplitude_weight": amplitude_min,
            "roles": role_reports,
            "mapping": self.netcon_encoder.mapping_report,
            "teacher_snapshot_modified": False,
            "heldout_boundary_targets_materialized": False,
            "heldout_event_targets_materialized": False,
        }
        _write_json(self.output_dir / "netcon_semantic_roundtrip_audit.json", report)
        return report

    def finalize_netcon_semantic_repair(
        self,
        mapping_report: Mapping[str, Any],
        coordinate_report: Mapping[str, Any],
        h2_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base = super().finalize_scale_repair(coordinate_report, h2_report)
        passed = bool(
            mapping_report["mapping_contract_passed"]
            and coordinate_report["state_input_contract_passed"]
            and h2_report["repaired_h2_input_contract_passed"]
        )
        report = {
            **base,
            "schema_version": "05i-b-final-report-v1",
            "decision": "SEMANTIC_INPUT_CONTRACT_ONLY_NO_HEAD_TRAINING",
            "diagnosis": (
                "NETCON_SEMANTIC_INPUT_CONTRACT_REPAIRED"
                if passed else "NETCON_SEMANTIC_INPUT_REPAIR_INSUFFICIENT"
            ),
            "input_contract_passed": passed,
            "full_training_authorized": False,
            "artifact_05i": self.artifact_05i_contract,
            "netcon_semantic_roundtrip": dict(mapping_report),
            "heldout_contract": {
                "boundary_targets_materialized": False,
                "event_targets_materialized": False,
                "frozen_h2_feature_extraction_performed": True,
                "candidate_head_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                **base["methodology"],
                "netcon_slots_decoded_by_point_process": True,
                "tsyn_boundary_relative_and_causal": True,
                "semantic_mapping_reversible": True,
                "thresholds_relaxed": False,
                "global_multiplier_only_patch": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": (
                    "05j_repaired_representation_train_development_recheck"
                    if passed else "05i_c_netcon_semantic_revision"
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
            "schema_version": "05i-b-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
