"""Causal data contract for the full-state flow-map baselines.

NEURON is deliberately absent here.  The module validates the immutable
diagnostic bundle, exposes its HDF5 transitions, and constructs inputs and
targets without leaking future microtraces or event labels into the model.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .diagnostic_contract import DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION


EXPECTED_TEACHER_COMMIT = "074c4666300a8ad246601dab179a97a6942f0f29"
DYNAMIC_CATEGORIES = (
    "voltage",
    "mechanism_states",
    "calcium_ions",
    "synapse_states",
)
EVENT_KINDS = (
    "axonal_spike",
    "somatic_spike",
    "backpropagating_ap",
    "calcium_spike",
    "nmda_spike",
    "nmda_plateau",
)
U1_FEATURE_NAMES = (
    "excitatory_count",
    "inhibitory_count",
    "excitatory_weight",
    "inhibitory_weight",
    "excitatory_gmax_weight",
    "inhibitory_gmax_weight",
    "excitatory_time_first_moment_ms",
    "inhibitory_time_first_moment_ms",
    "somatic_current_amplitude_na",
    "somatic_current_duration_ms",
    "somatic_current_offset_ms",
    "total_action_count",
)
U2_EVENT_FEATURE_NAMES = (
    "offset_ms",
    "weight_multiplier",
    "is_excitatory",
    "is_inhibitory",
    "is_somatic_current",
    "gmax_us",
    "release_observed_value",
    "release_observed_available",
    "current_amplitude_na",
    "current_duration_ms",
)


class FlowmapContractError(RuntimeError):
    """Raised when a dataset cannot be used by notebook 02."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_extract_zip(source: Path, destination: Path) -> None:
    destination = Path(destination).resolve()
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if destination != target and destination not in target.parents:
                raise FlowmapContractError(
                    f"unsafe path in dataset archive: {info.filename!r}"
                )
        archive.extractall(destination)


def _find_dataset_root(path: Path) -> Path:
    path = Path(path).resolve()
    if (path / "dataset_manifest.json").is_file():
        return path
    matches = list(path.rglob("dataset_manifest.json"))
    if len(matches) != 1:
        raise FlowmapContractError(
            "dataset source must contain exactly one dataset_manifest.json"
        )
    return matches[0].parent


@dataclass(frozen=True)
class FlowmapBundle:
    """Validated locations and provenance for one diagnostic bundle."""

    root: Path
    transition_path: Path
    manifest: Mapping[str, Any]
    state_schema: Mapping[str, Any]
    teacher_manifest: Mapping[str, Any]
    validation_report: Mapping[str, Any]
    artifact_validation: Mapping[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.manifest["schema_version"])


def prepare_flowmap_bundle(
    source: Path,
    *,
    cache_dir: Optional[Path] = None,
    verify_hashes: bool = True,
) -> FlowmapBundle:
    """Extract and validate the only dataset accepted by notebook 02."""

    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise FlowmapContractError("dataset source must be a ZIP or directory")
        if cache_dir is None:
            raise ValueError("cache_dir is required when the source is a ZIP")
        extraction = Path(cache_dir).resolve()
        marker = extraction / ".source_sha256"
        source_hash = _sha256_file(source)
        if not marker.is_file() or marker.read_text().strip() != source_hash:
            if extraction.exists():
                shutil.rmtree(extraction)
            extraction.mkdir(parents=True, exist_ok=True)
            _safe_extract_zip(source, extraction)
            marker.write_text(source_hash + "\n", encoding="utf-8")
        root = _find_dataset_root(extraction)
    else:
        root = _find_dataset_root(source)

    manifest = _read_json(root / "dataset_manifest.json")
    state_schema = _read_json(root / "state_schema.json")
    validation = _read_json(root / "validation_report.json")
    teacher_manifest = _read_json(root / "manifest.json")
    versions = {
        str(manifest.get("schema_version")),
        str(state_schema.get("schema_version")),
        str(validation.get("schema_version")),
    }
    if versions != {DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION}:
        raise FlowmapContractError(
            "notebook 02 accepts only diagnostic dataset schema "
            f"{DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION}; observed {sorted(versions)}"
        )
    if not bool(validation.get("valid")) or validation.get("blockers"):
        raise FlowmapContractError("dataset validation report is not green")
    if str(manifest.get("teacher_commit")) != EXPECTED_TEACHER_COMMIT:
        raise FlowmapContractError("teacher commit differs from the audited Hay model")
    if int(manifest.get("transition_count", -1)) != 1224:
        raise FlowmapContractError("diagnostic v1.0.1 transition count changed")
    if int(state_schema.get("core_state_width", -1)) != 17220:
        raise FlowmapContractError("core state width differs from the audited contract")
    if int(state_schema.get("privileged_state_width", -1)) != 9182:
        raise FlowmapContractError("privileged state width changed")

    index = _read_json(root / "artifact_index.json")
    missing: List[str] = []
    size_mismatches: List[str] = []
    hash_mismatches: List[str] = []
    for record in index.get("artifacts", []):
        relative = str(record["path"])
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            size_mismatches.append(relative)
        if verify_hashes and _sha256_file(path) != str(record["sha256"]):
            hash_mismatches.append(relative)
    artifact_validation = {
        "valid": not (missing or size_mismatches or hash_mismatches),
        "record_count": len(index.get("artifacts", [])),
        "hashes_checked": bool(verify_hashes),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
    }
    if not artifact_validation["valid"]:
        raise FlowmapContractError(
            f"dataset artifact validation failed: {artifact_validation}"
        )
    transition_path = root / str(manifest["transition_store"])
    if not transition_path.is_file():
        raise FlowmapContractError("transition HDF5 is missing")
    return FlowmapBundle(
        root=root,
        transition_path=transition_path,
        manifest=manifest,
        state_schema=state_schema,
        teacher_manifest=teacher_manifest,
        validation_report=validation,
        artifact_validation=artifact_validation,
    )


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class FlowmapLayout:
    """Semantic indexing shared by the loader and structured model."""

    def __init__(self, bundle: FlowmapBundle) -> None:
        import numpy as np

        self.np = np
        self.bundle = bundle
        self.variables = list(bundle.state_schema["variables"])
        self.segments = list(bundle.teacher_manifest["segments"])
        self.synapses = list(bundle.teacher_manifest["synapses"])
        self.segment_count = len(self.segments)
        self.category_widths = {
            name: int(bundle.state_schema["categories"][name]["width"])
            for name in DYNAMIC_CATEGORIES
        }
        self.category_slices: Dict[str, slice] = {}
        offset = 0
        for category in DYNAMIC_CATEGORIES:
            width = self.category_widths[category]
            self.category_slices[category] = slice(offset, offset + width)
            offset += width
        self.state_width = offset
        self.privileged_width = int(
            bundle.state_schema["categories"]["currents_conductances"]["width"]
        )
        self.synapse_to_segment = {
            int(row["id"]): int(row["segment_id"]) for row in self.synapses
        }
        self.synapse_type = {
            int(row["id"]): str(row["point_process"]) for row in self.synapses
        }
        self.synapse_gmax = {
            int(row["id"]): float(row.get("parameters", {}).get("gmax", 0.0))
            for row in self.synapses
        }
        self.core_records = self._records_for(DYNAMIC_CATEGORIES)
        self.privileged_records = self._records_for(("currents_conductances",))
        self.core_segment_ids = np.asarray(
            [self._record_segment(row) for row in self.core_records], dtype=np.int64
        )
        self.privileged_segment_ids = np.asarray(
            [self._record_segment(row) for row in self.privileged_records],
            dtype=np.int64,
        )
        self.core_category_ids, self.category_names = self._encode_field(
            self.core_records, "category"
        )
        all_records = self.core_records + self.privileged_records
        _, self.mechanism_names = self._encode_field(all_records, "mechanism")
        _, self.variable_names = self._encode_field(all_records, "variable")
        _, self.kind_names = self._encode_field(all_records, "kind")
        self.core_mechanism_ids = self._ids_from_vocab(
            self.core_records, "mechanism", self.mechanism_names
        )
        self.core_variable_ids = self._ids_from_vocab(
            self.core_records, "variable", self.variable_names
        )
        self.core_kind_ids = self._ids_from_vocab(
            self.core_records, "kind", self.kind_names
        )
        self.privileged_mechanism_ids = self._ids_from_vocab(
            self.privileged_records, "mechanism", self.mechanism_names
        )
        self.privileged_variable_ids = self._ids_from_vocab(
            self.privileged_records, "variable", self.variable_names
        )
        self.privileged_kind_ids = self._ids_from_vocab(
            self.privileged_records, "kind", self.kind_names
        )
        self.parent_ids = np.asarray(
            [
                int(row["parent_segment_id"])
                if row.get("parent_segment_id") is not None
                else int(row["id"])
                for row in self.segments
            ],
            dtype=np.int64,
        )
        children: List[List[int]] = [[] for _ in self.segments]
        for row in self.segments:
            parent = row.get("parent_segment_id")
            if parent is not None:
                children[int(parent)].append(int(row["id"]))
        self.children = children
        event_config = _read_json(bundle.root / "event_definition_config.json")
        regions = sorted(
            {str(row["region"]) for row in self.segments}
            | {
                str(row["region"])
                for row in event_config.get("definitions", [])
            }
        )
        self.region_names = regions
        region_index = {name: i for i, name in enumerate(regions)}
        self.segment_region_ids = np.asarray(
            [region_index[str(row["region"])] for row in self.segments],
            dtype=np.int64,
        )
        static = []
        for row in self.segments:
            static.append(
                [
                    math.log1p(float(row["area_um2"])),
                    math.log1p(float(row["length_um"])),
                    math.log1p(float(row["diameter_um"])),
                    math.log1p(max(0.0, float(row["axial_conductance_to_parent_us"]))),
                    math.log1p(max(0.0, float(row["membrane_capacitance_uf"]))),
                    math.log1p(max(0.0, float(row["passive_leak_conductance_us"]))),
                    float(row["passive_reversal_mv"]) / 100.0,
                ]
            )
        static_array = np.asarray(static, dtype=np.float32)
        self.segment_static = (
            static_array - static_array.mean(axis=0, keepdims=True)
        ) / np.maximum(static_array.std(axis=0, keepdims=True), 1e-6)
        self.aux_layout: Dict[str, slice] = {}

    def _records_for(self, categories: Sequence[str]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for category in categories:
            rows = sorted(
                (
                    row
                    for row in self.variables
                    if str(row["category"]) == category
                ),
                key=lambda row: int(row["index"]),
            )
            expected = int(self.bundle.state_schema["categories"][category]["width"])
            if len(rows) != expected:
                raise FlowmapContractError(f"variable index mismatch for {category}")
            selected.extend(rows)
        return selected

    def _record_segment(self, row: Mapping[str, Any]) -> int:
        if row["scope"] == "segment":
            return int(row["owner_id"])
        if row["scope"] == "synapse":
            return self.synapse_to_segment[int(row["owner_id"])]
        raise FlowmapContractError(f"unsupported dynamic scope {row['scope']!r}")

    def _encode_field(
        self, records: Sequence[Mapping[str, Any]], field: str
    ) -> Tuple[Any, List[str]]:
        names = sorted({str(row[field]) for row in records})
        return self._ids_from_vocab(records, field, names), names

    def _ids_from_vocab(
        self,
        records: Sequence[Mapping[str, Any]],
        field: str,
        names: Sequence[str],
    ) -> Any:
        index = {name: i for i, name in enumerate(names)}
        return self.np.asarray(
            [index[str(row[field])] for row in records], dtype=self.np.int64
        )

    def to_model_metadata(self) -> Dict[str, Any]:
        return {
            "state_width": self.state_width,
            "privileged_width": self.privileged_width,
            "segment_count": self.segment_count,
            "category_widths": self.category_widths,
            "category_slices": {
                key: [value.start, value.stop]
                for key, value in self.category_slices.items()
            },
            "category_names": self.category_names,
            "mechanism_names": self.mechanism_names,
            "variable_names": self.variable_names,
            "kind_names": self.kind_names,
            "region_names": self.region_names,
            "u1_feature_names": list(U1_FEATURE_NAMES),
            "u2_event_feature_names": list(U2_EVENT_FEATURE_NAMES),
            "event_kinds": list(EVENT_KINDS),
        }


class StateNormalizer:
    """Train-only semantic transforms and robust delta normalization."""

    IDENTITY = 0
    LOG1P = 1
    LOGIT = 2

    def __init__(self, layout: FlowmapLayout) -> None:
        self.layout = layout
        self.transform_codes = self._semantic_transform_codes()
        self.state_center = None
        self.state_scale = None
        self.delta_center = None
        self.delta_scale = None

    def _semantic_transform_codes(self) -> Any:
        np = self.layout.np
        codes = np.zeros(self.layout.state_width, dtype=np.int8)
        for index, row in enumerate(self.layout.core_records):
            category = str(row["category"])
            variable = str(row["variable"])
            mechanism = str(row["mechanism"])
            if category == "mechanism_states":
                codes[index] = self.LOGIT
            elif category == "calcium_ions":
                codes[index] = self.LOG1P
            elif category == "synapse_states":
                if mechanism == "NetCon" and variable in {"Pv", "Pr", "u"}:
                    codes[index] = self.LOGIT
                elif variable != "tsyn":
                    codes[index] = self.LOG1P
        return codes

    def transform(self, values: Any) -> Any:
        np = self.layout.np
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

    def inverse(self, values: Any) -> Any:
        np = self.layout.np
        result = np.asarray(values, dtype=np.float64).copy()
        log_mask = self.transform_codes == self.LOG1P
        logit_mask = self.transform_codes == self.LOGIT
        if log_mask.any():
            result[..., log_mask] = np.maximum(np.expm1(result[..., log_mask]), 0.0)
        if logit_mask.any():
            x = np.clip(result[..., logit_mask], -30.0, 30.0)
            result[..., logit_mask] = 1.0 / (1.0 + np.exp(-x))
        return result

    @staticmethod
    def _robust_stats(values: Any) -> Tuple[Any, Any]:
        import numpy as np

        center = np.median(values, axis=0)
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
        scale = (q75 - q25) / 1.349
        standard = np.std(values, axis=0)
        scale = np.where(scale > 1e-10, scale, standard)
        scale = np.where(scale > 1e-10, scale, 1.0)
        return center, scale

    def fit(self, state_t: Any, state_t_plus_1: Any) -> "StateNormalizer":
        z_t = self.transform(state_t)
        z_t1 = self.transform(state_t_plus_1)
        self.state_center, self.state_scale = self._robust_stats(z_t)
        self.delta_center, self.delta_scale = self._robust_stats(z_t1 - z_t)
        return self

    def normalize_state(self, raw: Any) -> Any:
        return (self.transform(raw) - self.state_center) / self.state_scale

    def normalize_delta(self, raw_t: Any, raw_t1: Any) -> Any:
        delta = self.transform(raw_t1) - self.transform(raw_t)
        return (delta - self.delta_center) / self.delta_scale

    def reconstruct(self, raw_t: Any, normalized_delta: Any) -> Any:
        delta = normalized_delta * self.delta_scale + self.delta_center
        return self.inverse(self.transform(raw_t) + delta)

    def to_dict(self) -> Dict[str, Any]:
        if self.state_center is None:
            raise RuntimeError("normalizer has not been fit")
        return {
            "fit_split": "train",
            "target": "transformed_delta",
            "reconstruction": "inverse(transform(S_t) + predicted_delta)",
            "transform_codes": self.transform_codes.tolist(),
            "transform_code_names": {
                "0": "identity",
                "1": "log1p_nonnegative",
                "2": "logit_clip_1e-6",
            },
            "state_center": self.state_center.tolist(),
            "state_scale": self.state_scale.tolist(),
            "delta_center": self.delta_center.tolist(),
            "delta_scale": self.delta_scale.tolist(),
        }

    @classmethod
    def from_dict(
        cls, layout: FlowmapLayout, payload: Mapping[str, Any]
    ) -> "StateNormalizer":
        np = layout.np
        result = cls(layout)
        result.transform_codes = np.asarray(payload["transform_codes"], dtype=np.int8)
        for name in ("state_center", "state_scale", "delta_center", "delta_scale"):
            setattr(result, name, np.asarray(payload[name], dtype=np.float64))
        return result


class FlowmapTransitionStore:
    """Lazy HDF5 reader with causal U1/U2 and event-target construction."""

    def __init__(self, bundle: FlowmapBundle, layout: FlowmapLayout) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError as error:
            raise RuntimeError("flow-map loading requires h5py and numpy") from error
        self.h5py = h5py
        self.np = np
        self.bundle = bundle
        self.layout = layout
        self._handle = None
        with h5py.File(bundle.transition_path, "r") as handle:
            self.count = int(handle.attrs["transition_count"])
            self.metadata = {
                name: np.asarray(
                    [_decode(value) for value in handle[f"metadata/{name}"][...]]
                )
                for name in (
                    "trajectory_id",
                    "category",
                    "protocol",
                    "protocol_id",
                    "protocol_variant",
                    "split",
                )
            }
            self.metadata.update(
                {
                    name: handle[f"metadata/{name}"][...]
                    for name in (
                        "transition_id",
                        "seed",
                        "step_index",
                        "start_time_ms",
                        "stimulus_relative_time_ms",
                        "negative_control",
                    )
                }
            )
            self.micro_observable_names = json.loads(
                handle.attrs.get("micro_observable_names_json", "[]")
            )
            self.probe_order = list(bundle.state_schema["probe_order"])
        self.split_indices = {
            split: self.np.flatnonzero(self.metadata["split"] == split)
            for split in sorted(set(self.metadata["split"].tolist()))
        }
        self.trajectory_indices: Dict[str, Any] = {}
        for trajectory in sorted(set(self.metadata["trajectory_id"].tolist())):
            indices = self.np.flatnonzero(
                self.metadata["trajectory_id"] == trajectory
            )
            order = self.np.argsort(self.metadata["step_index"][indices])
            self.trajectory_indices[trajectory] = indices[order]
        self.release_contract = {
            "rng_used_as_regression_target": False,
            "rng_used_as_surrogate_input": False,
            "teacher_rng_preserved_for_replay": True,
            "release_outcome_available": self._scan_release_availability(),
            "surrogate_input_policy": (
                "scheduled ordered events only; release_observed is unavailable"
            ),
        }

    def _open(self) -> Any:
        if self._handle is None:
            self._handle = self.h5py.File(self.bundle.transition_path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def _scan_release_availability(self) -> bool:
        with self.h5py.File(self.bundle.transition_path, "r") as handle:
            for encoded in handle["inputs/ordered_actions_json"][...]:
                for action in json.loads(_decode(encoded)):
                    if action.get("release_observed") is not None:
                        return True
        return False

    def read_state(self, indices: Sequence[int], boundary: str) -> Any:
        handle = self._open()
        rows = []
        ordered = self.np.asarray(indices, dtype=self.np.int64)
        for category in DYNAMIC_CATEGORIES:
            dataset = handle[f"states/{category}/{boundary}"]
            rows.append(self.np.stack([dataset[int(i), :] for i in ordered]))
        return self.np.concatenate(rows, axis=1)

    def read_privileged(self, indices: Sequence[int]) -> Any:
        handle = self._open()
        ordered = self.np.asarray(indices, dtype=self.np.int64)
        return self.np.stack(
            [
                handle["states/currents_conductances/t_plus_1"][int(i), :]
                for i in ordered
            ]
        )

    def _actions(self, index: int) -> List[Dict[str, Any]]:
        return json.loads(
            _decode(self._open()["inputs/ordered_actions_json"][int(index)])
        )

    def _events(self, index: int) -> List[Dict[str, Any]]:
        return json.loads(_decode(self._open()["events/labels_json"][int(index)]))

    def encode_inputs(self, indices: Sequence[int]) -> Dict[str, Any]:
        u1_rows = []
        event_rows: List[List[List[float]]] = []
        event_segments: List[List[int]] = []
        max_events = 1
        for index in indices:
            u1, rich, segments = self._encode_actions(self._actions(int(index)))
            u1_rows.append(u1)
            event_rows.append(rich)
            event_segments.append(segments)
            max_events = max(max_events, len(rich))
        features = self.np.zeros(
            (len(event_rows), max_events, len(U2_EVENT_FEATURE_NAMES)),
            dtype=self.np.float32,
        )
        segments = self.np.zeros((len(event_rows), max_events), dtype=self.np.int64)
        mask = self.np.zeros((len(event_rows), max_events), dtype=bool)
        for row, (values, owner_ids) in enumerate(zip(event_rows, event_segments)):
            if values:
                features[row, : len(values), :] = self.np.asarray(values)
                segments[row, : len(values)] = owner_ids
                mask[row, : len(values)] = True
        return {
            "u1": self.np.asarray(u1_rows, dtype=self.np.float32),
            "u2_features": features,
            "u2_segment_ids": segments,
            "u2_mask": mask,
        }

    def _encode_actions(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> Tuple[Any, List[List[float]], List[int]]:
        np = self.np
        u1 = np.zeros(
            (self.layout.segment_count, len(U1_FEATURE_NAMES)), dtype=np.float32
        )
        rich: List[List[float]] = []
        segment_ids: List[int] = []
        time_weight = np.zeros((self.layout.segment_count, 2), dtype=np.float64)
        weight_sum = np.zeros((self.layout.segment_count, 2), dtype=np.float64)
        for action in actions:
            kind = str(action["kind"])
            if kind == "somatic_current":
                segment_id = 0
                amplitude = float(action.get("amplitude_na") or 0.0)
                duration = float(action.get("duration_ms") or 0.0)
                offset = float(action["offset_ms"])
                u1[segment_id, 8:11] = (amplitude, duration, offset)
                u1[segment_id, 11] += 1.0
                rich.append(
                    [offset, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, amplitude, duration]
                )
                segment_ids.append(segment_id)
                continue
            synapse_id = int(action["synapse_id"])
            segment_id = self.layout.synapse_to_segment[synapse_id]
            excitatory = self.layout.synapse_type[synapse_id] == "ProbAMPANMDA2"
            type_index = 0 if excitatory else 1
            weight = float(action.get("weight_multiplier", 1.0))
            offset = float(action["offset_ms"])
            gmax = float(self.layout.synapse_gmax[synapse_id])
            u1[segment_id, type_index] += 1.0
            u1[segment_id, 2 + type_index] += weight
            u1[segment_id, 4 + type_index] += weight * gmax
            time_weight[segment_id, type_index] += weight * offset
            weight_sum[segment_id, type_index] += weight
            u1[segment_id, 11] += 1.0
            release = action.get("release_observed")
            rich.append(
                [
                    offset,
                    weight,
                    float(excitatory),
                    float(not excitatory),
                    0.0,
                    gmax,
                    float(bool(release)) if release is not None else 0.0,
                    float(release is not None),
                    0.0,
                    0.0,
                ]
            )
            segment_ids.append(segment_id)
        for type_index in (0, 1):
            valid = weight_sum[:, type_index] > 0.0
            u1[valid, 6 + type_index] = (
                time_weight[valid, type_index] / weight_sum[valid, type_index]
            )
        return u1, rich, segment_ids

    def event_targets(self, indices: Sequence[int]) -> Dict[str, Any]:
        kind_index = {name: i for i, name in enumerate(EVENT_KINDS)}
        region_index = {
            name: i for i, name in enumerate(self.layout.region_names)
        }
        presence = self.np.zeros((len(indices), len(EVENT_KINDS)), dtype=self.np.float32)
        timing = self.np.zeros((len(indices), len(EVENT_KINDS), 4), dtype=self.np.float32)
        timing_mask = self.np.zeros_like(timing, dtype=bool)
        regions = self.np.zeros((len(indices), len(EVENT_KINDS)), dtype=self.np.int64)
        region_mask = self.np.zeros_like(regions, dtype=bool)
        for row, index in enumerate(indices):
            start = float(self.metadata["start_time_ms"][int(index)])
            by_kind: Dict[str, Mapping[str, Any]] = {}
            for event in self._events(int(index)):
                kind = str(event["kind"])
                if kind not in kind_index:
                    continue
                previous = by_kind.get(kind)
                if previous is None or float(event["onset_ms"]) < float(
                    previous["onset_ms"]
                ):
                    by_kind[kind] = event
            for kind, event in by_kind.items():
                column = kind_index[kind]
                presence[row, column] = 1.0
                timing[row, column] = (
                    float(event["onset_ms"]) - start,
                    float(event["peak_ms"]) - start,
                    float(event["offset_ms"]) - start,
                    float(event["duration_ms"]),
                )
                timing_mask[row, column, :2] = True
                if not bool(event.get("right_censored", False)):
                    timing_mask[row, column, 2:] = True
                region = str(event.get("region", ""))
                if region in region_index:
                    regions[row, column] = region_index[region]
                    region_mask[row, column] = True
        return {
            "event_presence": presence,
            "event_timing": timing,
            "event_timing_mask": timing_mask,
            "event_region": regions,
            "event_region_mask": region_mask,
        }

    def auxiliary_targets(self, indices: Sequence[int]) -> Tuple[Any, Dict[str, slice]]:
        handle = self._open()
        rows = self.np.asarray(indices, dtype=self.np.int64)
        blocks: List[Any] = []
        layout: Dict[str, slice] = {}
        offset = 0

        def add(name: str, values: Any) -> None:
            nonlocal offset
            flat = self.np.asarray(values, dtype=self.np.float32).reshape(len(rows), -1)
            blocks.append(flat)
            layout[name] = slice(offset, offset + flat.shape[1])
            offset += flat.shape[1]

        add("currents_conductances_t_plus_1", self.read_privileged(rows))
        add(
            "probe_voltage_microtrace",
            self.np.stack(
                [handle["microtraces/probe_voltage"][int(i)] for i in rows]
            ),
        )
        if "protocol_observables" in handle["microtraces"]:
            add(
                "protocol_observables_microtrace",
                self.np.stack(
                    [
                        handle["microtraces/protocol_observables"][int(i)]
                        for i in rows
                    ]
                ),
            )
        for name in ("minimum_mv", "maximum_mv", "integral_mv_ms"):
            add(
                f"all_segment_voltage_{name}",
                self.np.stack(
                    [
                        handle[
                            f"microtraces/all_segment_voltage_summary/{name}"
                        ][int(i)]
                        for i in rows
                    ]
                ),
            )
        self.layout.aux_layout = layout
        return self.np.concatenate(blocks, axis=1), layout

    def load_batch(
        self,
        indices: Sequence[int],
        normalizer: StateNormalizer,
        *,
        include_auxiliary: bool = False,
    ) -> Dict[str, Any]:
        raw_t = self.read_state(indices, "t")
        raw_t1 = self.read_state(indices, "t_plus_1")
        batch = {
            "indices": self.np.asarray(indices, dtype=self.np.int64),
            "raw_state_t": raw_t,
            "raw_state_t_plus_1": raw_t1,
            "state_t": normalizer.normalize_state(raw_t).astype(self.np.float32),
            "delta_target": normalizer.normalize_delta(raw_t, raw_t1).astype(
                self.np.float32
            ),
        }
        batch.update(self.encode_inputs(indices))
        batch.update(self.event_targets(indices))
        if include_auxiliary:
            auxiliary, layout = self.auxiliary_targets(indices)
            batch["auxiliary_target_raw"] = auxiliary
            batch["auxiliary_layout"] = layout
        return batch

    def fit_state_normalizer(self) -> StateNormalizer:
        train = self.split_indices.get("train")
        if train is None or not len(train):
            raise FlowmapContractError("train split is empty")
        return StateNormalizer(self.layout).fit(
            self.read_state(train, "t"), self.read_state(train, "t_plus_1")
        )

    def rollout_windows(self, split: str, horizon: int) -> List[Any]:
        windows = []
        for indices in self.trajectory_indices.values():
            if not len(indices) or self.metadata["split"][indices[0]] != split:
                continue
            for start in range(0, max(0, len(indices) - int(horizon) + 1)):
                candidate = indices[start : start + int(horizon)]
                steps = self.metadata["step_index"][candidate]
                if self.np.array_equal(
                    steps, self.np.arange(steps[0], steps[0] + int(horizon))
                ):
                    windows.append(candidate)
        return windows


def batch_iterator(
    indices: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterable[Any]:
    """Deterministic dependency-light mini-batch ordering."""

    import numpy as np

    values = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(values)
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]
