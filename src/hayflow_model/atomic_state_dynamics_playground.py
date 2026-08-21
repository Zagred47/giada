"""Train-only atomic playground for the HayFlow explicit-state redesign.

The experiment deliberately does not train a full neuron surrogate.  It asks a
smaller causal question first: can the canonical mechanism STATE variables at
``t + 1 ms`` be predicted from their state at ``t`` and the causal input over
the interval?  A second, diagnostic-only arm also receives the teacher voltage
change over the same interval.  Validation and test states are never read.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import (
    CompositeFlowmapBundle,
    CompositeTransitionStore,
)
from .rollout_aware_architecture_canary import (
    CAUSAL_DRIVE_FEATURES,
    disjoint_episode_components_by_regime,
    encode_causal_realized_drive,
)


EXPECTED_05T_ARCHIVE_SHA256 = (
    "ef222130c1b6e33b302e99755a6083ea113fad63efed743bbc4e938e58c7e1f1"
)
EXPECTED_05T_INDEX_SHA256 = (
    "84221c3c4dde34e909ab81024c6ab3d44606db00ca44b76edd1374138c95fd34"
)
EXPECTED_05T_FINAL_SHA256 = (
    "7f061b3b58f9d0d8654873ae607084c62e2ebc26b4ccc17de6dc4d8d7f9a4ffe"
)

PILOT_ARMS = ("causal_start_voltage", "teacher_interval_voltage")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(source: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"unsafe artifact member {member.filename!r}")
        archive.extractall(destination)


def verified_05t_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Resolve the exact architecture-branch NO-GO that authorizes phase 06."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("05t source must be a ZIP or extracted directory")
        archive_hash = _sha256_file(source)
        if archive_hash != EXPECTED_05T_ARCHIVE_SHA256:
            # Kaggle can rebuild archive.zip around the original members.  In
            # that case member integrity below remains the authority.
            archive_hash = "kaggle-repacked"
        stamp = {
            "path": str(source),
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
        marker = cache_dir / ".source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != stamp:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            _safe_extract(source, cache_dir)
            marker.write_text(json.dumps(stamp, sort_keys=True), encoding="utf-8")
        search_root = cache_dir
    else:
        archive_hash = "extracted-directory"
        search_root = source
    matching: List[Path] = []
    for index_path in search_root.rglob("artifact_index.json"):
        if _sha256_file(index_path) == EXPECTED_05T_INDEX_SHA256:
            matching.append(index_path.parent)
    if len(matching) != 1:
        raise RuntimeError(
            f"expected one exact 05t artifact root; found {len(matching)}"
        )
    root = matching[0]
    index_payload = json.loads(
        (root / "artifact_index.json").read_text(encoding="utf-8")
    )
    indexed = index_payload.get("artifacts", [])
    member_failures = []
    for record in indexed:
        member = root / str(record["path"])
        if (
            not member.is_file()
            or member.stat().st_size != int(record["size_bytes"])
            or _sha256_file(member) != str(record["sha256"])
        ):
            member_failures.append(str(record["path"]))
    if member_failures:
        raise RuntimeError(f"05t indexed member verification failed: {member_failures}")
    final_path = root / "final_report.json"
    if not final_path.is_file() or _sha256_file(final_path) != EXPECTED_05T_FINAL_SHA256:
        raise RuntimeError("05t final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final.get("diagnosis") != "AUTOREGRESSIVE_REPRESENTATION_NO_GO":
        raise RuntimeError("05t does not contain the registered branch-closing NO-GO")
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_05T_INDEX_SHA256,
        "final_report_sha256": EXPECTED_05T_FINAL_SHA256,
        "indexed_member_count": len(indexed),
        "diagnosis": final["diagnosis"],
        "next_step": final.get("next_step"),
    }


@dataclass(frozen=True)
class AtomicStateDynamicsConfig:
    role_seed: int = 61001
    pilot_seed: int = 61017
    fit_components_per_regime: int = 3
    calibration_components_per_regime: int = 1
    development_components_per_regime: int = 1
    fit_transition_limit: int = 1024
    calibration_transition_limit: int = 256
    development_transition_limit: int = 256
    normalization_transition_limit: int = 1024
    hidden_width: int = 48
    embedding_width: int = 12
    training_steps: int = 300
    batch_transition_count: int = 8
    coordinates_per_transition: int = 512
    evaluation_interval: int = 25
    progress_interval: int = 25
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    gradient_clip_norm: float = 1.0
    normalized_delta_limit: float = 8.0
    active_delta_threshold: float = 0.25
    active_delta_weight: float = 4.0
    minimum_delta_scale: float = 1e-5
    rollout_horizons_ms: Tuple[int, ...] = (1, 2, 4, 8)
    rollout_windows_per_horizon: int = 16
    evaluation_coordinate_chunk: int = 2048
    minimum_pilot_improvement_fraction: float = 0.02

    def validate(self) -> None:
        positive = (
            self.fit_components_per_regime,
            self.calibration_components_per_regime,
            self.development_components_per_regime,
            self.fit_transition_limit,
            self.calibration_transition_limit,
            self.development_transition_limit,
            self.normalization_transition_limit,
            self.hidden_width,
            self.embedding_width,
            self.training_steps,
            self.batch_transition_count,
            self.coordinates_per_transition,
            self.evaluation_interval,
            self.progress_interval,
            self.rollout_windows_per_horizon,
            self.evaluation_coordinate_chunk,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("06a positive integer configuration is invalid")
        if tuple(sorted(set(self.rollout_horizons_ms))) != self.rollout_horizons_ms:
            raise ValueError("06a rollout horizons must be unique and increasing")
        if self.rollout_horizons_ms[0] != 1:
            raise ValueError("06a rollout horizons must start at one millisecond")
        if not 0 < self.learning_rate < 1 or self.weight_decay < 0:
            raise ValueError("06a optimizer configuration is invalid")
        if not 0 < self.active_delta_weight or self.active_delta_threshold < 0:
            raise ValueError("06a activity weighting is invalid")
        if self.minimum_delta_scale <= 0 or self.normalized_delta_limit <= 0:
            raise ValueError("06a state-domain limits are invalid")
        if not 0 < self.minimum_pilot_improvement_fraction < 1:
            raise ValueError("06a pilot materiality threshold is invalid")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AtomicStateDynamicsConfig":
        payload = dict(values)
        if "rollout_horizons_ms" in payload:
            payload["rollout_horizons_ms"] = tuple(
                map(int, payload["rollout_horizons_ms"])
            )
        result = cls(**payload)
        result.validate()
        return result


def mechanism_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def inverse_mechanism_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=axis)
    scale = (q75 - q25) / 1.349
    standard = np.std(values, axis=axis)
    return np.where(scale > 1e-10, scale, standard)


def fit_mechanism_statistics(
    state_t: np.ndarray,
    state_t_plus_1: np.ndarray,
    semantic_group_ids: np.ndarray,
    *,
    minimum_delta_scale: float,
) -> Dict[str, np.ndarray]:
    """Fit coordinate stats and lift weak scales using semantic peers only."""

    z_t = mechanism_logit(state_t)
    z_t1 = mechanism_logit(state_t_plus_1)
    center = np.median(z_t, axis=0)
    state_scale = _robust_scale(z_t)
    delta_scale = _robust_scale(z_t1 - z_t)
    state_scale = np.where(state_scale > 1e-8, state_scale, 1.0)
    repaired = delta_scale.copy()
    for group in np.unique(semantic_group_ids):
        mask = semantic_group_ids == group
        positive = delta_scale[mask & (delta_scale >= minimum_delta_scale)]
        floor = float(np.median(positive)) if len(positive) else minimum_delta_scale
        repaired[mask] = np.maximum(repaired[mask], max(floor, minimum_delta_scale))
    return {
        "state_center": center.astype(np.float32),
        "state_scale": state_scale.astype(np.float32),
        "delta_scale": repaired.astype(np.float32),
        "raw_delta_scale": delta_scale.astype(np.float32),
        "repaired_coordinate_count": np.asarray(
            [int(np.sum(delta_scale < minimum_delta_scale))], dtype=np.int64
        ),
    }


def build_atomic_episode_roles(
    episode_rows: Sequence[Mapping[str, Any]],
    config: AtomicStateDynamicsConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create seed/snapshot-disjoint roles from original train episodes only."""

    grouped = disjoint_episode_components_by_regime(
        episode_rows, role_seed=config.role_seed
    )
    counts = {
        "fit": config.fit_components_per_regime,
        "calibration": config.calibration_components_per_regime,
        "development": config.development_components_per_regime,
    }
    roles = {name: [] for name in counts}
    for regime, components in sorted(grouped.items()):
        required = sum(counts.values())
        if len(components) < required:
            raise RuntimeError(
                f"06a regime {regime!r} has {len(components)} components; "
                f"at least {required} are required"
            )
        cursor = 0
        for role, count in counts.items():
            for component in components[cursor : cursor + count]:
                for source in component:
                    row = dict(source)
                    row["06a_role"] = role
                    row["06a_regime"] = regime
                    roles[role].append(row)
            cursor += count
    if any(not rows for rows in roles.values()):
        raise RuntimeError("06a could not construct all train-derived roles")
    if any(str(row.get("split")) != "train" for rows in roles.values() for row in rows):
        raise RuntimeError("06a role construction leaked outside train")
    return roles


try:  # PyTorch remains optional for contract-only unit tests.
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    torch_functional = None


if nn is not None:

    class SemanticMechanismStateUpdater(nn.Module):
        """Shared, zero-initialized residual updater for one STATE coordinate."""

        def __init__(
            self,
            *,
            mechanism_count: int,
            variable_count: int,
            kind_count: int,
            region_count: int,
            static_width: int,
            drive_width: int,
            hidden_width: int,
            embedding_width: int,
            normalized_delta_limit: float,
        ) -> None:
            super().__init__()
            self.normalized_delta_limit = float(normalized_delta_limit)
            self.mechanism_embedding = nn.Embedding(mechanism_count, embedding_width)
            self.variable_embedding = nn.Embedding(variable_count, embedding_width)
            self.kind_embedding = nn.Embedding(kind_count, embedding_width)
            self.region_embedding = nn.Embedding(region_count, embedding_width)
            width = 3 + static_width + drive_width + 4 * embedding_width
            self.encoder = nn.Sequential(
                nn.Linear(width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, hidden_width),
                nn.SiLU(),
            )
            self.proposal = nn.Linear(hidden_width, 1)
            self.relaxation = nn.Linear(hidden_width, 1)
            nn.init.zeros_(self.proposal.weight)
            nn.init.zeros_(self.proposal.bias)

        def forward(
            self,
            state_value: Any,
            voltage_t: Any,
            voltage_delta: Any,
            drive: Any,
            static: Any,
            mechanism_id: Any,
            variable_id: Any,
            kind_id: Any,
            region_id: Any,
        ) -> Any:
            embedded = torch.cat(
                (
                    self.mechanism_embedding(mechanism_id),
                    self.variable_embedding(variable_id),
                    self.kind_embedding(kind_id),
                    self.region_embedding(region_id),
                ),
                dim=-1,
            )
            hidden = self.encoder(
                torch.cat(
                    (
                        state_value.unsqueeze(-1),
                        (voltage_t / 100.0).unsqueeze(-1),
                        (voltage_delta / 100.0).unsqueeze(-1),
                        drive,
                        static,
                        embedded,
                    ),
                    dim=-1,
                )
            )
            proposal = self.normalized_delta_limit * torch.tanh(
                self.proposal(hidden).squeeze(-1)
            )
            return torch.sigmoid(self.relaxation(hidden).squeeze(-1)) * proposal

else:  # pragma: no cover

    class SemanticMechanismStateUpdater:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06a pilot requires PyTorch")


class _CompactProgress:
    def __init__(self, label: str, total: int, interval: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.interval = max(1, int(interval))
        self.started = time.monotonic()

    def update(self, value: int, suffix: str = "") -> None:
        if value != 1 and value != self.total and value % self.interval:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        eta = elapsed / value * (self.total - value) if value else 0.0
        detail = f" {suffix}" if suffix else ""
        print(
            f"[HayFlow 06a][{self.label}] {value}/{self.total} "
            f"({100.0 * value / self.total:.1f}%) ETA {eta / 60:.1f} min{detail}",
            flush=True,
        )


class AtomicStateDynamicsPlayground:
    """End-to-end train-only session for notebook 06a."""

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: AtomicStateDynamicsConfig,
        artifact_05t_source: Path,
        *,
        code_revision: str,
    ) -> None:
        config.validate()
        self.bundle = bundle
        self.output_dir = Path(output_dir)
        self.config = config
        self.artifact_05t_source = Path(artifact_05t_source)
        self.code_revision = str(code_revision)
        self.store = CompositeTransitionStore(bundle)
        self.layout = self.store.layout
        self.roles: Dict[str, List[Dict[str, Any]]] = {}
        self.role_indices: Dict[str, np.ndarray] = {}
        self.mechanism_records: List[Mapping[str, Any]] = []
        self.ion_records: List[Mapping[str, Any]] = []
        self.coordinate: Dict[str, np.ndarray] = {}
        self.statistics: Dict[str, np.ndarray] = {}
        self.materialized: Dict[str, Dict[str, np.ndarray]] = {}
        self.models: Dict[str, Any] = {}
        self.run_reports: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        self.store.close()

    def _indices_for_rows(
        self, rows: Sequence[Mapping[str, Any]], limit: int, role: str
    ) -> np.ndarray:
        indices: List[int] = []
        for row in rows:
            indices.extend(
                map(int, self.store.trajectory_indices[str(row["trajectory_id"])])
            )
        unique = sorted(
            set(indices),
            key=lambda value: hashlib.sha256(
                f"{self.config.role_seed}|{role}|{value}".encode()
            ).hexdigest(),
        )
        return np.asarray(unique[:limit], dtype=np.int64)

    def _coordinate_metadata(self) -> None:
        records = [
            row
            for row in self.layout.core_records
            if str(row["category"]) == "mechanism_states"
        ]
        if len(records) != self.layout.category_widths["mechanism_states"]:
            raise RuntimeError("06a mechanism STATE coordinate count mismatch")
        mechanism_lookup = {name: i for i, name in enumerate(self.layout.mechanism_names)}
        variable_lookup = {name: i for i, name in enumerate(self.layout.variable_names)}
        kind_lookup = {name: i for i, name in enumerate(self.layout.kind_names)}
        semantic_names = [
            f"{row['mechanism']}|{row['variable']}|{row['kind']}" for row in records
        ]
        semantic_vocab = {name: i for i, name in enumerate(sorted(set(semantic_names)))}
        segment = np.asarray(
            [self.layout._record_segment(row) for row in records], dtype=np.int64
        )
        self.mechanism_records = records
        self.coordinate = {
            "segment": segment,
            "mechanism": np.asarray(
                [mechanism_lookup[str(row["mechanism"])] for row in records],
                dtype=np.int64,
            ),
            "variable": np.asarray(
                [variable_lookup[str(row["variable"])] for row in records],
                dtype=np.int64,
            ),
            "kind": np.asarray(
                [kind_lookup[str(row["kind"])] for row in records], dtype=np.int64
            ),
            "region": self.layout.segment_region_ids[segment],
            "semantic_group": np.asarray(
                [semantic_vocab[name] for name in semantic_names], dtype=np.int64
            ),
        }
        groups: Dict[int, np.ndarray] = {}
        for group in np.unique(self.coordinate["semantic_group"]):
            groups[int(group)] = np.flatnonzero(
                self.coordinate["semantic_group"] == group
            )
        self.coordinate_groups = groups
        ion_records = [
            row
            for row in self.layout.core_records
            if str(row["category"]) == "calcium_ions"
        ]
        if len(ion_records) != self.layout.category_widths["calcium_ions"]:
            raise RuntimeError("06a ion coordinate count mismatch")
        ion_names = sorted({str(row["variable"]) for row in ion_records})
        ion_lookup = {name: index for index, name in enumerate(ion_names)}
        self.ion_records = ion_records
        self.ion_feature_names = ion_names
        self.ion_segment_ids = np.asarray(
            [self.layout._record_segment(row) for row in ion_records], dtype=np.int64
        )
        self.ion_variable_ids = np.asarray(
            [ion_lookup[str(row["variable"])] for row in ion_records], dtype=np.int64
        )

    def _normalization_fit(self) -> None:
        indices = self.role_indices["fit"][: self.config.normalization_transition_limit]
        state_t = self.store.read_state(
            indices, "t", categories=("mechanism_states",)
        )
        state_t1 = self.store.read_state(
            indices, "t_plus_1", categories=("mechanism_states",)
        )
        self.statistics = fit_mechanism_statistics(
            state_t,
            state_t1,
            self.coordinate["semantic_group"],
            minimum_delta_scale=self.config.minimum_delta_scale,
        )
        ion_t = np.log1p(
            np.maximum(
                self.store.read_state(indices, "t", categories=("calcium_ions",)),
                0.0,
            )
        )
        ion_center = np.zeros(len(self.ion_feature_names), dtype=np.float32)
        ion_scale = np.ones(len(self.ion_feature_names), dtype=np.float32)
        for variable in range(len(self.ion_feature_names)):
            values = ion_t[:, self.ion_variable_ids == variable].reshape(-1)
            ion_center[variable] = float(np.median(values))
            scale = float(_robust_scale(values, axis=0))
            ion_scale[variable] = scale if scale > 1e-8 else 1.0
        self.statistics["ion_center"] = ion_center
        self.statistics["ion_scale"] = ion_scale
        np.savez_compressed(
            self.output_dir / "mechanism_state_normalization.npz",
            **self.statistics,
        )

    def _ion_context(self, indices: Sequence[int]) -> np.ndarray:
        raw = self.store.read_state(
            indices, "t", categories=("calcium_ions",)
        )
        transformed = np.log1p(np.maximum(raw, 0.0))
        normalized = (
            transformed
            - self.statistics["ion_center"][self.ion_variable_ids][None, :]
        ) / self.statistics["ion_scale"][self.ion_variable_ids][None, :]
        output = np.zeros(
            (
                len(indices),
                self.layout.segment_count,
                len(self.ion_feature_names),
            ),
            dtype=np.float32,
        )
        output[:, self.ion_segment_ids, self.ion_variable_ids] = normalized.astype(
            np.float32
        )
        return output

    def _materialize_role(self, role: str) -> Dict[str, np.ndarray]:
        indices = self.role_indices[role]
        raw_t = self.store.read_state(indices, "t", categories=("mechanism_states",))
        raw_t1 = self.store.read_state(
            indices, "t_plus_1", categories=("mechanism_states",)
        )
        transformed_t = mechanism_logit(raw_t).astype(np.float32)
        transformed_t1 = mechanism_logit(raw_t1).astype(np.float32)
        values = {
            "indices": indices,
            "state": (
                (transformed_t - self.statistics["state_center"])
                / self.statistics["state_scale"]
            ).astype(np.float32),
            "delta": (
                (transformed_t1 - transformed_t) / self.statistics["delta_scale"]
            ).astype(np.float32),
            "voltage_t": self.store.read_state(
                indices, "t", categories=("voltage",)
            ).astype(np.float32),
            "voltage_t1": self.store.read_state(
                indices, "t_plus_1", categories=("voltage",)
            ).astype(np.float32),
            "context": np.concatenate(
                (
                    encode_causal_realized_drive(self.store, indices),
                    self._ion_context(indices),
                ),
                axis=-1,
            ),
        }
        if not all(np.isfinite(value).all() for key, value in values.items() if key != "indices"):
            raise RuntimeError(f"06a materialized non-finite values in {role}")
        self.materialized[role] = values
        return values

    def prepare_playground(self) -> Dict[str, Any]:
        if self.output_dir.exists():
            raise FileExistsError(
                f"output already exists: {self.output_dir}; use a fresh session"
            )
        self.output_dir.mkdir(parents=True)
        _, source_report = verified_05t_artifact_root(
            self.artifact_05t_source,
            self.output_dir.parent / ".06a_artifact_cache" / "05t",
        )
        self.roles = build_atomic_episode_roles(self.store.episode_rows, self.config)
        limits = {
            "fit": self.config.fit_transition_limit,
            "calibration": self.config.calibration_transition_limit,
            "development": self.config.development_transition_limit,
        }
        self.role_indices = {
            role: self._indices_for_rows(rows, limits[role], role)
            for role, rows in self.roles.items()
        }
        if any(
            str(self.store.metadata["split"][index]) != "train"
            for indices in self.role_indices.values()
            for index in indices
        ):
            raise RuntimeError("06a attempted to read a non-train transition")
        trajectory_sets = {
            role: {str(row["trajectory_id"]) for row in rows}
            for role, rows in self.roles.items()
        }
        overlaps = {
            f"{left}:{right}": sorted(trajectory_sets[left] & trajectory_sets[right])
            for left, right in (("fit", "calibration"), ("fit", "development"), ("calibration", "development"))
        }
        if any(overlaps.values()):
            raise RuntimeError(f"06a role trajectory overlap: {overlaps}")
        self._coordinate_metadata()
        self._normalization_fit()
        for role in ("fit", "calibration", "development"):
            self._materialize_role(role)
        contract = {
            "schema_version": "06a-atomic-state-playground-contract-v1",
            "valid": True,
            "experiment": "atomic_state_dynamics_playground",
            "architecture_family": "HayFlow-ESI",
            "dataset_fingerprint": self.bundle.fingerprint,
            "code_revision": self.code_revision,
            "source_05t": source_report,
            "state_target": "canonical mechanism STATE at t_plus_1",
            "input_view": "U_realized",
            "arms": list(PILOT_ARMS),
            "teacher_interval_voltage_is_diagnostic_only": True,
            "state_and_outcome_splits_read": ["train"],
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "candidate_selection_performed": False,
            "role_episode_counts": {name: len(rows) for name, rows in self.roles.items()},
            "role_transition_counts": {
                name: int(len(indices)) for name, indices in self.role_indices.items()
            },
            "role_trajectory_overlap": overlaps,
            "mechanism_state_coordinate_count": len(self.mechanism_records),
            "semantic_group_count": len(self.coordinate_groups),
            "local_ion_context_names": self.ion_feature_names,
            "normalization_fit_split": "train/fit",
            "normalization_repaired_coordinate_count": int(
                self.statistics["repaired_coordinate_count"][0]
            ),
            "configuration": asdict(self.config),
        }
        _write_json(self.output_dir / "playground_contract.json", contract)
        return contract

    def _new_model(self, device: Any) -> Any:
        if torch is None:
            raise RuntimeError("06a pilot requires PyTorch")
        return SemanticMechanismStateUpdater(
            mechanism_count=len(self.layout.mechanism_names),
            variable_count=len(self.layout.variable_names),
            kind_count=len(self.layout.kind_names),
            region_count=len(self.layout.region_names),
            static_width=self.layout.segment_static.shape[1],
            drive_width=len(CAUSAL_DRIVE_FEATURES) + len(self.ion_feature_names),
            hidden_width=self.config.hidden_width,
            embedding_width=self.config.embedding_width,
            normalized_delta_limit=self.config.normalized_delta_limit,
        ).to(device)

    def _sample_coordinates(self, rng: np.random.Generator, count: int) -> np.ndarray:
        group_ids = rng.choice(list(self.coordinate_groups), size=count, replace=True)
        return np.asarray(
            [rng.choice(self.coordinate_groups[int(group)]) for group in group_ids],
            dtype=np.int64,
        )

    def _batch(
        self,
        values: Mapping[str, np.ndarray],
        transition_rows: np.ndarray,
        coordinates: np.ndarray,
        arm: str,
        device: Any,
    ) -> Tuple[Tuple[Any, ...], Any]:
        rows = np.repeat(transition_rows, coordinates.shape[1])
        cols = coordinates.reshape(-1)
        segments = self.coordinate["segment"][cols]
        voltage_delta = (
            values["voltage_t1"][rows, segments]
            - values["voltage_t"][rows, segments]
            if arm == "teacher_interval_voltage"
            else np.zeros(len(rows), dtype=np.float32)
        )
        tensor = lambda value, dtype=None: torch.as_tensor(
            value, dtype=dtype, device=device
        )
        inputs = (
            tensor(values["state"][rows, cols], torch.float32),
            tensor(values["voltage_t"][rows, segments], torch.float32),
            tensor(voltage_delta, torch.float32),
            tensor(values["context"][rows, segments], torch.float32),
            tensor(self.layout.segment_static[segments], torch.float32),
            tensor(self.coordinate["mechanism"][cols], torch.long),
            tensor(self.coordinate["variable"][cols], torch.long),
            tensor(self.coordinate["kind"][cols], torch.long),
            tensor(self.coordinate["region"][cols], torch.long),
        )
        target = tensor(values["delta"][rows, cols], torch.float32)
        return inputs, target

    def _evaluate_one_step(self, model: Any, role: str, arm: str, device: Any) -> Dict[str, Any]:
        values = self.materialized[role]
        squared_error = 0.0
        persistence_error = 0.0
        active_squared_error = 0.0
        active_persistence_error = 0.0
        count = 0
        active_count = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(self.mechanism_records), self.config.evaluation_coordinate_chunk):
                stop = min(len(self.mechanism_records), start + self.config.evaluation_coordinate_chunk)
                coordinate = np.arange(start, stop, dtype=np.int64)
                coordinates = np.broadcast_to(coordinate, (len(values["indices"]), len(coordinate)))
                rows = np.arange(len(values["indices"]), dtype=np.int64)
                inputs, target = self._batch(values, rows, coordinates, arm, device)
                prediction = model(*inputs)
                error = prediction - target
                active = target.abs() >= self.config.active_delta_threshold
                squared_error += float(torch.sum(error * error).cpu())
                persistence_error += float(torch.sum(target * target).cpu())
                if bool(active.any()):
                    active_squared_error += float(torch.sum(error[active] ** 2).cpu())
                    active_persistence_error += float(torch.sum(target[active] ** 2).cpu())
                    active_count += int(active.sum().item())
                count += int(target.numel())
        rmse = math.sqrt(squared_error / max(count, 1))
        persistence = math.sqrt(persistence_error / max(count, 1))
        active_rmse = math.sqrt(active_squared_error / max(active_count, 1))
        active_persistence = math.sqrt(active_persistence_error / max(active_count, 1))
        return {
            "normalized_delta_rmse": rmse,
            "persistence_normalized_delta_rmse": persistence,
            "improvement_vs_persistence_fraction": 1.0 - rmse / max(persistence, 1e-12),
            "active_normalized_delta_rmse": active_rmse,
            "active_persistence_normalized_delta_rmse": active_persistence,
            "active_improvement_vs_persistence_fraction": 1.0
            - active_rmse / max(active_persistence, 1e-12),
            "coordinate_example_count": count,
            "active_coordinate_example_count": active_count,
        }

    def _train_arm(self, arm: str, device: Any) -> Dict[str, Any]:
        torch.manual_seed(self.config.pilot_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.pilot_seed)
        rng = np.random.default_rng(self.config.pilot_seed)
        model = self._new_model(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        fit = self.materialized["fit"]
        best_loss = math.inf
        best_state: Optional[Dict[str, Any]] = None
        curve: List[Dict[str, Any]] = []
        progress = _CompactProgress(arm, self.config.training_steps, self.config.progress_interval)
        for step in range(1, self.config.training_steps + 1):
            model.train()
            rows = rng.integers(
                0, len(fit["indices"]), size=self.config.batch_transition_count
            )
            coordinates = self._sample_coordinates(
                rng,
                self.config.batch_transition_count * self.config.coordinates_per_transition,
            ).reshape(self.config.batch_transition_count, -1)
            inputs, target = self._batch(fit, rows, coordinates, arm, device)
            prediction = model(*inputs)
            weight = 1.0 + self.config.active_delta_weight * (
                target.abs() >= self.config.active_delta_threshold
            ).float()
            loss = torch.mean(weight * torch_functional.smooth_l1_loss(
                prediction, target, reduction="none"
            ))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            if step == 1 or step % self.config.evaluation_interval == 0 or step == self.config.training_steps:
                calibration = self._evaluate_one_step(model, "calibration", arm, device)
                calibration_loss = calibration["normalized_delta_rmse"]
                curve.append(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                        "calibration_normalized_delta_rmse": calibration_loss,
                    }
                )
                if calibration_loss < best_loss:
                    best_loss = calibration_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            progress.update(
                step,
                f"loss={float(loss.detach().cpu()):.4g} cal={best_loss:.4g}",
            )
        if best_state is None:
            raise RuntimeError("06a did not produce a calibration checkpoint")
        model.load_state_dict(best_state)
        self.models[arm] = model
        checkpoint = self.output_dir / f"pilot_{arm}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "arm": arm,
                "pilot_seed": self.config.pilot_seed,
                "configuration": asdict(self.config),
            },
            checkpoint,
        )
        development = self._evaluate_one_step(model, "development", arm, device)
        report = {
            "arm": arm,
            "paired_pilot_seed": self.config.pilot_seed,
            "parameter_count": int(sum(value.numel() for value in model.parameters())),
            "best_calibration_normalized_delta_rmse": best_loss,
            "development": development,
            "learning_curve": curve,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": _sha256_file(checkpoint),
            "finite": all(
                math.isfinite(float(value))
                for key, value in development.items()
                if isinstance(value, (float, int)) and "count" not in key
            ),
        }
        _write_json(self.output_dir / f"pilot_{arm}_report.json", report)
        return report

    def run_one_step_pilot(self) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reports = {arm: self._train_arm(arm, device) for arm in PILOT_ARMS}
        self.run_reports = reports
        payload = {
            "schema_version": "06a-one-step-pilot-v1",
            "valid": all(row["finite"] for row in reports.values()),
            "device": str(device),
            "candidate_selection_performed": False,
            "teacher_interval_voltage_is_diagnostic_only": True,
            "runs": reports,
        }
        _write_json(self.output_dir / "one_step_pilot.json", payload)
        return payload

    def _development_windows(self, horizon: int) -> List[np.ndarray]:
        allowed = {str(row["trajectory_id"]) for row in self.roles["development"]}
        windows: List[np.ndarray] = []
        for trajectory in sorted(allowed):
            indices = self.store.trajectory_indices[trajectory]
            for start in range(max(0, len(indices) - horizon + 1)):
                candidate = indices[start : start + horizon]
                steps = self.store.metadata["step_index"][candidate]
                if np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                    windows.append(candidate)
        windows = sorted(
            windows,
            key=lambda row: hashlib.sha256(
                f"{self.config.role_seed}|{horizon}|{','.join(map(str, row))}".encode()
            ).hexdigest(),
        )
        return windows[: self.config.rollout_windows_per_horizon]

    def _predict_full_delta(
        self,
        model: Any,
        normalized_state: np.ndarray,
        voltage_t: np.ndarray,
        voltage_t1: np.ndarray,
        context: np.ndarray,
        arm: str,
        device: Any,
    ) -> np.ndarray:
        batch = len(normalized_state)
        output = np.empty_like(normalized_state, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(self.mechanism_records), self.config.evaluation_coordinate_chunk):
                stop = min(len(self.mechanism_records), start + self.config.evaluation_coordinate_chunk)
                cols = np.arange(start, stop, dtype=np.int64)
                segments = self.coordinate["segment"][cols]
                rows = np.repeat(np.arange(batch), len(cols))
                flat_cols = np.tile(cols, batch)
                flat_segments = self.coordinate["segment"][flat_cols]
                vdelta = (
                    voltage_t1[rows, flat_segments] - voltage_t[rows, flat_segments]
                    if arm == "teacher_interval_voltage"
                    else np.zeros(len(rows), dtype=np.float32)
                )
                tensor = lambda value, dtype=None: torch.as_tensor(
                    value, dtype=dtype, device=device
                )
                prediction = model(
                    tensor(normalized_state[rows, flat_cols], torch.float32),
                    tensor(voltage_t[rows, flat_segments], torch.float32),
                    tensor(vdelta, torch.float32),
                    tensor(context[rows, flat_segments], torch.float32),
                    tensor(self.layout.segment_static[flat_segments], torch.float32),
                    tensor(self.coordinate["mechanism"][flat_cols], torch.long),
                    tensor(self.coordinate["variable"][flat_cols], torch.long),
                    tensor(self.coordinate["kind"][flat_cols], torch.long),
                    tensor(self.coordinate["region"][flat_cols], torch.long),
                ).reshape(batch, len(cols))
                output[:, start:stop] = prediction.cpu().numpy()
        return output

    def evaluate_teacher_voltage_rollouts(self) -> Dict[str, Any]:
        if set(self.models) != set(PILOT_ARMS):
            raise RuntimeError("run_one_step_pilot must precede rollout evaluation")
        device = next(iter(self.models.values())).proposal.weight.device
        report: Dict[str, Any] = {}
        for arm, model in self.models.items():
            arm_report: Dict[str, Any] = {}
            for horizon in self.config.rollout_horizons_ms:
                windows = self._development_windows(horizon)
                if not windows:
                    raise RuntimeError(f"06a found no development windows at {horizon} ms")
                first = np.asarray([row[0] for row in windows], dtype=np.int64)
                raw_initial = self.store.read_state(
                    first, "t", categories=("mechanism_states",)
                )
                transformed_initial = mechanism_logit(raw_initial).astype(np.float32)
                transformed = transformed_initial.copy()
                for step in range(horizon):
                    indices = np.asarray([row[step] for row in windows], dtype=np.int64)
                    voltage_t = self.store.read_state(
                        indices, "t", categories=("voltage",)
                    ).astype(np.float32)
                    voltage_t1 = self.store.read_state(
                        indices, "t_plus_1", categories=("voltage",)
                    ).astype(np.float32)
                    context = np.concatenate(
                        (
                            encode_causal_realized_drive(self.store, indices),
                            self._ion_context(indices),
                        ),
                        axis=-1,
                    )
                    normalized_state = (
                        transformed - self.statistics["state_center"]
                    ) / self.statistics["state_scale"]
                    predicted_delta = self._predict_full_delta(
                        model,
                        normalized_state.astype(np.float32),
                        voltage_t,
                        voltage_t1,
                        context,
                        arm,
                        device,
                    )
                    transformed += predicted_delta * self.statistics["delta_scale"]
                final_indices = np.asarray(
                    [row[-1] for row in windows], dtype=np.int64
                )
                target = mechanism_logit(
                    self.store.read_state(
                        final_indices,
                        "t_plus_1",
                        categories=("mechanism_states",),
                    )
                ).astype(np.float32)
                error = (transformed - target) / self.statistics["state_scale"]
                persistence_error = (
                    transformed_initial - target
                ) / self.statistics["state_scale"]
                rmse = float(np.sqrt(np.mean(error * error)))
                persistence = float(
                    np.sqrt(np.mean(persistence_error * persistence_error))
                )
                raw_prediction = inverse_mechanism_logit(transformed)
                arm_report[f"{horizon}_ms"] = {
                    "window_count": len(windows),
                    "normalized_state_rmse": rmse,
                    "persistence_normalized_state_rmse": persistence,
                    "improvement_vs_persistence_fraction": 1.0
                    - rmse / max(persistence, 1e-12),
                    "nonfinite_count": int(np.sum(~np.isfinite(raw_prediction))),
                    "domain_violation_count": int(
                        np.sum((raw_prediction < 0.0) | (raw_prediction > 1.0))
                    ),
                }
            report[arm] = arm_report
        payload = {
            "schema_version": "06a-teacher-voltage-rollout-v1",
            "valid": all(
                metrics["nonfinite_count"] == 0
                and metrics["domain_violation_count"] == 0
                for arm in report.values()
                for metrics in arm.values()
            ),
            "state_rollout_is_recursive": True,
            "membrane_voltage_is_teacher_forced": True,
            "validation_or_test_accessed": False,
            "arms": report,
        }
        _write_json(self.output_dir / "teacher_voltage_rollouts.json", payload)
        return payload

    def _plot(self, pilot: Mapping[str, Any], rollout: Mapping[str, Any]) -> List[str]:
        import matplotlib.pyplot as plt

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for arm, row in pilot["runs"].items():
            curve = row["learning_curve"]
            axes[0].plot(
                [point["step"] for point in curve],
                [point["calibration_normalized_delta_rmse"] for point in curve],
                marker="o",
                label=arm,
            )
            horizons = [int(key.split("_")[0]) for key in rollout["arms"][arm]]
            values = [
                rollout["arms"][arm][f"{horizon}_ms"][
                    "improvement_vs_persistence_fraction"
                ]
                for horizon in horizons
            ]
            axes[1].plot(horizons, values, marker="o", label=arm)
        axes[0].set(xlabel="training step", ylabel="calibration normalized-delta RMSE")
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set(
            xlabel="recursive state horizon (ms)",
            ylabel="improvement vs persistence",
            xticks=list(self.config.rollout_horizons_ms),
        )
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        path = figure_dir / "atomic_state_pilot_summary.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return [str(path.relative_to(self.output_dir))]

    def finalize(
        self,
        pilot_report: Mapping[str, Any],
        rollout_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        causal = pilot_report["runs"]["causal_start_voltage"]["development"]
        teacher = pilot_report["runs"]["teacher_interval_voltage"]["development"]
        technical_gate = (
            pilot_report.get("valid")
            and rollout_report.get("valid")
            and teacher["improvement_vs_persistence_fraction"]
            >= self.config.minimum_pilot_improvement_fraction
        )
        if not technical_gate:
            diagnosis = "ATOMIC_STATE_UPDATE_NOT_YET_LEARNABLE"
            next_step = "inspect_state_contract_normalization_or_local_context"
        elif causal["improvement_vs_persistence_fraction"] >= self.config.minimum_pilot_improvement_fraction:
            diagnosis = "ATOMIC_STATE_UPDATE_LEARNABLE_FROM_CAUSAL_BOUNDARY"
            next_step = "06b_explicit_state_updater_canary"
        else:
            diagnosis = "ATOMIC_STATE_UPDATE_REQUIRES_INTERVAL_VOLTAGE_CONTEXT"
            next_step = "06b_voltage_state_coupling_canary"
        figures = self._plot(pilot_report, rollout_report)
        final = {
            "schema_version": "06a-final-report-v1",
            "valid": bool(pilot_report.get("valid") and rollout_report.get("valid")),
            "decision_grade": False,
            "diagnosis": diagnosis,
            "architecture_family": "HayFlow-ESI",
            "candidate_selection_performed": False,
            "full_neuron_model_trained": False,
            "state_and_outcome_splits_read": ["train"],
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "fresh_test_generation_authorized": False,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "technical_gate_passed": bool(technical_gate),
            "minimum_pilot_improvement_fraction": self.config.minimum_pilot_improvement_fraction,
            "causal_one_step_improvement_vs_persistence_fraction": causal[
                "improvement_vs_persistence_fraction"
            ],
            "teacher_interval_one_step_improvement_vs_persistence_fraction": teacher[
                "improvement_vs_persistence_fraction"
            ],
            "teacher_interval_voltage_is_diagnostic_only": True,
            "rollout_horizons_ms": list(self.config.rollout_horizons_ms),
            "next_step": next_step,
            "figures": figures,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "final_report.json", final)
        self._write_artifact_index()
        return final

    def _write_artifact_index(self) -> None:
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.name == "artifact_index.json":
                continue
            records.append(
                {
                    "path": str(path.relative_to(self.output_dir)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        _write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "06a-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )


__all__ = [
    "EXPECTED_05T_ARCHIVE_SHA256",
    "EXPECTED_05T_INDEX_SHA256",
    "EXPECTED_05T_FINAL_SHA256",
    "PILOT_ARMS",
    "AtomicStateDynamicsConfig",
    "AtomicStateDynamicsPlayground",
    "SemanticMechanismStateUpdater",
    "build_atomic_episode_roles",
    "fit_mechanism_statistics",
    "inverse_mechanism_logit",
    "mechanism_logit",
    "verified_05t_artifact_root",
]
