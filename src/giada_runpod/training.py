"""GPU training for the paper-scale information-matched soma comparison."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from src.hayflow_model.branch_elm_information_matched_transition import (
    InformationMatchedBranchELM,
    InformationMatchedBridgeAdapter,
)
from src.hayflow_model.causal_voltage_state_coupling_forensic import CausalVoltageBridge


@dataclass(frozen=True)
class MatchedTrainingConfig:
    seeds: tuple[int, ...] = (61017, 61029, 61043)
    training_steps: int = 3000
    checkpoints: tuple[int, ...] = (100, 300, 1000, 3000)
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    voltage_scale_mv: float = 20.0
    delta_limit_mv: float = 100.0
    active_delta_threshold_mv: float = 5.0
    active_weight: float = 4.0
    normalization_sample_limit: int = 1_000_000
    evaluation_sample_limit: int = 1_000_000
    progress_interval: int = 50
    expected_input_width: int = 76
    expected_branch_elm_parameters: int = 8002
    expected_giada_parameters: int = 8985
    minimum_train_active_transitions: int = 256
    minimum_validation_active_transitions: int = 64
    minimum_validation_somatic_upcrossings: int = 16
    required_composite_stage: str | None = None
    checkpoint_selection: str = "final_preregistered"
    minimum_seed_wins: int = 2
    minimum_family_wins: int = 0
    minimum_protocol_wins: int = 0
    require_spike_transition_advantage: bool = False
    scaling_reference_seeds: tuple[int, ...] = ()
    expected_corpus_hashes: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.seeds or self.training_steps <= 0 or self.batch_size <= 0:
            raise ValueError("invalid matched training size")
        if not self.checkpoints or max(self.checkpoints) != self.training_steps:
            raise ValueError("final checkpoint must equal training_steps")
        if any(step <= 0 for step in self.checkpoints):
            raise ValueError("checkpoint steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.voltage_scale_mv <= 0 or self.delta_limit_mv <= 0:
            raise ValueError("invalid voltage scaling")
        if self.active_delta_threshold_mv != 5.0:
            raise ValueError("paper-scale active support threshold is fixed at 5 mV")
        if min(
            self.minimum_train_active_transitions,
            self.minimum_validation_active_transitions,
            self.minimum_validation_somatic_upcrossings,
        ) < 0:
            raise ValueError("support minima cannot be negative")
        if self.checkpoint_selection != "final_preregistered":
            raise ValueError("validation-based checkpoint selection is forbidden")
        if not 0 <= self.minimum_seed_wins <= len(self.seeds):
            raise ValueError("minimum_seed_wins must fit the registered seed count")
        if min(self.minimum_family_wins, self.minimum_protocol_wins) < 0:
            raise ValueError("breadth minima cannot be negative")
        if not set(self.scaling_reference_seeds).issubset(self.seeds):
            raise ValueError("scaling_reference_seeds must be registered training seeds")
        if self.required_composite_stage in {
            "s2_hybrid_production",
            "s3_hybrid_production",
        }:
            required_hashes = {
                "background_plan_sha256",
                "background_validation_sha256",
                "targeted_plan_sha256",
                "targeted_validation_sha256",
                "composite_manifest_sha256",
                "production_audit_sha256",
                "shard_marker_fingerprint_sha256",
            }
            if set(self.expected_corpus_hashes) != required_hashes:
                raise ValueError(
                    "S2/S3 requires the complete frozen corpus hash contract"
                )
            if any(
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                for value in self.expected_corpus_hashes.values()
            ):
                raise ValueError(
                    "S2/S3 corpus hashes must be lowercase SHA-256 values"
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MatchedTrainingConfig":
        payload = dict(values)
        for key in ("seeds", "checkpoints", "scaling_reference_seeds"):
            if key in payload:
                payload[key] = tuple(map(int, payload[key]))
        result = cls(**payload)
        result.validate()
        return result


class LeanSomaCorpus:
    """Lazy reader for validated soma_paper shards."""

    def __init__(self, root: Path) -> None:
        try:
            import h5py
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("paper-scale training requires h5py") from error
        self.h5py = h5py
        from .corpus_audit import corpus_components

        self.root = Path(root)
        manifest_path = self.root / "composite_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("valid"):
                raise RuntimeError("composite corpus is not validated")
        self.paths = []
        self.path_component: Dict[Path, str] = {}
        self.path_trajectory_labels: Dict[Path, Dict[int, Dict[str, str]]] = {}
        for component_id, component_root, plan_path in corpus_components(self.root):
            labels: Dict[int, Dict[str, str]] = {}
            if plan_path is not None and plan_path.is_file():
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                for shard in plan["shards"]:
                    for row in shard["trajectories"]:
                        labels[int(row["trajectory_index"])] = {
                            "protocol": str(row.get("protocol", "unknown")),
                            "family": str(row.get("protocol_family", "unknown")),
                            "arm": str(row.get("protocol_arm", "unknown")),
                        }
            for path in sorted((component_root / "shards").glob("shard-*.h5")):
                self.paths.append(path)
                self.path_component[path] = component_id
                self.path_trajectory_labels[path] = labels
        if not self.paths:
            raise FileNotFoundError(f"no paper-scale shards under {self.root}")
        self.rows: Dict[int, List[tuple[Path, np.ndarray]]] = {0: [], 1: []}
        self.metadata: Dict[str, Any] | None = None
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                metadata = json.loads(handle.attrs["schema_metadata_json"])
                if metadata.get("storage_profile") != "soma_paper":
                    raise RuntimeError(f"{path.name} is not a soma_paper shard")
                if self.metadata is None:
                    self.metadata = metadata
                else:
                    stable = ("mechanism_group_names", "ion_names", "causal_drive_features", "segment_ids", "mechanism_presence", "segment_static", "region_names", "segment_region_ids")
                    if any(self.metadata[key] != metadata[key] for key in stable):
                        raise RuntimeError("paper-scale shard feature schemas differ")
                split = np.asarray(handle["split_code"][...], dtype=np.uint8)
                for code in (0, 1):
                    indices = np.flatnonzero(split == code)
                    if len(indices):
                        self.rows[code].append((path, indices))
        if not self.rows[0] or not self.rows[1]:
            raise RuntimeError("paper-scale corpus requires train and validation rows")
        assert self.metadata is not None
        self.train_count = sum(len(indices) for _, indices in self.rows[0])
        self.validation_count = sum(len(indices) for _, indices in self.rows[1])
        self._handles: Dict[Path, Any] = {}

    def _handle(self, path: Path) -> Any:
        if path not in self._handles:
            self._handles[path] = self.h5py.File(path, "r")
        return self._handles[path]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    @staticmethod
    def _read_sorted(handle: Any, name: str, indices: np.ndarray) -> np.ndarray:
        # h5py requires strictly increasing fancy indices. Sampling with
        # replacement produces duplicates, so read each unique row once and
        # reconstruct the requested order afterward.
        unique, inverse = np.unique(indices, return_inverse=True)
        values = np.asarray(handle[name][unique])
        return values[inverse]

    def sample_raw(self, split_code: int, count: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        groups = self.rows[int(split_code)]
        sizes = np.asarray([len(indices) for _, indices in groups], dtype=np.float64)
        selected_group = int(rng.choice(len(groups), p=sizes / sizes.sum()))
        path, available = groups[selected_group]
        positions = rng.integers(0, len(available), size=int(count))
        indices = available[positions]
        handle = self._handle(path)
        names = (
            "voltage_t_mv", "voltage_t_plus_1_mv", "parent_delta_t_mv",
            "mean_child_delta_t_mv", "mechanism_state_t", "ion_state_t",
            "causal_drive",
        )
        return {name: self._read_sorted(handle, name, indices)[:, 0] if handle[name].ndim == 2 else self._read_sorted(handle, name, indices)[:, 0, :] for name in names}

    def sample_raw_global(
        self,
        split_code: int,
        count: int,
        rng: np.random.Generator,
        *,
        include_labels: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample the complete split proportionally instead of one shard.

        ``sample_raw`` intentionally selects a single shard for each training
        minibatch.  A diagnostic evaluation sample must instead represent the
        physical mixture of the complete split.  Multinomial allocation keeps
        every row equally likely while making the sampled rows and labels
        exactly reproducible from ``rng``.
        """
        requested = int(count)
        if requested <= 0:
            raise ValueError("global sample count must be positive")
        groups = self.rows[int(split_code)]
        sizes = np.asarray([len(indices) for _, indices in groups], dtype=np.float64)
        allocations = rng.multinomial(requested, sizes / sizes.sum())
        names = (
            "voltage_t_mv", "voltage_t_plus_1_mv", "parent_delta_t_mv",
            "mean_child_delta_t_mv", "mechanism_state_t", "ion_state_t",
            "causal_drive",
        )
        collected: Dict[str, List[np.ndarray]] = {name: [] for name in names}
        labels: Dict[str, List[np.ndarray]] = {
            "_component_label": [],
            "_protocol_label": [],
            "_family_label": [],
            "_arm_label": [],
        }
        for allocation, (path, available) in zip(allocations, groups):
            if allocation == 0:
                continue
            positions = rng.integers(0, len(available), size=int(allocation))
            indices = available[positions]
            handle = self._handle(path)
            for name in names:
                values = self._read_sorted(handle, name, indices)
                collected[name].append(
                    values[:, 0] if handle[name].ndim == 2 else values[:, 0, :]
                )
            if include_labels:
                trajectory_indices = self._read_sorted(
                    handle, "trajectory_index", indices
                ).astype(np.int64)
                lookup = self.path_trajectory_labels[path]
                rows = [lookup.get(int(index), {}) for index in trajectory_indices]
                labels["_component_label"].append(
                    np.full(int(allocation), self.path_component[path], dtype=object)
                )
                labels["_protocol_label"].append(
                    np.asarray(
                        [row.get("protocol", "unknown") for row in rows],
                        dtype=object,
                    )
                )
                labels["_family_label"].append(
                    np.asarray(
                        [row.get("family", "unknown") for row in rows],
                        dtype=object,
                    )
                )
                labels["_arm_label"].append(
                    np.asarray(
                        [row.get("arm", "unknown") for row in rows], dtype=object
                    )
                )
        result = {
            name: np.concatenate(chunks, axis=0) for name, chunks in collected.items()
        }
        if include_labels:
            result.update(
                {
                    name: np.concatenate(chunks, axis=0)
                    for name, chunks in labels.items()
                }
            )
        permutation = rng.permutation(requested)
        return {name: values[permutation] for name, values in result.items()}

    def iter_raw(
        self,
        split_code: int,
        limit: int,
        chunk: int = 65536,
        *,
        include_labels: bool = False,
    ) -> Iterable[Dict[str, np.ndarray]]:
        remaining = int(limit)
        for path, available in self.rows[int(split_code)]:
            if remaining <= 0:
                break
            handle = self._handle(path)
            take = available[:remaining]
            for start in range(0, len(take), chunk):
                indices = take[start : start + chunk]
                names = (
                    "voltage_t_mv", "voltage_t_plus_1_mv", "parent_delta_t_mv",
                    "mean_child_delta_t_mv", "mechanism_state_t", "ion_state_t",
                    "causal_drive",
                )
                result = {
                    name: np.asarray(handle[name][indices])[:, 0]
                    if handle[name].ndim == 2
                    else np.asarray(handle[name][indices])[:, 0, :]
                    for name in names
                }
                if include_labels:
                    trajectory_indices = np.asarray(
                        handle["trajectory_index"][indices], dtype=np.int64
                    )
                    lookup = self.path_trajectory_labels[path]
                    rows = [lookup.get(int(index), {}) for index in trajectory_indices]
                    result["_component_label"] = np.full(
                        len(indices), self.path_component[path], dtype=object
                    )
                    result["_protocol_label"] = np.asarray(
                        [row.get("protocol", "unknown") for row in rows], dtype=object
                    )
                    result["_family_label"] = np.asarray(
                        [row.get("family", "unknown") for row in rows], dtype=object
                    )
                    result["_arm_label"] = np.asarray(
                        [row.get("arm", "unknown") for row in rows], dtype=object
                    )
                yield result
            remaining -= len(take)


class FeatureTransform:
    def __init__(self, corpus: LeanSomaCorpus, config: MatchedTrainingConfig) -> None:
        self.corpus = corpus
        self.config = config
        metadata = corpus.metadata
        self.presence = np.asarray(metadata["mechanism_presence"][0], dtype=np.float32)
        self.static = np.asarray(metadata["segment_static"][0], dtype=np.float32)
        self.region_names = list(metadata["region_names"])
        self.region_id = int(metadata["segment_region_ids"][0])
        self.state_center = np.zeros(len(self.presence), dtype=np.float32)
        self.state_scale = np.ones(len(self.presence), dtype=np.float32)
        self.ion_center = np.zeros(len(metadata["ion_names"]), dtype=np.float32)
        self.ion_scale = np.ones(len(metadata["ion_names"]), dtype=np.float32)
        self.slices: Dict[str, slice] = {}

    @staticmethod
    def _robust(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(values, axis=0)
        q25, q75 = np.percentile(values, (25.0, 75.0), axis=0)
        scale = (q75 - q25) / 1.349
        std = np.std(values, axis=0)
        return center, np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))

    def fit(self) -> None:
        target = min(self.config.normalization_sample_limit, self.corpus.train_count)
        chunks = list(self.corpus.iter_raw(0, target))
        raw = {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in chunks[0]
        }
        state = np.log(
            np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6)
            / (1.0 - np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6))
        )
        active = self.presence.astype(bool)
        if np.any(active):
            self.state_center[active], self.state_scale[active] = self._robust(state[:, active])
        ions = np.log1p(np.maximum(raw["ion_state_t"], 0.0))
        if ions.shape[1]:
            self.ion_center, self.ion_scale = self._robust(ions)
        widths = {
            "axial_voltage": 3,
            "mechanism_state": len(self.presence),
            "mechanism_presence": len(self.presence),
            "causal_context": raw["causal_drive"].shape[1] + ions.shape[1],
            "segment_static": len(self.static),
            "region_one_hot": len(self.region_names),
        }
        start = 0
        for name, width in widths.items():
            self.slices[name] = slice(start, start + width)
            start += width
        if start % 2:
            self.slices["zero_padding"] = slice(start, start + 1)
            start += 1
        self.width = start
        if self.width != self.config.expected_input_width:
            raise RuntimeError(f"paper-scale common input width {self.width} != {self.config.expected_input_width}")

    def apply(self, raw: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        state = np.log(
            np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6)
            / (1.0 - np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6))
        )
        state = (state - self.state_center) / self.state_scale
        state[:, ~self.presence.astype(bool)] = 0.0
        ions = (np.log1p(np.maximum(raw["ion_state_t"], 0.0)) - self.ion_center) / self.ion_scale
        region = np.zeros((len(state), len(self.region_names)), dtype=np.float32)
        region[:, self.region_id] = 1.0
        blocks = [
            np.stack((raw["voltage_t_mv"], raw["parent_delta_t_mv"], raw["mean_child_delta_t_mv"]), axis=-1) / 100.0,
            state,
            np.broadcast_to(self.presence, state.shape),
            np.concatenate((raw["causal_drive"], ions), axis=-1),
            np.broadcast_to(self.static, (len(state), len(self.static))),
            region,
        ]
        if "zero_padding" in self.slices:
            blocks.append(np.zeros((len(state), 1), dtype=np.float32))
        features = np.concatenate(blocks, axis=-1).astype(np.float32)
        target = ((raw["voltage_t_plus_1_mv"] - raw["voltage_t_mv"]) / self.config.voltage_scale_mv).astype(np.float32)
        return features, target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit_split": "train",
            "state_transform": "logit_clip_1e-6",
            "ion_transform": "log1p_nonnegative",
            "state_center": self.state_center.tolist(),
            "state_scale": self.state_scale.tolist(),
            "ion_center": self.ion_center.tolist(),
            "ion_scale": self.ion_scale.tolist(),
            "feature_slices": {name: [value.start, value.stop] for name, value in self.slices.items()},
            "input_width": self.width,
        }


class PaperScaleMatchedTrainer:
    def __init__(self, corpus_root: Path, output_dir: Path, config: MatchedTrainingConfig, *, code_revision: str) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("paper-scale training requires PyTorch") from error
        config.validate()
        self.torch = torch
        from .corpus_audit import audit_soma_corpus

        self.corpus_contract = self._validate_corpus_contract(
            corpus_root,
            config,
            fingerprint_progress=self._report_corpus_fingerprint_progress,
        )
        self.support_preflight = audit_soma_corpus(corpus_root)
        train_support = self.support_preflight["splits"].get("train", {})
        validation_support = self.support_preflight["splits"].get(
            "validation", {}
        )
        blockers = []
        checks = (
            (
                "train |delta V| >= 5 mV",
                train_support.get("absolute_delta_ge_5mv_count", 0),
                config.minimum_train_active_transitions,
            ),
            (
                "validation |delta V| >= 5 mV",
                validation_support.get("absolute_delta_ge_5mv_count", 0),
                config.minimum_validation_active_transitions,
            ),
            (
                "validation somatic -55 mV upcrossings",
                validation_support.get("somatic_upcrossings_minus55mv", 0),
                config.minimum_validation_somatic_upcrossings,
            ),
        )
        for label, observed, required in checks:
            if int(observed) < int(required):
                blockers.append(
                    f"{label}: observed {int(observed)}, require {int(required)}"
                )
        self.support_preflight["training_support_blockers"] = blockers
        if blockers:
            raise RuntimeError(
                "matched training blocked before GPU use because corpus support "
                "is insufficient: " + "; ".join(blockers)
            )
        self.corpus = LeanSomaCorpus(corpus_root)
        self.output_dir = Path(output_dir)
        if (self.output_dir / "final_report.json").is_file():
            raise FileExistsError(
                f"completed result already exists in {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.code_revision = str(code_revision)
        self.transform = FeatureTransform(self.corpus, config)
        self.transform.fit()
        normalization_path = self.output_dir / "normalization.json"
        normalization_payload = self.transform.to_dict()
        if normalization_path.is_file():
            existing = json.loads(normalization_path.read_text(encoding="utf-8"))
            if existing != normalization_payload:
                raise RuntimeError(
                    "partial run normalization disagrees with the current corpus"
                )
        else:
            normalization_path.write_text(
                json.dumps(normalization_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("paper-scale matched training requires a CUDA GPU pod")

    @staticmethod
    def _report_corpus_fingerprint_progress(index: int, total: int) -> None:
        if index == 1 or index == total or index % 200 == 0:
            print(
                f"[GIADA RunPod][training corpus verification] {index}/{total} shards",
                flush=True,
            )

    @staticmethod
    def _validate_corpus_contract(
        corpus_root: Path,
        config: MatchedTrainingConfig,
        *,
        fingerprint_progress: Callable[[int, int], None] | None = None,
    ) -> Dict[str, Any]:
        root = Path(corpus_root)
        manifest_path = root / "composite_manifest.json"
        if config.required_composite_stage is None:
            return {"required_composite_stage": None, "verified": True}
        if not manifest_path.is_file():
            raise RuntimeError("matched training requires a composite corpus manifest")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not manifest.get("valid"):
            raise RuntimeError("composite corpus manifest is not valid")
        if manifest.get("stage") != config.required_composite_stage:
            raise RuntimeError(
                f"wrong composite stage {manifest.get('stage')!r}; "
                f"expected {config.required_composite_stage!r}"
            )
        audit_path = root / str(manifest.get("production_audit", "production_audit.json"))
        if not audit_path.is_file():
            raise RuntimeError("composite production audit is missing")
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        if not audit.get("valid") or audit.get("blockers"):
            raise RuntimeError("composite production audit did not pass")
        actual_hashes = {
            "composite_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "production_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        }
        fingerprint_report = None
        if config.expected_corpus_hashes:
            from .production_corpus import fingerprint_validated_shards

            # The complete component-level identity contract was introduced for
            # S2.  Historical S1e runs deliberately carry no frozen hash map and
            # remain verifiable through their sealed manifest and audit alone.
            components = {
                str(row["component_id"]): (root / str(row["root"])).resolve()
                for row in manifest.get("components", [])
            }
            for component_id in ("background", "targeted"):
                component_root = components.get(component_id)
                if component_root is None:
                    raise RuntimeError(
                        f"composite component {component_id!r} is missing"
                    )
                for filename, suffix in (
                    ("plan.json", "plan_sha256"),
                    ("validation_report.json", "validation_sha256"),
                ):
                    path = component_root / filename
                    if not path.is_file():
                        raise RuntimeError(
                            f"missing frozen corpus file {component_id}/{filename}"
                        )
                    actual_hashes[f"{component_id}_{suffix}"] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()

            fingerprint_report = fingerprint_validated_shards(
                root, progress=fingerprint_progress
            )
            actual_hashes["shard_marker_fingerprint_sha256"] = fingerprint_report[
                "marker_fingerprint_sha256"
            ]
            if not fingerprint_report["valid"]:
                raise RuntimeError(
                    "one or more physical frozen shards no longer match their completion marker"
                )
            mismatches = {
                key: {"expected": expected, "observed": actual_hashes.get(key)}
                for key, expected in config.expected_corpus_hashes.items()
                if actual_hashes.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"frozen corpus hash mismatch: {mismatches}")
        return {
            "required_composite_stage": config.required_composite_stage,
            "verified": True,
            "manifest_sha256": actual_hashes["composite_manifest_sha256"],
            "production_audit_sha256": actual_hashes["production_audit_sha256"],
            "verified_corpus_hashes": actual_hashes,
            "shard_fingerprint_report": fingerprint_report,
            "total_transition_count": manifest.get("total_transition_count"),
            "split_transition_counts": manifest.get("split_transition_counts"),
        }

    @staticmethod
    def _trainable_count(model: Any) -> int:
        return sum(value.numel() for value in model.parameters() if value.requires_grad)

    def _models(self) -> Dict[str, Any]:
        meta = self.corpus.metadata
        bridge = CausalVoltageBridge(
            state_width=len(meta["mechanism_group_names"]),
            presence_width=len(meta["mechanism_group_names"]),
            context_width=len(meta["causal_drive_features"]) + len(meta["ion_names"]),
            static_width=len(meta["segment_static"][0]),
            region_count=len(meta["region_names"]),
            region_embedding_width=8,
            hidden_width=64,
            normalized_delta_limit=self.config.delta_limit_mv / self.config.voltage_scale_mv,
        )
        models = {
            "branch_elm_core": InformationMatchedBranchELM(
                self.transform.width,
                self.config.delta_limit_mv / self.config.voltage_scale_mv,
            ),
            "giada_voltage_bridge": InformationMatchedBridgeAdapter(
                bridge, self.transform.slices
            ),
        }
        counts = {name: self._trainable_count(model) for name, model in models.items()}
        expected = {
            "branch_elm_core": self.config.expected_branch_elm_parameters,
            "giada_voltage_bridge": self.config.expected_giada_parameters,
        }
        if counts != expected:
            raise RuntimeError(f"matched parameter contract changed: {counts} != {expected}")
        return {name: model.to(self.device) for name, model in models.items()}

    @staticmethod
    def _empty_metric_state() -> Dict[str, float | int]:
        return {"squared": 0.0, "persistence_squared": 0.0, "count": 0}

    @staticmethod
    def _update_metric_state(
        state: Dict[str, float | int],
        error: np.ndarray,
        target_mv: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if not np.any(mask):
            return
        state["squared"] += float(np.sum(error[mask].astype(np.float64) ** 2))
        state["persistence_squared"] += float(
            np.sum(target_mv[mask].astype(np.float64) ** 2)
        )
        state["count"] += int(np.count_nonzero(mask))

    @staticmethod
    def _finish_metric_state(state: Mapping[str, float | int]) -> Dict[str, Any]:
        count = int(state["count"])
        if count == 0:
            return {
                "soma_rmse_mv": None,
                "persistence_soma_rmse_mv": None,
                "improvement_vs_persistence_fraction": None,
                "example_count": 0,
            }
        rmse = math.sqrt(float(state["squared"]) / count)
        persistence = math.sqrt(float(state["persistence_squared"]) / count)
        return {
            "soma_rmse_mv": rmse,
            "persistence_soma_rmse_mv": persistence,
            "improvement_vs_persistence_fraction": 1.0
            - rmse / max(persistence, 1e-12),
            "example_count": count,
        }

    def _evaluate(self, model: Any) -> Dict[str, Any]:
        squared = 0.0
        persistence_squared = 0.0
        active_squared = 0.0
        active_count = 0
        count = 0
        component_states: Dict[str, Dict[str, float | int]] = {}
        protocol_states: Dict[str, Dict[str, float | int]] = {}
        family_states: Dict[str, Dict[str, float | int]] = {}
        activity_states = {
            "quiescent_abs_delta_lt_1mv": self._empty_metric_state(),
            "moderate_abs_delta_1_to_5mv": self._empty_metric_state(),
            "active_abs_delta_ge_5mv": self._empty_metric_state(),
            "somatic_upcrossing_minus55mv": self._empty_metric_state(),
        }
        model.eval()
        with self.torch.no_grad():
            for raw in self.corpus.iter_raw(
                1, self.config.evaluation_sample_limit, include_labels=True
            ):
                features, target = self.transform.apply(raw)
                prediction = model(self.torch.as_tensor(features, device=self.device)).cpu().numpy() * self.config.voltage_scale_mv
                target_mv = target * self.config.voltage_scale_mv
                error = prediction - target_mv
                squared += float(np.sum(error.astype(np.float64) ** 2))
                persistence_squared += float(np.sum(target_mv.astype(np.float64) ** 2))
                active = np.abs(target_mv) >= self.config.active_delta_threshold_mv
                active_squared += float(np.sum(error[active].astype(np.float64) ** 2))
                active_count += int(active.sum())
                count += len(error)
                absolute = np.abs(target_mv)
                activity_masks = {
                    "quiescent_abs_delta_lt_1mv": absolute < 1.0,
                    "moderate_abs_delta_1_to_5mv": (absolute >= 1.0)
                    & (absolute < 5.0),
                    "active_abs_delta_ge_5mv": active,
                    "somatic_upcrossing_minus55mv": (
                        (raw["voltage_t_mv"] < -55.0)
                        & (raw["voltage_t_plus_1_mv"] >= -55.0)
                    ),
                }
                for label, mask in activity_masks.items():
                    self._update_metric_state(
                        activity_states[label], error, target_mv, mask
                    )
                for labels, states in (
                    (raw["_component_label"], component_states),
                    (raw["_protocol_label"], protocol_states),
                    (raw["_family_label"], family_states),
                ):
                    for label in np.unique(labels):
                        self._update_metric_state(
                            states.setdefault(str(label), self._empty_metric_state()),
                            error,
                            target_mv,
                            labels == label,
                        )
        rmse = math.sqrt(squared / count)
        persistence = math.sqrt(persistence_squared / count)
        return {
            "soma_rmse_mv": rmse,
            "persistence_soma_rmse_mv": persistence,
            "improvement_vs_persistence_fraction": 1.0 - rmse / max(persistence, 1e-12),
            # An empty stratum is unsupported, never a perfect zero-error score.
            "active_soma_rmse_mv": (
                math.sqrt(active_squared / active_count) if active_count else None
            ),
            "active_count": active_count,
            "example_count": count,
            "activity_regime_metrics": {
                label: self._finish_metric_state(state)
                for label, state in activity_states.items()
            },
            "component_metrics": {
                label: self._finish_metric_state(state)
                for label, state in sorted(component_states.items())
            },
            "protocol_family_metrics": {
                label: self._finish_metric_state(state)
                for label, state in sorted(family_states.items())
            },
            "protocol_metrics": {
                label: self._finish_metric_state(state)
                for label, state in sorted(protocol_states.items())
            },
        }

    def train(self) -> Dict[str, Any]:
        runs = []
        resumed_seeds = []
        configuration_payload = json.loads(json.dumps(asdict(self.config)))
        for seed in self.config.seeds:
            seed_report_path = self.output_dir / f"seed{seed}_completed.json"
            if seed_report_path.is_file():
                seed_report = json.loads(
                    seed_report_path.read_text(encoding="utf-8")
                )
                if (
                    seed_report.get("code_revision") != self.code_revision
                    or seed_report.get("configuration") != configuration_payload
                ):
                    raise RuntimeError(
                        f"completed seed {seed} belongs to a different frozen run"
                    )
                seed_runs = seed_report.get("runs", [])
                expected_rows = 2 * len(self.config.checkpoints)
                if len(seed_runs) != expected_rows:
                    raise RuntimeError(f"completed seed {seed} report is incomplete")
                runs.extend(seed_runs)
                resumed_seeds.append(seed)
                print(
                    f"[GIADA RunPod][matched seed={seed}] resumed completed seed",
                    flush=True,
                )
                continue
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
            models = self._models()
            optimizers = {
                name: self.torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
                for name, model in models.items()
            }
            rng = np.random.default_rng(seed)
            started = time.perf_counter()
            for step in range(1, self.config.training_steps + 1):
                raw = self.corpus.sample_raw(0, self.config.batch_size, rng)
                features, target = self.transform.apply(raw)
                x = self.torch.as_tensor(features, device=self.device)
                y = self.torch.as_tensor(target, device=self.device)
                active = self.torch.abs(y * self.config.voltage_scale_mv) >= self.config.active_delta_threshold_mv
                weight = self.torch.where(active, self.config.active_weight, 1.0)
                losses = {}
                for name, model in models.items():
                    model.train()
                    optimizers[name].zero_grad(set_to_none=True)
                    prediction = model(x)
                    loss = self.torch.mean(weight * (prediction - y) ** 2)
                    loss.backward()
                    self.torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    optimizers[name].step()
                    losses[name] = float(loss.detach().cpu())
                if step in self.config.checkpoints:
                    for name, model in models.items():
                        metrics = self._evaluate(model)
                        runs.append({"seed": seed, "model": name, "step": step, **metrics})
                        self.torch.save(
                            {
                                "seed": seed,
                                "model": name,
                                "step": step,
                                "state_dict": model.state_dict(),
                                "normalization": self.transform.to_dict(),
                            },
                            self.output_dir / f"{name}_seed{seed}_step{step}.pt",
                        )
                if step == 1 or step == self.config.training_steps or step % self.config.progress_interval == 0:
                    eta = (time.perf_counter() - started) / step * (self.config.training_steps - step)
                    compact = " ".join(f"{name}={value:.4g}" for name, value in losses.items())
                    print(f"[GIADA RunPod][matched seed={seed}] {step}/{self.config.training_steps} ETA {eta/60:.1f} min {compact}", flush=True)
            for name, model in models.items():
                self.torch.save(
                    {
                        "seed": seed,
                        "model": name,
                        "step": self.config.training_steps,
                        "state_dict": model.state_dict(),
                        "normalization": self.transform.to_dict(),
                    },
                    self.output_dir / f"{name}_seed{seed}.pt",
                )
            seed_runs = [row for row in runs if row["seed"] == seed]
            seed_payload = {
                "schema_version": "giada-paper-scale-matched-seed-v1",
                "code_revision": self.code_revision,
                "configuration": configuration_payload,
                "seed": seed,
                "runs": seed_runs,
            }
            temporary = seed_report_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(seed_payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(seed_report_path)
        final = [row for row in runs if row["step"] == self.config.training_steps]
        medians = {
            name: float(np.median([row["soma_rmse_mv"] for row in final if row["model"] == name]))
            for name in ("branch_elm_core", "giada_voltage_bridge")
        }

        def nested_value(row: Mapping[str, Any], *path: str) -> Any:
            value: Any = row
            for key in path:
                value = value[key]
            return value

        def paired_medians(*path: str) -> Dict[str, float | None]:
            result: Dict[str, float | None] = {}
            for name in ("branch_elm_core", "giada_voltage_bridge"):
                values = [
                    nested_value(row, *path)
                    for row in final
                    if row["model"] == name
                ]
                finite = [float(value) for value in values if value is not None]
                result[name] = float(np.median(finite)) if finite else None
            return result

        active_medians = paired_medians("active_soma_rmse_mv")
        spike_medians = paired_medians(
            "activity_regime_metrics",
            "somatic_upcrossing_minus55mv",
            "soma_rmse_mv",
        )
        component_medians = {
            component: paired_medians(
                "component_metrics", component, "soma_rmse_mv"
            )
            for component in ("background", "targeted")
        }

        def stratified_medians(section: str) -> Dict[str, Dict[str, float | None]]:
            labels = sorted({
                label
                for row in final
                for label in row.get(section, {})
            })
            return {
                label: paired_medians(section, label, "soma_rmse_mv")
                for label in labels
            }

        family_medians = stratified_medians("protocol_family_metrics")
        protocol_medians = stratified_medians("protocol_metrics")
        seed_wins = {
            str(seed): next(
                row["soma_rmse_mv"]
                for row in final
                if row["seed"] == seed and row["model"] == "giada_voltage_bridge"
            )
            < next(
                row["soma_rmse_mv"]
                for row in final
                if row["seed"] == seed and row["model"] == "branch_elm_core"
            )
            for seed in self.config.seeds
        }
        primary_passed = (
            medians["giada_voltage_bridge"] < medians["branch_elm_core"]
        )
        active_passed = bool(
            active_medians["giada_voltage_bridge"] is not None
            and active_medians["branch_elm_core"] is not None
            and active_medians["giada_voltage_bridge"]
            < active_medians["branch_elm_core"]
        )
        spike_transition_passed = bool(
            spike_medians["giada_voltage_bridge"] is not None
            and spike_medians["branch_elm_core"] is not None
            and spike_medians["giada_voltage_bridge"]
            < spike_medians["branch_elm_core"]
        )
        seed_robustness_passed = (
            sum(seed_wins.values()) >= self.config.minimum_seed_wins
        )
        family_wins = {
            label: bool(
                values["giada_voltage_bridge"] is not None
                and values["branch_elm_core"] is not None
                and values["giada_voltage_bridge"] < values["branch_elm_core"]
            )
            for label, values in family_medians.items()
        }
        protocol_wins = {
            label: bool(
                values["giada_voltage_bridge"] is not None
                and values["branch_elm_core"] is not None
                and values["giada_voltage_bridge"] < values["branch_elm_core"]
            )
            for label, values in protocol_medians.items()
        }
        breadth_passed = (
            sum(family_wins.values()) >= self.config.minimum_family_wins
            and sum(protocol_wins.values()) >= self.config.minimum_protocol_wins
        )

        def seed_dispersion(*path: str) -> Dict[str, Dict[str, float | int | None]]:
            result = {}
            for name in ("branch_elm_core", "giada_voltage_bridge"):
                values = [
                    nested_value(row, *path)
                    for row in final
                    if row["model"] == name
                ]
                finite = np.asarray(
                    [float(value) for value in values if value is not None],
                    dtype=np.float64,
                )
                result[name] = {
                    "mean": float(np.mean(finite)) if len(finite) else None,
                    "sample_standard_deviation": (
                        float(np.std(finite, ddof=1)) if len(finite) > 1 else None
                    ),
                    "median": float(np.median(finite)) if len(finite) else None,
                    "seed_count": int(len(finite)),
                }
            return result

        branch_by_seed = np.asarray([
            next(
                row["soma_rmse_mv"] for row in final
                if row["seed"] == seed and row["model"] == "branch_elm_core"
            )
            for seed in self.config.seeds
        ], dtype=np.float64)
        giada_by_seed = np.asarray([
            next(
                row["soma_rmse_mv"] for row in final
                if row["seed"] == seed and row["model"] == "giada_voltage_bridge"
            )
            for seed in self.config.seeds
        ], dtype=np.float64)
        paired_gain = branch_by_seed - giada_by_seed
        try:
            from scipy.stats import ttest_rel

            paired_t = ttest_rel(branch_by_seed, giada_by_seed, alternative="greater")
            paired_t_statistic = float(paired_t.statistic)
            paired_t_pvalue = float(paired_t.pvalue)
            if not math.isfinite(paired_t_statistic):
                paired_t_statistic = None
            if not math.isfinite(paired_t_pvalue):
                paired_t_pvalue = None
        except (ImportError, TypeError):  # pragma: no cover - old SciPy fallback
            paired_t_statistic = None
            paired_t_pvalue = None
        bootstrap_rng = np.random.default_rng(20_260_906)
        bootstrap_indices = bootstrap_rng.integers(
            0, len(paired_gain), size=(20_000, len(paired_gain))
        )
        bootstrap_means = np.mean(paired_gain[bootstrap_indices], axis=1)
        inference = {
            "unit": "independent_training_seed_on_the_same_validation_corpus",
            "paired_difference": "branch_elm_rmse_minus_giada_rmse_mv",
            "mean_difference_mv": float(np.mean(paired_gain)),
            "sample_standard_deviation_mv": (
                float(np.std(paired_gain, ddof=1)) if len(paired_gain) > 1 else None
            ),
            "bootstrap_95_percent_ci_mean_difference_mv": [
                float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])
            ],
            "paired_t_test_one_sided_giada_lower": {
                "statistic": paired_t_statistic,
                "p_value": paired_t_pvalue,
            },
            "interpretation_limit": (
                "This quantifies optimization-seed variability; it does not replace "
                "a fresh independent teacher-trajectory test."
            ),
        }
        scaling_reference = None
        if self.config.scaling_reference_seeds:
            reference = set(self.config.scaling_reference_seeds)
            reference_runs = [row for row in runs if row["seed"] in reference]
            scaling_reference = {
                "seeds": list(self.config.scaling_reference_seeds),
                "final_median_soma_rmse_mv": {
                    name: float(np.median([
                        row["soma_rmse_mv"] for row in reference_runs
                        if row["model"] == name
                        and row["step"] == self.config.training_steps
                    ]))
                    for name in ("branch_elm_core", "giada_voltage_bridge")
                },
                "mini_scaling_law": [
                    {
                        "step": step,
                        **{
                            name: float(np.median([
                                row["soma_rmse_mv"] for row in reference_runs
                                if row["model"] == name and row["step"] == step
                            ]))
                            for name in ("branch_elm_core", "giada_voltage_bridge")
                        },
                    }
                    for step in self.config.checkpoints
                ],
            }
        all_registered_passed = (
            primary_passed
            and active_passed
            and seed_robustness_passed
            and breadth_passed
            and (
                spike_transition_passed
                or not self.config.require_spike_transition_advantage
            )
        )
        stage = self.config.required_composite_stage
        is_s1e = stage == "s1e_hybrid_production"
        is_s2 = stage == "s2_hybrid_production"
        is_s3 = stage == "s3_hybrid_production"
        report = {
            "schema_version": "giada-paper-scale-matched-training-v1",
            "valid": True,
            "code_revision": self.code_revision,
            "device": str(self.device),
            "same_numeric_input": True,
            "same_target": "authentic_NEURON_one_ms_soma_voltage_transition",
            "same_sample_order": True,
            "same_optimizer_and_loss": True,
            "rollout_claimed": False,
            "checkpoint_selection": self.config.checkpoint_selection,
            "validation_used_for_checkpoint_selection": False,
            "corpus_contract": self.corpus_contract,
            "train_transition_count": self.corpus.train_count,
            "validation_transition_count": self.corpus.validation_count,
            "configuration": asdict(self.config),
            "corpus_support_preflight": self.support_preflight,
            "mini_scaling_law_runs": runs,
            "final_median_soma_rmse_mv": medians,
            "final_median_active_soma_rmse_mv": active_medians,
            "final_median_spike_transition_rmse_mv": spike_medians,
            "final_median_component_rmse_mv": component_medians,
            "final_median_protocol_family_rmse_mv": family_medians,
            "final_median_protocol_rmse_mv": protocol_medians,
            "final_seed_wins": seed_wins,
            "final_family_wins": family_wins,
            "final_protocol_wins": protocol_wins,
            "seed_dispersion": {
                "overall_soma_rmse_mv": seed_dispersion("soma_rmse_mv"),
                "active_soma_rmse_mv": seed_dispersion("active_soma_rmse_mv"),
                "somatic_upcrossing_rmse_mv": seed_dispersion(
                    "activity_regime_metrics",
                    "somatic_upcrossing_minus55mv",
                    "soma_rmse_mv",
                ),
            },
            "paired_statistical_inference": inference,
            "common_seed_scaling_reference": scaling_reference,
            "giada_relative_rmse_reduction_vs_branch_elm": 1.0 - medians["giada_voltage_bridge"] / medians["branch_elm_core"],
            "registered_decision": {
                "primary_overall_median_passed": primary_passed,
                "active_stratum_median_passed": active_passed,
                "spike_transition_advantage_required": (
                    self.config.require_spike_transition_advantage
                ),
                "spike_transition_median_passed": spike_transition_passed,
                "minimum_seed_wins_required": self.config.minimum_seed_wins,
                "observed_seed_wins": int(sum(seed_wins.values())),
                "seed_robustness_passed": seed_robustness_passed,
                "at_least_two_of_three_seed_wins": (
                    seed_robustness_passed
                    if len(self.config.seeds) == 3
                    and self.config.minimum_seed_wins == 2
                    else None
                ),
                "minimum_family_wins_required": self.config.minimum_family_wins,
                "observed_family_wins": int(sum(family_wins.values())),
                "minimum_protocol_wins_required": self.config.minimum_protocol_wins,
                "observed_protocol_wins": int(sum(protocol_wins.values())),
                "breadth_passed": breadth_passed,
                "all_registered_gates_passed": all_registered_passed,
                "s1e_advantage_confirmed": (
                    all_registered_passed if is_s1e else None
                ),
                "s2_advantage_confirmed": (
                    all_registered_passed if is_s2 else None
                ),
                "s3_advantage_confirmed": (
                    all_registered_passed if is_s3 else None
                ),
                "s2_authorized": (
                    all_registered_passed if is_s1e else None
                ),
                "s3_authorized": (
                    all_registered_passed if is_s2 else None
                ),
                "s4_authorized": (
                    all_registered_passed if is_s3 else None
                ),
            },
            "resumed_completed_seeds": resumed_seeds,
        }
        final_path = self.output_dir / "final_report.json"
        temporary = final_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(final_path)
        return report
