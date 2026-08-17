"""05j-n: refit the registered direct-tree decoder on expanded train support."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..hayflow_data.composite_flowmap import CompositeShard, CompositeTransitionStore
from ..hayflow_data.flowmap_dataset import FlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_regenerative_confirmation import (
    _IndependentBundle,
    _verified_artifact_root,
)
from .hines_repaired_representation_revision import pair_gate_selection_score
from .hines_residual_safety_gate import HinesResidualSafetyGateCanary
from .hines_spatial_support_revision import (
    apply_channel_pca,
    axial_tree_diffusion,
    deterministic_pca_components,
)
from .hines_trainable_topology_canary import TrainableTopologyResidualHead

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05JL_ARCHIVE_SHA256 = (
    "f4685a3322dd868aaf2c271e7e8cb949e5ba81ca8d6b3631eb470fe473adb40c"
)
EXPECTED_05JL_INDEX_SHA256 = (
    "dbe30639cb88dfafcf2349a56d91489bf765aa425c0d47c35fefb973cef58c82"
)
EXPECTED_05JL_FINAL_SHA256 = (
    "2eed425e145307ad6f4a93f035a21490ae749397163281b3b61b22e14cb785f0"
)
EXPECTED_05JM_ARCHIVE_SHA256 = (
    "70010063448be6de8c7478037398f58fbd2d4cd8d81a14fade1d1561b47bcf20"
)
EXPECTED_05JM_INDEX_SHA256 = (
    "90a46ddab7d27cc00f70e177f30ba97e8147a14f8fca390b809ed8eb81c868a1"
)
EXPECTED_05JM_FINAL_SHA256 = (
    "27eb696bfee58efa118a6929d8630ccc8141f558ed42d9394d6e0e8a00fed795"
)
EXPECTED_05JM_TRANSITION_SHA256 = (
    "97eb347e5348cfcb2198352dd654a09104cd78acfc790433ff471b42e60ab302"
)

_05JM_TRAINING_MEMBERS = frozenset(
    {
        "acquisition_contract.json",
        "artifact_index.json",
        "branching_pairs.parquet",
        "dataset_manifest.json",
        "dataset_card.json",
        "episodes.parquet",
        "event_definition_config.json",
        "events.parquet",
        "final_report.json",
        "manifest.json",
        "release_outcomes.parquet",
        "segments.parquet",
        "splits.json",
        "state_schema.json",
        "synapses.parquet",
        "training_pairs.parquet",
        "transition_dataset.h5",
        "validation_report.json",
    }
)


def _stream_sha256(handle: Any, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        block = handle.read(block_size)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _training_cache_stamp(source_fingerprint: str) -> str:
    payload = {
        "source_fingerprint": str(source_fingerprint),
        "materialized_members": sorted(_05JM_TRAINING_MEMBERS),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _verified_training_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """Verify every 05j-m byte but materialize training members only.

    The sealed fresh-test plan is hashed as an opaque archive member for
    provenance.  It is never parsed, copied, or exposed below the returned
    root, so downstream training code cannot access its inputs accidentally.
    """

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    observed_archive = None
    indexed_count = 0
    if source.is_file():
        observed_archive = sha256_file(source)
        if observed_archive != EXPECTED_05JM_ARCHIVE_SHA256:
            raise RuntimeError("05j-m archive SHA-256 mismatch")
        with zipfile.ZipFile(source) as archive:
            names = [name.replace("\\", "/") for name in archive.namelist()]
            indices = [name for name in names if name.endswith("/artifact_index.json")]
            matches = []
            for index_name in indices:
                prefix = index_name[: -len("artifact_index.json")]
                if prefix + "acquisition_contract.json" in names:
                    matches.append((index_name, prefix))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one 05j-m artifact in ZIP; found {len(matches)}"
                )
            index_name, prefix = matches[0]
            index_bytes = archive.read(index_name)
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JM_INDEX_SHA256:
                raise RuntimeError("05j-m artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            indexed_count = len(index["artifacts"])
            name_lookup = {name.replace("\\", "/"): name for name in archive.namelist()}
            for row in index["artifacts"]:
                relative = str(row["path"]).replace("\\", "/")
                member_name = prefix + relative
                if member_name not in name_lookup:
                    raise RuntimeError(f"05j-m indexed member missing: {relative}")
                info = archive.getinfo(name_lookup[member_name])
                with archive.open(info) as handle:
                    observed = _stream_sha256(handle)
                if info.file_size != int(row["size_bytes"]) or observed != str(
                    row["sha256"]
                ):
                    raise RuntimeError(f"05j-m indexed member mismatch: {relative}")
            cache_stamp = _training_cache_stamp(observed_archive)
            marker = cache_dir / ".training_source_sha256"
            if not marker.is_file() or marker.read_text().strip() != cache_stamp:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                root = cache_dir / "hayflow_regenerative_training_support"
                root.mkdir(parents=True)
                for basename in sorted(_05JM_TRAINING_MEMBERS):
                    member_name = prefix + basename
                    if member_name not in name_lookup:
                        raise RuntimeError(f"05j-m required training member missing: {basename}")
                    with archive.open(name_lookup[member_name]) as src, (
                        root / basename
                    ).open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                marker.write_text(cache_stamp + "\n")
            root = cache_dir / "hayflow_regenerative_training_support"
        source_kind = "original_zip_training_members_only"
    elif source.is_dir():
        indices = [
            path
            for path in source.rglob("artifact_index.json")
            if (path.parent / "acquisition_contract.json").is_file()
        ]
        if len(indices) != 1:
            raise RuntimeError(
                f"expected one extracted 05j-m artifact; found {len(indices)}"
            )
        source_root = indices[0].parent
        index_bytes = indices[0].read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JM_INDEX_SHA256:
            raise RuntimeError("05j-m artifact index SHA-256 mismatch")
        index = json.loads(index_bytes)
        indexed_count = len(index["artifacts"])
        for row in index["artifacts"]:
            path = source_root / str(row["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(row["size_bytes"])
                or sha256_file(path) != str(row["sha256"])
            ):
                raise RuntimeError(f"05j-m indexed member mismatch: {row['path']}")
        stamp = _training_cache_stamp(EXPECTED_05JM_INDEX_SHA256)
        marker = cache_dir / ".training_source_sha256"
        if not marker.is_file() or marker.read_text().strip() != stamp:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            root = cache_dir / "hayflow_regenerative_training_support"
            root.mkdir(parents=True)
            for basename in sorted(_05JM_TRAINING_MEMBERS):
                path = source_root / basename
                if not path.is_file():
                    raise RuntimeError(f"05j-m required training member missing: {basename}")
                shutil.copy2(path, root / basename)
            marker.write_text(stamp + "\n")
        root = cache_dir / "hayflow_regenerative_training_support"
        source_kind = "kaggle_directory_training_members_only"
    else:
        raise RuntimeError(f"05j-m artifact source does not exist: {source}")

    if (root / "sealed_fresh_test_plan.json").exists():
        raise RuntimeError("05j-m filtered training root exposes the sealed fresh test")
    final_bytes = (root / "final_report.json").read_bytes()
    if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JM_FINAL_SHA256:
        raise RuntimeError("05j-m final report SHA-256 mismatch")
    contract = {
        "source_kind": source_kind,
        "source_path": str(source),
        "archive_sha256": observed_archive,
        "artifact_index_sha256": EXPECTED_05JM_INDEX_SHA256,
        "final_report_sha256": EXPECTED_05JM_FINAL_SHA256,
        "verified_member_count": indexed_count,
        "all_indexed_members_verified": True,
        "fresh_test_member_extracted": False,
        "fresh_test_member_parsed": False,
    }
    return root, json.loads(final_bytes), contract


@dataclass(frozen=True)
class HinesRegenerativeDecoderRefitConfig:
    old_fit_calibration_pair_count: int = 11
    new_train_calibration_pair_count: int = 19
    expected_old_fit_pair_count: int = 54
    expected_new_train_pair_count: int = 96
    seeds: Tuple[int, ...] = (17, 29, 43)
    epochs: int = 1200
    evaluation_interval: int = 20
    patience: int = 240
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    minimum_internal_improvement_vs_h2_fraction: float = 0.10
    minimum_development_improvement_vs_best_baseline_fraction: float = 0.05
    minimum_development_branching_retention: float = 0.5
    maximum_development_branching_retention: float = 2.0
    maximum_development_error_ratio_vs_h2: float = 1.0
    minimum_passing_seeds: int = 2

    def validate(self) -> None:
        if self.expected_old_fit_pair_count != 54 or self.expected_new_train_pair_count != 96:
            raise ValueError("05j-n pair cardinalities are preregistered")
        if not 0 < self.old_fit_calibration_pair_count < self.expected_old_fit_pair_count:
            raise ValueError("old-fit internal calibration count is invalid")
        if not 0 < self.new_train_calibration_pair_count < self.expected_new_train_pair_count:
            raise ValueError("new-train internal calibration count is invalid")
        if tuple(self.seeds) != (17, 29, 43):
            raise ValueError("05j-n uses the registered three-seed ensemble")
        if min(
            self.epochs,
            self.evaluation_interval,
            self.patience,
            self.learning_rate,
            self.gradient_clip_norm,
        ) <= 0:
            raise ValueError("positive refit values must be positive")
        if self.patience < self.evaluation_interval:
            raise ValueError("patience is below one evaluation interval")
        fractions = (
            self.minimum_internal_improvement_vs_h2_fraction,
            self.minimum_development_improvement_vs_best_baseline_fraction,
        )
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("improvement fractions must lie in (0, 1)")
        if not (
            0 < self.minimum_development_branching_retention
            < self.maximum_development_branching_retention
        ):
            raise ValueError("development branching-retention interval is invalid")
        if self.maximum_development_error_ratio_vs_h2 <= 0:
            raise ValueError("development error ratio must be positive")
        if not 1 <= self.minimum_passing_seeds <= len(self.seeds):
            raise ValueError("passing-seed gate is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRegenerativeDecoderRefitConfig":
        payload = dict(values)
        if "seeds" in payload:
            payload["seeds"] = tuple(map(int, payload["seeds"]))
        result = cls(**payload)
        result.validate()
        return result


def deterministic_pair_partition(
    pair_ids: Sequence[str], calibration_count: int, *, salt: str
) -> Dict[str, List[int]]:
    """Split complete pairs deterministically without inspecting outcomes."""

    identities = list(map(str, pair_ids))
    if len(set(identities)) != len(identities):
        raise ValueError("pair identities must be unique")
    if not 0 < int(calibration_count) < len(identities):
        raise ValueError("calibration count must leave non-empty fit support")
    ranked = sorted(
        range(len(identities)),
        key=lambda index: (
            hashlib.sha256(f"{salt}:{identities[index]}".encode()).hexdigest(),
            identities[index],
        ),
    )
    calibration = sorted(ranked[: int(calibration_count)])
    fit = sorted(set(range(len(identities))) - set(calibration))
    return {"fit_pair_positions": fit, "calibration_pair_positions": calibration}


def _sample_positions(pair_positions: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [position for pair in pair_positions for position in (2 * pair, 2 * pair + 1)],
        dtype=np.int64,
    )


def _subset_role(role: Mapping[str, Any], positions: Sequence[int]) -> Dict[str, Any]:
    positions = np.asarray(positions, dtype=np.int64)
    sample_count = len(np.asarray(role["base"]))
    result: Dict[str, Any] = {}
    for name, value in role.items():
        array = np.asarray(value)
        result[name] = array[positions] if array.ndim and len(array) == sample_count else value
    return result


def _concatenate_roles(roles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not roles:
        raise ValueError("at least one role is required")
    result: Dict[str, Any] = {}
    for name in roles[0]:
        arrays = [np.asarray(role[name]) for role in roles]
        sample_counts = [len(np.asarray(role["base"])) for role in roles]
        if all(array.ndim and len(array) == count for array, count in zip(arrays, sample_counts)):
            result[name] = np.concatenate(arrays, axis=0)
        else:
            result[name] = roles[0][name]
    return result


class _TrainingSupportTransitionStore(CompositeTransitionStore):
    def _validate_contract(self) -> None:
        splits = sorted({str(row["split"]) for row in self.episode_rows})
        trajectories = [str(row["trajectory_id"]) for row in self.episode_rows]
        blockers = []
        if self.count != 2304:
            blockers.append("05j-m transition count is not 2304")
        if len(self.episode_rows) != 192:
            blockers.append("05j-m episode count is not 192")
        if splits != ["train"]:
            blockers.append("05j-m is not train-only")
        if len(set(trajectories)) != len(trajectories):
            blockers.append("05j-m repeats a trajectory id")
        self.contract_checks = {
            "transition_count_exact": self.count == 2304,
            "episode_count_exact": len(self.episode_rows) == 192,
            "train_only": splits == ["train"],
            "trajectory_ids_unique": len(set(trajectories)) == len(trajectories),
            "blockers": blockers,
        }
        if blockers:
            raise RuntimeError(f"05j-m store contract failed: {blockers}")

    def _shard_for(self, logical_index: int) -> CompositeShard:
        if not 0 <= int(logical_index) < self.count:
            raise IndexError(logical_index)
        return self.bundle.shard

    def report(self) -> Dict[str, Any]:
        return {
            "valid": not self.contract_checks["blockers"],
            "dataset_kind": "train_only_near_regenerative_shard",
            "fingerprint": self.bundle.fingerprint,
            "episode_count": len(self.episode_rows),
            "transition_count": self.count,
            "contract_checks": self.contract_checks,
            "state_loading": "lazy_single_shard",
        }


class HinesRegenerativeDecoderRefit(HinesResidualSafetyGateCanary):
    """Refit only the registered decoder; keep H2 and fresh test frozen."""

    def __init__(
        self,
        *args: Any,
        refit_config: HinesRegenerativeDecoderRefitConfig,
        artifact_05jl_source: Path,
        artifact_05jm_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        refit_config.validate()
        self.refit = refit_config
        self.artifact_05jl_source = Path(artifact_05jl_source).resolve()
        self.artifact_05jm_source = Path(artifact_05jm_source).resolve()
        self.artifact_05jl_contract: Dict[str, Any] = {}
        self.artifact_05jm_contract: Dict[str, Any] = {}
        self.artifact_05jm_report: Dict[str, Any] = {}
        self.training_store: _TrainingSupportTransitionStore | None = None
        self.training_pair_rows: List[Dict[str, Any]] = []
        self.refit_roles: Dict[str, Dict[str, Any]] = {}
        self.refit_designs: Dict[str, np.ndarray] = {}
        self.refit_transform: Dict[str, Any] = {}

    def prepare_regenerative_decoder_refit(self) -> Dict[str, Any]:
        base = self.prepare_residual_safety_gate()
        cache = self.output_dir.parent / ".05j_n_artifact_cache"
        _, report_l, contract_l = _verified_artifact_root(
            self.artifact_05jl_source,
            cache / "05jl",
            marker_name="residual_safety_gate_config.json",
            archive_sha256=EXPECTED_05JL_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JL_INDEX_SHA256,
            final_sha256=EXPECTED_05JL_FINAL_SHA256,
        )
        root_m, report_m, contract_m = _verified_training_artifact_root(
            self.artifact_05jm_source, cache / "05jm"
        )
        blockers = []
        if report_l.get("diagnosis") != "FIT_ONLY_SAFETY_GATE_DOES_NOT_RESCUE_EXTERNAL_DECODER":
            blockers.append("05j-l diagnosis does not route to decoder refit")
        if report_l.get("candidate_authorized") is not False:
            blockers.append("05j-l unexpectedly authorized a candidate")
        if report_m.get("diagnosis") != "TRAIN_SUPPORT_ACQUIRED_FRESH_TEST_PLAN_SEALED":
            blockers.append("05j-m diagnosis is unexpected")
        if not report_m.get("scientific_train_support_sufficient"):
            blockers.append("05j-m train support is insufficient")
        if int(report_m.get("realized_train_stratum_counts", {}).get("near_regenerative", 0)) < 72:
            blockers.append("05j-m near-regenerative support is below the floor")
        fresh = report_m.get("fresh_test", {})
        if fresh.get("outcomes_generated") is not False or not fresh.get("seeds_absent_from_training_shard"):
            blockers.append("05j-m fresh test is not sealed")
        transition = root_m / "transition_dataset.h5"
        if (
            not transition.is_file()
            or sha256_file(transition) != EXPECTED_05JM_TRANSITION_SHA256
        ):
            blockers.append("05j-m transition HDF SHA-256 mismatch")
        state_schema = json.loads((root_m / "state_schema.json").read_text())
        base_schema = self.bundle.layout_bundle.state_schema
        if json.dumps(state_schema, sort_keys=True) != json.dumps(
            base_schema, sort_keys=True
        ):
            blockers.append("05j-m state schema differs from the model dataset")
        if blockers:
            raise RuntimeError(f"05j-n provenance blockers: {blockers}")

        manifest = json.loads((root_m / "dataset_manifest.json").read_text())
        teacher = json.loads((root_m / "manifest.json").read_text())
        layout_bundle = FlowmapBundle(
            root=root_m,
            transition_path=transition,
            manifest=manifest,
            state_schema=state_schema,
            teacher_manifest=teacher,
            validation_report=report_m,
            artifact_validation={"valid": True, **contract_m},
        )
        shard = CompositeShard(
            shard_id="05jm_training",
            root=root_m,
            transition_path=transition,
            transition_count=2304,
            transition_sha256=EXPECTED_05JM_TRANSITION_SHA256,
            dataset_manifest=manifest,
            validation_report=report_m,
            offset=0,
        )
        independent = _IndependentBundle(
            root_m / "dataset_manifest.json",
            manifest,
            shard,
            layout_bundle,
            hashlib.sha256((EXPECTED_05JM_FINAL_SHA256 + EXPECTED_05JM_TRANSITION_SHA256).encode()).hexdigest(),
        )
        self.training_store = _TrainingSupportTransitionStore(independent)
        self.training_pair_rows = pd.read_parquet(root_m / "training_pairs.parquet").to_dict("records")
        self.artifact_05jl_contract = contract_l
        self.artifact_05jm_contract = contract_m
        self.artifact_05jm_report = report_m
        payload = {
            "schema_version": "05j-n-regenerative-decoder-refit-config-v1",
            "regenerative_decoder_refit": asdict(self.refit),
            "artifact_05jl": contract_l,
            "artifact_05jm": contract_m,
            "training_store": self.training_store.report(),
            "base_h2_frozen": True,
            "decoder_architecture": "registered_direct_tree_unchanged",
            "architecture_search_performed": False,
            "05ji_role": "development_after_checkpoint_freeze",
            "fresh_test_inputs_extracted": False,
            "fresh_test_outcomes_generated": False,
            "fresh_test_plan_member_parsed": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "regenerative_decoder_refit_config.json", payload)
        return {**base, **payload}

    def _training_logical_indices(self) -> np.ndarray:
        if self.training_store is None:
            raise RuntimeError("prepare_regenerative_decoder_refit() must run first")
        lookup = {
            int(value): index
            for index, value in enumerate(self.training_store.metadata["transition_id"].tolist())
        }
        indices = []
        for row in self.training_pair_rows:
            indices.extend([lookup[int(row["low_transition_id"])], lookup[int(row["high_transition_id"])]])
        return np.asarray(indices, dtype=np.int64)

    def _fit_transform(self, fit_role: Mapping[str, Any]) -> Dict[str, Any]:
        denominator = np.arcsinh(self.spatial.input_asinh_reference_z)
        surfaces = {}
        for name, rank in (("h2_raw", self.spatial.pca_rank_h2), ("causal_raw", self.spatial.pca_rank_causal)):
            values = np.asarray(fit_role[name], dtype=np.float64)
            mean = values.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(values.std(axis=(0, 1), keepdims=True), self.spatial.feature_epsilon)
            transformed = np.arcsinh((values - mean) / scale) / denominator
            channel_mean, components = deterministic_pca_components(transformed, rank)
            surfaces[name] = {"mean": mean, "scale": scale, "channel_mean": channel_mean, "components": components}
        transform: Dict[str, Any] = {"surfaces": surfaces}
        raw = self._raw_topology_design(fit_role, transform)
        transform["raw_mean"] = raw.mean(axis=(0, 1), keepdims=True)
        transform["raw_scale"] = np.maximum(raw.std(axis=(0, 1), keepdims=True), self.spatial.feature_epsilon)
        return transform

    def prepare_refit_roles(self) -> Dict[str, Any]:
        if self.training_store is None or not self.expanded_topology_transform:
            raise RuntimeError("05j-n requires reconstructed 05j-h/05j-i roles")
        require_torch()
        original_store = self.store
        old_fit = self.topology_roles["fit"]
        development = self.topology_roles["calibration"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2, _ = self._load_h2_checkpoint(device)
        h2.eval()
        indices = self._training_logical_indices()
        self.store = self.training_store
        try:
            new_train = self._extract_recheck_role(h2, indices)
        finally:
            self.store = original_store
            del h2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        old_count = len(old_fit["base"]) // 2
        new_count = len(new_train["base"]) // 2
        if old_count != self.refit.expected_old_fit_pair_count or new_count != self.refit.expected_new_train_pair_count:
            raise RuntimeError(f"05j-n pair cardinality mismatch: old={old_count}, new={new_count}")
        old_rows = self.regenerative_support["fit_pairs"]
        old_ids = [
            str(
                row.get(
                    "branch_pair_id",
                    "05jh:"
                    + str(row.get("left_episode_id", f"left-{index:04d}"))
                    + ":"
                    + str(row.get("right_episode_id", f"right-{index:04d}")),
                )
            )
            for index, row in enumerate(old_rows)
        ]
        new_ids = [str(row["branch_pair_id"]) for row in self.training_pair_rows]
        old_split = deterministic_pair_partition(old_ids, self.refit.old_fit_calibration_pair_count, salt="05j-n-old")
        new_split = deterministic_pair_partition(new_ids, self.refit.new_train_calibration_pair_count, salt="05j-n-new")
        fit_role = _concatenate_roles([
            _subset_role(old_fit, _sample_positions(old_split["fit_pair_positions"])),
            _subset_role(new_train, _sample_positions(new_split["fit_pair_positions"])),
        ])
        calibration_role = _concatenate_roles([
            _subset_role(old_fit, _sample_positions(old_split["calibration_pair_positions"])),
            _subset_role(new_train, _sample_positions(new_split["calibration_pair_positions"])),
        ])
        self.refit_roles = {"fit": fit_role, "calibration": calibration_role, "development": development}
        self.refit_transform = self._fit_transform(fit_role)
        self.refit_designs = {
            role: self._normalize_raw_topology(self._raw_topology_design(values, self.refit_transform), self.refit_transform)
            for role, values in self.refit_roles.items()
        }
        def episode_ids(row: Mapping[str, Any]) -> set[str]:
            if "trajectory_ids" in row:
                values = row["trajectory_ids"]
                if isinstance(values, str):
                    try:
                        values = json.loads(values)
                    except json.JSONDecodeError:
                        values = [values]
                return {str(value) for value in values}
            return {
                str(row.get("left_episode_id")),
                str(row.get("right_episode_id")),
            }

        fit_episodes = {
            value
            for positions, rows in (
                (old_split["fit_pair_positions"], old_rows),
                (new_split["fit_pair_positions"], self.training_pair_rows),
            )
            for position in positions
            for value in episode_ids(rows[position])
        }
        calibration_episodes = {
            value
            for positions, rows in (
                (old_split["calibration_pair_positions"], old_rows),
                (new_split["calibration_pair_positions"], self.training_pair_rows),
            )
            for position in positions
            for value in episode_ids(rows[position])
        }
        episode_overlap = sorted(fit_episodes & calibration_episodes)
        boundary_errors = []
        for pair in range(new_count):
            left, right = indices[2 * pair : 2 * pair + 2]
            state = self.training_store.read_state([left, right], "t")
            boundary_errors.append(float(np.max(np.abs(state[0] - state[1]))))
        report = {
            "schema_version": "05j-n-refit-role-contract-v1",
            "valid": bool(
                max(boundary_errors, default=0.0) <= 1e-12
                and not episode_overlap
                and all(
                    np.all(np.isfinite(value))
                    for value in self.refit_designs.values()
                )
            ),
            "old_fit_pair_count": old_count,
            "new_train_pair_count": new_count,
            "combined_internal_fit_pair_count": len(fit_role["base"]) // 2,
            "combined_internal_calibration_pair_count": len(calibration_role["base"]) // 2,
            "development_pair_count": len(development["base"]) // 2,
            "old_split": old_split,
            "new_split": new_split,
            "maximum_new_pair_boundary_state_error": max(boundary_errors, default=0.0),
            "internal_fit_calibration_episode_overlap": episode_overlap,
            "feature_width": int(self.refit_designs["fit"].shape[-1]),
            "representation_fit_roles": ["combined_internal_fit"],
            "development_used_to_fit_representation": False,
            "fresh_test_loaded": False,
        }
        np.savez_compressed(
            self.output_dir / "refit_feature_transform.npz",
            h2_mean=self.refit_transform["surfaces"]["h2_raw"]["mean"],
            h2_scale=self.refit_transform["surfaces"]["h2_raw"]["scale"],
            h2_channel_mean=self.refit_transform["surfaces"]["h2_raw"]["channel_mean"],
            h2_components=self.refit_transform["surfaces"]["h2_raw"]["components"],
            causal_mean=self.refit_transform["surfaces"]["causal_raw"]["mean"],
            causal_scale=self.refit_transform["surfaces"]["causal_raw"]["scale"],
            causal_channel_mean=self.refit_transform["surfaces"]["causal_raw"]["channel_mean"],
            causal_components=self.refit_transform["surfaces"]["causal_raw"]["components"],
            raw_mean=self.refit_transform["raw_mean"],
            raw_scale=self.refit_transform["raw_scale"],
        )
        _write_json(self.output_dir / "refit_role_contract.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-n refit role contract failed")
        return report

    def _refit_metrics(self, role: str, residual: np.ndarray) -> Dict[str, Any]:
        state = self.refit_roles[role]
        return self._pair_set_metrics(
            (np.asarray(state["base"]) + np.asarray(residual)).reshape(-1, 2, self.layout.segment_count),
            np.asarray(state["target"]).reshape(-1, 2, self.layout.segment_count),
        )

    def _baseline_metrics(self, role: str) -> Dict[str, Dict[str, Any]]:
        state = self.refit_roles[role]
        target = np.asarray(state["target"]).reshape(-1, 2, self.layout.segment_count)
        return {
            "h2": self._pair_set_metrics(np.asarray(state["base"]).reshape(-1, 2, self.layout.segment_count), target),
            "persistence": self._pair_set_metrics(np.asarray(state["voltage_t"]).reshape(-1, 2, self.layout.segment_count), target),
        }

    def run_decoder_refit(self) -> Dict[str, Any]:
        require_torch()
        if not self.refit_designs:
            raise RuntimeError("prepare_refit_roles() must run first")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = {}
        for role, state in self.refit_roles.items():
            tensors[role] = {
                "features": torch.as_tensor(self.refit_designs[role], dtype=torch.float32, device=device),
                "target": torch.as_tensor(np.asarray(state["target"]) - np.asarray(state["base"]), dtype=torch.float32, device=device),
            }
        baselines = {role: self._baseline_metrics(role) for role in self.refit_roles}
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        runs = []
        overall = Progress("05j-n decoder refit", len(self.refit.seeds))
        for run_index, seed in enumerate(self.refit.seeds, start=1):
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
            model = TrainableTopologyResidualHead(
                self.refit_designs["fit"].shape[-1],
                self.layout.segment_count,
                self.topology.hidden_width,
                self.topology.segment_embedding_dim,
                self.topology.target_residual_limit_mv,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.refit.learning_rate, weight_decay=self.refit.weight_decay)
            best_score, best_epoch, stale, best_state = math.inf, 0, 0, None
            history = []
            progress = Progress(f"05j-n direct_tree seed{seed}", self.refit.epochs)
            for epoch in range(1, self.refit.epochs + 1):
                model.train(); optimizer.zero_grad(set_to_none=True)
                prediction = model(tensors["fit"]["features"])
                loss = self._loss(prediction, tensors["fit"]["target"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.refit.gradient_clip_norm)
                optimizer.step()
                if epoch == 1 or epoch % self.refit.evaluation_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        residual = model(tensors["calibration"]["features"]).cpu().numpy()
                    metrics = self._refit_metrics("calibration", residual)
                    score = pair_gate_selection_score(metrics, max_error_weight=self.topology.selection_max_error_weight, branch_log_weight=self.topology.selection_branch_log_weight)
                    history.append({
                        "seed": seed, "epoch": epoch, "training_loss": float(loss.detach().cpu()),
                        "calibration_score": score,
                        "calibration_rmse_mv": metrics["aggregate_voltage_rmse_mv"],
                        "calibration_max_error_mv": metrics["maximum_segment_error_mv"],
                        "calibration_retention": metrics["median_branching_retention"],
                    })
                    if score < best_score - 1e-9:
                        best_score, best_epoch, stale = score, epoch, 0
                        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    else:
                        stale += self.refit.evaluation_interval
                    progress.update(epoch, f"loss={float(loss):.3g} cal={score:.3g}")
                    if stale >= self.refit.patience:
                        break
            if best_state is None:
                raise RuntimeError("05j-n failed to select a checkpoint")
            model.load_state_dict(best_state); model.eval()
            role_metrics = {}
            with torch.no_grad():
                for role in self.refit_roles:
                    role_metrics[role] = self._refit_metrics(
                        role, model(tensors[role]["features"]).cpu().numpy()
                    )
            internal_h2 = baselines["calibration"]["h2"]
            development_h2 = baselines["development"]["h2"]
            development_best = min(
                development_h2["aggregate_voltage_rmse_mv"],
                baselines["development"]["persistence"]["aggregate_voltage_rmse_mv"],
            )
            internal_gain = 1.0 - role_metrics["calibration"]["aggregate_voltage_rmse_mv"] / max(internal_h2["aggregate_voltage_rmse_mv"], 1e-12)
            development_gain = 1.0 - role_metrics["development"]["aggregate_voltage_rmse_mv"] / max(development_best, 1e-12)
            retention = role_metrics["development"]["median_branching_retention"]
            gate_values = (
                internal_gain,
                development_gain,
                retention,
                role_metrics["development"]["maximum_segment_error_mv"],
                development_h2["maximum_segment_error_mv"],
            )
            metrics_finite = all(math.isfinite(float(value)) for value in gate_values)
            passed = bool(
                metrics_finite
                and internal_gain >= self.refit.minimum_internal_improvement_vs_h2_fraction
                and development_gain >= self.refit.minimum_development_improvement_vs_best_baseline_fraction
                and self.refit.minimum_development_branching_retention <= retention <= self.refit.maximum_development_branching_retention
                and role_metrics["development"]["maximum_segment_error_mv"]
                <= self.refit.maximum_development_error_ratio_vs_h2 * development_h2["maximum_segment_error_mv"]
            )
            checkpoint = checkpoint_dir / f"direct_tree_refit_seed{seed}.pt"
            torch.save(
                {
                    "state_dict": best_state,
                    "family": "direct_tree_refit",
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "feature_width": int(self.refit_designs["fit"].shape[-1]),
                    "artifact_05jm_transition_sha256": EXPECTED_05JM_TRANSITION_SHA256,
                    "fresh_test_protocol_plan_sha256": self.artifact_05jm_report["fresh_test"]["protocol_plan_sha256"],
                },
                checkpoint,
            )
            runs.append({
                "family": "direct_tree_refit", "seed": seed, "best_epoch": best_epoch,
                "best_internal_calibration_score": best_score,
                "roles": role_metrics,
                "internal_improvement_vs_h2_fraction": internal_gain,
                "development_improvement_vs_best_baseline_fraction": development_gain,
                "gate_metrics_finite": metrics_finite,
                "run_passed": passed,
                "checkpoint": checkpoint.relative_to(self.output_dir).as_posix(),
            })
            write_parquet(self.output_dir / f"history_direct_tree_refit_seed{seed}.parquet", history)
            overall.update(run_index, f"seed={seed} pass={passed}")
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        passing = sum(row["run_passed"] for row in runs)
        report_valid = all(row["gate_metrics_finite"] for row in runs)
        report = {
            "schema_version": "05j-n-regenerative-decoder-refit-v1",
            "valid": report_valid,
            "device": str(device),
            "runs": runs,
            "baselines": baselines,
            "passing_seed_count": passing,
            "minimum_passing_seed_count": self.refit.minimum_passing_seeds,
            "robust_gate_passed": passing >= self.refit.minimum_passing_seeds,
            "checkpoint_selection_role": "internal_calibration_only",
            "development_used_for_checkpoint_selection": False,
            "development_inference_after_checkpoint_freeze": True,
            "fresh_test_inputs_extracted": False,
            "fresh_test_outcomes_generated": False,
            "architecture_search_performed": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "decoder_refit_report.json", report)
        write_parquet(self.output_dir / "decoder_refit_run_summary.parquet", [{
            "seed": row["seed"], "best_epoch": row["best_epoch"], "run_passed": row["run_passed"],
            "internal_improvement_vs_h2_fraction": row["internal_improvement_vs_h2_fraction"],
            "development_improvement_vs_best_baseline_fraction": row["development_improvement_vs_best_baseline_fraction"],
            **{f"{role}_rmse_mv": row["roles"][role]["aggregate_voltage_rmse_mv"] for role in ("fit", "calibration", "development")},
        } for row in runs])
        if not report_valid:
            raise RuntimeError("05j-n produced non-finite gate metrics")
        return report

    def finalize_decoder_refit(self, role_report: Mapping[str, Any], refit_report: Mapping[str, Any]) -> Dict[str, Any]:
        passed = bool(role_report["valid"] and refit_report["robust_gate_passed"])
        report = {
            "schema_version": "05j-n-final-report-v1",
            "valid": bool(role_report["valid"] and refit_report["valid"]),
            "decision": "REGISTERED_DECODER_REFIT_ON_EXPANDED_TRAIN_SUPPORT",
            "diagnosis": (
                "REFIT_PASSES_DEVELOPMENT_FRESH_TEST_GENERATION_AUTHORIZED"
                if passed
                else "REFIT_FAILS_DEVELOPMENT_FRESH_TEST_REMAINS_SEALED"
            ),
            "code_revision": self.code_revision,
            "artifact_05jl": self.artifact_05jl_contract,
            "artifact_05jm": self.artifact_05jm_contract,
            "role_contract": dict(role_report),
            "decoder_refit": dict(refit_report),
            "fresh_test_generation_authorized": passed,
            "candidate_model_authorized": False,
            "micro_rollout_authorized": False,
            "full_training_authorized": False,
            "methodology": {
                "base_h2_frozen": True,
                "decoder_architecture_unchanged": True,
                "representation_fit_on_internal_fit_only": True,
                "checkpoint_selected_on_internal_calibration_only": True,
                "05ji_development_used_after_checkpoint_freeze": True,
                "fresh_test_inputs_extracted": False,
                "fresh_test_outcomes_generated": False,
                "rollout_performed": False,
            },
            "next_step": (
                "05j_o_generate_and_evaluate_preregistered_fresh_test"
                if passed
                else "05j_n_b_decoder_failure_reassessment_without_unsealing_test"
            ),
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
        _write_json(self.output_dir / "artifact_index.json", {"schema_version": "05j-n-artifact-index-v1", "artifact_count": len(records), "artifacts": records})
        return report


__all__ = [
    "EXPECTED_05JL_ARCHIVE_SHA256",
    "EXPECTED_05JL_INDEX_SHA256",
    "EXPECTED_05JL_FINAL_SHA256",
    "EXPECTED_05JM_ARCHIVE_SHA256",
    "EXPECTED_05JM_INDEX_SHA256",
    "EXPECTED_05JM_FINAL_SHA256",
    "EXPECTED_05JM_TRANSITION_SHA256",
    "HinesRegenerativeDecoderRefitConfig",
    "HinesRegenerativeDecoderRefit",
    "deterministic_pair_partition",
]
