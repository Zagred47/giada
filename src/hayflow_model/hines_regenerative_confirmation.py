"""05j-j: frozen regenerative-state confirmation on the independent 05j-i shard."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import (
    CompositeShard,
    CompositeTransitionStore,
)
from ..hayflow_data.flowmap_dataset import FlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_regenerative_state_decomposition import (
    STATE_STATISTICS,
    aggregate_state_groups,
    semantic_group_ids,
)
from .hines_regenerative_support_expansion import (
    HinesRegenerativeSupportExpansion,
)
from .hines_trainable_topology_canary import (
    TrainableTopologyResidualHead,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05JH_ARCHIVE_SHA256 = (
    "b51137b6f73656f9d637a9a1948dc40ac2795e1c8c480148caf5d1831e0d228c"
)
EXPECTED_05JH_INDEX_SHA256 = (
    "6336704675d1c69a71c0fa55db9d61d85bd6679c9f27a40963489ee3d9b93090"
)
EXPECTED_05JH_FINAL_SHA256 = (
    "8ea2abf9ee0366b8c2341e0602613f5b4409fa832481e9e8bd8371430a9c8e5a"
)
EXPECTED_05JI_ARCHIVE_SHA256 = (
    "18235e9f643064df90fd7e305450635b5c255253a4958ce5b639433a558a8275"
)
EXPECTED_05JI_INDEX_SHA256 = (
    "70a0a72d51c52fd1fd0eb5d1c72c1332d8114cb241c8a9ec3bcaf82ac9839bce"
)
EXPECTED_05JI_FINAL_SHA256 = (
    "edb8ffcb60e12ab593f8761d2bf17fe8fefcf496479b090a8d445e8e47cbeeaf"
)
EXPECTED_05JI_TRANSITION_SHA256 = (
    "d1d372ffb7f5d0e513ed6bbfac5536ac8e35c5a726523f9c6aac8ada395aa6a6"
)


@dataclass(frozen=True)
class HinesRegenerativeConfirmationConfig:
    registered_pair_count: int = 24
    minimum_near_regenerative_pairs: int = 18
    minimum_pair_win_fraction: float = 2.0 / 3.0
    material_rmse_improvement_fraction: float = 0.20
    material_max_error_improvement_fraction: float = 0.10
    oracle_specificity_improvement_fraction: float = 0.15
    registered_05jh_specificity_fraction: float = 0.08345212759769333
    reproduction_atol: float = 2e-5

    def validate(self) -> None:
        if self.registered_pair_count != 24:
            raise ValueError("05j-j uses all 24 preregistered acquisition pairs")
        if not 0 < self.minimum_near_regenerative_pairs <= self.registered_pair_count:
            raise ValueError("05j-j near-regenerative minimum is invalid")
        fractions = (
            self.minimum_pair_win_fraction,
            self.material_rmse_improvement_fraction,
            self.material_max_error_improvement_fraction,
            self.oracle_specificity_improvement_fraction,
        )
        if any(not 0 < value <= 1 for value in fractions):
            raise ValueError("05j-j gate fractions must lie in (0, 1]")
        if self.reproduction_atol <= 0:
            raise ValueError("05j-j reproduction tolerance must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRegenerativeConfirmationConfig":
        result = cls(**dict(values))
        result.validate()
        return result


@dataclass(frozen=True)
class _IndependentBundle:
    manifest_path: Path
    manifest: Mapping[str, Any]
    shard: CompositeShard
    layout_bundle: FlowmapBundle
    fingerprint: str

    @property
    def shards(self) -> Tuple[CompositeShard, ...]:
        return (self.shard,)

    @property
    def transition_count(self) -> int:
        return self.shard.transition_count


class _IndependentTransitionStore(CompositeTransitionStore):
    """Reuse the causal v1.1 reader without the base/top-up cardinality rules."""

    def _validate_contract(self) -> None:
        splits = sorted({str(row["split"]) for row in self.episode_rows})
        trajectories = [str(row["trajectory_id"]) for row in self.episode_rows]
        blockers = []
        if self.count != 576:
            blockers.append("independent shard transition count is not 576")
        if len(self.episode_rows) != 48:
            blockers.append("independent shard episode count is not 48")
        if splits != ["validation"]:
            blockers.append("independent shard is not validation-only")
        if len(set(trajectories)) != len(trajectories):
            blockers.append("independent shard repeats a trajectory id")
        self.contract_checks = {
            "transition_count_exact": self.count == 576,
            "episode_count_exact": len(self.episode_rows) == 48,
            "validation_only": splits == ["validation"],
            "trajectory_ids_unique": len(set(trajectories)) == len(trajectories),
            "blockers": blockers,
        }
        if blockers:
            raise RuntimeError(f"independent confirmation contract failed: {blockers}")

    def _shard_for(self, logical_index: int) -> CompositeShard:
        if not 0 <= int(logical_index) < self.count:
            raise IndexError(logical_index)
        return self.bundle.shard

    def report(self) -> Dict[str, Any]:
        return {
            "valid": not self.contract_checks["blockers"],
            "dataset_kind": "independent_validation_confirmation_shard",
            "fingerprint": self.bundle.fingerprint,
            "episode_count": len(self.episode_rows),
            "transition_count": self.count,
            "contract_checks": self.contract_checks,
            "state_loading": "lazy_single_shard",
        }


def _safe_extract(source: Path, destination: Path, stamp: str) -> Path:
    marker = destination / ".source_sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == stamp:
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe artifact member: {member.filename}")
        archive.extractall(destination)
    marker.write_text(stamp + "\n", encoding="utf-8")
    return destination


def _verified_artifact_root(
    source: Path,
    cache_dir: Path,
    *,
    marker_name: str,
    archive_sha256: str,
    index_sha256: str,
    final_sha256: str,
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    source = Path(source).expanduser().resolve()
    observed_archive = None
    if source.is_file():
        observed_archive = sha256_file(source)
        if observed_archive != archive_sha256:
            raise RuntimeError(f"{marker_name} archive SHA-256 mismatch")
        search_root = _safe_extract(source, cache_dir, observed_archive)
        source_kind = "original_zip"
    elif source.is_dir():
        search_root = source
        source_kind = "kaggle_extracted_directory"
    else:
        raise RuntimeError(f"artifact source does not exist: {source}")
    indices = [
        path for path in search_root.rglob("artifact_index.json")
        if (path.parent / marker_name).is_file()
    ]
    if len(indices) != 1:
        raise RuntimeError(
            f"expected one {marker_name} artifact below {search_root}; found {len(indices)}"
        )
    index_path = indices[0]
    index_bytes = index_path.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != index_sha256:
        raise RuntimeError(f"{marker_name} artifact index SHA-256 mismatch")
    index = json.loads(index_bytes)
    root = index_path.parent
    for row in index["artifacts"]:
        path = root / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise RuntimeError(f"{marker_name} indexed member mismatch: {row['path']}")
    final_bytes = (root / "final_report.json").read_bytes()
    if hashlib.sha256(final_bytes).hexdigest() != final_sha256:
        raise RuntimeError(f"{marker_name} final report SHA-256 mismatch")
    contract = {
        "source_kind": source_kind,
        "source_path": str(source),
        "archive_sha256": observed_archive,
        "artifact_index_sha256": index_sha256,
        "final_report_sha256": final_sha256,
        "verified_member_count": len(index["artifacts"]),
        "all_indexed_members_verified": (
            len(index["artifacts"])
            == int(index.get("artifact_count", len(index["artifacts"])))
        ),
    }
    return root, json.loads(final_bytes), contract


def confirmation_decision(
    *,
    causal_rmse_gain: float,
    causal_max_gain: float,
    oracle_rmse_gain: float,
    oracle_max_gain: float,
    specificity_gain: float,
    pair_win_fraction: float,
    config: HinesRegenerativeConfirmationConfig,
) -> Tuple[str, str, bool, bool]:
    causal = bool(
        causal_rmse_gain >= config.material_rmse_improvement_fraction
        and causal_max_gain >= config.material_max_error_improvement_fraction
    )
    oracle = bool(
        oracle_rmse_gain >= config.material_rmse_improvement_fraction
        and oracle_max_gain >= config.material_max_error_improvement_fraction
        and specificity_gain >= config.oracle_specificity_improvement_fraction
        and pair_win_fraction >= config.minimum_pair_win_fraction
    )
    if causal:
        return (
            "INDEPENDENT_SUPPORT_CONFIRMS_CAUSAL_BOUNDARY_STATE_SIGNAL",
            "05j_k_explicit_regenerative_state_input_canary",
            True,
            oracle,
        )
    if oracle:
        return (
            "INDEPENDENT_SUPPORT_CONFIRMS_REGENERATIVE_STATE_TRANSITION_SIGNAL",
            "05j_k_joint_regenerative_state_transition_canary",
            False,
            True,
        )
    if oracle_rmse_gain >= config.material_rmse_improvement_fraction:
        return (
            "INDEPENDENT_REGENERATIVE_SIGNAL_IS_NOT_SPATIALLY_SPECIFIC",
            "05j_k_regime_conditioned_transition_objective",
            False,
            False,
        )
    return (
        "INDEPENDENT_SUPPORT_REJECTS_REGENERATIVE_STATE_EXPLANATION",
        "05j_k_voltage_decoder_objective_reassessment",
        False,
        False,
    )


class HinesRegenerativeStateConfirmation(HinesRegenerativeSupportExpansion):
    """Apply the unchanged 05j-h diagnostic to outcome-blind 05j-i support."""

    def __init__(
        self,
        *args: Any,
        confirmation_config: HinesRegenerativeConfirmationConfig,
        artifact_05jh_source: Path,
        artifact_05ji_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        confirmation_config.validate()
        self.confirmation = confirmation_config
        self.artifact_05jh_source = Path(artifact_05jh_source).resolve()
        self.artifact_05ji_source = Path(artifact_05ji_source).resolve()
        self.artifact_05jh_report: Dict[str, Any] = {}
        self.artifact_05ji_report: Dict[str, Any] = {}
        self.artifact_05jh_contract: Dict[str, Any] = {}
        self.artifact_05ji_contract: Dict[str, Any] = {}
        self.confirmation_store: _IndependentTransitionStore | None = None
        self.confirmation_indices = np.empty(0, dtype=np.int64)

    def prepare_independent_confirmation(self) -> Dict[str, Any]:
        base = self.prepare_regenerative_support_expansion()
        cache = self.output_dir.parent / ".05j_j_artifact_cache"
        _, report_h, contract_h = _verified_artifact_root(
            self.artifact_05jh_source,
            cache / "05jh",
            marker_name="regenerative_support_expansion_config.json",
            archive_sha256=EXPECTED_05JH_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JH_INDEX_SHA256,
            final_sha256=EXPECTED_05JH_FINAL_SHA256,
        )
        root_i, report_i, contract_i = _verified_artifact_root(
            self.artifact_05ji_source,
            cache / "05ji",
            marker_name="confirmation_plan.json",
            archive_sha256=EXPECTED_05JI_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JI_INDEX_SHA256,
            final_sha256=EXPECTED_05JI_FINAL_SHA256,
        )
        blockers: List[str] = []
        if report_h.get("diagnosis") != "EXISTING_DATASET_LACKS_INDEPENDENT_REGENERATIVE_CONFIRMATION_SUPPORT":
            blockers.append("05j-h diagnosis does not authorize independent support")
        if not report_i.get("valid") or not report_i.get("scientific_support_sufficient"):
            blockers.append("05j-i support is not scientifically sufficient")
        if report_i.get("diagnosis") != "NEAR_REGENERATIVE_CONFIRMATION_SUPPORT_ACQUIRED":
            blockers.append("05j-i diagnosis is unexpected")
        if int(report_i.get("registered_pair_count", -1)) != self.confirmation.registered_pair_count:
            blockers.append("05j-i registered pair count changed")
        near = int(report_i.get("realized_stratum_counts", {}).get("near_regenerative", 0))
        if near < self.confirmation.minimum_near_regenerative_pairs:
            blockers.append("05j-i near-regenerative support is insufficient")
        embedded = report_i.get("artifact_05jh", {})
        if embedded.get("artifact_index_sha256") != EXPECTED_05JH_INDEX_SHA256:
            blockers.append("05j-i embeds a different 05j-h provenance")
        transition = root_i / "transition_dataset.h5"
        if sha256_file(transition) != EXPECTED_05JI_TRANSITION_SHA256:
            blockers.append("05j-i transition HDF SHA-256 mismatch")
        state_schema = json.loads((root_i / "state_schema.json").read_text())
        base_schema = self.bundle.layout_bundle.state_schema
        if json.dumps(state_schema, sort_keys=True) != json.dumps(base_schema, sort_keys=True):
            blockers.append("05j-i state schema differs from the model dataset")
        if blockers:
            raise RuntimeError(f"05j-j provenance blockers: {blockers}")

        manifest = json.loads((root_i / "dataset_manifest.json").read_text())
        validation = json.loads((root_i / "validation_report.json").read_text())
        teacher = json.loads((root_i / "manifest.json").read_text())
        layout_bundle = FlowmapBundle(
            root=root_i,
            transition_path=transition,
            manifest=manifest,
            state_schema=state_schema,
            teacher_manifest=teacher,
            validation_report=validation,
            artifact_validation={"valid": True, **contract_i},
        )
        shard = CompositeShard(
            shard_id="05ji_confirmation",
            root=root_i,
            transition_path=transition,
            transition_count=576,
            transition_sha256=EXPECTED_05JI_TRANSITION_SHA256,
            dataset_manifest=manifest,
            validation_report=validation,
            offset=0,
        )
        fingerprint = hashlib.sha256(
            (EXPECTED_05JI_FINAL_SHA256 + EXPECTED_05JI_TRANSITION_SHA256).encode()
        ).hexdigest()
        independent = _IndependentBundle(
            root_i / "dataset_manifest.json", manifest, shard, layout_bundle, fingerprint
        )
        self.confirmation_store = _IndependentTransitionStore(independent)
        self.artifact_05jh_report, self.artifact_05jh_contract = report_h, contract_h
        self.artifact_05ji_report, self.artifact_05ji_contract = report_i, contract_i
        payload = {
            "schema_version": "05j-j-independent-confirmation-config-v1",
            "independent_confirmation": asdict(self.confirmation),
            "artifact_05jh": contract_h,
            "artifact_05ji": contract_i,
            "primary_support": "all_24_preregistered_pairs",
            "near_regenerative_subset_is_descriptive_only": True,
            "fit_support": "unchanged_05jh_train_fit_pairs",
            "feature_and_probe_normalizers_fit_on": ["unchanged_05jh_train_fit_pairs"],
            "confirmation_used_for_selection": False,
            "future_state_delta_is_diagnostic_oracle_only": True,
            "candidate_training_performed": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "independent_confirmation_config.json", payload)
        return {**base, **payload, "independent_store": self.confirmation_store.report()}

    def validate_registered_05jh_reproduction(
        self,
        support_report: Mapping[str, Any],
        probe_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        observed = float(
            probe_report["aligned_oracle_vs_spatial_shift_rmse_improvement_fraction"][
                "calibration"
            ]
        )
        expected = self.confirmation.registered_05jh_specificity_fraction
        errors = {
            "specificity_absolute_error": abs(observed - expected),
            "selected_pair_count_error": abs(
                int(support_report["selected_pair_count"])
                - int(self.artifact_05jh_report["support"]["selected_pair_count"])
            ),
        }
        report = {
            "schema_version": "05j-j-05jh-reproduction-v1",
            "valid": bool(
                errors["specificity_absolute_error"] <= self.confirmation.reproduction_atol
                and errors["selected_pair_count_error"] == 0
            ),
            "registered_specificity_fraction": expected,
            "reproduced_specificity_fraction": observed,
            "errors": errors,
            "new_support_used": False,
            "retraining_performed": False,
        }
        _write_json(self.output_dir / "registered_05jh_reproduction.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-j failed to reproduce registered 05j-h diagnostic")
        return report

    def _confirmation_logical_indices(self) -> np.ndarray:
        if self.confirmation_store is None:
            raise RuntimeError("prepare_independent_confirmation() must run first")
        transition_lookup = {
            int(value): index
            for index, value in enumerate(
                self.confirmation_store.metadata["transition_id"].tolist()
            )
        }
        result = []
        for row in self.artifact_05ji_report["pair_rows"]:
            result.extend([
                transition_lookup[int(row["low_transition_id"])],
                transition_lookup[int(row["high_transition_id"])],
            ])
        return np.asarray(result, dtype=np.int64)

    def prepare_external_confirmation_roles(self) -> Dict[str, Any]:
        if self.confirmation_store is None or not self.expanded_topology_transform:
            raise RuntimeError("05j-j preparation and 05j-h expanded roles are required")
        require_torch()
        original_store = self.store
        original_roles = self.topology_roles
        original_designs = self.topology_designs
        original_predictions = self.frozen_predictions[
            self.reassessment.audited_family
        ]
        indices = self._confirmation_logical_indices()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2, _ = self._load_h2_checkpoint(device)
        h2.eval()
        self.store = self.confirmation_store
        try:
            external_role = self._extract_recheck_role(h2, indices)
        finally:
            self.store = original_store
            del h2
        external_design = self._normalize_raw_topology(
            self._raw_topology_design(external_role, self.expanded_topology_transform),
            self.expanded_topology_transform,
        )
        registered = [
            row
            for row in self.artifact_05jd_report["trainable_topology_canary"]["runs"]
            if row["family"] == self.reassessment.audited_family
        ]
        external_predictions: Dict[int, np.ndarray] = {}
        progress = Progress("05j-j frozen direct-tree inference", len(registered))
        for position, row in enumerate(registered, start=1):
            seed = int(row["seed"])
            checkpoint = torch.load(
                io.BytesIO(self._read_05jd_checkpoint_bytes(str(row["checkpoint"]))),
                map_location=device,
                weights_only=False,
            )
            model = TrainableTopologyResidualHead(
                external_design.shape[-1],
                self.layout.segment_count,
                self.topology.hidden_width,
                self.topology.segment_embedding_dim,
                self.topology.target_residual_limit_mv,
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                external_predictions[seed] = model(
                    torch.as_tensor(external_design, device=device)
                ).cpu().numpy()
            del model
            progress.update(position, f"seed={seed}")
        self.confirmation_indices = indices
        self.topology_roles = {
            "fit": original_roles["fit"],
            "calibration": external_role,
            "development": original_roles["development"],
        }
        self.topology_designs = {
            "fit": original_designs["fit"],
            "calibration": external_design,
            "development": original_designs["development"],
        }
        self.frozen_predictions = {
            self.reassessment.audited_family: {
                seed: {
                    "fit": original_predictions[seed]["fit"],
                    "calibration": external_predictions[seed],
                    "development": original_predictions[seed]["development"],
                }
                for seed in self.reassessment.seeds
            }
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        boundary_errors = []
        for pair in range(len(indices) // 2):
            left, right = indices[2 * pair: 2 * pair + 2]
            states = self.confirmation_store.read_state([left, right], "t")
            boundary_errors.append(float(np.max(np.abs(states[0] - states[1]))))
        report = {
            "schema_version": "05j-j-external-roles-v1",
            "valid": bool(max(boundary_errors, default=0.0) <= 1e-12),
            "pair_count": len(indices) // 2,
            "sample_count": len(indices),
            "maximum_same_boundary_state_error": max(boundary_errors, default=0.0),
            "registered_05jd_transform_reused": True,
            "registered_direct_tree_checkpoints_frozen": True,
            "confirmation_used_to_fit_features": False,
            "confirmation_used_for_checkpoint_selection": False,
            "retraining_performed": False,
            "heldout_inputs_extracted": False,
        }
        _write_json(self.output_dir / "external_confirmation_roles.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-j branch boundaries disagree")
        return report

    def prepare_external_state_surfaces(self) -> Dict[str, Any]:
        if self.confirmation_store is None or not len(self.confirmation_indices):
            raise RuntimeError("prepare_external_confirmation_roles() must run first")
        records = list(self._normalization_records())
        group_ids = semantic_group_ids(records, self.decomposition.state_groups)
        segment_ids = np.asarray(self.layout.core_segment_ids, dtype=np.int64)
        stores = {
            "fit": self.store,
            "calibration": self.confirmation_store,
            "development": self.store,
        }
        surfaces: Dict[str, Dict[str, np.ndarray]] = {}
        for role, role_state in self.topology_roles.items():
            store = stores[role]
            indices = np.asarray(role_state["indices"], dtype=np.int64)
            old = self.store
            self.store = store
            try:
                raw_t = store.read_state(indices, "t")
                raw_t1 = store.read_state(indices, "t_plus_1")
                semantic_t = self._state_input_view(raw_t, indices, "t")
                semantic_t1 = self._state_input_view(raw_t1, indices, "t_plus_1")
            finally:
                self.store = old
            current = self.normalizer.normalize_state(semantic_t)
            delta, _ = self.normalizer.delta_and_activity(semantic_t, semantic_t1)
            surfaces[role] = {
                "current": aggregate_state_groups(
                    current,
                    segment_ids,
                    group_ids,
                    segment_count=self.layout.segment_count,
                    group_count=len(self.decomposition.state_groups),
                ),
                "delta": aggregate_state_groups(
                    delta,
                    segment_ids,
                    group_ids,
                    segment_count=self.layout.segment_count,
                    group_count=len(self.decomposition.state_groups),
                ),
            }
        self.state_surfaces = surfaces
        report = {
            "schema_version": "05j-j-external-state-surfaces-v1",
            "valid": all(
                np.all(np.isfinite(values[kind]))
                for values in surfaces.values()
                for kind in ("current", "delta")
            ),
            "roles": {
                role: {"sample_count": len(values["current"])}
                for role, values in surfaces.items()
            },
            "voltage_coordinate_count_excluded": int(np.sum(group_ids < 0)),
            "future_voltage_coordinate_excluded": True,
            "confirmation_used_to_fit_state_normalizer": False,
        }
        _write_json(self.output_dir / "external_state_surfaces.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-j state surfaces contain non-finite values")
        return report

    def run_fixed_external_probes(self) -> Dict[str, Any]:
        normalizers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        normalized: Dict[str, Dict[str, np.ndarray]] = {"current": {}, "delta": {}}
        for surface in ("current", "delta"):
            fit = np.asarray(self.state_surfaces["fit"][surface], dtype=np.float64)
            mean = fit.mean(axis=(0, 1), keepdims=True)
            scale = np.maximum(
                fit.std(axis=(0, 1), keepdims=True), self.decomposition.feature_epsilon
            )
            normalizers[surface] = (mean, scale)
            for role in self.state_surfaces:
                normalized[surface][role] = (
                    (np.asarray(self.state_surfaces[role][surface]) - mean) / scale
                ).astype(np.float32)
        np.savez_compressed(
            self.output_dir / "external_probe_normalizers.npz",
            **{
                f"{surface}_{kind}": value
                for surface, (mean, scale) in normalizers.items()
                for kind, value in (("mean", mean), ("scale", scale))
            },
        )
        baseline = {
            role: np.mean(
                np.stack([
                    self.frozen_predictions[self.reassessment.audited_family][seed][role]
                    for seed in self.reassessment.seeds
                ]),
                axis=0,
            )
            for role in self.topology_roles
        }
        baseline_roles = {
            role: self._role_metrics(role, residual)
            for role, residual in baseline.items()
        }
        designs = {
            "intercept_only_control": {
                role: np.zeros((*values["current"].shape[:2], 1), dtype=np.float32)
                for role, values in self.state_surfaces.items()
            },
            "causal_current_all": normalized["current"],
            "oracle_delta_aligned_all": normalized["delta"],
            "oracle_delta_spatial_shift_control": {
                role: np.roll(
                    values,
                    shift=self.decomposition.spatial_shift_segments,
                    axis=1,
                )
                for role, values in normalized["delta"].items()
            },
        }
        runs = []
        progress = Progress("05j-j fixed external probes", len(designs))
        for position, (family, family_designs) in enumerate(designs.items(), start=1):
            row = self._fit_probe(family, family_designs, baseline)
            runs.append(row)
            progress.update(
                position,
                f"{family} external={row['roles']['calibration']['aggregate_voltage_rmse_mv']:.3g}",
            )
        by_family = {row["family"]: row for row in runs}
        intercept = by_family["intercept_only_control"]
        for row in runs:
            row["improvement_vs_intercept_control"] = {
                role: {
                    "rmse_improvement_fraction": self._improvement(
                        intercept["roles"][role]["aggregate_voltage_rmse_mv"],
                        row["roles"][role]["aggregate_voltage_rmse_mv"],
                    ),
                    "maximum_error_improvement_fraction": self._improvement(
                        intercept["roles"][role]["maximum_segment_error_mv"],
                        row["roles"][role]["maximum_segment_error_mv"],
                    ),
                }
                for role in baseline_roles
            }
        aligned = by_family["oracle_delta_aligned_all"]
        shifted = by_family["oracle_delta_spatial_shift_control"]
        specificity = {
            role: self._improvement(
                shifted["roles"][role]["aggregate_voltage_rmse_mv"],
                aligned["roles"][role]["aggregate_voltage_rmse_mv"],
            )
            for role in baseline_roles
        }
        report = {
            "schema_version": "05j-j-fixed-external-probes-v1",
            "valid": True,
            "baseline_roles": baseline_roles,
            "runs": runs,
            "aligned_oracle_vs_spatial_shift_rmse_improvement_fraction": specificity,
            "probe_families_preregistered": list(designs),
            "individual_state_group_search_performed": False,
            "selection_roles": ["unchanged_05jh_fit_grouped_pair_cross_validation"],
            "confirmation_used_for_selection": False,
            "future_state_delta_used_as_diagnostic_oracle_only": True,
            "candidate_training_performed": False,
            "heldout_inputs_extracted": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "external_probe_report.json", report)
        write_parquet(
            self.output_dir / "external_probe_summary.parquet",
            [
                {
                    "family": row["family"],
                    "selected_ridge_lambda": row["selected_ridge_lambda"],
                    "external_rmse_mv": row["roles"]["calibration"]["aggregate_voltage_rmse_mv"],
                    "external_maximum_segment_error_mv": row["roles"]["calibration"]["maximum_segment_error_mv"],
                    "external_rmse_improvement_vs_intercept_fraction": row[
                        "improvement_vs_intercept_control"
                    ]["calibration"]["rmse_improvement_fraction"],
                }
                for row in runs
            ],
        )
        return report

    def finalize_independent_confirmation(
        self,
        reproduction_report: Mapping[str, Any],
        role_report: Mapping[str, Any],
        surface_report: Mapping[str, Any],
        probe_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        causal = self._probe(probe_report, "causal_current_all")
        aligned = self._probe(probe_report, "oracle_delta_aligned_all")
        shifted = self._probe(probe_report, "oracle_delta_spatial_shift_control")
        causal_gain = causal["improvement_vs_intercept_control"]["calibration"]
        oracle_gain = aligned["improvement_vs_intercept_control"]["calibration"]
        specificity = float(
            probe_report["aligned_oracle_vs_spatial_shift_rmse_improvement_fraction"][
                "calibration"
            ]
        )
        pairwise = self._pairwise_win_report(aligned, shifted, "calibration")
        diagnosis, next_step, causal_confirmed, transition_confirmed = (
            confirmation_decision(
                causal_rmse_gain=causal_gain["rmse_improvement_fraction"],
                causal_max_gain=causal_gain["maximum_error_improvement_fraction"],
                oracle_rmse_gain=oracle_gain["rmse_improvement_fraction"],
                oracle_max_gain=oracle_gain["maximum_error_improvement_fraction"],
                specificity_gain=specificity,
                pair_win_fraction=pairwise["win_fraction"],
                config=self.confirmation,
            )
        )
        report = {
            "schema_version": "05j-j-final-report-v1",
            "valid": bool(
                reproduction_report["valid"]
                and role_report["valid"]
                and surface_report["valid"]
                and probe_report["valid"]
            ),
            "decision": "INDEPENDENT_REGENERATIVE_STATE_CONFIRMATION",
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "base_dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05jh": self.artifact_05jh_contract,
            "artifact_05ji": self.artifact_05ji_contract,
            "registered_05jh_reproduction": dict(reproduction_report),
            "external_roles": dict(role_report),
            "external_state_surfaces": dict(surface_report),
            "external_probes": dict(probe_report),
            "primary_support": {
                "pair_count": self.confirmation.registered_pair_count,
                "realized_strata": self.artifact_05ji_report["realized_stratum_counts"],
                "all_preregistered_pairs_used": True,
                "near_only_subset_used_for_decision": False,
            },
            "external_aligned_vs_shifted": {
                "aggregate_rmse_improvement_fraction": specificity,
                "pairwise": pairwise,
                "registered_aggregate_threshold": self.confirmation.oracle_specificity_improvement_fraction,
                "registered_pair_win_threshold": self.confirmation.minimum_pair_win_fraction,
            },
            "causal_boundary_state_confirmed": causal_confirmed,
            "regenerative_state_transition_confirmed": transition_confirmed,
            "candidate_model_authorized": False,
            "micro_rollout_authorized": False,
            "full_training_authorized": False,
            "methodology": {
                "exact_05jh_and_05ji_artifacts_verified": True,
                "all_24_preregistered_pairs_used": True,
                "confirmation_outcomes_not_used_for_feature_or_probe_selection": True,
                "registered_05jh_fit_support_unchanged": True,
                "registered_05jd_transform_and_checkpoints_frozen": True,
                "future_voltage_coordinate_excluded": True,
                "future_state_delta_used_as_diagnostic_oracle_only": True,
                "candidate_training_performed": False,
                "heldout_inputs_extracted": False,
                "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True,
                "experiment": next_step,
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
            "schema_version": "05j-j-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
