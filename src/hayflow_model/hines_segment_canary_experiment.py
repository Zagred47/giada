"""Notebook-05f zero-output segment-conditioned HayFlow micro-canary."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import CompositeFlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_capacity_experiment import (
    HinesCapacityConfig,
    HinesSegmentCapacityExperiment,
)
from .hines_conditioning_experiment import HinesConditioningConfig
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import HinesIsolationConfig, sha256_file
from .hines_layer import require_torch

try:  # Keep provenance utilities importable without local PyTorch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - data-only environments.
    torch = None
    nn = None


EXPECTED_05E_ARCHIVE_SHA256 = (
    "a8e47a979678cef19ca45e5647528bf82dab4f49923dd4d5400705abbe104a48"
)
EXPECTED_05E_MEMBER_SHA256 = {
    "artifact_index.json": "9796e3fdb22bc8b9fe1b2a9be1ecf633a956ee5eddc7b329f876c65ed0a76a8f",
    "capacity_config.json": "486d6ce90e84a7b37be21e92d952e3c3387749c81a08f299f411d493f0aa6043",
    "capacity_probe_report.json": "8a290db42e1a12d4dc8edd7ced27b8c6f89e226569d5edae5239801ae31198fb",
    "capacity_probe_metrics.parquet": "19bad32b2408b5bf1b2ab9b68ce39d04132f488b335c646830e235d02bf0dc2d",
    "final_report.json": "0cab22303747c2df2e2f4b6865774d0f7b2d76fbae090ee73b67f3f109a902d0",
}


@dataclass(frozen=True)
class HinesSegmentCanaryConfig:
    ranks: Tuple[int, ...] = (64, 96)
    desired_train_pair_count: int = 8
    minimum_train_pair_count: int = 4
    desired_heldout_pair_count: int = 4
    minimum_heldout_pair_count: int = 2
    minimum_heldout_split_count: int = 2
    heldout_splits: Tuple[str, ...] = (
        "branching_near_test",
        "branching_far_test",
        "release_identifiability_test",
    )
    maximum_local_steps_searched: int = 4
    maximum_candidates_per_split: int = 512
    state_tolerance: float = 1e-5
    minimum_teacher_distance_mv: float = 0.05
    epochs: int = 1200
    learning_rate: float = 0.01
    bias_learning_rate: float = 0.05
    static_loss_weight: float = 1.0
    branch_delta_loss_weight: float = 1.0
    evaluation_interval: int = 20
    seed: int = 3705
    feature_epsilon: float = 1e-8
    svd_rcond: float = 1e-10
    pair_rmse_mv: float = 1.0
    pair_max_error_mv: float = 5.0
    pair_retention_minimum: float = 0.90
    pair_retention_maximum: float = 1.10

    def validate(self) -> None:
        if tuple(self.ranks) != (64, 96):
            raise ValueError("05f must compare ranks 64 and 96 in that order")
        positive = (
            self.desired_train_pair_count,
            self.minimum_train_pair_count,
            self.desired_heldout_pair_count,
            self.minimum_heldout_pair_count,
            self.minimum_heldout_split_count,
            self.maximum_local_steps_searched,
            self.maximum_candidates_per_split,
            self.state_tolerance,
            self.minimum_teacher_distance_mv,
            self.epochs,
            self.learning_rate,
            self.bias_learning_rate,
            self.static_loss_weight,
            self.branch_delta_loss_weight,
            self.evaluation_interval,
            self.feature_epsilon,
            self.svd_rcond,
        )
        if min(positive) <= 0:
            raise ValueError("05f counts, tolerances, and optimization values must be positive")
        if self.minimum_train_pair_count < 2:
            raise ValueError("05f requires more than one independent training pair")
        if self.desired_train_pair_count < self.minimum_train_pair_count:
            raise ValueError("desired train pairs must cover the minimum")
        if self.desired_heldout_pair_count < self.minimum_heldout_pair_count:
            raise ValueError("desired held-out pairs must cover the minimum")
        if not self.heldout_splits or len(set(self.heldout_splits)) != len(self.heldout_splits):
            raise ValueError("held-out splits must be non-empty and unique")
        if "train" in self.heldout_splits or "validation" in self.heldout_splits:
            raise ValueError("held-out counterfactual splits cannot include train or validation")
        if self.minimum_heldout_split_count > len(self.heldout_splits):
            raise ValueError("minimum held-out split count exceeds configured splits")
        if not 0 < self.pair_retention_minimum < self.pair_retention_maximum:
            raise ValueError("invalid pair-retention interval")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesSegmentCanaryConfig":
        payload = dict(values)
        for name in ("ranks", "heldout_splits"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


if nn is not None:

    class ZeroOutputSpectralSegmentResidual(nn.Module):
        """Low-rank segment residual with frozen train-only spectral basis.

        Segment factors and biases start at zero, so the complete module emits
        exactly zero before its first update. The spectral basis is derived
        only from training pairs and remains frozen.
        """

        def __init__(
            self, segment_count: int, feature_count: int, rank: int,
            spectral_basis: np.ndarray,
        ) -> None:
            super().__init__()
            basis = np.asarray(spectral_basis, dtype=np.float32)
            if basis.shape != (int(rank), int(feature_count)):
                raise ValueError(
                    f"spectral basis has shape {basis.shape}, expected "
                    f"{(int(rank), int(feature_count))}"
                )
            self.segment_factors = nn.Parameter(torch.zeros(
                int(segment_count), int(rank)
            ))
            self.segment_bias = nn.Parameter(torch.zeros(int(segment_count)))
            self.register_buffer("spectral_basis", torch.from_numpy(basis))

        def forward(self, features: Any) -> Any:
            projected = torch.einsum("...sf,rf->...sr", features, self.spectral_basis)
            dynamic = torch.einsum("...sr,sr->...s", projected, self.segment_factors)
            return dynamic + self.segment_bias

else:

    class ZeroOutputSpectralSegmentResidual:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


class HinesSegmentMicroCanaryExperiment(HinesSegmentCapacityExperiment):
    """Fresh held-out micro-canary; it cannot launch full training."""

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        model_config: Any,
        isolation_config: HinesIsolationConfig,
        conditioning_config: HinesConditioningConfig,
        capacity_config: HinesCapacityConfig,
        canary_config: HinesSegmentCanaryConfig,
        checkpoint_05b_source: Path,
        artifact_05c_source: Path,
        artifact_05d_source: Path,
        artifact_05e_source: Path,
        code_revision: Optional[str] = None,
    ) -> None:
        super().__init__(
            bundle, output_dir, model_config, isolation_config,
            conditioning_config, capacity_config, checkpoint_05b_source,
            artifact_05c_source, artifact_05d_source,
            code_revision=code_revision,
        )
        canary_config.validate()
        self.canary = canary_config
        self.artifact_05e_source = Path(artifact_05e_source).resolve()
        self.artifact_05e_contract: Dict[str, Any] = {}
        self.artifact_05e_report: Dict[str, Any] = {}
        self.pair_plan: Dict[str, Any] = {}
        self.training_rows: List[Dict[str, Any]] = []

    def _read_05e_source(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05e_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05E_ARCHIVE_SHA256:
                raise RuntimeError("05e archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                members: Dict[str, bytes] = {}
                resolved: Dict[str, str] = {}
                for suffix in EXPECTED_05E_MEMBER_SHA256:
                    matches = [
                        name for name in archive.namelist()
                        if name.replace("\\", "/").endswith(suffix)
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one 05e member ending in {suffix!r}, found {matches}"
                        )
                    resolved[suffix] = matches[0]
                    members[suffix] = archive.read(matches[0])
            contract: Dict[str, Any] = {
                "source_kind": "original_zip",
                "source_path": str(source),
                "archive_sha256": archive_hash,
                "final_report_member": resolved["final_report.json"],
            }
        elif source.is_dir():
            members = {}
            resolved = {}
            for suffix in EXPECTED_05E_MEMBER_SHA256:
                matches = list(source.rglob(Path(suffix).name))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one extracted 05e member ending in {suffix!r}, "
                        f"found {[str(path) for path in matches]}"
                    )
                resolved[suffix] = matches[0].relative_to(source).as_posix()
                members[suffix] = matches[0].read_bytes()
            contract = {
                "source_kind": "kaggle_extracted_directory",
                "source_path": str(source),
                "archive_sha256": None,
                "final_report_member": resolved["final_report.json"],
            }
        else:
            raise RuntimeError(f"05e artifact source does not exist: {source}")
        observed = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        }
        mismatches = {
            name: {"expected": EXPECTED_05E_MEMBER_SHA256[name], "observed": value}
            for name, value in observed.items()
            if value != EXPECTED_05E_MEMBER_SHA256[name]
        }
        if mismatches:
            raise RuntimeError(f"05e member SHA-256 mismatch: {mismatches}")
        contract["verified_member_sha256"] = observed
        return json.loads(members["final_report.json"]), contract

    def prepare_micro_canary(self) -> Dict[str, Any]:
        base = self.prepare_capacity_probe()
        report, contract = self._read_05e_source()
        blockers = []
        if report.get("diagnosis") != "SEGMENT_CONDITIONED_CAPACITY_SUFFICIENT":
            blockers.append(f"unexpected 05e diagnosis: {report.get('diagnosis')}")
        if report.get("full_training_authorized") is not False:
            blockers.append("05e unexpectedly authorizes full training")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05e and mounted composite fingerprints disagree")
        if int(report.get("selected_segment_conditioned_rank", -1)) != 96:
            blockers.append("05e did not select the registered rank 96")
        if tuple(report.get("branch_pair", ())) != tuple(self.branch_pair or ()):
            blockers.append("05e development pair disagrees with upstream provenance")
        if blockers:
            raise RuntimeError(f"05f provenance blockers: {blockers}")
        self.artifact_05e_report = report
        self.artifact_05e_contract = contract
        payload = {
            "schema_version": "05f-micro-canary-config-v1",
            "canary": asdict(self.canary),
            "artifact_05e": contract,
            "code_revision": self.code_revision,
            "development_pair_excluded_from_training": list(self.branch_pair or ()),
            "full_training_authorized": False,
        }
        _write_json(self.output_dir / "micro_canary_config.json", payload)
        return {**base, **payload}

    def _episode_identity(self, index: int) -> Tuple[str, str, str, int]:
        trajectory = str(self.store.metadata["trajectory_id"][int(index)])
        episode = self.store.episode_by_trajectory.get(trajectory, {})
        return (
            trajectory,
            str(episode.get("episode_id", trajectory)),
            str(episode.get("snapshot_id", episode.get("snapshot_source", ""))),
            int(self.store.metadata["seed"][int(index)]),
        )

    def _action_signature(self, index: int) -> str:
        payload = json.dumps(
            self.store.actions(int(index), "U_realized"),
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _pair_candidates_for_split(self, split: str) -> List[Dict[str, Any]]:
        by_step: Dict[int, List[int]] = {}
        for trajectory, indices in self.store.trajectory_indices.items():
            if not len(indices) or str(self.store.metadata["split"][indices[0]]) != split:
                continue
            for local_step, index in enumerate(
                indices[: self.canary.maximum_local_steps_searched]
            ):
                by_step.setdefault(local_step, []).append(int(index))
        candidate_keys = set()
        candidates: List[Tuple[int, int, float]] = []
        for local_step in sorted(by_step):
            indices = np.asarray(by_step[local_step], dtype=np.int64)
            if len(indices) < 2:
                continue
            states = self.store.read_state(indices, "t")
            voltage = np.round(
                states[:, : self.layout.segment_count], decimals=5
            ).astype(np.float32)
            buckets: Dict[bytes, List[int]] = {}
            for position, row in enumerate(voltage):
                signature = hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
                buckets.setdefault(signature, []).append(position)
            for positions in buckets.values():
                if len(positions) < 2:
                    continue
                action_groups: Dict[str, List[int]] = {}
                for position in positions:
                    index = int(indices[position])
                    action_groups.setdefault(self._action_signature(index), []).append(position)
                groups = [action_groups[key] for key in sorted(action_groups)]
                for left_group in range(len(groups)):
                    for right_group in range(left_group + 1, len(groups)):
                        for left_position in groups[left_group][:2]:
                            for right_position in groups[right_group][:2]:
                                left = int(indices[left_position])
                                right = int(indices[right_position])
                                if int(self.store.metadata["step_index"][left]) != int(
                                    self.store.metadata["step_index"][right]
                                ):
                                    continue
                                left_identity = self._episode_identity(left)
                                right_identity = self._episode_identity(right)
                                if left_identity[1] == right_identity[1]:
                                    continue
                                key = tuple(sorted((left, right)))
                                if key in candidate_keys:
                                    continue
                                state_error = float(np.max(np.abs(
                                    states[left_position] - states[right_position]
                                )))
                                if state_error > self.canary.state_tolerance:
                                    continue
                                candidate_keys.add(key)
                                candidates.append((left, right, state_error))
                                if len(candidates) >= self.canary.maximum_candidates_per_split:
                                    break
                            if len(candidates) >= self.canary.maximum_candidates_per_split:
                                break
                        if len(candidates) >= self.canary.maximum_candidates_per_split:
                            break
                    if len(candidates) >= self.canary.maximum_candidates_per_split:
                        break
                if len(candidates) >= self.canary.maximum_candidates_per_split:
                    break
            if len(candidates) >= self.canary.maximum_candidates_per_split:
                break
        if not candidates:
            print(f"[HayFlow 05f][pair search] {split}: 0 candidates", flush=True)
            return []
        unique_indices = sorted({value for left, right, _ in candidates for value in (left, right)})
        targets = self.store.read_state(unique_indices, "t_plus_1", categories=("voltage",))
        by_index = {index: targets[position] for position, index in enumerate(unique_indices)}
        rows = []
        for left, right, state_error in candidates:
            teacher_distance = float(np.sqrt(np.mean(
                (by_index[left] - by_index[right]) ** 2
            ) + 1e-12))
            if teacher_distance < self.canary.minimum_teacher_distance_mv:
                continue
            left_identity = self._episode_identity(left)
            right_identity = self._episode_identity(right)
            rows.append({
                "left_index": left,
                "right_index": right,
                "split": split,
                "step_index": int(self.store.metadata["step_index"][left]),
                "state_max_error": state_error,
                "teacher_distance_mv": teacher_distance,
                "left_trajectory_id": left_identity[0],
                "right_trajectory_id": right_identity[0],
                "left_episode_id": left_identity[1],
                "right_episode_id": right_identity[1],
                "left_snapshot_id": left_identity[2],
                "right_snapshot_id": right_identity[2],
                "left_seed": left_identity[3],
                "right_seed": right_identity[3],
                "left_action_sha256": self._action_signature(left),
                "right_action_sha256": self._action_signature(right),
            })
        rows.sort(key=lambda row: (
            -row["teacher_distance_mv"], row["left_index"], row["right_index"]
        ))
        print(
            f"[HayFlow 05f][pair search] {split}: {len(rows)} valid candidates",
            flush=True,
        )
        return rows

    @staticmethod
    def _select_disjoint_pairs(
        candidates: Sequence[Mapping[str, Any]], count: int,
        excluded_indices: Sequence[int] = (),
        excluded_episode_ids: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        excluded = {int(value) for value in excluded_indices}
        excluded_episodes = {str(value) for value in excluded_episode_ids}
        used_episodes = set()
        selected = []
        for raw in candidates:
            row = dict(raw)
            if {int(row["left_index"]), int(row["right_index"])} & excluded:
                continue
            episodes = {str(row["left_episode_id"]), str(row["right_episode_id"])}
            if episodes & excluded_episodes:
                continue
            if episodes & used_episodes:
                continue
            selected.append(row)
            used_episodes.update(episodes)
            if len(selected) == int(count):
                break
        return selected

    def _select_heldout_pairs(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        used_episodes = set()

        def add_first(rows: Sequence[Mapping[str, Any]]) -> bool:
            for raw in rows:
                row = dict(raw)
                episodes = {
                    str(row["left_episode_id"]), str(row["right_episode_id"])
                }
                if episodes & used_episodes:
                    continue
                selected.append(row)
                used_episodes.update(episodes)
                return True
            return False

        # Reserve protocol diversity before filling by teacher separation.
        for split in self.canary.heldout_splits:
            add_first([row for row in candidates if row["split"] == split])
            if len(selected) == self.canary.desired_heldout_pair_count:
                return selected
        for row in candidates:
            if len(selected) == self.canary.desired_heldout_pair_count:
                break
            if any(
                row["left_index"] == chosen["left_index"]
                and row["right_index"] == chosen["right_index"]
                for chosen in selected
            ):
                continue
            add_first([row])
        return selected

    def build_pair_plan(self) -> Dict[str, Any]:
        development = list(self.branch_pair or ())
        development_episodes = sorted({
            self._episode_identity(index)[1] for index in development
        })
        train_candidates = self._pair_candidates_for_split("train")
        train = self._select_disjoint_pairs(
            train_candidates, self.canary.desired_train_pair_count,
            excluded_indices=development,
            excluded_episode_ids=development_episodes,
        )
        heldout_candidates = []
        heldout_candidate_counts = {}
        for split in self.canary.heldout_splits:
            rows = self._pair_candidates_for_split(split)
            heldout_candidate_counts[split] = len(rows)
            heldout_candidates.extend(rows)
        heldout_candidates.sort(key=lambda row: (
            -row["teacher_distance_mv"], row["split"],
            row["left_index"], row["right_index"],
        ))
        heldout = self._select_heldout_pairs(heldout_candidates)
        blockers = []
        if len(train) < self.canary.minimum_train_pair_count:
            blockers.append(
                f"only {len(train)} disjoint train pairs; "
                f"minimum is {self.canary.minimum_train_pair_count}"
            )
        if len(heldout) < self.canary.minimum_heldout_pair_count:
            blockers.append(
                f"only {len(heldout)} disjoint held-out pairs; "
                f"minimum is {self.canary.minimum_heldout_pair_count}"
            )
        heldout_split_count = len({row["split"] for row in heldout})
        if heldout_split_count < self.canary.minimum_heldout_split_count:
            blockers.append(
                f"held-out pairs cover {heldout_split_count} splits; minimum is "
                f"{self.canary.minimum_heldout_split_count}"
            )
        train_episodes = {
            value for row in train
            for value in (row["left_episode_id"], row["right_episode_id"])
        }
        heldout_episodes = {
            value for row in heldout
            for value in (row["left_episode_id"], row["right_episode_id"])
        }
        overlap = sorted(train_episodes & heldout_episodes)
        if overlap:
            blockers.append(f"train/held-out episode overlap: {overlap}")
        if any(row["split"] != "train" for row in train):
            blockers.append("non-train pair entered the optimization set")
        if any(row["split"] in {"train", "validation"} for row in heldout):
            blockers.append("train or validation pair entered held-out evaluation")
        development_episode_overlap = sorted(train_episodes & set(development_episodes))
        if development_episode_overlap:
            blockers.append(
                f"05e development episodes entered training: {development_episode_overlap}"
            )
        payload_for_hash = {"train": train, "heldout": heldout, "development": development}
        plan_hash = hashlib.sha256(json.dumps(
            payload_for_hash, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        plan = {
            "schema_version": "05f-pair-plan-v1",
            "valid": not blockers,
            "blockers": blockers,
            "train_pair_count": len(train),
            "heldout_pair_count": len(heldout),
            "heldout_split_count": heldout_split_count,
            "heldout_splits": sorted({row["split"] for row in heldout}),
            "train_candidate_count": len(train_candidates),
            "heldout_candidate_counts": heldout_candidate_counts,
            "development_pair": development,
            "development_episode_ids": development_episodes,
            "development_pair_excluded_from_training": (
                not development_episode_overlap
                and not any(
                    set(development) & {row["left_index"], row["right_index"]}
                    for row in train
                )
            ),
            "episode_overlap": overlap,
            "train_pairs": train,
            "heldout_pairs": heldout,
            "pair_plan_sha256": plan_hash,
        }
        _write_json(self.output_dir / "pair_plan.json", plan)
        write_parquet(
            self.output_dir / "pair_plan.parquet",
            [
                {"role": role, "pair_position": position, **row}
                for role, rows in (("train", train), ("heldout", heldout))
                for position, row in enumerate(rows)
            ],
        )
        self.pair_plan = plan
        if blockers:
            raise RuntimeError(f"05f pair-plan blockers: {blockers}")
        return plan

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _pair_indices(rows: Sequence[Mapping[str, Any]]) -> List[int]:
        return [
            int(value) for row in rows
            for value in (row["left_index"], row["right_index"])
        ]

    def _extract_feature_sets(self) -> Dict[str, Any]:
        if not self.pair_plan.get("valid"):
            raise RuntimeError("build_pair_plan() must succeed first")
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        roles = {
            "train": self._pair_indices(self.pair_plan["train_pairs"]),
            "development": list(self.pair_plan["development_pair"]),
            "heldout": self._pair_indices(self.pair_plan["heldout_pairs"]),
        }
        ordered = []
        slices = {}
        for role, values in roles.items():
            start = len(ordered)
            ordered.extend(values)
            slices[role] = slice(start, len(ordered))
        base_t, features_t, target_t, compatibility = self._fixed_base_and_features(
            ordered, device
        )
        base = base_t.detach().cpu().double().numpy()
        features = features_t.detach().cpu().double().numpy()
        target = target_t.detach().cpu().double().numpy()
        train_slice = slices["train"]
        train_features = features[train_slice]
        mean = train_features.mean(axis=(0, 1), keepdims=True)
        raw_std = train_features.std(axis=(0, 1), keepdims=True)
        scale = np.maximum(raw_std, self.canary.feature_epsilon)
        standardized = (features - mean) / scale
        result = {}
        for role, selection in slices.items():
            count = len(roles[role]) // 2
            result[role] = {
                "indices": roles[role],
                "base": base[selection].reshape(count, 2, base.shape[1]),
                "features": standardized[selection].reshape(
                    count, 2, features.shape[1], features.shape[2]
                ),
                "target": target[selection].reshape(count, 2, target.shape[1]),
            }
        contract = {
            "schema_version": "05f-feature-contract-v1",
            "checkpoint_compatibility": compatibility,
            "feature_count": int(features.shape[2]),
            "segment_count": int(features.shape[1]),
            "feature_mean_sha256": hashlib.sha256(
                np.ascontiguousarray(mean).tobytes()
            ).hexdigest(),
            "feature_scale_sha256": hashlib.sha256(
                np.ascontiguousarray(scale).tobytes()
            ).hexdigest(),
            "minimum_raw_std": float(raw_std.min()),
            "maximum_raw_std": float(raw_std.max()),
            "constant_feature_count": int(np.sum(
                raw_std < self.canary.feature_epsilon
            )),
            "normalization_fit_roles": ["train"],
            "heldout_targets_used_for_normalization": False,
            "base_h2_frozen": True,
            "teacher_encoder_updated": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "feature_contract.json", contract)
        result["contract"] = contract
        return result

    def _spectral_basis(self, train: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        x = np.asarray(train["features"], dtype=np.float64).reshape(
            -1, train["features"].shape[-2], train["features"].shape[-1]
        )
        residual = np.asarray(train["target"] - train["base"], dtype=np.float64).reshape(
            -1, train["target"].shape[-1]
        )
        centered_x = x - x.mean(axis=0, keepdims=True)
        centered_y = residual - residual.mean(axis=0, keepdims=True)
        segment_count, feature_count = x.shape[1:]
        coefficients = np.zeros((segment_count, feature_count), dtype=np.float64)
        local_ranks = []
        for segment in range(segment_count):
            solution, _, rank, _ = np.linalg.lstsq(
                centered_x[:, segment, :], centered_y[:, segment],
                rcond=self.canary.svd_rcond,
            )
            coefficients[segment] = solution
            local_ranks.append(int(rank))
        _, singular_values, right = np.linalg.svd(coefficients, full_matrices=False)
        tolerance = self.canary.svd_rcond * (
            float(singular_values[0]) if len(singular_values) else 0.0
        )
        report = {
            "coefficient_matrix_rank": int(np.sum(singular_values > tolerance)),
            "coefficient_singular_values": [float(value) for value in singular_values],
            "minimum_local_design_rank": min(local_ranks),
            "maximum_local_design_rank": max(local_ranks),
            "basis_fit_roles": ["train"],
            "heldout_targets_used_for_basis": False,
            "coefficient_matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(coefficients).tobytes()
            ).hexdigest(),
        }
        return right, report

    def _pair_set_metrics(
        self, predicted: np.ndarray, target: np.ndarray
    ) -> Dict[str, Any]:
        pair_rows = []
        for position in range(len(predicted)):
            metrics = self._numpy_voltage_metrics(predicted[position], target[position])
            metrics["pair_position"] = position
            metrics["passed"] = self._passes(metrics, pair=True)
            pair_rows.append(metrics)
        error = predicted - target
        return {
            "pair_count": len(pair_rows),
            "aggregate_voltage_rmse_mv": float(np.sqrt(np.mean(error ** 2))),
            "maximum_segment_error_mv": float(np.max(np.abs(error))),
            "maximum_peak_error_mv": float(np.max(np.abs(
                predicted.max(axis=2) - target.max(axis=2)
            ))),
            "minimum_branching_retention": float(min(
                row["branching_retention"] for row in pair_rows
            )),
            "median_branching_retention": float(np.median([
                row["branching_retention"] for row in pair_rows
            ])),
            "maximum_branching_retention": float(max(
                row["branching_retention"] for row in pair_rows
            )),
            "all_pairs_passed": all(row["passed"] for row in pair_rows),
            "pair_metrics": pair_rows,
        }

    def _train_rank(
        self,
        rank: int,
        data: Mapping[str, Any],
        spectral_basis: np.ndarray,
        spectral_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(self.canary.seed + int(rank))
        train = data["train"]
        model = ZeroOutputSpectralSegmentResidual(
            train["features"].shape[-2], train["features"].shape[-1], rank,
            spectral_basis[:rank],
        ).to(device)
        tensors = {}
        for role in ("train", "development", "heldout"):
            tensors[role] = {
                name: torch.as_tensor(values, dtype=torch.float32, device=device)
                for name, values in data[role].items()
                if name in {"base", "features", "target"}
            }
        with torch.no_grad():
            initial_residual = model(tensors["train"]["features"])
        zero_output_max = float(initial_residual.abs().max().detach())
        if zero_output_max != 0.0:
            raise RuntimeError(f"rank {rank} residual is not exactly zero at initialization")
        optimizer = torch.optim.Adam([
            {
                "params": [model.segment_factors],
                "lr": self.canary.learning_rate,
            },
            {
                "params": [model.segment_bias],
                "lr": self.canary.bias_learning_rate,
            },
        ])
        progress = Progress(f"segment micro-canary rank {rank}", self.canary.epochs)
        history = []
        best_loss = math.inf
        best_epoch = 0
        best_state = None
        nonfinite = False
        for epoch in range(self.canary.epochs):
            optimizer.zero_grad(set_to_none=True)
            residual = model(tensors["train"]["features"])
            target_residual = tensors["train"]["target"] - tensors["train"]["base"]
            point_loss = torch.mean((residual - target_residual) ** 2)
            static_loss = torch.mean((
                residual.mean(dim=1) - target_residual.mean(dim=1)
            ) ** 2)
            branch_loss = torch.mean((
                (residual[:, 0] - residual[:, 1])
                - (target_residual[:, 0] - target_residual[:, 1])
            ) ** 2)
            loss = (
                point_loss
                + self.canary.static_loss_weight * static_loss
                + self.canary.branch_delta_loss_weight * branch_loss
            )
            if not bool(torch.isfinite(loss)):
                nonfinite = True
                break
            loss.backward()
            gradient_norm = float(torch.sqrt(sum(
                torch.sum(parameter.grad.detach() ** 2)
                for parameter in model.parameters() if parameter.grad is not None
            )).detach())
            if not math.isfinite(gradient_norm):
                nonfinite = True
                break
            optimizer.step()
            if (
                epoch == 0
                or (epoch + 1) % self.canary.evaluation_interval == 0
                or epoch + 1 == self.canary.epochs
            ):
                with torch.no_grad():
                    post_residual = model(tensors["train"]["features"])
                    post_point = torch.mean((post_residual - target_residual) ** 2)
                    post_static = torch.mean((
                        post_residual.mean(dim=1) - target_residual.mean(dim=1)
                    ) ** 2)
                    post_branch = torch.mean((
                        (post_residual[:, 0] - post_residual[:, 1])
                        - (target_residual[:, 0] - target_residual[:, 1])
                    ) ** 2)
                    post_loss = (
                        post_point
                        + self.canary.static_loss_weight * post_static
                        + self.canary.branch_delta_loss_weight * post_branch
                    )
                    train_prediction = (
                        tensors["train"]["base"] + post_residual
                    ).detach().cpu().numpy()
                loss_value = float(post_loss.detach())
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                train_metrics = self._pair_set_metrics(
                    train_prediction, train["target"]
                )
                row = {
                    "rank": int(rank),
                    "epoch": epoch + 1,
                    "loss": loss_value,
                    "point_loss": float(post_point.detach()),
                    "static_loss": float(post_static.detach()),
                    "branch_delta_loss": float(post_branch.detach()),
                    "gradient_norm": gradient_norm,
                    "train_aggregate_rmse_mv": train_metrics["aggregate_voltage_rmse_mv"],
                    "train_maximum_segment_error_mv": train_metrics["maximum_segment_error_mv"],
                    "train_median_branching_retention": train_metrics["median_branching_retention"],
                    "train_all_pairs_passed": train_metrics["all_pairs_passed"],
                }
                history.append(row)
                self.training_rows.append(dict(row))
                progress.update(
                    epoch + 1,
                    f"loss={loss_value:.4g} "
                    f"V={train_metrics['aggregate_voltage_rmse_mv']:.3g} "
                    f"max={train_metrics['maximum_segment_error_mv']:.3g}",
                )
        if best_state is None:
            raise RuntimeError(f"rank {rank} did not produce a finite checkpoint")
        model.load_state_dict(best_state)
        evaluations = {}
        with torch.no_grad():
            for role in ("train", "development", "heldout"):
                predicted = (
                    tensors[role]["base"] + model(tensors[role]["features"])
                ).detach().cpu().numpy()
                evaluations[role] = self._pair_set_metrics(
                    predicted, data[role]["target"]
                )
        checkpoint = {
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "rank": int(rank),
            "best_epoch": best_epoch,
            "best_train_loss": best_loss,
            "pair_plan_sha256": self.pair_plan["pair_plan_sha256"],
            "feature_contract": data["contract"],
        }
        checkpoint_path = self.output_dir / "checkpoints" / f"rank_{rank}.pt"
        torch.save(checkpoint, checkpoint_path)
        result = {
            "rank": int(rank),
            "zero_output_max_absolute_mv": zero_output_max,
            "zero_initialized": zero_output_max == 0.0,
            "spectral_basis_frozen": True,
            "spectral_basis_fit_roles": ["train"],
            "heldout_targets_used_for_basis": False,
            "heldout_metrics_used_for_checkpoint_selection": False,
            "epochs_budget": self.canary.epochs,
            "factor_learning_rate": self.canary.learning_rate,
            "bias_learning_rate": self.canary.bias_learning_rate,
            "epochs_completed": history[-1]["epoch"] if history else 0,
            "best_epoch_by_train_loss": best_epoch,
            "best_train_loss": best_loss,
            "nonfinite": nonfinite,
            "gradient_clipping_applied": False,
            "trainable_parameter_count": int(sum(
                parameter.numel() for parameter in model.parameters()
            )),
            "deployed_parameter_count": int(
                train["features"].shape[-2]
                + rank * (
                    train["features"].shape[-2] + train["features"].shape[-1]
                )
            ),
            "spectral_coefficient_matrix_rank": spectral_report[
                "coefficient_matrix_rank"
            ],
            "evaluations": evaluations,
            "passed_train": evaluations["train"]["all_pairs_passed"],
            "passed_heldout": evaluations["heldout"]["all_pairs_passed"],
            "checkpoint": checkpoint_path.relative_to(self.output_dir).as_posix(),
        }
        _write_json(self.output_dir / f"rank_{rank}_report.json", result)
        return result

    def run_micro_canary(self) -> Dict[str, Any]:
        require_torch()
        data = self._extract_feature_sets()
        spectral_basis, spectral_report = self._spectral_basis(data["train"])
        _write_json(self.output_dir / "spectral_basis_report.json", spectral_report)
        runs = [
            self._train_rank(rank, data, spectral_basis, spectral_report)
            for rank in self.canary.ranks
        ]
        write_parquet(
            self.output_dir / "micro_canary_training_history.parquet",
            self.training_rows,
        )
        summary_rows = []
        for run in runs:
            for role, metrics in run["evaluations"].items():
                summary_rows.append({
                    "rank": run["rank"],
                    "role": role,
                    "pair_count": metrics["pair_count"],
                    "aggregate_voltage_rmse_mv": metrics["aggregate_voltage_rmse_mv"],
                    "maximum_segment_error_mv": metrics["maximum_segment_error_mv"],
                    "maximum_peak_error_mv": metrics["maximum_peak_error_mv"],
                    "minimum_branching_retention": metrics["minimum_branching_retention"],
                    "median_branching_retention": metrics["median_branching_retention"],
                    "maximum_branching_retention": metrics["maximum_branching_retention"],
                    "all_pairs_passed": metrics["all_pairs_passed"],
                })
        write_parquet(
            self.output_dir / "micro_canary_metrics.parquet", summary_rows
        )
        report = {
            "schema_version": "05f-micro-canary-v1",
            "valid": bool(
                len(runs) == 2
                and all(run["zero_initialized"] and not run["nonfinite"] for run in runs)
                and data["contract"]["base_h2_frozen"]
                and data["contract"]["heldout_targets_used_for_normalization"] is False
            ),
            "pair_plan_sha256": self.pair_plan["pair_plan_sha256"],
            "feature_contract": data["contract"],
            "spectral_basis": spectral_report,
            "runs": runs,
            "any_rank_passed_heldout": any(run["passed_heldout"] for run in runs),
            "smallest_passing_heldout_rank": next(
                (run["rank"] for run in runs if run["passed_heldout"]), None
            ),
            "full_training_authorized": False,
        }
        _write_json(self.output_dir / "micro_canary_report.json", report)
        self._plot_micro_canary(report)
        return report

    def _plot_micro_canary(self, report: Mapping[str, Any]) -> None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for run in report["runs"]:
            rows = [row for row in self.training_rows if row["rank"] == run["rank"]]
            axes[0].semilogy(
                [row["epoch"] for row in rows],
                [max(row["loss"], 1e-12) for row in rows],
                label=f"rank {run['rank']}",
            )
        axes[0].set(title="Train-only optimization", xlabel="epoch", ylabel="loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        ranks = [run["rank"] for run in report["runs"]]
        width = 0.34
        x = np.arange(len(ranks))
        axes[1].bar(
            x - width / 2,
            [run["evaluations"]["train"]["aggregate_voltage_rmse_mv"] for run in report["runs"]],
            width, label="train",
        )
        axes[1].bar(
            x + width / 2,
            [run["evaluations"]["heldout"]["aggregate_voltage_rmse_mv"] for run in report["runs"]],
            width, label="held-out",
        )
        axes[1].axhline(self.canary.pair_rmse_mv, color="black", linestyle="--", label="RMSE gate")
        axes[1].set(xticks=x, xticklabels=ranks, xlabel="rank", ylabel="mV", title="Counterfactual generalization")
        axes[1].grid(alpha=0.3, axis="y")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "micro_canary_diagnostics.png", dpi=160)
        plt.close(figure)

    def finalize_micro_canary(self, canary_report: Mapping[str, Any]) -> Dict[str, Any]:
        by_rank = {int(run["rank"]): run for run in canary_report["runs"]}
        passing = [rank for rank in self.canary.ranks if by_rank[rank]["passed_heldout"]]
        if passing:
            diagnosis = (
                "RANK64_SEGMENT_CONDITIONING_GENERALIZES"
                if min(passing) == 64
                else "RANK96_SEGMENT_CONDITIONING_GENERALIZES"
            )
            next_experiment = "05g_segment_conditioned_multistep_micro_rollout"
        elif any(run["passed_train"] for run in canary_report["runs"]):
            diagnosis = "SEGMENT_CONDITIONED_OVERFIT_NO_HELDOUT_GENERALIZATION"
            next_experiment = "05g_segment_conditioning_regularization_revision"
        else:
            diagnosis = "SEGMENT_CONDITIONED_MICRO_CANARY_OPTIMIZATION_FAILURE"
            next_experiment = "05g_segment_conditioning_optimization_audit"
        report = {
            "schema_version": "05f-final-report-v1",
            "valid": bool(canary_report.get("valid") and self.pair_plan.get("valid")),
            "decision": "DIAGNOSTIC_ONLY_NO_FULL_TRAINING",
            "full_training_authorized": False,
            "diagnosis": diagnosis,
            "selected_rank_for_next_diagnostic": min(passing) if passing else None,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "pair_plan": self.pair_plan,
            "checkpoint_05b": self.checkpoint_contract,
            "artifact_05c": self.artifact_05c_contract,
            "artifact_05d": self.artifact_05d_contract,
            "artifact_05e": self.artifact_05e_contract,
            "micro_canary": canary_report,
            "methodology": {
                "base_h2_frozen": True,
                "spectral_basis_fit_on_train_only": True,
                "heldout_targets_used_for_training": False,
                "heldout_metrics_used_for_checkpoint_selection": False,
                "development_pair_from_05e_excluded_from_training": True,
                "teacher_encoder_updated": False,
                "rollout_performed": False,
                "full_training_path_present": False,
            },
            "next_step": {
                "experiment": next_experiment,
                "full_training_authorized": False,
                "requires_new_notebook": True,
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
            "schema_version": "05f-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
