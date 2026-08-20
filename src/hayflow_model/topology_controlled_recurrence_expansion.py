"""Train-only topology controls for the rollout-aware recurrent signal.

05l established that both a morphology GraphGRU and a parameter-matched
ordered ConvGRU beat persistence in closed-loop eight-millisecond rollouts,
but their difference was too small to attribute the gain to morphology.  05m
keeps the recurrent training contract fixed, expands unused train support and
adds a topology-preserving relabeling control.  No validation, test or sealed
fresh-test transition is read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import CompositeFlowmapBundle

from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .rollout_aware_architecture_canary import (
    MorphologyGraphGRU,
    OrderedConvGRUControl,
    RolloutAwareArchitectureCanary,
    RolloutAwareArchitectureCanaryConfig,
    disjoint_episode_components_by_regime,
    model_parameter_count,
    torch,
)


EXPECTED_05L_ARCHIVE_SHA256 = (
    "9a5c423aba80da51830a57c6f97808a3366069c6e42807193143d49de13be634"
)
EXPECTED_05L_INDEX_SHA256 = (
    "03e77d8a7104a25b2636f05f406031b01b81604c9e15d1dcb4d12a859ac0d757"
)
EXPECTED_05L_FINAL_SHA256 = (
    "a43227c545088b7bc1f9dbf2bafbac9f73534504033cbd365b9e6fe805cca930"
)


def verified_rollout_architecture_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="rollout_aware_architecture_canary_config.json",
        archive_sha256=EXPECTED_05L_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05L_INDEX_SHA256,
        final_sha256=EXPECTED_05L_FINAL_SHA256,
    )


@dataclass(frozen=True)
class TopologyControlledRecurrenceConfig(RolloutAwareArchitectureCanaryConfig):
    windows_per_episode: int = 3
    maximum_extra_fit_components_per_regime: int = 8
    topology_relabel_seed_offset: int = 905100
    minimum_topology_gain_fraction: float = 0.05
    minimum_paired_topology_win_count: int = 2

    def validate(self) -> None:
        super().validate()
        if self.maximum_extra_fit_components_per_regime <= 0:
            raise ValueError("05m train-support expansion must be positive")
        if not 0 < self.minimum_topology_gain_fraction < 1:
            raise ValueError("05m topology materiality threshold is invalid")
        if self.minimum_paired_topology_win_count not in {2, 3}:
            raise ValueError("05m paired topology win gate is invalid")


def expanded_train_episode_roles(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    config: TopologyControlledRecurrenceConfig,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Preserve 05l calibration/development and add unused components to fit."""

    grouped = disjoint_episode_components_by_regime(
        episode_rows, role_seed=config.role_seed
    )
    roles: Dict[str, List[Dict[str, Any]]] = {
        "fit": [], "calibration": [], "development": []
    }
    expansion: Dict[str, Any] = {}
    for regime, components in sorted(grouped.items()):
        minimum = (
            config.fit_groups_per_regime
            + config.calibration_groups_per_regime
            + config.development_groups_per_regime
        )
        if len(components) < minimum:
            raise RuntimeError(
                f"05m regime {regime!r} has {len(components)} components; "
                f"at least {minimum} are required"
            )
        fit_stop = config.fit_groups_per_regime
        calibration_stop = fit_stop + config.calibration_groups_per_regime
        development_stop = calibration_stop + config.development_groups_per_regime
        selected = {
            "fit": list(components[:fit_stop]),
            "calibration": list(components[fit_stop:calibration_stop]),
            "development": list(components[calibration_stop:development_stop]),
        }
        extras = list(
            components[
                development_stop : development_stop
                + config.maximum_extra_fit_components_per_regime
            ]
        )
        selected["fit"].extend(extras)
        expansion[regime] = {
            "available_component_count": len(components),
            "original_fit_component_count": config.fit_groups_per_regime,
            "extra_fit_component_count": len(extras),
            "expanded_fit_component_count": len(selected["fit"]),
        }
        for role, role_components in selected.items():
            for component in role_components:
                for source in component:
                    row = dict(source)
                    row["05l_regime"] = regime
                    row["05m_regime"] = regime
                    row["05m_role"] = role
                    roles[role].append(row)
    if not any(row["extra_fit_component_count"] for row in expansion.values()):
        raise RuntimeError("05m found no unused train components to expand")
    return roles, expansion


def topology_relabelled_parent_ids(
    parent_ids: Sequence[int], *, seed: int
) -> np.ndarray:
    """Relabel the exact tree while keeping roots fixed and features unmoved."""

    parent = np.asarray(parent_ids, dtype=np.int64)
    count = len(parent)
    if count == 0 or np.any(parent < 0) or np.any(parent >= count):
        raise ValueError("invalid parent array")
    roots = np.flatnonzero(parent == np.arange(count))
    movable = np.flatnonzero(parent != np.arange(count))
    permutation = np.arange(count, dtype=np.int64)
    shuffled = movable.copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)
    permutation[movable] = shuffled
    permutation[roots] = roots
    result = np.empty_like(parent)
    for old_segment in range(count):
        result[permutation[old_segment]] = permutation[parent[old_segment]]
    return result


class TopologyControlledRecurrenceExpansion(RolloutAwareArchitectureCanary):
    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: TopologyControlledRecurrenceConfig,
        artifact_05l_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05l_source,
            code_revision=code_revision,
        )
        self.artifact_05l_source = Path(artifact_05l_source).resolve()
        self.expansion_report: Dict[str, Any] = {}
        self._active_model_seed = int(config.seeds[0])

    @property
    def topology_config(self) -> TopologyControlledRecurrenceConfig:
        return self.config  # type: ignore[return-value]

    def _models(self, device: Any) -> Dict[str, Any]:
        parent = self._parent_ids()
        common = {
            "segment_static": self.store.layout.segment_static,
            "hidden_width": self.config.hidden_width,
            "voltage_delta_limit_mv": self.config.voltage_delta_limit_mv,
        }
        relabelled = topology_relabelled_parent_ids(
            parent,
            seed=(
                self.topology_config.topology_relabel_seed_offset
                + int(self._active_model_seed)
            ),
        )
        authentic = MorphologyGraphGRU(parent_ids=parent, **common).to(device)
        relabel_control = MorphologyGraphGRU(
            parent_ids=relabelled, **common
        ).to(device)
        # Copy parameters, not buffers: the graph pair starts identically while
        # retaining different parent/child topology tensors.
        with torch.no_grad():
            authentic_parameters = dict(authentic.named_parameters())
            relabel_parameters = dict(relabel_control.named_parameters())
            if authentic_parameters.keys() != relabel_parameters.keys():
                raise RuntimeError("05m graph parameter contracts differ")
            for name, value in authentic_parameters.items():
                relabel_parameters[name].copy_(value)
        convolution = OrderedConvGRUControl(parent_ids=parent, **common).to(device)
        return {
            "authentic_morphology_graph_gru": authentic,
            "relabelled_morphology_graph_control": relabel_control,
            "ordered_convgru_control": convolution,
        }

    def prepare(self) -> Dict[str, Any]:
        _, final, contract = verified_rollout_architecture_artifact_root(
            self.artifact_05l_source,
            self.output_dir.parent / ".05m_artifact_cache" / "05l",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05l artifact is not decision-grade")
        if final.get("diagnosis") != "ROLLOUT_AWARE_RECURRENCE_SIGNAL_TOPOLOGY_UNRESOLVED":
            blockers.append("05l diagnosis does not authorize topology isolation")
        if final.get("next_step") != "05m_topology_controlled_recurrence_expansion":
            blockers.append("05l next step is not 05m")
        if final.get("full_training_authorized"):
            blockers.append("05l unexpectedly authorized full training")
        self.artifact_contract = contract
        self.roles, self.expansion_report = expanded_train_episode_roles(
            self.store.episode_rows, config=self.topology_config
        )
        seed_overlap = self._role_overlap(self.roles, "seed")
        snapshot_overlap = self._role_overlap(self.roles, "snapshot_id")
        snapshot_source_overlap = self._role_overlap(
            self.roles, "snapshot_source"
        )
        trajectory_overlap = self._role_overlap(self.roles, "trajectory_id")
        if (
            seed_overlap
            or snapshot_overlap
            or snapshot_source_overlap
            or trajectory_overlap
        ):
            blockers.append("05m role isolation failed")
        self.windows = self._build_windows()
        self.materialized = {
            role: self._materialize_role(values)
            for role, values in self.windows.items()
        }
        if torch is None:
            blockers.append("PyTorch is unavailable")
            counts: Dict[str, int] = {}
            ratio = math.inf
        else:
            models = self._models(torch.device("cpu"))
            counts = {
                name: model_parameter_count(model) for name, model in models.items()
            }
            ratio = max(counts.values()) / min(counts.values())
            if ratio > self.config.maximum_parameter_ratio:
                blockers.append("05m candidate parameter counts differ")
        report = {
            "schema_version": "05m-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05l": contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "source_split_used": "train_only",
            "validation_or_test_loaded": False,
            "fresh_05jo_loaded": False,
            "role_episode_counts": {
                role: len(rows) for role, rows in self.roles.items()
            },
            "role_window_counts": {
                role: len(rows) for role, rows in self.windows.items()
            },
            "role_overlap": {
                "seed": seed_overlap,
                "snapshot": snapshot_overlap,
                "snapshot_source": snapshot_source_overlap,
                "trajectory": trajectory_overlap,
            },
            "support_expansion": self.expansion_report,
            "extra_fit_component_count": sum(
                row["extra_fit_component_count"]
                for row in self.expansion_report.values()
            ),
            "parameter_counts": counts,
            "parameter_ratio": ratio,
            "paired_graph_initialization_by_seed": True,
            "relabel_control_preserves_tree_shape": True,
            "relabel_control_keeps_segment_features_unmoved": True,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "topology_controlled_recurrence_config.json",
            {
                "schema_version": "05m-config-v1",
                "config": asdict(self.topology_config),
                "artifact_05l": contract,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05m preflight failed: {blockers}")
        return report

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None:
            raise RuntimeError("05m requires PyTorch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        development = self._tensor_role("development", device)
        families = (
            "authentic_morphology_graph_gru",
            "relabelled_morphology_graph_control",
            "ordered_convgru_control",
        )
        runs: Dict[str, Dict[str, Any]] = {family: {} for family in families}
        for family in families:
            for seed in self.config.seeds:
                self._active_model_seed = int(seed)
                model, single = self._train_one(family, int(seed), device)
                metrics = self._evaluate_arrays(model, development)
                eight = metrics["8"]
                improvement = 1.0 - eight["endpoint_rmse_mv"] / max(
                    eight["persistence_endpoint_rmse_mv"], 1e-12
                )
                single.update(
                    development=metrics,
                    development_improvement_vs_persistence_fraction=improvement,
                    passed=bool(
                        all(row["finite"] for row in metrics.values())
                        and sum(
                            row["physical_voltage_violation_count"]
                            for row in metrics.values()
                        )
                        == 0
                        and improvement
                        >= self.config.minimum_improvement_vs_persistence_fraction
                    ),
                )
                runs[family][str(seed)] = single
        rmse = {
            family: {
                seed: row["development"]["8"]["endpoint_rmse_mv"]
                for seed, row in values.items()
            }
            for family, values in runs.items()
        }
        passing = {
            family: sum(bool(row["passed"]) for row in values.values())
            for family, values in runs.items()
        }
        median_rmse = {
            family: float(np.median(list(values.values())))
            for family, values in rmse.items()
        }
        authentic = "authentic_morphology_graph_gru"
        relabelled = "relabelled_morphology_graph_control"
        convolution = "ordered_convgru_control"
        paired = {}
        for control in (relabelled, convolution):
            gains = {
                seed: 1.0 - rmse[authentic][seed] / max(rmse[control][seed], 1e-12)
                for seed in rmse[authentic]
            }
            paired[control] = {
                "gain_by_seed": gains,
                "median_gain_fraction": float(np.median(list(gains.values()))),
                "authentic_win_count": sum(value > 0 for value in gains.values()),
            }
        authentic_robust = (
            passing[authentic] >= self.config.minimum_passing_seed_count
        )
        topology_supported = bool(
            authentic_robust
            and all(
                row["median_gain_fraction"]
                >= self.topology_config.minimum_topology_gain_fraction
                and row["authentic_win_count"]
                >= self.topology_config.minimum_paired_topology_win_count
                for row in paired.values()
            )
        )
        if topology_supported:
            diagnosis = "AUTHENTIC_MORPHOLOGY_TOPOLOGY_SIGNAL"
            next_step = "05n_graphgru_mechanism_state_development_canary"
        elif authentic_robust:
            diagnosis = "RECURRENCE_SIGNAL_REPLICATED_TOPOLOGY_STILL_UNRESOLVED"
            next_step = "05n_graph_operator_and_state_contract_reassessment"
        else:
            diagnosis = "EXPANDED_SUPPORT_RECURRENT_SIGNAL_NOT_ROBUST"
            next_step = "05n_recurrent_training_contract_reassessment"
        report = {
            "schema_version": "05m-final-report-v1",
            "valid": True,
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05l": self.artifact_contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "preflight": self.prepare_report,
            "runs": runs,
            "passing_seed_count": passing,
            "median_development_rmse_8ms_mv": median_rmse,
            "paired_topology_comparisons": paired,
            "topology_supported": topology_supported,
            "development_reused_for_development_decision_only": True,
            "validation_or_test_split_loaded": False,
            "fresh_test_loaded": False,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append(
                    {
                        "path": path.relative_to(self.output_dir).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        _write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "05m-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05L_ARCHIVE_SHA256",
    "EXPECTED_05L_FINAL_SHA256",
    "EXPECTED_05L_INDEX_SHA256",
    "TopologyControlledRecurrenceConfig",
    "TopologyControlledRecurrenceExpansion",
    "expanded_train_episode_roles",
    "topology_relabelled_parent_ids",
    "verified_rollout_architecture_artifact_root",
]
