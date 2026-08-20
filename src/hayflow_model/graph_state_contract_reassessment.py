"""One-step information audit for graph operators and initial state contract.

This decision-grade development diagnostic follows 05m.  It does not propose
another recurrent architecture.  Instead, four fixed ridge probes test the
incremental predictive value of authentic axial-neighbour features and a
causal semantic sketch of the complete initial teacher state.  All probes use
the same train/calibration/development episode partition and predict the same
one-millisecond voltage delta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import (
    CompositeFlowmapBundle,
    CompositeTransitionStore,
)
from src.hayflow_data.flowmap_dataset import StateNormalizer

from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .hines_netcon_semantic_repair import HinesNetConSemanticRepairConfig
from .hines_synaptic_domain_repair import (
    BoundedSynapticStateEncoder,
    HinesSynapticDomainRepairConfig,
)
from .rollout_aware_architecture_canary import encode_causal_realized_drive
from .topology_controlled_recurrence_expansion import (
    TopologyControlledRecurrenceConfig,
    expanded_train_episode_roles,
)


EXPECTED_05M_ARCHIVE_SHA256 = (
    "155d7234e6ce27e8bd7eaa4378f6b90eb240d350cc6befa454cf8a5d3b8eccc6"
)
EXPECTED_05M_INDEX_SHA256 = (
    "d7e0d88a9e90fdf97f00041157beba54fd948c2b6d146a46fdda58182aae91ba"
)
EXPECTED_05M_FINAL_SHA256 = (
    "6c100e4fe6983dfb0477afe2938567590cd523b640e9864dbb5940bbaaf5bd98"
)


def verified_topology_expansion_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="topology_controlled_recurrence_config.json",
        archive_sha256=EXPECTED_05M_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05M_INDEX_SHA256,
        final_sha256=EXPECTED_05M_FINAL_SHA256,
    )


@dataclass(frozen=True)
class GraphStateContractReassessmentConfig:
    state_sketch_dim: int = 16
    state_clip: float = 8.0
    ridge_lambdas: Tuple[float, ...] = (
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
        10.0,
    )
    regenerative_delta_threshold_mv: float = 5.0
    regenerative_voltage_threshold_mv: float = -20.0
    calibration_regenerative_weight: float = 0.25
    material_gain_fraction: float = 0.05
    noninferiority_margin_fraction: float = 0.01
    sketch_seed: int = 510071

    def validate(self) -> None:
        if self.state_sketch_dim <= 0 or self.state_sketch_dim > 128:
            raise ValueError("05n state sketch width is invalid")
        if self.state_clip <= 0:
            raise ValueError("05n state clip must be positive")
        if not self.ridge_lambdas or any(value <= 0 for value in self.ridge_lambdas):
            raise ValueError("05n ridge ladder is invalid")
        if tuple(sorted(self.ridge_lambdas)) != self.ridge_lambdas:
            raise ValueError("05n ridge ladder must be sorted")
        if not 0 < self.material_gain_fraction < 1:
            raise ValueError("05n materiality threshold is invalid")
        if not 0 <= self.noninferiority_margin_fraction < 1:
            raise ValueError("05n noninferiority margin is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "GraphStateContractReassessmentConfig":
        payload = dict(values)
        if "ridge_lambdas" in payload:
            payload["ridge_lambdas"] = tuple(map(float, payload["ridge_lambdas"]))
        result = cls(**payload)
        result.validate()
        return result


def _episode_transition_starts(
    store: CompositeTransitionStore,
    roles: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    windows_per_episode: int = 3,
    horizon: int = 8,
) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for role, rows in roles.items():
        selected: List[int] = []
        for row in rows:
            indices = store.trajectory_indices[str(row["trajectory_id"])]
            starts = []
            for start in range(max(0, len(indices) - horizon + 1)):
                candidate = indices[start : start + horizon]
                steps = store.metadata["step_index"][candidate]
                if np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                    starts.append(start)
            if not starts:
                continue
            scored = [
                (
                    sum(
                        len(store.events(int(value)))
                        for value in indices[start : start + horizon]
                    ),
                    start,
                )
                for start in starts
            ]
            chosen = [max(scored, key=lambda value: (value[0], -value[1]))[1]]
            evenly = np.linspace(0, len(starts) - 1, windows_per_episode)
            chosen.extend(starts[int(round(value))] for value in evenly)
            chosen = sorted(set(chosen))[:windows_per_episode]
            selected.extend(int(indices[start]) for start in chosen)
        if not selected:
            raise RuntimeError(f"05n role {role} has no transition starts")
        output[role] = np.asarray(selected, dtype=np.int64)
    return output


def semantic_state_projection(
    layout: Any, *, dimension: int, seed: int
) -> np.ndarray:
    """Fixed signed semantic projection; voltage tokens are excluded."""

    rng = np.random.default_rng(int(seed))
    projection = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(layout.state_width, int(dimension)),
    ) / math.sqrt(float(dimension))
    for index, record in enumerate(layout.core_records):
        if str(record["category"]) == "voltage":
            projection[index] = 0.0
    return projection.astype(np.float32)


def sketch_normalized_state(
    normalized_state: np.ndarray,
    layout: Any,
    projection: np.ndarray,
    *,
    clip: float,
) -> np.ndarray:
    values = np.clip(np.asarray(normalized_state), -clip, clip).astype(np.float32)
    batch = values.shape[0]
    result = np.zeros(
        (batch, layout.segment_count, projection.shape[1]), dtype=np.float32
    )
    segment_ids = np.asarray(layout.core_segment_ids, dtype=np.int64)
    for segment in range(layout.segment_count):
        indices = np.flatnonzero(segment_ids == segment)
        if not len(indices):
            continue
        active = projection[indices]
        count = max(1, int(np.count_nonzero(np.any(active != 0, axis=1))))
        result[:, segment, :] = (
            values[:, indices] @ active / math.sqrt(float(count))
        )
    return result


def axial_voltage_features(voltage: np.ndarray, layout: Any) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=np.float32)
    parent = np.asarray(layout.parent_ids, dtype=np.int64)
    segments = layout.segments
    conductance = np.asarray(
        [max(0.0, float(row["axial_conductance_to_parent_us"])) for row in segments],
        dtype=np.float32,
    )
    positive = conductance[conductance > 0]
    reference = float(np.median(positive)) if len(positive) else 1.0
    scaled = np.clip(conductance / max(reference, 1e-12), 0.0, 20.0)
    parent_delta = voltage[:, parent] - voltage
    child_delta = np.zeros_like(voltage)
    child_axial = np.zeros_like(voltage)
    child_count = np.zeros(layout.segment_count, dtype=np.float32)
    child_weight = np.zeros(layout.segment_count, dtype=np.float32)
    for child, owner in enumerate(parent):
        if child == owner:
            continue
        child_delta[:, owner] += voltage[:, child] - voltage[:, owner]
        child_count[owner] += 1.0
        child_axial[:, owner] += scaled[child] * (
            voltage[:, child] - voltage[:, owner]
        )
        child_weight[owner] += scaled[child]
    child_delta /= np.maximum(child_count, 1.0)[None, :]
    child_axial /= np.maximum(child_weight, 1.0)[None, :]
    return np.stack(
        [
            parent_delta / 100.0,
            child_delta / 100.0,
            scaled[None, :] * parent_delta / 100.0,
            child_axial / 100.0,
        ],
        axis=-1,
    ).astype(np.float32)


def _ridge_sufficient_statistics(
    features: np.ndarray, target: np.ndarray
) -> Dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - center) / scale
    y_center = float(y.mean())
    return {
        "feature_center": center,
        "feature_scale": scale,
        "gram": z.T @ z,
        "rhs": z.T @ (y - y_center),
        "target_center": np.asarray(y_center),
    }


def _ridge_from_statistics(
    statistics: Mapping[str, np.ndarray], ridge: float
) -> Dict[str, np.ndarray]:
    gram = np.asarray(statistics["gram"], dtype=np.float64).copy()
    gram.flat[:: len(gram) + 1] += float(ridge)
    weight = np.linalg.solve(gram, statistics["rhs"])
    return {
        "feature_center": statistics["feature_center"],
        "feature_scale": statistics["feature_scale"],
        "weight": weight,
        "target_center": statistics["target_center"],
    }


def _ridge_fit(
    features: np.ndarray, target: np.ndarray, ridge: float
) -> Dict[str, np.ndarray]:
    return _ridge_from_statistics(
        _ridge_sufficient_statistics(features, target), ridge
    )


def _ridge_predict(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    z = (
        np.asarray(features, dtype=np.float64) - model["feature_center"]
    ) / model["feature_scale"]
    return z @ model["weight"] + float(model["target_center"])


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    regenerative: np.ndarray,
) -> Dict[str, Any]:
    error = np.asarray(prediction) - np.asarray(target)
    regenerative = np.asarray(regenerative, dtype=bool)
    return {
        "rmse_mv": float(np.sqrt(np.mean(error * error))),
        "mean_drift_mv": float(np.mean(error)),
        "regenerative_rmse_mv": float(
            np.sqrt(np.mean(error[regenerative] ** 2))
        ) if regenerative.any() else math.nan,
        "row_count": int(len(error)),
        "regenerative_row_count": int(regenerative.sum()),
        "finite": bool(np.isfinite(prediction).all()),
    }


class GraphStateContractReassessment:
    CONTRACT_FEATURES = {
        "voltage_input_local": ("base",),
        "voltage_input_axial_graph": ("base", "graph"),
        "rich_state_local": ("base", "state"),
        "rich_state_axial_graph": ("base", "graph", "state"),
    }

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: GraphStateContractReassessmentConfig,
        artifact_05m_source: Path,
        *,
        code_revision: str,
    ) -> None:
        config.validate()
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.artifact_05m_source = Path(artifact_05m_source).resolve()
        self.code_revision = str(code_revision)
        self.store = CompositeTransitionStore(bundle)
        self.roles: Dict[str, List[Dict[str, Any]]] = {}
        self.indices: Dict[str, np.ndarray] = {}
        self.artifact_contract: Dict[str, Any] = {}
        self.prepare_report: Dict[str, Any] = {}
        self.normalizer: Any = None
        self.projection: Any = None
        self.netcon_config = HinesNetConSemanticRepairConfig()
        self.domain_config = HinesSynapticDomainRepairConfig()
        self.semantic_encoder = BoundedSynapticStateEncoder(
            self.store.layout, self.netcon_config, self.domain_config
        )

    def close(self) -> None:
        self.store.close()

    def prepare(self) -> Dict[str, Any]:
        _, final, contract = verified_topology_expansion_artifact_root(
            self.artifact_05m_source,
            self.output_dir.parent / ".05n_artifact_cache" / "05m",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05m artifact is not decision-grade")
        if final.get("diagnosis") != "RECURRENCE_SIGNAL_REPLICATED_TOPOLOGY_STILL_UNRESOLVED":
            blockers.append("05m diagnosis does not authorize 05n")
        if final.get("next_step") != "05n_graph_operator_and_state_contract_reassessment":
            blockers.append("05m next step is not 05n")
        if final.get("topology_supported"):
            blockers.append("05m unexpectedly supported authentic topology")
        mapping = self.semantic_encoder.mapping_report
        if int(mapping.get("unmapped_coordinate_count", -1)) != 0:
            blockers.append("05n NetCon semantic mapping is incomplete")
        if (
            len(self.semantic_encoder.recency_indices)
            != self.domain_config.expected_tsyn_coordinate_count
        ):
            blockers.append("05n bounded recency coordinate count changed")
        if (
            len(self.semantic_encoder.trace_indices)
            != self.domain_config.expected_dynamic_trace_coordinate_count
        ):
            blockers.append("05n synaptic trace coordinate count changed")
        self.artifact_contract = contract
        role_config = TopologyControlledRecurrenceConfig()
        self.roles, expansion = expanded_train_episode_roles(
            self.store.episode_rows, config=role_config
        )
        self.indices = _episode_transition_starts(self.store, self.roles)
        role_sets = {name: set(map(int, values)) for name, values in self.indices.items()}
        overlap = {
            f"{left}__{right}": sorted(role_sets[left] & role_sets[right])
            for position, left in enumerate(sorted(role_sets))
            for right in sorted(role_sets)[position + 1 :]
            if role_sets[left] & role_sets[right]
        }
        if overlap:
            blockers.append("05n transition roles overlap")
        report = {
            "schema_version": "05n-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05m": contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "source_split_used": "train_only",
            "validation_or_test_loaded": False,
            "fresh_test_loaded": False,
            "role_episode_counts": {
                name: len(rows) for name, rows in self.roles.items()
            },
            "role_transition_counts": {
                name: len(rows) for name, rows in self.indices.items()
            },
            "transition_overlap": overlap,
            "state_width": int(self.store.layout.state_width),
            "segment_count": int(self.store.layout.segment_count),
            "state_sketch_dim": self.config.state_sketch_dim,
            "netcon_semantic_mapping": {
                name: self.semantic_encoder.mapping_report[name]
                for name in (
                    "mapped_coordinate_count",
                    "unmapped_coordinate_count",
                    "class_coordinate_counts",
                    "semantic_coordinate_counts",
                )
            },
            "bounded_recency_coordinate_count": int(
                len(self.semantic_encoder.recency_indices)
            ),
            "contracts": self.CONTRACT_FEATURES,
            "teacher_future_state_used_as_input": False,
            "support_expansion": expansion,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "graph_state_contract_reassessment_config.json",
            {
                "schema_version": "05n-config-v1",
                "config": asdict(self.config),
                "artifact_05m": contract,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05n preflight failed: {blockers}")
        return report

    def _fit_normalizer(self) -> None:
        raw = self.store.read_state(self.indices["fit"], "t")
        boundary_times = np.asarray(
            self.store.metadata["start_time_ms"][self.indices["fit"]],
            dtype=np.float64,
        )
        semantic = self.semantic_encoder.encode(raw, boundary_times)
        self.normalizer = StateNormalizer(self.store.layout)
        self.semantic_encoder.configure_transform_codes(self.normalizer)
        self.normalizer.fit(semantic, semantic)
        if len(self.semantic_encoder.recency_indices):
            indices = self.semantic_encoder.recency_indices
            self.normalizer.state_scale[indices] = np.maximum(
                self.normalizer.state_scale[indices],
                self.domain_config.bounded_recency_scale_floor,
            )
        if len(self.semantic_encoder.trace_indices):
            indices = self.semantic_encoder.trace_indices
            self.normalizer.state_scale[indices] = np.maximum(
                self.normalizer.state_scale[indices],
                self.domain_config.synaptic_trace_log1p_scale_floor,
            )
        self.projection = semantic_state_projection(
            self.store.layout,
            dimension=self.config.state_sketch_dim,
            seed=self.config.sketch_seed,
        )
        np.savez_compressed(
            self.output_dir / "initial_state_normalizer_and_projection.npz",
            transform_codes=self.normalizer.transform_codes,
            state_center=self.normalizer.state_center,
            state_scale=self.normalizer.state_scale,
            projection=self.projection,
        )

    def _role_arrays(self, role: str) -> Dict[str, np.ndarray]:
        indices = self.indices[role]
        raw = self.store.read_state(indices, "t")
        voltage = raw[:, : self.store.layout.segment_count].astype(np.float32)
        next_voltage = self.store.read_state(
            indices, "t_plus_1", categories=("voltage",)
        ).astype(np.float32)
        drive = encode_causal_realized_drive(self.store, indices)
        static = np.broadcast_to(
            self.store.layout.segment_static[None, :, :],
            (len(indices),) + self.store.layout.segment_static.shape,
        ).astype(np.float32)
        base = np.concatenate(
            [(voltage / 100.0)[..., None], drive, static], axis=-1
        )
        graph = axial_voltage_features(voltage, self.store.layout)
        boundary_times = np.asarray(
            self.store.metadata["start_time_ms"][indices], dtype=np.float64
        )
        semantic = self.semantic_encoder.encode(raw, boundary_times)
        normalized = self.normalizer.normalize_state(semantic)
        state = sketch_normalized_state(
            normalized,
            self.store.layout,
            self.projection,
            clip=self.config.state_clip,
        )
        target = next_voltage - voltage
        regenerative = (
            (np.abs(target) >= self.config.regenerative_delta_threshold_mv)
            | (next_voltage >= self.config.regenerative_voltage_threshold_mv)
        )
        return {
            "base": base.reshape(-1, base.shape[-1]),
            "graph": graph.reshape(-1, graph.shape[-1]),
            "state": state.reshape(-1, state.shape[-1]),
            "target": target.reshape(-1),
            "regenerative": regenerative.reshape(-1),
        }

    @staticmethod
    def _contract_matrix(
        arrays: Mapping[str, np.ndarray], components: Sequence[str]
    ) -> np.ndarray:
        return np.concatenate([arrays[name] for name in components], axis=1)

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        started = time.monotonic()
        self._fit_normalizer()
        arrays = {}
        for position, role in enumerate(("fit", "calibration", "development"), 1):
            arrays[role] = self._role_arrays(role)
            elapsed = time.monotonic() - started
            print(
                f"[HayFlow 05n][feature audit] {position}/3 roles "
                f"elapsed={elapsed / 60:.1f} min role={role}",
                flush=True,
            )
        contracts = {}
        model_dir = self.output_dir / "ridge_models"
        model_dir.mkdir()
        for name, components in self.CONTRACT_FEATURES.items():
            fit_x = self._contract_matrix(arrays["fit"], components)
            calibration_x = self._contract_matrix(arrays["calibration"], components)
            statistics = _ridge_sufficient_statistics(
                fit_x, arrays["fit"]["target"]
            )
            ladder = []
            best = None
            for ridge in self.config.ridge_lambdas:
                model = _ridge_from_statistics(statistics, ridge)
                prediction = _ridge_predict(model, calibration_x)
                metric = _metrics(
                    prediction,
                    arrays["calibration"]["target"],
                    arrays["calibration"]["regenerative"],
                )
                score = metric["rmse_mv"] + self.config.calibration_regenerative_weight * metric["regenerative_rmse_mv"]
                row = {"ridge": ridge, "selection_score": score, **metric}
                ladder.append(row)
                if best is None or (score, ridge) < (best[0], best[1]):
                    best = (score, ridge, model, metric)
            assert best is not None
            development_x = self._contract_matrix(arrays["development"], components)
            development_prediction = _ridge_predict(best[2], development_x)
            development = _metrics(
                development_prediction,
                arrays["development"]["target"],
                arrays["development"]["regenerative"],
            )
            path = model_dir / f"{name}.npz"
            np.savez_compressed(path, **best[2])
            contracts[name] = {
                "feature_components": list(components),
                "feature_width": int(fit_x.shape[1]),
                "selected_ridge": best[1],
                "calibration": best[3],
                "calibration_ladder": ladder,
                "development": development,
                "model": path.relative_to(self.output_dir).as_posix(),
                "model_sha256": sha256_file(path),
            }
            print(
                f"[HayFlow 05n][ridge] {name} width={fit_x.shape[1]} "
                f"dev={development['rmse_mv']:.4g} "
                f"regen={development['regenerative_rmse_mv']:.4g}",
                flush=True,
            )
        def gain(candidate: str, baseline: str, metric: str) -> float:
            candidate_value = contracts[candidate]["development"][metric]
            baseline_value = contracts[baseline]["development"][metric]
            return 1.0 - candidate_value / max(baseline_value, 1e-12)
        gains = {
            "graph_given_voltage": {
                metric: gain("voltage_input_axial_graph", "voltage_input_local", metric)
                for metric in ("rmse_mv", "regenerative_rmse_mv")
            },
            "graph_given_rich_state": {
                metric: gain("rich_state_axial_graph", "rich_state_local", metric)
                for metric in ("rmse_mv", "regenerative_rmse_mv")
            },
            "state_given_local": {
                metric: gain("rich_state_local", "voltage_input_local", metric)
                for metric in ("rmse_mv", "regenerative_rmse_mv")
            },
            "state_given_graph": {
                metric: gain("rich_state_axial_graph", "voltage_input_axial_graph", metric)
                for metric in ("rmse_mv", "regenerative_rmse_mv")
            },
        }
        graph_median = float(np.median([
            gains["graph_given_voltage"]["rmse_mv"],
            gains["graph_given_rich_state"]["rmse_mv"],
        ]))
        graph_regen = float(np.median([
            gains["graph_given_voltage"]["regenerative_rmse_mv"],
            gains["graph_given_rich_state"]["regenerative_rmse_mv"],
        ]))
        state_median = float(np.median([
            gains["state_given_local"]["rmse_mv"],
            gains["state_given_graph"]["rmse_mv"],
        ]))
        state_regen = float(np.median([
            gains["state_given_local"]["regenerative_rmse_mv"],
            gains["state_given_graph"]["regenerative_rmse_mv"],
        ]))
        graph_signal = bool(
            graph_median >= self.config.material_gain_fraction
            and graph_regen >= -self.config.noninferiority_margin_fraction
        )
        state_signal = bool(
            state_median >= self.config.material_gain_fraction
            and state_regen >= -self.config.noninferiority_margin_fraction
        )
        if graph_signal and state_signal:
            diagnosis = "JOINT_AXIAL_OPERATOR_AND_STATE_CONTRACT_SIGNAL"
            next_step = "05o_axial_graphgru_rich_initial_state_canary"
        elif state_signal:
            diagnosis = "INITIAL_STATE_CONTRACT_INFORMATION_BOTTLENECK"
            next_step = "05o_rich_initial_state_recurrent_canary"
        elif graph_signal:
            diagnosis = "AXIAL_GRAPH_OPERATOR_INFORMATION_SIGNAL"
            next_step = "05o_axial_graphgru_voltage_state_canary"
        else:
            diagnosis = "NO_LINEAR_GRAPH_OR_STATE_CONTRACT_SIGNAL"
            next_step = "05o_nonlinear_observability_reassessment"
        report = {
            "schema_version": "05n-final-report-v1",
            "valid": bool(
                all(
                    row["development"]["finite"]
                    and row["development"]["regenerative_row_count"] > 0
                    for row in contracts.values()
                )
            ),
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "artifact_05m": self.artifact_contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "preflight": self.prepare_report,
            "contracts": contracts,
            "incremental_gains": gains,
            "graph_median_gain_fraction": graph_median,
            "graph_regenerative_median_gain_fraction": graph_regen,
            "state_median_gain_fraction": state_median,
            "state_regenerative_median_gain_fraction": state_regen,
            "graph_information_signal": graph_signal,
            "initial_state_information_signal": state_signal,
            "interpretation_scope": "linear_one_step_development_diagnostic_only",
            "teacher_future_state_used_as_input": False,
            "validation_or_test_loaded": False,
            "fresh_test_loaded": False,
            "training_authorized": False,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
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
            "schema_version": "05n-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report


__all__ = [
    "EXPECTED_05M_ARCHIVE_SHA256",
    "EXPECTED_05M_FINAL_SHA256",
    "EXPECTED_05M_INDEX_SHA256",
    "GraphStateContractReassessment",
    "GraphStateContractReassessmentConfig",
    "axial_voltage_features",
    "semantic_state_projection",
    "sketch_normalized_state",
    "verified_topology_expansion_artifact_root",
]
