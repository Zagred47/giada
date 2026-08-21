"""Frozen validation-only autoregressive go/no-go for the 05s candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import CompositeFlowmapBundle

from .graph_state_contract_reassessment import sketch_normalized_state
from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .mechanism_state_encoder_canary import (
    MechanismStateEncoderCanary,
    MechanismStateEncoderCanaryConfig,
    shared_semantic_state_projection,
)
from .rollout_aware_architecture_canary import torch


EXPECTED_05S_ARCHIVE_SHA256 = (
    "cd879a8cf8fe25b42e9b139544424d7674144b983d3de126a34a882c4a1fd090"
)
EXPECTED_05S_INDEX_SHA256 = (
    "eac72ae71568dec45f3e454d6729733b8ebb0afbc38d3b6d447d980b2fb5ccda"
)
EXPECTED_05S_FINAL_SHA256 = (
    "500cdd8a7920d47412b727af915bc44d2392a5f674397aed3f06037a546068bb"
)


def verified_mechanism_encoder_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="mechanism_state_encoder_canary_config.json",
        archive_sha256=EXPECTED_05S_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05S_INDEX_SHA256,
        final_sha256=EXPECTED_05S_FINAL_SHA256,
    )


@dataclass(frozen=True)
class ConsolidatedAutoregressiveGoNoGoConfig(
    MechanismStateEncoderCanaryConfig
):
    validation_horizons_ms: Tuple[int, ...] = (8, 16, 32)
    validation_windows_per_episode: int = 2
    minimum_gain_vs_legacy_fraction: float = 0.05
    minimum_gain_vs_persistence_fraction: float = 0.10
    maximum_absolute_endpoint_drift_mv: float = 5.0
    source_metric_reproduction_atol_mv: float = 1.0e-4

    def validate(self) -> None:
        super().validate()
        if self.validation_horizons_ms != (8, 16, 32):
            raise ValueError("05t validation horizons must remain 8/16/32 ms")
        if self.validation_windows_per_episode != 2:
            raise ValueError("05t uses two preregistered windows per episode")
        if not 0 < self.minimum_gain_vs_legacy_fraction < 1:
            raise ValueError("05t legacy gain gate is invalid")
        if not 0 < self.minimum_gain_vs_persistence_fraction < 1:
            raise ValueError("05t persistence gain gate is invalid")
        if not 0 < self.maximum_absolute_endpoint_drift_mv <= 20:
            raise ValueError("05t drift gate is invalid")
        if not 0 < self.source_metric_reproduction_atol_mv <= 1.0e-3:
            raise ValueError("05t reproduction tolerance is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ConsolidatedAutoregressiveGoNoGoConfig":
        payload = dict(values)
        for name in ("horizons_ms", "seeds", "validation_horizons_ms"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class ConsolidatedAutoregressiveGoNoGo(MechanismStateEncoderCanary):
    EVALUATED_REPRESENTATIONS = (
        "legacy_full_signed",
        "semantic_full",
        "semantic_mechanism_states",
    )
    CANDIDATES = ("semantic_full", "semantic_mechanism_states")

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: ConsolidatedAutoregressiveGoNoGoConfig,
        artifact_05s_source: Path,
        artifact_05r_source: Path,
        artifact_05q_source: Path,
        artifact_05p_source: Path,
        artifact_05o_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05r_source,
            artifact_05q_source,
            artifact_05p_source,
            artifact_05o_source,
            code_revision=code_revision,
        )
        self.artifact_05s_source = Path(artifact_05s_source).resolve()
        self.artifact_05s_root: Path | None = None
        self.artifact_05s_report: Dict[str, Any] = {}
        self.artifact_05s_contract: Dict[str, Any] = {}
        self.validation_windows: List[np.ndarray] = []
        self.validation_materialized: Dict[str, np.ndarray] = {}
        self.validation_states: Dict[str, np.ndarray] = {}

    @property
    def go_no_go_config(self) -> ConsolidatedAutoregressiveGoNoGoConfig:
        return self.config  # type: ignore[return-value]

    def _validation_episode_windows(self, indices: np.ndarray) -> List[np.ndarray]:
        horizon = max(self.go_no_go_config.validation_horizons_ms)
        if len(indices) < horizon:
            return []
        starts = []
        for start in range(len(indices) - horizon + 1):
            candidate = indices[start : start + horizon]
            steps = self.store.metadata["step_index"][candidate]
            if np.array_equal(
                steps, np.arange(steps[0], steps[0] + horizon)
            ):
                starts.append(start)
        if not starts:
            return []
        scored = [
            (
                sum(
                    len(self.store.events(int(value)))
                    for value in indices[start : start + horizon]
                ),
                start,
            )
            for start in starts
        ]
        selected = [max(scored, key=lambda row: (row[0], -row[1]))[1]]
        if self.go_no_go_config.validation_windows_per_episode > 1:
            midpoint = starts[len(starts) // 2]
            selected.append(midpoint)
        selected = sorted(set(selected))
        if len(selected) < self.go_no_go_config.validation_windows_per_episode:
            for start in (starts[0], starts[-1]):
                if start not in selected:
                    selected.append(start)
                if len(selected) == self.go_no_go_config.validation_windows_per_episode:
                    break
        return [
            indices[start : start + horizon]
            for start in selected[: self.go_no_go_config.validation_windows_per_episode]
        ]

    def prepare(self) -> Dict[str, Any]:
        support = super().prepare()
        root, final, contract = verified_mechanism_encoder_artifact_root(
            self.artifact_05s_source,
            self.output_dir.parent / ".05t_artifact_cache" / "05s",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05s is not decision-grade")
        if final.get("diagnosis") != "MECHANISM_STATE_ENCODER_CANARY_PASSED":
            blockers.append("05s did not pass its encoder canary")
        if not final.get("bounded_autoregressive_go_no_go_authorized"):
            blockers.append("05s did not authorize 05t")
        if final.get("next_step") != "05t_consolidated_autoregressive_go_no_go":
            blockers.append("05s next step is not 05t")
        if set(final.get("runs", {})) != set(self.EVALUATED_REPRESENTATIONS):
            blockers.append("05s representation set changed")
        self.artifact_05s_root = root
        self.artifact_05s_report = final
        self.artifact_05s_contract = contract
        validation_rows = [
            row
            for row in self.store.episode_rows
            if str(row.get("split")) == "validation"
        ]
        if not validation_rows:
            blockers.append("05t found no validation episodes")
        for row in validation_rows:
            indices = self.store.trajectory_indices[str(row["trajectory_id"])]
            self.validation_windows.extend(
                self._validation_episode_windows(indices)
            )
        if not self.validation_windows:
            blockers.append("05t found no contiguous validation windows")
        if self.validation_windows:
            self.validation_materialized = self._materialize_role(
                self.validation_windows
            )
            starts = np.asarray(
                [int(window[0]) for window in self.validation_windows],
                dtype=np.int64,
            )
            if set(self.store.metadata["split"][starts].tolist()) != {"validation"}:
                blockers.append("05t validation windows crossed split boundaries")
            normalized = self._normalized_state(starts)
            semantic_full = shared_semantic_state_projection(
                self.store.layout,
                dimension=self.encoder_config.state_sketch_dim,
                seed=self.encoder_config.representation_seed,
                category=None,
            )
            semantic_mechanism = shared_semantic_state_projection(
                self.store.layout,
                dimension=self.encoder_config.state_sketch_dim,
                seed=self.encoder_config.representation_seed,
                category="mechanism_states",
            )
            self.validation_states = {
                "legacy_full_signed": np.asarray(
                    self.validation_materialized["initial_state"],
                    dtype=np.float32,
                ),
                "semantic_full": sketch_normalized_state(
                    normalized,
                    self.store.layout,
                    semantic_full,
                    clip=self.encoder_config.state_clip,
                ),
                "semantic_mechanism_states": sketch_normalized_state(
                    normalized,
                    self.store.layout,
                    semantic_mechanism,
                    clip=self.encoder_config.state_clip,
                ),
            }
        train_trajectories = {
            str(row["trajectory_id"])
            for values in self.roles.values()
            for row in values
        }
        validation_trajectories = {
            str(row["trajectory_id"]) for row in validation_rows
        }
        overlap = sorted(train_trajectories & validation_trajectories)
        if overlap:
            blockers.append("05t train/validation trajectory overlap detected")
        report = {
            "schema_version": "05t-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05s": contract,
            "artifact_05r": support["artifact_05r"],
            "dataset_fingerprint": self.bundle.fingerprint,
            "frozen_checkpoint_only": True,
            "retraining_performed": False,
            "evaluated_representations": list(self.EVALUATED_REPRESENTATIONS),
            "advanced_candidates": list(self.CANDIDATES),
            "validation_episode_count": len(validation_rows),
            "validation_window_count": len(self.validation_windows),
            "validation_horizons_ms": list(
                self.go_no_go_config.validation_horizons_ms
            ),
            "train_validation_trajectory_overlap": overlap,
            "validation_used_for_final_candidate_selection": True,
            "test_splits_loaded": False,
            "sealed_fresh_test_loaded": False,
            "fresh_test_outcomes_generated": False,
            "teacher_future_state_used_as_model_input": False,
            "support_reconstruction": support,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "consolidated_autoregressive_go_no_go_config.json",
            {
                "schema_version": "05t-config-v1",
                "config": asdict(self.go_no_go_config),
                "artifact_05s": contract,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05t preflight failed: {blockers}")
        return report

    def _tensor_validation(self, representation: str, device: Any) -> Dict[str, Any]:
        values = {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in self.validation_materialized.items()
        }
        values["initial_state"] = torch.as_tensor(
            self.validation_states[representation],
            dtype=torch.float32,
            device=device,
        )
        return values

    def _rollout_metrics(
        self, prediction: Any, values: Mapping[str, Any], horizon: int
    ) -> Dict[str, Any]:
        target = values["target_voltage"][:, :horizon]
        endpoint_error = prediction[:, horizon - 1] - target[:, horizon - 1]
        persistence_error = values["initial_voltage"] - target[:, horizon - 1]
        regenerative = (
            torch.abs(target[:, horizon - 1] - values["initial_voltage"]) >= 5.0
        ) | (target[:, horizon - 1] >= -20.0)
        regenerative_error = endpoint_error[regenerative]
        violations = (
            (prediction[:, :horizon] < self.config.physical_voltage_floor_mv)
            | (prediction[:, :horizon] > self.config.physical_voltage_ceiling_mv)
        )
        return {
            "endpoint_rmse_mv": float(
                torch.sqrt(torch.mean(endpoint_error.square())).cpu()
            ),
            "regenerative_endpoint_rmse_mv": float(
                torch.sqrt(torch.mean(regenerative_error.square())).cpu()
            )
            if regenerative_error.numel()
            else math.nan,
            "path_rmse_mv": float(
                torch.sqrt(
                    torch.mean((prediction[:, :horizon] - target).square())
                ).cpu()
            ),
            "endpoint_mean_drift_mv": float(torch.mean(endpoint_error).cpu()),
            "persistence_endpoint_rmse_mv": float(
                torch.sqrt(torch.mean(persistence_error.square())).cpu()
            ),
            "regenerative_coordinate_count": int(regenerative.sum().cpu()),
            "physical_voltage_violation_count": int(violations.sum().cpu()),
            "finite": bool(torch.isfinite(prediction[:, :horizon]).all().cpu()),
        }

    def _paired_comparison(
        self,
        results: Mapping[str, Any],
        candidate: str,
        baseline: str,
        horizon: int,
    ) -> Dict[str, Any]:
        by_seed = {}
        key = str(horizon)
        for seed in map(str, self.config.seeds):
            left = results[candidate][seed][key]
            right = results[baseline][seed][key]
            by_seed[seed] = {
                "rmse_gain_fraction": 1.0
                - left["endpoint_rmse_mv"]
                / max(right["endpoint_rmse_mv"], 1e-12),
                "regenerative_rmse_gain_fraction": 1.0
                - left["regenerative_endpoint_rmse_mv"]
                / max(right["regenerative_endpoint_rmse_mv"], 1e-12),
                "persistence_gain_fraction": 1.0
                - left["endpoint_rmse_mv"]
                / max(left["persistence_endpoint_rmse_mv"], 1e-12),
                "absolute_endpoint_drift_mv": abs(left["endpoint_mean_drift_mv"]),
            }
        return {
            "candidate": candidate,
            "baseline": baseline,
            "horizon_ms": horizon,
            "by_seed": by_seed,
            "median_rmse_gain_fraction": float(np.median([
                row["rmse_gain_fraction"] for row in by_seed.values()
            ])),
            "median_regenerative_gain_fraction": float(np.median([
                row["regenerative_rmse_gain_fraction"]
                for row in by_seed.values()
            ])),
            "median_persistence_gain_fraction": float(np.median([
                row["persistence_gain_fraction"] for row in by_seed.values()
            ])),
            "maximum_absolute_endpoint_drift_mv": max(
                row["absolute_endpoint_drift_mv"] for row in by_seed.values()
            ),
            "positive_win_count": sum(
                row["rmse_gain_fraction"] > 0 for row in by_seed.values()
            ),
        }

    def _candidate_passes(
        self,
        candidate: str,
        comparisons: Mapping[str, Mapping[str, Any]],
        results: Mapping[str, Any],
    ) -> bool:
        for horizon in self.go_no_go_config.validation_horizons_ms:
            row = comparisons[str(horizon)]
            if row["median_rmse_gain_fraction"] < self.go_no_go_config.minimum_gain_vs_legacy_fraction:
                return False
            if row["median_regenerative_gain_fraction"] < -self.config.regenerative_noninferiority_margin_fraction:
                return False
            if row["median_persistence_gain_fraction"] < self.go_no_go_config.minimum_gain_vs_persistence_fraction:
                return False
            if row["maximum_absolute_endpoint_drift_mv"] > self.go_no_go_config.maximum_absolute_endpoint_drift_mv:
                return False
            if row["positive_win_count"] < self.encoder_config.minimum_paired_win_count:
                return False
            for seed in map(str, self.config.seeds):
                metric = results[candidate][seed][str(horizon)]
                if not metric["finite"] or metric["physical_voltage_violation_count"]:
                    return False
        return True

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None or self.artifact_05s_root is None:
            raise RuntimeError("05t requires PyTorch and a verified 05s artifact")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results: Dict[str, Dict[str, Any]] = {
            name: {} for name in self.EVALUATED_REPRESENTATIONS
        }
        reproduction_errors = []
        with torch.no_grad():
            for representation in self.EVALUATED_REPRESENTATIONS:
                validation = self._tensor_validation(representation, device)
                for seed in self.config.seeds:
                    seed_key = str(seed)
                    model = self._model_for_representation(device)
                    source = self.artifact_05s_report["runs"][representation][seed_key]
                    checkpoint = torch.load(
                        self.artifact_05s_root / source["checkpoint"],
                        map_location=device,
                        weights_only=False,
                    )
                    model.load_state_dict(checkpoint["state_dict"])
                    model.eval()
                    development = self._tensor_representation_role(
                        "development", representation, device
                    )
                    reproduced = self._evaluate_arrays(model, development)["8"]
                    stored = source["development"]["8"]
                    reproduction_errors.extend(
                        (
                            abs(
                                reproduced["endpoint_rmse_mv"]
                                - stored["endpoint_rmse_mv"]
                            ),
                            abs(
                                reproduced["regenerative_endpoint_rmse_mv"]
                                - stored["regenerative_endpoint_rmse_mv"]
                            ),
                        )
                    )
                    prediction = model(
                        validation["initial_voltage"],
                        validation["causal_drive"],
                        validation["initial_state"],
                    )["voltage"]
                    results[representation][seed_key] = {
                        str(horizon): self._rollout_metrics(
                            prediction, validation, horizon
                        )
                        for horizon in self.go_no_go_config.validation_horizons_ms
                    }
                    print(
                        f"[HayFlow 05t][frozen validation] {representation} "
                        f"seed={seed} rmse8="
                        f"{results[representation][seed_key]['8']['endpoint_rmse_mv']:.3f} "
                        f"rmse32="
                        f"{results[representation][seed_key]['32']['endpoint_rmse_mv']:.3f}",
                        flush=True,
                    )
        maximum_reproduction_error = max(reproduction_errors, default=math.inf)
        if (
            maximum_reproduction_error
            > self.go_no_go_config.source_metric_reproduction_atol_mv
        ):
            raise RuntimeError("05t checkpoint reproduction disagrees with 05s")
        comparisons = {
            candidate: {
                str(horizon): self._paired_comparison(
                    results,
                    candidate,
                    "legacy_full_signed",
                    horizon,
                )
                for horizon in self.go_no_go_config.validation_horizons_ms
            }
            for candidate in self.CANDIDATES
        }
        candidate_pass = {
            candidate: self._candidate_passes(
                candidate, comparisons[candidate], results
            )
            for candidate in self.CANDIDATES
        }
        passing = [candidate for candidate in self.CANDIDATES if candidate_pass[candidate]]
        if passing:
            selected = max(
                passing,
                key=lambda candidate: min(
                    comparisons[candidate][str(horizon)][
                        "median_rmse_gain_fraction"
                    ]
                    for horizon in self.go_no_go_config.validation_horizons_ms
                ),
            )
            diagnosis = "AUTOREGRESSIVE_REPRESENTATION_GO"
            next_step = "06_preregistered_fresh_autoregressive_evaluation"
        else:
            selected = None
            diagnosis = "AUTOREGRESSIVE_REPRESENTATION_NO_GO"
            next_step = "stop_current_state_encoder_branch"
        all_finite = all(
            metric["finite"]
            for representation in results.values()
            for by_horizon in representation.values()
            for metric in by_horizon.values()
        )
        report = {
            "schema_version": "05t-final-report-v1",
            "valid": all_finite,
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05s": self.artifact_05s_contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "maximum_source_metric_reproduction_error_mv": maximum_reproduction_error,
            "validation_results": results,
            "candidate_comparisons_vs_legacy": comparisons,
            "candidate_pass": candidate_pass,
            "selected_representation": selected,
            "retraining_performed": False,
            "model_selection_source": "validation_only",
            "validation_consumed_for_final_candidate_selection": True,
            "test_splits_loaded": False,
            "sealed_fresh_test_loaded": False,
            "existing_sealed_tests_reusable_for_selected_candidate": False,
            "new_sealed_test_required_if_go": bool(selected),
            "fresh_sealed_test_generation_authorized": bool(selected),
            "full_training_authorized": False,
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
                "schema_version": "05t-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05S_ARCHIVE_SHA256",
    "EXPECTED_05S_FINAL_SHA256",
    "EXPECTED_05S_INDEX_SHA256",
    "ConsolidatedAutoregressiveGoNoGo",
    "ConsolidatedAutoregressiveGoNoGoConfig",
    "verified_mechanism_encoder_artifact_root",
]
