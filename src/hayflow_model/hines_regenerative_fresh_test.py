"""Frozen 05j-n decoder evaluation on the preregistered 05j-o fresh test."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from ..hayflow_data.composite_flowmap import CompositeShard, CompositeTransitionStore
from ..hayflow_data.flowmap_dataset import FlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_regenerative_confirmation import _IndependentBundle, _verified_artifact_root
from .hines_regenerative_decoder_refit import HinesRegenerativeDecoderRefit
from .hines_trainable_topology_canary import TrainableTopologyResidualHead

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05JN_ARCHIVE_SHA256 = (
    "1d7e5aab979be1c7bf5dcbc5016861a5186769b4ad98a18edd556366419d89f4"
)
EXPECTED_05JN_INDEX_SHA256 = (
    "c3c4199b0a6da73b8f0ec3b986e22a7f2a9999cc74947039ac72811583c8e610"
)
EXPECTED_05JN_FINAL_SHA256 = (
    "4575983a8a81b2ee6b0160bcde06738a00ec689bb6c8c3f8d4957333b563c5cd"
)
EXPECTED_FRESH_PROTOCOL_PLAN_SHA256 = (
    "31b42befaa2a3cf6b76e304a45daef133371f87f2c069a76ade030c4a68ec642"
)


@dataclass(frozen=True)
class HinesRegenerativeFreshTestConfig:
    seeds: Tuple[int, ...] = (17, 29, 43)
    minimum_improvement_vs_best_baseline_fraction: float = 0.05
    minimum_branching_retention: float = 0.5
    maximum_branching_retention: float = 2.0
    maximum_segment_error_ratio_vs_h2: float = 1.0
    minimum_passing_seeds: int = 2
    transform_reproduction_atol: float = 1e-6

    def validate(self) -> None:
        if tuple(self.seeds) != (17, 29, 43):
            raise ValueError("05j-o evaluates the three frozen registered seeds")
        if not 0 < self.minimum_improvement_vs_best_baseline_fraction < 1:
            raise ValueError("fresh-test improvement threshold is invalid")
        if not 0 < self.minimum_branching_retention < self.maximum_branching_retention:
            raise ValueError("fresh-test branching interval is invalid")
        if self.maximum_segment_error_ratio_vs_h2 <= 0:
            raise ValueError("fresh-test maximum-error ratio must be positive")
        if not 1 <= self.minimum_passing_seeds <= len(self.seeds):
            raise ValueError("fresh-test seed gate is invalid")
        if self.transform_reproduction_atol <= 0:
            raise ValueError("transform tolerance must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRegenerativeFreshTestConfig":
        payload = dict(values)
        if "seeds" in payload:
            payload["seeds"] = tuple(map(int, payload["seeds"]))
        result = cls(**payload)
        result.validate()
        return result


class _FreshTestTransitionStore(CompositeTransitionStore):
    def _validate_contract(self) -> None:
        splits = sorted({str(row["split"]) for row in self.episode_rows})
        trajectories = [str(row["trajectory_id"]) for row in self.episode_rows]
        blockers = []
        if self.count != 768:
            blockers.append("fresh-test transition count is not 768")
        if len(self.episode_rows) != 64:
            blockers.append("fresh-test episode count is not 64")
        if splits != ["test"]:
            blockers.append("fresh-test shard is not test-only")
        if len(set(trajectories)) != len(trajectories):
            blockers.append("fresh-test trajectory ids are not unique")
        self.contract_checks = {
            "transition_count_exact": self.count == 768,
            "episode_count_exact": len(self.episode_rows) == 64,
            "test_only": splits == ["test"],
            "trajectory_ids_unique": len(set(trajectories)) == len(trajectories),
            "blockers": blockers,
        }
        if blockers:
            raise RuntimeError(f"05j-o fresh-test store contract failed: {blockers}")

    def _shard_for(self, logical_index: int) -> CompositeShard:
        if not 0 <= int(logical_index) < self.count:
            raise IndexError(logical_index)
        return self.bundle.shard

    def report(self) -> Dict[str, Any]:
        return {
            "valid": not self.contract_checks["blockers"],
            "dataset_kind": "preregistered_regenerative_fresh_test",
            "fingerprint": self.bundle.fingerprint,
            "episode_count": len(self.episode_rows),
            "transition_count": self.count,
            "contract_checks": self.contract_checks,
            "state_loading": "lazy_single_shard",
        }


class HinesRegenerativeFreshTestEvaluation(HinesRegenerativeDecoderRefit):
    """Evaluate frozen checkpoints once; do not select, tune, or retrain."""

    def __init__(
        self,
        *args: Any,
        fresh_test_config: HinesRegenerativeFreshTestConfig,
        artifact_05jn_source: Path,
        fresh_dataset_root: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        fresh_test_config.validate()
        self.fresh_config = fresh_test_config
        self.artifact_05jn_source = Path(artifact_05jn_source).resolve()
        self.fresh_dataset_root = Path(fresh_dataset_root).resolve()
        self.artifact_05jn_root = Path()
        self.artifact_05jn_report: Dict[str, Any] = {}
        self.artifact_05jn_contract: Dict[str, Any] = {}
        self.fresh_store: _FreshTestTransitionStore | None = None
        self.fresh_pair_rows: List[Dict[str, Any]] = []

    def prepare_fresh_test_evaluation(self) -> Dict[str, Any]:
        base = self.prepare_regenerative_decoder_refit()
        root_n, report_n, contract_n = _verified_artifact_root(
            self.artifact_05jn_source,
            self.output_dir.parent / ".05j_o_artifact_cache" / "05jn",
            marker_name="regenerative_decoder_refit_config.json",
            archive_sha256=EXPECTED_05JN_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JN_INDEX_SHA256,
            final_sha256=EXPECTED_05JN_FINAL_SHA256,
        )
        teacher_report = json.loads(
            (self.fresh_dataset_root / "teacher_fresh_test_report.json").read_text()
        )
        manifest = json.loads(
            (self.fresh_dataset_root / "dataset_manifest.json").read_text()
        )
        state_schema = json.loads(
            (self.fresh_dataset_root / "state_schema.json").read_text()
        )
        teacher_manifest = json.loads(
            (self.fresh_dataset_root / "manifest.json").read_text()
        )
        transition = self.fresh_dataset_root / "transition_dataset.h5"
        blockers = []
        if not report_n.get("valid") or not report_n.get("fresh_test_generation_authorized"):
            blockers.append("05j-n does not authorize fresh-test evaluation")
        if report_n.get("diagnosis") != "REFIT_PASSES_DEVELOPMENT_FRESH_TEST_GENERATION_AUTHORIZED":
            blockers.append("05j-n diagnosis is incompatible with 05j-o")
        if not teacher_report.get("valid"):
            blockers.append("fresh-test teacher shard is invalid")
        if teacher_report.get("protocol_plan_sha256") != EXPECTED_FRESH_PROTOCOL_PLAN_SHA256:
            blockers.append("fresh-test teacher plan SHA-256 mismatch")
        if not transition.is_file() or sha256_file(transition) != teacher_report.get(
            "transition_store_sha256"
        ):
            blockers.append("fresh-test transition store SHA-256 mismatch")
        if json.dumps(state_schema, sort_keys=True) != json.dumps(
            self.bundle.layout_bundle.state_schema, sort_keys=True
        ):
            blockers.append("fresh-test state schema differs from model schema")
        if blockers:
            raise RuntimeError(f"05j-o model-evaluation blockers: {blockers}")
        layout_bundle = FlowmapBundle(
            root=self.fresh_dataset_root,
            transition_path=transition,
            manifest=manifest,
            state_schema=state_schema,
            teacher_manifest=teacher_manifest,
            validation_report=teacher_report,
            artifact_validation={"valid": True, "fresh_test": True},
        )
        shard = CompositeShard(
            shard_id="05jo_fresh_test",
            root=self.fresh_dataset_root,
            transition_path=transition,
            transition_count=768,
            transition_sha256=str(teacher_report["transition_store_sha256"]),
            dataset_manifest=manifest,
            validation_report=teacher_report,
            offset=0,
        )
        independent = _IndependentBundle(
            self.fresh_dataset_root / "dataset_manifest.json",
            manifest,
            shard,
            layout_bundle,
            str(teacher_report["transition_store_sha256"]),
        )
        self.fresh_store = _FreshTestTransitionStore(independent)
        self.fresh_pair_rows = pd.read_parquet(
            self.fresh_dataset_root / "fresh_test_pairs.parquet"
        ).to_dict("records")
        if len(self.fresh_pair_rows) != 32:
            raise RuntimeError("fresh-test pair table does not contain 32 pairs")
        self.artifact_05jn_root = root_n
        self.artifact_05jn_report = report_n
        self.artifact_05jn_contract = contract_n
        payload = {
            "schema_version": "05j-o-frozen-model-evaluation-config-v1",
            "frozen_model_gate": asdict(self.fresh_config),
            "artifact_05jn": contract_n,
            "fresh_store": self.fresh_store.report(),
            "teacher_fresh_test_report": teacher_report,
            "checkpoint_selection_performed": False,
            "retraining_performed": False,
            "fresh_test_used_for_model_selection": False,
            "rollout_performed": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "frozen_model_evaluation_config.json", payload)
        return {**base, **payload}

    def _fresh_logical_indices(self) -> np.ndarray:
        if self.fresh_store is None:
            raise RuntimeError("prepare_fresh_test_evaluation() must run first")
        lookup = {
            int(value): index
            for index, value in enumerate(
                self.fresh_store.metadata["transition_id"].tolist()
            )
        }
        result = []
        for row in self.fresh_pair_rows:
            result.extend(
                [
                    lookup[int(row["low_transition_id"])],
                    lookup[int(row["high_transition_id"])],
                ]
            )
        return np.asarray(result, dtype=np.int64)

    def _load_registered_transform(self) -> Tuple[Dict[str, Any], float]:
        path = self.artifact_05jn_root / "refit_feature_transform.npz"
        with np.load(path) as archive:
            loaded = {
                "surfaces": {
                    "h2_raw": {
                        "mean": archive["h2_mean"],
                        "scale": archive["h2_scale"],
                        "channel_mean": archive["h2_channel_mean"],
                        "components": archive["h2_components"],
                    },
                    "causal_raw": {
                        "mean": archive["causal_mean"],
                        "scale": archive["causal_scale"],
                        "channel_mean": archive["causal_channel_mean"],
                        "components": archive["causal_components"],
                    },
                },
                "raw_mean": archive["raw_mean"],
                "raw_scale": archive["raw_scale"],
            }
        errors = []
        for surface in ("h2_raw", "causal_raw"):
            for name in ("mean", "scale", "channel_mean", "components"):
                errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(loaded["surfaces"][surface][name])
                                - np.asarray(self.refit_transform["surfaces"][surface][name])
                            )
                        )
                    )
                )
        for name in ("raw_mean", "raw_scale"):
            errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(loaded[name])
                            - np.asarray(self.refit_transform[name])
                        )
                    )
                )
            )
        return loaded, max(errors, default=0.0)

    def evaluate_frozen_checkpoints(self) -> Dict[str, Any]:
        require_torch()
        if self.fresh_store is None or not self.refit_transform:
            raise RuntimeError("05j-o requires reconstructed 05j-n roles and transform")
        registered_transform, transform_error = self._load_registered_transform()
        if transform_error > self.fresh_config.transform_reproduction_atol:
            raise RuntimeError(
                f"05j-o registered transform reproduction mismatch: {transform_error}"
            )
        indices = self._fresh_logical_indices()
        original_store = self.store
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2, _ = self._load_h2_checkpoint(device)
        h2.eval()
        self.store = self.fresh_store
        try:
            fresh_role = self._extract_recheck_role(h2, indices)
        finally:
            self.store = original_store
            del h2
        design = self._normalize_raw_topology(
            self._raw_topology_design(fresh_role, registered_transform),
            registered_transform,
        )
        target = np.asarray(fresh_role["target"]).reshape(
            -1, 2, self.layout.segment_count
        )
        baselines = {
            "h2": self._pair_set_metrics(
                np.asarray(fresh_role["base"]).reshape(
                    -1, 2, self.layout.segment_count
                ),
                target,
            ),
            "persistence": self._pair_set_metrics(
                np.asarray(fresh_role["voltage_t"]).reshape(
                    -1, 2, self.layout.segment_count
                ),
                target,
            ),
        }
        best_baseline_rmse = min(
            baselines["h2"]["aggregate_voltage_rmse_mv"],
            baselines["persistence"]["aggregate_voltage_rmse_mv"],
        )
        rows_by_seed = {
            int(row["seed"]): row
            for row in self.artifact_05jn_report["decoder_refit"]["runs"]
        }
        predictions = []
        runs = []
        progress = Progress("05j-o frozen fresh-test inference", len(self.fresh_config.seeds))
        features = torch.as_tensor(design, dtype=torch.float32, device=device)
        for position, seed in enumerate(self.fresh_config.seeds, start=1):
            registered = rows_by_seed[int(seed)]
            checkpoint = torch.load(
                self.artifact_05jn_root / str(registered["checkpoint"]),
                map_location=device,
                weights_only=False,
            )
            if int(checkpoint["seed"]) != int(seed) or checkpoint["family"] != "direct_tree_refit":
                raise RuntimeError(f"05j-o checkpoint identity mismatch for seed {seed}")
            model = TrainableTopologyResidualHead(
                design.shape[-1],
                self.layout.segment_count,
                self.topology.hidden_width,
                self.topology.segment_embedding_dim,
                self.topology.target_residual_limit_mv,
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                residual = model(features).cpu().numpy()
            prediction = np.asarray(fresh_role["base"]) + residual
            metrics = self._pair_set_metrics(
                prediction.reshape(-1, 2, self.layout.segment_count), target
            )
            improvement = 1.0 - metrics["aggregate_voltage_rmse_mv"] / max(
                best_baseline_rmse, 1e-12
            )
            retention = metrics["median_branching_retention"]
            finite = all(
                math.isfinite(float(value))
                for value in (
                    improvement,
                    retention,
                    metrics["maximum_segment_error_mv"],
                )
            )
            passed = bool(
                finite
                and improvement
                >= self.fresh_config.minimum_improvement_vs_best_baseline_fraction
                and self.fresh_config.minimum_branching_retention
                <= retention
                <= self.fresh_config.maximum_branching_retention
                and metrics["maximum_segment_error_mv"]
                <= self.fresh_config.maximum_segment_error_ratio_vs_h2
                * baselines["h2"]["maximum_segment_error_mv"]
            )
            predictions.append(prediction)
            runs.append(
                {
                    "family": "direct_tree_refit",
                    "seed": int(seed),
                    "checkpoint": str(registered["checkpoint"]),
                    "metrics": metrics,
                    "improvement_vs_best_baseline_fraction": improvement,
                    "gate_metrics_finite": finite,
                    "run_passed": passed,
                }
            )
            progress.update(position, f"seed={seed} pass={passed}")
            del model
        ensemble_prediction = np.mean(np.stack(predictions), axis=0)
        ensemble_metrics = self._pair_set_metrics(
            ensemble_prediction.reshape(-1, 2, self.layout.segment_count), target
        )
        passing = sum(row["run_passed"] for row in runs)
        valid = all(row["gate_metrics_finite"] for row in runs)
        report = {
            "schema_version": "05j-o-frozen-fresh-test-evaluation-v1",
            "valid": valid,
            "device": str(device),
            "protocol_plan_sha256": EXPECTED_FRESH_PROTOCOL_PLAN_SHA256,
            "pair_count": len(indices) // 2,
            "all_pairs_evaluated": len(indices) == 64,
            "baselines": baselines,
            "runs": runs,
            "ensemble_mean_metrics": ensemble_metrics,
            "passing_seed_count": passing,
            "minimum_passing_seed_count": self.fresh_config.minimum_passing_seeds,
            "robust_fresh_test_gate_passed": passing
            >= self.fresh_config.minimum_passing_seeds,
            "maximum_transform_reproduction_error": transform_error,
            "checkpoint_selection_performed": False,
            "retraining_performed": False,
            "fresh_test_used_for_model_selection": False,
            "architecture_search_performed": False,
            "rollout_performed": False,
        }
        _write_json(self.output_dir / "frozen_fresh_test_evaluation.json", report)
        write_parquet(
            self.output_dir / "frozen_fresh_test_run_summary.parquet",
            [
                {
                    "seed": row["seed"],
                    "rmse_mv": row["metrics"]["aggregate_voltage_rmse_mv"],
                    "maximum_segment_error_mv": row["metrics"]["maximum_segment_error_mv"],
                    "median_branching_retention": row["metrics"]["median_branching_retention"],
                    "improvement_vs_best_baseline_fraction": row[
                        "improvement_vs_best_baseline_fraction"
                    ],
                    "run_passed": row["run_passed"],
                }
                for row in runs
            ],
        )
        if not valid:
            raise RuntimeError("05j-o produced non-finite fresh-test metrics")
        return report

    def finalize_fresh_test(
        self,
        teacher_report: Mapping[str, Any],
        evaluation_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        passed = bool(
            teacher_report["valid"]
            and evaluation_report["valid"]
            and evaluation_report["robust_fresh_test_gate_passed"]
        )
        report = {
            "schema_version": "05j-o-final-report-v1",
            "valid": bool(teacher_report["valid"] and evaluation_report["valid"]),
            "decision": "PREREGISTERED_REGENERATIVE_FRESH_TEST",
            "diagnosis": (
                "FRESH_TEST_CONFIRMS_REFIT_ONE_STEP_CANDIDATE"
                if passed
                else "FRESH_TEST_REJECTS_REFIT_CANDIDATE"
            ),
            "code_revision": self.code_revision,
            "artifact_05jn": self.artifact_05jn_contract,
            "teacher_fresh_test": dict(teacher_report),
            "frozen_model_evaluation": dict(evaluation_report),
            "candidate_model_authorized": passed,
            "candidate_authorization_scope": "one_step_only" if passed else None,
            "micro_rollout_authorized": passed,
            "full_training_authorized": False,
            "methodology": {
                "all_32_preregistered_pairs_generated": True,
                "all_32_preregistered_pairs_evaluated": True,
                "checkpoints_frozen_before_outcomes": True,
                "checkpoint_selection_performed": False,
                "retraining_performed": False,
                "fresh_test_used_for_model_selection": False,
                "rollout_performed": False,
            },
            "next_step": (
                "05k_frozen_candidate_micro_rollout"
                if passed
                else "05j_p_fresh_test_failure_reassessment"
            ),
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
                "schema_version": "05j-o-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05JN_ARCHIVE_SHA256",
    "EXPECTED_05JN_INDEX_SHA256",
    "EXPECTED_05JN_FINAL_SHA256",
    "EXPECTED_FRESH_PROTOCOL_PLAN_SHA256",
    "HinesRegenerativeFreshTestConfig",
    "HinesRegenerativeFreshTestEvaluation",
]
