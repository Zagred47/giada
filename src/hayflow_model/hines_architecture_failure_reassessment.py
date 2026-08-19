"""Decision-only reassessment after the failed state-consistent 05k-c repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root


EXPECTED_05KC_ARCHIVE_SHA256 = (
    "ec6cbdb0a015c8a61720bf7713feb20cbffb34b7142c69165030e7673c06f4b4"
)
EXPECTED_05KC_INDEX_SHA256 = (
    "822bde9a5917d68d056a1c7e85a517248e5bb55d7c0427d137c37761f1de7a1c"
)
EXPECTED_05KC_FINAL_SHA256 = (
    "bfc793e13a6184e56f9081ed161b40c9273ff4a5b28b08cbbe6e4d7fecfa861a"
)


@dataclass(frozen=True)
class HinesArchitectureFailureReassessmentConfig:
    horizon_ms: int = 8
    expected_seed_count: int = 3
    maximum_recommit_improvement_fraction: float = 0.10
    minimum_h2_to_persistence_rmse_ratio: float = 3.0
    maximum_oracle_to_persistence_rmse_ratio: float = 0.75
    proposed_canary_families: Tuple[str, ...] = (
        "morphology_graph_gru",
        "ordered_convgru_control",
    )

    def validate(self) -> None:
        if self.horizon_ms != 8 or self.expected_seed_count != 3:
            raise ValueError("05k-d horizon and frozen seed count are fixed")
        if not 0 < self.maximum_recommit_improvement_fraction < 1:
            raise ValueError("05k-d recommit materiality threshold is invalid")
        if self.minimum_h2_to_persistence_rmse_ratio <= 1:
            raise ValueError("05k-d H2 instability ratio must exceed one")
        if not 0 < self.maximum_oracle_to_persistence_rmse_ratio < 1:
            raise ValueError("05k-d oracle recovery ratio is invalid")
        if self.proposed_canary_families != (
            "morphology_graph_gru",
            "ordered_convgru_control",
        ):
            raise ValueError("05k-d architecture fork is preregistered")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesArchitectureFailureReassessmentConfig":
        payload = dict(values)
        if "proposed_canary_families" in payload:
            payload["proposed_canary_families"] = tuple(
                map(str, payload["proposed_canary_families"])
            )
        result = cls(**payload)
        result.validate()
        return result


def verified_development_repair_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="development_autoregressive_repair_config.json",
        archive_sha256=EXPECTED_05KC_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05KC_INDEX_SHA256,
        final_sha256=EXPECTED_05KC_FINAL_SHA256,
    )


class HinesArchitectureFailureReassessment:
    def __init__(
        self,
        output_dir: Path,
        config: HinesArchitectureFailureReassessmentConfig,
        artifact_05kc_source: Path,
        *,
        code_revision: str,
    ) -> None:
        config.validate()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.artifact_05kc_source = Path(artifact_05kc_source).resolve()
        self.code_revision = str(code_revision)
        self.artifact_root = Path()
        self.artifact_report: Dict[str, Any] = {}
        self.artifact_contract: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        root, final, contract = verified_development_repair_artifact_root(
            self.artifact_05kc_source,
            self.output_dir.parent / ".05k_d_artifact_cache" / "05kc",
        )
        repair = final.get("development_repair", {})
        blockers = []
        if not final.get("valid"):
            blockers.append("05k-c artifact is invalid")
        if final.get("diagnosis") != "STATE_CONSISTENT_RECOMMIT_NOT_SUFFICIENT_ON_DEVELOPMENT":
            blockers.append("05k-c did not reject state-consistent recommit")
        if final.get("fresh_05jo_used"):
            blockers.append("05k-c unexpectedly used fresh 05j-o")
        if final.get("rollout_aware_training_canary_authorized"):
            blockers.append("05k-c unexpectedly authorized rollout training")
        if final.get("next_step") != "05k_d_architecture_reassessment":
            blockers.append("05k-c did not prescribe architecture reassessment")
        seeds = sorted(repair.get("runs", {}))
        horizon = str(self.config.horizon_ms)
        persistence = float(
            repair.get("baselines", {}).get("persistence", {}).get(horizon, {}).get(
                "endpoint_voltage_rmse_mv", np.nan
            )
        )
        h2 = float(
            repair.get("baselines", {}).get("h2", {}).get(horizon, {}).get(
                "endpoint_voltage_rmse_mv", np.nan
            )
        )
        recommit_improvements = []
        oracle_ratios = []
        for seed in seeds:
            modes = repair["runs"][seed]
            recommit_improvements.append(
                float(
                    modes["state_consistent_recommit"][horizon][
                        "error_reduction_vs_standard_fraction"
                    ]
                )
            )
            oracle_ratios.append(
                float(modes["teacher_boundary_reset"][horizon]["endpoint_voltage_rmse_mv"])
                / max(persistence, 1e-12)
            )
        finite = bool(
            np.isfinite(persistence)
            and np.isfinite(h2)
            and np.all(np.isfinite(recommit_improvements))
            and np.all(np.isfinite(oracle_ratios))
        )
        conditions = {
            "all_three_seeds_failed": len(seeds) == self.config.expected_seed_count
            and int(repair.get("passing_seed_count", -1)) == 0,
            "h2_far_worse_than_persistence": h2 / max(persistence, 1e-12)
            >= self.config.minimum_h2_to_persistence_rmse_ratio,
            "recommit_improvement_immaterial": max(recommit_improvements, default=np.inf)
            <= self.config.maximum_recommit_improvement_fraction,
            "teacher_reset_oracle_recovers_boundary_map": max(oracle_ratios, default=np.inf)
            <= self.config.maximum_oracle_to_persistence_rmse_ratio,
            "fresh_05jo_not_used": not bool(final.get("fresh_05jo_used")),
            "numerically_finite": finite,
        }
        retire = bool(not blockers and all(conditions.values()))
        decision = {
            "schema_version": "05k-d-architecture-failure-reassessment-v1",
            "valid": bool(not blockers and finite),
            "blockers": blockers,
            "conditions": conditions,
            "horizon_ms": self.config.horizon_ms,
            "h2_rmse_mv": h2,
            "persistence_rmse_mv": persistence,
            "h2_to_persistence_rmse_ratio": h2 / max(persistence, 1e-12),
            "recommit_improvement_by_seed": dict(zip(seeds, recommit_improvements)),
            "teacher_reset_oracle_to_persistence_ratio_by_seed": dict(
                zip(seeds, oracle_ratios)
            ),
            "retire_free_running_h2_latent_recurrence": retire,
            "rollout_aware_architecture_canary_authorized": retire,
            "proposed_canary_families": list(self.config.proposed_canary_families),
            "fresh_05jo_loaded": False,
            "training_performed": False,
            "model_selection_performed": False,
        }
        _write_json(self.output_dir / "architecture_failure_reassessment.json", decision)
        report = {
            "schema_version": "05k-d-final-report-v1",
            "valid": decision["valid"],
            "decision": "FREE_RUNNING_RECURRENT_ARCHITECTURE_REASSESSMENT",
            "diagnosis": (
                "RETIRE_FREE_RUNNING_H2_LATENT_RECURRENCE"
                if retire
                else "ARCHITECTURE_FAILURE_NOT_YET_ISOLATED"
            ),
            "code_revision": self.code_revision,
            "artifact_05kc": contract,
            "architecture_reassessment": decision,
            "current_candidate_retired": retire,
            "rollout_aware_architecture_canary_authorized": retire,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "fresh_test_generation_authorized": False,
            "next_step": (
                "05l_rollout_aware_graphgru_vs_convgru_canary"
                if retire
                else "05k_e_failure_investigation"
            ),
        }
        _write_json(self.output_dir / "architecture_failure_reassessment_config.json", {
            "schema_version": "05k-d-config-v1",
            "config": asdict(self.config),
            "artifact_05kc": contract,
            "code_revision": self.code_revision,
        })
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
            "schema_version": "05k-d-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        if not decision["valid"]:
            raise RuntimeError(f"05k-d reassessment invalid: {blockers}")
        return report


__all__ = [
    "EXPECTED_05KC_ARCHIVE_SHA256",
    "EXPECTED_05KC_FINAL_SHA256",
    "EXPECTED_05KC_INDEX_SHA256",
    "HinesArchitectureFailureReassessment",
    "HinesArchitectureFailureReassessmentConfig",
    "verified_development_repair_artifact_root",
]
