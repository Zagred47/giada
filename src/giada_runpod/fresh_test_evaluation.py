"""Evaluation-only confirmation on sealed fresh S3 teacher trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .hybrid_inputs import PRODUCTION_BACKGROUND_PROTOCOLS, PRODUCTION_TARGET_PROTOCOLS
from .optimization_forensic import MODEL_NAMES, _atomic_json, _metric, _sha256
from .production_corpus import fingerprint_validated_shards
from .training import (
    FeatureTransform,
    LeanSomaCorpus,
    MatchedTrainingConfig,
    PaperScaleMatchedTrainer,
)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class FreshTeacherTestConfig:
    seeds: tuple[int, ...]
    source_absolute_step: int
    source_code_revision: str
    source_configuration_sha256: str
    source_final_report_sha256: str
    source_normalization_sha256: str
    source_state_hashes: Mapping[str, str]
    expected_fresh_stage: str
    expected_transition_count: int
    background_plan_sha256: str
    targeted_plan_sha256: str
    minimum_seed_wins: int
    minimum_family_wins: int
    minimum_protocol_wins: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FreshTeacherTestConfig":
        payload = dict(values)
        payload["seeds"] = tuple(map(int, payload["seeds"]))
        result = cls(**payload)
        result.validate()
        return result

    def validate(self) -> None:
        if self.seeds != (61017, 61029, 61043, 61071, 61103):
            raise ValueError("fresh test must evaluate the five frozen S3 seeds")
        if self.source_absolute_step != 144_000:
            raise ValueError("fresh test must use the matched 144k checkpoints")
        if self.expected_fresh_stage != "s3_fresh_teacher_test":
            raise ValueError("fresh test stage identity changed")
        if self.expected_transition_count != 1_680_000:
            raise ValueError("fresh test size changed after preregistration")
        hashes = {
            "source_configuration": self.source_configuration_sha256,
            "source_final_report": self.source_final_report_sha256,
            "source_normalization": self.source_normalization_sha256,
            "background_plan": self.background_plan_sha256,
            "targeted_plan": self.targeted_plan_sha256,
            **dict(self.source_state_hashes),
        }
        if any(not _valid_sha256(value) for value in hashes.values()):
            raise ValueError("fresh-test identity requires lowercase SHA-256 values")
        expected_states = {
            f"seeds/seed{seed}/state_step72000.pt" for seed in self.seeds
        }
        if set(self.source_state_hashes) != expected_states:
            raise ValueError("fresh-test source checkpoint map is incomplete")
        if (self.minimum_seed_wins, self.minimum_family_wins, self.minimum_protocol_wins) != (4, 5, 12):
            raise ValueError("fresh-test decision gates changed")


class S3FreshTeacherTestEvaluator(PaperScaleMatchedTrainer):
    """Load frozen models and evaluate them once on an independent corpus."""

    def __init__(
        self,
        corpus_root: Path,
        source_root: Path,
        output_dir: Path,
        config: FreshTeacherTestConfig,
        *,
        code_revision: str,
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("fresh test evaluation requires PyTorch") from error
        config.validate()
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("fresh test evaluation requires a CUDA GPU pod")
        self.source_root = Path(source_root)
        self.output_dir = Path(output_dir)
        if (self.output_dir / "final_report.json").is_file():
            raise FileExistsError(f"completed result already exists in {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fresh_config = config
        self.code_revision = str(code_revision)
        self.source_report, self.normalization = self._verify_source()
        self.corpus_contract = self._verify_fresh_corpus(Path(corpus_root))
        source_training = MatchedTrainingConfig.from_mapping(
            self.source_report["configuration"]["base_training"]
        )
        self.config = replace(
            source_training,
            evaluation_sample_limit=config.expected_transition_count,
        )
        self.corpus = LeanSomaCorpus(Path(corpus_root), require_train_split=False)
        if self.corpus.validation_count != config.expected_transition_count:
            raise RuntimeError("fresh test row count differs from the sealed contract")
        self.transform = FeatureTransform(self.corpus, self.config)
        self.transform.load_dict(self.normalization)

    def _verify_source(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        config = self.fresh_config
        report_path = self.source_root / "final_report.json"
        normalization_path = self.source_root / "normalization.json"
        if _sha256(report_path) != config.source_final_report_sha256:
            raise RuntimeError("frozen matched-exposure report SHA-256 mismatch")
        if _sha256(normalization_path) != config.source_normalization_sha256:
            raise RuntimeError("frozen training normalization SHA-256 mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("code_revision") != config.source_code_revision
            or report.get("configuration_sha256") != config.source_configuration_sha256
            or not report.get("valid")
            or not report.get("full_development_validation", {}).get(
                "matched_exposure_stability_passed"
            )
        ):
            raise RuntimeError("frozen matched-exposure source identity is invalid")
        for relative, expected in config.source_state_hashes.items():
            if _sha256(self.source_root / relative) != expected:
                raise RuntimeError(f"frozen source checkpoint mismatch: {relative}")
        return report, json.loads(normalization_path.read_text(encoding="utf-8"))

    def _verify_fresh_corpus(self, root: Path) -> Dict[str, Any]:
        config = self.fresh_config
        manifest_path = root / "composite_manifest.json"
        audit_path = root / "production_audit.json"
        fingerprint_path = root / "corpus_fingerprint.json"
        for path in (manifest_path, audit_path, fingerprint_path):
            if not path.is_file():
                raise RuntimeError(f"fresh corpus contract file missing: {path.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        saved_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if (
            not manifest.get("valid")
            or manifest.get("stage") != config.expected_fresh_stage
            or manifest.get("selection_role") != "sealed_fresh_test"
            or int(manifest.get("total_transition_count", -1))
            != config.expected_transition_count
            or manifest.get("split_transition_counts")
            != {"validation": config.expected_transition_count}
        ):
            raise RuntimeError("fresh composite manifest violates the preregistration")
        if not audit.get("valid") or audit.get("blockers") or audit.get("paper_test_claimed"):
            raise RuntimeError("fresh corpus audit did not pass cleanly")
        components = {
            row["component_id"]: (root / row["root"]).resolve()
            for row in manifest["components"]
        }
        plan_hashes = {
            "background": config.background_plan_sha256,
            "targeted": config.targeted_plan_sha256,
        }
        for component, expected in plan_hashes.items():
            plan_path = components[component] / "plan.json"
            if _sha256(plan_path) != expected:
                raise RuntimeError(f"fresh {component} plan changed after sealing")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if any(
                trajectory.get("split") != "validation"
                for shard in plan["shards"]
                for trajectory in shard["trajectories"]
            ):
                raise RuntimeError("fresh test contains a non-test trajectory")
        observed_fingerprint = fingerprint_validated_shards(
            root, progress=self._report_corpus_fingerprint_progress
        )
        if not observed_fingerprint["valid"] or observed_fingerprint != saved_fingerprint:
            raise RuntimeError("fresh physical corpus fingerprint mismatch")
        return {
            "verified": True,
            "selection_role": "sealed_fresh_test",
            "manifest_sha256": _sha256(manifest_path),
            "audit_sha256": _sha256(audit_path),
            "fingerprint_sha256": _sha256(fingerprint_path),
            "physical_fingerprint": observed_fingerprint,
            "plan_sha256": plan_hashes,
        }

    def _load_models(self, seed: int) -> Dict[str, Any]:
        path = self.source_root / f"seeds/seed{seed}/state_step72000.pt"
        state = self.torch.load(path, map_location=self.device, weights_only=False)
        if int(state.get("step", -1)) != 72_000 or int(state.get("seed", -1)) != seed:
            raise RuntimeError(f"frozen checkpoint metadata mismatch for seed {seed}")
        models = self._models()
        for name in MODEL_NAMES:
            models[name].load_state_dict(state["models"][name])
        return models

    def _evaluate_seed(self, seed: int) -> Dict[str, Any]:
        path = self.output_dir / "seeds" / f"seed{seed}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("source_state_sha256") != self.fresh_config.source_state_hashes[
                f"seeds/seed{seed}/state_step72000.pt"
            ]:
                raise RuntimeError(f"resumed fresh evaluation changed contract: seed {seed}")
            return row
        models = self._load_models(seed)
        row = {
            "schema_version": "giada-s3-fresh-teacher-test-seed-v1",
            "seed": seed,
            "source_absolute_step": self.fresh_config.source_absolute_step,
            "source_state_sha256": self.fresh_config.source_state_hashes[
                f"seeds/seed{seed}/state_step72000.pt"
            ],
            "readout": "raw",
            "metrics": {name: self._evaluate(models[name]) for name in MODEL_NAMES},
        }
        _atomic_json(path, row)
        print(f"[GIADA RunPod][S3 fresh test] seed {seed}/5 complete", flush=True)
        return row

    @staticmethod
    def summarize(rows: Sequence[Mapping[str, Any]], config: FreshTeacherTestConfig) -> Dict[str, Any]:
        by_seed = {int(row["seed"]): row["metrics"] for row in rows}
        if set(by_seed) != set(config.seeds):
            raise ValueError("fresh-test seed results are incomplete")

        def medians_for(getter) -> Dict[str, float | None]:
            result: Dict[str, float | None] = {}
            for model in MODEL_NAMES:
                values = [getter(by_seed[seed][model]) for seed in config.seeds]
                finite = [float(value) for value in values if value is not None]
                result[model] = float(np.median(finite)) if finite else None
            return result

        medians = {
            stratum: medians_for(lambda metrics, s=stratum: _metric(metrics, s))
            for stratum in ("overall", "active", "spike")
        }
        seed_wins = {
            str(seed): float(_metric(by_seed[seed]["giada_voltage_bridge"], "overall"))
            < float(_metric(by_seed[seed]["branch_elm_core"], "overall"))
            for seed in config.seeds
        }

        def stratified(section: str, labels: Sequence[str]) -> tuple[Dict[str, Dict[str, float | None]], Dict[str, bool]]:
            values = {
                label: medians_for(
                    lambda metrics, l=label: metrics.get(section, {})
                    .get(l, {})
                    .get("soma_rmse_mv")
                )
                for label in labels
            }
            wins = {
                label: bool(
                    pair["branch_elm_core"] is not None
                    and pair["giada_voltage_bridge"] is not None
                    and pair["giada_voltage_bridge"] < pair["branch_elm_core"]
                )
                for label, pair in values.items()
            }
            return values, wins

        families = ("neuronio_background", "somatic_repair", "nmda_boundary", "calcium_boundary", "bap_repair_matrix")
        protocols = (*PRODUCTION_BACKGROUND_PROTOCOLS, *PRODUCTION_TARGET_PROTOCOLS)
        family_medians, family_wins = stratified("protocol_family_metrics", families)
        protocol_medians, protocol_wins = stratified("protocol_metrics", protocols)
        stratum_wins = {
            name: bool(
                pair["branch_elm_core"] is not None
                and pair["giada_voltage_bridge"] is not None
                and pair["giada_voltage_bridge"] < pair["branch_elm_core"]
            )
            for name, pair in medians.items()
        }
        decision = {
            "overall_median_passed": stratum_wins["overall"],
            "active_median_passed": stratum_wins["active"],
            "spike_median_passed": stratum_wins["spike"],
            "minimum_seed_wins_required": config.minimum_seed_wins,
            "observed_seed_wins": int(sum(seed_wins.values())),
            "minimum_family_wins_required": config.minimum_family_wins,
            "observed_family_wins": int(sum(family_wins.values())),
            "minimum_protocol_wins_required": config.minimum_protocol_wins,
            "observed_protocol_wins": int(sum(protocol_wins.values())),
        }
        decision["all_preregistered_gates_passed"] = bool(
            all(stratum_wins.values())
            and decision["observed_seed_wins"] >= config.minimum_seed_wins
            and decision["observed_family_wins"] >= config.minimum_family_wins
            and decision["observed_protocol_wins"] >= config.minimum_protocol_wins
        )
        decision["s4_authorized"] = decision["all_preregistered_gates_passed"]
        branch_by_seed = np.asarray(
            [
                _metric(by_seed[seed]["branch_elm_core"], "overall")
                for seed in config.seeds
            ],
            dtype=np.float64,
        )
        giada_by_seed = np.asarray(
            [
                _metric(by_seed[seed]["giada_voltage_bridge"], "overall")
                for seed in config.seeds
            ],
            dtype=np.float64,
        )
        paired_gain = branch_by_seed - giada_by_seed
        bootstrap_rng = np.random.default_rng(20_260_913)
        bootstrap_indices = bootstrap_rng.integers(
            0, len(paired_gain), size=(20_000, len(paired_gain))
        )
        bootstrap_means = np.mean(paired_gain[bootstrap_indices], axis=1)
        try:
            from scipy.stats import ttest_rel

            paired_t = ttest_rel(branch_by_seed, giada_by_seed, alternative="greater")
            statistic = float(paired_t.statistic)
            p_value = float(paired_t.pvalue)
            if not math.isfinite(statistic):
                statistic = None
            if not math.isfinite(p_value):
                p_value = None
        except (ImportError, TypeError):  # pragma: no cover
            statistic = None
            p_value = None

        def dispersion(model: str, stratum: str) -> Dict[str, Any]:
            values = np.asarray(
                [_metric(by_seed[seed][model], stratum) for seed in config.seeds],
                dtype=np.float64,
            )
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sample_standard_deviation": float(np.std(values, ddof=1)),
                "seed_count": len(values),
            }

        return {
            "median_rmse_mv": medians,
            "seed_wins": seed_wins,
            "family_median_rmse_mv": family_medians,
            "family_wins": family_wins,
            "protocol_median_rmse_mv": protocol_medians,
            "protocol_wins": protocol_wins,
            "relative_overall_reduction_vs_branch_elm": 1.0
            - float(medians["overall"]["giada_voltage_bridge"])
            / float(medians["overall"]["branch_elm_core"]),
            "seed_dispersion": {
                stratum: {
                    model: dispersion(model, stratum) for model in MODEL_NAMES
                }
                for stratum in ("overall", "active", "spike")
            },
            "paired_statistical_inference": {
                "unit": "independent_training_seed_on_the_same_sealed_fresh_teacher_test_corpus",
                "paired_difference": "branch_elm_rmse_minus_giada_rmse_mv",
                "mean_difference_mv": float(np.mean(paired_gain)),
                "sample_standard_deviation_mv": float(np.std(paired_gain, ddof=1)),
                "bootstrap_95_percent_ci_mean_difference_mv": [
                    float(value)
                    for value in np.quantile(bootstrap_means, [0.025, 0.975])
                ],
                "paired_t_test_one_sided_giada_lower": {
                    "statistic": statistic,
                    "p_value": p_value,
                },
                "interpretation_limit": (
                    "This quantifies optimization-seed variability on one common "
                    "independent teacher-test corpus; it is not five independent "
                    "biological datasets."
                ),
            },
            "decision": decision,
        }

    def run(self) -> Dict[str, Any]:
        rows = [self._evaluate_seed(seed) for seed in self.fresh_config.seeds]
        summary = self.summarize(rows, self.fresh_config)
        report = {
            "schema_version": "giada-s3-fresh-teacher-test-v1",
            "valid": True,
            "code_revision": self.code_revision,
            "evaluation_only": True,
            "training_performed": False,
            "checkpoint_or_hyperparameter_selection_performed": False,
            "fresh_trajectories_used_for_training_or_tuning": False,
            "same_frozen_models_and_normalization": True,
            "preregistered_configuration": asdict(self.fresh_config),
            "source_identity": {
                "code_revision": self.fresh_config.source_code_revision,
                "configuration_sha256": self.fresh_config.source_configuration_sha256,
                "final_report_sha256": self.fresh_config.source_final_report_sha256,
                "normalization_sha256": self.fresh_config.source_normalization_sha256,
                "absolute_step": self.fresh_config.source_absolute_step,
            },
            "corpus_contract": self.corpus_contract,
            "fresh_test_transition_count": self.corpus.validation_count,
            "seed_rows": rows,
            **summary,
            "interpretation": {
                "fresh_independent_teacher_trajectory_confirmation": summary["decision"]["all_preregistered_gates_passed"],
                "paper_confirmation_claimed": summary["decision"]["all_preregistered_gates_passed"],
                "s4_authorized": summary["decision"]["s4_authorized"],
                "limit": "Training seeds share one sealed independent teacher-test corpus; rows are not treated as independent biological experiments.",
            },
        }
        _atomic_json(self.output_dir / "final_report.json", report)
        return report
