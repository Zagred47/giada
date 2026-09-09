"""Exact 84k-to-144k continuation of the selected S3 forensic arm."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .optimization_forensic import MODEL_NAMES, _atomic_json, _metric, _sha256
from .training import MatchedTrainingConfig, PaperScaleMatchedTrainer


@dataclass(frozen=True)
class MatchedExposureExtensionConfig:
    seeds: tuple[int, ...] = (61017, 61029, 61043, 61071, 61103)
    source_continuation_step: int = 12_000
    target_continuation_step: int = 72_000
    checkpoints: tuple[int, ...] = (24_000, 36_000, 48_000, 60_000, 72_000)
    learning_rate: float = 3e-4
    readout: str = "raw"
    ema_decay: float = 0.999
    diagnostic_evaluation_samples: int = 262_144
    diagnostic_evaluation_seed: int = 73_100_001
    progress_interval: int = 500
    source_code_revision: str = ""
    source_configuration_sha256: str = ""
    source_final_report_sha256: str = ""
    source_selection_sha256: str = ""
    source_sample_manifest_sha256: str = ""
    source_normalization_sha256: str = ""
    source_state_hashes: Mapping[str, str] = field(default_factory=dict)

    def validate(self, base: MatchedTrainingConfig | None = None) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("matched-exposure seeds must be non-empty and unique")
        if not 0 < self.source_continuation_step < self.target_continuation_step:
            raise ValueError("invalid matched-exposure continuation range")
        if (
            not self.checkpoints
            or max(self.checkpoints) != self.target_continuation_step
            or min(self.checkpoints) <= self.source_continuation_step
        ):
            raise ValueError("matched-exposure checkpoints must extend to the target")
        if self.learning_rate != 3e-4 or self.readout != "raw":
            raise ValueError("extension must preserve the selected forensic candidate")
        if not 0 < self.ema_decay < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        if min(self.diagnostic_evaluation_samples, self.progress_interval) <= 0:
            raise ValueError("invalid matched-exposure diagnostic size")
        for label, digest in (
            ("source configuration", self.source_configuration_sha256),
            ("source final report", self.source_final_report_sha256),
            ("source selection", self.source_selection_sha256),
            ("source sample manifest", self.source_sample_manifest_sha256),
            ("source normalization", self.source_normalization_sha256),
        ):
            _validate_sha256(label, digest)
        expected = {
            f"arms/seed{seed}_lr3e-04/state_step{self.source_continuation_step}.pt"
            for seed in self.seeds
        }
        if set(self.source_state_hashes) != expected:
            raise ValueError("matched-exposure source-state hash map is incomplete")
        for name, digest in self.source_state_hashes.items():
            _validate_sha256(name, digest)
        if base is not None:
            if tuple(base.seeds) != self.seeds:
                raise ValueError("extension must preserve frozen S3 seed order")
            if base.training_steps != 144_000:
                raise ValueError("extension requires the frozen S3 exposure target")
            if self.target_continuation_step * base.batch_size / 23_040_000 != 12.8:
                raise ValueError("extension no longer reaches 25.6 total S3 passes")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base: MatchedTrainingConfig | None = None,
    ) -> "MatchedExposureExtensionConfig":
        payload = dict(values)
        for key in ("seeds", "checkpoints"):
            if key in payload:
                payload[key] = tuple(map(int, payload[key]))
        result = cls(**payload)
        result.validate(base)
        return result


def _validate_sha256(label: str, digest: str) -> None:
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 value")


class S3MatchedExposureExtension(PaperScaleMatchedTrainer):
    def __init__(
        self,
        corpus_root: Path,
        source_root: Path,
        output_dir: Path,
        base_config: MatchedTrainingConfig,
        extension_config: MatchedExposureExtensionConfig,
        *,
        code_revision: str,
    ) -> None:
        extension_config.validate(base_config)
        self.source_root = Path(source_root)
        self.extension_config = extension_config
        super().__init__(
            corpus_root, output_dir, base_config, code_revision=code_revision
        )
        self._verify_source()
        self.configuration_payload = {
            "extension_code_revision": self.code_revision,
            "base_training": json.loads(json.dumps(asdict(base_config))),
            "matched_exposure_extension": json.loads(
                json.dumps(asdict(extension_config))
            ),
        }
        self.configuration_sha256 = hashlib.sha256(
            json.dumps(
                self.configuration_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self.diagnostic_raw = self.corpus.sample_raw_global(
            1,
            min(
                extension_config.diagnostic_evaluation_samples,
                self.corpus.validation_count,
            ),
            np.random.default_rng(extension_config.diagnostic_evaluation_seed),
            include_labels=True,
        )
        self.corpus.close()

    def _verify_source(self) -> None:
        config = self.extension_config
        paths = {
            "final_report": self.source_root / "final_report.json",
            "selection": self.source_root / "diagnostic_selection.json",
            "sample": self.source_root / "diagnostic_sample_manifest.json",
            "normalization": self.source_root / "normalization.json",
        }
        expected = {
            "final_report": config.source_final_report_sha256,
            "selection": config.source_selection_sha256,
            "sample": config.source_sample_manifest_sha256,
            "normalization": config.source_normalization_sha256,
        }
        for label, path in paths.items():
            if not path.is_file() or _sha256(path) != expected[label]:
                raise RuntimeError(f"forensic source {label} mismatch")
        report = json.loads(paths["final_report"].read_text(encoding="utf-8"))
        if (
            report.get("code_revision") != config.source_code_revision
            or report.get("configuration_sha256")
            != config.source_configuration_sha256
        ):
            raise RuntimeError("forensic source identity mismatch")
        selected = report.get("diagnostic_selection", {}).get("selected", {})
        if (
            float(selected.get("learning_rate", -1)) != config.learning_rate
            or selected.get("readout") != config.readout
        ):
            raise RuntimeError("forensic source did not select the frozen candidate")
        manifest = json.loads(paths["sample"].read_text(encoding="utf-8"))
        if (
            int(manifest.get("seed", -1)) != config.diagnostic_evaluation_seed
            or int(manifest.get("example_count", -1))
            != config.diagnostic_evaluation_samples
        ):
            raise RuntimeError("forensic diagnostic sample contract changed")
        normalization = json.loads(
            paths["normalization"].read_text(encoding="utf-8")
        )
        if normalization != self.transform.to_dict():
            raise RuntimeError("forensic normalization differs from current S3 transform")
        for relative, digest in config.source_state_hashes.items():
            path = self.source_root / relative
            if not path.is_file() or _sha256(path) != digest:
                raise RuntimeError(f"forensic optimizer state mismatch: {relative}")

    def _evaluate_fixed(self, model: Any) -> Dict[str, Any]:
        original_iter = self.corpus.iter_raw

        def fixed_iter(*_args: Any, **_kwargs: Any):
            total = len(self.diagnostic_raw["voltage_t_mv"])
            for start in range(0, total, 65_536):
                stop = min(start + 65_536, total)
                yield {
                    name: values[start:stop]
                    for name, values in self.diagnostic_raw.items()
                }

        self.corpus.iter_raw = fixed_iter  # type: ignore[method-assign]
        try:
            return self._evaluate(model)
        finally:
            self.corpus.iter_raw = original_iter  # type: ignore[method-assign]

    @staticmethod
    def _ema_update(ema_model: Any, model: Any, decay: float) -> None:
        import torch

        with torch.no_grad():
            source = model.state_dict()
            for name, value in ema_model.state_dict().items():
                candidate = source[name]
                if value.is_floating_point():
                    value.mul_(decay).add_(candidate, alpha=1.0 - decay)
                else:
                    value.copy_(candidate)

    def _source_state_path(self, seed: int) -> Path:
        return self.source_root / (
            f"arms/seed{seed}_lr3e-04/"
            f"state_step{self.extension_config.source_continuation_step}.pt"
        )

    def _new_models_and_optimizers(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        models = self._models()
        ema_models = self._models()
        optimizers = {
            name: self.torch.optim.AdamW(
                models[name].parameters(),
                lr=self.extension_config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            for name in MODEL_NAMES
        }
        return models, ema_models, optimizers

    def _run_seed(self, seed: int) -> Dict[str, Any]:
        run_dir = self.output_dir / "seeds" / f"seed{seed}"
        completion_path = run_dir / "completed.json"
        if completion_path.is_file():
            payload = json.loads(completion_path.read_text(encoding="utf-8"))
            if payload.get("configuration_sha256") != self.configuration_sha256:
                raise RuntimeError(f"completed extension seed changed contract: {seed}")
            print(f"[GIADA RunPod][S3 matched exposure seed={seed}] resumed", flush=True)
            return payload
        run_dir.mkdir(parents=True, exist_ok=True)
        models, ema_models, optimizers = self._new_models_and_optimizers()
        source = self.torch.load(
            self._source_state_path(seed),
            map_location=self.device,
            weights_only=False,
        )
        if (
            int(source.get("seed", -1)) != seed
            or float(source.get("learning_rate", -1))
            != self.extension_config.learning_rate
            or int(source.get("step", -1))
            != self.extension_config.source_continuation_step
            or source.get("configuration_sha256")
            != self.extension_config.source_configuration_sha256
        ):
            raise RuntimeError(f"forensic source-state metadata mismatch for seed {seed}")
        for name in MODEL_NAMES:
            models[name].load_state_dict(source["models"][name])
            ema_models[name].load_state_dict(source["ema_models"][name])
            optimizers[name].load_state_dict(source["optimizers"][name])
        rng = np.random.default_rng()
        rng.bit_generator.state = source["numpy_rng_state"]
        rows: list[Dict[str, Any]] = []
        start_step = self.extension_config.source_continuation_step
        state_paths = list(run_dir.glob("state_step*.pt"))
        if state_paths:
            state_path = max(
                state_paths,
                key=lambda path: int(path.stem.removeprefix("state_step")),
            )
            state = self.torch.load(
                state_path, map_location=self.device, weights_only=False
            )
            if state.get("configuration_sha256") != self.configuration_sha256:
                raise RuntimeError(f"resumable extension state changed contract: {state_path}")
            start_step = int(state["step"])
            rows = list(state["rows"])
            rng.bit_generator.state = state["numpy_rng_state"]
            for name in MODEL_NAMES:
                models[name].load_state_dict(state["models"][name])
                ema_models[name].load_state_dict(state["ema_models"][name])
                optimizers[name].load_state_dict(state["optimizers"][name])
            print(
                f"[GIADA RunPod][S3 matched exposure seed={seed}] resume step {start_step}",
                flush=True,
            )
        started = time.perf_counter()
        for step in range(start_step + 1, self.extension_config.target_continuation_step + 1):
            raw = self.corpus.sample_raw(0, self.config.batch_size, rng)
            features, target = self.transform.apply(raw)
            x = self.torch.as_tensor(features, device=self.device)
            y = self.torch.as_tensor(target, device=self.device)
            active = (
                self.torch.abs(y * self.config.voltage_scale_mv)
                >= self.config.active_delta_threshold_mv
            )
            weight = self.torch.where(active, self.config.active_weight, 1.0)
            losses = {}
            for name in MODEL_NAMES:
                model = models[name]
                model.train()
                optimizers[name].zero_grad(set_to_none=True)
                prediction = model(x)
                loss = self.torch.mean(weight * (prediction - y) ** 2)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.gradient_clip_norm
                )
                optimizers[name].step()
                self._ema_update(
                    ema_models[name], model, self.extension_config.ema_decay
                )
                losses[name] = float(loss.detach().cpu())
            if step in self.extension_config.checkpoints:
                for name in MODEL_NAMES:
                    for readout, model in (
                        ("raw", models[name]),
                        ("ema", ema_models[name]),
                    ):
                        rows.append(
                            {
                                "seed": seed,
                                "continuation_step": step,
                                "absolute_s3_step": 72_000 + step,
                                "model": name,
                                "readout": readout,
                                "metrics": self._evaluate_fixed(model),
                            }
                        )
                state_path = run_dir / f"state_step{step}.pt"
                temporary = state_path.with_suffix(".pt.tmp")
                self.torch.save(
                    {
                        "schema_version": "giada-s3-matched-exposure-resume-v1",
                        "configuration_sha256": self.configuration_sha256,
                        "seed": seed,
                        "learning_rate": self.extension_config.learning_rate,
                        "step": step,
                        "models": {name: models[name].state_dict() for name in MODEL_NAMES},
                        "ema_models": {name: ema_models[name].state_dict() for name in MODEL_NAMES},
                        "optimizers": {name: optimizers[name].state_dict() for name in MODEL_NAMES},
                        "numpy_rng_state": rng.bit_generator.state,
                        "rows": rows,
                    },
                    temporary,
                )
                temporary.replace(state_path)
            if (
                step == start_step + 1
                or step == self.extension_config.target_continuation_step
                or step % self.extension_config.progress_interval == 0
            ):
                completed = step - start_step
                eta = (time.perf_counter() - started) / max(completed, 1) * (
                    self.extension_config.target_continuation_step - step
                )
                compact = " ".join(
                    f"{name}={value:.4g}" for name, value in losses.items()
                )
                print(
                    f"[GIADA RunPod][S3 matched exposure seed={seed}] "
                    f"{step}/{self.extension_config.target_continuation_step} "
                    f"ETA {eta/60:.1f} min {compact}",
                    flush=True,
                )
        completion = {
            "schema_version": "giada-s3-matched-exposure-seed-v1",
            "configuration_sha256": self.configuration_sha256,
            "seed": seed,
            "source_continuation_step": self.extension_config.source_continuation_step,
            "target_continuation_step": self.extension_config.target_continuation_step,
            "exact_optimizer_and_rng_continuation": True,
            "rows": rows,
        }
        _atomic_json(completion_path, completion)
        return completion

    def _load_final_models(self, seed: int) -> Dict[str, Any]:
        path = (
            self.output_dir
            / "seeds"
            / f"seed{seed}"
            / f"state_step{self.extension_config.target_continuation_step}.pt"
        )
        state = self.torch.load(path, map_location=self.device, weights_only=False)
        models = self._models()
        for name in MODEL_NAMES:
            models[name].load_state_dict(state["models"][name])
        return models

    def _full_evaluation(self, seed: int) -> Dict[str, Any]:
        path = self.output_dir / "full_evaluation" / f"seed{seed}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("configuration_sha256") != self.configuration_sha256:
                raise RuntimeError(f"full extension evaluation changed contract: {path}")
            return payload
        models = self._load_final_models(seed)
        payload = {
            "schema_version": "giada-s3-matched-exposure-full-evaluation-v1",
            "configuration_sha256": self.configuration_sha256,
            "seed": seed,
            "absolute_s3_step": 144_000,
            "readout": "raw",
            "metrics": {name: self._evaluate(models[name]) for name in MODEL_NAMES},
        }
        _atomic_json(path, payload)
        print(
            f"[GIADA RunPod][S3 matched exposure full evaluation] seed {seed} complete",
            flush=True,
        )
        return payload

    @staticmethod
    def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        seed_wins = {}
        medians: Dict[str, Dict[str, float | None]] = {}
        for row in rows:
            seed = int(row["seed"])
            metrics = row["metrics"]
            seed_wins[str(seed)] = float(
                _metric(metrics["giada_voltage_bridge"], "overall")
            ) < float(_metric(metrics["branch_elm_core"], "overall"))
        for stratum in ("overall", "active", "spike"):
            medians[stratum] = {}
            for model in MODEL_NAMES:
                values = [_metric(row["metrics"][model], stratum) for row in rows]
                finite = [float(value) for value in values if value is not None]
                medians[stratum][model] = float(np.median(finite)) if finite else None
        stratum_wins = {
            stratum: bool(
                medians[stratum]["giada_voltage_bridge"]
                < medians[stratum]["branch_elm_core"]
            )
            for stratum in medians
        }
        return {
            "seed_wins": seed_wins,
            "observed_seed_wins": int(sum(seed_wins.values())),
            "median_rmse_mv": medians,
            "stratum_wins": stratum_wins,
            "matched_exposure_stability_passed": bool(
                sum(seed_wins.values()) >= 4 and all(stratum_wins.values())
            ),
        }

    def run(self) -> Dict[str, Any]:
        checkpoint_rows = []
        for seed in self.extension_config.seeds:
            completion = self._run_seed(seed)
            checkpoint_rows.extend(completion["rows"])
        full_rows = [self._full_evaluation(seed) for seed in self.extension_config.seeds]
        summary = self._summary(full_rows)
        report = {
            "schema_version": "giada-s3-matched-exposure-extension-v1",
            "valid": True,
            "code_revision": self.code_revision,
            "configuration_sha256": self.configuration_sha256,
            "configuration": self.configuration_payload,
            "continuation_contract": {
                "source_absolute_s3_step": 84_000,
                "target_absolute_s3_step": 144_000,
                "additional_steps": 60_000,
                "learning_rate": self.extension_config.learning_rate,
                "exact_optimizer_state_continuation": True,
                "exact_numpy_rng_state_continuation": True,
                "architecture_modified": False,
                "loss_modified": False,
                "data_modified": False,
                "hyperparameter_selection_during_extension": False,
            },
            "diagnostic_checkpoint_rows": checkpoint_rows,
            "full_development_validation": summary,
            "full_evaluation_rows": full_rows,
            "interpretation": {
                "matched_exposure_schedule_stability_diagnosed": summary[
                    "matched_exposure_stability_passed"
                ],
                "paper_confirmation_claimed": False,
                "s3_result_overridden": False,
                "s4_authorized": False,
                "reason": (
                    "The 3e-4 schedule was selected on this S3 development-validation "
                    "corpus; this extension tests equal-exposure stability but is not "
                    "a fresh independent confirmation."
                ),
            },
        }
        _atomic_json(self.output_dir / "final_report.json", report)
        return report
