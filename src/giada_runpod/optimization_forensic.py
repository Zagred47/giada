"""Preregistered, bounded forensic of S3 late optimization behavior.

This module deliberately does not modify either compared architecture.  It
loads the frozen 72k S3 weights, resets AdamW identically in every arm, and
tests learning rate plus raw/EMA readout on shared minibatch streams.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .training import MatchedTrainingConfig, PaperScaleMatchedTrainer


MODEL_NAMES = ("branch_elm_core", "giada_voltage_bridge")


@dataclass(frozen=True)
class LateOptimizationForensicConfig:
    source_checkpoint_step: int = 72_000
    discovery_seeds: tuple[int, ...] = (61017, 61043, 61103)
    confirmation_seeds: tuple[int, ...] = (61029, 61071)
    learning_rates: tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    continuation_steps: int = 12_000
    checkpoints: tuple[int, ...] = (3_000, 6_000, 12_000)
    ema_decay: float = 0.999
    diagnostic_evaluation_samples: int = 262_144
    diagnostic_evaluation_seed: int = 73_100_001
    continuation_stream_seed_offset: int = 74_000_000
    progress_interval: int = 500
    source_code_revision: str = ""
    source_final_report_sha256: str = ""
    source_normalization_sha256: str = ""
    source_checkpoint_hashes: Mapping[str, str] = field(default_factory=dict)
    selection_rule: str = "lexicographic_registered_gates_seed_wins_margin"

    @property
    def all_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds + self.confirmation_seeds

    def validate(self, base: MatchedTrainingConfig | None = None) -> None:
        if not self.discovery_seeds or not self.confirmation_seeds:
            raise ValueError("forensic requires discovery and confirmation seeds")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("discovery and confirmation seeds must be disjoint")
        if len(set(self.all_seeds)) != len(self.all_seeds):
            raise ValueError("forensic seeds must be unique")
        if not self.learning_rates or any(rate <= 0 for rate in self.learning_rates):
            raise ValueError("forensic learning rates must be positive")
        if len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("forensic learning rates must be unique")
        if self.continuation_steps <= 0:
            raise ValueError("continuation_steps must be positive")
        if not self.checkpoints or max(self.checkpoints) != self.continuation_steps:
            raise ValueError("final forensic checkpoint must equal continuation_steps")
        if any(step <= 0 for step in self.checkpoints):
            raise ValueError("forensic checkpoints must be positive")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in (0, 1)")
        if self.diagnostic_evaluation_samples < 1:
            raise ValueError("diagnostic evaluation sample must be non-empty")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be positive")
        if self.selection_rule != "lexicographic_registered_gates_seed_wins_margin":
            raise ValueError("unknown forensic selection rule")
        for label, digest in (
            ("source final report", self.source_final_report_sha256),
            ("source normalization", self.source_normalization_sha256),
        ):
            _validate_sha256(label, digest)
        expected_names = {
            f"{model}_seed{seed}_step{self.source_checkpoint_step}.pt"
            for seed in self.all_seeds
            for model in MODEL_NAMES
        }
        if set(self.source_checkpoint_hashes) != expected_names:
            raise ValueError("source checkpoint hash map is incomplete or unexpected")
        for name, digest in self.source_checkpoint_hashes.items():
            _validate_sha256(name, digest)
        if base is not None:
            if set(base.seeds) != set(self.all_seeds):
                raise ValueError("forensic seeds must preserve the frozen S3 seed set")
            if self.source_checkpoint_step not in base.checkpoints:
                raise ValueError("source step is not a frozen S3 checkpoint")
            if base.checkpoint_selection != "final_preregistered":
                raise ValueError("source run did not use the frozen final checkpoint")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base: MatchedTrainingConfig | None = None,
    ) -> "LateOptimizationForensicConfig":
        payload = dict(values)
        for key in ("discovery_seeds", "confirmation_seeds", "checkpoints"):
            if key in payload:
                payload[key] = tuple(map(int, payload[key]))
        if "learning_rates" in payload:
            payload["learning_rates"] = tuple(map(float, payload["learning_rates"]))
        result = cls(**payload)
        result.validate(base)
        return result


def _validate_sha256(label: str, digest: str) -> None:
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 value")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    if name == "overall":
        return metrics.get("soma_rmse_mv")
    if name == "active":
        return metrics.get("active_soma_rmse_mv")
    if name == "spike":
        return (
            metrics.get("activity_regime_metrics", {})
            .get("somatic_upcrossing_minus55mv", {})
            .get("soma_rmse_mv")
        )
    raise KeyError(name)


def summarize_candidates(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    learning_rates: Sequence[float],
) -> list[Dict[str, Any]]:
    """Apply the preregistered lexicographic selection table."""
    summaries = []
    for rate_index, rate in enumerate(learning_rates):
        for readout_index, readout in enumerate(("raw", "ema")):
            selected = [
                row
                for row in rows
                if float(row["learning_rate"]) == float(rate)
                and row["readout"] == readout
                and int(row["seed"]) in seeds
            ]
            by_seed_model = {
                (int(row["seed"]), str(row["model"])): row["metrics"]
                for row in selected
            }
            expected = {(int(seed), model) for seed in seeds for model in MODEL_NAMES}
            if set(by_seed_model) != expected:
                raise ValueError(
                    f"candidate lr={rate} readout={readout} is incomplete"
                )
            seed_wins = {
                str(seed): float(_metric(by_seed_model[(seed, "giada_voltage_bridge")], "overall"))
                < float(_metric(by_seed_model[(seed, "branch_elm_core")], "overall"))
                for seed in seeds
            }
            medians: Dict[str, Dict[str, float | None]] = {}
            for stratum in ("overall", "active", "spike"):
                medians[stratum] = {}
                for model in MODEL_NAMES:
                    values = [
                        _metric(by_seed_model[(seed, model)], stratum)
                        for seed in seeds
                    ]
                    finite = [float(value) for value in values if value is not None]
                    medians[stratum][model] = (
                        float(np.median(finite)) if finite else None
                    )
            stratum_wins = {
                stratum: bool(
                    medians[stratum]["giada_voltage_bridge"] is not None
                    and medians[stratum]["branch_elm_core"] is not None
                    and medians[stratum]["giada_voltage_bridge"]
                    < medians[stratum]["branch_elm_core"]
                )
                for stratum in ("overall", "active", "spike")
            }
            margin = (
                float(medians["overall"]["branch_elm_core"])
                - float(medians["overall"]["giada_voltage_bridge"])
            )
            registered_gates = bool(
                all(seed_wins.values()) and all(stratum_wins.values())
            )
            # Last two negative indices preserve config order on exact ties.
            score = (
                int(registered_gates),
                sum(seed_wins.values()),
                sum(stratum_wins.values()),
                margin,
                -rate_index,
                -readout_index,
            )
            summaries.append(
                {
                    "learning_rate": float(rate),
                    "readout": readout,
                    "seed_wins": seed_wins,
                    "observed_seed_wins": int(sum(seed_wins.values())),
                    "median_rmse_mv": medians,
                    "stratum_wins": stratum_wins,
                    "all_diagnostic_gates_passed": registered_gates,
                    "overall_median_margin_mv": margin,
                    "selection_score": list(score),
                }
            )
    return summaries


class S3LateOptimizationForensic(PaperScaleMatchedTrainer):
    def __init__(
        self,
        corpus_root: Path,
        source_root: Path,
        output_dir: Path,
        base_config: MatchedTrainingConfig,
        forensic_config: LateOptimizationForensicConfig,
        *,
        code_revision: str,
    ) -> None:
        forensic_config.validate(base_config)
        self.source_root = Path(source_root)
        self.forensic_config = forensic_config
        super().__init__(
            corpus_root, output_dir, base_config, code_revision=code_revision
        )
        self._verify_source()
        self.configuration_payload = {
            "forensic_code_revision": self.code_revision,
            "base_training": json.loads(json.dumps(asdict(base_config))),
            "forensic": json.loads(json.dumps(asdict(forensic_config))),
        }
        self.configuration_sha256 = hashlib.sha256(
            json.dumps(
                self.configuration_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self.diagnostic_raw = self.corpus.sample_raw_global(
            1,
            min(
                forensic_config.diagnostic_evaluation_samples,
                self.corpus.validation_count,
            ),
            np.random.default_rng(forensic_config.diagnostic_evaluation_seed),
            include_labels=True,
        )
        # The global sample touches most validation shards. Release those
        # descriptors before the long training loop; future reads reopen lazily.
        self.corpus.close()
        _atomic_json(
            self.output_dir / "diagnostic_sample_manifest.json",
            {
                "schema_version": "giada-s3-forensic-diagnostic-sample-v1",
                "selection_split": "development_validation",
                "sampling": "global_row_proportional_with_replacement",
                "seed": forensic_config.diagnostic_evaluation_seed,
                "example_count": len(self.diagnostic_raw["voltage_t_mv"]),
                "corpus_marker_fingerprint_sha256": base_config.expected_corpus_hashes.get(
                    "shard_marker_fingerprint_sha256"
                ),
                "used_for_paper_confirmation": False,
            },
        )

    def _verify_source(self) -> None:
        config = self.forensic_config
        final_report_path = self.source_root / "final_report.json"
        normalization_path = self.source_root / "normalization.json"
        for path in (final_report_path, normalization_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen S3 source artifact: {path}")
        if _sha256(final_report_path) != config.source_final_report_sha256:
            raise RuntimeError("frozen S3 final report hash mismatch")
        if _sha256(normalization_path) != config.source_normalization_sha256:
            raise RuntimeError("frozen S3 normalization hash mismatch")
        report = json.loads(final_report_path.read_text(encoding="utf-8"))
        if report.get("code_revision") != config.source_code_revision:
            raise RuntimeError("frozen S3 source code revision mismatch")
        source_normalization = json.loads(
            normalization_path.read_text(encoding="utf-8")
        )
        if source_normalization != self.transform.to_dict():
            raise RuntimeError("current S3 transform differs from frozen normalization")
        for filename, expected_hash in config.source_checkpoint_hashes.items():
            path = self.source_root / filename
            if not path.is_file() or _sha256(path) != expected_hash:
                raise RuntimeError(f"frozen S3 checkpoint mismatch: {filename}")

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

    def _evaluate_fixed(self, model: Any) -> Dict[str, Any]:
        # Reuse the production metric implementation without rescanning HDF5.
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

    def _source_checkpoint(self, seed: int, model: str) -> Path:
        return self.source_root / (
            f"{model}_seed{seed}_step{self.forensic_config.source_checkpoint_step}.pt"
        )

    def _load_models(self, seed: int) -> Dict[str, Any]:
        models = self._models()
        for name, model in models.items():
            payload = self.torch.load(
                self._source_checkpoint(seed, name),
                map_location=self.device,
                weights_only=False,
            )
            if (
                int(payload.get("seed", -1)) != seed
                or payload.get("model") != name
                or int(payload.get("step", -1))
                != self.forensic_config.source_checkpoint_step
                or payload.get("normalization") != self.transform.to_dict()
            ):
                raise RuntimeError(f"source checkpoint metadata mismatch for {name}/{seed}")
            model.load_state_dict(payload["state_dict"])
        return models

    def _run_name(self, seed: int, learning_rate: float) -> str:
        return f"seed{seed}_lr{learning_rate:.0e}".replace("+", "")

    def _source_diagnostic(self, seed: int) -> Dict[str, Any]:
        path = self.output_dir / "source_diagnostic" / f"seed{seed}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("configuration_sha256") != self.configuration_sha256:
                raise RuntimeError(f"source diagnostic has changed contract: {path}")
            return payload
        models = self._load_models(seed)
        payload = {
            "schema_version": "giada-s3-forensic-source-diagnostic-v1",
            "configuration_sha256": self.configuration_sha256,
            "seed": seed,
            "source_checkpoint_step": self.forensic_config.source_checkpoint_step,
            "metrics": {
                name: self._evaluate_fixed(models[name]) for name in MODEL_NAMES
            },
        }
        _atomic_json(path, payload)
        return payload

    def _run_arm(self, seed: int, learning_rate: float) -> Dict[str, Any]:
        run_dir = self.output_dir / "arms" / self._run_name(seed, learning_rate)
        completion_path = run_dir / "completed.json"
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("configuration_sha256") != self.configuration_sha256:
                raise RuntimeError(f"partial forensic arm has changed contract: {run_dir}")
            print(
                f"[GIADA RunPod][S3 forensic seed={seed} lr={learning_rate:g}] resumed",
                flush=True,
            )
            return completion
        run_dir.mkdir(parents=True, exist_ok=True)
        models = self._load_models(seed)
        # ELM exposes scripted/derived tensors whose deep-copied parameters can
        # cease to be leaves.  Build independent architecture-identical EMA
        # modules and load values instead.  They are never passed to an
        # optimizer and every EMA forward runs under ``torch.no_grad()`` in the
        # evaluation path, so changing their requires_grad flags is unnecessary.
        ema_models = self._models()
        for name in MODEL_NAMES:
            ema_models[name].load_state_dict(models[name].state_dict())
        optimizers = {
            name: self.torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=self.config.weight_decay,
            )
            for name, model in models.items()
        }
        rng = np.random.default_rng(
            seed + self.forensic_config.continuation_stream_seed_offset
        )
        rows: list[Dict[str, Any]] = []
        start_step = 0
        state_paths = sorted(run_dir.glob("state_step*.pt"))
        if state_paths:
            state_path = max(
                state_paths,
                key=lambda path: int(path.stem.removeprefix("state_step")),
            )
            state = self.torch.load(
                state_path, map_location=self.device, weights_only=False
            )
            if state.get("configuration_sha256") != self.configuration_sha256:
                raise RuntimeError(f"resumable forensic state has changed contract: {state_path}")
            start_step = int(state["step"])
            rows = list(state["rows"])
            rng.bit_generator.state = state["numpy_rng_state"]
            for name in MODEL_NAMES:
                models[name].load_state_dict(state["models"][name])
                ema_models[name].load_state_dict(state["ema_models"][name])
                optimizers[name].load_state_dict(state["optimizers"][name])
            print(
                f"[GIADA RunPod][S3 forensic seed={seed} lr={learning_rate:g}] "
                f"resume step {start_step}",
                flush=True,
            )
        started = time.perf_counter()
        for step in range(start_step + 1, self.forensic_config.continuation_steps + 1):
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
                    ema_models[name], model, self.forensic_config.ema_decay
                )
                losses[name] = float(loss.detach().cpu())
            if step in self.forensic_config.checkpoints:
                for name in MODEL_NAMES:
                    for readout, model in (
                        ("raw", models[name]),
                        ("ema", ema_models[name]),
                    ):
                        rows.append(
                            {
                                "seed": seed,
                                "learning_rate": learning_rate,
                                "continuation_step": step,
                                "absolute_source_step": (
                                    self.forensic_config.source_checkpoint_step + step
                                ),
                                "model": name,
                                "readout": readout,
                                "metrics": self._evaluate_fixed(model),
                            }
                        )
                state_path = run_dir / f"state_step{step}.pt"
                temporary = state_path.with_suffix(".pt.tmp")
                self.torch.save(
                    {
                        "schema_version": "giada-s3-forensic-resume-v1",
                        "configuration_sha256": self.configuration_sha256,
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "step": step,
                        "models": {
                            name: models[name].state_dict() for name in MODEL_NAMES
                        },
                        "ema_models": {
                            name: ema_models[name].state_dict() for name in MODEL_NAMES
                        },
                        "optimizers": {
                            name: optimizers[name].state_dict() for name in MODEL_NAMES
                        },
                        "numpy_rng_state": rng.bit_generator.state,
                        "rows": rows,
                    },
                    temporary,
                )
                temporary.replace(state_path)
            if (
                step == 1
                or step == self.forensic_config.continuation_steps
                or step % self.forensic_config.progress_interval == 0
            ):
                elapsed = max(time.perf_counter() - started, 1e-9)
                completed_now = step - start_step
                eta = elapsed / completed_now * (
                    self.forensic_config.continuation_steps - step
                )
                compact = " ".join(
                    f"{name}={value:.4g}" for name, value in losses.items()
                )
                print(
                    f"[GIADA RunPod][S3 forensic seed={seed} lr={learning_rate:g}] "
                    f"{step}/{self.forensic_config.continuation_steps} "
                    f"ETA {eta/60:.1f} min {compact}",
                    flush=True,
                )
        final_state_path = run_dir / "final_state.pt"
        temporary = final_state_path.with_suffix(".pt.tmp")
        self.torch.save(
            {
                "schema_version": "giada-s3-forensic-final-state-v1",
                "configuration_sha256": self.configuration_sha256,
                "seed": seed,
                "learning_rate": learning_rate,
                "step": self.forensic_config.continuation_steps,
                "models": {name: models[name].state_dict() for name in MODEL_NAMES},
                "ema_models": {
                    name: ema_models[name].state_dict() for name in MODEL_NAMES
                },
            },
            temporary,
        )
        temporary.replace(final_state_path)
        completion = {
            "schema_version": "giada-s3-forensic-arm-v1",
            "configuration_sha256": self.configuration_sha256,
            "seed": seed,
            "learning_rate": learning_rate,
            "optimizer": "fresh_AdamW_from_source_weights",
            "shared_stream_seed": (
                seed + self.forensic_config.continuation_stream_seed_offset
            ),
            "rows": rows,
        }
        _atomic_json(completion_path, completion)
        return completion

    def _load_readout_models(
        self, seed: int, learning_rate: float, readout: str
    ) -> Dict[str, Any]:
        path = (
            self.output_dir
            / "arms"
            / self._run_name(seed, learning_rate)
            / "final_state.pt"
        )
        payload = self.torch.load(path, map_location=self.device, weights_only=False)
        key = "models" if readout == "raw" else "ema_models"
        models = self._models()
        for name in MODEL_NAMES:
            models[name].load_state_dict(payload[key][name])
        return models

    def _evaluate_full_candidate(
        self, seed: int, learning_rate: float, readout: str
    ) -> Dict[str, Any]:
        path = self.output_dir / "full_evaluation" / f"seed{seed}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("configuration_sha256") != self.configuration_sha256:
                raise RuntimeError(f"full evaluation has changed contract: {path}")
            return payload
        models = self._load_readout_models(seed, learning_rate, readout)
        metrics = {name: self._evaluate(models[name]) for name in MODEL_NAMES}
        payload = {
            "schema_version": "giada-s3-forensic-full-evaluation-v1",
            "configuration_sha256": self.configuration_sha256,
            "seed": seed,
            "learning_rate": learning_rate,
            "readout": readout,
            "metrics": metrics,
        }
        _atomic_json(path, payload)
        print(
            f"[GIADA RunPod][S3 forensic full evaluation] seed {seed} complete",
            flush=True,
        )
        return payload

    @staticmethod
    def _pair_summary(full_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        seed_wins = {}
        medians: Dict[str, Dict[str, float | None]] = {}
        for row in full_rows:
            seed = int(row["seed"])
            metrics = row["metrics"]
            seed_wins[str(seed)] = float(
                _metric(metrics["giada_voltage_bridge"], "overall")
            ) < float(_metric(metrics["branch_elm_core"], "overall"))
        for stratum in ("overall", "active", "spike"):
            medians[stratum] = {}
            for model in MODEL_NAMES:
                values = [_metric(row["metrics"][model], stratum) for row in full_rows]
                finite = [float(value) for value in values if value is not None]
                medians[stratum][model] = (
                    float(np.median(finite)) if finite else None
                )
        return {
            "seed_wins": seed_wins,
            "observed_seed_wins": int(sum(seed_wins.values())),
            "median_rmse_mv": medians,
            "all_three_median_strata_favor_giada": all(
                medians[stratum]["giada_voltage_bridge"]
                < medians[stratum]["branch_elm_core"]
                for stratum in medians
            ),
        }

    def run(self) -> Dict[str, Any]:
        discovery_rows: list[Dict[str, Any]] = []
        diagnostic_checkpoint_rows: list[Dict[str, Any]] = []
        source_diagnostics = [
            self._source_diagnostic(seed)
            for seed in self.forensic_config.discovery_seeds
        ]
        for seed in self.forensic_config.discovery_seeds:
            for learning_rate in self.forensic_config.learning_rates:
                completion = self._run_arm(seed, learning_rate)
                diagnostic_checkpoint_rows.extend(completion["rows"])
                discovery_rows.extend(
                    row
                    for row in completion["rows"]
                    if int(row["continuation_step"])
                    == self.forensic_config.continuation_steps
                )
        candidates = summarize_candidates(
            discovery_rows,
            self.forensic_config.discovery_seeds,
            self.forensic_config.learning_rates,
        )
        selected = max(candidates, key=lambda row: tuple(row["selection_score"]))
        selected_rate = float(selected["learning_rate"])
        selected_readout = str(selected["readout"])
        _atomic_json(
            self.output_dir / "diagnostic_selection.json",
            {
                "schema_version": "giada-s3-forensic-selection-v1",
                "selection_rule": self.forensic_config.selection_rule,
                "development_validation_used_for_selection": True,
                "candidates": candidates,
                "selected": selected,
            },
        )
        print(
            f"[GIADA RunPod][S3 forensic] selected lr={selected_rate:g} "
            f"readout={selected_readout}",
            flush=True,
        )
        for seed in self.forensic_config.confirmation_seeds:
            self._run_arm(seed, selected_rate)
        full_rows = [
            self._evaluate_full_candidate(seed, selected_rate, selected_readout)
            for seed in self.forensic_config.all_seeds
        ]
        full_summary = self._pair_summary(full_rows)
        report = {
            "schema_version": "giada-s3-late-optimization-forensic-v1",
            "valid": True,
            "code_revision": self.code_revision,
            "configuration_sha256": self.configuration_sha256,
            "configuration": self.configuration_payload,
            "source_contract": {
                "checkpoint_step": self.forensic_config.source_checkpoint_step,
                "source_code_revision": self.forensic_config.source_code_revision,
                "optimizer_state_available": False,
                "exact_continuation_claimed": False,
                "all_arms_reset_adamw": True,
                "architecture_modified": False,
            },
            "diagnostic_selection": {
                "development_validation_used_for_selection": True,
                "source_checkpoint_diagnostics": source_diagnostics,
                "checkpoint_rows": diagnostic_checkpoint_rows,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selected": selected,
            },
            "full_development_validation": full_summary,
            "interpretation": {
                "purpose": "diagnose late S3 optimization instability",
                "paper_confirmation_claimed": False,
                "s3_result_overridden": False,
                "s4_authorized": False,
                "reason": (
                    "The schedule/readout was selected on the S3 development "
                    "validation corpus and all arms reset AdamW because historical "
                    "optimizer state was not saved."
                ),
            },
        }
        _atomic_json(self.output_dir / "final_report.json", report)
        return report
