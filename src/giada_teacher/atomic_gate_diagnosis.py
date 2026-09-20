"""Task 1b: multifactorial diagnosis of residual Ca_HVA m-gate error."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_gate_playground import (
    _evaluate_model,
    _file_sha256,
    _metrics,
    _models,
    _rollout_metrics,
    _sha256,
    _state_dict_cpu,
    materialize_atomic_gate_dataset,
)
from .double_oracle import ExtractedGateFormula
from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest


@dataclass(frozen=True)
class GateMDiagnosisConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    densities: tuple[str, ...] = ("base", "dense")
    widths: tuple[int, ...] = (8, 16, 32)
    objectives: tuple[str, ...] = ("endpoint_only", "endpoint_plus_privileged_rates")
    learning_rate: float = 0.003
    batch_size: int = 1024
    common_checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000)
    extended_checkpoints: tuple[int, ...] = (30000, 50000)
    auxiliary_inf_weight: float = 0.1
    auxiliary_log_tau_weight: float = 0.01
    extended_variants: tuple[str, ...] = (
        "base-w16-endpoint_only",
        "dense-w16-endpoint_only",
        "base-w32-endpoint_only",
        "base-w16-endpoint_plus_privileged_rates",
        "dense-w32-endpoint_plus_privileged_rates",
    )
    effect_threshold_fraction: float = 0.20
    rollout_steps: tuple[int, ...] = (10, 100, 1000)

    def validate(self) -> None:
        expected = {
            f"{density}-w{width}-{objective}"
            for density in self.densities for width in self.widths for objective in self.objectives
        }
        if not set(self.extended_variants) <= expected:
            raise ValueError("extended variants must belong to the registered factorial")
        if self.common_checkpoints[0] != 0 or self.common_checkpoints[-1] >= self.extended_checkpoints[0]:
            raise ValueError("checkpoint ladders are invalid")
        if min(self.widths) <= 0 or min(self.learning_rate, self.batch_size) <= 0:
            raise ValueError("invalid optimization configuration")


def variant_id(density: str, width: int, objective: str) -> str:
    return f"{density}-w{width}-{objective}"


def _reserved_voltage(voltage: float) -> bool:
    windows = ((-85.0, -79.0), (-60.0, -54.0), (-15.0, -9.0), (15.0, 21.0))
    return (
        voltage in {-100.0, 40.0}
        or -28.0 <= voltage <= -26.0
        or any(low - 2.0 <= voltage <= high + 2.0 for low, high in windows)
    )


def _row(formula: ExtractedGateFormula, inputs: np.ndarray) -> dict[str, Any]:
    rates = [formula.rates(float(voltage)) for voltage in inputs[:, 0]]
    return {
        "role": "fit",
        "axis": "dense_support",
        "inputs": np.asarray(inputs, dtype=np.float64),
        "targets": np.asarray([
            formula.step("m", float(state), float(voltage), float(dt))
            for voltage, state, dt in inputs
        ], dtype=np.float64),
        "privileged_inf": np.asarray([item["m_inf"] for item in rates], dtype=np.float64),
        "privileged_tau_ms": np.asarray([item["m_tau_ms"] for item in rates], dtype=np.float64),
    }


def build_dense_fit(formula: ExtractedGateFormula) -> dict[str, Any]:
    voltages = [
        float(value) for value in np.arange(-99.75, 40.0, 0.25)
        if not _reserved_voltage(float(value))
    ]
    states = np.linspace(0.025, 0.975, 20)
    dt_values = np.geomspace(0.025, 1.0, 8)
    inputs = np.asarray([
        (voltage, float(state), float(dt))
        for voltage in voltages for state in states for dt in dt_values
    ], dtype=np.float64)
    return _row(formula, inputs)


def build_fresh_confirmation(formula: ExtractedGateFormula) -> dict[str, dict[str, Any]]:
    in_support_voltage = [
        float(value) for value in np.arange(-97.5, 38.0, 2.5)
        if not _reserved_voltage(float(value))
    ]
    ood_voltage = [-117.5, -112.5, -107.5, 47.5, 52.5, 57.5]
    states = (0.025, 0.125, 0.375, 0.625, 0.875, 0.975)
    in_support_dt = (0.075, 0.3, 0.8)
    make = lambda voltages, dt_values: np.asarray([
        (voltage, state, dt) for voltage in voltages for state in states for dt in dt_values
    ], dtype=np.float64)
    inside = _row(formula, make(in_support_voltage, in_support_dt))
    outside = _row(formula, make(ood_voltage, in_support_dt))
    ood_dt = _row(formula, make((-92.5, -67.5, -42.5, -2.5, 32.5), (1.25,)))
    inside.update(role="fresh_confirmation", axis="disjoint_in_support")
    outside.update(role="fresh_confirmation", axis="new_ood_voltage")
    ood_dt.update(role="fresh_confirmation", axis="new_ood_dt")
    return {"fresh_in_support": inside, "fresh_ood_voltage": outside, "fresh_ood_dt": ood_dt}


def _dataset_fingerprint(row: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(row["inputs"], dtype="<f8").tobytes())
    digest.update(np.asarray(row["targets"], dtype="<f8").tobytes())
    return digest.hexdigest()


def prepare_gate_m_diagnosis(formula: ExtractedGateFormula) -> dict[str, Any]:
    atomic = materialize_atomic_gate_dataset(formula, gate="m")
    fit = {"base": atomic["strata"]["train"], "dense": build_dense_fit(formula)}
    development = {
        name: row for name, row in atomic["strata"].items() if row["role"] == "development"
    }
    fresh = build_fresh_confirmation(formula)
    old_sealed_fingerprint = hashlib.sha256(b"".join(
        np.asarray(row["inputs"], dtype="<f8").tobytes()
        for row in atomic["strata"].values() if row["role"] == "sealed_test"
    )).hexdigest()
    fresh_keys = {tuple(values) for row in fresh.values() for values in row["inputs"]}
    old_keys = {
        tuple(values) for row in atomic["strata"].values() if row["role"] == "sealed_test"
        for values in row["inputs"]
    }
    if fresh_keys & old_keys:
        raise RuntimeError("fresh confirmation overlaps the opened Task 1 sealed set")
    selection_keys = {
        tuple(values) for row in [*fit.values(), *development.values()] for values in row["inputs"]
    }
    if fresh_keys & selection_keys:
        raise RuntimeError("fresh confirmation overlaps fit or development selection data")
    return {
        "formula": formula, "fit": fit, "development": development, "fresh_confirmation": fresh,
        "contract": {
            "schema_version": "giada-task1b-data-v1",
            "base_fit_count": len(fit["base"]["inputs"]),
            "dense_fit_count": len(fit["dense"]["inputs"]),
            "development_count": sum(len(row["inputs"]) for row in development.values()),
            "fresh_confirmation_counts": {name: len(row["inputs"]) for name, row in fresh.items()},
            "fit_fingerprints": {name: _dataset_fingerprint(row) for name, row in fit.items()},
            "fresh_fingerprints": {name: _dataset_fingerprint(row) for name, row in fresh.items()},
            "old_task1_sealed_fingerprint": old_sealed_fingerprint,
            "old_task1_sealed_rows_reused": False,
            "fit_or_development_rows_reused": False,
            "fresh_used_for_selection": False,
        },
    }


def _improvement(reference: float, candidate: float) -> float:
    return float((reference - candidate) / max(reference, 1e-15))


def _paired_batch_generator(length: int, batch_size: int, seed: int):
    """Replayable O(batch) stream; avoids materializing tens of millions of indices."""

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(length)
    cursor = 0
    while True:
        remaining = length - cursor
        if remaining >= batch_size:
            batch = permutation[cursor : cursor + batch_size]
            cursor += batch_size
            yield batch
            continue
        head = permutation[cursor:]
        permutation = rng.permutation(length)
        needed = batch_size - remaining
        batch = np.concatenate((head, permutation[:needed]))
        cursor = needed
        yield batch


def run_gate_m_diagnosis(
    bundle: dict[str, Any], output_dir: str | Path, config: GateMDiagnosisConfig | None = None
) -> dict[str, Any]:
    config = config or GateMDiagnosisConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch = configure_torch_runtime(config.seeds[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development = list(bundle["development"].items())
    stream_specs, tensors = {}, {}
    max_steps_by_variant = {}
    for density in config.densities:
        fit = bundle["fit"][density]
        tensors[density] = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in {
                "inputs": fit["inputs"], "targets": fit["targets"],
                "inf": fit["privileged_inf"], "tau": fit["privileged_tau_ms"],
            }.items()
        }
        for seed in config.seeds:
            spec = {
                "length": len(fit["inputs"]), "batch_size": config.batch_size,
                "seed": seed + 200000, "algorithm": "numpy-PCG64-epoch-permutation-v1",
            }
            spec["sha256"] = hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            stream_specs[(density, seed)] = spec
    variants = [
        (density, width, objective, variant_id(density, width, objective))
        for density in config.densities for width in config.widths for objective in config.objectives
    ]
    total_steps = sum(
        (max(config.extended_checkpoints) if name in config.extended_variants else max(config.common_checkpoints))
        * len(config.seeds) for _, _, _, name in variants
    )
    completed_steps, started = 0, time.perf_counter()
    runs, snapshots = [], {}
    for density, width, objective, name in variants:
        maximum = max(config.extended_checkpoints) if name in config.extended_variants else max(config.common_checkpoints)
        checkpoints = config.common_checkpoints + (config.extended_checkpoints if name in config.extended_variants else ())
        max_steps_by_variant[name] = maximum
        constructor = _models(torch, width)["physical_tau"]
        for seed in config.seeds:
            configure_torch_runtime(seed)
            model = constructor().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
            data = tensors[density]
            stream = _paired_batch_generator(
                len(bundle["fit"][density]["inputs"]), config.batch_size, seed + 200000
            )
            checkpoint_rows = []
            for step in range(maximum + 1):
                if step in checkpoints:
                    metrics = _evaluate_model(model, development, torch, device)
                    key = f"{name}-seed{seed}-step{step}"
                    snapshots[key] = _state_dict_cpu(model, torch)
                    checkpoint_rows.append({
                        "step": step, "development_score": float(np.mean([x["rmse"] for x in metrics.values()])),
                        "checkpoint_key": key,
                    })
                    elapsed = time.perf_counter() - started
                    progressed = completed_steps + step
                    eta = elapsed / max(progressed, 1) * max(total_steps - progressed, 0)
                    print(
                        f"[GIADA Task 1b] {100*progressed/total_steps:.1f}% ETA {eta/60:.1f} min "
                        f"{name} seed={seed} step={step} dev={checkpoint_rows[-1]['development_score']:.4g}"
                    )
                if step == maximum:
                    break
                indices = torch.as_tensor(next(stream), dtype=torch.long, device=device)
                inputs = data["inputs"].index_select(0, indices)
                targets = data["targets"].index_select(0, indices)
                optimizer.zero_grad(set_to_none=True)
                prediction, details = model(inputs, diagnostics=True)
                loss = torch.mean((prediction - targets) ** 2)
                if objective == "endpoint_plus_privileged_rates":
                    inf_target = data["inf"].index_select(0, indices)
                    tau_target = data["tau"].index_select(0, indices)
                    loss = loss + config.auxiliary_inf_weight * torch.mean((details["inf"] - inf_target) ** 2)
                    loss = loss + config.auxiliary_log_tau_weight * torch.mean(
                        (torch.log(details["tau_ms"]) - torch.log(tau_target)) ** 2
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            completed_steps += maximum
            runs.append({
                "variant": name, "density": density, "width": width, "objective": objective,
                "seed": seed, "parameter_count": sum(p.numel() for p in model.parameters()),
                "stream_sha256": stream_specs[(density, seed)]["sha256"], "checkpoints": checkpoint_rows,
            })
    selections = {}
    for _, _, _, name in variants:
        variant_runs = [run for run in runs if run["variant"] == name]
        candidates = []
        for step in [row["step"] for row in variant_runs[0]["checkpoints"]]:
            replicas = [(run, next(x for x in run["checkpoints"] if x["step"] == step)) for run in variant_runs]
            candidates.append((float(np.mean([x["development_score"] for _, x in replicas])), step, replicas))
        score, step, replicas = min(candidates, key=lambda item: (item[0], item[1]))
        selections[name] = {
            "step": step, "mean_development_score": score,
            "seed_checkpoint_keys": {str(run["seed"]): row["checkpoint_key"] for run, row in replicas},
        }
    winner = min(selections, key=lambda name: (selections[name]["mean_development_score"], name))
    saved = {
        "selected": {
            name: {seed: snapshots[key] for seed, key in row["seed_checkpoint_keys"].items()}
            for name, row in selections.items()
        },
        "budget_probes": {
            name: {
                str(step): {
                    str(run["seed"]): snapshots[next(x["checkpoint_key"] for x in run["checkpoints"] if x["step"] == step)]
                    for run in runs if run["variant"] == name
                }
                for step in (10000, 30000, 50000)
            }
            for name in config.extended_variants
        },
        "common_10000": {
            name: {
                str(run["seed"]): snapshots[next(
                    x["checkpoint_key"] for x in run["checkpoints"] if x["step"] == 10000
                )]
                for run in runs if run["variant"] == name
            }
            for _, _, _, name in variants
        },
    }
    checkpoint_path = output_dir / "frozen_checkpoints.pt"
    torch.save(saved, checkpoint_path)
    freeze = {
        "schema_version": "giada-task1b-freeze-v1", "config": config.__dict__,
        "data_contract": bundle["contract"], "selection": selections, "winner": winner,
        "checkpoint_sha256": _file_sha256(checkpoint_path), "fresh_confirmation_accessed": False,
    }
    freeze["freeze_sha256"] = _sha256(freeze)
    (output_dir / "selection_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    training_report = {
        "schema_version": "giada-task1b-training-v1", "valid": True,
        "environment": environment_manifest(torch), "data_contract": bundle["contract"],
        "runs": runs, "selection": selections, "winner": winner,
        "fresh_confirmation_accessed": False, "selection_freeze_sha256": freeze["freeze_sha256"],
    }
    (output_dir / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    return training_report


def evaluate_gate_m_diagnosis(
    bundle: dict[str, Any], output_dir: str | Path, config: GateMDiagnosisConfig | None = None
) -> dict[str, Any]:
    config = config or GateMDiagnosisConfig()
    output_dir = Path(output_dir)
    if (output_dir / "fresh_confirmation_opened.json").exists():
        raise RuntimeError("fresh confirmation was already opened")
    freeze = json.loads((output_dir / "selection_freeze.json").read_text(encoding="utf-8"))
    claimed = freeze.pop("freeze_sha256")
    if _sha256(freeze) != claimed or freeze["fresh_confirmation_accessed"]:
        raise RuntimeError("invalid Task 1b selection freeze")
    checkpoint_path = output_dir / "frozen_checkpoints.pt"
    if _file_sha256(checkpoint_path) != freeze["checkpoint_sha256"]:
        raise RuntimeError("Task 1b checkpoint SHA-256 mismatch")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    fresh = list(bundle["fresh_confirmation"].items())
    per_variant, per_seed = {}, {}
    for name, selection in freeze["selection"].items():
        width = int(name.split("-w", 1)[1].split("-", 1)[0])
        constructor = _models(torch, width)["physical_tau"]
        seed_rows = {}
        for seed in config.seeds:
            model = constructor().to(device)
            model.load_state_dict(saved["selected"][name][str(seed)])
            seed_rows[str(seed)] = _evaluate_model(model, fresh, torch, device)
        per_seed[name] = seed_rows
        per_variant[name] = {
            stratum: {
                metric: float(np.mean([seed_rows[str(seed)][stratum][metric] for seed in config.seeds]))
                for metric in next(iter(seed_rows.values()))[stratum]
            } for stratum in bundle["fresh_confirmation"]
        }
    macro = {
        name: float(np.mean([row["rmse"] for row in strata.values()]))
        for name, strata in per_variant.items()
    }
    common_10000_metrics = {}
    for name in freeze["selection"]:
        width = int(name.split("-w", 1)[1].split("-", 1)[0])
        constructor = _models(torch, width)["physical_tau"]
        scores = []
        for seed in config.seeds:
            model = constructor().to(device)
            model.load_state_dict(saved["common_10000"][name][str(seed)])
            metrics = _evaluate_model(model, fresh, torch, device)
            scores.append(float(np.mean([row["rmse"] for row in metrics.values()])))
        common_10000_metrics[name] = float(np.mean(scores))
    winner = freeze["winner"]
    winner_width = int(winner.split("-w", 1)[1].split("-", 1)[0])
    winner_rollouts = []
    for seed in config.seeds:
        model = _models(torch, winner_width)["physical_tau"]().to(device)
        model.load_state_dict(saved["selected"][winner][str(seed)])
        winner_rollouts.append(_rollout_metrics(
            model, fresh, bundle["formula"], "m", config.rollout_steps, torch, device
        ))
    rollout = {
        horizon: float(np.mean([row[horizon] for row in winner_rollouts]))
        for horizon in map(str, config.rollout_steps)
    }
    def matched_effect(axis: str):
        effects = []
        if axis == "density":
            for width in config.widths:
                for objective in config.objectives:
                    a, b = variant_id("base", width, objective), variant_id("dense", width, objective)
                    effects.append({"pair": [a, b], "checkpoint": 10000, "improvement_fraction": _improvement(common_10000_metrics[a], common_10000_metrics[b])})
        elif axis == "capacity":
            for density in config.densities:
                for objective in config.objectives:
                    a, b = variant_id(density, 16, objective), variant_id(density, 32, objective)
                    effects.append({"pair": [a, b], "checkpoint": 10000, "improvement_fraction": _improvement(common_10000_metrics[a], common_10000_metrics[b])})
        else:
            for density in config.densities:
                for width in config.widths:
                    a = variant_id(density, width, "endpoint_only")
                    b = variant_id(density, width, "endpoint_plus_privileged_rates")
                    effects.append({"pair": [a, b], "checkpoint": 10000, "improvement_fraction": _improvement(common_10000_metrics[a], common_10000_metrics[b])})
        return effects
    effects = {axis: matched_effect(axis) for axis in ("density", "capacity", "objective")}
    budget = {}
    for name in config.extended_variants:
        width = int(name.split("-w", 1)[1].split("-", 1)[0])
        constructor = _models(torch, width)["physical_tau"]
        budget[name] = {}
        for step in (10000, 30000, 50000):
            seed_scores = []
            for seed in config.seeds:
                model = constructor().to(device)
                model.load_state_dict(saved["budget_probes"][name][str(step)][str(seed)])
                metrics = _evaluate_model(model, fresh, torch, device)
                seed_scores.append(float(np.mean([row["rmse"] for row in metrics.values()])))
            budget[name][str(step)] = float(np.mean(seed_scores))
    violations = sum(
        row["occupancy_violation_count"] for row in per_variant[winner].values()
    )
    scientific = macro[winner] <= 0.001 and rollout["1000"] <= 0.005 and violations == 0
    engineering = macro[winner] <= 0.0025 and rollout["1000"] <= 0.005 and violations == 0
    diagnosis = {
        axis: bool(np.median([row["improvement_fraction"] for row in rows]) >= config.effect_threshold_fraction)
        for axis, rows in effects.items()
    }
    diagnosis["budget_limited"] = any(
        _improvement(rows["10000"], rows["50000"]) >= config.effect_threshold_fraction
        for rows in budget.values()
    )
    report = {
        "schema_version": "giada-task1b-final-v1", "valid": True,
        "selection_freeze_sha256": claimed, "winner": winner,
        "fresh_confirmation_macro_rmse": macro, "fresh_metrics": per_variant,
        "common_10000_fresh_macro_rmse": common_10000_metrics,
        "per_seed_metrics": per_seed, "winner_rollout_rmse": rollout,
        "factor_effects": effects, "budget_probes": budget, "diagnosis": diagnosis,
        "scientific_gate_passed": scientific,
        "engineering_continuation_gate_passed": engineering,
        "old_task1_sealed_rows_reused": False, "fresh_confirmation_used_for_selection": False,
        "claim_scope": "held-voltage Ca_HVA m only",
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "fresh_confirmation_opened.json").write_text(json.dumps({
        "opened_once": True, "selection_freeze_sha256": claimed,
        "final_report_sha256": _file_sha256(output_dir / "final_report.json"),
    }, indent=2), encoding="utf-8")
    return report
