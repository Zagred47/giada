"""Task 1: paired atomic learnability playground for the Ca_HVA activation gate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .domain_splits import AtomicDomainSplitConfig, build_atomic_domain_splits
from .double_oracle import ExtractedGateFormula
from .gpu_baseline_runtime import (
    benchmark_cuda,
    configure_torch_runtime,
    environment_manifest,
    paired_index_generator,
    paired_index_stream,
)


@dataclass(frozen=True)
class AtomicGateTaskConfig:
    gate: str = "m"
    seeds: tuple[int, ...] = (17, 29, 43)
    learning_rates: tuple[float, ...] = (0.01, 0.003, 0.001)
    checkpoints: tuple[int, ...] = (0, 100, 300, 1000, 3000, 10000)
    batch_size: int = 1024
    hidden_width: int = 16
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 1.0
    rollout_steps: tuple[int, ...] = (10, 100, 1000)

    def validate(self) -> None:
        if self.gate not in {"m", "h"}:
            raise ValueError("atomic Ca_HVA tasks are restricted to the m and h gates")
        if self.checkpoints[0] != 0 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted, unique and start at zero")
        if min(self.seeds) < 0 or min(self.learning_rates) <= 0:
            raise ValueError("seeds and learning rates must be valid")
        if min(self.batch_size, self.hidden_width) <= 0:
            raise ValueError("batch size and hidden width must be positive")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratum_rows(split: dict[str, Any], gate: str) -> np.ndarray:
    rows = [
        (float(v), float(x), float(dt))
        for v in split["voltages_mv"]
        for x in split["states"]
        for dt in split["dt_ms"]
    ]
    result = np.asarray(rows, dtype=np.float64)
    expected = split["case_count"] // 2
    if gate not in {"m", "h"} or result.shape != (expected, 3):
        raise RuntimeError(f"stratum materialization mismatch: {result.shape} != {(expected, 3)}")
    return result


def materialize_atomic_gate_dataset(
    formula: ExtractedGateFormula,
    *,
    gate: str = "m",
    split_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize causal inputs and teacher targets without opening sealed rows."""

    split_report = split_report or build_atomic_domain_splits(AtomicDomainSplitConfig())
    strata: dict[str, Any] = {}
    for name, split in split_report["strata"].items():
        inputs = _stratum_rows(split, gate)
        targets = np.asarray(
            [formula.step(gate, state, voltage, dt) for voltage, state, dt in inputs],
            dtype=np.float64,
        )
        rates = [formula.rates(voltage) for voltage in inputs[:, 0]]
        strata[name] = {
            "role": split["role"],
            "axis": split["varied_axis"],
            "inputs": inputs,
            "targets": targets,
            "privileged_inf": np.asarray([row[f"{gate}_inf"] for row in rates]),
            "privileged_tau_ms": np.asarray([row[f"{gate}_tau_ms"] for row in rates]),
        }
    return {
        "gate": gate,
        "teacher_source_sha256": formula.source_sha256,
        "split_schema_version": split_report["schema_version"],
        "strata": strata,
    }


def _models(torch, hidden_width: int):
    nn = torch.nn

    class InputTransform(nn.Module):
        def forward(self, values):
            voltage, state, dt = values.unbind(-1)
            return torch.stack(((voltage + 30.0) / 90.0, state, torch.log(dt) / 4.0), dim=-1)

    class DirectMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.transform = InputTransform()
            self.network = nn.Sequential(
                nn.Linear(3, hidden_width), nn.SiLU(),
                nn.Linear(hidden_width, hidden_width), nn.SiLU(),
                nn.Linear(hidden_width, 1),
            )

        def forward(self, values, diagnostics: bool = False):
            prediction = torch.sigmoid(self.network(self.transform(values))).squeeze(-1)
            return (prediction, {}) if diagnostics else prediction

    class PhysicalTau(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(1, hidden_width), nn.SiLU(),
                nn.Linear(hidden_width, hidden_width), nn.SiLU(),
            )
            self.inf_head = nn.Linear(hidden_width, 1)
            self.tau_head = nn.Linear(hidden_width, 1)

        def forward(self, values, diagnostics: bool = False):
            voltage, state, dt = values.unbind(-1)
            features = self.trunk(((voltage + 30.0) / 90.0).unsqueeze(-1))
            inf = torch.sigmoid(self.inf_head(features)).squeeze(-1)
            tau = torch.nn.functional.softplus(self.tau_head(features)).squeeze(-1) + 1e-5
            z = -torch.expm1(-dt / tau)
            prediction = (1.0 - z) * state + z * inf
            details = {"inf": inf, "tau_ms": tau, "z": z}
            return (prediction, details) if diagnostics else prediction

    class DirectZ(nn.Module):
        def __init__(self):
            super().__init__()
            self.inf_net = nn.Sequential(
                nn.Linear(1, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 1)
            )
            self.z_net = nn.Sequential(
                nn.Linear(2, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 1)
            )

        def forward(self, values, diagnostics: bool = False):
            voltage, state, dt = values.unbind(-1)
            normalized_v = ((voltage + 30.0) / 90.0).unsqueeze(-1)
            inf = torch.sigmoid(self.inf_net(normalized_v)).squeeze(-1)
            z_input = torch.cat((normalized_v, (torch.log(dt) / 4.0).unsqueeze(-1)), dim=-1)
            z = torch.sigmoid(self.z_net(z_input)).squeeze(-1)
            prediction = (1.0 - z) * state + z * inf
            details = {"inf": inf, "z": z}
            return (prediction, details) if diagnostics else prediction

    return {"direct_mlp": DirectMLP, "physical_tau": PhysicalTau, "direct_z": DirectZ}


def build_atomic_gate_models(hidden_width: int = 16, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    return {name: constructor() for name, constructor in _models(torch_module, hidden_width).items()}


def _metrics(
    prediction: np.ndarray, target: np.ndarray, *, conductance_power: int = 2
) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    finite = bool(np.isfinite(prediction).all())
    violations = int(np.count_nonzero((prediction < 0.0) | (prediction > 1.0)))
    result = {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "conductance_factor_rmse": float(np.sqrt(np.mean(
            (prediction**conductance_power - target**conductance_power) ** 2
        ))),
        "conductance_power": int(conductance_power),
        "finite": finite,
        "occupancy_violation_count": violations,
    }
    if conductance_power == 2:
        result["conductance_factor_m2_rmse"] = result["conductance_factor_rmse"]
    return result


def _evaluate_model(
    model, rows: Iterable[tuple[str, dict[str, Any]]], torch, device, *, gate: str = "m"
):
    model.eval()
    report = {}
    with torch.inference_mode():
        for name, row in rows:
            inputs = torch.as_tensor(row["inputs"], dtype=torch.float32, device=device)
            prediction, details = model(inputs, diagnostics=True)
            prediction_np = prediction.detach().cpu().double().numpy()
            metrics = _metrics(
                prediction_np, row["targets"], conductance_power=2 if gate == "m" else 1
            )
            if "inf" in details:
                metrics["privileged_inf_rmse"] = float(np.sqrt(np.mean(
                    (details["inf"].detach().cpu().double().numpy() - row["privileged_inf"]) ** 2
                )))
            if "tau_ms" in details:
                predicted_tau = details["tau_ms"].detach().cpu().double().numpy()
                metrics["privileged_tau_log_rmse"] = float(np.sqrt(np.mean(
                    (np.log(predicted_tau) - np.log(row["privileged_tau_ms"])) ** 2
                )))
            report[name] = metrics
    return report


def _development_score(metrics: dict[str, dict[str, Any]]) -> float:
    return float(np.mean([row["rmse"] for row in metrics.values()]))


def _training_support_lut(fit: dict[str, Any]):
    voltages = np.unique(fit["inputs"][:, 0])
    inf = np.asarray([
        np.mean(fit["privileged_inf"][fit["inputs"][:, 0] == voltage]) for voltage in voltages
    ])
    tau = np.asarray([
        np.mean(fit["privileged_tau_ms"][fit["inputs"][:, 0] == voltage]) for voltage in voltages
    ])

    def predict(inputs: np.ndarray) -> np.ndarray:
        voltage, state, dt = np.asarray(inputs, dtype=np.float64).T
        interpolated_inf = np.interp(voltage, voltages, inf)
        interpolated_tau = np.exp(np.interp(voltage, voltages, np.log(tau)))
        return interpolated_inf + (state - interpolated_inf) * np.exp(-dt / interpolated_tau)

    return predict


def _rollout_metrics(model, rows, formula, gate, horizons, torch, device):
    result = {str(horizon): [] for horizon in horizons}
    model.eval()
    with torch.inference_mode():
        for _, row in rows:
            inputs = torch.as_tensor(row["inputs"], dtype=torch.float32, device=device)
            voltage = inputs[:, 0]
            state = inputs[:, 1]
            dt = inputs[:, 2]
            for step in range(1, max(horizons) + 1):
                state = model(torch.stack((voltage, state, dt), dim=-1))
                if step in horizons:
                    target = np.asarray([
                        formula.step(gate, float(x), float(v), float(delta * step))
                        for v, x, delta in row["inputs"]
                    ])
                    result[str(step)].append(_metrics(
                        state.cpu().double().numpy(), target,
                        conductance_power=2 if gate == "m" else 1,
                    )["rmse"])
    return {horizon: float(np.mean(values)) for horizon, values in result.items()}


def _state_dict_cpu(model, torch) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_and_select_atomic_gate(
    dataset: dict[str, Any], output_dir: str | Path, config: AtomicGateTaskConfig | None = None
) -> dict[str, Any]:
    """Train paired arms and freeze selection without reading sealed-test arrays."""

    config = config or AtomicGateTaskConfig()
    config.validate()
    torch = configure_torch_runtime(config.seeds[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    fit = dataset["strata"]["train"]
    development = [(name, row) for name, row in dataset["strata"].items() if row["role"] == "development"]
    # This is the firewall: no sealed row is iterated, tensorized or scored in this function.
    max_steps = config.checkpoints[-1]
    if config.gate == "m":
        stream_by_seed = {
            seed: paired_index_stream(
                len(fit["inputs"]), config.batch_size, max_steps, seed + 100000
            )
            for seed in config.seeds
        }
        stream_hashes = {str(seed): row["sha256"] for seed, row in stream_by_seed.items()}
    else:
        stream_by_seed = {}
        stream_hashes = {}
        for seed in config.seeds:
            spec = {
                "length": len(fit["inputs"]),
                "batch_size": config.batch_size,
                "seed": seed + 100000,
                "algorithm": "python-random-shuffle-epoch-v1",
            }
            stream_hashes[str(seed)] = hashlib.sha256(_canonical_json(spec)).hexdigest()
    model_types = _models(torch, config.hidden_width)
    runs, snapshots = [], {}
    total_runs = len(model_types) * len(config.seeds) * len(config.learning_rates)
    completed = 0
    started = time.perf_counter()
    fit_inputs = torch.as_tensor(fit["inputs"], dtype=torch.float32, device=device)
    fit_targets = torch.as_tensor(fit["targets"], dtype=torch.float32, device=device)
    for family, constructor in model_types.items():
        for seed in config.seeds:
            for learning_rate in config.learning_rates:
                batches = (
                    stream_by_seed[seed]["batches"]
                    if config.gate == "m"
                    else paired_index_generator(
                        len(fit["inputs"]), config.batch_size, seed + 100000
                    )
                )
                configure_torch_runtime(seed)
                model = constructor().to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
                )
                checkpoint_rows = []
                for step in range(max_steps + 1):
                    if step in config.checkpoints:
                        development_metrics = _evaluate_model(
                            model, development, torch, device, gate=config.gate
                        )
                        key = f"{family}-seed{seed}-lr{learning_rate:g}-step{step}"
                        snapshots[key] = _state_dict_cpu(model, torch)
                        checkpoint_rows.append({
                            "step": step,
                            "development_score": _development_score(development_metrics),
                            "development_metrics": development_metrics,
                            "checkpoint_key": key,
                        })
                    if step == max_steps:
                        break
                    batch = batches[step] if config.gate == "m" else next(batches)
                    indices = torch.as_tensor(batch, dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(fit_inputs.index_select(0, indices))
                    loss = torch.mean((prediction - fit_targets.index_select(0, indices)) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                runs.append({
                    "family": family, "seed": seed, "learning_rate": learning_rate,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "paired_stream_sha256": stream_hashes[str(seed)], "checkpoints": checkpoint_rows,
                })
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (total_runs - completed)
                print(
                    f"[GIADA Task {'1' if config.gate == 'm' else '2'}] "
                    f"{completed}/{total_runs} ({100*completed/total_runs:.1f}%) "
                    f"ETA {eta/60:.1f} min {family} seed={seed} lr={learning_rate:g}"
                )
    selections = {}
    for family in model_types:
        family_runs = [run for run in runs if run["family"] == family]
        family_candidates = []
        for learning_rate in config.learning_rates:
            for step in config.checkpoints:
                replicas = [
                    (run, next(row for row in run["checkpoints"] if row["step"] == step))
                    for run in family_runs if run["learning_rate"] == learning_rate
                ]
                family_candidates.append((
                    float(np.mean([row["development_score"] for _, row in replicas])),
                    learning_rate,
                    step,
                    replicas,
                ))
        score, learning_rate, step, replicas = min(
            family_candidates, key=lambda item: (item[0], item[1], item[2])
        )
        selections[family] = {
            "learning_rate": learning_rate, "step": step,
            "mean_development_score_across_seeds": score,
            "seed_checkpoint_keys": {
                str(run["seed"]): checkpoint["checkpoint_key"] for run, checkpoint in replicas
            },
        }
    checkpoint_path = output_dir / "selected_checkpoints.pt"
    torch.save({
        family: {
            seed: snapshots[checkpoint_key]
            for seed, checkpoint_key in selection["seed_checkpoint_keys"].items()
        }
        for family, selection in selections.items()
    }, checkpoint_path)
    freeze = {
        "schema_version": f"giada-task{'1' if config.gate == 'm' else '2'}-selection-freeze-v1",
        "gate": config.gate,
        "config": asdict(config),
        "teacher_source_sha256": dataset["teacher_source_sha256"],
        "paired_stream_sha256": stream_hashes,
        "selection": selections,
        "selected_checkpoints_sha256": _file_sha256(checkpoint_path),
        "sealed_test_accessed": False,
    }
    freeze["freeze_sha256"] = _sha256(freeze)
    (output_dir / "selection_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    training_report = {
        "schema_version": f"giada-task{'1' if config.gate == 'm' else '2'}-training-v1",
        "valid": True,
        "device": str(device), "environment": environment_manifest(torch), "runs": runs,
        "selection_freeze_sha256": freeze["freeze_sha256"], "sealed_test_accessed": False,
    }
    (output_dir / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    return training_report


def evaluate_frozen_atomic_gate(
    dataset: dict[str, Any], output_dir: str | Path, config: AtomicGateTaskConfig | None = None,
    *, formula: ExtractedGateFormula | None = None,
) -> dict[str, Any]:
    """Open sealed strata exactly once after verifying the immutable selection freeze."""

    config = config or AtomicGateTaskConfig()
    output_dir = Path(output_dir)
    if (output_dir / "sealed_test_opened.json").exists() or (output_dir / "final_report.json").exists():
        raise RuntimeError("sealed test has already been opened for this frozen experiment")
    freeze_path = output_dir / "selection_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    claimed_hash = freeze.pop("freeze_sha256")
    if _sha256(freeze) != claimed_hash or freeze["sealed_test_accessed"]:
        raise RuntimeError("invalid or previously opened selection freeze")
    freeze["freeze_sha256"] = claimed_hash
    checkpoint_path = output_dir / "selected_checkpoints.pt"
    if _file_sha256(checkpoint_path) != freeze["selected_checkpoints_sha256"]:
        raise RuntimeError("selected checkpoint SHA-256 mismatch")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sealed = [(name, row) for name, row in dataset["strata"].items() if row["role"] == "sealed_test"]
    if formula is None:
        raise ValueError("the extracted teacher formula is required for rollout and LUT audits")
    results, rollout, per_seed_results = {}, {}, {}
    constructors = _models(torch, config.hidden_width)
    for family, selection in freeze["selection"].items():
        family_seed_metrics, family_seed_rollouts = {}, {}
        for seed in config.seeds:
            model = constructors[family]().to(device)
            model.load_state_dict(checkpoints[family][str(seed)])
            family_seed_metrics[str(seed)] = _evaluate_model(
                model, sealed, torch, device, gate=config.gate
            )
            family_seed_rollouts[str(seed)] = _rollout_metrics(
                model, sealed, formula, config.gate, config.rollout_steps, torch, device
            )
        per_seed_results[family] = family_seed_metrics
        results[family] = {
            stratum: {
                metric: float(np.mean([
                    family_seed_metrics[str(seed)][stratum][metric] for seed in config.seeds
                ]))
                for metric in next(iter(family_seed_metrics.values()))[stratum]
            }
            for stratum in dict(sealed)
        }
        rollout[family] = {
            horizon: float(np.mean([
                family_seed_rollouts[str(seed)][horizon] for seed in config.seeds
            ]))
            for horizon in map(str, config.rollout_steps)
        }
    # Exact teacher and a causal persistence control bound interpretation of learned errors.
    analytic = {}
    persistence = {}
    for name, row in sealed:
        conductance_power = 2 if config.gate == "m" else 1
        analytic[name] = _metrics(
            row["targets"], row["targets"], conductance_power=conductance_power
        )
        persistence[name] = _metrics(
            row["inputs"][:, 1], row["targets"], conductance_power=conductance_power
        )
    lut = _training_support_lut(dataset["strata"]["train"])
    results["formula_oracle"] = analytic
    results["persistence_control"] = persistence
    results["linear_rate_lut"] = {
        name: _metrics(
            lut(row["inputs"]), row["targets"],
            conductance_power=2 if config.gate == "m" else 1,
        ) for name, row in sealed
    }
    aggregate = {
        family: float(np.mean([row["rmse"] for row in metrics.values()]))
        for family, metrics in results.items()
    }
    latency: dict[str, Any] = {"available": bool(torch.cuda.is_available()), "families": {}}
    if torch.cuda.is_available():
        for family in constructors:
            model = constructors[family]().to(device)
            timing_seed = str(config.seeds[0])
            model.load_state_dict(checkpoints[family][timing_seed])
            model.eval()
            family_rows = {"eager": {}}
            for batch_size in (1, 64, 1024):
                sample = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
                sample[:, 0], sample[:, 1], sample[:, 2] = -40.0, 0.5, 1.0
                family_rows["eager"][str(batch_size)] = benchmark_cuda(lambda: model(sample))
            if hasattr(torch, "compile"):
                compiled = torch.compile(model)
                compiled(sample)
                family_rows["compiled"] = {}
                for batch_size in (1, 64, 1024):
                    sample = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
                    sample[:, 0], sample[:, 1], sample[:, 2] = -40.0, 0.5, 1.0
                    family_rows["compiled"][str(batch_size)] = benchmark_cuda(lambda: compiled(sample))
            latency["families"][family] = family_rows
    report = {
        "schema_version": f"giada-task{'1' if config.gate == 'm' else '2'}-final-v1",
        "valid": all(row["finite"] for family in results.values() for row in family.values()),
        "gate": config.gate, "selection_freeze_sha256": claimed_hash,
        "sealed_strata": [name for name, _ in sealed], "metrics": results,
        "aggregate_sealed_rmse": aggregate,
        "per_seed_metrics": per_seed_results,
        "constant_voltage_rollout_rmse": rollout,
        "gpu_latency": latency,
        "selection_used_sealed_test": False,
        "claim_scope": (
            f"Ca_HVA {config.gate} under held voltage only; "
            "no coupled-voltage or embedded-neuron claim"
        ),
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "sealed_test_opened.json").write_text(json.dumps({
        "selection_freeze_sha256": claimed_hash,
        "final_report_sha256": _file_sha256(output_dir / "final_report.json"),
        "opened_once": True,
    }, indent=2), encoding="utf-8")
    return report
