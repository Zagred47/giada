"""Task 3b: causal diagnosis of shared Ca_HVA m+h optimization."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_gate_playground import _state_dict_cpu
from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest, paired_index_generator
from .joint_gate_cell_playground import (
    JointGateCellConfig,
    _constructors,
    _development_score,
    _evaluate,
)


@dataclass(frozen=True)
class JointGateOptimizationDiagnosisConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    arms: tuple[str, ...] = (
        "baseline_extended",
        "symmetric_rates",
        "pcgrad",
        "symmetric_rates_pcgrad",
        "rate_pretrain",
    )
    learning_rate: float = 0.003
    batch_size: int = 1024
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000, 30000, 50000)
    baseline_extended_steps: int = 100000
    pretrain_steps: int = 5000
    rate_inf_weight: float = 0.1
    rate_log_tau_weight: float = 0.01
    diagnostic_batches_per_checkpoint: int = 3
    material_improvement_fraction: float = 0.20

    def validate(self) -> None:
        expected = {
            "baseline_extended", "symmetric_rates", "pcgrad",
            "symmetric_rates_pcgrad", "rate_pretrain",
        }
        if set(self.arms) != expected:
            raise ValueError("Task 3b requires the registered five-arm diagnosis")
        if self.checkpoints[0] != 0 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted, unique and start at zero")
        if self.baseline_extended_steps <= self.checkpoints[-1] or self.pretrain_steps <= 0:
            raise ValueError("invalid registered budget contrast")
        if min(self.seeds) < 0 or min(self.learning_rate, self.batch_size) <= 0:
            raise ValueError("invalid optimization configuration")


def augment_joint_gate_rate_targets(bundle: dict[str, Any]) -> dict[str, Any]:
    """Add exact m rate labels without changing any tuple or existing target."""
    formula = bundle["formula"]
    for group in (bundle["fit"], *bundle["development"].values()):
        rates = [formula.rates(float(v)) for v in group["inputs"][:, 0]]
        group["m_inf"] = np.asarray([row["m_inf"] for row in rates], dtype=np.float64)
        group["m_tau_ms"] = np.asarray([row["m_tau_ms"] for row in rates], dtype=np.float64)
    return bundle


def _arm_properties(arm: str, step: int, config: JointGateOptimizationDiagnosisConfig) -> dict[str, bool]:
    return {
        "symmetric_rates": arm in {"symmetric_rates", "symmetric_rates_pcgrad", "rate_pretrain"},
        "pcgrad": arm in {"pcgrad", "symmetric_rates_pcgrad"},
        "rate_only": arm == "rate_pretrain" and step < config.pretrain_steps,
    }


def _losses(details, pred, target, index, tensors, props, torch, config):
    mse = torch.mean
    m_endpoint = mse((pred[:, 0] - target[:, 0]) ** 2)
    h_endpoint = mse((pred[:, 1] - target[:, 1]) ** 2)
    m_rate = config.rate_inf_weight * mse((details["m_inf"] - tensors["m_inf"].index_select(0, index)) ** 2)
    m_rate = m_rate + config.rate_log_tau_weight * mse(
        (torch.log(details["m_tau_ms"]) - torch.log(tensors["m_tau"].index_select(0, index))) ** 2
    )
    h_rate = config.rate_inf_weight * mse((details["h_inf"] - tensors["h_inf"].index_select(0, index)) ** 2)
    h_rate = h_rate + config.rate_log_tau_weight * mse(
        (torch.log(details["h_tau_ms"]) - torch.log(tensors["h_tau"].index_select(0, index))) ** 2
    )
    m_loss = m_rate if props["rate_only"] else m_endpoint + (m_rate if props["symmetric_rates"] else 0.0)
    h_loss = h_rate if props["rate_only"] else h_endpoint + h_rate
    return m_loss, h_loss


def _gradient_geometry(m_loss, h_loss, parameters, torch, retain_graph=True):
    gm = torch.autograd.grad(m_loss, parameters, retain_graph=True, allow_unused=True)
    gh = torch.autograd.grad(h_loss, parameters, retain_graph=retain_graph, allow_unused=True)
    pairs = [(a, b) for a, b in zip(gm, gh) if a is not None and b is not None]
    dot = sum((a * b).sum() for a, b in pairs)
    nm = sum((a * a).sum() for a, _ in pairs)
    nh = sum((b * b).sum() for _, b in pairs)
    cosine = dot / torch.sqrt(nm * nh + 1e-30)
    return gm, gh, dot, nm, nh, cosine


def _apply_pcgrad(model, m_loss, h_loss, torch):
    shared = tuple(model.trunk.parameters())
    gm, gh, dot, nm, nh, cosine = _gradient_geometry(m_loss, h_loss, shared, torch)
    (m_loss + h_loss).backward()
    negative = torch.minimum(dot, torch.zeros_like(dot))
    for parameter, a, b in zip(shared, gm, gh):
        if a is not None and b is not None:
            parameter.grad = a - negative * b / (nh + 1e-30) + b - negative * a / (nm + 1e-30)
    return float(cosine.detach().cpu()), bool(dot.detach().cpu() < 0)


def _diagnose_gradient(model, x, targets, tensors, stream, torch, device, props, config):
    shared = tuple(model.trunk.parameters()); cosines = []
    for _ in range(config.diagnostic_batches_per_checkpoint):
        index = torch.as_tensor(next(stream), dtype=torch.long, device=device)
        batch = x.index_select(0, index); target = targets.index_select(0, index)
        pred, details = model(batch, diagnostics=True)
        m_loss, h_loss = _losses(details, pred, target, index, tensors, props, torch, config)
        _, _, dot, _, _, cosine = _gradient_geometry(m_loss, h_loss, shared, torch, retain_graph=False)
        cosines.append((float(cosine.detach().cpu()), bool(dot.detach().cpu() < 0)))
    return {
        "mean_cosine": float(np.mean([row[0] for row in cosines])),
        "conflict_fraction": float(np.mean([row[1] for row in cosines])),
    }


def run_joint_gate_optimization_diagnosis(bundle, output_dir, config=None, *, code_revision="unknown"):
    config = config or JointGateOptimizationDiagnosisConfig(); config.validate()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    bundle = augment_joint_gate_rate_targets(bundle)
    torch = configure_torch_runtime(config.seeds[0]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = bundle["fit"]
    x = torch.as_tensor(fit["inputs"], dtype=torch.float32, device=device)
    targets = torch.as_tensor(np.stack((fit["m_target"], fit["h_target"]), -1), dtype=torch.float32, device=device)
    tensors = {
        key: torch.as_tensor(fit[source], dtype=torch.float32, device=device)
        for key, source in (("m_inf", "m_inf"), ("m_tau", "m_tau_ms"), ("h_inf", "h_inf"), ("h_tau", "h_tau_ms"))
    }
    constructor = _constructors(torch, JointGateCellConfig())["shared_matched"]
    runs = []; snapshots = {}; total = len(config.seeds) * (config.baseline_extended_steps + (len(config.arms) - 1) * config.checkpoints[-1])
    completed = 0; started = time.perf_counter()
    for arm in config.arms:
        maximum = config.baseline_extended_steps if arm == "baseline_extended" else config.checkpoints[-1]
        arm_checkpoints = tuple(sorted(set(config.checkpoints + ((maximum,) if maximum not in config.checkpoints else ()))))
        for seed in config.seeds:
            configure_torch_runtime(seed); model = constructor().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
            stream = paired_index_generator(len(fit["inputs"]), config.batch_size, seed + 400000)
            diagnostic_stream = paired_index_generator(len(fit["inputs"]), config.batch_size, seed + 940000)
            checkpoints = []; conflict_steps = 0
            for step in range(maximum + 1):
                props = _arm_properties(arm, step, config)
                if step in arm_checkpoints:
                    metrics = _evaluate(model, bundle["development"].items(), torch, device)
                    score = _development_score(metrics)
                    geometry = _diagnose_gradient(model, x, targets, tensors, diagnostic_stream, torch, device, props, config)
                    key = f"{arm}-seed{seed}-step{step}"; snapshots[key] = _state_dict_cpu(model, torch)
                    checkpoints.append({"step": step, "score": score, "metrics": metrics, "gradient_geometry": geometry, "checkpoint_key": key})
                    progress = completed + step; elapsed = time.perf_counter() - started
                    eta = elapsed / max(progress, 1) * max(total - progress, 0)
                    print(f"[GIADA Task 3b] {100*progress/total:.1f}% ETA {eta/60:.1f} min {arm} seed={seed} step={step} score={score:.4g} cos={geometry['mean_cosine']:.3f}")
                if step == maximum: break
                index = torch.as_tensor(next(stream), dtype=torch.long, device=device)
                batch = x.index_select(0, index); target = targets.index_select(0, index)
                optimizer.zero_grad(set_to_none=True); pred, details = model(batch, diagnostics=True)
                m_loss, h_loss = _losses(details, pred, target, index, tensors, props, torch, config)
                if props["pcgrad"]:
                    _, conflicted = _apply_pcgrad(model, m_loss, h_loss, torch); conflict_steps += int(conflicted)
                else:
                    (m_loss + h_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            completed += maximum
            runs.append({
                "arm": arm, "seed": seed, "parameter_count": sum(p.numel() for p in model.parameters()),
                "trained_steps": maximum, "pcgrad_conflict_step_count": conflict_steps, "checkpoints": checkpoints,
            })
    torch.save(snapshots, output_dir / "diagnostic_checkpoints.pt")
    report = _finalize_diagnosis(runs, config)
    report.update({
        "schema_version": "giada-task3b-optimization-diagnosis-v1", "valid": True,
        "environment": environment_manifest(torch), "config": asdict(config), "runs": runs,
        "code_revision": str(code_revision), "fresh_accessed": False, "task4_authorized": False,
    })
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _mean_score(runs, arm, step):
    values = [next(c["score"] for c in row["checkpoints"] if c["step"] == step) for row in runs if row["arm"] == arm]
    return float(np.mean(values))


def _finalize_diagnosis(runs, config):
    at50 = {arm: _mean_score(runs, arm, 50000) for arm in config.arms}
    baseline50 = at50["baseline_extended"]; baseline100 = _mean_score(runs, "baseline_extended", config.baseline_extended_steps)
    improvement = lambda score: float((baseline50 - score) / baseline50)
    effects = {
        "symmetric_rate_supervision": improvement(at50["symmetric_rates"]),
        "pcgrad": improvement(at50["pcgrad"]),
        "interaction": improvement(at50["symmetric_rates_pcgrad"]),
        "rate_pretraining_vs_symmetric": float((at50["symmetric_rates"] - at50["rate_pretrain"]) / at50["symmetric_rates"]),
        "extended_budget": float((baseline50 - baseline100) / baseline50),
    }
    threshold = config.material_improvement_fraction
    supported = {name: value >= threshold for name, value in effects.items()}
    baseline_geometry = [c["gradient_geometry"] for row in runs if row["arm"] == "baseline_extended" for c in row["checkpoints"] if c["step"] in (1000, 3000, 10000)]
    observed_conflict = float(np.mean([row["conflict_fraction"] for row in baseline_geometry]))
    if supported["interaction"] and not supported["symmetric_rate_supervision"] and not supported["pcgrad"]:
        diagnosis = "RATE_INFORMATION_AND_GRADIENT_INTERFERENCE_INTERACTION"
    elif supported["symmetric_rate_supervision"]:
        diagnosis = "OBJECTIVE_INFORMATION_IMBALANCE"
    elif supported["pcgrad"] and observed_conflict >= 0.25:
        diagnosis = "SHARED_TRUNK_GRADIENT_INTERFERENCE"
    elif supported["rate_pretraining_vs_symmetric"]:
        diagnosis = "INITIALIZATION_OR_OPTIMIZATION_PATH"
    elif supported["extended_budget"]:
        diagnosis = "TRAINING_BUDGET_LIMIT"
    else:
        diagnosis = "NO_SINGLE_REGISTERED_OPTIMIZATION_CAUSE"
    return {
        "scores_at_50000": at50, "baseline_score_at_100000": baseline100,
        "causal_effect_fractions": effects, "supported_causes": supported,
        "baseline_early_gradient_conflict_fraction": observed_conflict,
        "diagnosis": diagnosis,
        "interpretation_rule": f"a registered cause requires >= {threshold:.0%} score improvement",
        "next_step": "preregister one minimal repair from the supported cause; do not open fresh confirmation in Task 3b",
    }
