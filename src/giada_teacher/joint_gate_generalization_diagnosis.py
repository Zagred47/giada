"""Task 3d: development-only causal matrix for joint m+h generalization.

The opened Task 3c fresh set is deliberately not an input.  Shared candidates
are trained as one batched parameter ensemble (arm x seed) on one GPU.  The
independent-trunk control is a second, seed-vectorized ensemble.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest, paired_index_generator
from .joint_gate_cell_playground import JointGateCellConfig, _constructors, _development_score, _evaluate, _joint_row
from .joint_gate_optimization_diagnosis import augment_joint_gate_rate_targets
from .joint_gate_symmetric_confirmation import _batched_model, _clip_per_seed, _single_states


@dataclass(frozen=True)
class JointGateGeneralizationDiagnosisConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    shared_arms: tuple[str, ...] = (
        "baseline_current",
        "same_support_dense",
        "balanced_support",
        "expanded_support",
        "multihorizon",
        "shape_constrained",
        "expanded_multihorizon",
        "full_repair",
    )
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000, 30000, 50000)
    learning_rate: float = 0.003
    weight_decay: float = 1e-6
    batch_size: int = 768
    width: int = 23
    rate_inf_weight: float = 0.1
    rate_log_tau_weight: float = 0.01
    shape_weight: float = 0.02
    gradient_clip_norm: float = 1.0
    pool_size: int = 24000
    probe_count: int = 48
    material_improvement_fraction: float = 0.20
    compile_model: bool = True

    def validate(self) -> None:
        expected = {
            "baseline_current", "same_support_dense", "balanced_support",
            "expanded_support", "multihorizon", "shape_constrained",
            "expanded_multihorizon", "full_repair",
        }
        if set(self.shared_arms) != expected:
            raise ValueError("Task 3d requires the preregistered eight-arm matrix")
        if self.checkpoints[0] != 0 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted, unique and start at zero")
        if self.checkpoints[-1] != 50000 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Task 3d freezes three seeds at 50k")
        if min(self.batch_size, self.pool_size, self.probe_count, self.width) <= 0:
            raise ValueError("invalid Task 3d configuration")


def _rates(formula, voltage):
    rows = [formula.rates(float(v)) for v in voltage]
    return {
        "m_inf": np.asarray([r["m_inf"] for r in rows]),
        "m_tau_ms": np.asarray([r["m_tau_ms"] for r in rows]),
        "h_inf": np.asarray([r["h_inf"] for r in rows]),
        "h_tau_ms": np.asarray([r["h_tau_ms"] for r in rows]),
    }


def _random_row(formula, *, seed, count, voltage_ranges, dt_values, role, axis):
    rng = np.random.default_rng(seed)
    ranges = np.asarray(voltage_ranges, dtype=np.float64)
    band = rng.integers(0, len(ranges), size=count)
    voltage = rng.uniform(ranges[band, 0], ranges[band, 1])
    m = rng.uniform(0.0, 1.0, size=count); h = rng.uniform(0.0, 1.0, size=count)
    dt = rng.choice(np.asarray(dt_values, dtype=np.float64), size=count)
    return _joint_row(formula, np.stack((voltage, m, h, dt), -1), role, axis)


def _enrich(row, formula):
    row = dict(row); row.update(_rates(formula, row["inputs"][:, 0])); return row


def prepare_joint_gate_generalization_diagnosis(formula, base_bundle, config=None):
    config = config or JointGateGeneralizationDiagnosisConfig(); config.validate()
    base_bundle = augment_joint_gate_rate_targets(base_bundle)
    current = _enrich(base_bundle["fit"], formula)
    current_range = ((-100.0, 40.0),)
    balanced_ranges = ((-100.0, -70.0), (-70.0, -35.0), (-35.0, 0.0), (0.0, 40.0))
    expanded_ranges = ((-135.0, -100.0), *balanced_ranges, (40.0, 75.0))
    dense = _enrich(_random_row(formula, seed=73101, count=config.pool_size, voltage_ranges=current_range,
                                dt_values=(0.025, 0.1, 0.2, 0.5, 0.8, 1.0), role="fit", axis="same_support_dense"), formula)
    balanced = _enrich(_random_row(formula, seed=73102, count=config.pool_size, voltage_ranges=balanced_ranges,
                                   dt_values=(0.025, 0.1, 0.2, 0.5, 0.8, 1.0), role="fit", axis="balanced_support"), formula)
    expanded = _enrich(_random_row(formula, seed=73103, count=config.pool_size, voltage_ranges=expanded_ranges,
                                   dt_values=(0.025, 0.1, 0.2, 0.5, 0.8, 1.0), role="fit", axis="expanded_support"), formula)
    multihorizon = _enrich(_random_row(formula, seed=73104, count=config.pool_size, voltage_ranges=balanced_ranges,
                                       dt_values=(0.025, 0.1, 0.5, 1.0, 5.0, 25.0, 100.0), role="fit", axis="multihorizon"), formula)
    expanded_multi = _enrich(_random_row(formula, seed=73105, count=config.pool_size, voltage_ranges=expanded_ranges,
                                         dt_values=(0.025, 0.1, 0.5, 1.0, 5.0, 25.0, 100.0), role="fit", axis="expanded_multihorizon"), formula)
    pools = {
        "baseline_current": current, "same_support_dense": dense,
        "balanced_support": balanced, "expanded_support": expanded,
        "multihorizon": multihorizon, "shape_constrained": balanced,
        "expanded_multihorizon": expanded_multi, "full_repair": expanded_multi,
    }
    development = {
        "central_interpolation": _enrich(_random_row(formula, seed=73201, count=4096, voltage_ranges=balanced_ranges,
            dt_values=(0.075, 0.35, 0.9), role="development", axis="central"), formula),
        "voltage_tail_stress": _enrich(_random_row(formula, seed=73202, count=2048,
            voltage_ranges=((-132.5, -102.5), (42.5, 72.5)), dt_values=(0.075, 0.35, 0.9),
            role="development", axis="voltage_tail"), formula),
        "long_horizon_stress": _enrich(_random_row(formula, seed=73203, count=2048, voltage_ranges=expanded_ranges,
            dt_values=(10.0, 50.0, 100.0), role="development", axis="long_horizon"), formula),
    }
    probe_v = np.linspace(-140.0, 80.0, config.probe_count, dtype=np.float64)
    contract = {
        "schema_version": "giada-task3d-development-contract-v1",
        "fit_counts": {name: int(len(row["inputs"])) for name, row in pools.items()},
        "development_counts": {name: int(len(row["inputs"])) for name, row in development.items()},
        "fresh_task3c_accessed": False, "fresh_used_for_selection": False,
        "all_targets_from_extracted_teacher_formula": True,
        "causal_axes": ["support_density", "support_balance", "voltage_extent", "temporal_horizon", "shape_prior", "topology", "budget"],
    }
    return {"formula": formula, "pools": pools, "development": development, "probe_voltage": probe_v, "contract": contract}


def _tensor_row(row, torch, device):
    names = ("inputs", "m_target", "h_target", "m_inf", "m_tau_ms", "h_inf", "h_tau_ms")
    return {name: torch.as_tensor(row[name], dtype=torch.float32, device=device) for name in names}


def _batch_loss(pred, details, batch, arm_names, torch, config, model):
    mean = lambda x: x.mean(dim=1)
    loss = mean((pred[..., 0] - batch["m_target"]) ** 2) + mean((pred[..., 1] - batch["h_target"]) ** 2)
    loss = loss + config.rate_inf_weight * (mean((details["m_inf"] - batch["m_inf"]) ** 2) + mean((details["h_inf"] - batch["h_inf"]) ** 2))
    loss = loss + config.rate_log_tau_weight * (
        mean((torch.log(details["m_tau_ms"]) - torch.log(batch["m_tau_ms"])) ** 2)
        + mean((torch.log(details["h_tau_ms"]) - torch.log(batch["h_tau_ms"])) ** 2)
    )
    shape_indices = [i for i, name in enumerate(arm_names) if name in {"shape_constrained", "full_repair"}]
    if shape_indices:
        probe = batch["shape_probe"]
        _, rates = model(probe)
        m_wrong = torch.relu(rates["m_inf"][:, :-1] - rates["m_inf"][:, 1:]).mean(1)
        h_wrong = torch.relu(rates["h_inf"][:, 1:] - rates["h_inf"][:, :-1]).mean(1)
        mask = torch.zeros_like(loss); mask[shape_indices] = 1.0
        loss = loss + config.shape_weight * mask * (m_wrong + h_wrong)
    return loss.sum(), loss


def _evaluate_candidate(model, development, torch, device):
    metrics = _evaluate(model, development.items(), torch, device)
    central = _development_score({"central_interpolation": metrics["central_interpolation"]})
    tail = _development_score({"voltage_tail_stress": metrics["voltage_tail_stress"]})
    long = _development_score({"long_horizon_stress": metrics["long_horizon_stress"]})
    return {"score": max(central, tail, long), "central_score": central, "tail_score": tail,
            "long_horizon_score": long, "metrics": metrics}


def _train_shared_matrix(bundle, config, torch, device, progress):
    arms = config.shared_arms; labels = tuple(f"{arm}|seed={seed}" for arm in arms for seed in config.seeds)
    constructor, base_states = _single_states(torch, type("C", (), {**asdict(config), "seeds": config.seeds})())
    states = [base_states[i] for _arm in arms for i in range(len(config.seeds))]
    model = _batched_model(torch, states).to(device); training_model = model
    if config.compile_model:
        try: training_model = torch.compile(model, dynamic=False)
        except Exception: training_model = model
    pools = {name: _tensor_row(row, torch, device) for name, row in bundle["pools"].items()}
    streams = [paired_index_generator(len(bundle["pools"][arm]["inputs"]), config.batch_size, 810000 + 101*a + seed)
               for a, arm in enumerate(arms) for seed in config.seeds]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    probe = torch.as_tensor(np.stack((bundle["probe_voltage"], np.full(config.probe_count, .5),
        np.full(config.probe_count, .5), np.ones(config.probe_count)), -1), dtype=torch.float32, device=device)
    checkpoints = []; maximum = config.checkpoints[-1]; started = time.perf_counter()
    for step in range(maximum + 1):
        if step in config.checkpoints:
            rows = []
            for index, label in enumerate(labels):
                candidate = constructor().to(device); candidate.load_state_dict(model.single_state(index))
                rows.append({"candidate": label, **_evaluate_candidate(candidate, bundle["development"], torch, device)})
            checkpoints.append({"step": step, "candidates": rows})
            progress(step, maximum, time.perf_counter() - started, rows)
        if step == maximum: break
        batches = []
        for index, label in enumerate(labels):
            arm = label.split("|", 1)[0]; idx = torch.as_tensor(next(streams[index]), dtype=torch.long, device=device)
            batches.append({name: value.index_select(0, idx) for name, value in pools[arm].items()})
        batch = {name: torch.stack([row[name] for row in batches], 0) for name in batches[0]}
        batch["shape_probe"] = probe.unsqueeze(0).expand(len(labels), -1, -1)
        optimizer.zero_grad(set_to_none=True); pred, details = training_model(batch["inputs"])
        loss, _ = _batch_loss(pred, details, batch, tuple(label.split("|", 1)[0] for label in labels), torch, config, training_model)
        if not bool(torch.isfinite(loss).detach().cpu()): raise RuntimeError(f"non-finite shared loss at step {step}")
        loss.backward(); _clip_per_seed(model, torch, config.gradient_clip_norm); optimizer.step()
    return model, constructor, labels, checkpoints, time.perf_counter() - started


def _batched_independent_model(torch, states):
    nn = torch.nn
    class BatchedIndependent(nn.Module):
        def __init__(self):
            super().__init__()
            stack = lambda name: torch.stack([row[name] for row in states], 0)
            for prefix in ("mt", "ht"):
                setattr(self, prefix + "w1", nn.Parameter(stack(prefix + ".0.weight")))
                setattr(self, prefix + "b1", nn.Parameter(stack(prefix + ".0.bias")))
                setattr(self, prefix + "w2", nn.Parameter(stack(prefix + ".2.weight")))
                setattr(self, prefix + "b2", nn.Parameter(stack(prefix + ".2.bias")))
            for prefix in ("mh", "hh"):
                setattr(self, prefix + "iw", nn.Parameter(stack(prefix + ".inf.weight").squeeze(1)))
                setattr(self, prefix + "ib", nn.Parameter(stack(prefix + ".inf.bias").squeeze(-1)))
                setattr(self, prefix + "tw", nn.Parameter(stack(prefix + ".tau.weight").squeeze(1)))
                setattr(self, prefix + "tb", nn.Parameter(stack(prefix + ".tau.bias").squeeze(-1)))
        def _features(self, x, prefix):
            f = torch.nn.functional.silu(torch.einsum("sbi,soi->sbo", x, getattr(self, prefix + "w1")) + getattr(self, prefix + "b1")[:, None, :])
            return torch.nn.functional.silu(torch.einsum("sbi,soi->sbo", f, getattr(self, prefix + "w2")) + getattr(self, prefix + "b2")[:, None, :])
        def _head(self, f, prefix):
            linear = lambda w, b: torch.einsum("sbw,sw->sb", f, getattr(self, w)) + getattr(self, b)[:, None]
            return torch.sigmoid(linear(prefix + "iw", prefix + "ib")), torch.nn.functional.softplus(linear(prefix + "tw", prefix + "tb")) + 1e-5
        def forward(self, values):
            voltage, m, h, dt = values.unbind(-1); x = ((voltage + 30.0) / 90.0).unsqueeze(-1)
            mi, mt = self._head(self._features(x, "mt"), "mh"); hi, ht = self._head(self._features(x, "ht"), "hh")
            mz = -torch.expm1(-dt / mt); hz = -torch.expm1(-dt / ht)
            mp = (1.0 - mz) * m + mz * mi; hp = (1.0 - hz) * h + hz * hi
            return torch.stack((mp, hp), -1), {"m_inf": mi, "m_tau_ms": mt, "h_inf": hi, "h_tau_ms": ht, "open": mp.square() * hp}
        def single_state(self, i):
            out = {}
            for prefix in ("mt", "ht"):
                out[prefix + ".0.weight"] = getattr(self, prefix + "w1")[i].detach().clone(); out[prefix + ".0.bias"] = getattr(self, prefix + "b1")[i].detach().clone()
                out[prefix + ".2.weight"] = getattr(self, prefix + "w2")[i].detach().clone(); out[prefix + ".2.bias"] = getattr(self, prefix + "b2")[i].detach().clone()
            for prefix in ("mh", "hh"):
                out[prefix + ".inf.weight"] = getattr(self, prefix + "iw")[i].detach().clone().unsqueeze(0); out[prefix + ".inf.bias"] = getattr(self, prefix + "ib")[i].detach().clone().reshape(1)
                out[prefix + ".tau.weight"] = getattr(self, prefix + "tw")[i].detach().clone().unsqueeze(0); out[prefix + ".tau.bias"] = getattr(self, prefix + "tb")[i].detach().clone().reshape(1)
            return out
    return BatchedIndependent()


def _train_independent_control(bundle, config, torch, device):
    base_config = JointGateCellConfig(seeds=config.seeds, private_width=16, matched_width=config.width)
    constructor = _constructors(torch, base_config)["independent"]
    states = []
    for seed in config.seeds:
        configure_torch_runtime(seed); states.append({k: v.detach().clone() for k, v in constructor().state_dict().items()})
    model = _batched_independent_model(torch, states).to(device); training_model = model
    if config.compile_model:
        try: training_model = torch.compile(model, dynamic=False)
        except Exception: training_model = model
    pool = _tensor_row(bundle["pools"]["full_repair"], torch, device)
    streams = [paired_index_generator(len(bundle["pools"]["full_repair"]["inputs"]), config.batch_size, 830000 + seed) for seed in config.seeds]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    probe = torch.as_tensor(np.stack((bundle["probe_voltage"], np.full(config.probe_count, .5), np.full(config.probe_count, .5), np.ones(config.probe_count)), -1), dtype=torch.float32, device=device)
    checkpoints = []; started = time.perf_counter(); maximum = config.checkpoints[-1]
    for step in range(maximum + 1):
        if step in config.checkpoints:
            rows = []
            for i, seed in enumerate(config.seeds):
                candidate = constructor().to(device); candidate.load_state_dict(model.single_state(i))
                rows.append({"candidate": f"full_repair_independent|seed={seed}", **_evaluate_candidate(candidate, bundle["development"], torch, device)})
            checkpoints.append({"step": step, "candidates": rows})
            elapsed = time.perf_counter() - started; eta = elapsed / max(step, 1) * (maximum - step)
            print(f"[GIADA Task 3d][independent {len(rows)} candidates] {step}/{maximum} ({100*step/maximum:.1f}%) ETA {eta/60:.1f} min mean={np.mean([r['score'] for r in rows]):.3g}")
        if step == maximum: break
        batches = []
        for stream in streams:
            idx = torch.as_tensor(next(stream), dtype=torch.long, device=device); batches.append({name: value.index_select(0, idx) for name, value in pool.items()})
        batch = {name: torch.stack([row[name] for row in batches], 0) for name in batches[0]}; batch["shape_probe"] = probe.unsqueeze(0).expand(len(config.seeds), -1, -1)
        optimizer.zero_grad(set_to_none=True); pred, details = training_model(batch["inputs"])
        loss, _ = _batch_loss(pred, details, batch, tuple("full_repair" for _ in config.seeds), torch, config, training_model)
        if not bool(torch.isfinite(loss).detach().cpu()): raise RuntimeError(f"non-finite independent loss at step {step}")
        loss.backward(); _clip_per_seed(model, torch, config.gradient_clip_norm); optimizer.step()
    return model, checkpoints, time.perf_counter() - started


def _train_wide_control(bundle, config, torch, device):
    wide_width = 46
    fake = type("WideConfig", (), {"seeds": config.seeds, "width": wide_width})()
    constructor, states = _single_states(torch, fake)
    model = _batched_model(torch, states).to(device); training_model = model
    if config.compile_model:
        try: training_model = torch.compile(model, dynamic=False)
        except Exception: training_model = model
    pool = _tensor_row(bundle["pools"]["full_repair"], torch, device)
    streams = [paired_index_generator(len(bundle["pools"]["full_repair"]["inputs"]), config.batch_size, 840000 + seed) for seed in config.seeds]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    probe = torch.as_tensor(np.stack((bundle["probe_voltage"], np.full(config.probe_count, .5), np.full(config.probe_count, .5), np.ones(config.probe_count)), -1), dtype=torch.float32, device=device)
    checkpoints = []; started = time.perf_counter(); maximum = config.checkpoints[-1]
    for step in range(maximum + 1):
        if step in config.checkpoints:
            rows = []
            for i, seed in enumerate(config.seeds):
                candidate = constructor().to(device); candidate.load_state_dict(model.single_state(i))
                rows.append({"candidate": f"full_repair_wide|seed={seed}", **_evaluate_candidate(candidate, bundle["development"], torch, device)})
            checkpoints.append({"step": step, "candidates": rows})
            elapsed = time.perf_counter() - started; eta = elapsed / max(step, 1) * (maximum - step)
            print(f"[GIADA Task 3d][wide {len(rows)} candidates] {step}/{maximum} ({100*step/maximum:.1f}%) ETA {eta/60:.1f} min mean={np.mean([r['score'] for r in rows]):.3g}")
        if step == maximum: break
        batches = []
        for stream in streams:
            idx = torch.as_tensor(next(stream), dtype=torch.long, device=device); batches.append({name: value.index_select(0, idx) for name, value in pool.items()})
        batch = {name: torch.stack([row[name] for row in batches], 0) for name in batches[0]}; batch["shape_probe"] = probe.unsqueeze(0).expand(len(config.seeds), -1, -1)
        optimizer.zero_grad(set_to_none=True); pred, details = training_model(batch["inputs"])
        loss, _ = _batch_loss(pred, details, batch, tuple("full_repair" for _ in config.seeds), torch, config, training_model)
        if not bool(torch.isfinite(loss).detach().cpu()): raise RuntimeError(f"non-finite wide loss at step {step}")
        loss.backward(); _clip_per_seed(model, torch, config.gradient_clip_norm); optimizer.step()
    return model, checkpoints, time.perf_counter() - started


def run_joint_gate_generalization_diagnosis(bundle, output_dir, config=None, *, code_revision="unknown"):
    config = config or JointGateGeneralizationDiagnosisConfig(); config.validate()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    torch = configure_torch_runtime(config.seeds[0]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Task 3d requires CUDA")
    def progress(step, maximum, elapsed, rows):
        eta = elapsed / max(step, 1) * (maximum - step)
        scores = [r["score"] for r in rows]
        print(f"[GIADA Task 3d][shared {len(rows)} candidates] {step}/{maximum} ({100*step/maximum:.1f}%) ETA {eta/60:.1f} min score[min/median/max]={min(scores):.3g}/{np.median(scores):.3g}/{max(scores):.3g}")
    model, constructor, labels, checkpoints, elapsed = _train_shared_matrix(bundle, config, torch, device, progress)
    independent_model, independent_checkpoints, independent_elapsed = _train_independent_control(bundle, config, torch, device)
    wide_model, wide_checkpoints, wide_elapsed = _train_wide_control(bundle, config, torch, device)
    final = checkpoints[-1]["candidates"]
    summaries = {}
    for arm in config.shared_arms:
        rows = [r for r in final if r["candidate"].startswith(arm + "|")]
        summaries[arm] = {key: float(np.mean([r[key] for r in rows])) for key in ("score", "central_score", "tail_score", "long_horizon_score")}
        summaries[arm]["seed_scores"] = {r["candidate"].split("=")[-1]: r["score"] for r in rows}
    baseline = summaries["baseline_current"]["score"]
    if not np.isfinite([row["score"] for row in summaries.values()]).all() or baseline <= 0:
        raise RuntimeError("Task 3d produced non-finite or invalid development scores")
    effects = {arm: float((baseline - row["score"]) / baseline) for arm, row in summaries.items() if arm != "baseline_current"}
    independent_rows = independent_checkpoints[-1]["candidates"]
    summaries["full_repair_independent"] = {
        key: float(np.mean([r[key] for r in independent_rows]))
        for key in ("score", "central_score", "tail_score", "long_horizon_score")
    }
    summaries["full_repair_independent"]["seed_scores"] = {r["candidate"].split("=")[-1]: r["score"] for r in independent_rows}
    wide_rows = wide_checkpoints[-1]["candidates"]
    summaries["full_repair_wide"] = {
        key: float(np.mean([r[key] for r in wide_rows]))
        for key in ("score", "central_score", "tail_score", "long_horizon_score")
    }
    summaries["full_repair_wide"]["seed_scores"] = {r["candidate"].split("=")[-1]: r["score"] for r in wide_rows}
    axes = {
        "more_same_support": effects["same_support_dense"],
        "balanced_sampling": effects["balanced_support"],
        "expanded_voltage_support": effects["expanded_support"],
        "temporal_horizon": effects["multihorizon"],
        "shape_prior": effects["shape_constrained"],
        "support_horizon_interaction": effects["expanded_multihorizon"],
        "full_repair": effects["full_repair"],
        "independent_topology_over_full_repair_shared": float((summaries["full_repair"]["score"] - summaries["full_repair_independent"]["score"]) / summaries["full_repair"]["score"]),
        "capacity_wide_over_full_repair_shared": float((summaries["full_repair"]["score"] - summaries["full_repair_wide"]["score"]) / summaries["full_repair"]["score"]),
    }
    supported = {name: value >= config.material_improvement_fraction for name, value in axes.items()}
    best = min(summaries, key=lambda name: summaries[name]["score"])
    report = {
        "schema_version": "giada-task3d-generalization-diagnosis-v1", "valid": True,
        "code_revision": str(code_revision), "environment": environment_manifest(torch),
        "config": asdict(config), "data_contract": bundle["contract"],
        "execution": {"parallel_shared_candidates": len(labels), "parallel_independent_candidates": len(config.seeds),
                      "parallel_wide_candidates": len(config.seeds), "shared_elapsed_seconds": elapsed,
                      "independent_elapsed_seconds": independent_elapsed, "wide_elapsed_seconds": wide_elapsed,
                      "candidate_steps_per_second": (len(labels) + 2 * len(config.seeds)) * config.checkpoints[-1] / (elapsed + independent_elapsed + wide_elapsed)},
        "checkpoints": {"shared": checkpoints, "independent": independent_checkpoints, "wide": wide_checkpoints}, "summaries": summaries, "causal_effect_fractions": axes,
        "supported_causes": supported, "best_development_arm": best,
        "fresh_task3c_accessed": False, "task4_authorized": False,
        "next_step": "Preregister only the supported minimal repair, then confirm it on a newly sealed independent set.",
    }
    saved = {label: model.single_state(i) for i, label in enumerate(labels)}
    saved.update({f"full_repair_independent|seed={seed}": independent_model.single_state(i) for i, seed in enumerate(config.seeds)})
    saved.update({f"full_repair_wide|seed={seed}": wide_model.single_state(i) for i, seed in enumerate(config.seeds)})
    torch.save(saved, output_dir / "development_checkpoints.pt")
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
