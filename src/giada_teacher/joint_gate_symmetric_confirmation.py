"""Task 3c: vectorized multi-seed confirmation of the repaired joint m+h cell."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest, paired_index_generator
from .joint_gate_cell_playground import (
    JointGateCellConfig,
    _constructors,
    _development_score,
    _evaluate,
    _file_sha,
    _joint_row,
    _rollout,
    _sha,
    prepare_joint_gate_dataset,
)
from .joint_gate_optimization_diagnosis import augment_joint_gate_rate_targets


@dataclass(frozen=True)
class JointGateSymmetricConfirmationConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    learning_rate: float = 0.003
    weight_decay: float = 1e-6
    batch_size: int = 1024
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000, 30000, 50000)
    width: int = 23
    rate_inf_weight: float = 0.1
    rate_log_tau_weight: float = 0.01
    gradient_clip_norm: float = 1.0
    equivalence_atol: float = 1e-5
    enable_torch_compile: bool = True
    rollout_horizons: tuple[int, ...] = (10, 100, 1000)
    development_mean_score_max: float = 0.15
    development_max_seed_score_max: float = 0.20

    def validate(self) -> None:
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2:
            raise ValueError("Task 3c requires distinct replicated seeds")
        if self.checkpoints[0] != 0 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted, unique and start at zero")
        if self.checkpoints[-1] != 50000 or self.width != 23:
            raise ValueError("Task 3c freezes the registered 50k width-23 repair")
        if min(self.learning_rate, self.batch_size, self.rate_inf_weight, self.rate_log_tau_weight) <= 0:
            raise ValueError("invalid Task 3c optimization configuration")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _new_fresh_rows(formula):
    states_m = (0.07, 0.31, 0.61, 0.89, 0.97)
    states_h = (0.04, 0.22, 0.52, 0.82, 0.96)
    make = lambda vs, ds: np.asarray(
        [(v, m, h, d) for v in vs for m in states_m for h in states_h for d in ds],
        dtype=np.float64,
    )
    support_v = tuple(float(v) for v in np.arange(-94.75, 39.0, 4.0))
    return {
        "fresh3c_in_support": _joint_row(formula, make(support_v, (0.2, 0.8)), "fresh3c", "in_support"),
        "fresh3c_ood_voltage": _joint_row(
            formula, make((-121.75, -111.75, 51.75, 61.75), (0.2, 0.8)), "fresh3c", "ood_voltage"
        ),
        "fresh3c_ood_dt": _joint_row(
            formula, make((-89.75, -57.75, -25.75, 34.25), (1.75,)), "fresh3c", "ood_dt"
        ),
    }


def prepare_joint_gate_symmetric_confirmation(formula) -> dict[str, Any]:
    base = augment_joint_gate_rate_targets(prepare_joint_gate_dataset(formula))
    new_fresh = _new_fresh_rows(formula)
    old = {
        tuple(row)
        for group in (base["fit"], *base["development"].values(), *base["fresh"].values())
        for row in group["inputs"]
    }
    new = {tuple(row) for group in new_fresh.values() for row in group["inputs"]}
    overlap = old & new
    if overlap:
        raise RuntimeError(f"Task 3c fresh tuples overlap prior domains: {len(overlap)}")
    base["fresh3c"] = new_fresh
    base["confirmation_contract"] = {
        "schema_version": "giada-task3c-data-v1",
        "new_fresh_counts": {name: len(row["inputs"]) for name, row in new_fresh.items()},
        "overlap_with_task3_or_development": 0,
        "fresh_used_for_training_or_freeze": False,
    }
    return base


def _single_states(torch, config):
    constructor = _constructors(torch, JointGateCellConfig(matched_width=config.width))["shared_matched"]
    states = []
    for seed in config.seeds:
        configure_torch_runtime(seed)
        states.append({name: value.detach().clone() for name, value in constructor().state_dict().items()})
    return constructor, states


def _batched_model(torch, states):
    nn = torch.nn

    class BatchedShared(nn.Module):
        def __init__(self):
            super().__init__()
            stack = lambda name: torch.stack([row[name] for row in states], 0)
            self.w1 = nn.Parameter(stack("trunk.0.weight")); self.b1 = nn.Parameter(stack("trunk.0.bias"))
            self.w2 = nn.Parameter(stack("trunk.2.weight")); self.b2 = nn.Parameter(stack("trunk.2.bias"))
            self.miw = nn.Parameter(stack("mh.inf.weight").squeeze(1)); self.mib = nn.Parameter(stack("mh.inf.bias").squeeze(-1))
            self.mtw = nn.Parameter(stack("mh.tau.weight").squeeze(1)); self.mtb = nn.Parameter(stack("mh.tau.bias").squeeze(-1))
            self.hiw = nn.Parameter(stack("hh.inf.weight").squeeze(1)); self.hib = nn.Parameter(stack("hh.inf.bias").squeeze(-1))
            self.htw = nn.Parameter(stack("hh.tau.weight").squeeze(1)); self.htb = nn.Parameter(stack("hh.tau.bias").squeeze(-1))

        def forward(self, values):
            voltage, m, h, dt = values.unbind(-1)
            x = ((voltage + 30.0) / 90.0).unsqueeze(-1)
            features = torch.nn.functional.silu(torch.einsum("sbi,soi->sbo", x, self.w1) + self.b1[:, None, :])
            features = torch.nn.functional.silu(torch.einsum("sbi,soi->sbo", features, self.w2) + self.b2[:, None, :])
            head = lambda weight, bias: torch.einsum("sbw,sw->sb", features, weight) + bias[:, None]
            mi = torch.sigmoid(head(self.miw, self.mib)); mt = torch.nn.functional.softplus(head(self.mtw, self.mtb)) + 1e-5
            hi = torch.sigmoid(head(self.hiw, self.hib)); ht = torch.nn.functional.softplus(head(self.htw, self.htb)) + 1e-5
            mz = -torch.expm1(-dt / mt); hz = -torch.expm1(-dt / ht)
            mp = (1.0 - mz) * m + mz * mi; hp = (1.0 - hz) * h + hz * hi
            return torch.stack((mp, hp), -1), {
                "m_inf": mi, "m_tau_ms": mt, "h_inf": hi, "h_tau_ms": ht, "open": mp.square() * hp,
            }

        def single_state(self, index):
            return {
                "trunk.0.weight": self.w1[index].detach().clone(), "trunk.0.bias": self.b1[index].detach().clone(),
                "trunk.2.weight": self.w2[index].detach().clone(), "trunk.2.bias": self.b2[index].detach().clone(),
                "mh.inf.weight": self.miw[index].detach().clone().unsqueeze(0), "mh.inf.bias": self.mib[index].detach().clone().reshape(1),
                "mh.tau.weight": self.mtw[index].detach().clone().unsqueeze(0), "mh.tau.bias": self.mtb[index].detach().clone().reshape(1),
                "hh.inf.weight": self.hiw[index].detach().clone().unsqueeze(0), "hh.inf.bias": self.hib[index].detach().clone().reshape(1),
                "hh.tau.weight": self.htw[index].detach().clone().unsqueeze(0), "hh.tau.bias": self.htb[index].detach().clone().reshape(1),
            }

    return BatchedShared()


def _gather_training_tensors(bundle, torch, device):
    fit = bundle["fit"]
    mapping = {
        "inputs": "inputs", "m_target": "m_target", "h_target": "h_target",
        "m_inf": "m_inf", "m_tau": "m_tau_ms", "h_inf": "h_inf", "h_tau": "h_tau_ms",
    }
    return {name: torch.as_tensor(fit[source], dtype=torch.float32, device=device) for name, source in mapping.items()}


def _batch(tensors, indices):
    return {name: tensor[indices] for name, tensor in tensors.items()}


def _symmetric_loss(pred, details, batch, torch, config):
    mean = lambda value: value.mean(dim=1)
    per_seed = mean((pred[..., 0] - batch["m_target"]) ** 2) + mean((pred[..., 1] - batch["h_target"]) ** 2)
    per_seed = per_seed + config.rate_inf_weight * (
        mean((details["m_inf"] - batch["m_inf"]) ** 2) + mean((details["h_inf"] - batch["h_inf"]) ** 2)
    )
    per_seed = per_seed + config.rate_log_tau_weight * (
        mean((torch.log(details["m_tau_ms"]) - torch.log(batch["m_tau"])) ** 2)
        + mean((torch.log(details["h_tau_ms"]) - torch.log(batch["h_tau"])) ** 2)
    )
    return per_seed.sum(), per_seed


def _clip_per_seed(model, torch, maximum):
    squared = None
    for parameter in model.parameters():
        axes = tuple(range(1, parameter.grad.ndim))
        contribution = parameter.grad.square().sum(dim=axes)
        squared = contribution if squared is None else squared + contribution
    norms = torch.sqrt(squared)
    scale = torch.clamp(maximum / (norms + 1e-6), max=1.0)
    for parameter in model.parameters():
        parameter.grad.mul_(scale.reshape((-1,) + (1,) * (parameter.grad.ndim - 1)))
    return norms


def _scalar_loss(model, batch, torch, config):
    pred, details = model(batch["inputs"], diagnostics=True)
    loss = torch.mean((pred[:, 0] - batch["m_target"]) ** 2) + torch.mean((pred[:, 1] - batch["h_target"]) ** 2)
    loss = loss + config.rate_inf_weight * (
        torch.mean((details["m_inf"] - batch["m_inf"]) ** 2) + torch.mean((details["h_inf"] - batch["h_inf"]) ** 2)
    )
    loss = loss + config.rate_log_tau_weight * (
        torch.mean((torch.log(details["m_tau_ms"]) - torch.log(batch["m_tau"])) ** 2)
        + torch.mean((torch.log(details["h_tau_ms"]) - torch.log(batch["h_tau"])) ** 2)
    )
    return loss, pred


def verify_vectorized_equivalence(bundle, config, torch, device):
    constructor, states = _single_states(torch, config)
    vector = _batched_model(torch, states).to(device)
    scalars = [constructor().to(device) for _ in config.seeds]
    for model, state in zip(scalars, states): model.load_state_dict(state)
    tensors = _gather_training_tensors(bundle, torch, device)
    streams = [paired_index_generator(len(bundle["fit"]["inputs"]), config.batch_size, seed + 400000) for seed in config.seeds]
    indices = torch.as_tensor(np.asarray([next(stream) for stream in streams]), dtype=torch.long, device=device)
    batch = _batch(tensors, indices)
    with torch.no_grad():
        vp, _ = vector(batch["inputs"])
        sp = torch.stack([model(batch["inputs"][i]) for i, model in enumerate(scalars)], 0)
        prediction_error = float(torch.max(torch.abs(vp - sp)).cpu())
    vector_optimizer = torch.optim.AdamW(vector.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    vector_optimizer.zero_grad(set_to_none=True); prediction, details = vector(batch["inputs"])
    loss, _ = _symmetric_loss(prediction, details, batch, torch, config); loss.backward()
    _clip_per_seed(vector, torch, config.gradient_clip_norm); vector_optimizer.step()
    for i, model in enumerate(scalars):
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimizer.zero_grad(set_to_none=True)
        scalar_batch = {name: value[i] for name, value in batch.items()}
        scalar_loss, _ = _scalar_loss(model, scalar_batch, torch, config); scalar_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm); optimizer.step()
    state_error = 0.0
    for i, model in enumerate(scalars):
        vector_state = vector.single_state(i)
        state_error = max(state_error, max(float(torch.max(torch.abs(vector_state[name] - value)).cpu()) for name, value in model.state_dict().items()))
    valid = prediction_error <= config.equivalence_atol and state_error <= config.equivalence_atol
    report = {
        "valid": valid, "prediction_max_abs_error": prediction_error,
        "one_step_parameter_max_abs_error": state_error, "atol": config.equivalence_atol,
        "independent_adam_moments": True, "independent_gradient_clipping": True,
    }
    if not valid: raise RuntimeError(f"vectorized seed ensemble failed equivalence preflight: {report}")
    return report

def _development_metrics(vector, constructor, bundle, torch, device, config):
    per_seed = {}; scores = []
    for index, seed in enumerate(config.seeds):
        model = constructor().to(device); model.load_state_dict(vector.single_state(index))
        metrics = _evaluate(model, bundle["development"].items(), torch, device)
        score = _development_score(metrics); scores.append(score)
        per_seed[str(seed)] = {"score": score, "metrics": metrics}
    return {"mean_score": float(np.mean(scores)), "max_seed_score": float(np.max(scores)), "per_seed": per_seed}


def train_vectorized_symmetric_joint_gate(bundle, output_dir, config=None, *, code_revision="unknown"):
    config = config or JointGateSymmetricConfirmationConfig(); config.validate()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    torch = configure_torch_runtime(config.seeds[0]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Task 3c registered execution requires CUDA")
    equivalence = verify_vectorized_equivalence(bundle, config, torch, device)
    constructor, states = _single_states(torch, config); vector = _batched_model(torch, states).to(device)
    training_model = vector; compile_report = {"requested": config.enable_torch_compile, "enabled": False, "fallback_reason": None}
    if config.enable_torch_compile and hasattr(torch, "compile"):
        try:
            training_model = torch.compile(vector, mode="reduce-overhead", fullgraph=True, dynamic=False)
            probe = torch.zeros((len(config.seeds), config.batch_size, 4), dtype=torch.float32, device=device)
            eager, _ = vector(probe); compiled, _ = training_model(probe); torch.cuda.synchronize()
            error = float(torch.max(torch.abs(eager - compiled)).cpu())
            if error > config.equivalence_atol: raise RuntimeError(f"compiled forward error {error}")
            compile_report.update({"enabled": True, "forward_max_abs_error": error})
        except Exception as exc:
            training_model = vector
            compile_report["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    tensors = _gather_training_tensors(bundle, torch, device)
    streams = [paired_index_generator(len(bundle["fit"]["inputs"]), config.batch_size, seed + 400000) for seed in config.seeds]
    optimizer = torch.optim.AdamW(vector.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    checkpoints = []; started = time.perf_counter(); maximum = config.checkpoints[-1]
    for step in range(maximum + 1):
        if step in config.checkpoints:
            metrics = _development_metrics(vector, constructor, bundle, torch, device, config)
            elapsed = time.perf_counter() - started; eta = elapsed / max(step, 1) * max(maximum - step, 0)
            checkpoints.append({"step": step, **metrics})
            print(f"[GIADA Task 3c] {step}/{maximum} ({100*step/maximum:.1f}%) ETA {eta/60:.1f} min mean={metrics['mean_score']:.4g} maxseed={metrics['max_seed_score']:.4g}")
        if step == maximum: break
        indices = torch.as_tensor(np.asarray([next(stream) for stream in streams]), dtype=torch.long, device=device)
        batch = _batch(tensors, indices)
        optimizer.zero_grad(set_to_none=True); prediction, details = training_model(batch["inputs"])
        loss, _ = _symmetric_loss(prediction, details, batch, torch, config); loss.backward()
        _clip_per_seed(vector, torch, config.gradient_clip_norm); optimizer.step()
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    checkpoint_path = output_dir / "frozen_vectorized_checkpoint.pt"
    torch.save({"batched_state": vector.state_dict(), "per_seed_state": {str(seed): vector.single_state(i) for i, seed in enumerate(config.seeds)}}, checkpoint_path)
    final_development = checkpoints[-1]
    freeze = {
        "schema_version": "giada-task3c-freeze-v1", "code_revision": str(code_revision),
        "config": asdict(config), "data_contract": bundle["confirmation_contract"],
        "equivalence_preflight": equivalence, "compile": compile_report,
        "fixed_checkpoint_step": maximum, "development": final_development,
        "development_gate_passed": bool(
            final_development["mean_score"] <= config.development_mean_score_max
            and final_development["max_seed_score"] <= config.development_max_seed_score_max
        ),
        "checkpoint_sha256": _file_sha(checkpoint_path), "fresh_accessed": False,
    }
    freeze["freeze_sha256"] = _sha(freeze)
    (output_dir / "selection_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    report = {
        "schema_version": "giada-task3c-training-v1", "valid": True,
        "environment": environment_manifest(torch), "code_revision": str(code_revision),
        "execution": {"vectorized_seed_count": len(config.seeds), "elapsed_seconds": elapsed, "model_steps_per_second": len(config.seeds) * maximum / elapsed},
        "equivalence_preflight": equivalence, "compile": compile_report,
        "checkpoints": checkpoints, "fresh_accessed": False, "freeze_sha256": freeze["freeze_sha256"],
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_frozen_symmetric_joint_gate(bundle, output_dir, config=None):
    config = config or JointGateSymmetricConfirmationConfig(); config.validate(); output_dir = Path(output_dir)
    marker = output_dir / "fresh_confirmation_opened.json"
    if marker.exists() or (output_dir / "final_report.json").exists():
        raise RuntimeError("Task 3c fresh confirmation has already been opened")
    freeze = json.loads((output_dir / "selection_freeze.json").read_text(encoding="utf-8")); claimed = freeze.pop("freeze_sha256")
    if _sha(freeze) != claimed or freeze["fresh_accessed"] or not freeze["development_gate_passed"]:
        raise RuntimeError("invalid Task 3c freeze or development gate not passed")
    checkpoint_path = output_dir / "frozen_vectorized_checkpoint.pt"
    if _file_sha(checkpoint_path) != freeze["checkpoint_sha256"]: raise RuntimeError("Task 3c checkpoint SHA-256 mismatch")
    torch = configure_torch_runtime(config.seeds[0]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)["per_seed_state"]
    constructor = _constructors(torch, JointGateCellConfig(matched_width=config.width))["shared_matched"]
    seed_metrics = {}; seed_rollouts = {}
    for seed in config.seeds:
        model = constructor().to(device); model.load_state_dict(saved[str(seed)])
        seed_metrics[str(seed)] = _evaluate(model, bundle["fresh3c"].items(), torch, device)
        rollout_rows = {}
        for name, row in bundle["fresh3c"].items():
            enriched = dict(row); enriched["formula"] = bundle["formula"]
            rollout_rows[name] = {str(h): _rollout(model, enriched, h, torch, device) for h in config.rollout_horizons}
        seed_rollouts[str(seed)] = rollout_rows
    names = tuple(bundle["fresh3c"])
    mean_metric = lambda key: float(np.mean([seed_metrics[str(seed)][name][key] for seed in config.seeds for name in names]))
    mean_rollout = lambda horizon, key: float(np.mean([seed_rollouts[str(seed)][name][str(horizon)][key] for seed in config.seeds for name in names]))
    decision = {
        "fresh_m_rmse": mean_metric("m_rmse"), "fresh_h_rmse": mean_metric("h_rmse"),
        "fresh_open_rmse": mean_metric("open_rmse"),
        "rollout1000_m_rmse": mean_rollout(1000, "m_rmse"),
        "rollout1000_h_rmse": mean_rollout(1000, "h_rmse"),
        "rollout1000_open_rmse": mean_rollout(1000, "open_rmse"),
        "occupancy_violations": int(sum(seed_metrics[str(seed)][name]["occupancy_violation_count"] for seed in config.seeds for name in names)),
        "development_mean_score": freeze["development"]["mean_score"],
        "development_max_seed_score": freeze["development"]["max_seed_score"],
    }
    finite = bool(np.isfinite(list(decision.values())).all())
    decision["task3c_passed"] = bool(
        finite and decision["occupancy_violations"] == 0
        and decision["fresh_m_rmse"] <= 0.0025 and decision["fresh_h_rmse"] <= 0.001
        and decision["fresh_open_rmse"] <= 0.0025 and decision["rollout1000_m_rmse"] <= 0.0025
        and decision["rollout1000_h_rmse"] <= 0.005
    )
    decision["task4_authorized"] = decision["task3c_passed"]
    report = {
        "schema_version": "giada-task3c-final-v1", "valid": finite,
        "selection_used_fresh": False, "fresh_contract": bundle["confirmation_contract"],
        "per_seed_metrics": seed_metrics, "per_seed_rollouts": seed_rollouts, "decision": decision,
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    marker.write_text(json.dumps({"freeze_sha256": claimed, "final_report_sha256": _file_sha(output_dir / "final_report.json"), "opened_once": True}, indent=2), encoding="utf-8")
    return report
