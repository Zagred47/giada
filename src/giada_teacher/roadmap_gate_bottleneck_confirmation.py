"""GIADA Task 15b: paired 2x2 loss intervention on learned Ca-HVA gates."""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .roadmap_causal_operator import _build_model
from .roadmap_current_architecture_comparison import (
    CURRENT_SCALE_MA_CM2, _checkpoint_sha, _counterfactual_roles,
    _decode_outputs, _evaluation, _freeze_hash, _parameterized_role,
    _predict, _recursive_rollout, _rmse, prepare_current_role,
)


EXPECTED_TASK15_FINAL_SHA256 = "6c4144caf95785576bdc9a786eeecd43e4da7151147999287d7a65614f3c7bc3"
TASK15_CODE_REVISION = "c4a91c79f853225840fe548e1a4aab170539dab3"
ARMS = {
    "baseline": (4.0, 0.0),
    "gate_weighted": (16.0, 0.0),
    "current_supervised": (4.0, 1.0),
    "joint": (16.0, 1.0),
}


@dataclass(frozen=True)
class GateBottleneckConfig:
    train_seed: int = 15211
    development_seed: int = 15229
    sealed_seed: int = 15259
    counterfactual_seed: int = 15359
    train_episodes: int = 192
    development_episodes: int = 48
    sealed_episodes: int = 48
    counterfactual_episodes: int = 48
    duration_ms: int = 16
    seeds: tuple[int, ...] = (17, 29, 43)
    steps: int = 800
    checkpoints: tuple[int, ...] = (400, 800)
    batch_size: int = 256
    width: int = 62
    learning_rate: float = 0.001
    parameter_mask_off_fraction: float = 0.125

    def validate(self):
        if asdict(self) != asdict(GateBottleneckConfig()):
            raise ValueError("Task 15b differs from preregistration")


def _read_result_member(source, filename):
    source = Path(source)
    if source.is_dir():
        matches = list(source.rglob(filename))
        if len(matches) != 1:
            raise ValueError(f"Task 15 {filename} not unique")
        return matches[0].read_bytes()
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("Task 15 source must be ZIP or extracted directory")
    with zipfile.ZipFile(source) as outer:
        names = [n for n in outer.namelist() if n.endswith("/" + filename)]
        if len(names) == 1:
            return outer.read(names[0])
        nested = [n for n in outer.namelist()
                  if n.endswith("giada_roadmap_task15_current_architecture_comparison.zip")]
        if len(nested) != 1:
            raise ValueError("Task 15 nested ZIP not unique")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
            names = [n for n in inner.namelist() if n.endswith("/" + filename)]
            if len(names) != 1:
                raise ValueError(f"Nested Task 15 {filename} not unique")
            return inner.read(names[0])


def verified_task15_result(source):
    raw = _read_result_member(source, "final_report.json")
    report = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if (digest != EXPECTED_TASK15_FINAL_SHA256 or not report.get("valid")
            or report.get("code_revision") != TASK15_CODE_REVISION
            or report.get("schema_version") != "giada-roadmap-task15-current-comparison-v1"
            or report.get("sealed_or_counterfactual_used_for_selection") is not False):
        raise ValueError("Task 15 immutable result contract mismatch")
    return {"final_report_sha256": digest, "code_revision": TASK15_CODE_REVISION,
            "baseline_15_sealed_m_rmse": report["sealed_one_step_and_available_rollout"]
            ["decomposed_formula"]["m_rmse"]}


def _loss(torch, model, x, knots, endpoint_gates, target_current,
          gate_weight, current_weight):
    path, gates, _ = _decode_outputs("decomposed_formula", model(x), x)
    voltage_loss = torch.mean(((path - knots) / 20.0) ** 2)
    gate_loss = torch.mean((gates - endpoint_gates) ** 2)
    nominal_gbar = x[:, 3] * (8.0 / 1e5)
    effective_gbar = nominal_gbar * x[:, 4]
    eca = 120.0 + 40.0 * x[:, 5]
    predicted_current = effective_gbar * gates[:, 0] ** 2 * gates[:, 1] * (path[:, -1] - eca)
    current_loss = torch.mean(((predicted_current - target_current) / CURRENT_SCALE_MA_CM2) ** 2)
    return voltage_loss + gate_weight * gate_loss + current_weight * current_loss


def _load_model(torch, output_dir, checkpoint_row, device, width):
    path = Path(output_dir) / "checkpoints" / checkpoint_row["file"]
    if _checkpoint_sha(path) != checkpoint_row["sha256"]:
        raise RuntimeError("Task 15b checkpoint SHA-256 mismatch")
    saved = torch.load(path, map_location=device, weights_only=True)
    if (saved["arm"] != checkpoint_row["arm"] or saved["seed"] != checkpoint_row["seed"]
            or saved["step"] != checkpoint_row["step"]):
        raise RuntimeError("Task 15b checkpoint identity mismatch")
    model = _build_model(torch, width, 10, 6).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def train_and_freeze(task15_source, output_dir, config=GateBottleneckConfig(), *, device="cpu"):
    """Fit all arms on paired streams; select per arm on development m only."""
    import torch

    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Task 15b output exists: {output_dir}")
    prerequisite = verified_task15_result(task15_source)
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    train = prepare_current_role(_parameterized_role(config.train_seed, config.train_episodes, config))
    development = prepare_current_role(
        _parameterized_role(config.development_seed, config.development_episodes, config))
    contract = {"schema_version": "giada-roadmap-task15b-prepare-v1",
                "prerequisite": prerequisite, "config": asdict(config),
                "arms": {name: {"gate_weight": weights[0], "current_weight": weights[1]}
                         for name, weights in ARMS.items()},
                "train_x_sha256": hashlib.sha256(train["x"].tobytes()).hexdigest(),
                "development_x_sha256": hashlib.sha256(development["x"].tobytes()).hexdigest(),
                "same_numeric_input_and_architecture": True,
                "sealed_or_counterfactual_generated": False}
    (output_dir / "prepare_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8")
    x = torch.as_tensor(train["x"], device=device)
    knots = torch.as_tensor(train["knots"], device=device, dtype=torch.float32)
    gates = torch.as_tensor(train["endpoint"][:, 1:3], device=device, dtype=torch.float32)
    current = torch.as_tensor(train["current"], device=device, dtype=torch.float32)
    records = {}
    total = len(ARMS) * len(config.seeds) * len(config.checkpoints)
    done, started = 0, time.perf_counter()
    for seed in config.seeds:
        batch_indices = np.random.default_rng(seed).integers(
            0, len(x), size=(config.steps, config.batch_size), dtype=np.int64)
        batch_indices = torch.as_tensor(batch_indices, device=device)
        for arm, (gate_weight, current_weight) in ARMS.items():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = _build_model(torch, config.width, 10, 6).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            history = []
            for step in range(1, config.steps + 1):
                model.train()
                index = batch_indices[step - 1]
                loss = _loss(torch, model, x[index], knots[index], gates[index], current[index],
                             gate_weight, current_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step in config.checkpoints:
                    model.eval()
                    metrics, _ = _evaluation(torch, "decomposed_formula", model, development, device)
                    if not metrics["finite"] or not np.isfinite(metrics["m_rmse"]):
                        raise RuntimeError(f"Task 15b nonfinite development: {arm} seed{seed}")
                    filename = f"{arm}-seed{seed}-step{step}.pt"
                    path = output_dir / "checkpoints" / filename
                    torch.save({"arm": arm, "seed": seed, "step": step,
                                "model": model.state_dict()}, path)
                    history.append({"step": step, "metrics": metrics,
                                    "checkpoint": {"arm": arm, "seed": seed, "step": step,
                                                   "file": filename, "sha256": _checkpoint_sha(path)}})
                    done += 1
                    eta = (time.perf_counter() - started) / done * (total - done) / 60.0
                    print(f"[GIADA 15b] {done}/{total} {arm} seed={seed} step={step} "
                          f"dev_m={metrics['m_rmse']:.4g} dev_I={metrics['all_rmse_ma_cm2']:.3g} "
                          f"ETA {eta:.1f} min", flush=True)
            records[f"{arm}-seed{seed}"] = history
    selected = {}
    for arm in ARMS:
        rows = [(seed, records[f"{arm}-seed{seed}"]) for seed in config.seeds]
        seed, history = min(rows, key=lambda item: (item[1][-1]["metrics"]["m_rmse"], item[0]))
        selected[arm] = {"seed": seed, "selection_metric": "development_m_rmse_step800",
                         "development_m_rmse": history[-1]["metrics"]["m_rmse"],
                         "checkpoints": {str(row["step"]): row["checkpoint"] for row in history}}
    freeze = {"schema_version": "giada-roadmap-task15b-freeze-v1",
              "prerequisite": prerequisite, "config": asdict(config), "records": records,
              "selected": selected, "selection_source": "development_m_rmse_step800_only",
              "sealed_accessed": False, "counterfactual_accessed": False,
              "same_architecture_initialization_and_batch_indices_within_seed": True}
    freeze["freeze_sha256"] = _freeze_hash(freeze)
    (output_dir / "selection_freeze.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")
    return freeze


def evaluate_frozen(freeze, output_dir, config=GateBottleneckConfig(), *, device="cpu",
                    code_revision="unknown"):
    """Open fresh sealed roles once after verifying the development-only freeze."""
    import torch

    config.validate()
    output_dir = Path(output_dir)
    stored = json.loads((output_dir / "selection_freeze.json").read_text(encoding="utf-8"))
    claimed = stored.pop("freeze_sha256")
    if (_freeze_hash(stored) != claimed or claimed != freeze.get("freeze_sha256")
            or stored["sealed_accessed"] or stored["counterfactual_accessed"]
            or stored["selection_source"] != "development_m_rmse_step800_only"):
        raise RuntimeError("Task 15b invalid pre-test freeze")
    for history in stored["records"].values():
        for row in history:
            checkpoint = row["checkpoint"]
            if _checkpoint_sha(output_dir / "checkpoints" / checkpoint["file"]) != checkpoint["sha256"]:
                raise RuntimeError("Task 15b historical checkpoint SHA-256 mismatch")
    models = {}
    for arm, selected in stored["selected"].items():
        for step, checkpoint in selected["checkpoints"].items():
            models[(arm, step)] = _load_model(torch, output_dir, checkpoint, device, config.width)
    sealed_role = _parameterized_role(config.sealed_seed, config.sealed_episodes, config)
    sealed = prepare_current_role(sealed_role)
    sealed_metrics = {}
    for (arm, step), model in models.items():
        metrics, _ = _evaluation(torch, "decomposed_formula", model, sealed, device)
        metrics["recursive_16ms"] = _recursive_rollout(
            torch, "decomposed_formula", model, sealed_role, device)
        sealed_metrics[f"{arm}@{step}"] = metrics
    paired_roles = _counterfactual_roles(config)
    prepared = {name: prepare_current_role(role) for name, role in paired_roles.items()}
    baseline = prepared["baseline"]
    counterfactuals = {}
    for name, role in prepared.items():
        if name == "baseline":
            continue
        teacher_delta_i = role["current"] - baseline["current"]
        teacher_delta_v = role["endpoint"][:, 0] - baseline["endpoint"][:, 0]
        row = {"teacher_delta_current_rms_ma_cm2": _rmse(teacher_delta_i, 0.0),
               "teacher_delta_voltage_rms_mv": _rmse(teacher_delta_v, 0.0), "arms": {}}
        for arm in ARMS:
            model = models[(arm, str(config.steps))]
            base_prediction = _predict(torch, "decomposed_formula", model, baseline, device)
            changed = _predict(torch, "decomposed_formula", model, role, device)
            row["arms"][arm] = {
                "delta_current_rmse_ma_cm2": _rmse(changed["current"] - base_prediction["current"], teacher_delta_i),
                "delta_voltage_rmse_mv": _rmse(changed["state"][:, 0] - base_prediction["state"][:, 0], teacher_delta_v)}
        counterfactuals[name] = row
        print(f"[GIADA 15b paired] {len(counterfactuals)}/{len(prepared)-1} {name}", flush=True)
    def median_delta(arm):
        return float(np.median([row["arms"][arm]["delta_current_rmse_ma_cm2"]
                                for row in counterfactuals.values()]))
    base400, base800 = sealed_metrics["baseline@400"], sealed_metrics["baseline@800"]
    decisions = {"baseline_more_budget_reduces_m_by_25pct": bool(
                    base800["m_rmse"] <= .75 * base400["m_rmse"]
                    and base800["all_rmse_ma_cm2"] <= 1.10 * base400["all_rmse_ma_cm2"]),
                 "intervention_gate_pass": {},
                 "median_delta_current_rmse_ma_cm2": {arm: median_delta(arm) for arm in ARMS},
                 "gate_c_or_embedded_promotion_authorized": False}
    for arm in ARMS:
        if arm == "baseline":
            continue
        candidate = sealed_metrics[f"{arm}@800"]
        candidate_rollout = candidate["recursive_16ms"]
        baseline_rollout = base800["recursive_16ms"]
        decisions["intervention_gate_pass"][arm] = bool(
            candidate["m_rmse"] <= .5 * base800["m_rmse"]
            and candidate["all_rmse_ma_cm2"] <= 1.10 * base800["all_rmse_ma_cm2"]
            and candidate_rollout.get("valid", False)
            and baseline_rollout.get("valid", False)
            and candidate_rollout["voltage_rmse_mv"]
                <= 1.10 * baseline_rollout["voltage_rmse_mv"]
            and median_delta(arm) <= median_delta("baseline"))
    finite = (all(row["finite"] and np.isfinite(row["m_rmse"])
                  for row in sealed_metrics.values())
              and all(np.isfinite(arm_row["delta_current_rmse_ma_cm2"])
                      and np.isfinite(arm_row["delta_voltage_rmse_mv"])
                      for row in counterfactuals.values() for arm_row in row["arms"].values()))
    report = {"schema_version": "giada-roadmap-task15b-gate-bottleneck-v1",
              "valid": bool(finite), "code_revision": code_revision,
              "prerequisite": stored["prerequisite"], "freeze_sha256": claimed,
              "selected": stored["selected"], "sealed": sealed_metrics,
              "counterfactuals": counterfactuals, "registered_decisions": decisions,
              "selection_used_sealed_or_counterfactual": False,
              "same_numeric_input_and_architecture": True,
              "scope": "Ca_HVA+pas synthetic one compartment; not Task16 embedded confirmation or Gate C"}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
