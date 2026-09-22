"""Task 8: controlled continuous Ca_HVA voltage paths, frozen candidates.

The prescribed path is exogenous. It is not an input available to a future
closed-loop membrane model. No candidate is selected on these outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .double_oracle import NeuronIsolatedGateOracle
from .gpu_baseline_runtime import configure_torch_runtime
from .primitive_matrix_playground import PrimitiveMatrixConfig, _learned_models, _sha
from .primitive_scaling import _gpu_rate_table, gpu_lut_predict
from .voltage_path_stress import _formula_step, _scores, verified_task5_root


@dataclass(frozen=True)
class ContinuousPathConfig:
    steps_per_ms: int = 40
    coarse_steps: int = 8
    development_per_family: int = 128
    sealed_per_family: int = 256
    development_seed: int = 88031
    sealed_seed: int = 88032
    pilot_per_family: int = 12
    lut_points: int = 513

    def validate(self):
        if self.steps_per_ms != 40 or self.coarse_steps != 8 or self.steps_per_ms % self.coarse_steps:
            raise ValueError("Task 8 fine/coarse path grid differs from preregistration")
        if (self.development_per_family, self.sealed_per_family, self.lut_points) != (128, 256, 513):
            raise ValueError("Task 8 support or LUT contract differs from preregistration")


FAMILIES = ("slow_ramp", "fast_ramp", "low_chirp", "high_chirp")


def make_continuous_paths(config: ContinuousPathConfig, *, role: str):
    config.validate()
    if role not in {"development", "sealed"}:
        raise ValueError(role)
    count = config.development_per_family if role == "development" else config.sealed_per_family
    rng = np.random.default_rng(config.development_seed if role == "development" else config.sealed_seed)
    # Each value is the voltage at the *start* of its 0.025 ms subinterval.
    t = np.arange(config.steps_per_ms, dtype=np.float64) / config.steps_per_ms
    result = {}
    for family in FAMILIES:
        baseline = rng.uniform(-80., -45., size=(count, 1))
        amplitude = rng.uniform(10., 35., size=(count, 1))
        direction = rng.choice(np.array([-1., 1.]), size=(count, 1))
        if family == "slow_ramp":
            waveform = direction * amplitude * t[None, :]
        elif family == "fast_ramp":
            # Same start/end range, but most movement occurs in 0.25 ms.
            onset = rng.uniform(.2, .6, size=(count, 1))
            waveform = direction * amplitude * np.clip((t[None, :] - onset) / .25, 0., 1.)
        else:
            f0, f1 = (1., 3.) if family == "low_chirp" else (4., 12.)
            phase = 2 * np.pi * (f0 * t[None, :] + .5 * (f1 - f0) * t[None, :] ** 2)
            waveform = amplitude * np.sin(phase + rng.uniform(-np.pi, np.pi, size=(count, 1)))
        voltage = np.clip(baseline + waveform, -120., 60.)
        occupancy = rng.uniform(.01, .99, size=(count, 2))
        result[family] = np.concatenate((voltage, occupancy), axis=1)
    return result


def formula_path_targets(formula, paths, config):
    """Exact fine-path teacher and two causal-information controls."""
    targets, coarse, start = {}, {}, {}
    stride = config.steps_per_ms // config.coarse_steps
    for family, rows in paths.items():
        exact = np.empty((len(rows), 2), dtype=np.float64)
        coarse_values = np.empty_like(exact)
        start_values = np.empty_like(exact)
        for i, row in enumerate(rows):
            state, coarse_state = row[-2:].copy(), row[-2:].copy()
            for k, voltage in enumerate(row[:config.steps_per_ms]):
                state = _formula_step(formula, voltage, state, 1 / config.steps_per_ms)
                # Coarse path is sampled at the same substep convention as the fine path.
                if k % stride == 0:
                    coarse_state = _formula_step(formula, voltage, coarse_state, 1 / config.coarse_steps)
            exact[i], coarse_values[i] = state, coarse_state
            start_values[i] = _formula_step(formula, row[0], row[-2:], 1.)
        targets[family], coarse[family], start[family] = exact, coarse_values, start_values
    return targets, coarse, start


def authentic_path_pilot(formula, paths, config, mechanism_root):
    oracle = NeuronIsolatedGateOracle(mechanism_root)
    maximum = 0.
    count = 0
    for rows in paths.values():
        for row in rows[:config.pilot_per_family]:
            predicted = row[-2:].copy()
            observed = predicted.copy()
            for voltage in row[:config.steps_per_ms]:
                predicted = _formula_step(formula, voltage, predicted, 1 / config.steps_per_ms)
                observed = np.array([oracle.step(gate, observed[j], voltage, 1 / config.steps_per_ms)
                                     for j, gate in enumerate(("m", "h"))])
            maximum = max(maximum, float(np.max(np.abs(predicted - observed))))
            count += 1
    result = {"valid": maximum <= 1e-9 and oracle.maximum_absolute_voltage_change_mv <= 1e-8,
              "case_count": count, "maximum_gate_error": maximum,
              "maximum_voltage_change_mv": oracle.maximum_absolute_voltage_change_mv}
    if not result["valid"]:
        raise RuntimeError(f"Task 8 authentic NEURON pilot failed: {result}")
    return result


def _candidate_predictions(torch, model, table, paths, config, device, *, view):
    seeds, lut = {}, {}
    stride = config.steps_per_ms // config.coarse_steps
    repetitions = {"start_only": 1, "coarse_path": config.coarse_steps,
                   "fine_path": config.steps_per_ms}[view]
    dt = 1 / repetitions
    with torch.inference_mode():
        for family, rows in paths.items():
            voltage = torch.as_tensor(rows[:, :config.steps_per_ms], device=device, dtype=torch.float32)
            initial = torch.as_tensor(rows[:, -2:], device=device, dtype=torch.float32)
            state_l = initial.clone()
            state_m = initial.unsqueeze(0).expand(3, -1, -1).clone()
            for k in range(repetitions):
                index = k * stride if view == "coarse_path" else k if view == "fine_path" else 0
                v = voltage[:, index]
                x_l = torch.cat((v[:, None], state_l, torch.full_like(v[:, None], dt)), dim=1)
                state_l = gpu_lut_predict(torch, x_l, table, linear=True)
                x_m = torch.cat((v[None, :, None].expand(3, -1, -1), state_m,
                                 torch.full((3, len(rows), 1), dt, device=device)), dim=2)
                state_m = model(x_m)[0]
            lut[family] = state_l.cpu().double().numpy()
            seeds[family] = state_m.cpu().double().numpy()
    return lut, [{family: values[s] for family, values in seeds.items()} for s in range(3)]


def evaluate_continuous_role(formula, task5_root, paths, config, *, role):
    torch = configure_torch_runtime(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 8 requires CUDA for frozen candidate comparisons")
    device = torch.device("cuda")
    model = _learned_models(torch, PrimitiveMatrixConfig(physical_width=32))["physical_tau"].to(device)
    states = torch.load(Path(task5_root) / "frozen_scaling_checkpoints.pt", map_location=device, weights_only=True)
    model.load_state_dict(states["32"])
    model.eval()
    table = _gpu_rate_table(torch, formula, config.lut_points, device, torch.float32)
    targets, coarse, start = formula_path_targets(formula, paths, config)
    metrics = {"formula_start_only": _scores(start, targets),
               "formula_coarse_path": _scores(coarse, targets)}
    for view in ("start_only", "coarse_path", "fine_path"):
        lut, seeds = _candidate_predictions(torch, model, table, paths, config, device, view=view)
        metrics[f"lut_{view}"] = _scores(lut, targets)
        metrics[f"physical_{view}"] = {str(seed): _scores(value, targets)
                                          for seed, value in zip((17, 29, 43), seeds)}
    fingerprint = hashlib.sha256(b"".join(np.asarray(targets[key], dtype="<f8").tobytes()
                                         for key in sorted(targets))).hexdigest()
    return {"role": role, "family_counts": {key: len(value) for key, value in paths.items()},
            "target_sha256": fingerprint, "metrics": metrics}


def run_continuous_path_stress(formula, task5_source, mechanism_root, output_dir,
                               config=None, *, code_revision="unknown"):
    config = config or ContinuousPathConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    task5_root = verified_task5_root(task5_source, output_dir / ".verified_task5")
    development = make_continuous_paths(config, role="development")
    pilot = authentic_path_pilot(formula, development, config, mechanism_root)
    print(f"[GIADA Task 8] pilot NEURON: {pilot['case_count']} casi validi", flush=True)
    dev = evaluate_continuous_role(formula, task5_root, development, config, role="development")
    freeze = {"schema_version": "giada-task8-freeze-v1", "code_revision": code_revision,
              "config": asdict(config), "development_target_sha256": dev["target_sha256"],
              "frozen_candidates": ["formula", "lut_linear_513", "physical_tau_32_seeds_17_29_43"],
              "sealed_accessed": False, "selection_performed": False, "retraining_performed": False}
    freeze["freeze_sha256"] = _sha(freeze)
    (output_dir / "development_report.json").write_text(json.dumps({"pilot": pilot, **dev}, indent=2))
    (output_dir / "selection_freeze.json").write_text(json.dumps(freeze, indent=2))
    print("[GIADA Task 8] development concluso; freeze salvato", flush=True)
    sealed = make_continuous_paths(config, role="sealed")
    overlap = {tuple(row) for rows in development.values() for row in rows} & \
              {tuple(row) for rows in sealed.values() for row in rows}
    if overlap:
        raise RuntimeError("Task 8 development/sealed path overlap")
    sealed_report = evaluate_continuous_role(formula, task5_root, sealed, config, role="sealed")
    scores = []
    violations = 0
    for arm, values in sealed_report["metrics"].items():
        families = (row for seed in values.values() for row in seed.values()) if arm.startswith("physical_") else values.values()
        for row in families:
            scores.append(row["normalized_score"])
            violations += row["occupancy_violations"]
    report = {"schema_version": "giada-task8-continuous-path-v1",
              "valid": bool(pilot["valid"] and np.isfinite(scores).all() and violations == 0),
              "code_revision": code_revision, "pilot": pilot, "development": dev, "sealed": sealed_report,
              "sealed_overlap_count": 0, "selection_used_sealed": False,
              "candidate_retrained": False, "endogenous_voltage_tested": False,
              "interpretation": "Exogenous voltage-path stress only; full 642-segment and endogenous claims prohibited."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "sealed_opened.json").write_text(json.dumps({"freeze_sha256": freeze["freeze_sha256"],
        "final_report_sha256": hashlib.sha256((output_dir / "final_report.json").read_bytes()).hexdigest(),
        "opened_once": True}, indent=2))
    print("[GIADA Task 8] sealed valutato; report salvato", flush=True)
    return report
