"""Task 6: frozen Ca_HVA primitives under known varying-voltage paths.

This is an exogenous path test, not an endogenous membrane-coupling claim.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .double_oracle import NeuronIsolatedGateOracle
from .gpu_baseline_runtime import configure_torch_runtime
from .primitive_matrix_playground import PrimitiveMatrixConfig, _file_sha, _learned_models, _sha
from .primitive_scaling import _gpu_rate_table, gpu_lut_predict


EXPECTED_TASK5_ARCHIVE_SHA256 = "ec6e691fe8a3c7fa30ec70db967f508f358f05b6bbaec58f3b7bcf0efed383f3"
EXPECTED_TASK5_REPORT_SHA256 = "027c06ded49a9c83daeadf2c8a5bc3bb00884991df6e9b387a1b5b120f5536f5"


@dataclass(frozen=True)
class VoltagePathStressConfig:
    segments_per_ms: int = 8
    development_per_family: int = 256
    sealed_per_family: int = 512
    development_seed: int = 86031
    sealed_seed: int = 86032
    oracle_pilot_count: int = 24
    selected_width: int = 32
    selected_lut_points: int = 513
    voltage_min_mv: float = -120.0
    voltage_max_mv: float = 60.0

    def validate(self):
        if (self.segments_per_ms, self.development_per_family, self.sealed_per_family) != (8, 256, 512):
            raise ValueError("Task 6 path grid differs from preregistration")
        if (self.selected_width, self.selected_lut_points) != (32, 513):
            raise ValueError("Task 6 must use Task 5 development-selected candidates")


def verified_task5_root(source: Path, cache: Path) -> Path:
    source, cache = Path(source), Path(cache)
    if source.is_file():
        if _file_sha(source) != EXPECTED_TASK5_ARCHIVE_SHA256:
            raise RuntimeError("Task 5 archive SHA-256 mismatch")
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (cache / member.filename).resolve()
                if cache.resolve() not in target.parents and target != cache.resolve():
                    raise RuntimeError("Unsafe Task 5 archive member")
            archive.extractall(cache)
        candidates = list(cache.rglob("final_report.json"))
        if len(candidates) != 1:
            raise RuntimeError("Ambiguous Task 5 report")
        root = candidates[0].parent
    else:
        root = source
    if _file_sha(root / "final_report.json") != EXPECTED_TASK5_REPORT_SHA256:
        raise RuntimeError("Task 5 report SHA-256 mismatch")
    report = json.loads((root / "final_report.json").read_text())
    freeze = json.loads((root / "selection_freeze.json").read_text())
    claimed = freeze.pop("freeze_sha256")
    if _sha(freeze) != claimed or _file_sha(root / "frozen_scaling_checkpoints.pt") != freeze["checkpoint_sha256"]:
        raise RuntimeError("Task 5 freeze or checkpoint hash mismatch")
    if not report["valid"] or report["decision"]["best_physical_width"] != 32 or report["decision"]["best_linear_lut"] != "lut_linear_513":
        raise RuntimeError("Task 5 decision contract mismatch")
    return root


def make_paths(config: VoltagePathStressConfig, *, role: str) -> dict[str, np.ndarray]:
    """Generate disjoint exogenous voltage-clamp schedules and initial gates."""
    config.validate()
    if role not in {"development", "sealed"}:
        raise ValueError(role)
    count = config.development_per_family if role == "development" else config.sealed_per_family
    rng = np.random.default_rng(config.development_seed if role == "development" else config.sealed_seed)
    t = np.arange(config.segments_per_ms, dtype=np.float64) / config.segments_per_ms
    result = {}
    for family in ("step_up", "step_down", "ramp", "biphasic", "near_singularity", "voltage_extremes"):
        baseline = rng.uniform(-120., -25., size=(count, 1))
        swing = rng.uniform(25., 95., size=(count, 1))
        if family == "step_up":
            path = baseline + swing * (t[None, :] >= .5)
        elif family == "step_down":
            path = baseline + swing * (t[None, :] < .5)
        elif family == "ramp":
            path = baseline + swing * t[None, :]
        elif family == "biphasic":
            path = baseline + swing * ((t[None, :] >= .25) & (t[None, :] < .75))
        elif family == "near_singularity":
            path = -27. + rng.choice(np.array([-.01, -.0001, 0., .0001, .01]), size=(count, config.segments_per_ms))
        else:
            path = np.where(t[None, :] < .5,
                            rng.uniform(-120., -105., size=(count, 1)),
                            rng.uniform(45., 60., size=(count, 1)))
        path = np.clip(path, config.voltage_min_mv, config.voltage_max_mv)
        initial = rng.uniform(.01, .99, size=(count, 2))
        result[family] = np.concatenate((path, initial), axis=1)
    return result


def _formula_step(formula, voltage, state, dt):
    rates = formula.rates(float(voltage))
    inf = np.array([rates["m_inf"], rates["h_inf"]], dtype=np.float64)
    tau = np.array([rates["m_tau_ms"], rates["h_tau_ms"]], dtype=np.float64)
    return inf + (state - inf) * np.exp(-dt / tau)


def formula_targets(formula, paths, config):
    target, held = {}, {}
    dt = 1.0 / config.segments_per_ms
    for family, rows in paths.items():
        truth = np.empty((len(rows), 2), dtype=np.float64)
        start_only = np.empty_like(truth)
        for i, row in enumerate(rows):
            state = row[-2:].copy()
            for voltage in row[:config.segments_per_ms]:
                state = _formula_step(formula, voltage, state, dt)
            truth[i] = state
            start_only[i] = _formula_step(formula, row[0], row[-2:], 1.0)
        target[family], held[family] = truth, start_only
    return target, held


def authentic_path_pilot(formula, paths, config, mechanism_root):
    """Check the compositional formula against the compiled NEURON mechanism."""
    oracle = NeuronIsolatedGateOracle(mechanism_root)
    errors = []
    dt = 1.0 / config.segments_per_ms
    for family, rows in paths.items():
        for row in rows[:config.oracle_pilot_count]:
            predicted = row[-2:].copy()
            observed = row[-2:].copy()
            for voltage in row[:config.segments_per_ms]:
                predicted = _formula_step(formula, voltage, predicted, dt)
                observed = np.array([oracle.step(gate, observed[j], voltage, dt) for j, gate in enumerate(("m", "h"))])
            errors.append(float(np.max(np.abs(predicted - observed))))
    maximum = max(errors)
    report = {"valid": maximum <= 1e-9 and oracle.maximum_absolute_voltage_change_mv <= 1e-8,
              "case_count": len(errors), "maximum_gate_error": maximum,
              "maximum_voltage_change_mv": oracle.maximum_absolute_voltage_change_mv,
              "tolerance": 1e-9}
    if not report["valid"]:
        raise RuntimeError(f"NEURON varying-path oracle pilot failed: {report}")
    return report


def _scores(prediction, targets):
    result = {}
    for family, truth in targets.items():
        pred = prediction[family]
        m = float(np.sqrt(np.mean((pred[:, 0] - truth[:, 0]) ** 2)))
        h = float(np.sqrt(np.mean((pred[:, 1] - truth[:, 1]) ** 2)))
        open_error = pred[:, 0] ** 2 * pred[:, 1] - truth[:, 0] ** 2 * truth[:, 1]
        o = float(np.sqrt(np.mean(open_error ** 2)))
        result[family] = {"m_rmse": m, "h_rmse": h, "open_rmse": o,
                          "normalized_score": max(m / .0025, h / .001, o / .0025),
                          "occupancy_violations": int(np.count_nonzero((pred < 0) | (pred > 1)))}
    return result


def _candidate_predictions(torch, model, table, paths, config, device, *, path_aware):
    all_seeds, lut = [], {}
    dt = 1.0 / config.segments_per_ms
    with torch.inference_mode():
        for family, rows in paths.items():
            voltage = torch.as_tensor(rows[:, :config.segments_per_ms], device=device, dtype=torch.float32)
            initial = torch.as_tensor(rows[:, -2:], device=device, dtype=torch.float32)
            state_m = initial.unsqueeze(0).expand(3, -1, -1).clone()
            state_l = initial.clone()
            repetitions = config.segments_per_ms if path_aware else 1
            for k in range(repetitions):
                v = voltage[:, k] if path_aware else voltage[:, 0]
                step = dt if path_aware else 1.0
                x_l = torch.cat((v[:, None], state_l, torch.full_like(v[:, None], step)), dim=1)
                state_l = gpu_lut_predict(torch, x_l, table, linear=True)
                x_m = torch.cat((v[None, :, None].expand(3, -1, -1), state_m,
                                 torch.full((3, len(rows), 1), step, device=device)), dim=2)
                state_m = model(x_m)[0]
            lut[family] = state_l.cpu().double().numpy()
            all_seeds.append((family, state_m.cpu().double().numpy()))
    per_seed = [{family: value[s] for family, value in all_seeds} for s in range(3)]
    return per_seed, lut


def evaluate_path_role(formula, root, paths, config, *, role):
    torch = configure_torch_runtime(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 6 requires CUDA")
    device = torch.device("cuda")
    model = _learned_models(torch, PrimitiveMatrixConfig(physical_width=32))["physical_tau"].to(device)
    states = torch.load(root / "frozen_scaling_checkpoints.pt", map_location=device, weights_only=True)
    model.load_state_dict(states["32"])
    model.eval()
    table = _gpu_rate_table(torch, formula, config.selected_lut_points, device, torch.float32)
    targets, held = formula_targets(formula, paths, config)
    predictions = {"formula_start_only": _scores(held, targets)}
    for path_aware in (False, True):
        seeds, lut = _candidate_predictions(torch, model, table, paths, config, device, path_aware=path_aware)
        suffix = "known_path" if path_aware else "start_only"
        predictions[f"lut_{suffix}"] = _scores(lut, targets)
        predictions[f"physical_{suffix}"] = {str(seed): _scores(rows, targets) for seed, rows in zip((17, 29, 43), seeds)}
    return {"role": role, "family_counts": {k: len(v) for k, v in paths.items()},
            "metrics": predictions, "target_sha256": hashlib.sha256(
                b"".join(np.asarray(targets[k], dtype="<f8").tobytes() for k in sorted(targets))).hexdigest()}


def run_voltage_path_stress(formula, task5_root, mechanism_root, output_dir, config=None, *, code_revision="unknown"):
    config = config or VoltagePathStressConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    root = verified_task5_root(task5_root, output_dir / ".verified_task5")
    print("[GIADA Task 6] 1/4 sorgente Task 5 e checkpoint verificati", flush=True)
    development = make_paths(config, role="development")
    pilot = authentic_path_pilot(formula, development, config, mechanism_root)
    print(f"[GIADA Task 6] 2/4 pilot NEURON valido: {pilot['case_count']} casi", flush=True)
    dev_report = evaluate_path_role(formula, root, development, config, role="development")
    freeze = {"schema_version": "giada-task6-freeze-v1", "code_revision": code_revision,
              "config": asdict(config), "task5_archive_sha256": EXPECTED_TASK5_ARCHIVE_SHA256,
              "frozen_candidates": ["physical_width_32_seeds_17_29_43", "lut_linear_513", "formula_start_only"],
              "development_target_sha256": dev_report["target_sha256"],
              "sealed_accessed": False, "no_model_selection_or_retraining": True}
    freeze["freeze_sha256"] = _sha(freeze)
    (output_dir / "development_report.json").write_text(json.dumps({"pilot": pilot, **dev_report}, indent=2))
    (output_dir / "selection_freeze.json").write_text(json.dumps(freeze, indent=2))
    print("[GIADA Task 6] 3/4 development concluso e candidati congelati", flush=True)
    sealed = make_paths(config, role="sealed")
    old = {tuple(row) for group in development.values() for row in group}
    new = {tuple(row) for group in sealed.values() for row in group}
    if old & new:
        raise RuntimeError("Task 6 sealed paths overlap development")
    sealed_report = evaluate_path_role(formula, root, sealed, config, role="sealed")
    values = []
    for arm, rows in sealed_report["metrics"].items():
        families = rows.values() if not arm.startswith("physical_") else (family for seed in rows.values() for family in seed.values())
        values.extend(metric["normalized_score"] for metric in families)
    report = {"schema_version": "giada-task6-final-v1", "valid": bool(pilot["valid"] and np.isfinite(values).all()),
              "code_revision": code_revision, "oracle_pilot": pilot, "development": dev_report,
              "sealed": sealed_report, "sealed_overlap_count": 0, "selection_used_sealed": False,
              "teacher_voltage_path_is_exogenous": True, "endogenous_membrane_coupling_tested": False,
              "interpretation_policy": "Known exogenous path is a diagnostic upper bound, not a causal input available from an endogenous future membrane trajectory."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "sealed_opened.json").write_text(json.dumps({"freeze_sha256": freeze["freeze_sha256"],
        "final_report_sha256": _file_sha(output_dir / "final_report.json"), "opened_once": True}, indent=2))
    print("[GIADA Task 6] 4/4 sealed concluso e report salvato", flush=True)
    return report
