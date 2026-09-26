"""GIADA roadmap Task 14: paired Ca-HVA parameter counterfactuals.

Frozen Task 11 models are evaluated without retraining.  The numerical
reference is the same 0.025 ms Ca-HVA+pas recurrence as Task 11, generalized
only by replacing its previously fixed calcium reversal potential.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .roadmap_causal_operator import (
    CausalOperatorConfig, _decode, _rates, _step, flatten_role,
    generate_role, integrate_predicted_path,
)
from .roadmap_current_contract import (
    _frozen_model, analytical_current, verified_task11_checkpoints,
)


@dataclass(frozen=True)
class ParameterVariationConfig:
    source_seed: int = 14059
    episode_count: int = 48
    duration_ms: int = 16
    reference_dt_ms: float = .025
    eca_baseline_mv: float = 120.
    eca_lower_mv: float = 100.
    eca_upper_mv: float = 140.
    gbar_low_multiplier: float = .5
    gbar_high_multiplier: float = 1.5
    m_initial_increase: float = .1
    h_initial_decrease: float = .1

    def validate(self):
        if asdict(self) != asdict(ParameterVariationConfig()):
            raise ValueError("Task 14 differs from preregistered parameter matrix")


def _step_with_eca(state, current_density, gbar, eca, dt):
    """Exact Task 11 finite-step equations with explicit E_Ca parameter."""
    v, m, h = np.moveaxis(np.asarray(state, dtype=np.float64), -1, 0)
    gca = np.asarray(gbar) * m*m*h
    capacitance = .001/dt
    new_v = (capacitance*v + .0001*-76. + gca*eca + current_density) / (
        capacitance + .0001 + gca)
    minf, hinf, mtau, htau = _rates(new_v)
    new_m = minf + (m-minf)*np.exp(-dt/mtau)
    new_h = hinf + (h-hinf)*np.exp(-dt/htau)
    return np.stack((new_v, new_m, new_h), axis=-1)


def _reference_role(base, *, gbar_multiplier=1., eca=120., m_shift=0., h_shift=0., mask=1.):
    source = np.asarray(base["states"][:, 0, 0], dtype=np.float64)
    initial = source.copy()
    initial[:, 1] = np.clip(initial[:, 1]+m_shift, 0., 1.)
    initial[:, 2] = np.clip(initial[:, 2]+h_shift, 0., 1.)
    gbar = np.asarray(base["gbar"])*gbar_multiplier*mask
    density = np.asarray(base["current_density"])
    episodes, duration = density.shape[:2]
    states = np.empty((episodes, duration, 41, 3), dtype=np.float64)
    state = initial
    for t in range(duration):
        states[:, t, 0] = state
        for tick in range(40):
            state = _step_with_eca(state, density[:, t, tick//10], gbar, eca, .025)
            states[:, t, tick+1] = state
    return {"states": states, "current_na": base["current_na"],
            "current_density": density, "gbar": gbar,
            "coefficients": np.zeros((episodes, duration, 2, 2), dtype=np.float64),
            "role_seed": base["role_seed"]}


def _eca_one_step_twins(base, eca):
    """Each 1 ms counterfactual starts from the *same* baseline boundary state."""
    density = np.asarray(base["current_density"])
    episodes, duration = density.shape[:2]
    states = np.empty_like(base["states"])
    for t in range(duration):
        state = np.asarray(base["states"][:, t, 0], dtype=np.float64)
        states[:, t, 0] = state
        for tick in range(40):
            state = _step_with_eca(state, density[:, t, tick//10], base["gbar"], eca, .025)
            states[:, t, tick+1] = state
    return {"states": states, "current_na": base["current_na"],
            "current_density": density, "gbar": base["gbar"],
            "coefficients": np.zeros((episodes, duration, 2, 2), dtype=np.float64),
            "role_seed": base["role_seed"]}


def _scenario_definitions(config):
    return {
        "baseline": {"axis": "baseline"},
        "gbar_half": {"axis": "gbar", "gbar_multiplier": config.gbar_low_multiplier},
        "gbar_one_half": {"axis": "gbar", "gbar_multiplier": config.gbar_high_multiplier},
        "eca_100": {"axis": "eca_unobserved", "eca": config.eca_lower_mv},
        "eca_140": {"axis": "eca_unobserved", "eca": config.eca_upper_mv},
        "initial_m_plus": {"axis": "initial_gate", "m_shift": config.m_initial_increase},
        "initial_h_minus": {"axis": "initial_gate", "h_shift": -config.h_initial_decrease},
        "mechanism_off_visible": {"axis": "visible_effective_gbar", "mask": 0.},
        "mechanism_off_hidden": {"axis": "negative_control_hidden_mask", "mask": 0., "hide_mask": True},
    }


def _rmse(prediction, target):
    return float(np.sqrt(np.mean((np.asarray(prediction)-np.asarray(target))**2)))


def _model_one_step(torch, model, arm, role, input_role, device):
    data = flatten_role(input_role)
    x = torch.as_tensor(data["x_full"], device=device)
    with torch.inference_mode():
        chunks = [_decode(torch, arm, model(part), part).cpu().numpy()
                  for part in x.split(4096)]
    decoded = np.concatenate(chunks).astype(np.float64)
    predicted = (integrate_predicted_path(data["initial"], decoded)
                 if arm == "path_full" else decoded[:, :3])
    actual = flatten_role(role)["end"]
    return predicted, actual


def _recursive(torch, model, arm, role, input_gbar, device):
    states = role["states"]
    count, duration = states.shape[:2]
    result = np.empty((count, duration+1, 3), dtype=np.float64)
    result[:, 0] = states[:, 0, 0]
    for t in range(duration):
        old = result[:, t]
        x = np.column_stack(((old[:, 0]+60.)/40., old[:, 1], old[:, 2],
                             input_gbar*1e5/8., role["current_na"][:, t]/.08)).astype(np.float32)
        with torch.inference_mode():
            xt = torch.as_tensor(x, device=device)
            decoded = _decode(torch, arm, model(xt), xt).cpu().numpy().astype(np.float64)
        result[:, t+1] = (integrate_predicted_path(old, decoded)
                          if arm == "path_full" else decoded[:, :3])
        if not np.isfinite(result[:, t+1]).all() or np.max(np.abs(result[:, t+1, 0])) > 250.:
            return None
    return result


def _verify_task12_13(source):
    """This prerequisite is tiny; use it for provenance, never for selection."""
    import zipfile
    expected = {
        "final_report.json": "70b4b9d4d57bdd4edc46e372fce42281a83aaac8ef3f80e7df8809910a18284e",
        "task12_current_report.json": "d2695aa1a204a999114aaba6278c14e341aed8535a80e4d71fb5805d5f8c34f7",
        "task13_timing_report.json": "62078df2c5feb22cd56a07d79560c22427e9b5ec4246d85022cd458de935c00a",
    }
    source = Path(source)
    if source.is_dir():
        files = [p for p in source.rglob("final_report.json")
                 if (p.parent/"task12_current_report.json").is_file()
                 and (p.parent/"task13_timing_report.json").is_file()]
        if len(files) != 1:
            raise ValueError("Task12/13 source has no unique final report")
        read = lambda name: (files[0].parent/name).read_bytes()
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            finals = [n for n in names if n.endswith("/final_report.json")]
            if len(finals) == 1:
                prefix = finals[0][:-len("final_report.json")]
                members = {name: archive.read(prefix+name) for name in expected}
            else:
                nested = [n for n in names if n.endswith("giada_roadmap_task12_13_current_contract.zip")]
                if len(nested) != 1:
                    raise ValueError("Task12/13 ZIP has no unique final report")
                with zipfile.ZipFile(io.BytesIO(archive.read(nested[0]))) as inner:
                    inner_finals = [n for n in inner.namelist() if n.endswith("/final_report.json")]
                    if len(inner_finals) != 1:
                        raise ValueError("nested Task12/13 ZIP has no unique final report")
                    prefix = inner_finals[0][:-len("final_report.json")]
                    members = {name: inner.read(prefix+name) for name in expected}
        read = lambda name: members[name]
    else:
        raise ValueError("Task12/13 source must be a ZIP or extracted directory")
    if any(hashlib.sha256(read(name)).hexdigest() != digest for name, digest in expected.items()):
        raise ValueError("Task12/13 immutable member SHA-256 mismatch")
    final = json.loads(read("final_report.json"))
    task12 = json.loads(read("task12_current_report.json"))
    task13 = json.loads(read("task13_timing_report.json"))
    if (final.get("schema_version") != "giada-roadmap-task12-13-v1"
            or final.get("code_revision") != "48116b1e3edc4c0360413585abb21f9ae236aaa1"
            or not final.get("valid") or not task12.get("valid") or not task13.get("valid")
            or task12.get("independent_role_seed") != 12059 or task13.get("winner") != "pre_all"):
        raise ValueError("Task12/13 prerequisite contract mismatch")
    return {"final_report_sha256": hashlib.sha256(read("final_report.json")).hexdigest(),
            "task12_report_sha256": hashlib.sha256(read("task12_current_report.json")).hexdigest(),
            "task13_report_sha256": hashlib.sha256(read("task13_timing_report.json")).hexdigest()}


def evaluate_parameter_matrix(task11_source, task12_13_source, *, device="cpu",
                              config=ParameterVariationConfig()):
    import torch
    config.validate()
    provenance = _verify_task12_13(task12_13_source)
    freeze, checkpoints = verified_task11_checkpoints(task11_source)
    base = generate_role(config.source_seed, config.episode_count, CausalOperatorConfig())
    same = _reference_role(base)
    baseline_max = float(np.max(np.abs(same["states"]-base["states"])))
    if baseline_max > 1e-11:
        raise RuntimeError(f"E_Ca parameterized reference diverges from Task11 at baseline: {baseline_max}")
    models = {arm: _frozen_model(torch, freeze, checkpoints, arm, device)
              for arm in ("path_full", "effect_full")}
    scenarios = _scenario_definitions(config)
    roles = {}
    for name, spec in scenarios.items():
        role = _reference_role(base, gbar_multiplier=spec.get("gbar_multiplier", 1.),
                               eca=spec.get("eca", 120.), m_shift=spec.get("m_shift", 0.),
                               h_shift=spec.get("h_shift", 0.), mask=spec.get("mask", 1.))
        roles[name] = role
    base_end = flatten_role(roles["baseline"])["end"]
    base_current = analytical_current(np.repeat(base["gbar"], config.duration_ms),
                                      base_end[:, 0], base_end[:, 1], base_end[:, 2])
    results = {}
    started = time.perf_counter()
    for index, (name, spec) in enumerate(scenarios.items(), start=1):
        role = roles[name]
        hidden = spec.get("hide_mask", False)
        input_role = {**role, "gbar": base["gbar"]} if hidden else role
        teacher = flatten_role(role)["end"]
        gbar = np.repeat(role["gbar"], config.duration_ms)
        eca = spec.get("eca", 120.)
        true_current = analytical_current(gbar, teacher[:, 0], teacher[:, 1], teacher[:, 2], eca)
        arm_results = {}
        for arm, model in models.items():
            pred, target = _model_one_step(torch, model, arm, role, input_role, device)
            pred_current = analytical_current(gbar if not hidden else np.repeat(base["gbar"], config.duration_ms),
                                              pred[:, 0], pred[:, 1], pred[:, 2], eca)
            rollout = _recursive(torch, model, arm, role, input_role["gbar"], device)
            teacher_recursive = np.concatenate((role["states"][:, 0, 0, None, :],
                                                role["states"][:, :, -1, :]), axis=1)
            arm_results[arm] = {
                "one_step_voltage_rmse_mv": _rmse(pred[:, 0], target[:, 0]),
                "one_step_m_rmse": _rmse(pred[:, 1], target[:, 1]),
                "one_step_h_rmse": _rmse(pred[:, 2], target[:, 2]),
                "one_step_current_rmse_ma_cm2": _rmse(pred_current, true_current),
                "recursive_valid": rollout is not None,
                "recursive_16ms_voltage_rmse_mv": (_rmse(rollout[:, 1:, 0], teacher_recursive[:, 1:, 0]) if rollout is not None else None),
                "recursive_16ms_m_rmse": (_rmse(rollout[:, 1:, 1], teacher_recursive[:, 1:, 1]) if rollout is not None else None),
                "predicted_delta_voltage_vs_baseline_rmse_mv": None,
                "predicted_delta_current_vs_baseline_rmse_ma_cm2": None,
                "_prediction_voltage": pred[:, 0],
                "_prediction_current": pred_current,
            }
        results[name] = {"axis": spec["axis"], "parameters": {k:v for k,v in spec.items() if k!="axis"},
                         "teacher_delta_voltage_vs_baseline_rms_mv": _rmse(teacher[:, 0], base_end[:, 0]),
                         "teacher_delta_current_vs_baseline_rms_ma_cm2": _rmse(true_current, base_current),
                         "teacher_current_rms_ma_cm2": float(np.sqrt(np.mean(true_current**2))),
                         "arms": arm_results}
        elapsed = time.perf_counter()-started
        remaining = elapsed/index*(len(scenarios)-index)
        print(f"[GIADA Task14] {index}/{len(scenarios)} {name} ETA {remaining/60:.1f} min", flush=True)
    for name, row in results.items():
        for arm, metrics in row["arms"].items():
            predicted = metrics.pop("_prediction_voltage")
            predicted_current = metrics.pop("_prediction_current")
            if name != "baseline":
                baseline_pred = results["baseline"]["arms"][arm]["_baseline_prediction_voltage"]
                baseline_pred_current = results["baseline"]["arms"][arm]["_baseline_prediction_current"]
                teacher_delta = flatten_role(roles[name])["end"][:, 0] - base_end[:, 0]
                metrics["predicted_delta_voltage_vs_baseline_rmse_mv"] = _rmse(predicted-baseline_pred, teacher_delta)
                true_state = flatten_role(roles[name])["end"]
                actual_gbar = np.repeat(roles[name]["gbar"], config.duration_ms)
                actual_current = analytical_current(actual_gbar, true_state[:, 0], true_state[:, 1], true_state[:, 2],
                                                    scenarios[name].get("eca", 120.))
                metrics["predicted_delta_current_vs_baseline_rmse_ma_cm2"] = _rmse(
                    predicted_current-baseline_pred_current, actual_current-base_current)
            else:
                metrics["_baseline_prediction_voltage"] = predicted
                metrics["_baseline_prediction_current"] = predicted_current
    for arm in models:
        results["baseline"]["arms"][arm].pop("_baseline_prediction_voltage")
        results["baseline"]["arms"][arm].pop("_baseline_prediction_current")
    eca_pair = (_eca_one_step_twins(base, config.eca_lower_mv),
                _eca_one_step_twins(base, config.eca_upper_mv))
    same_input = bool(np.array_equal(flatten_role(eca_pair[0])["x_full"],
                                     flatten_role(eca_pair[1])["x_full"]))
    eca_target_difference = _rmse(flatten_role(eca_pair[0])["end"][:, 0],
                                  flatten_role(eca_pair[1])["end"][:, 0])
    comparison_axes = ("gbar_half", "initial_m_plus", "initial_h_minus", "mechanism_off_visible")
    baseline_path = results["baseline"]["arms"]["path_full"]
    gates = {
        name: {
            "voltage_within_2x_baseline": results[name]["arms"]["path_full"]["one_step_voltage_rmse_mv"]
                    <= 2*baseline_path["one_step_voltage_rmse_mv"],
            "m_within_2x_baseline": results[name]["arms"]["path_full"]["one_step_m_rmse"]
                    <= 2*baseline_path["one_step_m_rmse"],
            "current_within_3x_baseline": results[name]["arms"]["path_full"]["one_step_current_rmse_ma_cm2"]
                    <= 3*baseline_path["one_step_current_rmse_ma_cm2"],
        } for name in comparison_axes
    }
    metric_names = ("one_step_voltage_rmse_mv", "one_step_m_rmse", "one_step_h_rmse",
                    "one_step_current_rmse_ma_cm2", "recursive_16ms_voltage_rmse_mv",
                    "recursive_16ms_m_rmse")
    valid = bool(same_input and eca_target_difference > 0 and np.isfinite(baseline_max)
                 and all(r["arms"][a]["recursive_valid"]
                         and all(r["arms"][a][key] is not None
                                 and np.isfinite(r["arms"][a][key]) for key in metric_names)
                         for r in results.values() for a in models))
    return {"schema_version": "giada-roadmap-task14-parameter-matrix-v1", "valid": valid,
            "code_scope": "Ca_HVA+pas one compartment; synthetic Task11 reference generalized to E_Ca",
            "task12_13_provenance": provenance, "task11_freeze_sha256": freeze["freeze_sha256"],
            "config": asdict(config), "scenario_count": len(results), "baseline_reference_max_abs_error": baseline_max,
            "eca_one_step_twins_equal_numeric_input_tensor": same_input,
            "eca_equal_input_teacher_voltage_difference_rms_mv": eca_target_difference,
            "eca_pair_unavoidable_worst_rmse_lower_bound_mv": eca_target_difference/2,
            "eca_full_trajectory_inputs_identical": False,
            "eca_is_model_input": False, "mask_is_model_input": False,
            "effective_gbar_is_model_input": True,
            "hidden_mask_is_negative_control_not_model_failure_claim": True,
            "no_training_or_model_selection": True,
            "registered_in_domain_performance_gates": gates,
            "results": results}
