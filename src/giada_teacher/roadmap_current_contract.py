"""Roadmap Tasks 12 and 13: analytical current and native sample timing."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from .cahva_boundary_semantics_reassessment import verified_task7b_trace_bytes
from .roadmap_causal_operator import (
    CausalOperatorConfig, _build_model, _decode, _predict_numpy, flatten_role,
    generate_role, integrate_predicted_path,
)

EXPECTED_TASK11_FINAL_SHA256 = "60643af34b7ed54b3d900670d3bed906d343de6aaf803d6f2a2846d6b26f2835"
EXPECTED_TASK11_FREEZE_FILE_SHA256 = "f99edf0df2f5f6b009a5ee33e33d6a08fef6f3e0816e3c2d95d12dad26e4c463"


def analytical_conductance(gbar, m, h):
    return np.asarray(gbar) * np.asarray(m) ** 2 * np.asarray(h)


def analytical_current(gbar, voltage, m, h, eca=120.0):
    """Ca_HVA current density, mA/cm²; inward current is negative."""
    return analytical_conductance(gbar, m, h) * (np.asarray(voltage) - eca)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _task11_members(source):
    source = Path(source)
    if source.is_dir():
        roots = [p.parent for p in source.rglob("selection_freeze.json")
                 if (p.parent / "final_report.json").is_file()]
        if len(roots) != 1:
            raise ValueError("Task 11: expected one extracted artifact root")
        root = roots[0]
        return lambda name: (root / name).read_bytes()
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("Task 11 source must be a ZIP or extracted directory")
    with zipfile.ZipFile(source) as outer:
        names = outer.namelist()
        freezes = [n for n in names if n.endswith("/selection_freeze.json")]
        if len(freezes) == 1:
            prefix = freezes[0][:-len("selection_freeze.json")]
            members = {n[len(prefix):]: outer.read(n) for n in names if n.startswith(prefix) and not n.endswith("/")}
        else:
            nested = [n for n in names if n.endswith("giada_roadmap_task11_causal_operator.zip")]
            if len(nested) != 1:
                raise ValueError("Task 11 members not found in archive")
            with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                inner_names = inner.namelist()
                inner_freezes = [n for n in inner_names if n.endswith("/selection_freeze.json")]
                if len(inner_freezes) != 1:
                    raise ValueError("nested Task 11 archive has no unique freeze")
                inner_prefix = inner_freezes[0][:-len("selection_freeze.json")]
                members = {n[len(inner_prefix):]: inner.read(n) for n in inner_names
                           if n.startswith(inner_prefix) and not n.endswith("/")}
    return lambda name: members[name]


def verified_task11_checkpoints(source):
    """Verify freeze and all checkpoints before exposing the frozen selections."""
    read = _task11_members(source)
    if (_sha(read("final_report.json")) != EXPECTED_TASK11_FINAL_SHA256
            or _sha(read("selection_freeze.json")) != EXPECTED_TASK11_FREEZE_FILE_SHA256):
        raise ValueError("Task 11 immutable report/freeze member mismatch")
    freeze = json.loads(read("selection_freeze.json"))
    claimed = freeze.pop("freeze_sha256")
    if _sha(json.dumps(freeze, sort_keys=True).encode()) != claimed:
        raise ValueError("Task 11 selection freeze SHA-256 mismatch")
    freeze["freeze_sha256"] = claimed
    final = json.loads(read("final_report.json"))
    if (not final.get("valid")
            or final.get("schema_version") != "giada-roadmap-task11-causal-operator-v1"
            or final.get("code_revision") != "c2ffeb54133ccfce284518561b83e59a05f6dee9"
            or final.get("selection_freeze_sha256") != claimed
            or freeze["sealed_test_accessed"]
            or freeze["selection_source"] != "development_only_final_checkpoint"):
        raise ValueError("Task 11 validity or selection contract mismatch")
    checkpoints = {}
    for row in freeze["records"].values():
        name = row["checkpoint"]
        payload = read("checkpoints/" + name)
        if _sha(payload) != row["checkpoint_sha256"]:
            raise ValueError("Task 11 checkpoint SHA-256 mismatch: " + name)
        checkpoints[name] = payload
    return freeze, checkpoints


def _frozen_model(torch, freeze, checkpoints, arm, device):
    row = freeze["records"][freeze["selected"][arm]]
    checkpoint = torch.load(io.BytesIO(checkpoints[row["checkpoint"]]), map_location=device, weights_only=True)
    if checkpoint["arm"] != arm or checkpoint["seed"] != row["seed"]:
        raise ValueError("Task 11 checkpoint identity mismatch")
    model = _build_model(torch, 64, 8, 4 if arm == "path_full" else 5).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _metric(predicted, target, mask):
    a, b = np.asarray(predicted)[mask], np.asarray(target)[mask]
    if not len(a):
        return {"count": 0, "rmse_ma_cm2": None, "mae_ma_cm2": None}
    return {"count": int(len(a)), "rmse_ma_cm2": float(np.sqrt(np.mean((a-b)**2))),
            "mae_ma_cm2": float(np.mean(np.abs(a-b)))}


def evaluate_task12(task11_source, *, device="cpu", episode_count=48, seed=12059):
    """Independent reference episodes; no fitting or model selection."""
    import torch
    freeze, checkpoints = verified_task11_checkpoints(task11_source)
    config = CausalOperatorConfig()
    role = generate_role(seed, episode_count, config)
    data = flatten_role(role)
    teacher = data["end"]
    gbar = data["gbar"]
    true_current = analytical_current(gbar, teacher[:, 0], teacher[:, 1], teacher[:, 2])
    signal = np.abs(true_current)
    threshold = float(np.quantile(signal[signal > 0], .75)) if np.any(signal > 0) else 0.
    masks = {"all": np.ones(len(signal), dtype=bool),
             "active_top_quartile_nonzero": signal >= threshold if threshold > 0 else signal > 0}
    arms = {}
    for arm in ("path_full", "effect_full"):
        model = _frozen_model(torch, freeze, checkpoints, arm, device)
        predicted, _ = _predict_numpy(torch, model, arm, role, device)
        if not np.isfinite(predicted).all():
            raise RuntimeError("nonfinite frozen Task 11 state predictions")
        comparisons = {
            "all_predicted": analytical_current(gbar, predicted[:, 0], predicted[:, 1], predicted[:, 2]),
            "gate_only_error": analytical_current(gbar, teacher[:, 0], predicted[:, 1], predicted[:, 2]),
            "voltage_only_error": analytical_current(gbar, predicted[:, 0], teacher[:, 1], teacher[:, 2]),
        }
        arms[arm] = {name: {regime: _metric(values, true_current, mask) for regime, mask in masks.items()}
                     for name, values in comparisons.items()}
        arms[arm]["state_rmse"] = {name: float(np.sqrt(np.mean((predicted[:, i]-teacher[:, i])**2)))
                                   for i, name in enumerate(("voltage_mv", "m", "h"))}
    return {"schema_version": "giada-roadmap-task12-current-v1", "valid": True,
            "input_task11_freeze_sha256": freeze["freeze_sha256"],
            "independent_role_seed": seed, "episode_count": episode_count,
            "window_count": len(signal), "current_unit": "mA/cm2",
            "active_threshold_abs_current_ma_cm2": threshold,
            "teacher_current_rms_ma_cm2": float(np.sqrt(np.mean(true_current**2))),
            "arms": arms, "training_performed": False, "model_selection_performed": False,
            "scope": "one-step analytic current from frozen states; Ca_HVA+pas one compartment"}


def evaluate_task13(task7b_source):
    """Compare native current samples with causal pre/post boundary assignments."""
    report, raw = verified_task7b_trace_bytes(task7b_source)
    errors = {name: [] for name in ("pre_all", "post_all", "pre_gates_post_voltage", "post_gates_pre_voltage")}
    active_errors = {name: [] for name in errors}
    g_errors = {"pre_gates": [], "post_gates": []}
    episode_count = 0
    with np.load(io.BytesIO(raw)) as arrays:
        keys = sorted(k[:-8] for k in arrays.files if k.endswith("_teacher"))
        for key in keys:
            values = np.asarray(arrays[key + "_teacher"], dtype=np.float64)
            gbar = 1e-5 * float(report["episodes"][key]["gbar_multiplier"])
            if gbar <= 0:
                continue  # zero-current controls cannot identify the temporal contract
            episode_count += 1
            old, new = values[:-1], values[1:]
            current = new[:, 4]
            eca = new[:, 5]
            variants = {
                "pre_all": analytical_current(gbar, old[:, 1], old[:, 2], old[:, 3], eca),
                "post_all": analytical_current(gbar, new[:, 1], new[:, 2], new[:, 3], eca),
                "pre_gates_post_voltage": analytical_current(gbar, new[:, 1], old[:, 2], old[:, 3], eca),
                "post_gates_pre_voltage": analytical_current(gbar, old[:, 1], new[:, 2], new[:, 3], eca),
            }
            active = np.abs(current) >= max(1e-12, float(np.quantile(np.abs(current), .75)))
            for name, predicted in variants.items():
                errors[name].append(predicted-current)
                active_errors[name].append((predicted-current)[active])
            # Effective conductance is inferred from native I and the identified pre-step driving force.
            safe = np.abs(old[:, 1]-eca) > 1.
            observed_g = current[safe]/(old[safe, 1]-eca[safe])
            g_errors["pre_gates"].append(analytical_conductance(gbar, old[safe, 2], old[safe, 3])-observed_g)
            g_errors["post_gates"].append(analytical_conductance(gbar, new[safe, 2], new[safe, 3])-observed_g)
    if episode_count == 0:
        raise RuntimeError("no nonzero-gbar native episodes")
    summarize = lambda arrays: {"count": int(sum(len(a) for a in arrays)),
                                "rmse": float(np.sqrt(np.mean(np.concatenate(arrays)**2))),
                                "maximum_absolute_error": float(np.max(np.abs(np.concatenate(arrays))))}
    all_metrics = {k: summarize(v) for k, v in errors.items()}
    active_metrics = {k: summarize(v) for k, v in active_errors.items()}
    winner = min(active_metrics, key=lambda k: active_metrics[k]["rmse"])
    valid = winner == "pre_all" and all_metrics["pre_all"]["maximum_absolute_error"] < 1e-10
    return {"schema_version": "giada-roadmap-task13-timing-v1", "valid": valid,
            "native_episode_count": episode_count, "native_task7b_valid": report["valid"],
            "native_current_unit": "mA/cm2", "sample_dt_ms": .025,
            "current_all_samples": all_metrics, "current_active_samples": active_metrics,
            "inferred_conductance_metrics_us_cm2": {k: summarize([a*1e6 for a in v]) for k,v in g_errors.items()},
            "winner": winner,
            "interpretation": "sampled native I(t+dt) uses pre-step V,m,h; sampled V,m,h at t+dt are post-step",
            "conductance_caveat": "gCa_HVA was not stored directly; observed conductance is inferred from I/(V_pre-E_Ca)",
            "scope": "fixed-step .025 ms Task 7b Ca_HVA+pas traces, not arbitrary CVode/full-neuron semantics"}
