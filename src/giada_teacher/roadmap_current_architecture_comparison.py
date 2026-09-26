"""GIADA roadmap Task 15: information-matched Ca-HVA current architectures.

All three learned arms receive the same causal ten-dimensional tensor.  The
teacher target is endpoint analytical current, never the lagged native `ica`
sample.  This is a one-compartment playground, not Gate C promotion.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .roadmap_causal_operator import (
    CausalOperatorConfig, _build_model, flatten_role, generate_role,
    integrate_predicted_path,
)
from .roadmap_current_contract import analytical_current
from .roadmap_parameter_variation import (
    ParameterVariationConfig, _reference_role, _scenario_definitions,
)


EXPECTED_TASK14_REPORT_SHA256 = "0e1aed13dd73a78084c4009311b071b2445f5f9d8d823d4fed5201accb187a2e"
CURRENT_SCALE_MA_CM2 = 1e-4


@dataclass(frozen=True)
class CurrentArchitectureConfig:
    train_seed: int = 15011
    development_seed: int = 15029
    sealed_seed: int = 15059
    counterfactual_seed: int = 15159
    train_episodes: int = 192
    development_episodes: int = 48
    sealed_episodes: int = 48
    counterfactual_episodes: int = 48
    duration_ms: int = 16
    seeds: tuple[int, ...] = (17, 29, 43)
    steps: int = 400
    checkpoints: tuple[int, ...] = (50, 100, 200, 400)
    batch_size: int = 256
    decomposed_width: int = 62
    direct_width: int = 64
    auxiliary_width: int = 62
    learning_rate: float = .001
    auxiliary_path_weight: float = .5
    parameter_mask_off_fraction: float = .125

    def validate(self):
        if asdict(self) != asdict(CurrentArchitectureConfig()):
            raise ValueError("Task15 configuration differs from preregistration")


def verified_task14_report(source):
    """Verify exact Task14 member even when Kaggle repacks its ZIP."""
    source = Path(source)
    filename = "task14_parameter_matrix_report.json"
    if source.is_dir():
        matches = list(source.rglob(filename))
        if len(matches) != 1:
            raise ValueError("Task14 extracted report not unique")
        raw = matches[0].read_bytes()
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as outer:
            matches = [n for n in outer.namelist() if n.endswith("/"+filename)]
            if len(matches) == 1:
                raw = outer.read(matches[0])
            else:
                nested = [n for n in outer.namelist() if n.endswith("giada_roadmap_task14_parameter_variation.zip")]
                if len(nested) != 1:
                    raise ValueError("Task14 ZIP or nested ZIP not found")
                with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                    inner_matches = [n for n in inner.namelist() if n.endswith("/"+filename)]
                    if len(inner_matches) != 1:
                        raise ValueError("nested Task14 report not unique")
                    raw = inner.read(inner_matches[0])
    else:
        raise ValueError("Task14 source must be ZIP or extracted directory")
    digest = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    if (digest != EXPECTED_TASK14_REPORT_SHA256
            or report.get("schema_version") != "giada-roadmap-task14-parameter-matrix-v1"
            or report.get("code_revision") != "3372465ff37dd8e10b4eaed5f12e73f92dc0f036"
            or not report.get("valid") or report.get("scenario_count") != 9
            or not report.get("eca_one_step_twins_equal_numeric_input_tensor")):
        raise ValueError("Task14 immutable result contract mismatch")
    return {"report_sha256": digest, "scenario_count": 9,
            "all_registered_in_domain_gates_passed": all(
                all(row.values()) for row in report["registered_in_domain_performance_gates"].values())}


def _parameterized_role(seed, count, config):
    base = generate_role(seed, count, CausalOperatorConfig())
    rng = np.random.default_rng(seed + 150000)
    scale = rng.choice(np.array([.5, 1., 1.5]), size=count)
    eca = rng.uniform(100., 140., size=count)
    mask = (rng.random(count) >= config.parameter_mask_off_fraction).astype(np.float64)
    m_shift = rng.choice(np.array([-.1, 0., .1]), size=count)
    h_shift = rng.choice(np.array([-.1, 0., .1]), size=count)
    role = _reference_role(base, gbar_multiplier=scale, eca=eca,
                           m_shift=m_shift, h_shift=h_shift, mask=mask)
    role["nominal_gbar"] = base["gbar"]*scale
    role["mask"] = mask
    role["eca"] = eca
    return role


def _counterfactual_roles(config):
    base = generate_role(config.counterfactual_seed, config.counterfactual_episodes,
                         CausalOperatorConfig())
    scenarios = _scenario_definitions(ParameterVariationConfig())
    roles = {}
    for name, spec in scenarios.items():
        if name == "mechanism_off_hidden":
            continue  # deliberately missing mask is not a fair Task15 input
        multiplier = spec.get("gbar_multiplier", 1.)
        mask = spec.get("mask", 1.)
        role = _reference_role(base, gbar_multiplier=multiplier,
                               eca=spec.get("eca", 120.), m_shift=spec.get("m_shift", 0.),
                               h_shift=spec.get("h_shift", 0.), mask=mask)
        role["nominal_gbar"] = base["gbar"]*multiplier
        role["mask"] = np.full(config.counterfactual_episodes, mask)
        role["eca"] = np.full(config.counterfactual_episodes, spec.get("eca", 120.))
        roles[name] = role
    return roles


def common_tensor(role):
    """[V0,m0,h0,gbar_nominal,mask,E_Ca,I_0,...,I_3], all causal."""
    states = role["states"]
    count, duration = states.shape[:2]
    initial = states[:, :, 0, :].reshape(-1, 3)
    nominal = np.repeat(np.asarray(role["nominal_gbar"]), duration)
    mask = np.repeat(np.asarray(role["mask"]), duration)
    eca = np.repeat(np.asarray(role["eca"]), duration)
    schedule = role["current_na"].reshape(-1, 4)
    x = np.column_stack(((initial[:, 0]+60.)/40., initial[:, 1], initial[:, 2],
                         nominal*1e5/8., mask, (eca-120.)/40., schedule/.08))
    return x.astype(np.float32)


def prepare_current_role(role):
    data = flatten_role(role)
    current = analytical_current(data["gbar"], data["end"][:, 0],
                                 data["end"][:, 1], data["end"][:, 2],
                                 np.repeat(role["eca"], role["states"].shape[1]))
    x = common_tensor(role)
    if x.shape[1] != 10 or not np.isfinite(x).all() or not np.isfinite(current).all():
        raise RuntimeError("nonfinite or wrong-width Task15 role")
    return {"x": x, "initial": data["initial"], "knots": data["knots"],
            "endpoint": data["end"], "current": current,
            "gbar": data["gbar"], "eca": np.repeat(role["eca"], role["states"].shape[1]),
            "mask": np.repeat(role["mask"], role["states"].shape[1]),
            "episode_count": role["states"].shape[0], "duration_ms": role["states"].shape[1]}


def _decode_outputs(arm, raw, x):
    v0 = x[:, 0]*40.-60.
    if arm == "direct_current":
        return None, None, raw[:, 0]*CURRENT_SCALE_MA_CM2
    path = v0[:, None] + 20.*raw[:, :4]
    gates = raw[:, 4:6].sigmoid()
    current = raw[:, 6]*CURRENT_SCALE_MA_CM2 if arm == "auxiliary_current" else None
    return path, gates, current


def _torch_loss(torch, arm, model, x, knots, endpoint_gates, current):
    raw = model(x)
    path, gates, current_prediction = _decode_outputs(arm, raw, x)
    state_loss = torch.mean(((path-knots)/20.)**2) + 4*torch.mean((gates-endpoint_gates)**2) if path is not None else None
    if arm == "decomposed_formula":
        return state_loss
    current_loss = torch.mean(((current_prediction-current)/CURRENT_SCALE_MA_CM2)**2)
    if arm == "direct_current":
        return current_loss
    return current_loss + .5*state_loss


def _predict(torch, arm, model, prepared, device):
    x = torch.as_tensor(prepared["x"], device=device)
    path_parts, gate_parts, current_parts = [], [], []
    with torch.inference_mode():
        for part in x.split(4096):
            path, gates, current = _decode_outputs(arm, model(part), part)
            if path is not None:
                path_parts.append(path.cpu().numpy())
                gate_parts.append(gates.cpu().numpy())
            if current is not None:
                current_parts.append(current.cpu().numpy())
    if arm == "direct_current":
        return {"current": np.concatenate(current_parts).astype(np.float64),
                "state": None, "path": None}
    path = np.concatenate(path_parts).astype(np.float64)
    gates = np.concatenate(gate_parts).astype(np.float64)
    state = np.column_stack((path[:, -1], gates))
    if arm == "decomposed_formula":
        current = analytical_current(prepared["gbar"], state[:, 0], state[:, 1],
                                     state[:, 2], prepared["eca"])
    else:
        current = np.concatenate(current_parts).astype(np.float64)
    return {"current": current, "state": state, "path": path}


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def _current_metrics(predicted, prepared):
    actual = prepared["current"]
    signal = np.abs(actual)
    nonzero = signal > 0
    threshold = float(np.quantile(signal[nonzero], .75)) if nonzero.any() else 0.
    active = signal >= threshold if threshold > 0 else nonzero
    off = prepared["mask"] == 0
    metrics = {"all_rmse_ma_cm2": _rmse(predicted, actual),
               "all_mae_ma_cm2": float(np.mean(np.abs(predicted-actual))),
               "active_count": int(active.sum()),
               "active_rmse_ma_cm2": _rmse(predicted[active], actual[active]) if active.any() else None,
               "off_count": int(off.sum()),
               "off_rmse_ma_cm2": _rmse(predicted[off], actual[off]) if off.any() else None,
               "positive_predicted_inward_current_count": int(np.count_nonzero(
                   (actual < 0) & (predicted > 0))),
               "finite": bool(np.isfinite(predicted).all())}
    return metrics


def _evaluation(torch, arm, model, prepared, device):
    prediction = _predict(torch, arm, model, prepared, device)
    current = prediction["current"]
    metrics = _current_metrics(current, prepared)
    if prediction["state"] is not None:
        state = prediction["state"]
        metrics["voltage_rmse_mv"] = _rmse(state[:, 0], prepared["endpoint"][:, 0])
        metrics["m_rmse"] = _rmse(state[:, 1], prepared["endpoint"][:, 1])
        metrics["h_rmse"] = _rmse(state[:, 2], prepared["endpoint"][:, 2])
        metrics["gate_occupancy_violations"] = int(np.count_nonzero(
            (state[:, 1:] < 0) | (state[:, 1:] > 1)))
        probe = integrate_predicted_path(prepared["initial"], prediction["path"])
        metrics["learned_vs_analytic_gate_probe_rmse"] = {
            "m": _rmse(state[:, 1], probe[:, 1]),
            "h": _rmse(state[:, 2], probe[:, 2]),
            "analytic_m_vs_teacher": _rmse(probe[:, 1], prepared["endpoint"][:, 1]),
            "analytic_h_vs_teacher": _rmse(probe[:, 2], prepared["endpoint"][:, 2])}
    else:
        metrics.update({"voltage_rmse_mv": None, "m_rmse": None,
                        "h_rmse": None, "gate_occupancy_violations": None})
    if arm in ("direct_current", "auxiliary_current"):
        projected = current*prepared["mask"]
        metrics["known_mask_projection_all_rmse_ma_cm2"] = _rmse(projected, prepared["current"])
        if arm == "auxiliary_current":
            state = prediction["state"]
            analytic = analytical_current(prepared["gbar"], state[:, 0], state[:, 1],
                                          state[:, 2], prepared["eca"])
            metrics["direct_vs_own_analytic_identity_rmse_ma_cm2"] = _rmse(current, analytic)
    return metrics, prediction


def _checkpoint_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _freeze_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_frozen(torch, freeze, arm, output_dir, device):
    row = freeze["selected"][arm]
    path = Path(output_dir)/"checkpoints"/row["checkpoint"]
    if _checkpoint_sha(path) != row["checkpoint_sha256"]:
        raise RuntimeError("Task15 frozen checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["arm"] != arm or checkpoint["seed"] != row["seed"]:
        raise RuntimeError("Task15 frozen checkpoint identity mismatch")
    architecture = {"decomposed_formula": (62, 6), "direct_current": (64, 1),
                    "auxiliary_current": (62, 7)}
    width, output_width = architecture[arm]
    model = _build_model(torch, width, 10, output_width).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def train_and_freeze(task14_source, output_dir, config=CurrentArchitectureConfig(), *, device="cpu"):
    """Train three arms on paired streams; freeze on development only."""
    import torch
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Task15 output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir/"checkpoints").mkdir()
    prerequisite = verified_task14_report(task14_source)
    train_role = _parameterized_role(config.train_seed, config.train_episodes, config)
    development_role = _parameterized_role(config.development_seed, config.development_episodes, config)
    train = prepare_current_role(train_role)
    development = prepare_current_role(development_role)
    contract = {"schema_version": "giada-roadmap-task15-prepare-v1",
                "task14": prerequisite, "config": asdict(config),
                "train_windows": len(train["x"]), "development_windows": len(development["x"]),
                "common_input_width": 10, "common_numeric_input_tensor": True,
                "input_columns": ["V0_normalized", "m0", "h0", "gbar_nominal_normalized",
                                  "mechanism_mask", "E_Ca_normalized", "I_schedule_q0",
                                  "I_schedule_q1", "I_schedule_q2", "I_schedule_q3"],
                "train_x_sha256": hashlib.sha256(train["x"].tobytes()).hexdigest(),
                "development_x_sha256": hashlib.sha256(development["x"].tobytes()).hexdigest(),
                "target_current": "analytic endpoint I=gbar_effective*m1^2*h1*(V1-E_Ca)",
                "native_ica_sample_used_as_target": False,
                "sealed_generated": False}
    (output_dir/"prepare_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    train_x = torch.as_tensor(train["x"], device=device)
    train_knots = torch.as_tensor(train["knots"], device=device, dtype=torch.float32)
    train_gates = torch.as_tensor(train["endpoint"][:, 1:3], device=device, dtype=torch.float32)
    train_current = torch.as_tensor(train["current"], device=device, dtype=torch.float32)
    arms = ("decomposed_formula", "direct_current", "auxiliary_current")
    widths = {"decomposed_formula": (config.decomposed_width, 6),
              "direct_current": (config.direct_width, 1),
              "auxiliary_current": (config.auxiliary_width, 7)}
    records = {}
    started = time.perf_counter()
    total = len(arms)*len(config.seeds)*len(config.checkpoints)
    done = 0
    for seed in config.seeds:
        batches = np.random.default_rng(seed).integers(
            0, len(train["x"]), size=(config.steps, config.batch_size), dtype=np.int64)
        batches = torch.as_tensor(batches, device=device)
        for arm in arms:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            width, output_width = widths[arm]
            model = _build_model(torch, width, 10, output_width).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            history = []
            for step in range(1, config.steps+1):
                model.train()
                idx = batches[step-1]
                loss = _torch_loss(torch, arm, model, train_x[idx],
                                   train_knots[idx], train_gates[idx], train_current[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if step in config.checkpoints:
                    model.eval()
                    metrics, _ = _evaluation(torch, arm, model, development, device)
                    if not metrics["finite"] or not np.isfinite(metrics["all_rmse_ma_cm2"]):
                        raise RuntimeError(f"nonfinite Task15 development predictions: {arm} seed{seed}")
                    history.append({"step": step, "development": metrics})
                    done += 1
                    elapsed = time.perf_counter()-started
                    remaining = elapsed/max(1, done)*(total-done)
                    print(f"[GIADA Task15] {done}/{total} {arm} seed={seed} step={step} "
                          f"dev_I={metrics['all_rmse_ma_cm2']:.3g} ETA {remaining/60:.1f} min", flush=True)
            name = f"{arm}-seed{seed}.pt"
            path = output_dir/"checkpoints"/name
            torch.save({"arm": arm, "seed": seed, "model": model.state_dict()}, path)
            records[f"{arm}-seed{seed}"] = {
                "arm": arm, "seed": seed, "checkpoint": name,
                "checkpoint_sha256": _checkpoint_sha(path),
                "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
                "history": history,
                "selection_metric_at_final_step": history[-1]["development"]["all_rmse_ma_cm2"]}
    selected = {}
    for arm in arms:
        candidates = [row for row in records.values() if row["arm"] == arm]
        winner = min(candidates, key=lambda row: (row["selection_metric_at_final_step"], row["seed"]))
        selected[arm] = {k: winner[k] for k in ("seed", "checkpoint", "checkpoint_sha256",
                                                "trainable_parameter_count", "selection_metric_at_final_step")}
    counts = [row["trainable_parameter_count"] for row in selected.values()]
    parameter_count_spread = (max(counts)-min(counts))/min(counts)
    if parameter_count_spread > .06:
        raise RuntimeError("Task15 model parameter counts exceed registered 6% spread")
    freeze = {"schema_version": "giada-roadmap-task15-freeze-v1", "config": asdict(config),
              "prerequisite_task14": prerequisite, "selected": selected, "records": records,
              "selection_source": "development_final_400_step_current_rmse_only",
              "sealed_test_accessed": False, "counterfactual_test_accessed": False,
              "paired_minibatch_indices_within_seed": True,
              "common_numeric_input_tensor": True,
              "same_steps_and_batch_size": True,
              "selected_parameter_count_relative_spread": parameter_count_spread}
    freeze["freeze_sha256"] = _freeze_hash(freeze)
    (output_dir/"selection_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return freeze


def _recursive_rollout(torch, arm, model, role, device):
    if arm == "direct_current":
        return {"available": False, "reason": "direct current head has no state transition"}
    states = role["states"]
    count, duration = states.shape[:2]
    predicted = np.empty((count, duration+1, 3), dtype=np.float64)
    predicted[:, 0] = states[:, 0, 0]
    predicted_current = np.empty((count, duration), dtype=np.float64)
    for t in range(duration):
        old = predicted[:, t]
        schedule = role["current_na"][:, t]
        x = np.column_stack(((old[:, 0]+60.)/40., old[:, 1], old[:, 2],
                             role["nominal_gbar"]*1e5/8., role["mask"],
                             (role["eca"]-120.)/40., schedule/.08)).astype(np.float32)
        with torch.inference_mode():
            xt = torch.as_tensor(x, device=device)
            path_t, gates_t, current_t = _decode_outputs(arm, model(xt), xt)
            path = path_t.cpu().numpy().astype(np.float64)
            gates = gates_t.cpu().numpy().astype(np.float64)
            direct_current = current_t.cpu().numpy().astype(np.float64) if current_t is not None else None
        predicted[:, t+1] = np.column_stack((path[:, -1], gates))
        if arm == "decomposed_formula":
            state = predicted[:, t+1]
            predicted_current[:, t] = analytical_current(
                role["gbar"], state[:, 0], state[:, 1], state[:, 2], role["eca"])
        else:
            predicted_current[:, t] = direct_current
        if (not np.isfinite(predicted[:, t+1]).all()
                or not np.isfinite(predicted_current[:, t]).all()
                or np.max(np.abs(predicted[:, t+1, 0])) > 250.):
            return {"available": True, "valid": False, "failed_step": t+1}
    actual = states[:, :, -1, :]
    actual_current = analytical_current(role["gbar"][:, None], actual[:, :, 0],
                                        actual[:, :, 1], actual[:, :, 2], role["eca"][:, None])
    return {"available": True, "valid": True, "horizon_ms": duration,
            "voltage_rmse_mv": _rmse(predicted[:, 1:, 0], actual[:, :, 0]),
            "m_rmse": _rmse(predicted[:, 1:, 1], actual[:, :, 1]),
            "current_rmse_ma_cm2": _rmse(predicted_current, actual_current),
            "model_current_did_not_drive_voltage": arm == "auxiliary_current"}


def evaluate_frozen(freeze, output_dir, config=CurrentArchitectureConfig(), *, device="cpu",
                    code_revision="unknown"):
    """Open sealed and paired counterfactual roles only after validating freeze."""
    import torch
    config.validate()
    output_dir = Path(output_dir)
    stored = json.loads((output_dir/"selection_freeze.json").read_text(encoding="utf-8"))
    claimed = stored.pop("freeze_sha256")
    if (_freeze_hash(stored) != claimed or claimed != freeze.get("freeze_sha256")
            or stored["sealed_test_accessed"] or stored["counterfactual_test_accessed"]
            or stored["selection_source"] != "development_final_400_step_current_rmse_only"):
        raise RuntimeError("invalid Task15 pre-test freeze")
    stored["freeze_sha256"] = claimed
    models = {arm: _load_frozen(torch, stored, arm, output_dir, device)
              for arm in stored["selected"]}
    sealed_role = _parameterized_role(config.sealed_seed, config.sealed_episodes, config)
    sealed = prepare_current_role(sealed_role)
    sealed_results = {}
    for arm, model in models.items():
        metrics, _ = _evaluation(torch, arm, model, sealed, device)
        metrics["recursive_16ms"] = _recursive_rollout(torch, arm, model, sealed_role, device)
        sealed_results[arm] = metrics
    paired_roles = _counterfactual_roles(config)
    prepared = {name: prepare_current_role(role) for name, role in paired_roles.items()}
    baseline = prepared["baseline"]
    paired_results = {}
    for name, data in prepared.items():
        if name == "baseline":
            continue
        teacher_delta_current = data["current"]-baseline["current"]
        teacher_delta_voltage = data["endpoint"][:, 0]-baseline["endpoint"][:, 0]
        row = {"teacher_delta_current_rms_ma_cm2": _rmse(teacher_delta_current, 0.),
               "teacher_delta_voltage_rms_mv": _rmse(teacher_delta_voltage, 0.),
               "arms": {}}
        for arm, model in models.items():
            base_prediction = _predict(torch, arm, model, baseline, device)
            modified = _predict(torch, arm, model, data, device)
            predicted_delta_current = modified["current"]-base_prediction["current"]
            arm_row = {"delta_current_rmse_ma_cm2": _rmse(predicted_delta_current, teacher_delta_current),
                       "modified_current_rmse_ma_cm2": _rmse(modified["current"], data["current"])}
            if modified["state"] is not None:
                predicted_delta_voltage = modified["state"][:, 0]-base_prediction["state"][:, 0]
                arm_row["delta_voltage_rmse_mv"] = _rmse(predicted_delta_voltage, teacher_delta_voltage)
            else:
                arm_row["delta_voltage_rmse_mv"] = None
            row["arms"][arm] = arm_row
        paired_results[name] = row
        print(f"[GIADA Task15 paired] {len(paired_results)}/{len(prepared)-1} {name}", flush=True)
    finite = all(bool(row["finite"]) and np.isfinite(row["all_rmse_ma_cm2"])
                 for row in sealed_results.values())
    a = sealed_results["decomposed_formula"]
    b = sealed_results["direct_current"]
    c = sealed_results["auxiliary_current"]
    def median_delta(arm):
        return float(np.median([row["arms"][arm]["delta_current_rmse_ma_cm2"]
                                for row in paired_results.values()]))
    b_delta, c_delta = median_delta("direct_current"), median_delta("auxiliary_current")
    decision = {
        "direct_current_advantage_over_formula": bool(
            b["all_rmse_ma_cm2"] <= .8*a["all_rmse_ma_cm2"]
            and b["active_rmse_ma_cm2"] <= a["active_rmse_ma_cm2"]),
        "auxiliary_improves_direct": bool(
            c["all_rmse_ma_cm2"] <= .9*b["all_rmse_ma_cm2"] and c_delta <= b_delta),
        "median_paired_delta_current_rmse_ma_cm2": {
            arm: median_delta(arm) for arm in models},
        "full_mechanism_or_gate_c_promotion_authorized": False,
        "direct_current_has_no_state_rollout": True,
    }
    report = {"schema_version": "giada-roadmap-task15-current-comparison-v1",
              "valid": bool(finite), "code_revision": code_revision,
              "selection_freeze_sha256": claimed,
              "task14_prerequisite": stored["prerequisite_task14"],
              "selected": stored["selected"], "roles": {
                  "train_seed": config.train_seed, "development_seed": config.development_seed,
                  "sealed_seed": config.sealed_seed, "counterfactual_seed": config.counterfactual_seed,
                  "train_episodes": config.train_episodes,
                  "development_episodes": config.development_episodes,
                  "sealed_episodes": config.sealed_episodes,
                  "counterfactual_episodes": config.counterfactual_episodes},
              "common_numeric_input_tensor": True, "input_width": 10,
              "same_seed_minibatches_and_training_budget": True,
              "selected_parameter_count_relative_spread": stored["selected_parameter_count_relative_spread"],
              "sealed_or_counterfactual_used_for_selection": False,
              "native_lagged_ica_used_as_target": False,
              "teacher_endpoint_as_model_input": False,
              "direct_current_rollout_not_claimed": True,
              "auxiliary_current_did_not_drive_voltage": True,
              "sealed_one_step_and_available_rollout": sealed_results,
              "paired_counterfactuals": paired_results,
              "registered_decision": decision,
              "scope": "Ca_HVA+pas one compartment, synthetic reference anchored to Task11/14; not full neuron or Gate C"}
    (output_dir/"final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
