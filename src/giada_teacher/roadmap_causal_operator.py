"""Roadmap Task 11: paired, causal one-compartment Ca-HVA operator matrix.

The 0.025 ms formula reference is independently checked against saved NEURON
Task 7b traces before synthetic episodes are used. All learned inference
receives only boundary state and the scheduled current for the next 1 ms.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cahva_boundary_semantics_reassessment import verified_task7b_trace_bytes
from .cahva_closed_loop_microcanary import ActiveClosedLoopCaHVAConfig, _injected_density, _pulses
from .gpu_baseline_runtime import configure_torch_runtime, paired_index_stream

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class CausalOperatorConfig:
    train_episodes: int = 192
    development_episodes: int = 48
    sealed_episodes: int = 48
    sealed_role_seed: int = 11059
    duration_ms: int = 16
    reference_dt_ms: float = .025
    seeds: tuple[int, ...] = (17, 29, 43)
    steps: int = 400
    checkpoints: tuple[int, ...] = (50, 100, 200, 400)
    batch_size: int = 256
    hidden_width: int = 64
    learning_rate: float = .001
    adaptive_voltage_threshold_mv: float = .5
    adaptive_gate_threshold: float = .01

    def validate(self):
        if asdict(self) != asdict(CausalOperatorConfig()):
            raise ValueError("Task 11 differs from preregistered matrix")


def _rates(voltage):
    """Vectorized, algebraically identical Ca_HVA.mod rate equations."""
    v = np.asarray(voltage, dtype=np.float64)
    # Match the exact equality branch in the teacher's Ca_HVA.mod.
    v = np.where(v == -27., v + .0001, v)
    delta = -27. - v
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        alpha_m = .055 * delta / np.expm1(delta / 3.8)
    alpha_m = np.where(np.abs(delta) < 1e-10, .055 * 3.8, alpha_m)
    beta_m = .94 * np.exp((-75. - v) / 17.)
    alpha_h = .000457 * np.exp((-13. - v) / 50.)
    beta_h = .0065 / (np.exp((-v - 15.) / 28.) + 1.)
    return (alpha_m / (alpha_m + beta_m), alpha_h / (alpha_h + beta_h),
            1. / (alpha_m + beta_m), 1. / (alpha_h + beta_h))


def _step(state, current_density, gbar, dt):
    """NEURON-matched order: membrane from old gates, then cnexp at new V."""
    state = np.asarray(state, dtype=np.float64)
    v, m, h = np.moveaxis(state, -1, 0)
    gca = np.asarray(gbar) * m * m * h
    capacitance = .001 / dt
    new_v = (capacitance * v + .0001 * -76. + gca * 120. + current_density) / (
        capacitance + .0001 + gca)
    minf, hinf, mtau, htau = _rates(new_v)
    a_m, a_h = np.exp(-dt / mtau), np.exp(-dt / htau)
    new_m = minf + (m - minf) * a_m
    new_h = hinf + (h - hinf) * a_h
    return np.stack((new_v, new_m, new_h), axis=-1), np.stack((a_m, a_h), axis=-1), np.stack((minf * (1-a_m), hinf * (1-a_h)), axis=-1)


def _predictor_corrector_step(state, current_density, gbar, dt):
    """Causal midpoint conductance and voltage, then exponential gate update."""
    half, _, _ = _step(state, current_density, gbar, dt/2)
    v, m, h = np.moveaxis(state, -1, 0)
    gmid = np.asarray(gbar)*half[...,1]**2*half[...,2]
    capacitance = .001/dt
    new_v = (capacitance*v+.0001*-76.+gmid*120.+current_density)/(
        capacitance+.0001+gmid)
    minf, hinf, mtau, htau = _rates((v+new_v)/2)
    new_m = minf+(m-minf)*np.exp(-dt/mtau)
    new_h = hinf+(h-hinf)*np.exp(-dt/htau)
    return np.stack((new_v,new_m,new_h),axis=-1)


def verify_native_anchor(source, formula):
    report, trace_bytes = verified_task7b_trace_bytes(source)
    with np.load(io.BytesIO(trace_bytes)) as archive:
        keys = sorted(key[:-8] for key in archive.files if key.endswith("_teacher"))
        if len(keys) != 24:
            raise RuntimeError("Task 11 needs all 24 native anchor episodes")
        maximum_rate_error = 0.
        for v in (-110., -80., -60., -45., -27.0001, -27., -26.9999, 0., 40.):
            reference = formula.rates(v)
            candidate = _rates(v)
            maximum_rate_error = max(maximum_rate_error, *(
                abs(float(candidate[i]) - reference[name]) for i, name in enumerate(
                    ("m_inf", "h_inf", "m_tau_ms", "h_tau_ms"))))
        if maximum_rate_error > 1e-9:
            raise RuntimeError("vectorized rates disagree with extracted NMODL formula")
        worst_v, worst_gate = 0., 0.
        for key in keys:
            teacher = np.asarray(archive[f"{key}_teacher"], dtype=np.float64)
            metadata = report["episodes"][key]
            area = float(metadata["area_um2"])
            gbar = 1e-5 * float(metadata["gbar_multiplier"])
            pulses = _pulses(str(metadata["protocol"]))
            state = teacher[0, 1:4].copy()
            for index in range(len(teacher)-1):
                current = _injected_density(index*.025, pulses, area)
                state, _, _ = _step(state, current, gbar, .025)
                worst_v = max(worst_v, abs(float(state[0]-teacher[index+1, 1])))
                worst_gate = max(worst_gate, float(np.max(np.abs(state[1:]-teacher[index+1, 2:4]))))
    valid = worst_v <= 1e-8 and worst_gate <= 1e-8
    result = {"valid": valid, "native_episode_count": len(keys),
              "maximum_rate_error": maximum_rate_error,
              "maximum_voltage_error_mv": worst_v,
              "maximum_gate_error": worst_gate,
              "native_report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()}
    if not valid:
        raise RuntimeError(f"Task 11 native formula anchor failed: {result}")
    return result


def _schedule(rng, count, duration):
    """All inputs are scheduled at 0.25 ms boundaries before the episode."""
    choices = np.array([0., 0., 0., 0., .008, .025, .05, .08])
    result = rng.choice(choices, size=(count, duration, 4))
    active = rng.random((count, duration)) < .42
    result[~active] = 0.
    # A distinct delayed pulse makes current-at-start genuinely ambiguous.
    result[:, 3:13, 1:] += np.where(rng.random((count, 10, 3)) < .12,
                                       rng.choice((.008, .025, .05), size=(count, 10, 3)), 0.)
    return result


def generate_role(seed, count, config):
    rng = np.random.default_rng(seed)
    schedule = _schedule(rng, count, config.duration_ms)
    initial_v = rng.uniform(-83., -47., size=count)
    minf, hinf, _, _ = _rates(initial_v)
    initial = np.stack((initial_v, np.clip(minf+rng.normal(0,.03,count),0,1),
                        np.clip(hinf+rng.normal(0,.03,count),0,1)), axis=-1)
    gbar = rng.uniform(0., 8., size=count) * 1e-5
    area = math.pi * 10. * 10.
    density = schedule / (.01 * area)
    states = np.empty((count, config.duration_ms, 41, 3), dtype=np.float64)
    coefficients = np.empty((count, config.duration_ms, 2, 2), dtype=np.float64)
    state = initial.copy()
    for millisecond in range(config.duration_ms):
        states[:, millisecond, 0] = state
        a_total = np.ones((count,2))
        b_total = np.zeros((count,2))
        for tick in range(40):
            state, a, b = _step(state, density[:, millisecond, tick//10], gbar,
                                config.reference_dt_ms)
            a_total, b_total = a*a_total, a*b_total+b
            states[:, millisecond, tick+1] = state
        coefficients[:, millisecond, 0] = a_total
        coefficients[:, millisecond, 1] = b_total
    if not np.isfinite(states).all() or np.any(states[...,1:] < -1e-9) or np.any(states[...,1:] > 1+1e-9):
        raise RuntimeError("nonfinite or unphysical reference episodes")
    return {"states": states, "current_na": schedule, "current_density": density,
            "gbar": gbar, "coefficients": coefficients, "role_seed": seed}


def flatten_role(role):
    states = role["states"]
    episodes, intervals = states.shape[:2]
    initial = states[:,:,0].reshape(-1,3)
    schedule = role["current_na"].reshape(-1,4)
    gbar = np.repeat(role["gbar"]*1e5, intervals)
    inputs = np.column_stack(((initial[:,0]+60.)/40., initial[:,1], initial[:,2],
                              gbar/8., schedule/.08))
    reduced = inputs.copy()
    reduced[:,5:] = 0.
    return {"x_full": inputs.astype(np.float32),
            "x_reduced": reduced.astype(np.float32),
            "initial": initial, "end": states[:,:,-1].reshape(-1,3),
            "knots": states[:,:,[10,20,30,40],0].reshape(-1,4),
            "coefficients": role["coefficients"].reshape(-1,2,2),
            "current_density": role["current_density"].reshape(-1,4),
            "gbar": np.repeat(role["gbar"], intervals),
            "episode_count": episodes, "intervals": intervals}


def integrate_reference(initial, current_density, gbar, steps, *, scheme="sequential"):
    if steps not in (1,2,4,8,40) or scheme not in ("sequential","predictor_corrector"):
        raise ValueError("unregistered stage count")
    state = np.asarray(initial, dtype=np.float64).copy()
    for index in range(steps):
        current = current_density[:, min(3, int((index+.5)*4/steps))]
        if scheme == "sequential":
            state, _, _ = _step(state, current, gbar, 1./steps)
        else:
            state = _predictor_corrector_step(state, current, gbar, 1./steps)
    return state


def integrate_predicted_path(initial, knots):
    state = np.asarray(initial, dtype=np.float64).copy()
    voltages = np.column_stack((initial[:,0], knots))
    for index in range(40):
        quarter = index//10
        fraction = ((index%10)+.5)/10
        voltage = voltages[:,quarter]*(1-fraction)+voltages[:,quarter+1]*fraction
        minf, hinf, mtau, htau = _rates(voltage)
        state[:,1] = minf+(state[:,1]-minf)*np.exp(-.025/mtau)
        state[:,2] = hinf+(state[:,2]-hinf)*np.exp(-.025/htau)
    state[:,0] = knots[:,-1]
    return state


def _metrics(prediction, target):
    p = np.asarray(prediction)
    y = np.asarray(target)
    return {"voltage_rmse_mv": float(np.sqrt(np.mean((p[:,0]-y[:,0])**2))),
            "m_rmse": float(np.sqrt(np.mean((p[:,1]-y[:,1])**2))),
            "h_rmse": float(np.sqrt(np.mean((p[:,2]-y[:,2])**2))),
            "open_rmse": float(np.sqrt(np.mean((p[:,1]**2*p[:,2]-y[:,1]**2*y[:,2])**2))),
            "maximum_voltage_error_mv": float(np.max(np.abs(p[:,0]-y[:,0]))),
            "occupancy_violations": int(np.count_nonzero((p[:,1:] < 0)|(p[:,1:] > 1)))}


def numerical_matrix(role, config):
    data = flatten_role(role)
    results = {}
    for steps in (1,2,4,8,40):
        start = time.perf_counter()
        predicted = integrate_reference(data["initial"], data["current_density"], data["gbar"], steps)
        results[str(steps)] = {**_metrics(predicted, data["end"]),
                               "wall_seconds": time.perf_counter()-start}
    for steps in (1,2,4,8):
        start = time.perf_counter()
        predicted = integrate_reference(data["initial"], data["current_density"],
                                        data["gbar"], steps, scheme="predictor_corrector")
        results[f"pc_{steps}"] = {**_metrics(predicted, data["end"]),
                                  "wall_seconds":time.perf_counter()-start}
    adaptive_start = time.perf_counter()
    one = integrate_reference(data["initial"], data["current_density"], data["gbar"], 1)
    two = integrate_reference(data["initial"], data["current_density"], data["gbar"], 2)
    hard = (np.abs(one[:,0]-two[:,0]) > config.adaptive_voltage_threshold_mv) | (
        np.max(np.abs(one[:,1:]-two[:,1:]),axis=1) > config.adaptive_gate_threshold)
    # Schedule changes are known before t; never gate on a teacher future state.
    hard |= np.any(np.abs(data["current_density"][:,1:]-data["current_density"][:,:-1]) > 0,axis=1)
    adaptive = two.copy()
    if np.any(hard):
        adaptive[hard] = integrate_reference(data["initial"][hard],data["current_density"][hard],
                                             data["gbar"][hard],8)
    results["adaptive_2_or_8"] = {**_metrics(adaptive,data["end"]),
                                  "fraction_8_stages": float(np.mean(hard)),
                                  "wall_seconds": time.perf_counter()-adaptive_start,
                                  "causal_indicator": "one-vs-two local estimate or known schedule change"}
    return results


def missing_input_counterfactual(role):
    """Same visible reduced input, different known current later in the ms."""
    data = flatten_role(role)
    initial = data["initial"][:64]
    gbar = data["gbar"][:64]
    zero = np.zeros((len(initial),4),dtype=np.float64)
    delayed = zero.copy()
    delayed[:,2] = .08/(.01*math.pi*10.*10.)
    first = integrate_reference(initial,zero,gbar,40)
    second = integrate_reference(initial,delayed,gbar,40)
    difference = np.abs(second-first)
    return {"paired_count":len(initial),"same_reduced_features":True,
            "maximum_voltage_difference_mv":float(np.max(difference[:,0])),
            "maximum_m_difference":float(np.max(difference[:,1])),
            "minimum_voltage_difference_mv":float(np.min(difference[:,0]))}


def _build_model(torch, width, input_width, output_width):
    return torch.nn.Sequential(torch.nn.Linear(input_width,width),torch.nn.SiLU(),
                               torch.nn.Linear(width,width),torch.nn.SiLU(),
                               torch.nn.Linear(width,output_width))


def _decode(torch, arm, raw, x):
    v0 = x[:,0]*40.-60.
    if arm == "path_full":
        return v0[:,None] + 20.*raw
    a = raw[:,:2].sigmoid()
    q = raw[:,2:4].sigmoid()
    v = v0+20.*raw[:,4]
    gate0 = x[:,1:3]
    return torch.cat((v[:,None],a*gate0+(1-a)*q,a,(1-a)*q),dim=1)


def _loss(torch, arm, decoded, batch, index):
    if arm == "path_full":
        target = batch["knots"][index]
        return torch.mean(((decoded-target)/20.)**2)
    endpoint = batch["end"][index]
    coeff = batch["coeff"][index]
    predicted_gates = decoded[:,1:3]
    return (torch.mean(((decoded[:,0]-endpoint[:,0])/20.)**2)
            + 4*torch.mean((predicted_gates-endpoint[:,1:3])**2)
            + .25*torch.mean((decoded[:,3:5]-coeff[:,0])**2)
            + .25*torch.mean((decoded[:,5:7]-coeff[:,1])**2))


def _tensors(torch, data, device):
    return {"x_full": torch.as_tensor(data["x_full"],device=device),
            "x_reduced": torch.as_tensor(data["x_reduced"],device=device),
            "end": torch.as_tensor(data["end"],device=device,dtype=torch.float32),
            "knots": torch.as_tensor(data["knots"],device=device,dtype=torch.float32),
            "coeff": torch.as_tensor(data["coefficients"],device=device,dtype=torch.float32)}


def _predict_numpy(torch, model, arm, role, device):
    data = flatten_role(role)
    key = "x_reduced" if arm == "effect_reduced" else "x_full"
    with torch.inference_mode():
        x = torch.as_tensor(data[key],device=device,dtype=torch.float32)
        outputs = []
        for part in x.split(4096):
            outputs.append(_decode(torch,arm,model(part),part).cpu().numpy())
    values = np.concatenate(outputs).astype(np.float64)
    if arm == "path_full":
        return integrate_predicted_path(data["initial"],values),values
    return values[:,:3],values


def _compact_metrics(prediction, target):
    if not np.isfinite(prediction).all():
        return {"valid":False,"selection_score":1e9,"failure":"nonfinite one-step prediction"}
    score = _metrics(prediction,target)
    score["selection_score"] = (score["voltage_rmse_mv"]/20. +
                                score["m_rmse"]+score["h_rmse"]+score["open_rmse"])
    score["valid"] = True
    return score


def _regimes(data):
    change = np.abs(data["end"][:,0]-data["initial"][:,0])
    return {"quiet_lt_1mV":change<1.,
            "moderate_1_to_5mV":(change>=1.)&(change<5.),
            "active_ge_5mV":change>=5.}


def _stratified_metrics(prediction, data):
    result = {}
    for name, mask in _regimes(data).items():
        result[name] = {"count":int(mask.sum()),
                        "metrics":_metrics(prediction[mask],data["end"][mask]) if mask.any() else None}
    return result


def train_matrix(train_role, development_role, config, output_dir):
    torch = configure_torch_runtime(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 11 registered learned matrix requires a CUDA GPU")
    device = torch.device("cuda")
    train = flatten_role(train_role)
    dev = flatten_role(development_role)
    tensors = _tensors(torch,train,device)
    records = {}
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir/"checkpoints"
    checkpoint_dir.mkdir(parents=True,exist_ok=False)
    for seed in config.seeds:
        stream = paired_index_stream(len(train["initial"]),config.batch_size,config.steps,seed)
        for arm in ("effect_full","effect_reduced","path_full"):
            torch.manual_seed(seed)
            in_width = 8  # identical capacity; hidden input columns are masked
            out_width = 4 if arm == "path_full" else 5
            model = _build_model(torch,config.hidden_width,in_width,out_width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
            key = f"{arm}-seed{seed}"
            history = []
            started = time.perf_counter()
            for step, ids in enumerate(stream["batches"],1):
                idx = torch.as_tensor(ids,device=device,dtype=torch.long)
                x = tensors["x_reduced" if arm == "effect_reduced" else "x_full"][idx]
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(torch,arm,_decode(torch,arm,model(x),x),tensors,idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step in config.checkpoints:
                    prediction, _ = _predict_numpy(torch,model,arm,development_role,device)
                    metrics = _compact_metrics(prediction,dev["end"])
                    history.append({"step":step,"development":metrics})
                    print(f"[GIADA Task 11][{key}] {step}/{config.steps} dev_score={metrics['selection_score']:.5f} V={metrics['voltage_rmse_mv']:.4f} m={metrics['m_rmse']:.5f}",flush=True)
            torch.cuda.synchronize()
            filename = f"{key}.pt"
            checkpoint_path = checkpoint_dir/filename
            torch.save({"model":model.state_dict(),"arm":arm,"seed":seed},checkpoint_path)
            records[key] = {"arm":arm,"seed":seed,"history":history,
                            "seconds":time.perf_counter()-started,
                            "checkpoint":filename,
                            "checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                            "stream_sha256":stream["sha256"],
                            "parameter_count":sum(p.numel() for p in model.parameters())}
    selected = {}
    for arm in ("effect_full","effect_reduced","path_full"):
        candidates = [(records[f"{arm}-seed{seed}"]["history"][-1]["development"]["selection_score"],seed)
                      for seed in config.seeds]
        _, chosen = min(candidates)
        selected[arm] = f"{arm}-seed{chosen}"
    freeze = {"selected":selected,"records":records,
              "selection_source":"development_only_final_checkpoint",
              "sealed_test_accessed":False,"paired_streams":True,
              "train_episode_seed":11011,"development_episode_seed":11029,
              "config":asdict(config)}
    encoded = json.dumps(freeze,sort_keys=True).encode()
    freeze["freeze_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output_dir/"selection_freeze.json").write_text(json.dumps(freeze,indent=2))
    return freeze


def _load_selected(torch, freeze, arm, output_dir, device):
    key = freeze["selected"][arm]
    row = freeze["records"][key]
    model = _build_model(torch,64,8,
                         4 if arm=="path_full" else 5).to(device)
    checkpoint_path = Path(output_dir)/"checkpoints"/row["checkpoint"]
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=row["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path,
                            map_location=device,weights_only=True)
    if checkpoint["arm"] != arm or checkpoint["seed"] != row["seed"]:
        raise RuntimeError("selected checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _rollout(torch, arm, model, role, device):
    states = role["states"]
    count, duration = states.shape[:2]
    predicted = np.empty((count,duration+1,3),dtype=np.float64)
    predicted[:,0] = states[:,0,0]
    inputs = role["current_na"]
    for t in range(duration):
        current_state = predicted[:,t]
        x = np.column_stack(((current_state[:,0]+60)/40,current_state[:,1],current_state[:,2],
                             role["gbar"]*1e5/8,inputs[:,t]/.08)).astype(np.float32)
        if arm == "effect_reduced":x[:,5:] = 0.
        with torch.inference_mode():
            xt = torch.as_tensor(x,device=device)
            decoded = _decode(torch,arm,model(xt),xt).cpu().numpy().astype(np.float64)
        if not np.isfinite(decoded).all() or np.max(np.abs(decoded[:,0 if arm!="path_full" else slice(None)])) > 250.:
            return {"valid":False,"failed_interval_ms":t+1,
                    "failure":"nonfinite or extreme predicted voltage"}
        if arm == "path_full":
            predicted[:,t+1] = integrate_predicted_path(current_state,decoded)
        else:
            predicted[:,t+1] = decoded[:,:3]
        if not np.isfinite(predicted[:,t+1]).all() or np.max(np.abs(predicted[:,t+1,0])) > 250.:
            return {"valid":False,"failed_interval_ms":t+1,
                    "failure":"recursive state nonfinite or beyond 250 mV"}
    reference = np.concatenate((states[:,0,0,None,:],states[:,:,-1,:]),axis=1)
    p, y = predicted[:,1:].reshape(-1,3),reference[:,1:].reshape(-1,3)
    threshold = 0.
    pred_events = (predicted[:,1:,0]>=threshold)&(predicted[:,:-1,0]<threshold)
    true_events = (reference[:,1:,0]>=threshold)&(reference[:,:-1,0]<threshold)
    supported = bool(true_events.any() and (~true_events).any())
    return {**_metrics(p,y),"valid":True,"horizon_ms":duration,
            "predicted_threshold_crossings":int(pred_events.sum()),
            "teacher_threshold_crossings":int(true_events.sum()),
            "event_f1_has_support":supported,
            "exact_interval_event_f1":(float(2*np.logical_and(pred_events,true_events).sum()/
                max(1,pred_events.sum()+true_events.sum())) if supported else None)}


def evaluate_frozen(freeze, sealed_role, config, output_dir, native_anchor, *, code_revision="unknown"):
    claimed = freeze.get("freeze_sha256")
    content = {k:v for k,v in freeze.items() if k!="freeze_sha256"}
    if (hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()!=claimed
            or freeze["sealed_test_accessed"] or freeze["selection_source"]!="development_only_final_checkpoint"):
        raise RuntimeError("invalid Task 11 development freeze")
    torch = configure_torch_runtime(17)
    device = torch.device("cuda")
    if not torch.cuda.is_available():raise RuntimeError("Task 11 sealed evaluation requires CUDA")
    data = flatten_role(sealed_role)
    analytic = numerical_matrix(sealed_role,config)
    oracle_path = integrate_predicted_path(data["initial"],
                                           sealed_role["states"][:,:,[10,20,30,40],0].reshape(-1,4))
    coefficients = data["coefficients"]
    oracle_effect = np.column_stack((data["end"][:,0],
                                     coefficients[:,0]*data["initial"][:,1:]+coefficients[:,1]))
    predictions = {}
    probes = {}
    strata = {}
    rollouts = {}
    for arm in ("effect_full","effect_reduced","path_full"):
        model = _load_selected(torch,freeze,arm,output_dir,device)
        prediction,decoded = _predict_numpy(torch,model,arm,sealed_role,device)
        prediction_valid = bool(np.isfinite(prediction).all())
        predictions[arm] = ({**_metrics(prediction,data["end"]),"valid":True} if prediction_valid
                            else {"valid":False,"failure":"nonfinite one-step prediction"})
        strata[arm] = _stratified_metrics(prediction,data) if prediction_valid else None
        if not np.isfinite(decoded).all():
            probes[arm] = {"valid":False,"failure":"nonfinite internal probe"}
        elif arm == "path_full":
            probes[arm] = {"knot_voltage_rmse_mv":float(np.sqrt(np.mean(
                (decoded-data["knots"])**2)))}
        else:
            probes[arm] = {"A_rmse":float(np.sqrt(np.mean((decoded[:,3:5]-coefficients[:,0])**2))),
                           "B_rmse":float(np.sqrt(np.mean((decoded[:,5:7]-coefficients[:,1])**2)))}
        rollouts[arm] = _rollout(torch,arm,model,sealed_role,device)
    report = {"schema_version":"giada-roadmap-task11-causal-operator-v1",
              "valid":bool(native_anchor["valid"]),"code_revision":code_revision,
              "native_anchor":native_anchor,"reference":"CaHVA+pas one compartment; .025ms formula; ECa fixed",
              "external_step_ms":1,"reference_internal_dt_ms":.025,
              "roles":{"train_episodes":config.train_episodes,
                       "development_episodes":config.development_episodes,
                       "sealed_episodes":config.sealed_episodes},
              "selection_freeze_sha256":claimed,"sealed_used_for_selection":False,
              "future_teacher_voltage_as_model_input":False,
              "H0_input_contract":{"reduced":"V,m,h,gbar,current at start quarter",
                                   "full":"V,m,h,gbar,all four scheduled current quarters",
                                   "paired_arms":["effect_reduced","effect_full"],
                                   "same_reduced_input_counterfactual":missing_input_counterfactual(sealed_role)},
              "H1_coupled_numerical_stages":analytic,
              "H2_oracle_compact_path":_metrics(oracle_path,data["end"]),
              "H3_oracle_integrated_effect":_metrics(oracle_effect,data["end"]),
              "learned_one_step":predictions,"learned_internal_probes":probes,
              "sealed_regime_support":{k:int(v.sum()) for k,v in _regimes(data).items()},
              "one_step_by_regime":strata,"learned_recursive_16ms":rollouts,
              "H4_adaptive":"adaptive_2_or_8",
              "scope_limit":"No dynamic calcium, other mechanisms, synapses, axial coupling or 642-segment teacher."}
    (Path(output_dir)/"final_report.json").write_text(json.dumps(report,indent=2))
    return report
