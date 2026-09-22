"""Task 7: one-compartment causal Ca_HVA gate-current-voltage microcanary.

This does not claim validity for the 642-segment Hay teacher.  The authentic
single-compartment NEURON mechanism is an independent reference; a formula
arm measures the numerical voltage-solver discrepancy before LUT/model arms
are interpreted.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .gpu_baseline_runtime import configure_torch_runtime
from .primitive_matrix_playground import PrimitiveMatrixConfig, _file_sha, _learned_models
from .primitive_scaling import _gpu_rate_table, gpu_lut_predict
from .voltage_path_stress import verified_task5_root


@dataclass(frozen=True)
class ClosedLoopCaHVAConfig:
    dt_ms: float = 0.025
    duration_ms: float = 20.0
    initial_voltage_mv: tuple[float, ...] = (-76., -62.)
    gbar_multipliers: tuple[float, ...] = (1., 4.)
    protocol_names: tuple[str, ...] = ("rest", "single_pulse", "paired_pulse")
    section_length_um: float = 10.
    section_diameter_um: float = 10.
    cm_uf_cm2: float = 1.
    g_pas_s_cm2: float = 0.0001
    e_pas_mv: float = -76.
    e_ca_mv: float = 120.
    canonical_gbar_s_cm2: float = 0.00001
    pilot_formula_voltage_rmse_limit_mv: float = 0.25
    pilot_formula_gate_rmse_limit: float = 0.005

    def validate(self):
        if (self.dt_ms, self.duration_ms, self.initial_voltage_mv,
                self.gbar_multipliers, self.protocol_names) != (
                .025, 20., (-76., -62.), (1., 4.),
                ("rest", "single_pulse", "paired_pulse")):
            raise ValueError("Task 7 design differs from preregistration")


def _pulses(name):
    if name == "rest":
        return ()
    if name == "single_pulse":
        return ((4., 12., .003),)
    if name == "paired_pulse":
        return ((3., 7., .003), (11., 15., .003))
    raise ValueError(name)


def _injected_density(time_ms, pulses, area_um2):
    # One mA/cm2 through one um2 is 0.01 nA.
    return sum(amp for start, end, amp in pulses if start <= time_ms < end) / (.01 * area_um2)


def _teacher_episode(mechanism_root, config, initial_voltage, gbar_multiplier, protocol):
    try:
        from neuron import h
    except ImportError as exc:
        raise RuntimeError("NEURON must be installed for Task 7") from exc
    section = h.Section(name="giada_cahva_microcanary")
    section.L = config.section_length_um
    section.diam = config.section_diameter_um
    section.nseg = 1
    section.cm = config.cm_uf_cm2
    section.insert("pas")
    section.g_pas = config.g_pas_s_cm2
    section.e_pas = config.e_pas_mv
    section.insert("Ca_HVA")
    segment = section(.5)
    segment.gCa_HVAbar_Ca_HVA = config.canonical_gbar_s_cm2 * gbar_multiplier
    segment.eca = config.e_ca_mv
    clamps = []
    for start, end, amp in _pulses(protocol):
        clamp = h.IClamp(segment)
        clamp.delay, clamp.dur, clamp.amp = start, end - start, amp
        clamps.append(clamp)
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = config.dt_ms
    h.finitialize(initial_voltage)
    segment.eca = config.e_ca_mv
    count = int(round(config.duration_ms / config.dt_ms))
    rows = np.empty((count + 1, 6), dtype=np.float64)

    def capture(index):
        rows[index] = (float(h.t), float(segment.v), float(segment.m_Ca_HVA),
                       float(segment.h_Ca_HVA), float(segment.ica), float(segment.eca))

    capture(0)
    for step in range(count):
        h.fadvance()
        capture(step + 1)
    if not np.isfinite(rows).all() or np.max(np.abs(rows[:, 5] - config.e_ca_mv)) > 1e-9:
        raise RuntimeError("Nonfinite teacher state or changing E_Ca in fixed-reversal microcanary")
    # Keep section and clamps alive through the complete trajectory, then let
    # NEURON/Python release them before creating the next independent episode.
    return rows, float(h.area(.5, sec=section))


def _formula_gate(formula, voltage, m, h, dt):
    rates = formula.rates(float(voltage))
    m_next = rates["m_inf"] + (m - rates["m_inf"]) * math.exp(-dt / rates["m_tau_ms"])
    h_next = rates["h_inf"] + (h - rates["h_inf"]) * math.exp(-dt / rates["h_tau_ms"])
    return m_next, h_next


def _voltage_step(v, m, h, gbar, iinj, config):
    """Semi-implicit passive + Ca conductance voltage step (density units)."""
    gca = gbar * m * m * h
    capacitance = .001 * config.cm_uf_cm2 / config.dt_ms
    return ((capacitance * v + config.g_pas_s_cm2 * config.e_pas_mv
             + gca * config.e_ca_mv + iinj)
            / (capacitance + config.g_pas_s_cm2 + gca))


def _formula_closed_loop(formula, teacher, area_um2, config, gbar_multiplier, protocol):
    out = np.empty_like(teacher)
    out[0] = teacher[0]
    gbar = config.canonical_gbar_s_cm2 * gbar_multiplier
    pulses = _pulses(protocol)
    for step in range(len(teacher) - 1):
        time_ms, v, m, h = out[step, :4]
        mn, hn = _formula_gate(formula, v, m, h, config.dt_ms)
        iinj = _injected_density(step * config.dt_ms, pulses, area_um2)
        vn = _voltage_step(v, mn, hn, gbar, iinj, config)
        ica = gbar * mn * mn * hn * (vn - config.e_ca_mv)
        out[step + 1] = ((step + 1) * config.dt_ms, vn, mn, hn, ica, config.e_ca_mv)
    return out


def _frozen_candidates(task5_root, formula, device):
    torch = configure_torch_runtime(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 7 frozen model comparison requires CUDA")
    model = _learned_models(torch, PrimitiveMatrixConfig(physical_width=32))["physical_tau"].to(device)
    states = torch.load(task5_root / "frozen_scaling_checkpoints.pt", map_location=device, weights_only=True)
    model.load_state_dict(states["32"])
    model.eval()
    table = _gpu_rate_table(torch, formula, 513, device, torch.float32)
    return torch, model, table


def _candidate_closed_loop(torch, model, table, teacher, area_um2, config, gbar_multiplier, protocol, device):
    count = len(teacher) - 1
    out_lut = np.empty_like(teacher)
    out_lut[0] = teacher[0]
    out_model = {seed: np.empty_like(teacher) for seed in (17, 29, 43)}
    for row in out_model.values():
        row[0] = teacher[0]
    gbar = config.canonical_gbar_s_cm2 * gbar_multiplier
    state_l = torch.as_tensor(teacher[0, 1:4], device=device, dtype=torch.float32).reshape(1, 3)
    state_m = state_l.expand(3, -1).clone()
    with torch.inference_mode():
        for step in range(count):
            iinj = _injected_density(step * config.dt_ms, _pulses(protocol), area_um2)
            dt_l = torch.full((1, 1), config.dt_ms, device=device)
            x_l = torch.cat((state_l, dt_l), dim=1)
            gates_l = gpu_lut_predict(torch, x_l, table, linear=True)
            v_l = _voltage_step(state_l[:, 0], gates_l[:, 0], gates_l[:, 1], gbar, iinj, config)
            state_l = torch.cat((v_l[:, None], gates_l), dim=1)

            dt_m = torch.full((3, 1, 1), config.dt_ms, device=device)
            x_m = torch.cat((state_m[:, None, :], dt_m), dim=2)
            gates_m = model(x_m)[0][:, 0, :]
            v_m = _voltage_step(state_m[:, 0], gates_m[:, 0], gates_m[:, 1], gbar, iinj, config)
            state_m = torch.cat((v_m[:, None], gates_m), dim=1)
            out_lut[step + 1, 0] = (step + 1) * config.dt_ms
            # Copy to host once after rollout; intermediate states stay on GPU.
            if step == 0:
                l_states = torch.empty((count, 3), device=device)
                m_states = torch.empty((3, count, 3), device=device)
            l_states[step] = state_l[0]
            m_states[:, step] = state_m
    l_np = l_states.cpu().double().numpy()
    m_np = m_states.cpu().double().numpy()
    out_lut[1:, 1:4] = l_np
    out_lut[1:, 4] = gbar * l_np[:, 1] ** 2 * l_np[:, 2] * (l_np[:, 0] - config.e_ca_mv)
    out_lut[1:, 5] = config.e_ca_mv
    for index, seed in enumerate((17, 29, 43)):
        out = out_model[seed]
        values = m_np[index]
        out[1:, 0] = np.arange(1, count + 1) * config.dt_ms
        out[1:, 1:4] = values
        out[1:, 4] = gbar * values[:, 1] ** 2 * values[:, 2] * (values[:, 0] - config.e_ca_mv)
        out[1:, 5] = config.e_ca_mv
    return out_lut, out_model


def _metrics(prediction, teacher):
    delta = prediction[1:, 1:5] - teacher[1:, 1:5]
    return {"voltage_rmse_mv": float(np.sqrt(np.mean(delta[:, 0] ** 2))),
            "voltage_max_abs_mv": float(np.max(np.abs(delta[:, 0]))),
            "m_rmse": float(np.sqrt(np.mean(delta[:, 1] ** 2))),
            "h_rmse": float(np.sqrt(np.mean(delta[:, 2] ** 2))),
            "ica_rmse_ma_cm2": float(np.sqrt(np.mean(delta[:, 3] ** 2))),
            "endpoint_voltage_error_mv": float(delta[-1, 0]),
            "finite": bool(np.isfinite(prediction).all()),
            "occupancy_violations": int(np.count_nonzero((prediction[:, 2:4] < 0) | (prediction[:, 2:4] > 1)))}


def run_closed_loop_microcanary(formula, task5_source, mechanism_root, output_dir,
                                config=None, *, code_revision="unknown"):
    config = config or ClosedLoopCaHVAConfig()
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir() or any(
            child.name != ".verified_task5" for child in output_dir.iterdir()
        ):
            raise FileExistsError(f"Output già presente e non vuoto: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    root = verified_task5_root(task5_source, output_dir / ".verified_task5")
    from neuron import load_mechanisms
    load_mechanisms(str(mechanism_root))
    torch = configure_torch_runtime(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch, model, table = _frozen_candidates(root, formula, device)
    episodes = {}
    trace_store = {}
    total = len(config.initial_voltage_mv) * len(config.gbar_multipliers) * len(config.protocol_names)
    started = time.perf_counter()
    for index, (initial, multiplier, protocol) in enumerate(
        ((v, g, p) for v in config.initial_voltage_mv
         for g in config.gbar_multipliers for p in config.protocol_names), start=1
    ):
        teacher, area = _teacher_episode(mechanism_root, config, initial, multiplier, protocol)
        formula_rollout = _formula_closed_loop(formula, teacher, area, config, multiplier, protocol)
        lut_rollout, physical_rollouts = _candidate_closed_loop(
            torch, model, table, teacher, area, config, multiplier, protocol, device)
        key = f"v{initial:g}-g{multiplier:g}-{protocol}"
        trace_store[f"{key}_teacher"] = teacher
        trace_store[f"{key}_formula"] = formula_rollout
        trace_store[f"{key}_lut"] = lut_rollout
        for seed, value in physical_rollouts.items():
            trace_store[f"{key}_physical_{seed}"] = value
        episodes[key] = {"initial_voltage_mv": initial, "gbar_multiplier": multiplier,
                         "protocol": protocol, "area_um2": area,
                         "teacher_voltage_range_mv": [float(teacher[:, 1].min()), float(teacher[:, 1].max())],
                         "formula": _metrics(formula_rollout, teacher),
                         "lut": _metrics(lut_rollout, teacher),
                         "physical": {str(seed): _metrics(value, teacher) for seed, value in physical_rollouts.items()}}
        eta = (time.perf_counter() - started) / index * (total - index) / 60
        print(f"[GIADA closed loop] {index}/{total} ({100*index/total:.1f}%) ETA {eta:.1f} min "
              f"formulaV={episodes[key]['formula']['voltage_rmse_mv']:.4g} "
              f"lutV={episodes[key]['lut']['voltage_rmse_mv']:.4g}", flush=True)
    formula_worst = max(row["formula"]["voltage_rmse_mv"] for row in episodes.values())
    formula_gate_worst = max(max(row["formula"]["m_rmse"], row["formula"]["h_rmse"]) for row in episodes.values())
    finite = all(row[arm]["finite"] for row in episodes.values() for arm in ("formula", "lut"))
    finite = finite and all(metric["finite"] for row in episodes.values() for metric in row["physical"].values())
    comparison_valid = bool(formula_worst <= config.pilot_formula_voltage_rmse_limit_mv
                            and formula_gate_worst <= config.pilot_formula_gate_rmse_limit)
    report = {"schema_version": "giada-task7-microcanary-v1", "code_revision": code_revision,
              "valid": bool(finite and comparison_valid), "finite": bool(finite),
              "reference_solver_calibrated": comparison_valid,
              "formula_worst_voltage_rmse_mv": formula_worst,
              "formula_worst_gate_rmse": formula_gate_worst,
              "config": asdict(config), "episode_count": len(episodes), "episodes": episodes,
              "teacher_kind": "authentic_single_compartment_NEURON_Ca_HVA_plus_pas",
              "coupling_kind": "causal_voltage_gate_current_closed_loop_at_0.025ms",
              "full_642_segment_teacher_tested": False,
              "trained_voltage_network_tested": False,
              "interpretation_policy": "Candidate errors against NEURON are interpretable only if formula arm calibrates the fixed-step membrane solver; compare candidate excess over formula before attribution."}
    np.savez_compressed(output_dir / "trajectories.npz", **trace_store)
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    return report
