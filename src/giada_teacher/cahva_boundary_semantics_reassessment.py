"""Task 7c: forensic boundary-state reassessment of the saved Task 7b run.

No new NEURON trajectory, retraining, or fresh test is generated.  The
previously saved authentic teacher traces are immutable inputs; corrected
candidate rollouts use V_(n+1) to advance the boundary gates after the
membrane step, matching NEURON's observed fixed-step state ordering.
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

from .cahva_closed_loop_microcanary import (
    ActiveClosedLoopCaHVAConfig,
    _frozen_candidates,
    _injected_density,
    _metrics,
    _pulses,
    _voltage_step,
)
from .gpu_baseline_runtime import configure_torch_runtime
from .primitive_scaling import gpu_lut_predict
from .voltage_path_stress import verified_task5_root


EXPECTED_TASK7B_REPORT_SHA256 = "17a91aaaee8652c86a48e670f22ff1ae2a98bc30fd98ee250941948878d66a71"
EXPECTED_TASK7B_TRAJECTORIES_SHA256 = "89583fdded8cceca5f7ae7fd6140a5074801b55db5080fed097db843c5474f86"


@dataclass(frozen=True)
class BoundarySemanticsConfig:
    formula_voltage_max_abs_limit_mv: float = 1e-8
    formula_gate_max_rmse_limit: float = 1e-8
    formula_current_max_abs_limit_ma_cm2: float = 1e-10
    lag_identity_max_abs_limit: float = 1e-8
    expected_episode_count: int = 24
    frozen_seeds: tuple[int, ...] = (17, 29, 43)

    def validate(self) -> None:
        if asdict(self) != asdict(BoundarySemanticsConfig()):
            raise ValueError("Task 7c forensic analysis differs from its registered design")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _from_archive(archive: zipfile.ZipFile) -> tuple[bytes, bytes] | None:
    reports = [name for name in archive.namelist() if name.endswith("/final_report.json")]
    for name in reports:
        prefix = name[: -len("final_report.json")]
        trace_name = prefix + "trajectories.npz"
        if trace_name in archive.namelist() and "giada_cahva_active_closed_loop_microcanary" in prefix:
            return archive.read(name), archive.read(trace_name)
    return None


def verified_task7b_trace_bytes(source: str | Path) -> tuple[dict, bytes]:
    """Verify immutable member hashes, including when Kaggle repacks the ZIP."""
    source = Path(source)
    found = None
    if source.is_dir():
        candidates = [p for p in source.rglob("final_report.json")
                      if "giada_cahva_active_closed_loop_microcanary" in str(p.parent)
                      and (p.parent / "trajectories.npz").is_file()]
        if len(candidates) == 1:
            found = (candidates[0].read_bytes(),
                     (candidates[0].parent / "trajectories.npz").read_bytes())
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            found = _from_archive(archive)
            if found is None:
                nested = [name for name in archive.namelist()
                          if name.endswith("giada_cahva_active_closed_loop_microcanary.zip")]
                if len(nested) == 1:
                    with zipfile.ZipFile(io.BytesIO(archive.read(nested[0]))) as inner:
                        found = _from_archive(inner)
    if found is None:
        raise ValueError("Task 7b report and trajectories not found as one artifact")
    report_bytes, trace_bytes = found
    if (_sha_bytes(report_bytes) != EXPECTED_TASK7B_REPORT_SHA256
            or _sha_bytes(trace_bytes) != EXPECTED_TASK7B_TRAJECTORIES_SHA256):
        raise ValueError("Task 7b artifact member SHA-256 mismatch")
    report = json.loads(report_bytes)
    if (report.get("schema_version") != "giada-task7b-active-microcanary-v1"
            or not report.get("valid") or report.get("episode_count") != 24):
        raise ValueError("Task 7b result contract mismatch")
    return report, trace_bytes


def _gate_formula(formula, voltage, m, h, dt):
    rate = formula.rates(float(voltage))
    mn = rate["m_inf"] + (m - rate["m_inf"]) * np.exp(-dt / rate["m_tau_ms"])
    hn = rate["h_inf"] + (h - rate["h_inf"]) * np.exp(-dt / rate["h_tau_ms"])
    return float(mn), float(hn)


def corrected_formula_rollout(formula, teacher, area_um2, config, multiplier, protocol):
    """NEURON order: voltage from boundary gates, then gates from new voltage."""
    out = np.empty_like(teacher)
    out[0] = teacher[0]
    gbar = config.canonical_gbar_s_cm2 * multiplier
    pulses = _pulses(protocol)
    for step in range(len(teacher) - 1):
        _, v, m, h = out[step, :4]
        iinj = _injected_density(step * config.dt_ms, pulses, area_um2)
        vn = _voltage_step(v, m, h, gbar, iinj, config)
        mn, hn = _gate_formula(formula, vn, m, h, config.dt_ms)
        out[step + 1] = ((step + 1) * config.dt_ms, vn, mn, hn,
                         gbar * m * m * h * (v - config.e_ca_mv), config.e_ca_mv)
    return out


def corrected_candidate_rollout(torch, model, table, teacher, area_um2,
                                config, multiplier, protocol, device):
    count = len(teacher) - 1
    lut = np.empty_like(teacher)
    lut[0] = teacher[0]
    physical = {seed: np.empty_like(teacher) for seed in (17, 29, 43)}
    for row in physical.values():
        row[0] = teacher[0]
    gbar = config.canonical_gbar_s_cm2 * multiplier
    pulses = _pulses(protocol)
    state_l = torch.as_tensor(teacher[0, 1:4], device=device, dtype=torch.float32).reshape(1, 3)
    state_m = state_l.expand(3, -1).clone()
    with torch.inference_mode():
        l_states = torch.empty((count, 3), device=device)
        m_states = torch.empty((3, count, 3), device=device)
        for step in range(count):
            iinj = _injected_density(step * config.dt_ms, pulses, area_um2)
            vl = _voltage_step(state_l[:, 0], state_l[:, 1], state_l[:, 2], gbar, iinj, config)
            vm = _voltage_step(state_m[:, 0], state_m[:, 1], state_m[:, 2], gbar, iinj, config)
            xl = torch.cat((vl[:, None], state_l[:, 1:],
                            torch.full((1, 1), config.dt_ms, device=device)), dim=1)
            xm = torch.cat((vm[:, None], state_m[:, 1:],
                            torch.full((3, 1), config.dt_ms, device=device)), dim=1)
            # One input tuple per frozen seed, preserving the Task 5 ensemble axis.
            gl = gpu_lut_predict(torch, xl, table, linear=True)
            gm = model(xm[:, None, :])[0][:, 0, :]
            state_l = torch.cat((vl[:, None], gl), dim=1)
            state_m = torch.cat((vm[:, None], gm), dim=1)
            l_states[step] = state_l[0]
            m_states[:, step] = state_m
    l_np = l_states.cpu().double().numpy()
    m_np = m_states.cpu().double().numpy()
    lut[1:, 0] = np.arange(1, count + 1) * config.dt_ms
    lut[1:, 1:4] = l_np
    l_old = np.vstack((teacher[0, 1:4], l_np[:-1]))
    lut[1:, 4] = gbar * l_old[:, 1] ** 2 * l_old[:, 2] * (l_old[:, 0] - config.e_ca_mv)
    lut[1:, 5] = config.e_ca_mv
    for index, seed in enumerate((17, 29, 43)):
        row = physical[seed]
        states = m_np[index]
        row[1:, 0] = np.arange(1, count + 1) * config.dt_ms
        row[1:, 1:4] = states
        old = np.vstack((teacher[0, 1:4], states[:-1]))
        row[1:, 4] = gbar * old[:, 1] ** 2 * old[:, 2] * (old[:, 0] - config.e_ca_mv)
        row[1:, 5] = config.e_ca_mv
    return lut, physical


def reassess_boundary_semantics(formula, task5_source, task7b_source, output_dir,
                                config=None, *, code_revision="unknown"):
    config = config or BoundarySemanticsConfig()
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(p.name != ".verified_task5" for p in output_dir.iterdir()):
        raise FileExistsError(f"Output already contains results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    old_report, trace_bytes = verified_task7b_trace_bytes(task7b_source)
    task5_root = verified_task5_root(task5_source, output_dir / ".verified_task5")
    source = np.load(io.BytesIO(trace_bytes))
    teacher_keys = sorted(name[:-8] for name in source.files if name.endswith("_teacher"))
    if len(teacher_keys) != config.expected_episode_count:
        raise ValueError("Task 7b teacher episode count mismatch")
    torch = configure_torch_runtime(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 7c frozen physical-tau comparison requires CUDA")
    device = torch.device("cuda")
    torch, model, table = _frozen_candidates(task5_root, formula, device)
    experiment = ActiveClosedLoopCaHVAConfig()
    experiment.validate()
    episodes = {}
    corrected_traces = {}
    started = time.perf_counter()
    for index, key in enumerate(teacher_keys, 1):
        old_row = old_report["episodes"][key]
        teacher = source[f"{key}_teacher"]
        area = float(old_row["area_um2"])
        multiplier = float(old_row["gbar_multiplier"])
        protocol = str(old_row["protocol"])
        formula_trace = corrected_formula_rollout(
            formula, teacher, area, experiment, multiplier, protocol
        )
        lut_trace, physical_traces = corrected_candidate_rollout(
            torch, model, table, teacher, area, experiment, multiplier, protocol, device
        )
        original_formula = source[f"{key}_formula"]
        lag_error = float(np.max(np.abs(original_formula[1:, 2:4] - teacher[:-1, 2:4])))
        corrected_traces[f"{key}_teacher"] = teacher
        corrected_traces[f"{key}_formula"] = formula_trace
        corrected_traces[f"{key}_lut"] = lut_trace
        for seed, values in physical_traces.items():
            corrected_traces[f"{key}_physical_{seed}"] = values
        episodes[key] = {
            "old_formula_one_step_gate_lag_max_abs": lag_error,
            "formula": _metrics(formula_trace, teacher),
            "formula_current_max_abs_ma_cm2": float(
                np.max(np.abs(formula_trace[:, 4] - teacher[:, 4]))
            ),
            "lut": _metrics(lut_trace, teacher),
            "physical": {str(seed): _metrics(values, teacher)
                         for seed, values in physical_traces.items()},
            "original_lut_voltage_rmse_mv": old_row["lut"]["voltage_rmse_mv"],
        }
        eta = (time.perf_counter() - started) / index * (len(teacher_keys) - index) / 60
        print(f"[GIADA Task 7c] {index}/{len(teacher_keys)} ETA {eta:.1f} min "
              f"formulaV={episodes[key]['formula']['voltage_rmse_mv']:.3g} "
              f"formulaM={episodes[key]['formula']['m_rmse']:.3g}", flush=True)
    formula_voltage_max = max(row["formula"]["voltage_max_abs_mv"] for row in episodes.values())
    formula_gate_max = max(max(row["formula"]["m_rmse"], row["formula"]["h_rmse"])
                           for row in episodes.values())
    formula_current_max = max(row["formula_current_max_abs_ma_cm2"]
                              for row in episodes.values())
    lag_max = max(row["old_formula_one_step_gate_lag_max_abs"] for row in episodes.values())
    finite = all(row[arm]["finite"] for row in episodes.values() for arm in ("formula", "lut"))
    finite = finite and all(metric["finite"] for row in episodes.values()
                            for metric in row["physical"].values())
    report = {
        "schema_version": "giada-task7c-boundary-semantics-v1",
        "code_revision": code_revision,
        "valid": bool(finite and formula_voltage_max <= config.formula_voltage_max_abs_limit_mv
                      and formula_gate_max <= config.formula_gate_max_rmse_limit
                      and formula_current_max <= config.formula_current_max_abs_limit_ma_cm2
                      and lag_max <= config.lag_identity_max_abs_limit),
        "source_task7b_report_sha256": EXPECTED_TASK7B_REPORT_SHA256,
        "source_task7b_trajectories_sha256": EXPECTED_TASK7B_TRAJECTORIES_SHA256,
        "teacher_regenerated": False,
        "model_retrained": False,
        "fresh_test_claimed": False,
        "episode_count": len(episodes),
        "old_formula_one_step_gate_lag_max_abs": lag_max,
        "corrected_formula_voltage_max_abs_mv": formula_voltage_max,
        "corrected_formula_gate_max_rmse": formula_gate_max,
        "corrected_formula_current_max_abs_ma_cm2": formula_current_max,
        "config": asdict(config),
        "episodes": episodes,
        "interpretation": "Task 7b original gate/current boundary metrics were phase-shifted. Corrected rollout uses V from boundary gates, records the old-state current at t+1, then updates gates at new V. This reuses opened Task 7b trajectories and is forensic, not an independent confirmation.",
    }
    np.savez_compressed(output_dir / "corrected_trajectories.npz", **corrected_traces)
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
