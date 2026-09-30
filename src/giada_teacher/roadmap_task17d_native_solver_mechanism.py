"""Task 17d: native teacher solver-state and event attribution.

Diagnostic interventions only. No surrogate is selected or trained here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.hayflow_teacher.dendritic_calibration import DendriticProtocolCalibrator
from src.hayflow_teacher.diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession

from .native_process import native_phase
from .roadmap_causal_cahva_replacement import (
    SITES, _canonical_gbar, _compare, _sample, _trial, candidate_protocols,
)


@dataclass(frozen=True)
class Task17dConfig:
    conditions: tuple[tuple[int, float], ...] = (
        (170029, 1.0), (170083, 1.5), (170029, 0.5))
    protocol_index: int = 1
    duration_ms: int = 60
    sample_interval_ms: float = 0.025
    dense_interval_ms: float = 0.005
    calcium_scale: float = 1e-4
    voltage_scale: float = 1e-2
    default_atol: float = 1e-3
    default_rtol: float = 0.0
    tight_atol: float = 1e-5
    tight_rtol: float = 1e-6
    ultra_atol: float = 1e-7
    ultra_rtol: float = 1e-8

    def validate(self):
        if asdict(self) != asdict(Task17dConfig()):
            raise ValueError("Task 17d config differs from preregistration")


def _extended_sample(session, suffix):
    result = _sample(session, suffix)
    for site in SITES:
        segment = session.audit.live_segments[site]
        row = result[str(site)]
        for field, attr in (("cai_mM", "cai"), ("sk_z", "z_SK_E2"),
                            ("ik_total", "ik"), ("ica_total", "ica")):
            if hasattr(segment, attr):
                row[field] = float(getattr(segment, attr))
    return result


def _state_names(session):
    vector = session.h.Vector()
    session.cvode.states(vector)
    reference = session.h.ref("")
    basenames = []
    for index in range(int(vector.size())):
        session.cvode.statename(index, reference, 2)
        basenames.append(str(reference[0]))
    return sorted(set(basenames))


def _select_calcium_state(basenames):
    preferred = "cai_CaDynamics_E2"
    if preferred in basenames:
        return preferred
    matches = [name for name in basenames if "cai" in name.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot identify unique calcium STATE basename: {matches}")
    return matches[0]


def _set_policy(session, label, calcium_state, config):
    cvode = session.cvode
    cvode.atol(config.default_atol)
    cvode.rtol(config.default_rtol)
    cvode.atolscale(calcium_state, 1.0)
    cvode.atolscale("v", 1.0)
    if label == "tight":
        cvode.atol(config.tight_atol)
        cvode.rtol(config.tight_rtol)
    elif label in ("ultra", "ultra_dense"):
        cvode.atol(config.ultra_atol)
        cvode.rtol(config.ultra_rtol)
    elif label == "calcium_scaled":
        cvode.atolscale(calcium_state, config.calcium_scale)
    elif label == "voltage_scaled":
        cvode.atolscale("v", config.voltage_scale)
    elif label != "default":
        raise ValueError(label)
    cvode.re_init()
    expected = {
        "atol": config.ultra_atol if label.startswith("ultra") else
                config.tight_atol if label == "tight" else config.default_atol,
        "rtol": config.ultra_rtol if label.startswith("ultra") else
                config.tight_rtol if label == "tight" else config.default_rtol,
        "calcium_scale": config.calcium_scale if label == "calcium_scaled" else 1.0,
        "voltage_scale": config.voltage_scale if label == "voltage_scaled" else 1.0,
    }
    actual = {"atol": float(cvode.atol()), "rtol": float(cvode.rtol()),
              "calcium_scale": float(cvode.atolscale(calcium_state)),
              "voltage_scale": float(cvode.atolscale("v"))}
    if any(not np.isclose(actual[k], v, rtol=1e-10, atol=1e-15)
           for k, v in expected.items()):
        raise RuntimeError(f"Task 17d policy not applied: {label}: {actual} != {expected}")
    return actual


def _release_fingerprint(trial):
    rows = trial["release_rows"]
    if len(rows) != 60:
        raise RuntimeError("Task 17d incomplete release audit")
    if any(not row["verification"].get("valid", False) for row in rows):
        raise RuntimeError("Task 17d causal release boundary verification failed")
    payload = [row["outcomes"] for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _crossings(values, dt):
    values = np.asarray(values, dtype=float)
    indices = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
    return [float((i - values[i] / (values[i + 1] - values[i])) * dt)
            for i in indices]


def _timing_metrics(reference, candidate, dt):
    result = {}
    for site in SITES:
        key = str(site)
        a = np.asarray(reference["traces"][key]["v"])
        b = np.asarray(candidate["traces"][key]["v"])
        if len(a) != len(b):
            raise RuntimeError("Task 17d non-paired trace lengths")
        ta, tb = _crossings(a, dt), _crossings(b, dt)
        result[key] = {"zero_crossings_reference_ms": ta,
                       "zero_crossings_candidate_ms": tb,
                       "first_crossing_delay_ms": tb[0] - ta[0] if ta and tb else None,
                       "peak_reference_mv": float(a.max()),
                       "peak_candidate_mv": float(b.max()),
                       "peak_time_reference_ms": float(a.argmax() * dt),
                       "peak_time_candidate_ms": float(b.argmax() * dt),
                       "late_20_60_rmse_mv": float(np.sqrt(np.mean((a[int(20/dt):] - b[int(20/dt):])**2))),
                       "reference_last_voltage_mv": float(a[-1]),
                       "candidate_last_voltage_mv": float(b[-1])}
    return result


def _downsample_dense(trial, ratio):
    result = dict(trial)
    result["traces"] = {site: {key: values[::ratio] for key, values in fields.items()}
                        for site, fields in trial["traces"].items()}
    return result


def _state_metrics(reference, candidate):
    result = {}
    for site in SITES:
        key = str(site)
        left, right = reference["traces"][key], candidate["traces"][key]
        result[key] = {}
        for field in ("cai_mM", "sk_z", "ik_total", "ica_total"):
            if field in left and field in right:
                error = np.asarray(right[field]) - np.asarray(left[field])
                result[key][field] = {"rmse": float(np.sqrt(np.mean(error**2))),
                                       "maximum_absolute_error": float(np.max(np.abs(error)))}
    return result


def run_task17d(elm_repo, teacher_repo, native_mod, output_dir,
                config=Task17dConfig(), *, code_revision="unknown"):
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, calibration_source=Path(native_mod),
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        output_dir=output_dir / "teacher_workspace")
    teacher = session.prepare_teacher()
    if teacher["segment_count"] != 642:
        raise RuntimeError("Task 17d canonical 642-segment teacher missing")
    session.run_burn_in()
    session.task17_snapshot_suffix = "Ca_HVA"
    calibrator = DendriticProtocolCalibrator(session, output_dir=output_dir / "protocol_workspace")
    gbar = _canonical_gbar(session)
    names = _state_names(session)
    calcium_state = _select_calcium_state(names)
    if "v" not in names:
        raise RuntimeError("CVode voltage STATE basename unavailable")
    starting_tolerances = (float(session.cvode.atol()), float(session.cvode.rtol()))
    if not np.allclose(starting_tolerances, (config.default_atol, config.default_rtol),
                       rtol=1e-10, atol=1e-15):
        raise RuntimeError(f"Canonical teacher CVode tolerance changed: {starting_tolerances}")
    protocol = candidate_protocols()[config.protocol_index]
    policies = ("default", "tight", "ultra", "calcium_scaled", "voltage_scaled")
    cache, repeats, policy_audit, release_hashes, runtimes = {}, {}, {}, {}, {}
    for policy in policies:
        policy_audit[policy] = _set_policy(session, policy, calcium_state, config)
        for seed, multiplier in config.conditions:
            native_phase("task17d_native_trial", policy=policy, seed=seed, multiplier=multiplier)
            started = time.perf_counter()
            row = _trial(session, calibrator, protocol, seed, multiplier,
                         "native", config, gbar, sample_fn=_extended_sample,
                         capture_release=True)
            runtimes[f"{policy}/{seed}/{multiplier}"] = time.perf_counter() - started
            fingerprint = _release_fingerprint(row)
            cache[(policy, seed, multiplier)] = row
            release_hashes[f"{policy}/{seed}/{multiplier}"] = fingerprint
            if (seed, multiplier) == config.conditions[0]:
                repeat = _trial(session, calibrator, protocol, seed, multiplier,
                                "native", config, gbar, sample_fn=_extended_sample,
                                capture_release=True)
                if _release_fingerprint(repeat) != fingerprint:
                    raise RuntimeError(f"Task 17d release repeat failed: {policy}")
                metrics = _compare(row, repeat)
                if any(max(site["voltage_max_error_mv"], site["m_max_error"],
                           site["h_max_error"]) > 1e-9 for site in metrics.values()):
                    raise RuntimeError(f"Task 17d native repeat failed: {policy}")
                repeats[policy] = metrics
    dense_config = replace(config, sample_interval_ms=config.dense_interval_ms)
    policy_audit["ultra_dense"] = _set_policy(session, "ultra_dense", calcium_state, config)
    for seed, multiplier in config.conditions[:2]:
        native_phase("task17d_dense_driver", seed=seed, multiplier=multiplier)
        started = time.perf_counter()
        row = _trial(session, calibrator, protocol, seed, multiplier, "native",
                     dense_config, gbar, sample_fn=_extended_sample,
                     capture_release=True)
        runtimes[f"ultra_dense/{seed}/{multiplier}"] = time.perf_counter() - started
        cache[("ultra_dense", seed, multiplier)] = row
        release_hashes[f"ultra_dense/{seed}/{multiplier}"] = _release_fingerprint(row)
    comparisons = []
    for seed, multiplier in config.conditions:
        reference = cache[("ultra", seed, multiplier)]
        for policy in policies:
            row = cache[(policy, seed, multiplier)]
            comparisons.append({"seed": seed, "gbar_multiplier": multiplier,
                                "policy": policy, "against": "ultra",
                                "metrics": _compare(reference, row),
                                "state_metrics": _state_metrics(reference, row),
                                "timing": _timing_metrics(reference, row, config.sample_interval_ms),
                                "release_match": release_hashes[f"{policy}/{seed}/{multiplier}"] ==
                                                 release_hashes[f"ultra/{seed}/{multiplier}"]})
        if (seed, multiplier) in config.conditions[:2]:
            dense = _downsample_dense(cache[("ultra_dense", seed, multiplier)],
                                      int(round(config.sample_interval_ms/config.dense_interval_ms)))
            comparisons.append({"seed": seed, "gbar_multiplier": multiplier,
                                "policy": "ultra_dense", "against": "ultra",
                                "metrics": _compare(reference, dense),
                                "state_metrics": _state_metrics(reference, dense),
                                "timing": _timing_metrics(reference, dense, config.sample_interval_ms),
                                "release_match": release_hashes[f"ultra_dense/{seed}/{multiplier}"] ==
                                                 release_hashes[f"ultra/{seed}/{multiplier}"]})
    traces = {f"{policy}/{seed}/{multiplier}": cache[(policy, seed, multiplier)]["traces"]
              for policy in policies for seed, multiplier in config.conditions}
    for seed, multiplier in config.conditions[:2]:
        traces[f"ultra_dense/{seed}/{multiplier}"] = cache[("ultra_dense", seed, multiplier)]["traces"]
    (output_dir / "native_traces.json").write_text(json.dumps(traces), encoding="utf-8")
    (output_dir / "paired_metrics.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (output_dir / "release_hashes.json").write_text(json.dumps(release_hashes, indent=2), encoding="utf-8")
    (output_dir / "trial_runtime_seconds.json").write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    report = {"schema_version": "giada-roadmap-task17d-native-solver-mechanism-v1",
              "valid": True, "decision": "DIAGNOSTIC_ONLY", "gate_c_authorized": False,
              "task17a_result_preserved": "CAUSAL_MICROCANARY_NO_GO",
              "code_revision": code_revision, "config": asdict(config),
              "teacher_segment_count": teacher["segment_count"],
              "calcium_state_basename": calcium_state,
              "cvode_state_basenames": names,
              "policy_audit": policy_audit, "repeat_preflight": repeats,
              "paired_comparison_count": len(comparisons),
              "all_release_hashes_match_ultra": all(row["release_match"] for row in comparisons),
              "warning": "Known stress conditions and native-only attribution; no candidate selection."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
