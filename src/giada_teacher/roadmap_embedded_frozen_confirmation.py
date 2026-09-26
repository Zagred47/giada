"""Task 16: frozen Ca_HVA LUT shadow during authentic native teacher replay.

The teacher is never modified. Its replayed V(t), E_Ca(t), and native gates
drive a parallel frozen LUT state. This is embedded teacher-forced confirmation,
not causal mechanism replacement or deployment with a future voltage path.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.hayflow_teacher.diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession

from .physiological_voltage_paths import REGIMES, _decode, _metrics, classify_regime, verify_source
from .primitive_scaling import _gpu_rate_table, gpu_lut_predict
from .roadmap_embedded_interface_bridge import verified_task5_any
from .voltage_path_stress import _formula_step


EXPECTED_15C_REPORT_SHA256 = "ffde98ea986cc6dc293688b0e9335efb6af60639fe95281440df17523c4b2fc9"
EXPECTED_15C_CODE_REVISION = "fb8a55cdb038628af6dd35cac9b576dfb4b3ff99"


@dataclass(frozen=True)
class FrozenEmbeddedConfig:
    split: str = "deterministic_test"
    site_ids: tuple[int, ...] = (0, 387, 460, 469)
    regimes: tuple[str, ...] = REGIMES
    rows_per_group: int = 8
    minimum_per_group: int = 4
    sample_seed: int = 16017
    replay_atol: float = 1e-5
    voltage_trace_atol_mv: float = 1e-5
    formula_floor_limit: float = 0.005
    candidate_gate_limit: float = 0.005
    current_rmse_limit_ma_cm2: float = 0.0001

    def validate(self):
        if asdict(self) != asdict(FrozenEmbeddedConfig()):
            raise ValueError("Task 16 differs from preregistration")


def verified_15c(source):
    source = Path(source)
    if source.is_dir():
        matches = list(source.rglob("final_report.json"))
        if len(matches) != 1:
            raise ValueError("Task 15c extracted report is not unique")
        raw = matches[0].read_bytes()
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as outer:
            reports = [name for name in outer.namelist()
                       if name == "final_report.json" or name.endswith("/final_report.json")]
            if len(reports) == 1:
                raw = outer.read(reports[0])
            else:
                nested = [name for name in outer.namelist()
                          if name.endswith("giada_roadmap_task15c_embedded_interface_bridge.zip")]
                if len(nested) != 1:
                    raise ValueError("Task 15c nested ZIP is not unique")
                with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                    reports = [name for name in inner.namelist()
                               if name == "final_report.json" or name.endswith("/final_report.json")]
                    if len(reports) != 1:
                        raise ValueError("Task 15c nested report is not unique")
                    raw = inner.read(reports[0])
    else:
        raise ValueError("Task 15c source must be ZIP or extracted directory")
    report = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != EXPECTED_15C_REPORT_SHA256
            or report.get("code_revision") != EXPECTED_15C_CODE_REVISION
            or report.get("teacher_forced_candidate_for_task16") != "lut_fine_path"
            or report.get("task16_frozen_evaluation_authorized") is not True
            or report.get("teacher_test_splits_opened") is not False):
        raise ValueError("Task 15c frozen selection contract mismatch")
    return {"report_sha256": EXPECTED_15C_REPORT_SHA256,
            "selected_candidate": "lut_fine_path"}


def select_test_rows(source, config=FrozenEmbeddedConfig()):
    import h5py

    config.validate()
    groups = {(site, regime): [] for site in config.site_ids for regime in config.regimes}
    with h5py.File(source["h5_path"], "r") as handle:
        if int(handle.attrs["transition_count"]) != source["transition_count"]:
            raise RuntimeError("Task 16 teacher HDF5 count mismatch")
        grid = np.asarray(handle["microtraces/time_offsets_ms"][:])
        if grid.shape != (41,) or not np.allclose(grid, np.arange(41) / 40, atol=1e-12):
            raise RuntimeError("Task 16 voltage grid mismatch")
        splits = np.asarray([_decode(x) for x in handle["metadata/split"][:]])
        eligible = np.flatnonzero(splits == config.split)
        if len(eligible) == 0:
            raise RuntimeError("Task 16 deterministic test split missing")
        checked = 0
        started = time.perf_counter()
        for index in np.random.default_rng(config.sample_seed).permutation(eligible):
            checked += 1
            voltage = handle["microtraces/all_segment_voltage"][int(index)]
            for site in config.site_ids:
                path = np.asarray(voltage[:, source["mapping"][site]["voltage_index"]], dtype=float)
                regime = classify_regime(path)
                group = (site, regime)
                if regime not in config.regimes or len(groups[group]) >= config.rows_per_group:
                    continue
                ref = _decode(handle["metadata/native_snapshot_ref"][int(index)])
                if not ref or Path(ref).is_absolute() or ".." in Path(ref).parts:
                    raise RuntimeError(f"Task 16 unsafe/missing native snapshot ref {index}")
                groups[group].append({"transition_index": int(index), "segment_id": site,
                                      "regime": regime, "split": config.split,
                                      "trajectory_id": _decode(handle["metadata/trajectory_id"][int(index)]),
                                      "seed": int(handle["metadata/seed"][int(index)]),
                                      "native_snapshot_ref": ref})
            if checked % 500 == 0:
                filled = sum(map(len, groups.values()))
                print(f"[GIADA Task 16] selezione test {checked}/{len(eligible)} "
                      f"path={filled} elapsed={(time.perf_counter()-started)/60:.1f} min",
                      flush=True)
            if all(len(items) == config.rows_per_group for items in groups.values()):
                break
    rows = [row for group in groups.values() for row in group]
    support = {"eligible_transition_count": len(eligible), "checked_transition_count": checked,
               "selected_site_transition_count": len(rows),
               "unique_transition_count": len({r["transition_index"] for r in rows}),
               "unique_trajectory_count": len({r["trajectory_id"] for r in rows}),
               "groups": {f"{site}:{regime}": len(groups[(site, regime)])
                          for site in config.site_ids for regime in config.regimes},
               "minimum_support_met": all(len(items) >= config.minimum_per_group
                                          for items in groups.values())}
    return rows, support


def stage_snapshots(dataset_source, rows, source, destination):
    """Stage only selected immutable native snapshots, never the whole bank."""
    destination = Path(destination)
    refs = sorted({row["native_snapshot_ref"] for row in rows})
    for ref in refs:
        target = (destination / ref).resolve()
        if destination.resolve() not in target.parents:
            raise RuntimeError("Task 16 snapshot path escapes replay workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        local = Path(source["root"]) / ref
        if local.is_file():
            shutil.copy2(local, target)
            continue
        archive_path = Path(dataset_source)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing native snapshot: {ref}")
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name == ref or name.endswith("/" + ref)]
            if len(members) != 1:
                raise RuntimeError(f"Task 16 snapshot member ambiguous: {ref}")
            with archive.open(members[0]) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
    return {"staged_snapshot_count": len(refs), "refs": refs}


def _lut_table(formula):
    grid = np.linspace(-135., 75., 513)
    return np.asarray([[formula.rates(float(v))[key]
                        for key in ("m_inf", "h_inf", "m_tau_ms", "h_tau_ms")]
                       for v in grid], dtype=np.float32)


def _lut_step(table, voltage, state):
    v = np.float32(voltage)
    coordinate = np.clip(np.float32((v + np.float32(135.)) * np.float32(512. / 210.)),
                         np.float32(0.), np.float32(512.))
    low = min(int(np.floor(coordinate)), 511)
    fraction = np.float32(coordinate - low)
    curve = table[low] * np.float32(1. - fraction) + table[low + 1] * fraction
    z = -np.expm1(np.float32(-.025) / curve[2:])
    old = np.asarray(state, dtype=np.float32)
    return np.asarray((np.float32(1.) - z) * old + z * curve[:2], dtype=np.float32)


def _gpu_equivalence_preflight(formula, table):
    from .gpu_baseline_runtime import configure_torch_runtime

    torch = configure_torch_runtime(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 16 LUT equivalence preflight requires CUDA")
    device = torch.device("cuda")
    torch_table = _gpu_rate_table(torch, formula, 513, device, torch.float32)
    if not np.array_equal(table, torch_table.cpu().numpy()):
        raise RuntimeError("Task 16 CPU and Task5 GPU LUT tables differ")
    voltage = np.asarray([-135., -100., -55.123, -27., -10., 0., 60., 75.], dtype=np.float32)
    states = np.asarray([[.1 + .05 * i, .9 - .04 * i] for i in range(len(voltage))], dtype=np.float32)
    values = np.column_stack((voltage, states, np.full(len(voltage), .025, dtype=np.float32)))
    with torch.inference_mode():
        gpu = gpu_lut_predict(torch, torch.as_tensor(values, device=device), torch_table,
                              linear=True).cpu().numpy()
    cpu = np.stack([_lut_step(table, row[0], row[1:3]) for row in values])
    error = float(np.max(np.abs(cpu - gpu)))
    if error > 5e-7:
        raise RuntimeError(f"Task 16 online LUT differs from frozen GPU reference: {error}")
    return {"valid": True, "maximum_absolute_error": error, "atol": 5e-7}


class _ShadowReplaySession(TargetedDiagnosticDatasetSession):
    def __init__(self, *args, site_ids, lut_table, **kwargs):
        super().__init__(*args, **kwargs)
        self.shadow_site_ids = tuple(site_ids)
        self.shadow_lut_table = np.asarray(lut_table, dtype=np.float32)
        self.shadow_target_index = None
        self.shadow_capture = False
        self.shadow_samples = []

    def _run_transition(self, transition_id, trajectory, step_index, actions, snapshot_path):
        self.shadow_capture = int(transition_id) == self.shadow_target_index
        if self.shadow_capture:
            self.shadow_samples = []
        try:
            return super()._run_transition(transition_id, trajectory, step_index, actions, snapshot_path)
        finally:
            self.shadow_capture = False

    def _sample_transition_point(self):
        sample = super()._sample_transition_point()
        if self.shadow_capture:
            sites = {}
            for site in self.shadow_site_ids:
                segment = self.audit.live_segments[site]
                voltage = float(segment.v)
                if self.shadow_samples:
                    previous = self.shadow_samples[-1][site]
                    candidate = _lut_step(self.shadow_lut_table,
                                          (previous["v"] + voltage) / 2,
                                          previous["candidate"])
                else:
                    candidate = np.asarray([float(segment.m_Ca_HVA),
                                            float(segment.h_Ca_HVA)], dtype=np.float32)
                sites[site] = {"v": float(segment.v), "m": float(segment.m_Ca_HVA),
                               "h": float(segment.h_Ca_HVA), "eca": float(segment.eca),
                               "gbar": float(segment.gCa_HVAbar_Ca_HVA),
                               "candidate": candidate}
            self.shadow_samples.append(sites)
        return sample


def _group_metrics(prediction, target, rows, config):
    result = {}
    for site in config.site_ids:
        for regime in config.regimes:
            indices = [i for i, row in enumerate(rows)
                       if row["segment_id"] == site and row["regime"] == regime]
            if indices:
                result[f"{site}:{regime}"] = {"count": len(indices),
                                              **_metrics(prediction[indices], target[indices])}
    return result


def run_frozen_embedded_confirmation(formula, elm_repo, teacher_repo, dataset_root,
                                     dataset_source, task5_source, task15c_source,
                                     output_dir, config=FrozenEmbeddedConfig(),
                                     *, code_revision="unknown", progress=None):
    import h5py

    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    selection = verified_15c(task15c_source)
    source = verify_source(Path(dataset_root), progress=progress)
    task5_root = verified_task5_any(task5_source, output_dir.parent / ".task16_task5_cache")
    table = _lut_table(formula)
    equivalence = _gpu_equivalence_preflight(formula, table)
    rows, support = select_test_rows(source, config)
    output_dir.mkdir(parents=True)
    (output_dir / "selected_test_paths.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not support["minimum_support_met"]:
        report = {"schema_version": "giada-roadmap-task16-embedded-v1", "valid": False,
                  "diagnosis": "independent_test_support_shortfall", "support": support,
                  "candidate": selection, "test_selection_not_used_for_model_choice": True}
        (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    replay_root = output_dir / "native_replay_workspace"
    replay_root.mkdir()
    staged = stage_snapshots(dataset_source, rows, source, replay_root)
    session = _ShadowReplaySession(elm_repo, teacher_repo, output_dir=replay_root,
                                   site_ids=config.site_ids, lut_table=table)
    session.prepare_teacher()
    stored_schema = json.loads((Path(source["root"]) / "state_schema.json").read_text(encoding="utf-8"))
    for category in session.state_variables:
        if (session.state_schema["categories"][category]["variable_ids"]
                != stored_schema["categories"][category]["variable_ids"]):
            raise RuntimeError(f"Task 16 live teacher state order differs from stored {category}")
    recorded = []
    started = time.perf_counter()
    # Replaying every selected target from its own immutable native checkpoint
    # keeps the teacher's adaptive CVode, synapses and RNG path authentic.
    with h5py.File(source["h5_path"], "r") as handle:
        by_index = {}
        for row in rows:
            by_index.setdefault(row["transition_index"], []).append(row)
        trajectory_index_map = {}
        for row_index, trajectory_id in enumerate(handle["metadata/trajectory_id"][:]):
            trajectory_index_map.setdefault(_decode(trajectory_id), []).append(row_index)
        for number, (index, same_transition_rows) in enumerate(by_index.items(), 1):
            session.shadow_target_index = index
            session.shadow_samples = []
            replay = session._replay_hdf5_transition(
                handle, index, trajectory_index_map=trajectory_index_map)
            session.shadow_target_index = None
            samples = session.shadow_samples
            if len(samples) != 41:
                raise RuntimeError(f"Task 16 missing live shadow samples: {index}")
            for row in same_transition_rows:
                site = row["segment_id"]
                live = {key: np.asarray([sample[site][key] for sample in samples], dtype=float)
                        for key in ("v", "m", "h", "eca", "gbar")}
                live["candidate"] = np.stack([sample[site]["candidate"] for sample in samples]).astype(float)
                stored_v = np.asarray(handle["microtraces/all_segment_voltage"][index, :,
                                     source["mapping"][site]["voltage_index"]], dtype=float)
                stored_state_row = handle["states/mechanism_states/t_plus_1"][index]
                stored_end = np.asarray(
                    stored_state_row[source["mapping"][site]["state_indices"]], dtype=float)
                voltage_error = float(np.max(np.abs(live["v"] - stored_v)))
                endpoint_error = float(np.max(np.abs(np.array([live["m"][-1], live["h"][-1]]) - stored_end)))
                recorded.append({"metadata": row, "live": live, "replay": replay,
                                 "voltage_reproduction_error_mv": voltage_error,
                                 "native_gate_endpoint_error": endpoint_error})
            if number == 1 or number % 8 == 0 or number == len(by_index):
                eta = (time.perf_counter() - started) * (len(by_index) / number - 1) / 60
                print(f"[GIADA Task 16] live replay {number}/{len(by_index)} ETA {eta:.1f} min", flush=True)
    row_order = {(row["transition_index"], row["segment_id"]): i
                 for i, row in enumerate(rows)}
    recorded.sort(key=lambda item: row_order[(item["metadata"]["transition_index"],
                                              item["metadata"]["segment_id"])])
    replay_valid = all(item["replay"]["reproduced"]
                       and item["voltage_reproduction_error_mv"] <= config.voltage_trace_atol_mv
                       and item["native_gate_endpoint_error"] <= config.replay_atol
                       for item in recorded)
    if not replay_valid:
        report = {"schema_version": "giada-roadmap-task16-embedded-v1", "valid": False,
                  "diagnosis": "native_teacher_replay_mismatch", "support": support,
                  "maximum_voltage_reproduction_error_mv": max(x["voltage_reproduction_error_mv"] for x in recorded),
                  "maximum_gate_endpoint_error": max(x["native_gate_endpoint_error"] for x in recorded),
                  "failed_replays": [x["metadata"] for x in recorded if not x["replay"]["reproduced"]],
                  "test_selection_not_used_for_model_choice": True}
        (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    paths = np.stack([x["live"]["v"] for x in recorded])
    native = np.stack([np.stack((x["live"]["m"], x["live"]["h"]), axis=-1) for x in recorded])
    eca = np.stack([x["live"]["eca"] for x in recorded])
    gbar = np.stack([x["live"]["gbar"] for x in recorded])
    candidate = np.stack([x["live"]["candidate"] for x in recorded])
    # The formula control is integrated sequentially, independently of native
    # endpoints; it isolates the path-discretization floor from LUT error.
    controls = []
    for path, initial in zip(paths, native[:, 0, :]):
        states = [initial.copy()]
        for left, right in zip(path[:-1], path[1:]):
            states.append(_formula_step(formula, (left + right) / 2, states[-1], .025))
        controls.append(states)
    controls = np.asarray(controls)
    native_current = gbar * native[..., 0] ** 2 * native[..., 1] * (paths - eca)
    candidate_current = gbar * candidate[..., 0] ** 2 * candidate[..., 1] * (paths - eca)
    control_current = gbar * controls[..., 0] ** 2 * controls[..., 1] * (paths - eca)
    endpoint_native, endpoint_candidate, endpoint_control = native[:, -1], candidate[:, -1], controls[:, -1]
    floor = _group_metrics(endpoint_control, endpoint_native, rows, config)
    groups = _group_metrics(endpoint_candidate, endpoint_native, rows, config)
    floor_valid = all(max(x["m_rmse"], x["h_rmse"]) <= config.formula_floor_limit for x in floor.values())
    gate_valid = all(max(x["m_rmse"], x["h_rmse"]) <= config.candidate_gate_limit
                     and x["occupancy_violations"] == 0 for x in groups.values())
    current_group_rmse = {}
    for site in config.site_ids:
        for regime in config.regimes:
            subset = [i for i, row in enumerate(rows)
                      if row["segment_id"] == site and row["regime"] == regime]
            current_group_rmse[f"{site}:{regime}"] = float(np.sqrt(np.mean(
                (candidate_current[subset] - native_current[subset]) ** 2)))
    current_valid = all(value <= config.current_rmse_limit_ma_cm2
                        for value in current_group_rmse.values())
    report = {"schema_version": "giada-roadmap-task16-embedded-v1", "valid": True,
              "code_revision": code_revision, "config": asdict(config), "task15c_selection": selection,
              "task5_report_sha256": hashlib.sha256((task5_root / "final_report.json").read_bytes()).hexdigest(),
              "teacher_h5_sha256": source["h5_sha256"], "support": support,
              "staged_snapshot_count": staged["staged_snapshot_count"],
              "maximum_voltage_reproduction_error_mv": max(x["voltage_reproduction_error_mv"] for x in recorded),
              "maximum_gate_endpoint_error": max(x["native_gate_endpoint_error"] for x in recorded),
              "maximum_native_state_replay_error": max(
                  value for item in recorded for value in item["replay"]["max_state_error_by_category"].values()),
              "maximum_native_rng_replay_error": max(
                  item["replay"]["max_rng_sequence_error"] for item in recorded),
              "native_replay_valid": replay_valid, "formula_floor_by_group": floor,
              "online_lut_gpu_equivalence": equivalence,
              "candidate_endpoint_by_group": groups,
              "candidate_current_path_rmse_by_group_ma_cm2": current_group_rmse,
              "formula_current_path_rmse_ma_cm2": float(np.sqrt(np.mean((control_current-native_current)**2))),
              "candidate_current_path_rmse_ma_cm2": float(np.sqrt(np.mean((candidate_current-native_current)**2))),
              "formula_gate_endpoint_global": _metrics(endpoint_control, endpoint_native),
              "candidate_gate_endpoint_global": _metrics(endpoint_candidate, endpoint_native),
              "floor_valid": floor_valid, "gate_valid": gate_valid,
              "current_valid": current_valid,
              "embedded_teacher_forced_confirmation_passed": bool(floor_valid and gate_valid and current_valid),
              "teacher_voltage_or_membrane_modified": False,
              "teacher_test_used_for_model_selection": False,
              "mechanism_replaced": False, "gate_c_authorized": False,
              "current_reference_kind": "analytic_CaHVA_current_from_native_gate_states_not_total_ica",
              "scope": "Frozen LUT shadow during exact 642-segment NEURON replay; V(t) supplied by native teacher at each observed substep. Midpoint uses the next recorded substep V and is not causal at the start of the millisecond. Task17 replacement remains untested."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
