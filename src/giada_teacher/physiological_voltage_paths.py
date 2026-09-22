"""Task 9: frozen Ca_HVA gates on recorded full-teacher voltage paths.

All 1 ms paths come from the authentic 642-segment teacher's *train* split.
The future path is diagnostic/teacher-forced, never a deployment input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .gpu_baseline_runtime import configure_torch_runtime
from .primitive_matrix_playground import PrimitiveMatrixConfig, _learned_models
from .primitive_scaling import _gpu_rate_table, gpu_lut_predict
from .voltage_path_stress import _formula_step, verified_task5_root


EXPECTED_TEACHER_COMMIT = "074c4666300a8ad246601dab179a97a6942f0f29"
EXPECTED_TRANSITION_SHA256 = "3fef415544a82b55801461e3cec069ed292faca0075f1f9f431e9dce8f5ea6d8"
EXPECTED_TRANSITIONS = 29240
REGIMES = ("quiet", "rising", "spike", "falling")


@dataclass(frozen=True)
class PhysiologicalPathConfig:
    site_ids: tuple[int, ...] = (0, 387, 460, 469)
    candidate_pool_limit: int = 4000
    maximum_per_site_regime: int = 24
    sample_seed: int = 89031
    source_split: str = "train"
    fine_steps_per_ms: int = 40
    coarse_steps_per_ms: int = 8
    teacher_floor_limit: float = 0.005

    def validate(self):
        if self.site_ids != (0, 387, 460, 469) or self.source_split != "train":
            raise ValueError("Task 9 sites/split differ from preregistration")
        if (self.candidate_pool_limit, self.maximum_per_site_regime) != (4000, 24):
            raise ValueError("Task 9 sampling budget differs from preregistration")
        if (self.fine_steps_per_ms, self.coarse_steps_per_ms) != (40, 8):
            raise ValueError("Task 9 path grid differs from preregistration")


def sha256_file(path: Path, progress=None) -> str:
    digest = hashlib.sha256()
    size = Path(path).stat().st_size
    done = 0
    next_notice = 0.1
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
            done += len(block)
            if progress is not None and done / size >= next_notice:
                progress(round(100 * done / size), "transition_dataset.h5")
                next_notice += 0.1
    return digest.hexdigest()


def classify_regime(voltage: np.ndarray) -> str:
    v = np.asarray(voltage, dtype=np.float64)
    if v.shape != (41,) or not np.isfinite(v).all():
        raise ValueError("Task 9 requires a finite 41-point voltage path")
    if v.max() >= 0 and v.min() < -20:
        return "spike"
    if v[-1] - v[0] >= 5:
        return "rising"
    if v[0] - v[-1] >= 5:
        return "falling"
    if np.ptp(v) < 2:
        return "quiet"
    return "mixed"


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def verify_source(dataset_root: Path, *, progress=None):
    root = Path(dataset_root)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "state_schema.json").read_text(encoding="utf-8"))
    if manifest.get("teacher_commit") != EXPECTED_TEACHER_COMMIT:
        raise RuntimeError("Task 9 teacher commit mismatch")
    if int(manifest.get("transition_count", -1)) != EXPECTED_TRANSITIONS:
        raise RuntimeError("Task 9 source must be the 29,240-transition targeted base")
    h5_path = root / manifest.get("transition_store", "transition_dataset.h5")
    actual = sha256_file(h5_path, progress=progress)
    if actual != EXPECTED_TRANSITION_SHA256:
        raise RuntimeError(f"Task 9 transition store SHA-256 mismatch: {actual}")
    variable_ids = schema["categories"]["mechanism_states"]["variable_ids"]
    voltage_ids = schema["categories"]["voltage"]["variable_ids"]
    mapping = {}
    for segment in PhysiologicalPathConfig().site_ids:
        pair = []
        for gate in ("m", "h"):
            key = f"segment:{segment}:Ca_HVA:{gate}_Ca_HVA"
            if key not in variable_ids:
                raise RuntimeError(f"Ca_HVA STATE missing from source: {key}")
            pair.append(variable_ids.index(key))
        voltage_key = f"segment:{segment}:v"
        mapping[segment] = {"state_indices": pair, "voltage_index": voltage_ids.index(voltage_key)}
    return {"root": str(root), "h5_path": str(h5_path), "h5_sha256": actual,
            "manifest_sha256": hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest(),
            "state_schema_sha256": hashlib.sha256((root / "state_schema.json").read_bytes()).hexdigest(),
            "teacher_commit": EXPECTED_TEACHER_COMMIT, "mapping": mapping,
            "transition_count": EXPECTED_TRANSITIONS}


def select_recorded_paths(source, config: PhysiologicalPathConfig):
    import h5py

    config.validate()
    rng = np.random.default_rng(config.sample_seed)
    mapping = source["mapping"]
    selected = {(site, regime): [] for site in config.site_ids for regime in REGIMES}
    counts = {key: 0 for key in selected}
    checked = 0
    maximum_boundary_error = 0.
    with h5py.File(source["h5_path"], "r") as handle:
        if int(handle.attrs["transition_count"]) != EXPECTED_TRANSITIONS:
            raise RuntimeError("HDF5 transition count differs from manifest")
        time = np.asarray(handle["microtraces/time_offsets_ms"][:], dtype=np.float64)
        if time.shape != (41,) or not np.allclose(time, np.arange(41) / 40, atol=1e-12):
            raise RuntimeError("Task 9 microtrace time grid mismatch")
        splits = [_decode(item) for item in handle["metadata/split"][:]]
        eligible = np.flatnonzero(np.asarray(splits) == config.source_split)
        pool = rng.permutation(eligible)[:config.candidate_pool_limit]
        for index in pool:
            checked += 1
            trace = handle["microtraces/all_segment_voltage"][int(index)]
            state_t = handle["states/mechanism_states/t"][int(index)]
            state_next = handle["states/mechanism_states/t_plus_1"][int(index)]
            boundary_t = handle["states/voltage/t"][int(index)]
            boundary_next = handle["states/voltage/t_plus_1"][int(index)]
            for site in config.site_ids:
                position = mapping[site]["voltage_index"]
                path = np.asarray(trace[:, position], dtype=np.float64)
                regime = classify_regime(path)
                if regime == "mixed":
                    continue
                key = (site, regime)
                counts[key] += 1
                if len(selected[key]) >= config.maximum_per_site_regime:
                    continue
                error = max(abs(path[0] - boundary_t[position]), abs(path[-1] - boundary_next[position]))
                maximum_boundary_error = max(maximum_boundary_error, float(error))
                if error > 1e-4:
                    raise RuntimeError(f"Task 9 voltage boundary mismatch at transition {index}, segment {site}")
                state_indices = mapping[site]["state_indices"]
                initial = np.asarray(state_t[state_indices], dtype=np.float64)
                target = np.asarray(state_next[state_indices], dtype=np.float64)
                if not (np.isfinite(path).all() and np.isfinite(initial).all() and np.isfinite(target).all()):
                    raise RuntimeError("Task 9 source contains NaN/Inf")
                selected[key].append({"transition_index": int(index), "segment_id": site,
                    "regime": regime, "path": path, "initial": initial, "teacher_endpoint": target,
                    "trajectory_id": _decode(handle["metadata/trajectory_id"][int(index)]),
                    "protocol_id": _decode(handle["metadata/protocol_id"][int(index)]),
                    "seed": int(handle["metadata/seed"][int(index)]), "split": splits[int(index)]})
            if all(len(value) >= config.maximum_per_site_regime for value in selected.values()):
                break
    rows = [row for key in selected for row in selected[key]]
    return rows, {"candidate_transitions_checked": checked,
                  "selected_path_count": len(rows),
                  "unique_selected_transition_count": len({row["transition_index"] for row in rows}),
                  "unique_selected_trajectory_count": len({row["trajectory_id"] for row in rows}),
                  "unique_selected_seed_count": len({row["seed"] for row in rows}),
                  "maximum_boundary_voltage_error_mv": maximum_boundary_error,
                  "candidate_counts": {f"{site}:{regime}": counts[(site, regime)]
                                       for site in config.site_ids for regime in REGIMES},
                  "selected_counts": {f"{site}:{regime}": len(selected[(site, regime)])
                                      for site in config.site_ids for regime in REGIMES},
                  "support_shortfalls": [f"{site}:{regime}" for site in config.site_ids for regime in REGIMES
                                         if len(selected[(site, regime)]) < config.maximum_per_site_regime]}


def formula_rollouts(formula, rows):
    variants = {name: np.empty((len(rows), 2), dtype=np.float64)
                for name in ("start_only", "coarse_path", "fine_path")}
    for i, row in enumerate(rows):
        v = row["path"]
        initial = row["initial"]
        variants["start_only"][i] = _formula_step(formula, v[0], initial, 1.)
        coarse = initial.copy()
        for voltage in v[5::5]:
            coarse = _formula_step(formula, voltage, coarse, .125)
        variants["coarse_path"][i] = coarse
        fine = initial.copy()
        for voltage in v[1:]:
            fine = _formula_step(formula, voltage, fine, .025)
        variants["fine_path"][i] = fine
    return variants


def frozen_candidate_rollouts(formula, task5_root, rows):
    torch = configure_torch_runtime(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 9 requires CUDA for frozen candidate comparisons")
    device = torch.device("cuda")
    model = _learned_models(torch, PrimitiveMatrixConfig(physical_width=32))["physical_tau"].to(device)
    states = torch.load(Path(task5_root) / "frozen_scaling_checkpoints.pt", map_location=device, weights_only=True)
    model.load_state_dict(states["32"])
    model.eval()
    table = _gpu_rate_table(torch, formula, 513, device, torch.float32)
    path = torch.as_tensor(np.stack([row["path"] for row in rows]), device=device, dtype=torch.float32)
    initial = torch.as_tensor(np.stack([row["initial"] for row in rows]), device=device, dtype=torch.float32)
    result = {}
    with torch.inference_mode():
        for view, indices, dt in (("start_only", [0], 1.),
                                  ("coarse_path", list(range(5, 41, 5)), .125),
                                  ("fine_path", list(range(1, 41)), .025)):
            lut = initial.clone()
            physical = initial.unsqueeze(0).expand(3, -1, -1).clone()
            for index in indices:
                voltage = path[:, index]
                x_l = torch.cat((voltage[:, None], lut, torch.full_like(voltage[:, None], dt)), dim=1)
                lut = gpu_lut_predict(torch, x_l, table, linear=True)
                x_m = torch.cat((voltage[None, :, None].expand(3, -1, -1), physical,
                                 torch.full((3, len(rows), 1), dt, device=device)), dim=2)
                physical = model(x_m)[0]
            result[f"lut_{view}"] = lut.cpu().double().numpy()
            for seed, values in zip((17, 29, 43), physical.cpu().double().numpy()):
                result[f"physical_{view}_seed{seed}"] = values
    return result


def _metrics(prediction, target):
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if not np.isfinite(p).all():
        raise RuntimeError("Task 9 candidate produced NaN/Inf")
    return {"m_rmse": float(np.sqrt(np.mean((p[:, 0] - t[:, 0]) ** 2))),
            "h_rmse": float(np.sqrt(np.mean((p[:, 1] - t[:, 1]) ** 2))),
            "open_rmse": float(np.sqrt(np.mean((p[:, 0] ** 2 * p[:, 1] - t[:, 0] ** 2 * t[:, 1]) ** 2))),
            "occupancy_violations": int(np.count_nonzero((p < 0) | (p > 1)))}


def run_physiological_path_diagnostic(formula, dataset_root, task5_source, output_dir,
                                       config=None, *, code_revision="unknown", progress=None):
    config = config or PhysiologicalPathConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source = verify_source(dataset_root, progress=progress)
    task5_root = verified_task5_root(task5_source, output_dir.parent / ".task9_verified_task5")
    print("[GIADA Task 9] teacher dataset e Task 5 verificati", flush=True)
    rows, support = select_recorded_paths(source, config)
    if not rows:
        raise RuntimeError("Task 9 found no physiological paths")
    print(f"[GIADA Task 9] selezionati {len(rows)} percorsi da {support['candidate_transitions_checked']} transizioni", flush=True)
    formula_values = formula_rollouts(formula, rows)
    candidate_values = frozen_candidate_rollouts(formula, task5_root, rows)
    teacher = np.stack([row["teacher_endpoint"] for row in rows])
    floor = _metrics(formula_values["fine_path"], teacher)
    groups = {}
    for site in config.site_ids:
        for regime in REGIMES:
            ix = np.asarray([i for i, row in enumerate(rows)
                             if row["segment_id"] == site and row["regime"] == regime], dtype=np.int64)
            if len(ix) == 0:
                continue
            groups[f"{site}:{regime}"] = {"count": len(ix),
                "formula_vs_teacher": _metrics(formula_values["fine_path"][ix], teacher[ix]),
                "vs_formula_fine": {**{f"formula_{view}": _metrics(value[ix], formula_values["fine_path"][ix])
                                        for view, value in formula_values.items() if view != "fine_path"},
                                    **{arm: _metrics(value[ix], formula_values["fine_path"][ix])
                                       for arm, value in candidate_values.items()}},
                "vs_teacher": {arm: _metrics(value[ix], teacher[ix]) for arm, value in candidate_values.items()}}
    calibrated = all(
        max(value["formula_vs_teacher"]["m_rmse"], value["formula_vs_teacher"]["h_rmse"])
        <= config.teacher_floor_limit for value in groups.values()
    )
    provenance = [{key: row[key] for key in ("transition_index", "segment_id", "regime", "trajectory_id",
                                           "protocol_id", "seed", "split")} for row in rows]
    candidate_violations = sum(
        metric["occupancy_violations"] for value in groups.values()
        for name, metric in value["vs_formula_fine"].items() if not name.startswith("formula_")
    )
    report = {"schema_version": "giada-task9-physiological-path-v1",
              "valid": candidate_violations == 0,
              "code_revision": code_revision, "config": asdict(config), "source": source,
              "support": support, "teacher_formula_floor": floor,
              "teacher_formula_floor_calibrated": calibrated,
              "candidate_occupancy_violation_count": candidate_violations,
              "candidate_vs_teacher_interpretable": calibrated,
              "group_metrics": groups, "selection_used_test": False,
              "source_split_train_only": True, "models_retrained": False,
              "future_teacher_voltage_used_for_diagnostic_only": True,
              "endogenous_rollout_claimed": False,
              "interpretation_policy": "If the formula-vs-teacher floor exceeds the preregistered limit, do not attribute candidate-vs-teacher error to the candidate. Candidate-vs-formula remains a numerical approximation diagnostic."}
    (output_dir / "selected_paths.json").write_text(json.dumps(provenance, indent=2))
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    print(f"[GIADA Task 9] concluso; teacher floor calibrato={calibrated}", flush=True)
    return report
