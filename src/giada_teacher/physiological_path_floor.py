"""Task 9b: forensic quadrature controls for the opened Task 9 teacher floor.

No model is trained or selected, no new teacher path is generated, and these
already-open train paths are not an independent validation set.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .physiological_voltage_paths import _decode, verify_source


EXPECTED_TASK9_ZIP_SHA256 = "6c28d4a1b0019a309a3ec65a4f62eae232f014b2bbb1b7680b82b1b16823ac44"
EXPECTED_TASK9_REPORT_SHA256 = "1743e55859ddda524872b512ae0645534d0f7af9f83b3a1aa40c96483e7d8363"
EXPECTED_TASK9_SELECTED_SHA256 = "ba1a731194fdd52f74f97099bc024c92c4dbd996d3d64bacde0b11dbb533efd2"
METHODS = ("right", "right_refined_5", "left", "midpoint", "linear_5", "linear_20")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_verified_task9(source: Path):
    source = Path(source)
    if source.is_file():
        if _sha(source.read_bytes()) != EXPECTED_TASK9_ZIP_SHA256:
            raise RuntimeError("Task 9 ZIP SHA-256 mismatch")
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            report_names = [n for n in names if n.endswith("/final_report.json")]
            selected_names = [n for n in names if n.endswith("/selected_paths.json")]
            if len(report_names) != 1 or len(selected_names) != 1:
                raise RuntimeError("Task 9 archive members ambiguous")
            report_bytes = archive.read(report_names[0])
            selected_bytes = archive.read(selected_names[0])
    else:
        report_bytes = (source / "final_report.json").read_bytes()
        selected_bytes = (source / "selected_paths.json").read_bytes()
    if _sha(report_bytes) != EXPECTED_TASK9_REPORT_SHA256 or _sha(selected_bytes) != EXPECTED_TASK9_SELECTED_SHA256:
        raise RuntimeError("Task 9 report or selected-path index SHA-256 mismatch")
    report = json.loads(report_bytes)
    selected = json.loads(selected_bytes)
    if (not report.get("valid") or not report.get("source_split_train_only")
            or report.get("teacher_formula_floor_calibrated")
            or len(selected) != report["support"]["selected_path_count"]):
        raise RuntimeError("Task 9 source state differs from the registered floor diagnosis")
    return report, selected


def _step(formula, voltage: float, state: np.ndarray, dt: float) -> np.ndarray:
    rates = formula.rates(float(voltage))
    inf = np.array([rates["m_inf"], rates["h_inf"]], dtype=np.float64)
    tau = np.array([rates["m_tau_ms"], rates["h_tau_ms"]], dtype=np.float64)
    return inf + (state - inf) * np.exp(-dt / tau)


def integrate_recorded_path(formula, path, initial, method: str):
    """Compose Ca_HVA exact constant-V updates under predeclared path views."""
    voltage = np.asarray(path, dtype=np.float64)
    state = np.asarray(initial, dtype=np.float64).copy()
    if voltage.shape != (41,) or state.shape != (2,) or method not in METHODS:
        raise ValueError("invalid Task 9b path/state/method")
    for start, end in zip(voltage[:-1], voltage[1:]):
        if method == "right":
            state = _step(formula, end, state, .025)
        elif method == "right_refined_5":
            for _ in range(5):
                state = _step(formula, end, state, .005)
        elif method == "left":
            state = _step(formula, start, state, .025)
        elif method == "midpoint":
            state = _step(formula, (start + end) / 2, state, .025)
        else:
            substeps = 5 if method == "linear_5" else 20
            dt = .025 / substeps
            for k in range(substeps):
                fraction = (k + .5) / substeps
                state = _step(formula, start + fraction * (end - start), state, dt)
    return state


def read_original_paths(dataset_root: Path, task9_report, selected, *, progress=None):
    import h5py

    source = verify_source(dataset_root, progress=progress)
    if source["h5_sha256"] != task9_report["source"]["h5_sha256"]:
        raise RuntimeError("Task 9 and 9b dataset fingerprints differ")
    rows = []
    mapping = source["mapping"]
    with h5py.File(source["h5_path"], "r") as handle:
        for public in selected:
            index = int(public["transition_index"])
            segment = int(public["segment_id"])
            if segment not in mapping or _decode(handle["metadata/split"][index]) != "train":
                raise RuntimeError("Task 9b selected row is outside the registered train/site contract")
            for name in ("trajectory_id", "protocol_id"):
                if _decode(handle[f"metadata/{name}"][index]) != public[name]:
                    raise RuntimeError(f"Task 9b {name} provenance mismatch")
            if int(handle["metadata/seed"][index]) != int(public["seed"]):
                raise RuntimeError("Task 9b seed provenance mismatch")
            position = mapping[segment]["voltage_index"]
            pair = mapping[segment]["state_indices"]
            path = np.asarray(handle["microtraces/all_segment_voltage"][index, :, position], dtype=np.float64)
            initial = np.asarray(handle["states/mechanism_states/t"][index, pair], dtype=np.float64)
            teacher = np.asarray(handle["states/mechanism_states/t_plus_1"][index, pair], dtype=np.float64)
            v0 = float(handle["states/voltage/t"][index, position])
            v1 = float(handle["states/voltage/t_plus_1"][index, position])
            if max(abs(path[0] - v0), abs(path[-1] - v1)) > 1e-4:
                raise RuntimeError("Task 9b boundary voltage mismatch")
            if not (np.isfinite(path).all() and np.isfinite(initial).all() and np.isfinite(teacher).all()):
                raise RuntimeError("Task 9b source contains NaN/Inf")
            rows.append({**public, "path": path, "initial": initial, "teacher_endpoint": teacher})
    return source, rows


def _metrics(prediction, target):
    delta = np.asarray(prediction) - np.asarray(target)
    p = np.asarray(prediction)
    t = np.asarray(target)
    return {"m_rmse": float(np.sqrt(np.mean(delta[:, 0] ** 2))),
            "h_rmse": float(np.sqrt(np.mean(delta[:, 1] ** 2))),
            "open_rmse": float(np.sqrt(np.mean((p[:, 0] ** 2 * p[:, 1] - t[:, 0] ** 2 * t[:, 1]) ** 2))),
            "m_mean_signed_error": float(np.mean(delta[:, 0])),
            "m_median_absolute_error": float(np.median(np.abs(delta[:, 0])))}


def run_physiological_path_floor(formula, dataset_root, task9_source, output_dir,
                                 *, code_revision="unknown", progress=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prior, selected = load_verified_task9(task9_source)
    source, rows = read_original_paths(dataset_root, prior, selected, progress=progress)
    print(f"[GIADA Task 9b] verificati {len(rows)} percorsi Task 9", flush=True)
    teacher = np.stack([row["teacher_endpoint"] for row in rows])
    predictions = {}
    for method_number, method in enumerate(METHODS, start=1):
        print(f"[GIADA Task 9b] metodo {method_number}/{len(METHODS)}: {method}", flush=True)
        values = []
        for index, row in enumerate(rows, start=1):
            values.append(integrate_recorded_path(formula, row["path"], row["initial"], method))
            if index % 100 == 0 or index == len(rows):
                print(f"[GIADA Task 9b] {method}: {index}/{len(rows)}", flush=True)
        predictions[method] = np.stack(values)
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Task 9b integration produced NaN/Inf")
    original = prior["teacher_formula_floor"]
    reproduced = _metrics(predictions["right"], teacher)
    reproduction_error = max(abs(reproduced[key] - original[key]) for key in ("m_rmse", "h_rmse", "open_rmse"))
    if reproduction_error > 1e-10:
        raise RuntimeError(f"Task 9b failed to reproduce Task 9 right-endpoint floor: {reproduction_error}")
    refinement_error = float(np.max(np.abs(predictions["right"] - predictions["right_refined_5"])))
    if refinement_error > 1e-10:
        raise RuntimeError("Task 9b constant-voltage refinement control failed")
    groups = {}
    for key in prior["group_metrics"]:
        site, regime = key.split(":")
        indices = np.asarray([i for i, row in enumerate(rows)
                              if row["segment_id"] == int(site) and row["regime"] == regime], dtype=np.int64)
        if len(indices) != int(prior["group_metrics"][key]["count"]):
            raise RuntimeError(f"Task 9b support differs for {key}")
        methods = {method: _metrics(values[indices], teacher[indices])
                   for method, values in predictions.items()}
        paired_improved = float(np.mean(np.abs(predictions["linear_20"][indices, 0] - teacher[indices, 0])
                                       < np.abs(predictions["right"][indices, 0] - teacher[indices, 0])))
        groups[key] = {"count": len(indices), "methods": methods,
                       "linear20_fraction_paths_m_improved_vs_right": paired_improved,
                       "linear5_vs_linear20_max_abs_gate": float(np.max(np.abs(
                           predictions["linear_5"][indices] - predictions["linear_20"][indices])))}
    spike = {key: value for key, value in groups.items() if key.endswith(":spike")}
    improved_spike_floor = bool(spike) and all(
        value["methods"]["linear_20"]["m_rmse"] <= .5 * value["methods"]["right"]["m_rmse"]
        and max(value["methods"]["linear_20"][gate] for gate in ("m_rmse", "h_rmse")) <= .005
        and value["linear5_vs_linear20_max_abs_gate"] <= .001
        for value in spike.values()
    )
    report = {"schema_version": "giada-task9b-path-floor-forensic-v1", "valid": True,
              "code_revision": code_revision, "task9_zip_sha256": EXPECTED_TASK9_ZIP_SHA256,
              "source": source, "path_count": len(rows), "source_split_train_only": True,
              "task9_right_floor_reproduction_max_error": reproduction_error,
              "right_substep_invariance_max_abs": refinement_error,
              "global_methods": {method: _metrics(value, teacher) for method, value in predictions.items()},
              "groups": groups, "all_spike_groups_meet_preregistered_interpolation_gate": improved_spike_floor,
              "new_teacher_paths_generated": False, "model_training_performed": False,
              "interpretation": "Interpolation improvement is consistent with temporal sampling loss, not proof of the unobserved true intra-sample voltage. Failure to improve leaves the floor unresolved; denser authentic teacher capture would be the next causal test."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    print(f"[GIADA Task 9b] concluso; spike interpolation gate={improved_spike_floor}", flush=True)
    return report
