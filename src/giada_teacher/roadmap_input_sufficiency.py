"""Roadmap Task 10: controlled voltage-path information and solver-step matrix.

All future-voltage views are teacher-forced diagnostic oracles. The only
causally available view in this experiment is the start voltage; no learned
voltage predictor or autonomous rollout is claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .physiological_path_floor import (
    _metrics,
    _step,
    integrate_recorded_path,
    load_verified_task9,
    read_original_paths,
)


VIEWS = ("start_only", "mean_oracle", "endpoints_oracle", "endpoints_mean_oracle", "five_oracle",
         "nine_oracle", "twentyone_oracle", "full41_oracle")
SUBSTEPS = (1, 2, 4, 8, 40)
SITE_REGIMES = ("quiet", "rising", "spike", "falling")


def reconstruct_voltage_path(path, view):
    original = np.asarray(path, dtype=np.float64)
    if original.shape != (41,) or not np.isfinite(original).all() or view not in VIEWS:
        raise ValueError("Task 10 requires a finite 41-point path and registered view")
    if view == "start_only":
        return np.full(41, original[0])
    if view == "mean_oracle":
        return np.full(41, np.mean(original))
    if view == "full41_oracle":
        return original.copy()
    if view == "endpoints_mean_oracle":
        linear = np.linspace(original[0], original[-1], 41)
        t = np.linspace(0, 1, 41)
        arch = 6 * t * (1 - t)
        return linear + (float(np.mean(original)) - float(np.mean(linear))) / float(np.mean(arch)) * arch
    count = {"endpoints_oracle": 2, "five_oracle": 5, "nine_oracle": 9,
             "twentyone_oracle": 21}[view]
    indices = np.linspace(0, 40, count, dtype=np.int64)
    return np.interp(np.arange(41), indices, original[indices])


def integrate_view(formula, path, initial, view, substeps):
    if substeps not in SUBSTEPS:
        raise ValueError("unregistered number of internal substeps")
    approx = reconstruct_voltage_path(path, view)
    state = np.asarray(initial, dtype=np.float64).copy()
    if state.shape != (2,):
        raise ValueError("Task 10 requires m/h initial state")
    for k in range(substeps):
        position = (k + .5) * 40 / substeps
        left = min(39, int(np.floor(position)))
        voltage = approx[left] + (position - left) * (approx[left + 1] - approx[left])
        state = _step(formula, voltage, state, 1.0 / substeps)
    return state


def _difference_metrics(prediction, reference):
    return _metrics(prediction, reference)


def run_roadmap_input_sufficiency(formula, dataset_root, task9_source, output_dir,
                                   *, code_revision="unknown", progress=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prior, selected = load_verified_task9(task9_source)
    source, rows = read_original_paths(dataset_root, prior, selected, progress=progress)
    if len(rows) != 350 or any(row["split"] != "train" for row in rows):
        raise RuntimeError("Roadmap Task 10 Task 9 train support changed")
    authentic = np.stack([row["teacher_endpoint"] for row in rows])
    predictions = {}
    for view in VIEWS:
        for n in SUBSTEPS:
            values = np.stack([integrate_view(formula, row["path"], row["initial"], view, n)
                               for row in rows])
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite Task 10 output: {view}/{n}")
            predictions[(view, n)] = values
        print(f"[GIADA roadmap Task 10] vista {VIEWS.index(view)+1}/{len(VIEWS)}: {view}",
              flush=True)
    full = predictions[("full41_oracle", 40)]
    midpoint = np.stack([integrate_recorded_path(formula, row["path"],
                                                  row["initial"], "midpoint") for row in rows])
    reproduction_error = float(np.max(np.abs(full - midpoint)))
    if reproduction_error > 1e-12:
        raise RuntimeError("Task 10 full41/40 failed to reproduce Task 9b midpoint")
    groups = {}
    for site in (0, 387, 460, 469):
        for regime in SITE_REGIMES:
            key = f"{site}:{regime}"
            indices = np.asarray([i for i, row in enumerate(rows)
                                  if int(row["segment_id"]) == site and row["regime"] == regime])
            if not len(indices):
                raise RuntimeError(f"Task 10 lacks registered group {key}")
            groups[key] = {"count": len(indices), "teacher_floor": _metrics(full[indices], authentic[indices]),
                           "views": {view: {str(n): {
                               "vs_full_path": _difference_metrics(predictions[(view, n)][indices], full[indices]),
                               "vs_teacher": _metrics(predictions[(view, n)][indices], authentic[indices])}
                               for n in SUBSTEPS} for view in VIEWS}}
    global_rows = {view: {str(n): {
        "vs_full_path": _difference_metrics(predictions[(view, n)], full),
        "vs_teacher": _metrics(predictions[(view, n)], authentic)}
        for n in SUBSTEPS} for view in VIEWS}
    candidate_views = ("start_only", "endpoints_oracle", "five_oracle", "nine_oracle",
                       "twentyone_oracle", "full41_oracle")
    minimum_view = next((view for view in candidate_views if all(
        max(groups[key]["views"][view]["40"]["vs_full_path"][metric]
            for metric in ("m_rmse", "h_rmse", "open_rmse")) <= .001
        for key in groups)), None)
    report = {"schema_version": "giada-roadmap-task10-input-sufficiency-v1",
              "valid": True, "code_revision": code_revision,
              "source_h5_sha256": source["h5_sha256"], "task9_selected_path_count": len(rows),
              "source_split_train_only": True, "new_teacher_paths_generated": False,
              "model_training_performed": False, "sealed_test_opened": False,
              "views": VIEWS, "internal_substeps": SUBSTEPS,
              "causally_available_views": ["start_only"],
              "oracle_views_not_available_during_autonomous_rollout": list(VIEWS[1:]),
              "full41_midpoint_reproduction_max_abs": reproduction_error,
              "full_path_teacher_floor": _metrics(full, authentic),
              "global": global_rows, "groups": groups,
              "minimum_view_at_registered_0p001_gate": minimum_view,
              "decision_scope": "Minimal oracle path view for isolated Ca_HVA gates on opened train paths; "
                "not proof that such a future path is causally available, nor a full-neuron closed-loop result."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    return report
