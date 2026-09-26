"""GIADA 15c: teacher-interface bridge, not embedded test or causal replacement.

The Task 15b MLP needs four scheduled local-current samples unavailable in
the multicompartment teacher.  This study never fabricates those values.  It
compares already-frozen voltage-driven gate primitives on teacher validation
paths and reserves all teacher test splits for roadmap Task 16.
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

from .physiological_voltage_paths import (
    REGIMES, PhysiologicalPathConfig, _decode, _metrics, classify_regime,
    frozen_candidate_rollouts, verify_source,
)
from .physiological_path_floor import integrate_recorded_path
from .roadmap_gate_bottleneck_confirmation import _read_result_member
from .voltage_path_stress import verified_task5_root


EXPECTED_TASK15B_REPORT_SHA256 = "a20c6536a5f852fe9ba3ae323d00b717d1d4c371e5433519567aa36f364dba97"
EXPECTED_TASK15B_CODE_REVISION = "4262849a7fd7c321734eacef61f8d03b89817f5d"
CANDIDATES = ("lut_fine_path", "physical_fine_path_seed17",
              "physical_fine_path_seed29", "physical_fine_path_seed43")


@dataclass(frozen=True)
class EmbeddedInterfaceConfig:
    source_split: str = "validation"
    site_ids: tuple[int, ...] = (0, 387, 460, 469)
    regimes: tuple[str, ...] = REGIMES
    rows_per_site_regime: int = 16
    minimum_per_site_regime: int = 8
    sample_seed: int = 15471
    formula_floor_limit: float = 0.005
    one_ms_samples: int = 41

    def validate(self):
        if asdict(self) != asdict(EmbeddedInterfaceConfig()):
            raise ValueError("Task 15c configuration differs from preregistration")


def verified_task15b_result(source):
    raw = _read_result_member(source, "final_report.json")
    report = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if (digest != EXPECTED_TASK15B_REPORT_SHA256
            or report.get("schema_version") != "giada-roadmap-task15b-gate-bottleneck-v1"
            or report.get("code_revision") != EXPECTED_TASK15B_CODE_REVISION
            or not report.get("valid") or report.get("selection_used_sealed_or_counterfactual") is not False):
        raise ValueError("Task 15b immutable result contract mismatch")
    return {"final_report_sha256": digest, "code_revision": EXPECTED_TASK15B_CODE_REVISION,
            "baseline_800_m_rmse": report["sealed"]["baseline@800"]["m_rmse"]}


def verified_task5_any(source, cache):
    """Handle an exact Task-5 ZIP, extracted root, or Kaggle repack."""
    source, cache = Path(source), Path(cache)
    if source.is_dir():
        roots = [p.parent for p in source.rglob("frozen_scaling_checkpoints.pt")
                 if (p.parent / "final_report.json").is_file()
                 and (p.parent / "selection_freeze.json").is_file()]
        if len(roots) != 1:
            raise ValueError("Task 5 extracted root not unique")
        return verified_task5_root(roots[0], cache)
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("Task 5 source must be ZIP or directory")
    try:
        return verified_task5_root(source, cache)
    except RuntimeError as exc:
        if "archive SHA-256 mismatch" not in str(exc):
            raise
    required = ("final_report.json", "selection_freeze.json",
                "frozen_scaling_checkpoints.pt")
    with zipfile.ZipFile(source) as outer:
        nested = [name for name in outer.namelist()
                  if name.endswith("giada_primitive_scaling_laws.zip")]
        if len(nested) == 1:
            with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                archive = inner
                members = {name: [item for item in archive.infolist()
                                  if item.filename.endswith("/" + name)] for name in required}
                if not all(len(items) == 1 for items in members.values()):
                    raise ValueError("Nested Task 5 members incomplete")
                root = cache / "flattened_task5"
                root.mkdir(parents=True, exist_ok=False)
                for name in required:
                    with archive.open(members[name][0]) as reader, (root / name).open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
        else:
            members = {name: [item for item in outer.infolist()
                              if item.filename.endswith("/" + name)] for name in required}
            if not all(len(items) == 1 for items in members.values()):
                raise ValueError("Repacked Task 5 members incomplete")
            root = cache / "flattened_task5"
            root.mkdir(parents=True, exist_ok=False)
            for name in required:
                with outer.open(members[name][0]) as reader, (root / name).open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
    return verified_task5_root(root, cache)


def voltage_view(path, view):
    """Transform recorded samples for the existing frozen Task-5 stepper."""
    voltage = np.asarray(path, dtype=np.float64)
    if voltage.shape != (41,) or not np.isfinite(voltage).all():
        raise ValueError("Task 15c needs 41 finite voltage samples")
    if view == "left_available":
        drivers = voltage[:-1]
    elif view == "midpoint_oracle":
        drivers = (voltage[:-1] + voltage[1:]) / 2.0
    elif view == "right_oracle":
        drivers = voltage[1:]
    else:
        raise ValueError(view)
    return np.concatenate((voltage[:1], drivers))


def select_validation_rows(source, config=EmbeddedInterfaceConfig(), *, progress=None):
    import h5py

    config.validate()
    selected = {(site, regime): [] for site in config.site_ids for regime in REGIMES}
    checked, start = 0, time.perf_counter()
    mapping = source["mapping"]
    with h5py.File(source["h5_path"], "r") as handle:
        if int(handle.attrs["transition_count"]) != source["transition_count"]:
            raise RuntimeError("Task 15c HDF5 transition count mismatch")
        time_grid = np.asarray(handle["microtraces/time_offsets_ms"][:], dtype=np.float64)
        if time_grid.shape != (config.one_ms_samples,) or not np.allclose(
                time_grid, np.arange(config.one_ms_samples) / 40, atol=1e-12):
            raise RuntimeError("Task 15c microtrace time grid mismatch")
        splits = np.asarray([_decode(item) for item in handle["metadata/split"][:]])
        eligible = np.flatnonzero(splits == config.source_split)
        if len(eligible) == 0:
            raise RuntimeError("Task 15c teacher validation split is empty")
        indices = np.random.default_rng(config.sample_seed).permutation(eligible)
        for index in indices:
            checked += 1
            trace = np.asarray(handle["microtraces/all_segment_voltage"][int(index)], dtype=np.float64)
            initial_row = handle["states/mechanism_states/t"][int(index)]
            target_row = handle["states/mechanism_states/t_plus_1"][int(index)]
            for site in config.site_ids:
                position = mapping[site]["voltage_index"]
                path = trace[:, position]
                regime = classify_regime(path)
                if regime not in REGIMES or len(selected[(site, regime)]) >= config.rows_per_site_regime:
                    continue
                pair = mapping[site]["state_indices"]
                initial = np.asarray(initial_row[pair], dtype=np.float64)
                target = np.asarray(target_row[pair], dtype=np.float64)
                if (path.shape != (41,) or not np.isfinite(path).all()
                        or not np.isfinite(initial).all() or not np.isfinite(target).all()
                        or np.any(initial < -1e-8) or np.any(initial > 1 + 1e-8)
                        or np.any(target < -1e-8) or np.any(target > 1 + 1e-8)):
                    raise RuntimeError(f"Task 15c invalid teacher row {index}/{site}")
                v0 = float(handle["states/voltage/t"][int(index), position])
                v1 = float(handle["states/voltage/t_plus_1"][int(index), position])
                if max(abs(path[0] - v0), abs(path[-1] - v1)) > 1e-4:
                    raise RuntimeError(f"Task 15c voltage boundary mismatch {index}/{site}")
                selected[(site, regime)].append({
                    "transition_index": int(index), "segment_id": site, "regime": regime,
                    "split": config.source_split, "path": path.copy(),
                    "initial": initial, "teacher_endpoint": target,
                    "trajectory_id": _decode(handle["metadata/trajectory_id"][int(index)]),
                    "seed": int(handle["metadata/seed"][int(index)])})
            if progress and checked % 500 == 0:
                progress(checked, len(indices), sum(map(len, selected.values())),
                         (time.perf_counter() - start) / 60.0)
            if all(len(group) >= config.rows_per_site_regime for group in selected.values()):
                break
    rows = [row for key in selected for row in selected[key]]
    support = {"source_split": config.source_split, "eligible_transition_count": len(eligible),
               "checked_transition_count": checked, "selected_row_count": len(rows),
               "unique_transition_count": len({row["transition_index"] for row in rows}),
               "unique_trajectory_count": len({row["trajectory_id"] for row in rows}),
               "groups": {f"{site}:{regime}": len(selected[(site, regime)])
                          for site in config.site_ids for regime in REGIMES},
               "minimum_support_met": all(len(group) >= config.minimum_per_site_regime
                                          for group in selected.values())}
    if not rows:
        raise RuntimeError("Task 15c selected no teacher paths")
    return rows, support


def _group_metrics(prediction, target, rows, config):
    result = {}
    for site in config.site_ids:
        for regime in REGIMES:
            indices = [i for i, row in enumerate(rows)
                       if row["segment_id"] == site and row["regime"] == regime]
            if indices:
                result[f"{site}:{regime}"] = {
                    "count": len(indices), **_metrics(prediction[indices], target[indices])}
    return result


def run_interface_bridge(formula, dataset_root, task5_source, task15b_source, output_dir,
                         config=EmbeddedInterfaceConfig(), *, code_revision="unknown", progress=None):
    """Validation-only frozen comparison. Never execute Task15b with fake drive."""
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Task 15c output exists: {output_dir}")
    prerequisite = verified_task15b_result(task15b_source)
    source = verify_source(Path(dataset_root), progress=progress)
    task5_root = verified_task5_any(Path(task5_source), output_dir.parent / ".15c_task5_cache")
    output_dir.mkdir(parents=True)
    rows, support = select_validation_rows(source, config, progress=progress)
    print(f"[GIADA 15c] validation: {len(rows)} percorsi da {support['checked_transition_count']} transizioni", flush=True)
    target = np.stack([row["teacher_endpoint"] for row in rows])
    views = {}
    for view in ("left_available", "midpoint_oracle", "right_oracle"):
        prepared = [{**row, "path": voltage_view(row["path"], view)} for row in rows]
        formula_prediction = np.stack([
            integrate_recorded_path(formula, row["path"], row["initial"],
                                    "left" if view == "left_available" else
                                    "midpoint" if view == "midpoint_oracle" else "right")
            for row in rows])
        # The frozen stepper reads prepared path[1:] at each 0.025-ms update.
        predictions = frozen_candidate_rollouts(formula, task5_root, prepared)
        floor = _group_metrics(formula_prediction, target, rows, config)
        candidates = {name: _group_metrics(predictions[name], target, rows, config)
                      for name in CANDIDATES}
        vs_formula = {name: _group_metrics(predictions[name], formula_prediction, rows, config)
                      for name in CANDIDATES}
        views[view] = {"formula_vs_teacher": floor,
                       "candidate_vs_teacher": candidates,
                       "candidate_vs_formula": vs_formula,
                       "global_formula_vs_teacher": _metrics(formula_prediction, target),
                       "global_candidate_vs_teacher": {name: _metrics(predictions[name], target)
                                                      for name in CANDIDATES}}
        print(f"[GIADA 15c] vista {view}: formula m RMSE={views[view]['global_formula_vs_teacher']['m_rmse']:.4g}", flush=True)
    midpoint = views["midpoint_oracle"]
    floor_calibrated = bool(support["minimum_support_met"] and all(
        max(group["m_rmse"], group["h_rmse"]) <= config.formula_floor_limit
        for group in midpoint["formula_vs_teacher"].values()))
    scores = {}
    for candidate in CANDIDATES:
        group_rows = midpoint["candidate_vs_teacher"][candidate]
        scores[candidate] = float(np.mean([
            max(row["m_rmse"], row["h_rmse"]) for row in group_rows.values()]))
    selected = min(scores, key=lambda name: (scores[name], name)) if floor_calibrated else None
    report = {"schema_version": "giada-roadmap-task15c-interface-bridge-v1",
              "valid": True, "code_revision": code_revision,
              "task15b_prerequisite": prerequisite,
              "task5_source": {"final_report_sha256": hashlib.sha256(
                  (task5_root / "final_report.json").read_bytes()).hexdigest()},
              "teacher_source": {k: source[k] for k in ("h5_sha256", "manifest_sha256",
                                                          "state_schema_sha256", "teacher_commit")},
              "config": asdict(config), "support": support, "views": views,
              "midpoint_formula_floor_calibrated": floor_calibrated,
              "development_macro_gate_scores": scores,
              "teacher_forced_candidate_for_task16": selected,
              "task16_frozen_evaluation_authorized": bool(floor_calibrated and selected),
              "task15b_checkpoint_evaluated_in_teacher": False,
              "task15b_interface": {
                  "required_local_inputs": "V0,m0,h0,gbar_nominal,mask,E_Ca,four scheduled local-current values",
                  "teacher_has_equivalent_four_local_current_values": False,
                  "teacher_somatic_iclamp_is_not_total_local_drive": True,
                  "future_voltage_path_is_not_causal_input_at_ms_start": True},
              "models_retrained": False, "teacher_test_splits_opened": False,
              "interpretation": "This is a validation-only interface bridge. Left voltage samples are recorded causal observations at each substep; midpoint/right views are teacher-forced oracles. No Task15b fake-input score or membrane replacement is claimed."}
    (output_dir / "selected_validation_paths.json").write_text(json.dumps([
        {k: row[k] for k in ("transition_index", "segment_id", "regime", "split",
                                  "trajectory_id", "seed")} for row in rows], indent=2), encoding="utf-8")
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
