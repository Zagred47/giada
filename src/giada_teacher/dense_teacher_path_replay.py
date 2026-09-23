"""Task 9c: small, authentic NEURON replay of already-selected train paths.

The 40-point replay is an acceptance control. Denser observations are never
used to train or select a model and do not change teacher mechanisms/inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .physiological_path_floor import _metrics, integrate_recorded_path, load_verified_task9
from .physiological_voltage_paths import _decode, verify_source


def select_replay_rows(selected):
    """Fixed, provenance-only subset: two spike and one quiet per site."""
    chosen = []
    for site in (0, 387, 460, 469):
        for regime, count in (("spike", 2), ("quiet", 1)):
            pool = sorted((row for row in selected if int(row["segment_id"]) == site
                           and row["regime"] == regime),
                          key=lambda row: (int(row["transition_index"]), row["trajectory_id"]))
            if len(pool) < count:
                raise RuntimeError(f"Task 9c lacks preregistered {site}:{regime} support")
            chosen.extend(pool[:count])
    return chosen


def _actions(handle, index):
    from ..hayflow_data import InputAction

    raw = json.loads(_decode(handle["inputs/ordered_actions_json"][index]))
    return [InputAction(kind=row["kind"], offset_ms=row["offset_ms"],
                        synapse_id=row.get("synapse_id"),
                        weight_multiplier=row.get("weight_multiplier", 1.0),
                        duration_ms=row.get("duration_ms"),
                        amplitude_na=row.get("amplitude_na"),
                        metadata=row.get("metadata", {})) for row in raw]


def _replay_once(session, handle, index, segment_id, sample_interval_ms, snapshot_root):
    from ..hayflow_data import ProtocolTrajectory

    target_step = int(handle["metadata/step_index"][index])
    checkpoint_step = int(handle["metadata/snapshot_step_index"][index])
    trajectory_id = _decode(handle["metadata/trajectory_id"][index])
    indices = [i for i in range(int(handle.attrs["transition_count"]))
               if _decode(handle["metadata/trajectory_id"][i]) == trajectory_id
               and checkpoint_step <= int(handle["metadata/step_index"][i]) <= target_step]
    indices.sort(key=lambda i: int(handle["metadata/step_index"][i]))
    if not indices or int(handle["metadata/step_index"][indices[0]]) != checkpoint_step:
        raise RuntimeError("Task 9c checkpoint prefix is incomplete")
    ref = _decode(handle["metadata/native_snapshot_ref"][index])
    snapshot = (Path(snapshot_root) / ref).resolve()
    if not snapshot.is_file() or not snapshot.is_relative_to(Path(snapshot_root).resolve()):
        raise RuntimeError("Task 9c native snapshot missing or outside source")
    seed = int(handle["metadata/seed"][index])
    session._restore_native_snapshot(snapshot, handle["rng_state/t"][indices[0], :], seed)
    trajectory = ProtocolTrajectory(
        trajectory_id, _decode(handle["metadata/category"][index]),
        _decode(handle["metadata/protocol"][index]), seed, max(1, target_step + 1),
        _decode(handle["metadata/split"][index]),
        protocol_id=_decode(handle["metadata/protocol_id"][index]),
        protocol_variant=_decode(handle["metadata/protocol_variant"][index]),
        snapshot_source=_decode(handle["metadata/snapshot_source"][index]))
    session._active_trajectory = trajectory
    try:
        target_trace = None
        for row_index in indices:
            session._active_transition_id = int(row_index)
            actions = _actions(handle, row_index)
            is_target = row_index == index
            interval = sample_interval_ms if is_target else .025
            segment = session.audit.live_segments[segment_id]
            observe = (lambda: float(segment.v)) if is_target else (lambda: None)
            _, _, samples = session._drive_one_ms(float(session.h.t), actions, observe,
                                                   sample_interval_ms=interval)
            if is_target:
                target_trace = np.asarray(samples, dtype=np.float64)
        state = session.capture_boundary_state()
        errors = {category: float(np.max(np.abs(
            state[category] - handle[f"states/{category}/t_plus_1"][index, :])))
            for category in session.state_variables}
        rng_error = float(np.max(np.abs(np.asarray(session.audit._snapshot_rng_sequences())
                                        - handle["rng_state/t_plus_1"][index, :])))
        return target_trace, errors, rng_error
    finally:
        session._active_trajectory = None


def evaluate_dense_replay(formula, dataset_root, task9_source, teacher_repo, elm_repo,
                          output_dir, *, code_revision="unknown", progress=None):
    """Run 12 paired replays; fail closed if canonical 40-point replay differs."""
    import h5py
    from ..hayflow_teacher import TargetedDiagnosticDatasetSession, expected_audit_hashes

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prior, selected = load_verified_task9(task9_source)
    source = verify_source(dataset_root, progress=progress)
    if source["h5_sha256"] != prior["source"]["h5_sha256"]:
        raise RuntimeError("Task 9c dataset differs from Task 9")
    chosen = select_replay_rows(selected)
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, output_dir=output_dir.parent / ".task9c_teacher_runtime",
        calibration_source=dataset_root,
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        seed=271828, expected_teacher_hashes=expected_audit_hashes())
    session.prepare_teacher()
    results = []
    with h5py.File(source["h5_path"], "r") as handle:
        for ordinal, row in enumerate(chosen, 1):
            index, site = int(row["transition_index"]), int(row["segment_id"])
            if _decode(handle["metadata/split"][index]) != "train":
                raise RuntimeError("Task 9c selected a non-train row")
            position = source["mapping"][site]["voltage_index"]
            pair = source["mapping"][site]["state_indices"]
            stored = np.asarray(handle["microtraces/all_segment_voltage"][index, :, position], dtype=np.float64)
            initial = np.asarray(handle["states/mechanism_states/t"][index, pair], dtype=np.float64)
            teacher = np.asarray(handle["states/mechanism_states/t_plus_1"][index, pair], dtype=np.float64)
            traces, checks = {}, {}
            for interval in (.025, .005, .001):
                trace, state_errors, rng_error = _replay_once(
                    session, handle, index, site, interval, dataset_root)
                traces[str(interval)] = trace
                checks[str(interval)] = {"max_state_error": max(state_errors.values()),
                                         "state_errors": state_errors, "rng_error": rng_error}
            stored_error = float(np.max(np.abs(traces["0.025"] - stored)))
            maximum_endpoint_error = max(v["max_state_error"] for v in checks.values())
            maximum_rng_error = max(v["rng_error"] for v in checks.values())
            if stored_error > 1e-5 or maximum_endpoint_error > 1e-5 or maximum_rng_error > 1e-5:
                failure = {"schema_version": "giada-task9c-dense-teacher-replay-v1",
                           "valid": False, "blocker": "canonical_replay_mismatch",
                           "transition_index": index, "segment_id": site,
                           "stored_replay_max_error_mv": stored_error,
                           "maximum_endpoint_state_error": maximum_endpoint_error,
                           "maximum_rng_error": maximum_rng_error,
                           "checks": checks, "completed_rows": results,
                           "interpretation": "No dense-path attribution is permitted."}
                (output_dir / "final_report.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                return failure
            predictions = {}
            for label, trace in traces.items():
                # Generic midpoint quadrature for any uniform authentic grid.
                state = initial.copy()
                dt = float(label)
                for left, right in zip(trace[:-1], trace[1:]):
                    rates = formula.rates(float((left + right) / 2))
                    inf = np.asarray([rates["m_inf"], rates["h_inf"]])
                    tau = np.asarray([rates["m_tau_ms"], rates["h_tau_ms"]])
                    state = inf + (state - inf) * np.exp(-dt / tau)
                predictions[label] = state
            interpolated = integrate_recorded_path(formula, stored, initial, "linear_20")
            results.append({"transition_index": index, "segment_id": site, "regime": row["regime"],
                            "trajectory_id": row["trajectory_id"], "seed": row["seed"],
                            "stored_replay_max_error_mv": stored_error, "checks": checks,
                            "teacher_endpoint": teacher.tolist(),
                            "predictions": {**{k: v.tolist() for k, v in predictions.items()},
                                            "linear_20": interpolated.tolist()},
                            "dense_vs_interpolated_max_voltage_mv": float(np.max(np.abs(
                                traces["0.001"][::25] - stored)))})
            print(f"[GIADA Task 9c] {ordinal}/{len(chosen)} transition={index} site={site} {row['regime']}", flush=True)
    target = np.asarray([r["teacher_endpoint"] for r in results])
    metrics = {name: _metrics(np.asarray([r["predictions"][name] for r in results]), target)
               for name in ("0.025", "0.005", "0.001", "linear_20")}
    dense_converged = float(np.max(np.abs(np.asarray([r["predictions"]["0.005"] for r in results])
                                          - np.asarray([r["predictions"]["0.001"] for r in results]))))
    groups = {}
    for site in (0, 387, 460, 469):
        for regime in ("spike", "quiet"):
            subset = [r for r in results if r["segment_id"] == site and r["regime"] == regime]
            truth = np.asarray([r["teacher_endpoint"] for r in subset])
            groups[f"{site}:{regime}"] = {"count": len(subset), "methods": {
                name: _metrics(np.asarray([r["predictions"][name] for r in subset]), truth)
                for name in ("0.025", "0.005", "0.001", "linear_20")}}
    floor_calibrated = bool(dense_converged <= .001 and all(
        max(group["methods"]["0.001"][gate] for gate in ("m_rmse", "h_rmse")) <= .005
        for group in groups.values()))
    report = {"schema_version": "giada-task9c-dense-teacher-replay-v1", "valid": True,
              "code_revision": code_revision, "source_h5_sha256": source["h5_sha256"],
              "selected_from_task9_train_only": True, "path_count": len(results),
              "new_teacher_trajectories_generated": False, "authentic_native_snapshot_replay": True,
              "model_training_performed": False, "sample_intervals_ms": [.025, .005, .001],
              "metrics": metrics, "groups": groups,
              "dense_formula_convergence_max_abs": dense_converged,
              "dense_teacher_formula_floor_calibrated": floor_calibrated,
              "rows": results,
              "interpretation": "The dense CVode observations are a replay control, not an independent test. "
                                "Endpoint/state/RNG equivalence is required before attributing any floor."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
