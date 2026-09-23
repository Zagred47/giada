"""TG-01: paired full-state canary for the number of internal updates per ms.

This is a bounded learnability screen, not a proof that any temporal resolution
is intrinsically sufficient for every possible surrogate architecture.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .dense_teacher_path_replay import _actions
from .physiological_path_floor import load_verified_task9
from .physiological_voltage_paths import _decode, verify_source


SUBSTEPS = (1, 2, 4, 8)
CHECKPOINTS = (100, 300, 600)
STATE_ATOL = 1e-5
EVENT_SLOTS_PER_BIN = 64
FEATURE_WIDTH = EVENT_SLOTS_PER_BIN * 4 + 4


def select_rows(selected):
    """One transition ID per frozen Task 9 provenance row, sorted before outcomes."""
    by_index = {}
    for row in selected:
        by_index.setdefault(int(row["transition_index"]), row)
    return [by_index[index] for index in sorted(by_index)]


def trajectory_role(trajectory_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(trajectory_id.encode()).digest()[:4], "big") % 10
    return "train" if bucket < 6 else "development" if bucket < 8 else "test"


def realized_schedule(rows, synapse_to_segment):
    """Ordered timestamped U_realized tokens plus exact current pulse fields.

    The bin is a routing container, not a quantization of event time.  Each
    token retains its continuous offset, synapse ID, segment and magnitude.
    """
    features = np.zeros((8, FEATURE_WIDTH), dtype=np.float32)
    slot_counts = np.zeros(8, dtype=np.int64)
    current_count = 0
    for row in sorted(rows, key=lambda item: (float(item["offset_ms"]),
                                             int(item.get("synapse_id") or -1))):
        offset = float(row["offset_ms"])
        if not 0 <= offset <= 1:
            raise ValueError("input offset is outside its macro-step")
        if row["kind"] == "synaptic_event":
            segment = synapse_to_segment[int(row["synapse_id"])]
            bin_index = min(7, int(offset * 8))
            slot = int(slot_counts[bin_index])
            if slot >= EVENT_SLOTS_PER_BIN:
                raise RuntimeError("U_realized exceeds registered event-token capacity")
            slot_counts[bin_index] += 1
            quantity = row.get("released_quantity")
            amount = float(1.0 if quantity is None else quantity)
            inhibitory = bool(row.get("inhibitory_state_increment"))
            start = slot * 4
            features[bin_index, start:start + 4] = (
                (segment + 1) / 643, offset,
                -amount if inhibitory else amount,
                (int(row["synapse_id"]) + 1) / 10000)
        elif row["kind"] == "somatic_current":
            current_count += 1
            if current_count > 1:
                raise RuntimeError("TG-01 requires at most one somatic current action per ms")
            duration = float(row["duration_ms"])
            amplitude = float(row["amplitude_na"])
            for i in range(8):
                overlap = max(0.0, min((i + 1) / 8, offset + duration) - max(i / 8, offset))
                if overlap > 0:
                    features[i, -4:] = (amplitude, offset, duration, overlap * 8)
        else:
            raise ValueError(f"unsupported causal input kind {row['kind']}")
    if not np.isfinite(features).all():
        raise RuntimeError("U_realized features contain NaN/Inf")
    return features


def _flat(state, categories):
    return np.concatenate([np.asarray(state[name], dtype=np.float64) for name in categories])


def _replay_dense(session, handle, index, prefix_indices, categories):
    from ..hayflow_data import ProtocolTrajectory

    step = int(handle["metadata/step_index"][index])
    checkpoint = int(handle["metadata/snapshot_step_index"][index])
    trajectory_id = _decode(handle["metadata/trajectory_id"][index])
    prefix = [i for i in prefix_indices[trajectory_id]
              if checkpoint <= int(handle["metadata/step_index"][i]) <= step]
    if not prefix or int(handle["metadata/step_index"][prefix[0]]) != checkpoint:
        raise RuntimeError("TG-01 missing native checkpoint prefix")
    root = Path(session.dataset_root).resolve()
    ref = _decode(handle["metadata/native_snapshot_ref"][index])
    snapshot = (root / ref).resolve()
    if not snapshot.is_relative_to(root) or not snapshot.is_file():
        raise RuntimeError("TG-01 snapshot missing or outside dataset root")
    seed = int(handle["metadata/seed"][index])
    session._restore_native_snapshot(snapshot, handle["rng_state/t"][prefix[0], :], seed)
    trajectory = ProtocolTrajectory(trajectory_id, _decode(handle["metadata/category"][index]),
                                    _decode(handle["metadata/protocol"][index]), seed,
                                    max(1, step + 1), "train",
                                    protocol_id=_decode(handle["metadata/protocol_id"][index]),
                                    protocol_variant=_decode(handle["metadata/protocol_variant"][index]),
                                    snapshot_source=_decode(handle["metadata/snapshot_source"][index]))
    session._active_trajectory = trajectory
    try:
        captured = None
        for row_index in prefix:
            session._active_transition_id = int(row_index)
            target = row_index == index
            observer = (lambda: _flat(session.capture_boundary_state(), categories)) if target else (lambda: None)
            _, _, samples = session._drive_one_ms(float(session.h.t), _actions(handle, row_index), observer,
                                                   sample_interval_ms=.125 if target else .025)
            if target:
                captured = np.stack(samples)
        final_state = session.capture_boundary_state()
        state_errors = {name: float(np.max(np.abs(
            final_state[name] - handle[f"states/{name}/t_plus_1"][index, :])))
            for name in categories}
        rng_error = float(np.max(np.abs(np.asarray(session.audit._snapshot_rng_sequences())
                                        - handle["rng_state/t_plus_1"][index, :])))
        return captured, state_errors, rng_error
    finally:
        session._active_trajectory = None


def acquire_full_state_matrix(dataset_root, task9_source, teacher_repo, elm_repo,
                              output_dir, *, progress=None):
    """Acquire only the frozen Task 9 train transitions, with hard replay gates."""
    import h5py
    import pandas as pd
    from ..hayflow_teacher import TargetedDiagnosticDatasetSession, expected_audit_hashes

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prior, selected = load_verified_task9(task9_source)
    source = verify_source(dataset_root, progress=progress)
    if source["h5_sha256"] != prior["source"]["h5_sha256"]:
        raise RuntimeError("TG-01 source differs from frozen Task 9")
    chosen = select_rows(selected)
    session = TargetedDiagnosticDatasetSession(
        elm_repo, teacher_repo, output_dir=output_dir.parent / ".task10_teacher_runtime",
        calibration_source=dataset_root,
        dataset_config_path=Path(elm_repo) / "configs/hayflow/targeted_transition_dataset_v1_1.yml",
        seed=271828, expected_teacher_hashes=expected_audit_hashes())
    session.dataset_root = Path(dataset_root)
    session.prepare_teacher()
    categories = ("voltage",) + tuple(name for name in session.state_variables if name != "voltage")
    widths = {name: len(session.state_variables[name]) for name in categories}
    mapping = {int(row.synapse_id): int(row.segment_id)
               for row in pd.read_parquet(Path(dataset_root) / "synapses.parquet").itertuples()}
    paths, inputs, ids, roles, checks = [], [], [], [], []
    acquisition_start = time.monotonic()
    with h5py.File(source["h5_path"], "r") as handle:
        prefix_indices = {}
        for i, split in enumerate(handle["metadata/split"][:]):
            if _decode(split) == "train":
                prefix_indices.setdefault(_decode(handle["metadata/trajectory_id"][i]), []).append(i)
        for values in prefix_indices.values():
            values.sort(key=lambda i: int(handle["metadata/step_index"][i]))
        for ordinal, row in enumerate(chosen, 1):
            index = int(row["transition_index"])
            if _decode(handle["metadata/split"][index]) != "train":
                raise RuntimeError("TG-01 opened non-train source")
            states, state_errors, rng_error = _replay_dense(
                session, handle, index, prefix_indices, categories)
            original_start = np.concatenate([handle[f"states/{name}/t"][index, :]
                                             for name in categories])
            original_end = np.concatenate([handle[f"states/{name}/t_plus_1"][index, :]
                                           for name in categories])
            boundary_error = float(max(np.max(np.abs(states[0] - original_start)),
                                       np.max(np.abs(states[-1] - original_end))))
            micro_voltage = np.asarray(handle["microtraces/all_segment_voltage"][index, ::5, :])
            micro_error = float(np.max(np.abs(states[:, :widths["voltage"]] - micro_voltage)))
            max_state_error = max(state_errors.values())
            if max(boundary_error, micro_error, max_state_error, rng_error) > STATE_ATOL:
                failure = {"valid": False, "blocker": "authentic_replay_mismatch", "transition_index": index,
                           "boundary_error": boundary_error, "state_errors": state_errors,
                           "micro_voltage_error_mv": micro_error,
                           "rng_error": rng_error, "completed_count": len(paths)}
                (output_dir / "acquisition_report.json").write_text(json.dumps(failure, indent=2))
                return failure
            realized = json.loads(_decode(handle["inputs/U_realized_json"][index]))
            features = realized_schedule(realized, mapping)
            paths.append(states.astype(np.float32))
            inputs.append(features)
            ids.append(index)
            roles.append(trajectory_role(str(row["trajectory_id"])))
            checks.append({"transition_index": index, "maximum_boundary_error": boundary_error,
                           "maximum_micro_voltage_error_mv": micro_error,
                           "maximum_state_error": max_state_error, "rng_error": rng_error})
            if ordinal % 10 == 0 or ordinal == len(chosen):
                elapsed = time.monotonic() - acquisition_start
                eta = elapsed / ordinal * (len(chosen) - ordinal) / 60
                print(f"[GIADA TG-01][teacher] {ordinal}/{len(chosen)} "
                      f"({ordinal/len(chosen):.1%}) ETA {eta:.1f} min", flush=True)
    if not all(roles.count(role) >= 12 for role in ("train", "development", "test")):
        raise RuntimeError("TG-01 lacks grouped train/development/test support")
    np.savez_compressed(output_dir / "training_support.npz", states=np.stack(paths),
                        inputs=np.stack(inputs), transition_ids=np.asarray(ids),
                        roles=np.asarray(roles))
    report = {"valid": True, "source_h5_sha256": source["h5_sha256"],
              "selected_transition_count": len(chosen), "role_counts": {
                  role: roles.count(role) for role in ("train", "development", "test")},
              "state_widths": widths, "state_category_order": categories,
              "maximum_boundary_error": max(r["maximum_boundary_error"] for r in checks),
              "maximum_micro_voltage_error_mv": max(r["maximum_micro_voltage_error_mv"] for r in checks),
              "maximum_state_error": max(r["maximum_state_error"] for r in checks),
              "maximum_rng_error": max(r["rng_error"] for r in checks),
              "support_sha256": hashlib.sha256((output_dir / "training_support.npz").read_bytes()).hexdigest(),
              "new_teacher_trajectories": False, "sealed_dataset_splits_opened": False,
              "checks": checks}
    (output_dir / "acquisition_report.json").write_text(json.dumps(report, indent=2))
    return report


def _substep_input(schedule, start_bin: int, bins_per_call: int):
    values = np.zeros_like(schedule)
    values[..., start_bin:start_bin + bins_per_call, :] = schedule[..., start_bin:start_bin + bins_per_call, :]
    return values.reshape(*schedule.shape[:-2], -1)


def _model_class(torch):
    class StepNet(torch.nn.Module):
        def __init__(self, state_width: int, hidden: int = 96):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_width + 8 * FEATURE_WIDTH + 1, hidden),
                torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                torch.nn.SiLU(), torch.nn.Linear(hidden, state_width))
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

        def forward(self, state, schedule, dt):
            dt_column = torch.full((state.shape[0], 1), dt, device=state.device,
                                   dtype=state.dtype)
            return state + dt * self.net(torch.cat((state, schedule, dt_column), dim=-1))

    return StepNet


def _rollout_one_ms(model, states, schedule, substeps, mean, scale, torch):
    normalized = (states - mean) / scale
    bins_per_call = 8 // substeps
    for step in range(substeps):
        masked = _substep_input(schedule, step * bins_per_call, bins_per_call)
        normalized = model(normalized, torch.as_tensor(masked, device=states.device),
                           1.0 / substeps)
    return normalized * scale + mean


def _metric_voltage(prediction, target, voltage_width):
    error = np.asarray(prediction)[..., :voltage_width] - np.asarray(target)[..., :voltage_width]
    return float(np.sqrt(np.mean(error ** 2)))


def _eight_ms_windows(dataset_root, selected_indices, original_states, source_categories,
                      synapse_to_segment, maximum=48):
    """Existing source trajectories only; no fresh teacher or new test split."""
    import h5py

    manifest = json.loads((Path(dataset_root) / "dataset_manifest.json").read_text())
    windows = []
    with h5py.File(Path(dataset_root) / manifest.get("transition_store", "transition_dataset.h5"), "r") as h:
        by_trajectory = {}
        for i, split in enumerate(h["metadata/split"][:]):
            if _decode(split) == "train":
                trajectory_id = _decode(h["metadata/trajectory_id"][i])
                by_trajectory.setdefault(trajectory_id, {})[
                    int(h["metadata/step_index"][i])] = i
        for local_id in selected_indices:
            index = int(local_id)
            trajectory_id = _decode(h["metadata/trajectory_id"][index])
            step = int(h["metadata/step_index"][index])
            members = [by_trajectory[trajectory_id].get(step + k) for k in range(8)]
            if any(i is None or _decode(h["metadata/split"][i]) != "train" for i in members):
                continue
            inputs = np.stack([realized_schedule(json.loads(_decode(h["inputs/U_realized_json"][i])),
                                                 synapse_to_segment) for i in members])
            final = np.concatenate([h[f"states/{name}/t_plus_1"][members[-1], :]
                                    for name in source_categories]).astype(np.float32)
            soma_trace = np.asarray([h["states/voltage/t"][members[0], 0]] + [
                h["states/voltage/t_plus_1"][i, 0] for i in members], dtype=np.float32)
            windows.append({"transition_index": index, "initial": original_states[index],
                            "inputs": inputs, "final": final,
                            "soma_boundary_spike": bool(soma_trace.max() >= 0)})
            if len(windows) >= maximum:
                break
    return windows


def _binary_f1(labels, predictions):
    truth = np.asarray(labels, dtype=bool)
    pred = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0


def train_paired_granularity_models(dataset_root, output_dir, *, code_revision="unknown"):
    """Same architecture, stream and optimizer steps; only internal calls vary.

    All source trajectories are in the original train split. Here test means
    a trajectory-held-out diagnostic subset, never the sealed dataset test.
    """
    import h5py
    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TG-01 matched full-state canary requires a CUDA GPU")
    output_dir = Path(output_dir)
    acquisition = json.loads((output_dir / "acquisition_report.json").read_text())
    if not acquisition["valid"]:
        raise RuntimeError("TG-01 acquisition did not pass authentic replay")
    support = output_dir / "training_support.npz"
    if hashlib.sha256(support.read_bytes()).hexdigest() != acquisition["support_sha256"]:
        raise RuntimeError("TG-01 training support fingerprint mismatch")
    with np.load(support) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        schedules = np.asarray(data["inputs"], dtype=np.float32)
        indices = np.asarray(data["transition_ids"], dtype=np.int64)
        roles = np.asarray(data["roles"])
    train_ids = np.flatnonzero(roles == "train")
    dev_ids = np.flatnonzero(roles == "development")
    test_ids = np.flatnonzero(roles == "test")
    if states.ndim != 3 or states.shape[1] != 9 or schedules.shape != (len(states), 8, FEATURE_WIDTH):
        raise RuntimeError("TG-01 fine-grid support shape mismatch")
    mean_np = states[train_ids].reshape(-1, states.shape[-1]).mean(axis=0)
    scale_np = states[train_ids].reshape(-1, states.shape[-1]).std(axis=0)
    scale_np = np.maximum(scale_np, 1e-3)
    mean = torch.as_tensor(mean_np, device="cuda")
    scale = torch.as_tensor(scale_np, device="cuda")
    state_t = torch.as_tensor(states, device="cuda")
    voltage_width = int(acquisition["state_widths"]["voltage"])
    weights = torch.ones(states.shape[-1], device="cuda")
    weights[:voltage_width] = 3.0
    StepNet = _model_class(torch)
    histories = []
    frozen = {}
    parameter_count = None
    start_clock = time.monotonic()

    def dev_score(model, n):
        model.eval()
        with torch.inference_mode():
            predictions = _rollout_one_ms(model, state_t[dev_ids, 0], schedules[dev_ids],
                                          n, mean, scale, torch)
            normalized_error = (predictions - state_t[dev_ids, -1]) / scale
            return float((weights * normalized_error.square()).mean().item())

    for n in SUBSTEPS:
        for seed in (17, 29):
            torch.manual_seed(seed)
            rng = np.random.default_rng(12000 + seed + n * 100)
            model = StepNet(states.shape[-1]).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
            count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_count is not None and count != parameter_count:
                raise RuntimeError("TG-01 candidate parameter counts differ")
            parameter_count = count
            best = (float("inf"), None, None)
            for step in range(1, CHECKPOINTS[-1] + 1):
                model.train()
                batch = rng.choice(train_ids, size=min(32, len(train_ids)), replace=True)
                starts = rng.integers(0, n, size=len(batch))
                begin = starts * (8 // n)
                end = begin + (8 // n)
                before = state_t[batch, begin]
                target = state_t[batch, end]
                input_rows = np.stack([_substep_input(schedules[row], int(start), 8 // n)
                                       for row, start in zip(batch, begin)])
                features = torch.as_tensor(input_rows, device="cuda")
                output = model((before - mean) / scale, features, 1.0 / n)
                delta = output - (target - mean) / scale
                loss = (weights * torch.nn.functional.smooth_l1_loss(
                    delta, torch.zeros_like(delta), reduction="none")).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"TG-01 nonfinite training loss: n={n}, seed={seed}, step={step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step in CHECKPOINTS:
                    score = dev_score(model, n)
                    histories.append({"substeps_per_ms": n, "seed": seed, "optimizer_step": step,
                                      "train_batch_loss": float(loss.item()),
                                      "development_weighted_normalized_mse": score})
                    print(f"[GIADA TG-01][GPU] n={n} seed={seed} step={step}/{CHECKPOINTS[-1]} "
                          f"dev={score:.5g} ETA~{(time.monotonic()-start_clock)/len(histories)*(24-len(histories))/60:.1f} min",
                          flush=True)
                    if score < best[0]:
                        best = (score, step, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()})
            frozen[(n, seed)] = best
    # All selections are frozen on grouped development trajectories. Open the
    # trajectory-held-out diagnostic subset only now.
    evaluations = []
    persistence = _metric_voltage(states[test_ids, 0], states[test_ids, -1], voltage_width)
    for n in SUBSTEPS:
        for seed in (17, 29):
            score, selected_step, state_dict = frozen[(n, seed)]
            model = StepNet(states.shape[-1]).cuda()
            model.load_state_dict(state_dict)
            model.eval()
            with torch.inference_mode():
                prediction = _rollout_one_ms(model, state_t[test_ids, 0],
                                             schedules[test_ids], n, mean, scale, torch)
            predicted = prediction.cpu().numpy()
            truth = states[test_ids, -1]
            evaluations.append({"substeps_per_ms": n, "seed": seed,
                                "selected_optimizer_step": selected_step,
                                "development_score": score,
                                "heldout_one_ms_voltage_rmse_mv": _metric_voltage(
                                    predicted, truth, voltage_width),
                                "heldout_one_ms_full_normalized_rmse": float(np.sqrt(np.mean(
                                    ((predicted - truth) / scale_np) ** 2))),
                                "heldout_voltage_violation_count": int(np.count_nonzero(
                                    (predicted[:, :voltage_width] < -120) |
                                    (predicted[:, :voltage_width] > 80))),
                                "heldout_nonfinite_count": int(np.size(predicted) - np.isfinite(predicted).sum())})
    import pandas as pd
    synapse_to_segment = {int(row.synapse_id): int(row.segment_id)
                          for row in pd.read_parquet(Path(dataset_root) / "synapses.parquet").itertuples()}
    originals = {int(index): states[i, 0] for i, index in enumerate(indices)}
    windows = _eight_ms_windows(dataset_root, indices[test_ids], originals,
                                acquisition["state_category_order"], synapse_to_segment)
    rollout_rows = []
    persistence_8ms = None
    if windows:
        truth_8 = np.stack([row["final"] for row in windows])
        start_8 = np.stack([row["initial"] for row in windows])
        input_8 = np.stack([row["inputs"] for row in windows])
        persistence_8ms = _metric_voltage(start_8, truth_8, voltage_width)
        for n in SUBSTEPS:
            for seed in (17, 29):
                model = StepNet(states.shape[-1]).cuda()
                model.load_state_dict(frozen[(n, seed)][2])
                model.eval()
                torch.cuda.synchronize()
                t0 = time.monotonic()
                with torch.inference_mode():
                    evolving = torch.as_tensor(start_8, device="cuda")
                    predicted_spikes = torch.zeros(len(windows), device="cuda", dtype=torch.bool)
                    voltage_violations = torch.zeros((), device="cuda", dtype=torch.int64)
                    for k in range(8):
                        evolving = _rollout_one_ms(model, evolving, input_8[:, k],
                                                  n, mean, scale, torch)
                        voltage_violations += torch.count_nonzero(
                            (evolving[:, :voltage_width] < -120) |
                            (evolving[:, :voltage_width] > 80))
                        predicted_spikes |= evolving[:, 0] >= 0
                torch.cuda.synchronize()
                elapsed = time.monotonic() - t0
                result = evolving.cpu().numpy()
                rollout_rows.append({"substeps_per_ms": n, "seed": seed,
                                     "eight_ms_voltage_rmse_mv": _metric_voltage(result, truth_8, voltage_width),
                                     "soma_boundary_spike_proxy_f1": _binary_f1(
                                         [row["soma_boundary_spike"] for row in windows],
                                         predicted_spikes.cpu().numpy()),
                                     "voltage_violation_count": int(voltage_violations.item()),
                                     "nonfinite_count": int(np.size(result) - np.isfinite(result).sum()),
                                     "seconds_per_simulated_ms": elapsed / (8 * len(windows))})
    # Predeclared decision is conservative: a candidate must beat persistence
    # and pass the boundary spike proxy, not merely win among failed arms.
    decision = "INCONCLUSIVE_INSUFFICIENT_LEARNABILITY_OR_SUPPORT"
    selected_substeps = None
    if len(windows) >= 8 and sum(row["soma_boundary_spike"] for row in windows) >= 2:
        medians = {n: {"rmse": float(np.median([r["eight_ms_voltage_rmse_mv"]
                                               for r in rollout_rows if r["substeps_per_ms"] == n])),
                       "f1": float(np.median([r["soma_boundary_spike_proxy_f1"]
                                             for r in rollout_rows if r["substeps_per_ms"] == n])),
                       "safe": all(r["voltage_violation_count"] == 0 and r["nonfinite_count"] == 0
                                   for r in rollout_rows if r["substeps_per_ms"] == n)}
                   for n in SUBSTEPS}
        safe = [row for row in medians.values() if row["safe"]]
        best_rmse = min((row["rmse"] for row in safe), default=float("inf"))
        best_f1 = max((row["f1"] for row in safe), default=0.0)
        if np.isfinite(best_rmse) and best_rmse <= .8 * persistence_8ms and best_f1 >= .5:
            eligible = [n for n in SUBSTEPS if medians[n]["safe"]
                        and medians[n]["rmse"] <= 1.10 * best_rmse
                        and medians[n]["f1"] >= best_f1 - .05]
            if eligible:
                selected_substeps = eligible[0]
                decision = ("ONE_MS_EXTERNAL_ONE_INTERNAL_STEP_SUFFICIENT_WITHIN_SCREEN"
                            if selected_substeps == 1 else
                            "ONE_MS_EXTERNAL_WITH_INTERNAL_SUBSTEPS_PREFERRED_WITHIN_SCREEN")
    report = {"schema_version": "giada-task10-temporal-granularity-matrix-v1",
              "valid": True, "code_revision": code_revision,
              "acquisition_support_sha256": acquisition["support_sha256"],
              "source_h5_sha256": acquisition["source_h5_sha256"],
              "candidate_kind": "full-state flat residual MLP canary",
              "same_parameter_count": parameter_count,
              "substeps_per_ms": SUBSTEPS, "checkpoint_steps": CHECKPOINTS,
              "seeds": [17, 29], "train_transition_count": len(train_ids),
              "development_transition_count": len(dev_ids),
              "heldout_transition_count": len(test_ids),
              "heldout_is_original_train_split": True,
              "heldout_used_for_selection": False,
              "persistence_heldout_one_ms_voltage_rmse_mv": persistence,
              "persistence_eight_ms_voltage_rmse_mv": persistence_8ms,
              "rollout_window_count": len(windows),
              "rollout_positive_boundary_spike_proxy_count": sum(
                  row["soma_boundary_spike"] for row in windows),
              "rollout_rows": rollout_rows, "decision": decision,
              "selected_internal_substeps_per_ms": selected_substeps,
              "training_history": histories, "evaluations": evaluations,
              "methodology_limit": "A failed small-capacity canary cannot prove 1 ms impossible. "
                                   "This compares internal composition over the same 1 ms external horizon, "
                                   "not a standalone full biological surrogate."}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    return report
