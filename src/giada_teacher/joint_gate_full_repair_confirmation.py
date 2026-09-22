"""Task 3e: exact frozen confirmation of the Task 3d full-repair candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest
from .joint_gate_cell_playground import JointGateCellConfig, _constructors, _evaluate, _file_sha, _joint_row, _sha
from .joint_gate_generalization_diagnosis import _enrich, _evaluate_candidate, _random_row


EXPECTED_TASK3D_ARCHIVE_SHA256 = "4733cc943abb668291cb05bc581f1e47809be7bdb099fae14676486f509c9edf"
EXPECTED_TASK3D_REPORT_SHA256 = "4519c485e0dc5ea37b24de6e823d48d0b4b703ea91cefa3f73cf3a22192b4827"
EXPECTED_TASK3D_CHECKPOINT_SHA256 = "d5d10a4b57bc0c5c07a2afd7b7f9d2ac3d31bd6ae69f040ca419d9ac1e64ac1f"


@dataclass(frozen=True)
class JointGateFullRepairConfirmationConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    width: int = 23
    fixed_step: int = 50000
    development_reproduction_atol: float = 1e-10
    sealed_mean_score_max: float = 1.0
    sealed_max_seed_score_max: float = 1.0
    m_inf_rmse_max: float = 0.01
    h_inf_rmse_max: float = 0.01
    log_tau_rmse_max: float = 0.15
    maximum_monotonic_reversal: float = 0.001

    def validate(self) -> None:
        if self.seeds != (17, 29, 43) or self.width != 23 or self.fixed_step != 50000:
            raise ValueError("Task 3e freezes the preregistered Task 3d candidate")
        if min(self.sealed_mean_score_max, self.sealed_max_seed_score_max, self.log_tau_rmse_max) <= 0:
            raise ValueError("invalid Task 3e gates")


def verified_task3d_root(source, cache_dir):
    source = Path(source); cache_dir = Path(cache_dir)
    if source.is_file():
        if _file_sha(source) != EXPECTED_TASK3D_ARCHIVE_SHA256:
            raise RuntimeError("Task 3d archive SHA-256 mismatch")
        if cache_dir.exists(): shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        with zipfile.ZipFile(source) as archive:
            root = cache_dir.resolve()
            for member in archive.infolist():
                target = (cache_dir / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError("unsafe Task 3d archive member")
            archive.extractall(cache_dir)
        search_root = cache_dir
    elif source.is_dir():
        search_root = source
    else:
        raise FileNotFoundError(source)
    matches = [p.parent for p in search_root.rglob("final_report.json") if (p.parent / "development_checkpoints.pt").is_file()]
    if len(matches) != 1: raise RuntimeError(f"expected one Task 3d artifact root, found {len(matches)}")
    root = matches[0]; report_path = root / "final_report.json"; checkpoint_path = root / "development_checkpoints.pt"
    if _file_sha(report_path) != EXPECTED_TASK3D_REPORT_SHA256: raise RuntimeError("Task 3d report SHA-256 mismatch")
    if _file_sha(checkpoint_path) != EXPECTED_TASK3D_CHECKPOINT_SHA256: raise RuntimeError("Task 3d checkpoint SHA-256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "giada-task3d-generalization-diagnosis-v1" or report.get("fresh_task3c_accessed"):
        raise RuntimeError("invalid or fresh-contaminated Task 3d artifact")
    return root, report


def freeze_full_repair_from_task3d(development_bundle, output_dir, task3d_source, config=None, *, code_revision="unknown"):
    config = config or JointGateFullRepairConfirmationConfig(); config.validate()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    root, source_report = verified_task3d_root(task3d_source, output_dir.parent / ".task3d_verified_cache")
    torch = configure_torch_runtime(config.seeds[0]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(root / "development_checkpoints.pt", map_location=device, weights_only=True)
    constructor = _constructors(torch, JointGateCellConfig(matched_width=config.width))["shared_matched"]
    selected = {}; per_seed = {}; errors = []
    expected = source_report["summaries"]["full_repair"]["seed_scores"]
    for seed in config.seeds:
        key = f"full_repair|seed={seed}"
        if key not in saved: raise RuntimeError(f"missing Task 3d checkpoint {key}")
        selected[str(seed)] = saved[key]
        model = constructor().to(device); model.load_state_dict(saved[key])
        metrics = _evaluate_candidate(model, development_bundle["development"], torch, device)
        errors.append(abs(metrics["score"] - float(expected[str(seed)])))
        per_seed[str(seed)] = metrics
    reproduction_error = float(max(errors))
    if reproduction_error > config.development_reproduction_atol:
        raise RuntimeError(f"Task 3d candidate reproduction mismatch: {reproduction_error}")
    checkpoint_path = output_dir / "frozen_full_repair_checkpoints.pt"
    torch.save({"per_seed_state": selected}, checkpoint_path)
    freeze = {
        "schema_version": "giada-task3e-freeze-v1", "code_revision": str(code_revision),
        "config": asdict(config), "checkpoint_origin": "task3d_full_repair_shared_width23_step50000",
        "task3d_report_sha256": EXPECTED_TASK3D_REPORT_SHA256,
        "task3d_checkpoint_sha256": EXPECTED_TASK3D_CHECKPOINT_SHA256,
        "development_reproduction_max_error": reproduction_error, "development": per_seed,
        "checkpoint_sha256": _file_sha(checkpoint_path), "retraining_performed": False,
        "sealed_accessed": False,
    }
    freeze["freeze_sha256"] = _sha(freeze)
    (output_dir / "selection_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    report = {
        "schema_version": "giada-task3e-freeze-report-v1", "valid": True,
        "environment": environment_manifest(torch), "source_verified": True,
        "retraining_performed": False, "development_reproduction_max_error": reproduction_error,
        "sealed_accessed": False, "freeze_sha256": freeze["freeze_sha256"],
    }
    (output_dir / "checkpoint_freeze_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _sealed_rows(formula):
    expanded = ((-134.5, -101.5), (-99.5, -70.5), (-69.5, -35.5), (-34.5, -0.5), (0.5, 39.5), (41.5, 74.5))
    return {
        "sealed_central": _enrich(_random_row(formula, seed=85101, count=4096,
            voltage_ranges=expanded[1:5], dt_values=(0.055, 0.275, 0.675, 0.95), role="sealed", axis="central"), formula),
        "sealed_voltage_tail": _enrich(_random_row(formula, seed=85102, count=2048,
            voltage_ranges=(expanded[0], expanded[-1]), dt_values=(0.055, 0.275, 0.675, 0.95), role="sealed", axis="voltage_tail"), formula),
        "sealed_long_horizon": _enrich(_random_row(formula, seed=85103, count=2048,
            voltage_ranges=expanded, dt_values=(7.5, 37.5, 125.0), role="sealed", axis="long_horizon"), formula),
    }


def _rate_metrics(model, rows, torch, device):
    result = {}
    with torch.inference_mode():
        for name, row in rows.items():
            x = torch.as_tensor(row["inputs"], dtype=torch.float32, device=device)
            _pred, detail = model(x, diagnostics=True)
            result[name] = {
                "m_inf_rmse": float(np.sqrt(np.mean((detail["m_inf"].cpu().double().numpy() - row["m_inf"]) ** 2))),
                "h_inf_rmse": float(np.sqrt(np.mean((detail["h_inf"].cpu().double().numpy() - row["h_inf"]) ** 2))),
                "m_log_tau_rmse": float(np.sqrt(np.mean((np.log(detail["m_tau_ms"].cpu().double().numpy()) - np.log(row["m_tau_ms"])) ** 2))),
                "h_log_tau_rmse": float(np.sqrt(np.mean((np.log(detail["h_tau_ms"].cpu().double().numpy()) - np.log(row["h_tau_ms"])) ** 2))),
            }
    return result


def _shape_metrics(model, torch, device):
    voltage = np.linspace(-140.0, 80.0, 401)
    values = np.stack((voltage, np.full_like(voltage, .5), np.full_like(voltage, .5), np.ones_like(voltage)), -1)
    with torch.inference_mode():
        _pred, detail = model(torch.as_tensor(values, dtype=torch.float32, device=device), diagnostics=True)
    mi = detail["m_inf"].cpu().double().numpy(); hi = detail["h_inf"].cpu().double().numpy()
    return {
        "maximum_m_inf_reversal": float(np.maximum(0.0, -np.diff(mi)).max(initial=0.0)),
        "maximum_h_inf_reversal": float(np.maximum(0.0, np.diff(hi)).max(initial=0.0)),
    }


def evaluate_frozen_full_repair(development_bundle, output_dir, config=None):
    config = config or JointGateFullRepairConfirmationConfig(); config.validate(); output_dir = Path(output_dir)
    marker = output_dir / "sealed_confirmation_opened.json"
    if marker.exists() or (output_dir / "final_report.json").exists(): raise RuntimeError("Task 3e sealed set already opened")
    freeze = json.loads((output_dir / "selection_freeze.json").read_text(encoding="utf-8")); claimed = freeze.pop("freeze_sha256")
    if _sha(freeze) != claimed or freeze["sealed_accessed"] or freeze["retraining_performed"]:
        raise RuntimeError("invalid Task 3e freeze")
    checkpoint_path = output_dir / "frozen_full_repair_checkpoints.pt"
    if _file_sha(checkpoint_path) != freeze["checkpoint_sha256"]: raise RuntimeError("Task 3e checkpoint SHA-256 mismatch")
    rows = _sealed_rows(development_bundle["formula"])
    prior = {tuple(x) for row in (*development_bundle["pools"].values(), *development_bundle["development"].values()) for x in row["inputs"]}
    sealed = {tuple(x) for row in rows.values() for x in row["inputs"]}
    overlap = len(prior & sealed)
    if overlap: raise RuntimeError(f"sealed tuples overlap Task 3d data: {overlap}")
    torch = configure_torch_runtime(config.seeds[0]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)["per_seed_state"]
    constructor = _constructors(torch, JointGateCellConfig(matched_width=config.width))["shared_matched"]
    per_seed = {}; seed_scores = []
    for seed in config.seeds:
        model = constructor().to(device); model.load_state_dict(saved[str(seed)])
        endpoint = _evaluate(model, rows.items(), torch, device); rates = _rate_metrics(model, rows, torch, device); shape = _shape_metrics(model, torch, device)
        score = max(max(v["m_rmse"] / .0025, v["h_rmse"] / .001, v["open_rmse"] / .0025) for v in endpoint.values())
        seed_scores.append(score); per_seed[str(seed)] = {"score": score, "endpoint": endpoint, "rates": rates, "shape": shape}
    rate_values = [metric for seed in per_seed.values() for row in seed["rates"].values() for metric in row.values()]
    m_inf = [row["m_inf_rmse"] for seed in per_seed.values() for row in seed["rates"].values()]
    h_inf = [row["h_inf_rmse"] for seed in per_seed.values() for row in seed["rates"].values()]
    log_tau = [value for seed in per_seed.values() for row in seed["rates"].values() for key, value in row.items() if "log_tau" in key]
    reversals = [value for seed in per_seed.values() for value in seed["shape"].values()]
    violations = int(sum(row["occupancy_violation_count"] for seed in per_seed.values() for row in seed["endpoint"].values()))
    decision = {
        "sealed_mean_score": float(np.mean(seed_scores)), "sealed_max_seed_score": float(np.max(seed_scores)),
        "maximum_m_inf_rmse": float(max(m_inf)), "maximum_h_inf_rmse": float(max(h_inf)),
        "maximum_log_tau_rmse": float(max(log_tau)), "maximum_monotonic_reversal": float(max(reversals)),
        "occupancy_violations": violations,
    }
    finite = bool(np.isfinite(seed_scores + rate_values + reversals).all())
    decision["task3e_passed"] = bool(
        finite and decision["sealed_mean_score"] <= config.sealed_mean_score_max
        and decision["sealed_max_seed_score"] <= config.sealed_max_seed_score_max
        and decision["maximum_m_inf_rmse"] <= config.m_inf_rmse_max
        and decision["maximum_h_inf_rmse"] <= config.h_inf_rmse_max
        and decision["maximum_log_tau_rmse"] <= config.log_tau_rmse_max
        and decision["maximum_monotonic_reversal"] <= config.maximum_monotonic_reversal
        and violations == 0
    )
    decision["task4_authorized"] = decision["task3e_passed"]
    report = {
        "schema_version": "giada-task3e-final-v1", "valid": finite,
        "selection_used_sealed": False,
        "sealed_contract": {"counts": {name: len(row["inputs"]) for name, row in rows.items()},
                            "overlap_with_task3d": overlap, "opened_once_after_freeze": True},
        "per_seed": per_seed, "decision": decision,
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    marker.write_text(json.dumps({"freeze_sha256": claimed, "final_report_sha256": _file_sha(output_dir / "final_report.json"), "opened_once": True}, indent=2), encoding="utf-8")
    return report

