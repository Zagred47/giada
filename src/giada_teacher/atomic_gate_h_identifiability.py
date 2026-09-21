"""Task 2b: causal identification repair for the slow Ca_HVA h gate."""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_gate_diagnosis import _paired_batch_generator, _reserved_voltage
from .atomic_gate_playground import (
    _evaluate_model,
    _file_sha256,
    _metrics,
    _models,
    _rollout_metrics,
    _sha256,
    _state_dict_cpu,
    materialize_atomic_gate_dataset,
)
from .double_oracle import ExtractedGateFormula
from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest


EXPECTED_TASK2_ARCHIVE_SHA256 = "e8ef3c46f0c1b02e529e627df21eab9d2e4bad7ceebdad26ba14d0ad742a699b"
EXPECTED_TASK2_FREEZE_SHA256 = "e855b5c55605b993af821441c2331968a4e4631861b47ee68aaebc09ba951a74"
EXPECTED_TASK2_FINAL_SHA256 = "44d2ec5e1c7257ad336f743a88da8addb87983ff2de05f98aa8f73f16bc66de5"
EXPECTED_TASK2_CHECKPOINT_SHA256 = "2fc443f28084559a3ff1691e1dcda9b293df79fe9f0466854b5826c18f4e15ce"


@dataclass(frozen=True)
class GateHIdentifiabilityConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    objectives: tuple[str, ...] = (
        "endpoint_only", "rate_supervision", "multi_horizon", "rate_plus_multi_horizon"
    )
    learning_rate: float = 0.003
    batch_size: int = 1024
    hidden_width: int = 16
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000, 30000, 50000)
    training_horizons: tuple[int, ...] = (1, 10, 100)
    selection_horizons: tuple[int, ...] = (1, 10, 100)
    rollout_horizons: tuple[int, ...] = (10, 100, 1000)
    auxiliary_inf_weight: float = 0.1
    auxiliary_log_tau_weight: float = 0.01

    def validate(self) -> None:
        expected = {
            "endpoint_only", "rate_supervision", "multi_horizon", "rate_plus_multi_horizon"
        }
        if set(self.objectives) != expected:
            raise ValueError("Task 2b requires the registered 2x2 objective factorial")
        if self.checkpoints[0] != 0 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("invalid checkpoint ladder")
        if self.training_horizons != (1, 10, 100):
            raise ValueError("registered training horizons changed")


def _safe_extract(source: Path, destination: Path) -> Path:
    with zipfile.ZipFile(source) as archive:
        base = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if base != target and base not in target.parents:
                raise RuntimeError("unsafe Task 2 archive member")
        archive.extractall(destination)
    return destination


def verified_task2_artifact_root(source: str | Path, cache_dir: str | Path) -> Path:
    source, cache_dir = Path(source), Path(cache_dir)
    if source.is_file():
        if _file_sha256(source) != EXPECTED_TASK2_ARCHIVE_SHA256:
            raise RuntimeError("Task 2 archive SHA-256 mismatch")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        root = _safe_extract(source, cache_dir)
    else:
        root = source
    candidates = [p.parent for p in root.rglob("final_report.json") if
                  (p.parent / "selection_freeze.json").is_file() and
                  (p.parent / "selected_checkpoints.pt").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Task 2 artifact root, found {len(candidates)}")
    root = candidates[0]
    final = json.loads((root / "final_report.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "selection_freeze.json").read_text(encoding="utf-8"))
    opened = json.loads((root / "sealed_test_opened.json").read_text(encoding="utf-8"))
    claimed = freeze.pop("freeze_sha256")
    checks = (
        final.get("schema_version") == "giada-task2-final-v1",
        final.get("gate") == "h",
        claimed == EXPECTED_TASK2_FREEZE_SHA256,
        _sha256(freeze) == claimed,
        _file_sha256(root / "final_report.json") == EXPECTED_TASK2_FINAL_SHA256,
        opened.get("final_report_sha256") == EXPECTED_TASK2_FINAL_SHA256,
        _file_sha256(root / "selected_checkpoints.pt") == EXPECTED_TASK2_CHECKPOINT_SHA256,
        freeze.get("selected_checkpoints_sha256") == EXPECTED_TASK2_CHECKPOINT_SHA256,
    )
    if not all(checks):
        raise RuntimeError("Task 2 artifact verification failed")
    return root


def _row(formula: ExtractedGateFormula, inputs: np.ndarray, role: str, axis: str) -> dict[str, Any]:
    rates = [formula.rates(float(voltage)) for voltage in inputs[:, 0]]
    return {
        "role": role, "axis": axis, "inputs": np.asarray(inputs, dtype=np.float64),
        "targets": np.asarray([formula.step("h", x, v, dt) for v, x, dt in inputs]),
        "privileged_inf": np.asarray([r["h_inf"] for r in rates]),
        "privileged_tau_ms": np.asarray([r["h_tau_ms"] for r in rates]),
    }


def prepare_gate_h_identifiability(formula: ExtractedGateFormula) -> dict[str, Any]:
    atomic = materialize_atomic_gate_dataset(formula, gate="h")
    fit = atomic["strata"]["train"]
    development = {n: r for n, r in atomic["strata"].items() if r["role"] == "development"}
    voltages = [
        float(v) for v in np.arange(-97.5, 38.0, 2.5)
        if not _reserved_voltage(float(v))
    ]
    states = (0.025, 0.125, 0.375, 0.625, 0.875, 0.975)
    make = lambda vs, ds: np.asarray([(v, x, d) for v in vs for x in states for d in ds])
    fresh = {
        "fresh_in_support": _row(formula, make(voltages, (0.075, 0.3, 0.8)), "fresh", "in_support"),
        "fresh_ood_voltage": _row(formula, make((-117.5,-112.5,-107.5,47.5,52.5,57.5), (0.075,0.3,0.8)), "fresh", "ood_voltage"),
        "fresh_ood_dt": _row(formula, make((-92.5,-67.5,-42.5,-2.5,32.5), (1.25,)), "fresh", "ood_dt"),
    }
    old = {tuple(x) for r in atomic["strata"].values() for x in r["inputs"]}
    new = {tuple(x) for r in fresh.values() for x in r["inputs"]}
    if old & new:
        raise RuntimeError("Task 2b fresh confirmation overlaps Task 2 rows")
    return {"formula": formula, "fit": fit, "development": development, "fresh": fresh,
            "contract": {"schema_version": "giada-task2b-data-v1",
                         "fit_count": len(fit["inputs"]),
                         "development_count": sum(len(r["inputs"]) for r in development.values()),
                         "fresh_counts": {n: len(r["inputs"]) for n, r in fresh.items()},
                         "task2_rows_reused_for_fresh": False, "fresh_used_for_selection": False}}


def _horizon_targets(formula: ExtractedGateFormula, inputs: np.ndarray, horizons) -> dict[int, np.ndarray]:
    return {k: np.asarray([formula.step("h", x, v, dt * k) for v, x, dt in inputs])
            for k in horizons}


def _predicted_horizon(details, state, dt, horizon):
    inf, tau = details["inf"], details["tau_ms"]
    return inf + (state - inf) * (-horizon * dt / tau).exp()


def _physical_tau_decision(metrics, rollout, rates, winner: str, fresh_names) -> dict[str, Any]:
    """Aggregate the preregistered fresh gates using the canonical metric names."""

    names = tuple(fresh_names)
    winner_macro = float(np.mean([metrics[winner][name]["rmse"] for name in names]))
    roll1000 = float(np.mean([rollout[winner][name]["1000"] for name in names]))
    inf = float(np.mean([rates[winner][name]["inf_rmse"] for name in names]))
    tau = float(np.mean([rates[winner][name]["log_tau_rmse"] for name in names]))
    occupancy_violations = int(sum(
        metrics[winner][name]["occupancy_violation_count"] for name in names
    ))
    finite = bool(np.isfinite([winner_macro, roll1000, inf, tau]).all())
    repaired = bool(
        finite and occupancy_violations == 0 and winner_macro <= 1e-3
        and roll1000 <= 5e-3 and inf <= 1e-2 and tau <= 1e-1
    )
    return {
        "winner": winner,
        "winner_fresh_macro_rmse": winner_macro,
        "winner_rollout_1000_rmse": roll1000,
        "winner_inf_rmse": inf,
        "winner_log_tau_rmse": tau,
        "winner_occupancy_violations": occupancy_violations,
        "winner_metrics_finite": finite,
        "physical_tau_repaired": repaired,
        "task3_authorized": repaired,
    }


def _score(model, rows, formula, horizons, torch, device) -> tuple[float, dict[str, Any]]:
    model.eval(); reports = {}
    with torch.inference_mode():
        for name, row in rows:
            inputs = torch.as_tensor(row["inputs"], dtype=torch.float32, device=device)
            _, details = model(inputs, diagnostics=True)
            per_horizon = {}
            for horizon in horizons:
                prediction = _predicted_horizon(details, inputs[:, 1], inputs[:, 2], horizon)
                target = _horizon_targets(formula, row["inputs"], (horizon,))[horizon]
                per_horizon[str(horizon)] = _metrics(
                    prediction.cpu().double().numpy(), target, conductance_power=1
                )["rmse"]
            reports[name] = per_horizon
    return float(np.mean([v for r in reports.values() for v in r.values()])), reports


def run_gate_h_identifiability(bundle, output_dir, config: GateHIdentifiabilityConfig | None = None):
    config = config or GateHIdentifiabilityConfig(); config.validate()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    torch = configure_torch_runtime(config.seeds[0]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = bundle["fit"]; inputs = torch.as_tensor(fit["inputs"], dtype=torch.float32, device=device)
    endpoint = torch.as_tensor(fit["targets"], dtype=torch.float32, device=device)
    inf_target = torch.as_tensor(fit["privileged_inf"], dtype=torch.float32, device=device)
    tau_target = torch.as_tensor(fit["privileged_tau_ms"], dtype=torch.float32, device=device)
    horizon_targets = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in
                       _horizon_targets(bundle["formula"], fit["inputs"], config.training_horizons).items()}
    total = len(config.objectives) * len(config.seeds) * config.checkpoints[-1]
    done = 0; started = time.perf_counter(); runs=[]; snapshots={}
    constructor = _models(torch, config.hidden_width)["physical_tau"]
    for objective in config.objectives:
        for seed in config.seeds:
            configure_torch_runtime(seed); model=constructor().to(device)
            optimizer=torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
            stream=_paired_batch_generator(len(fit["inputs"]), config.batch_size, seed+300000)
            checkpoints=[]
            for step in range(config.checkpoints[-1]+1):
                if step in config.checkpoints:
                    score, metrics=_score(model, bundle["development"].items(), bundle["formula"], config.selection_horizons, torch, device)
                    key=f"{objective}-seed{seed}-step{step}"; snapshots[key]=_state_dict_cpu(model,torch)
                    checkpoints.append({"step":step,"development_multihorizon_score":score,"metrics":metrics,"checkpoint_key":key})
                    elapsed=time.perf_counter()-started; progressed=done+step
                    eta=elapsed/max(progressed,1)*max(total-progressed,0)
                    print(f"[GIADA Task 2b] {100*progressed/total:.1f}% ETA {eta/60:.1f} min {objective} seed={seed} step={step} score={score:.4g}")
                if step==config.checkpoints[-1]: break
                index=torch.as_tensor(next(stream),dtype=torch.long,device=device); batch=inputs.index_select(0,index)
                optimizer.zero_grad(set_to_none=True); pred,details=model(batch,diagnostics=True)
                loss=torch.mean((pred-endpoint.index_select(0,index))**2)
                if "multi_horizon" in objective:
                    losses=[]
                    for horizon in config.training_horizons:
                        ph=_predicted_horizon(details,batch[:,1],batch[:,2],horizon)
                        losses.append(torch.mean((ph-horizon_targets[horizon].index_select(0,index))**2))
                    loss=torch.stack(losses).mean()
                if "rate" in objective:
                    loss=loss+config.auxiliary_inf_weight*torch.mean((details["inf"]-inf_target.index_select(0,index))**2)
                    loss=loss+config.auxiliary_log_tau_weight*torch.mean((torch.log(details["tau_ms"])-torch.log(tau_target.index_select(0,index)))**2)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            done += config.checkpoints[-1]
            runs.append({"objective":objective,"seed":seed,"checkpoints":checkpoints})
    selections={}
    for objective in config.objectives:
        subset=[r for r in runs if r["objective"]==objective]; candidates=[]
        for step in config.checkpoints:
            rows=[next(c for c in r["checkpoints"] if c["step"]==step) for r in subset]
            candidates.append((float(np.mean([r["development_multihorizon_score"] for r in rows])),step,rows,subset))
        score,step,rows,subset=min(candidates,key=lambda x:(x[0],x[1]))
        selections[objective]={"step":step,"mean_development_multihorizon_score":score,
            "seed_checkpoint_keys":{str(r["seed"]):c["checkpoint_key"] for r,c in zip(subset,rows)}}
    winner=min(selections,key=lambda n:(selections[n]["mean_development_multihorizon_score"],n))
    checkpoint=output_dir/"frozen_checkpoints.pt"
    torch.save({n:{s:snapshots[k] for s,k in row["seed_checkpoint_keys"].items()} for n,row in selections.items()},checkpoint)
    freeze={"schema_version":"giada-task2b-freeze-v1","config":asdict(config),"data_contract":bundle["contract"],
            "selection":selections,"winner":winner,"checkpoint_sha256":_file_sha256(checkpoint),"fresh_accessed":False}
    freeze["freeze_sha256"]=_sha256(freeze); (output_dir/"selection_freeze.json").write_text(json.dumps(freeze,indent=2))
    report={"schema_version":"giada-task2b-training-v1","valid":True,"environment":environment_manifest(torch),
            "runs":runs,"selection":selections,"winner":winner,"fresh_accessed":False}
    (output_dir/"training_report.json").write_text(json.dumps(report,indent=2)); return report


def evaluate_gate_h_identifiability(bundle, output_dir, task2_root, config: GateHIdentifiabilityConfig | None=None):
    config=config or GateHIdentifiabilityConfig(); output_dir=Path(output_dir)
    if (output_dir/"fresh_confirmation_opened.json").exists(): raise RuntimeError("fresh confirmation already opened")
    freeze=json.loads((output_dir/"selection_freeze.json").read_text()); claimed=freeze.pop("freeze_sha256")
    if _sha256(freeze)!=claimed or freeze["fresh_accessed"]: raise RuntimeError("invalid Task 2b freeze")
    import torch
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); constructor=_models(torch,config.hidden_width)["physical_tau"]
    saved=torch.load(output_dir/"frozen_checkpoints.pt",map_location=device,weights_only=True)
    metrics={}; rollout={}; rates={}
    for objective in config.objectives:
        seed_metrics=[]; seed_roll=[]; seed_rates=[]
        for seed in config.seeds:
            model=constructor().to(device); model.load_state_dict(saved[objective][str(seed)])
            seed_metrics.append(_evaluate_model(model,bundle["fresh"].items(),torch,device,gate="h"))
            score, detail=_score(model,bundle["fresh"].items(),bundle["formula"],config.rollout_horizons,torch,device); seed_roll.append(detail)
            rate_rows={}
            with torch.inference_mode():
                for name,row in bundle["fresh"].items():
                    x=torch.as_tensor(row["inputs"],dtype=torch.float32,device=device); _,d=model(x,diagnostics=True)
                    rate_rows[name]={"inf_rmse":float(np.sqrt(np.mean((d["inf"].cpu().numpy()-row["privileged_inf"])**2))),
                        "log_tau_rmse":float(np.sqrt(np.mean((np.log(d["tau_ms"].cpu().numpy())-np.log(row["privileged_tau_ms"]))**2)))}
            seed_rates.append(rate_rows)
        metrics[objective]={n:{k:float(np.mean([s[n][k] for s in seed_metrics])) for k in seed_metrics[0][n]} for n in bundle["fresh"]}
        rollout[objective]={n:{h:float(np.mean([s[n][h] for s in seed_roll])) for h in map(str,config.rollout_horizons)} for n in bundle["fresh"]}
        rates[objective]={n:{k:float(np.mean([s[n][k] for s in seed_rates])) for k in ("inf_rmse","log_tau_rmse")} for n in bundle["fresh"]}
    task2_saved=torch.load(Path(task2_root)/"selected_checkpoints.pt",map_location=device,weights_only=True)
    dz_constructor=_models(torch,16)["direct_z"]; dz_metrics=[]; dz_roll=[]
    for seed in config.seeds:
        model=dz_constructor().to(device); model.load_state_dict(task2_saved["direct_z"][str(seed)])
        dz_metrics.append(_evaluate_model(model,bundle["fresh"].items(),torch,device,gate="h"))
        detail={}
        for name,row in bundle["fresh"].items():
            values=_rollout_metrics(
                model, [(name,row)], bundle["formula"], "h",
                config.rollout_horizons, torch, device,
            )
            detail[name]=values
        dz_roll.append(detail)
    metrics["frozen_task2_direct_z"]={n:{k:float(np.mean([s[n][k] for s in dz_metrics])) for k in dz_metrics[0][n]} for n in bundle["fresh"]}
    rollout["frozen_task2_direct_z"]={n:{h:float(np.mean([s[n][h] for s in dz_roll])) for h in map(str,config.rollout_horizons)} for n in bundle["fresh"]}
    winner=freeze["winner"]
    decision=_physical_tau_decision(metrics,rollout,rates,winner,bundle["fresh"])
    report={"schema_version":"giada-task2b-final-v1","valid":decision["winner_metrics_finite"],"selection_used_fresh":False,
            "metrics":metrics,"rollout":rollout,"rates":rates,"decision":decision}
    (output_dir/"final_report.json").write_text(json.dumps(report,indent=2)); (output_dir/"registered_decision.json").write_text(json.dumps(decision,indent=2))
    (output_dir/"fresh_confirmation_opened.json").write_text(json.dumps({"freeze_sha256":claimed,"final_report_sha256":_file_sha256(output_dir/"final_report.json"),"opened_once":True},indent=2))
    return report
