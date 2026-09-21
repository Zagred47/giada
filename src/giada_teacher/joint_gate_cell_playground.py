"""Task 3: paired joint Ca_HVA m+h physical-tau cell playground."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_gate_playground import _metrics, _state_dict_cpu
from .domain_splits import AtomicDomainSplitConfig, build_atomic_domain_splits
from .double_oracle import ExtractedGateFormula
from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest, paired_index_generator


@dataclass(frozen=True)
class JointGateCellConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    families: tuple[str, ...] = ("independent", "shared_compact", "shared_matched")
    learning_rate: float = 0.003
    batch_size: int = 1024
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000, 30000, 50000)
    private_width: int = 16
    compact_width: int = 16
    matched_width: int = 23
    h_inf_weight: float = 0.1
    h_log_tau_weight: float = 0.01
    rollout_horizons: tuple[int, ...] = (10, 100, 1000)

    def validate(self) -> None:
        if set(self.families) != {"independent", "shared_compact", "shared_matched"}:
            raise ValueError("Task 3 requires the registered three-family contrast")
        if self.checkpoints[0] != 0 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted, unique and start at zero")
        if min(self.seeds) < 0 or min(self.learning_rate, self.batch_size, self.private_width) <= 0:
            raise ValueError("invalid Task 3 optimization configuration")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _joint_row(formula: ExtractedGateFormula, values: np.ndarray, role: str, axis: str) -> dict[str, Any]:
    rates = [formula.rates(float(v)) for v in values[:, 0]]
    m_target = np.asarray([formula.step("m", m, v, dt) for v, m, h, dt in values])
    h_target = np.asarray([formula.step("h", h, v, dt) for v, m, h, dt in values])
    return {
        "role": role, "axis": axis, "inputs": values,
        "m_target": m_target, "h_target": h_target,
        "open_target": m_target ** 2 * h_target,
        "h_inf": np.asarray([r["h_inf"] for r in rates]),
        "h_tau_ms": np.asarray([r["h_tau_ms"] for r in rates]),
    }


def _materialize_split(formula: ExtractedGateFormula, split: dict[str, Any]) -> dict[str, Any]:
    rows = np.asarray([
        (float(v), float(m), float(h), float(dt))
        for v in split["voltages_mv"] for m in split["states"]
        for h in split["states"] for dt in split["dt_ms"]
    ], dtype=np.float64)
    return _joint_row(formula, rows, split["role"], split["varied_axis"])


def prepare_joint_gate_dataset(formula: ExtractedGateFormula) -> dict[str, Any]:
    split_report = build_atomic_domain_splits(AtomicDomainSplitConfig())
    strata = {name: _materialize_split(formula, row) for name, row in split_report["strata"].items()}
    fit = strata["train"]
    development = {n: r for n, r in strata.items() if r["role"] == "development"}
    states_m = (0.05, 0.25, 0.55, 0.85, 0.95)
    states_h = (0.02, 0.18, 0.48, 0.78, 0.98)
    make = lambda vs, ds: np.asarray([(v, m, h, d) for v in vs for m in states_m for h in states_h for d in ds])
    support_v = tuple(float(v) for v in np.arange(-96.25, 37.0, 3.0))
    fresh = {
        "fresh_in_support": _joint_row(formula, make(support_v, (0.125, 0.625)), "fresh", "in_support"),
        "fresh_ood_voltage": _joint_row(formula, make((-116.25, -108.25, 48.25, 56.25), (0.125, 0.625)), "fresh", "ood_voltage"),
        "fresh_ood_dt": _joint_row(formula, make((-91.25, -61.25, -31.25, 28.75), (1.375,)), "fresh", "ood_dt"),
    }
    old = {tuple(x) for row in strata.values() for x in row["inputs"]}
    new = {tuple(x) for row in fresh.values() for x in row["inputs"]}
    if old & new:
        raise RuntimeError("Task 3 fresh joint rows overlap the atomic split rows")
    return {
        "formula": formula, "fit": fit, "development": development, "fresh": fresh,
        "contract": {
            "schema_version": "giada-task3-joint-data-v1",
            "fit_count": len(fit["inputs"]),
            "development_count": sum(len(r["inputs"]) for r in development.values()),
            "fresh_counts": {n: len(r["inputs"]) for n, r in fresh.items()},
            "fresh_overlap": 0, "fresh_used_for_selection": False,
        },
    }


def _constructors(torch, config: JointGateCellConfig):
    nn = torch.nn

    class Heads(nn.Module):
        def __init__(self, width):
            super().__init__(); self.inf = nn.Linear(width, 1); self.tau = nn.Linear(width, 1)
        def forward(self, features):
            return torch.sigmoid(self.inf(features)).squeeze(-1), torch.nn.functional.softplus(self.tau(features)).squeeze(-1) + 1e-5

    def trunk(width):
        return nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())

    class JointBase(nn.Module):
        def solve(self, values, m_rate, h_rate, diagnostics=False):
            v, m, h, dt = values.unbind(-1); mi, mt = m_rate; hi, ht = h_rate
            mz = -torch.expm1(-dt / mt); hz = -torch.expm1(-dt / ht)
            mp = (1-mz)*m + mz*mi; hp = (1-hz)*h + hz*hi
            pred = torch.stack((mp, hp), -1)
            details = {"m_inf": mi, "m_tau_ms": mt, "h_inf": hi, "h_tau_ms": ht, "open": mp**2*hp}
            return (pred, details) if diagnostics else pred

    class Independent(JointBase):
        def __init__(self):
            super().__init__(); w=config.private_width; self.mt=trunk(w); self.ht=trunk(w); self.mh=Heads(w); self.hh=Heads(w)
        def forward(self, values, diagnostics=False):
            x=((values[:,0]+30.0)/90.0).unsqueeze(-1)
            return self.solve(values, self.mh(self.mt(x)), self.hh(self.ht(x)), diagnostics)

    def shared_class(width):
        class Shared(JointBase):
            def __init__(self):
                super().__init__(); self.trunk=trunk(width); self.mh=Heads(width); self.hh=Heads(width)
            def forward(self, values, diagnostics=False):
                x=((values[:,0]+30.0)/90.0).unsqueeze(-1); features=self.trunk(x)
                return self.solve(values, self.mh(features), self.hh(features), diagnostics)
        return Shared

    return {"independent": Independent, "shared_compact": shared_class(config.compact_width), "shared_matched": shared_class(config.matched_width)}


def _evaluate(model, rows, torch, device) -> dict[str, Any]:
    result = {}; model.eval()
    with torch.inference_mode():
        for name, row in rows:
            x=torch.as_tensor(row["inputs"],dtype=torch.float32,device=device); pred,detail=model(x,diagnostics=True)
            p=pred.cpu().double().numpy(); op=detail["open"].cpu().double().numpy()
            result[name]={
                "m_rmse":_metrics(p[:,0],row["m_target"],conductance_power=1)["rmse"],
                "h_rmse":_metrics(p[:,1],row["h_target"],conductance_power=1)["rmse"],
                "open_rmse":float(np.sqrt(np.mean((op-row["open_target"])**2))),
                "occupancy_violation_count":int(np.count_nonzero((p<0)|(p>1))),
            }
    return result


def _development_score(metrics) -> float:
    m=float(np.mean([r["m_rmse"] for r in metrics.values()])); h=float(np.mean([r["h_rmse"] for r in metrics.values()])); o=float(np.mean([r["open_rmse"] for r in metrics.values()]))
    return max(m/0.0025, h/0.001, o/0.0025)


def run_joint_gate_playground(bundle, output_dir, config: JointGateCellConfig | None = None):
    config=config or JointGateCellConfig(); config.validate(); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    torch=configure_torch_runtime(config.seeds[0]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit=bundle["fit"]; x=torch.as_tensor(fit["inputs"],dtype=torch.float32,device=device)
    targets=torch.as_tensor(np.stack((fit["m_target"],fit["h_target"]),-1),dtype=torch.float32,device=device)
    hi=torch.as_tensor(fit["h_inf"],dtype=torch.float32,device=device); ht=torch.as_tensor(fit["h_tau_ms"],dtype=torch.float32,device=device)
    constructors=_constructors(torch,config); snapshots={}; runs=[]; total=len(config.families)*len(config.seeds)*config.checkpoints[-1]; done=0; started=time.perf_counter()
    for family in config.families:
        for seed in config.seeds:
            configure_torch_runtime(seed); model=constructors[family]().to(device); opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=1e-6)
            stream=paired_index_generator(len(fit["inputs"]),config.batch_size,seed+400000); checkpoints=[]
            for step in range(config.checkpoints[-1]+1):
                if step in config.checkpoints:
                    metrics=_evaluate(model,bundle["development"].items(),torch,device); score=_development_score(metrics); key=f"{family}-seed{seed}-step{step}"; snapshots[key]=_state_dict_cpu(model,torch)
                    checkpoints.append({"step":step,"score":score,"metrics":metrics,"checkpoint_key":key}); elapsed=time.perf_counter()-started; progress=done+step; eta=elapsed/max(progress,1)*max(total-progress,0)
                    print(f"[GIADA Task 3] {100*progress/total:.1f}% ETA {eta/60:.1f} min {family} seed={seed} step={step} score={score:.4g}")
                if step==config.checkpoints[-1]: break
                index=torch.as_tensor(next(stream),dtype=torch.long,device=device); batch=x.index_select(0,index); target=targets.index_select(0,index)
                opt.zero_grad(set_to_none=True); pred,details=model(batch,diagnostics=True)
                loss=torch.mean((pred[:,0]-target[:,0])**2)+torch.mean((pred[:,1]-target[:,1])**2)
                loss=loss+config.h_inf_weight*torch.mean((details["h_inf"]-hi.index_select(0,index))**2)
                loss=loss+config.h_log_tau_weight*torch.mean((torch.log(details["h_tau_ms"])-torch.log(ht.index_select(0,index)))**2)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            done+=config.checkpoints[-1]; runs.append({"family":family,"seed":seed,"parameter_count":sum(p.numel() for p in model.parameters()),"checkpoints":checkpoints})
    selections={}
    for family in config.families:
        subset=[r for r in runs if r["family"]==family]; candidates=[]
        for step in config.checkpoints:
            rows=[next(c for c in r["checkpoints"] if c["step"]==step) for r in subset]; candidates.append((float(np.mean([r["score"] for r in rows])),step,rows))
        score,step,rows=min(candidates,key=lambda q:(q[0],q[1])); selections[family]={"step":step,"score":score,"seed_checkpoint_keys":{str(r["seed"]):c["checkpoint_key"] for r,c in zip(subset,rows)}}
    shared=min(("shared_compact","shared_matched"),key=lambda n:(selections[n]["score"],n)); baseline=selections["independent"]["score"]
    winner=shared if selections[shared]["score"]<=1.10*baseline else "independent"
    checkpoint=output_dir/"frozen_checkpoints.pt"; torch.save({n:{s:snapshots[k] for s,k in row["seed_checkpoint_keys"].items()} for n,row in selections.items()},checkpoint)
    freeze={"schema_version":"giada-task3-freeze-v1","config":asdict(config),"data_contract":bundle["contract"],"selection":selections,"best_shared":shared,"winner":winner,"shared_within_ten_percent":selections[shared]["score"]<=1.10*baseline,"checkpoint_sha256":_file_sha(checkpoint),"fresh_accessed":False}; freeze["freeze_sha256"]=_sha(freeze)
    (output_dir/"selection_freeze.json").write_text(json.dumps(freeze,indent=2)); report={"schema_version":"giada-task3-training-v1","valid":True,"environment":environment_manifest(torch),"runs":runs,"selection":selections,"winner":winner,"fresh_accessed":False}; (output_dir/"training_report.json").write_text(json.dumps(report,indent=2)); return report


def _rollout(model,row,horizon,torch,device):
    x=torch.as_tensor(row["inputs"],dtype=torch.float32,device=device); v,m0,h0,dt=x.unbind(-1); state=x.clone()
    with torch.inference_mode():
        for _ in range(horizon):
            pred=model(state); state=torch.stack((v,pred[:,0],pred[:,1],dt),-1)
    formula=row["formula"]; mt=np.asarray([formula.step("m",m,vv,d*horizon) for vv,m,h,d in row["inputs"]]); ht=np.asarray([formula.step("h",h,vv,d*horizon) for vv,m,h,d in row["inputs"]]); p=pred.cpu().double().numpy()
    return {"m_rmse":float(np.sqrt(np.mean((p[:,0]-mt)**2))),"h_rmse":float(np.sqrt(np.mean((p[:,1]-ht)**2))),"open_rmse":float(np.sqrt(np.mean((p[:,0]**2*p[:,1]-mt**2*ht)**2)))}


def evaluate_joint_gate_playground(bundle, output_dir, config: JointGateCellConfig | None = None):
    config=config or JointGateCellConfig(); output_dir=Path(output_dir); marker=output_dir/"fresh_confirmation_opened.json"
    if marker.exists(): raise RuntimeError("Task 3 fresh confirmation already opened")
    freeze=json.loads((output_dir/"selection_freeze.json").read_text()); claimed=freeze.pop("freeze_sha256")
    if _sha(freeze)!=claimed or freeze["fresh_accessed"]: raise RuntimeError("invalid Task 3 freeze")
    torch=configure_torch_runtime(config.seeds[0]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); constructors=_constructors(torch,config); saved=torch.load(output_dir/"frozen_checkpoints.pt",map_location=device,weights_only=True)
    metrics={}; rollout={}
    for family in config.families:
        mm=[]; rr=[]
        for seed in config.seeds:
            model=constructors[family]().to(device); model.load_state_dict(saved[family][str(seed)]); mm.append(_evaluate(model,bundle["fresh"].items(),torch,device)); rows=[]
            for name,row in bundle["fresh"].items():
                enriched=dict(row); enriched["formula"]=bundle["formula"]; rows.append((name,{str(h):_rollout(model,enriched,h,torch,device) for h in config.rollout_horizons}))
            rr.append(dict(rows))
        metrics[family]={n:{k:float(np.mean([s[n][k] for s in mm])) for k in mm[0][n]} for n in bundle["fresh"]}
        rollout[family]={n:{str(h):{k:float(np.mean([s[n][str(h)][k] for s in rr])) for k in ("m_rmse","h_rmse","open_rmse")} for h in config.rollout_horizons} for n in bundle["fresh"]}
    chosen=freeze["best_shared"]; names=tuple(bundle["fresh"]); mean=lambda key:float(np.mean([metrics[chosen][n][key] for n in names])); r1000=lambda key:float(np.mean([rollout[chosen][n]["1000"][key] for n in names])); violations=int(sum(metrics[chosen][n]["occupancy_violation_count"] for n in names))
    decision={"selected_shared_family":chosen,"fresh_m_rmse":mean("m_rmse"),"fresh_h_rmse":mean("h_rmse"),"fresh_open_rmse":mean("open_rmse"),"rollout1000_m_rmse":r1000("m_rmse"),"rollout1000_h_rmse":r1000("h_rmse"),"rollout1000_open_rmse":r1000("open_rmse"),"occupancy_violations":violations}
    finite=bool(np.isfinite(list(decision.values())[1:]).all()); decision["task3_passed"]=bool(finite and violations==0 and decision["fresh_m_rmse"]<=0.0025 and decision["fresh_h_rmse"]<=0.001 and decision["fresh_open_rmse"]<=0.0025 and decision["rollout1000_m_rmse"]<=0.0025 and decision["rollout1000_h_rmse"]<=0.005 and freeze["shared_within_ten_percent"]); decision["task4_authorized"]=decision["task3_passed"]
    report={"schema_version":"giada-task3-final-v1","valid":finite,"selection_used_fresh":False,"metrics":metrics,"rollout":rollout,"decision":decision}; (output_dir/"final_report.json").write_text(json.dumps(report,indent=2)); (output_dir/"registered_decision.json").write_text(json.dumps(decision,indent=2)); marker.write_text(json.dumps({"freeze_sha256":claimed,"final_report_sha256":_file_sha(output_dir/"final_report.json"),"opened_once":True},indent=2)); return report
