"""Task 4: paired primitive matrix for the isolated Ca_HVA m+h cell.

The experiment compares numerical and learned primitives on identical held-
voltage tuples.  Learned families and seeds are evaluated on paired streams;
the four learned families execute concurrently on one GPU at every step.
Task 3e sealed tuples are never read or reconstructed.
"""

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

from .gpu_baseline_runtime import configure_torch_runtime, environment_manifest, paired_index_generator
from .joint_gate_cell_playground import _development_score, _joint_row
from .joint_gate_generalization_diagnosis import _enrich, _random_row

EXPECTED_TASK3E_ARCHIVE_SHA256 = "e10521f15fa2b543b4ccb543967f08d3fd3795542616d619a0672d08cdea17cd"
EXPECTED_TASK3E_FINAL_SHA256 = "639dcd2cbc1302b493cf017bb2f41f082c5e7a352caacd95666e5c61fa2aab79"


@dataclass(frozen=True)
class PrimitiveMatrixConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    learned_families: tuple[str, ...] = ("direct_mlp", "gru", "physical_tau", "direct_z")
    numerical_families: tuple[str, ...] = ("formula", "lut_nearest", "lut_linear", "chebyshev")
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000, 30000, 50000)
    batch_size: int = 768
    pool_size: int = 24000
    learning_rate: float = 0.003
    weight_decay: float = 1e-6
    direct_width: int = 32
    recurrent_width: int = 24
    physical_width: int = 23
    rate_inf_weight: float = 0.1
    rate_log_tau_weight: float = 0.01
    shape_weight: float = 0.02
    gradient_clip_norm: float = 1.0
    lut_points: int = 257
    chebyshev_degree: int = 16
    rollout_horizons: tuple[int, ...] = (10, 100, 1000)
    latency_batch_sizes: tuple[int, ...] = (1, 1024, 65536)
    compile_models: bool = True

    def validate(self) -> None:
        if set(self.learned_families) != {"direct_mlp", "gru", "physical_tau", "direct_z"}:
            raise ValueError("Task 4 requires four registered learned primitives")
        if set(self.numerical_families) != {"formula", "lut_nearest", "lut_linear", "chebyshev"}:
            raise ValueError("Task 4 requires four registered numerical primitives")
        if self.checkpoints[0] != 0 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted, unique and start at zero")
        if self.checkpoints[-1] != 50000 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Task 4 requires three seeds through 50k")
        if min(self.pool_size, self.batch_size, self.lut_points, self.chebyshev_degree) <= 0:
            raise ValueError("invalid Task 4 configuration")


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


def verified_task3e_root(source: Path, cache_dir: Path) -> Path:
    source=Path(source); cache_dir=Path(cache_dir)
    if source.is_file():
        if _file_sha(source)!=EXPECTED_TASK3E_ARCHIVE_SHA256: raise RuntimeError("Task 3e archive SHA-256 mismatch")
        cache_dir.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target=(cache_dir/member.filename).resolve()
                if cache_dir.resolve() not in target.parents and target!=cache_dir.resolve(): raise RuntimeError("unsafe Task 3e archive member")
            archive.extractall(cache_dir)
        candidates=list(cache_dir.rglob("final_report.json"))
        if len(candidates)!=1: raise RuntimeError("Task 3e final report is ambiguous")
        root=candidates[0].parent
    else: root=source
    report_path=root/"final_report.json"
    if not report_path.is_file() or _file_sha(report_path)!=EXPECTED_TASK3E_FINAL_SHA256: raise RuntimeError("Task 3e final report SHA-256 mismatch")
    report=json.loads(report_path.read_text())
    if not report.get("valid") or not report.get("decision",{}).get("task4_authorized"): raise RuntimeError("Task 3e does not authorize Task 4")
    return root


def prepare_primitive_matrix(formula, config: PrimitiveMatrixConfig | None = None) -> dict[str, Any]:
    config = config or PrimitiveMatrixConfig(); config.validate()
    expanded = ((-135.0, -100.0), (-100.0, -70.0), (-70.0, -35.0), (-35.0, 0.0), (0.0, 40.0), (40.0, 75.0))
    fit = _enrich(_random_row(formula, seed=74101, count=config.pool_size, voltage_ranges=expanded,
        dt_values=(0.025, 0.1, 0.5, 1.0, 5.0, 25.0, 100.0), role="fit", axis="expanded_multihorizon"), formula)
    development = {
        "central": _enrich(_random_row(formula, seed=74201, count=4096,
            voltage_ranges=((-100.0, 40.0),), dt_values=(0.075, 0.35, 0.9), role="development", axis="central"), formula),
        "voltage_tail": _enrich(_random_row(formula, seed=74202, count=2048,
            voltage_ranges=((-132.5, -102.5), (42.5, 72.5)), dt_values=(0.075, 0.35, 0.9), role="development", axis="voltage_tail"), formula),
        "long_horizon": _enrich(_random_row(formula, seed=74203, count=2048,
            voltage_ranges=expanded, dt_values=(10.0, 50.0, 100.0), role="development", axis="long_horizon"), formula),
    }
    # Materialized only by evaluate_frozen_primitive_matrix after a valid freeze.
    sealed_spec = {
        "central": {"seed": 74301, "count": 4096, "voltage_ranges": ((-100.0, 40.0),), "dt_values": (0.055, 0.275, 0.725)},
        "voltage_tail": {"seed": 74302, "count": 2048, "voltage_ranges": ((-134.0, -101.0), (41.0, 74.0)), "dt_values": (0.055, 0.275, 0.725)},
        "long_horizon": {"seed": 74303, "count": 2048, "voltage_ranges": expanded, "dt_values": (8.0, 40.0, 80.0)},
    }
    return {"formula": formula, "fit": fit, "development": development, "sealed_spec": sealed_spec,
        "contract": {"schema_version": "giada-task4-data-v1", "fit_count": len(fit["inputs"]),
            "development_counts": {k: len(v["inputs"]) for k, v in development.items()},
            "task3e_sealed_accessed": False, "paired_streams": True, "all_targets_from_extracted_formula": True}}


def _init_parameter(torch, shape, scale=0.08):
    return torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32) * scale)


class _BatchedPrimitiveBase:
    pass


def _learned_models(torch, config: PrimitiveMatrixConfig):
    nn = torch.nn; S = len(config.seeds)

    class Dense(nn.Module):
        def __init__(self, in_dim, width, out_dim):
            super().__init__(); self.w1=_init_parameter(torch,(S,width,in_dim)); self.b1=nn.Parameter(torch.zeros(S,width)); self.w2=_init_parameter(torch,(S,width,width)); self.b2=nn.Parameter(torch.zeros(S,width)); self.wo=_init_parameter(torch,(S,out_dim,width)); self.bo=nn.Parameter(torch.zeros(S,out_dim))
        def forward(self,x):
            f=torch.nn.functional.silu(torch.einsum('sbi,soi->sbo',x,self.w1)+self.b1[:,None]); f=torch.nn.functional.silu(torch.einsum('sbi,soi->sbo',f,self.w2)+self.b2[:,None]); return torch.einsum('sbi,soi->sbo',f,self.wo)+self.bo[:,None]

    class DirectMLP(nn.Module):
        def __init__(self): super().__init__(); self.net=Dense(4,config.direct_width,2)
        def forward(self,x):
            v,m,h,dt=x.unbind(-1); z=torch.stack(((v+30)/90,m,h,torch.log(dt.clamp_min(1e-5))),-1); p=torch.sigmoid(self.net(z)); return p,{"open":p[...,0].square()*p[...,1]}

    class PhysicalTau(nn.Module):
        def __init__(self): super().__init__(); self.net=Dense(1,config.physical_width,4)
        def forward(self,x):
            v,m,h,dt=x.unbind(-1); raw=self.net(((v+30)/90).unsqueeze(-1)); inf=torch.sigmoid(raw[...,:2]); tau=torch.nn.functional.softplus(raw[...,2:])+1e-5; z=-torch.expm1(-dt.unsqueeze(-1)/tau); p=(1-z)*torch.stack((m,h),-1)+z*inf
            return p,{"m_inf":inf[...,0],"h_inf":inf[...,1],"m_tau_ms":tau[...,0],"h_tau_ms":tau[...,1],"open":p[...,0].square()*p[...,1]}

    class DirectZ(nn.Module):
        def __init__(self): super().__init__(); self.net=Dense(2,config.physical_width,4)
        def forward(self,x):
            v,m,h,dt=x.unbind(-1); raw=self.net(torch.stack(((v+30)/90,torch.log(dt.clamp_min(1e-5))),-1)); inf=torch.sigmoid(raw[...,:2]); z=torch.sigmoid(raw[...,2:]); p=(1-z)*torch.stack((m,h),-1)+z*inf
            return p,{"m_inf":inf[...,0],"h_inf":inf[...,1],"open":p[...,0].square()*p[...,1]}

    class GRU(nn.Module):
        def __init__(self):
            super().__init__(); W=config.recurrent_width; self.enc=Dense(2,W,W); self.wx=_init_parameter(torch,(S,3*W,2)); self.wh=_init_parameter(torch,(S,3*W,W)); self.b=nn.Parameter(torch.zeros(S,3*W)); self.dec_w=_init_parameter(torch,(S,2,W)); self.dec_b=nn.Parameter(torch.zeros(S,2))
        def forward(self,x):
            v,m,h,dt=x.unbind(-1); state=self.enc(torch.stack((m,h),-1)); inp=torch.stack(((v+30)/90,torch.log(dt.clamp_min(1e-5))),-1); gx=torch.einsum('sbi,soi->sbo',inp,self.wx); gh=torch.einsum('sbi,soi->sbo',state,self.wh); ix,ir,inn=gx.chunk(3,-1); hx,hr,hn=gh.chunk(3,-1); bz,br,bn=self.b.chunk(3,-1); z=torch.sigmoid(ix+hx+bz[:,None]); r=torch.sigmoid(ir+hr+br[:,None]); n=torch.tanh(inn+r*hn+bn[:,None]); hidden=(1-z)*n+z*state; p=torch.sigmoid(torch.einsum('sbi,soi->sbo',hidden,self.dec_w)+self.dec_b[:,None]); return p,{"open":p[...,0].square()*p[...,1]}

    return {"direct_mlp":DirectMLP(),"gru":GRU(),"physical_tau":PhysicalTau(),"direct_z":DirectZ()}


def _initialize_seed_slices(models,torch,seeds):
    """Make the leading ensemble axis correspond exactly to registered seeds."""
    with torch.no_grad():
        for family_index,(family,model) in enumerate(models.items()):
            for parameter_index,parameter in enumerate(model.parameters()):
                if parameter.shape[0]!=len(seeds): raise RuntimeError(f"{family} parameter lacks seed axis")
                if bool((parameter==0).all()): continue
                for seed_index,seed in enumerate(seeds):
                    generator=torch.Generator(device=parameter.device); generator.manual_seed(int(seed)+1009*family_index+37*parameter_index)
                    parameter[seed_index].normal_(mean=0.0,std=0.08,generator=generator)


def _tensor_row(row, torch, device):
    return {name: torch.as_tensor(row[name],dtype=torch.float32,device=device) for name in ("inputs","m_target","h_target","m_inf","m_tau_ms","h_inf","h_tau_ms")}


def _evaluate_learned(model,row,torch,device):
    x=torch.as_tensor(row["inputs"],dtype=torch.float32,device=device).unsqueeze(0).expand(model.seed_count,-1,-1) if hasattr(model,"seed_count") else None
    if x is None: x=torch.as_tensor(row["inputs"],dtype=torch.float32,device=device).unsqueeze(0).expand(len(next(model.parameters())),-1,-1)
    with torch.inference_mode(): pred,detail=model(x)
    p=pred.cpu().double().numpy(); target=np.stack((row["m_target"],row["h_target"]),-1); out=[]
    for s in range(p.shape[0]):
        out.append({"m_rmse":float(np.sqrt(np.mean((p[s,:,0]-target[:,0])**2))),"h_rmse":float(np.sqrt(np.mean((p[s,:,1]-target[:,1])**2))),"open_rmse":float(np.sqrt(np.mean((p[s,:,0]**2*p[s,:,1]-row["open_target"])**2))),"occupancy_violation_count":int(np.count_nonzero((p[s]<0)|(p[s]>1)))})
    return out


def _seed_count(model):
    return int(next(model.parameters()).shape[0])


def _learned_eval(model,rows,torch,device):
    S=_seed_count(model); result=[{} for _ in range(S)]
    with torch.inference_mode():
        for name,row in rows.items():
            x=torch.as_tensor(row["inputs"],dtype=torch.float32,device=device).unsqueeze(0).expand(S,-1,-1); pred,_=model(x); p=pred.cpu().double().numpy()
            for s in range(S): result[s][name]={"m_rmse":float(np.sqrt(np.mean((p[s,:,0]-row["m_target"])**2))),"h_rmse":float(np.sqrt(np.mean((p[s,:,1]-row["h_target"])**2))),"open_rmse":float(np.sqrt(np.mean((p[s,:,0]**2*p[s,:,1]-row["open_target"])**2))),"occupancy_violation_count":int(np.count_nonzero((p[s]<0)|(p[s]>1)))}
    return result


def _score(metrics):
    return max(_development_score({name:row}) for name,row in metrics.items())


def _clip_models(models,torch,limit):
    for model in models.values(): torch.nn.utils.clip_grad_norm_(model.parameters(),limit)


def train_and_freeze_primitive_matrix(bundle,output_dir,config:PrimitiveMatrixConfig|None=None,code_revision="unknown"):
    config=config or PrimitiveMatrixConfig(); config.validate(); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    torch=configure_torch_runtime(config.seeds[0]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); models={k:v.to(device) for k,v in _learned_models(torch,config).items()}; S=len(config.seeds); _initialize_seed_slices(models,torch,config.seeds)
    for model in models.values(): model.seed_count=S
    training=models
    if config.compile_models:
        compiled={}
        for name,model in models.items():
            try: compiled[name]=torch.compile(model,dynamic=False)
            except Exception: compiled[name]=model
        training=compiled
    optimizer=torch.optim.AdamW([p for m in models.values() for p in m.parameters()],lr=config.learning_rate,weight_decay=config.weight_decay)
    fit=_tensor_row(bundle["fit"],torch,device); streams=[paired_index_generator(len(bundle["fit"]["inputs"]),config.batch_size,840000+s) for s in config.seeds]
    probe_v=torch.linspace(-140,80,96,device=device); checkpoints={name:[] for name in models}; states={}; started=time.perf_counter(); maximum=config.checkpoints[-1]
    for step in range(maximum+1):
        if step in config.checkpoints:
            compact=[]
            for name,model in models.items():
                per_seed=_learned_eval(model,bundle["development"],torch,device); scores=[_score(x) for x in per_seed]; checkpoints[name].append({"step":step,"mean_score":float(np.mean(scores)),"max_seed_score":float(np.max(scores)),"per_seed_scores":scores}); states[f"{name}:{step}"]={k:v.detach().cpu() for k,v in model.state_dict().items()}; compact.append(f"{name}={np.mean(scores):.3g}")
            elapsed=time.perf_counter()-started; eta=elapsed/max(step,1)*(maximum-step); print(f"[GIADA Task 4] {100*step/maximum:.1f}% ETA {eta/60:.1f} min "+" ".join(compact))
        if step==maximum: break
        indices=torch.stack([torch.as_tensor(next(stream),dtype=torch.long,device=device) for stream in streams]); batch={k:v.index_select(0,indices.reshape(-1)).reshape(S,config.batch_size,*v.shape[1:]) for k,v in fit.items()}
        target=torch.stack((batch["m_target"],batch["h_target"]),-1); optimizer.zero_grad(set_to_none=True); total=0.0
        for name,model in training.items():
            pred,detail=model(batch["inputs"]); loss=((pred-target)**2).mean()+((detail["open"]-target[...,0].square()*target[...,1])**2).mean()
            if name=="physical_tau":
                loss=loss+config.rate_inf_weight*(((detail["m_inf"]-batch["m_inf"])**2).mean()+((detail["h_inf"]-batch["h_inf"])**2).mean())
                loss=loss+config.rate_log_tau_weight*(((torch.log(detail["m_tau_ms"])-torch.log(batch["m_tau_ms"]))**2).mean()+((torch.log(detail["h_tau_ms"])-torch.log(batch["h_tau_ms"]))**2).mean())
                shape_x=torch.stack((probe_v,torch.full_like(probe_v,.5),torch.full_like(probe_v,.5),torch.ones_like(probe_v)),-1).unsqueeze(0).expand(S,-1,-1); _,rates=model(shape_x); loss=loss+config.shape_weight*(torch.relu(rates["m_inf"][:,:-1]-rates["m_inf"][:,1:]).mean()+torch.relu(rates["h_inf"][:,1:]-rates["h_inf"][:,:-1]).mean())
            total=total+loss
        if not bool(torch.isfinite(total).detach().cpu()): raise RuntimeError(f"non-finite Task 4 loss at step {step}")
        total.backward(); _clip_models(models,torch,config.gradient_clip_norm); optimizer.step()
    selection={}; frozen={}
    for name,rows in checkpoints.items():
        winner=min(rows,key=lambda r:(r["mean_score"],r["step"])); selection[name]=winner; frozen[name]=states[f"{name}:{winner['step']}"]
    checkpoint=output_dir/"frozen_primitive_checkpoints.pt"; torch.save(frozen,checkpoint)
    freeze={"schema_version":"giada-task4-freeze-v1","code_revision":code_revision,"config":asdict(config),"data_contract":bundle["contract"],"selection":selection,"checkpoint_sha256":_file_sha(checkpoint),"task3e_sealed_accessed":False,"sealed_accessed":False}; freeze["freeze_sha256"]=_sha(freeze); (output_dir/"selection_freeze.json").write_text(json.dumps(freeze,indent=2)); report={"schema_version":"giada-task4-training-v1","valid":True,"environment":environment_manifest(torch),"checkpoints":checkpoints,"selection":selection,"parallelization":"family-and-seed concurrent on one GPU","sealed_accessed":False}; (output_dir/"training_report.json").write_text(json.dumps(report,indent=2)); return report


def _numerical_predictor(formula,name,config):
    grid=np.linspace(-135,75,config.lut_points); rates=np.asarray([[formula.rates(float(v))[k] for k in ("m_inf","h_inf","m_tau_ms","h_tau_ms")] for v in grid])
    if name=="chebyshev":
        transformed=np.column_stack((np.log(np.clip(rates[:,:2],1e-8,1-1e-8)/(1-np.clip(rates[:,:2],1e-8,1-1e-8))),np.log(rates[:,2:])))
        coeff=np.stack([np.polynomial.chebyshev.chebfit((grid+30)/105,transformed[:,i],config.chebyshev_degree) for i in range(4)])
    def predict(values):
        v,m,h,dt=np.asarray(values).T
        if name=="formula": rr=np.asarray([[formula.rates(float(x))[k] for k in ("m_inf","h_inf","m_tau_ms","h_tau_ms")] for x in v])
        elif name=="lut_nearest": rr=rates[np.clip(np.rint((v-grid[0])/(grid[1]-grid[0])).astype(int),0,len(grid)-1)]
        elif name=="lut_linear": rr=np.column_stack([np.interp(v,grid,rates[:,i]) for i in range(4)])
        else:
            raw=np.column_stack([np.polynomial.chebyshev.chebval((v+30)/105,coeff[i]) for i in range(4)]); rr=np.column_stack((1/(1+np.exp(-raw[:,:2])),np.exp(raw[:,2:])))
        z=-np.expm1(-dt[:,None]/rr[:,2:]); p=(1-z)*np.column_stack((m,h))+z*rr[:,:2]; return p,rr
    return predict


def _evaluate_numpy(predictor,rows):
    out={}
    for name,row in rows.items():
        p,_=predictor(row["inputs"]); out[name]={"m_rmse":float(np.sqrt(np.mean((p[:,0]-row["m_target"])**2))),"h_rmse":float(np.sqrt(np.mean((p[:,1]-row["h_target"])**2))),"open_rmse":float(np.sqrt(np.mean((p[:,0]**2*p[:,1]-row["open_target"])**2))),"occupancy_violation_count":int(np.count_nonzero((p<0)|(p>1)))}
    return out


def _benchmark_learned(model,torch,device,batch_sizes):
    result={}; S=_seed_count(model); model.eval()
    for batch in batch_sizes:
        x=torch.zeros(S,batch,4,device=device); x[...,0]=-55.0; x[...,1]=.4; x[...,2]=.6; x[...,3]=1.0
        with torch.inference_mode():
            for _ in range(5): model(x)
            if device.type=="cuda": torch.cuda.synchronize()
            repeats=50 if batch<=1024 else 10; started=time.perf_counter()
            for _ in range(repeats): model(x)
            if device.type=="cuda": torch.cuda.synchronize()
        elapsed=(time.perf_counter()-started)/repeats
        result[str(batch)]={"three_seed_ensemble_latency_ms":1000*elapsed,"effective_rows_per_second":S*batch/elapsed}
    return result


def _benchmark_numpy(predictor,batch_sizes):
    result={}
    for batch in batch_sizes:
        x=np.zeros((batch,4)); x[:,0]=-55; x[:,1]=.4; x[:,2]=.6; x[:,3]=1
        for _ in range(3): predictor(x)
        repeats=30 if batch<=1024 else 5; started=time.perf_counter()
        for _ in range(repeats): predictor(x)
        elapsed=(time.perf_counter()-started)/repeats
        result[str(batch)]={"single_cpu_latency_ms":1000*elapsed,"rows_per_second":batch/elapsed}
    return result


def evaluate_frozen_primitive_matrix(bundle,output_dir,config:PrimitiveMatrixConfig|None=None):
    config=config or PrimitiveMatrixConfig(); output_dir=Path(output_dir); marker=output_dir/"sealed_opened.json"
    if marker.exists(): raise RuntimeError("Task 4 sealed evaluation already opened")
    freeze=json.loads((output_dir/"selection_freeze.json").read_text()); claimed=freeze.pop("freeze_sha256")
    if _sha(freeze)!=claimed or freeze["sealed_accessed"] or freeze["task3e_sealed_accessed"]: raise RuntimeError("invalid Task 4 freeze")
    formula=bundle["formula"]; sealed={name:_enrich(_random_row(formula,role="sealed",axis=name,**spec),formula) for name,spec in bundle["sealed_spec"].items()}
    old={tuple(x) for row in [bundle["fit"],*bundle["development"].values()] for x in row["inputs"]}; new={tuple(x) for row in sealed.values() for x in row["inputs"]}
    if old&new: raise RuntimeError("Task 4 sealed rows overlap fit/development")
    torch=configure_torch_runtime(config.seeds[0]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); models={k:v.to(device) for k,v in _learned_models(torch,config).items()}; saved=torch.load(output_dir/"frozen_primitive_checkpoints.pt",map_location=device,weights_only=True)
    learned={}; params={}
    for name,model in models.items():
        model.seed_count=len(config.seeds); model.load_state_dict(saved[name]); per=_learned_eval(model,sealed,torch,device); scores=[_score(x) for x in per]; params[name]=sum(p.numel() for p in model.parameters())//len(config.seeds)
        learned[name]={"per_seed":per,"per_seed_scores":scores,"mean_score":float(np.mean(scores)),"max_seed_score":float(np.max(scores)),"parameter_count_per_seed":params[name],"gpu_efficiency":_benchmark_learned(model,torch,device,config.latency_batch_sizes)}
    numerical={}
    for name in config.numerical_families:
        predictor=_numerical_predictor(formula,name,config); metrics=_evaluate_numpy(predictor,sealed); numerical[name]={"metrics":metrics,"score":_score(metrics),"parameter_count":0,"cpu_efficiency":_benchmark_numpy(predictor,config.latency_batch_sizes)}
    ranking=sorted([{"family":k,"kind":"learned","score":v["mean_score"],"parameters":params[k]} for k,v in learned.items()]+[{"family":k,"kind":"numerical","score":v["score"],"parameters":0} for k,v in numerical.items()],key=lambda r:(r["score"],r["parameters"]))
    best_learned=min(learned,key=lambda k:learned[k]["mean_score"]); physical=learned["physical_tau"]
    decision={"best_learned":best_learned,"physical_tau_within_ten_percent_of_best_learned":physical["mean_score"]<=1.1*learned[best_learned]["mean_score"],"formula_is_accuracy_oracle":numerical["formula"]["score"]<1e-8,"task5_scaling_curves_available":True,"task6_authorized":True}
    report={"schema_version":"giada-task4-final-v1","valid":bool(np.isfinite([r["score"] for r in ranking]).all()),"selection_used_sealed":False,"task3e_sealed_accessed":False,"efficiency_comparison_policy":"GPU learned ensembles and CPU numerical primitives are reported separately; cross-device latency is not ranked.","sealed_contract":{"counts":{k:len(v["inputs"]) for k,v in sealed.items()},"overlap_with_fit_development":0,"opened_once_after_freeze":True},"learned":learned,"numerical":numerical,"ranking":ranking,"decision":decision}; (output_dir/"final_report.json").write_text(json.dumps(report,indent=2)); marker.write_text(json.dumps({"freeze_sha256":claimed,"final_report_sha256":_file_sha(output_dir/"final_report.json"),"opened_once":True},indent=2)); return report
