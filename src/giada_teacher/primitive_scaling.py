"""Task 5: paired capacity, budget and numerical resolution study."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .gpu_baseline_runtime import configure_torch_runtime, paired_index_generator, environment_manifest
from .joint_gate_generalization_diagnosis import _enrich, _random_row
from .primitive_matrix_playground import (
    PrimitiveMatrixConfig, _file_sha, _sha, _learned_models,
    _initialize_seed_slices, _tensor_row, _learned_eval, _score,
    prepare_primitive_matrix,
)

EXPECTED_TASK4_ARCHIVE_SHA256 = "47b75fcf71bfde8ed492c249ed18d6473b48d60de4b8ca0a98c5b6efb9813096"
EXPECTED_TASK4_REPORT_SHA256 = "d45dd919b98b98c8afa89e48ed29f01c9d1a8c1af9ba430cd9d6e8434f5bb896"


@dataclass(frozen=True)
class PrimitiveScalingConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    widths: tuple[int, ...] = (16, 23, 32)
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 10000, 30000, 50000, 75000)
    lut_points: tuple[int, ...] = (33, 65, 129, 257, 513)
    chebyshev_degrees: tuple[int, ...] = (8, 16, 24)
    batch_size: int = 768
    learning_rate: float = .003
    weight_decay: float = 1e-6
    inf_weight: float = .1
    log_tau_weight: float = .01
    shape_weight: float = .02
    benchmark_batch_sizes: tuple[int, ...] = (1024, 65536)

    def validate(self):
        if self.widths != (16, 23, 32) or self.seeds != (17, 29, 43):
            raise ValueError("Task 5 width/seed matrix differs from preregistration")
        if self.checkpoints[0] != 0 or self.checkpoints[-1] != 75000 or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("Invalid registered budget axis")
        if self.lut_points != (33, 65, 129, 257, 513) or self.chebyshev_degrees != (8, 16, 24):
            raise ValueError("Invalid registered numerical resolution axis")


def verified_task4_root(source: Path, cache: Path) -> Path:
    source, cache = Path(source), Path(cache)
    if source.is_file():
        if _file_sha(source) != EXPECTED_TASK4_ARCHIVE_SHA256:
            raise RuntimeError("Task 4 archive SHA-256 mismatch")
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (cache / member.filename).resolve()
                if cache.resolve() not in target.parents and target != cache.resolve():
                    raise RuntimeError("Unsafe Task 4 archive member")
            archive.extractall(cache)
        candidates = list(cache.rglob("final_report.json"))
        if len(candidates) != 1:
            raise RuntimeError("Ambiguous Task 4 report")
        root = candidates[0].parent
    else:
        root = source
    if _file_sha(root / "final_report.json") != EXPECTED_TASK4_REPORT_SHA256:
        raise RuntimeError("Task 4 report SHA-256 mismatch")
    report = json.loads((root / "final_report.json").read_text())
    if not report["valid"] or not report["decision"]["task6_authorized"]:
        raise RuntimeError("Task 4 result is not valid")
    return root


def prepare_scaling_data(formula, config: PrimitiveScalingConfig):
    config.validate()
    base = prepare_primitive_matrix(formula, PrimitiveMatrixConfig(pool_size=24000))
    base["contract"] = {**base["contract"], "schema_version": "giada-task5-data-v1", "task4_sealed_accessed": False}
    base["sealed_spec"] = {
        "central": {"seed": 75301, "count": 4096, "voltage_ranges": ((-100., 40.),), "dt_values": (.065, .325, .825)},
        "voltage_tail": {"seed": 75302, "count": 2048, "voltage_ranges": ((-133., -101.), (41., 73.)), "dt_values": (.065, .325, .825)},
        "long_horizon": {"seed": 75303, "count": 2048, "voltage_ranges": ((-135., -100.), (-100., -70.), (-70., -35.), (-35., 0.), (0., 40.), (40., 75.)), "dt_values": (12., 60., 90.)},
    }
    return base


def _physical_models(torch, config, device):
    models = {}
    for width in config.widths:
        base = PrimitiveMatrixConfig(physical_width=width)
        model = _learned_models(torch, base)["physical_tau"].to(device)
        _initialize_seed_slices({f"physical_tau_{width}": model}, torch, config.seeds)
        models[width] = model
    return models


def _gpu_rate_table(torch, formula, points, device, dtype):
    grid = np.linspace(-135., 75., points)
    rates = np.asarray([[formula.rates(float(v))[key] for key in ("m_inf", "h_inf", "m_tau_ms", "h_tau_ms")] for v in grid])
    return torch.as_tensor(rates, device=device, dtype=dtype)


def gpu_lut_predict(torch, values, rates, *, linear=True):
    voltage = values[..., 0]
    coordinate = ((voltage + 135.) * (rates.shape[0] - 1) / 210.).clamp(0, rates.shape[0] - 1)
    if linear:
        low = coordinate.floor().long().clamp(max=rates.shape[0] - 2)
        fraction = (coordinate - low).unsqueeze(-1)
        curve = rates[low] * (1 - fraction) + rates[low + 1] * fraction
    else:
        curve = rates[coordinate.round().long()]
    inf, tau = curve[..., :2], curve[..., 2:]
    gate = values[..., 1:3]
    z = -torch.expm1(-values[..., 3:4] / tau)
    return (1-z)*gate+z*inf


def _chebyshev_table(formula, degree, points=513):
    grid = np.linspace(-135., 75., points)
    rates = np.asarray([[formula.rates(float(v))[key] for key in ("m_inf", "h_inf", "m_tau_ms", "h_tau_ms")] for v in grid])
    inf = np.clip(rates[:, :2], 1e-8, 1-1e-8)
    transformed = np.column_stack((np.log(inf/(1-inf)), np.log(rates[:, 2:])))
    coeff = np.stack([np.polynomial.chebyshev.chebfit((grid+30)/105, transformed[:, i], degree) for i in range(4)])
    return coeff


def gpu_chebyshev_predict(torch, values, coeff):
    x = ((values[..., 0]+30.)/105.).clamp(-1., 1.)
    # Clenshaw recurrence keeps the GPU path free of Python loops over rows.
    b1 = torch.zeros(*x.shape, 4, device=x.device, dtype=x.dtype)
    b2 = torch.zeros_like(b1)
    for c in reversed(coeff[1:]):
        b0 = 2*x.unsqueeze(-1)*b1 - b2 + c
        b2, b1 = b1, b0
    raw = x.unsqueeze(-1)*b1 - b2 + coeff[0]
    inf = torch.sigmoid(raw[..., :2]); tau = torch.exp(raw[..., 2:]).clamp_min(1e-5)
    z = -torch.expm1(-values[..., 3:4]/tau)
    return (1-z)*values[..., 1:3]+z*inf


def _gpu_metrics(torch, predictor, rows, device):
    metrics = {}
    with torch.inference_mode():
        for name, row in rows.items():
            x = torch.as_tensor(row["inputs"], device=device, dtype=torch.float32)
            pred = predictor(x).cpu().double().numpy()
            metrics[name] = {
                "m_rmse": float(np.sqrt(np.mean((pred[:, 0]-row["m_target"])**2))),
                "h_rmse": float(np.sqrt(np.mean((pred[:, 1]-row["h_target"])**2))),
                "open_rmse": float(np.sqrt(np.mean((pred[:, 0]**2*pred[:, 1]-row["open_target"])**2))),
                "occupancy_violation_count": int(np.count_nonzero((pred < 0) | (pred > 1))),
            }
    return metrics


def _gpu_benchmark(torch, predictor, device, sizes):
    result = {}
    for size in sizes:
        x = torch.zeros(size, 4, device=device); x[:, 0] = -55.; x[:, 1:3] = .5; x[:, 3] = 1.
        with torch.inference_mode():
            for _ in range(5): predictor(x)
            torch.cuda.synchronize(); repetitions = 30 if size <= 1024 else 10
            started = time.perf_counter()
            for _ in range(repetitions): predictor(x)
            torch.cuda.synchronize()
        seconds = (time.perf_counter()-started)/repetitions
        result[str(size)] = {"latency_ms": 1000*seconds, "rows_per_second": size/seconds}
    return result


def _numerical_matrix(torch, formula, rows, config, device, *, benchmark=True):
    result = {}
    for points in config.lut_points:
        rates = _gpu_rate_table(torch, formula, points, device, torch.float32)
        for linear in (True, False):
            label = f"lut_{'linear' if linear else 'nearest'}_{points}"
            predictor = lambda x, table=rates, interpolation=linear: gpu_lut_predict(torch, x, table, linear=interpolation)
            metrics = _gpu_metrics(torch, predictor, rows, device)
            result[label] = {"score": _score(metrics), "metrics": metrics, "table_bytes": rates.numel()*rates.element_size(), "gpu": _gpu_benchmark(torch, predictor, device, config.benchmark_batch_sizes) if benchmark else None}
    for degree in config.chebyshev_degrees:
        coeff = torch.as_tensor(_chebyshev_table(formula, degree).T, device=device, dtype=torch.float32)
        predictor = lambda x, c=coeff: gpu_chebyshev_predict(torch, x, c)
        metrics = _gpu_metrics(torch, predictor, rows, device)
        result[f"chebyshev_{degree}"] = {"score": _score(metrics), "metrics": metrics, "table_bytes": coeff.numel()*coeff.element_size(), "gpu": _gpu_benchmark(torch, predictor, device, config.benchmark_batch_sizes) if benchmark else None}
    return result


def train_and_freeze_scaling(bundle, output_dir, config: PrimitiveScalingConfig, code_revision="unknown"):
    config.validate(); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    torch = configure_torch_runtime(config.seeds[0])
    if not torch.cuda.is_available(): raise RuntimeError("Task 5 requires CUDA")
    device = torch.device("cuda"); models = _physical_models(torch, config, device); seed_count = len(config.seeds)
    optimizers = {width: torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay) for width, model in models.items()}
    fit = _tensor_row(bundle["fit"], torch, device)
    streams = [paired_index_generator(len(bundle["fit"]["inputs"]), config.batch_size, 850000+seed) for seed in config.seeds]
    probe = torch.linspace(-140., 80., 96, device=device)
    probe_x = torch.stack((probe, torch.full_like(probe, .5), torch.full_like(probe, .5), torch.ones_like(probe)), -1).unsqueeze(0).expand(seed_count, -1, -1)
    checkpoints = {str(width): [] for width in config.widths}; selected_states = {}; started = time.perf_counter(); maximum = config.checkpoints[-1]
    for step in range(maximum+1):
        if step in config.checkpoints:
            summary=[]
            for width, model in models.items():
                metrics = _learned_eval(model, bundle["development"], torch, device)
                scores = [_score(row) for row in metrics]
                checkpoints[str(width)].append({"step": step, "mean_score": float(np.mean(scores)), "max_seed_score": float(np.max(scores)), "per_seed_scores": scores})
                selected_states[(width, step)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                summary.append(f"w{width}={np.mean(scores):.3g}")
            eta = (time.perf_counter()-started)/max(step,1)*(maximum-step)
            print(f"[GIADA Task 5] {100*step/maximum:.1f}% ETA {eta/60:.1f} min {' '.join(summary)}", flush=True)
        if step == maximum: break
        indices = torch.stack([torch.as_tensor(next(stream), device=device, dtype=torch.long) for stream in streams])
        batch = {key: value[indices] for key,value in fit.items()}
        target = torch.stack((batch["m_target"],batch["h_target"]), -1)
        for width,model in models.items():
            optimizers[width].zero_grad(set_to_none=True)
            pred, rates = model(batch["inputs"])
            loss = ((pred-target)**2).mean()+((rates["open"]-target[...,0].square()*target[...,1])**2).mean()
            loss = loss+config.inf_weight*sum(((rates[k]-batch[k])**2).mean() for k in ("m_inf","h_inf"))
            loss = loss+config.log_tau_weight*sum(((torch.log(rates[k])-torch.log(batch[k]))**2).mean() for k in ("m_tau_ms","h_tau_ms"))
            _, shape = model(probe_x)
            loss = loss+config.shape_weight*(torch.relu(shape["m_inf"][:,:-1]-shape["m_inf"][:,1:]).mean()+torch.relu(shape["h_inf"][:,1:]-shape["h_inf"][:,:-1]).mean())
            if not bool(torch.isfinite(loss).detach().cpu()): raise RuntimeError(f"Nonfinite loss width={width} step={step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizers[width].step()
    selection = {str(width): min(rows, key=lambda row: (row["mean_score"],row["step"])) for width,rows in ((w,checkpoints[str(w)]) for w in config.widths)}
    numerical_development = _numerical_matrix(torch, bundle["formula"], bundle["development"], config, device, benchmark=False)
    numerical_selection = {
        "linear_lut": min((key for key in numerical_development if key.startswith("lut_linear_")), key=lambda key: numerical_development[key]["score"]),
        "nearest_lut": min((key for key in numerical_development if key.startswith("lut_nearest_")), key=lambda key: numerical_development[key]["score"]),
        "chebyshev": min((key for key in numerical_development if key.startswith("chebyshev_")), key=lambda key: numerical_development[key]["score"]),
    }
    best_physical_width = min(selection, key=lambda key: selection[key]["mean_score"])
    checkpoint = output_dir/"frozen_scaling_checkpoints.pt"
    torch.save({str(width): selected_states[(width,selection[str(width)]["step"])] for width in config.widths}, checkpoint)
    freeze = {"schema_version":"giada-task5-freeze-v1","code_revision":code_revision,"config":asdict(config),"selection":selection,"numerical_selection":numerical_selection,"best_physical_width":best_physical_width,"checkpoint_sha256":_file_sha(checkpoint),"task4_sealed_accessed":False,"task5_sealed_accessed":False}
    freeze["freeze_sha256"] = _sha(freeze)
    (output_dir/"selection_freeze.json").write_text(json.dumps(freeze,indent=2))
    report = {"schema_version":"giada-task5-training-v1","valid":True,"environment":environment_manifest(torch),"checkpoint_curves":checkpoints,"selection":selection,"numerical_development":numerical_development,"numerical_selection":numerical_selection,"best_physical_width":best_physical_width,"same_minibatches_across_widths":True,"task4_sealed_accessed":False}
    (output_dir/"training_report.json").write_text(json.dumps(report,indent=2))
    return report


def evaluate_frozen_scaling(bundle, output_dir, config: PrimitiveScalingConfig):
    output_dir = Path(output_dir); marker = output_dir/"sealed_opened.json"
    if marker.exists(): raise RuntimeError("Task 5 sealed was already opened")
    freeze = json.loads((output_dir/"selection_freeze.json").read_text()); digest=freeze.pop("freeze_sha256")
    if _sha(freeze)!=digest or freeze["task5_sealed_accessed"] or freeze["task4_sealed_accessed"]: raise RuntimeError("Invalid Task 5 freeze")
    if _sha(freeze["config"]) != _sha(asdict(config)): raise RuntimeError("Task 5 config changed after freeze")
    checkpoint = output_dir/"frozen_scaling_checkpoints.pt"
    if _file_sha(checkpoint)!=freeze["checkpoint_sha256"]: raise RuntimeError("Task 5 checkpoint hash mismatch")
    formula = bundle["formula"]
    sealed = {name:_enrich(_random_row(formula, role="sealed", axis=name, **spec),formula) for name,spec in bundle["sealed_spec"].items()}
    old = {tuple(x) for row in [bundle["fit"],*bundle["development"].values()] for x in row["inputs"]}
    new = {tuple(x) for row in sealed.values() for x in row["inputs"]}
    if old & new: raise RuntimeError("Task 5 sealed overlaps fit/development")
    torch = configure_torch_runtime(config.seeds[0]); device=torch.device("cuda")
    if not torch.cuda.is_available(): raise RuntimeError("Task 5 requires CUDA")
    states=torch.load(checkpoint,map_location=device,weights_only=True)
    physical={}
    for width,model in _physical_models(torch,config,device).items():
        model.load_state_dict(states[str(width)])
        per_seed=_learned_eval(model,sealed,torch,device)
        scores=[_score(row) for row in per_seed]
        single_config=PrimitiveMatrixConfig(seeds=(config.seeds[0],),physical_width=width)
        single=_learned_models(torch,single_config)["physical_tau"].to(device)
        single.load_state_dict({key:value[:1].clone() for key,value in states[str(width)].items()})
        single.eval()
        predictor=lambda x, network=single: network(x.unsqueeze(0))[0][0]
        physical[str(width)]={"per_seed_scores":scores,"mean_score":float(np.mean(scores)),"max_seed_score":float(np.max(scores)),"parameter_count_per_seed":sum(p.numel() for p in model.parameters())//len(config.seeds),"gpu":_gpu_benchmark(torch,predictor,device,config.benchmark_batch_sizes)}
    numerical=_numerical_matrix(torch,formula,sealed,config,device)
    best_physical=freeze["best_physical_width"]
    best_lut=freeze["numerical_selection"]["linear_lut"]
    report={"schema_version":"giada-task5-final-v1","valid":bool(np.isfinite([v["mean_score"] for v in physical.values()]+[v["score"] for v in numerical.values()]).all()),"selection_used_sealed":False,"task4_sealed_accessed":False,"sealed_contract":{"count":sum(len(row["inputs"]) for row in sealed.values()),"overlap":0,"opened_once_after_freeze":True},"physical":physical,"numerical":numerical,"decision":{"best_physical_width":int(best_physical),"best_linear_lut":best_lut,"same_gpu_comparison":True,"task6_authorized":True}}
    (output_dir/"final_report.json").write_text(json.dumps(report,indent=2))
    marker.write_text(json.dumps({"freeze_sha256":digest,"final_report_sha256":_file_sha(output_dir/"final_report.json"),"opened_once":True},indent=2))
    return report
