"""Task21: isolated canonical Nap_Et2 h; duration support x rate supervision.

No full Nap current, coupled voltage, or acceleration claim. The preceding
m-rate branch at -38mV is preserved because it changes local h-rate voltage.
"""
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from . import single_gate_transfer as engine
from .hh_family_transfer import TEACHER_REVISION, sha, write

TASK_LABEL = 'Task21'
CHANNELS = ('short_dt', 'multiscale_dt')
SHORT_DT = [.025, .1, .5, 1.]
LONG_DT = [.025, .1, .5, 1., 5., 25., 100., 500., 1000., 5000., 10000.]
model_factory = engine.model_factory
equivalent = engine.equivalent


def rates(channel, voltage):
    if channel not in CHANNELS:
        raise ValueError(channel)
    v = np.asarray(voltage, dtype=np.float64).copy()
    for singular in (-38., -17., -64.4):
        v = np.where(v == singular, v + .0001, v)
    inf = 1 / (1 + np.exp((v + 48.8) / 10))
    a = -2.88e-6 * (v + 17) / (-np.expm1((v + 17) / 4.63))
    b = 6.94e-6 * (v + 64.4) / (-np.expm1(-(v + 64.4) / 2.63))
    tau = 1 / (a + b) / 2.3 ** 1.3
    if not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError('Invalid canonical slow rates')
    return inf, tau


def data(seed, count, domain=(-135, 75)):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(*domain, count), rng.uniform(0, 1, count),
                 rng.choice(LONG_DT, count), rng.choice(SHORT_DT, count)]


def fit(config):
    x = data(210101, config['pool_size'])
    n = len(x) // 2
    x[:n, 0] = np.random.default_rng(210102).uniform(-75, -25, n)
    return x


def fit_targets(values, descriptors, inf, tau):
    durations = np.stack([values[:, 3 if c == 'short_dt' else 2] for c, _, _ in descriptors])
    return engine.update(values[:, 1], inf, tau, durations)


def training_batch(x, indices, descriptors):
    batch = x[indices][None].expand(len(descriptors), -1, -1).clone()
    for n, (condition, _, _) in enumerate(descriptors):
        if condition == 'short_dt':
            batch[n, :, 2] = batch[n, :, 3]
    return batch


def grid(fresh=False):
    n, offset = (401, .37) if fresh else (201, .5)
    voltage = -135 + 210 * (np.arange(n) + offset) / n
    if fresh:
        voltage = np.r_[voltage, -135., 75., -38., -17., -64.4]
    v, dt, state = np.meshgrid(voltage, LONG_DT, [0., 1.], indexing='ij')
    return np.c_[v.ravel(), state.ravel(), dt.ravel(), dt.ravel()]


def development():
    return dict(uniform=data(210201, 2048), negative_tail=data(210202, 2048, (-135, -115)), state_extrema=grid())


def fresh():
    return dict(in_support=data(210401, 4096), negative_tail=data(210404, 4096, (-135, -115)), state_extrema=grid(True),
                ood_negative=data(210402, 2048, (-155, -135)), ood_positive=data(210403, 2048, (75, 95)))


def roll_inputs():
    v = np.random.default_rng(210405).uniform(-135, 75, 128)
    return np.c_[np.repeat(v, 2), np.tile([0., 1.], len(v)), np.ones(2*len(v)), np.ones(2*len(v))]


def measurements(x, prediction, inf, tau, channel):
    ti, tt = rates(channel, x[:, 0])
    error = prediction - engine.update(x[:, 1], ti, tt, x[:, 2])
    return dict(gate_rmse=float(np.sqrt(np.mean(error**2))), gate_max_error=float(abs(error).max()),
                inf_rmse=float(np.sqrt(np.mean((inf-ti)**2))),
                log_tau_rmse=float(np.sqrt(np.mean((np.log(tau)-np.log(tt))**2))),
                open_rmse=float(np.sqrt(np.mean(error**2))),
                occupancy_violations=int(np.sum((prediction<0)|(prediction>1))),
                finite=bool(np.isfinite(prediction).all() and np.isfinite(inf).all() and np.isfinite(tau).all() and (tau>0).all()))


def evaluate(model, values, descriptors, torch, device):
    x = torch.tensor(values, dtype=torch.float32, device=device)[None].expand(len(descriptors), -1, -1)
    with torch.no_grad():
        outputs = [o.cpu().double().numpy() for o in model(x)]
    return [measurements(values, *(o[n] for o in outputs), c) for n, (c, _, _) in enumerate(descriptors)]


def rollout(model, descriptors, values, horizons, torch, device):
    x = torch.tensor(values, dtype=torch.float32, device=device)[None].expand(len(descriptors), -1, -1)
    ti, tt = rates(CHANNELS[0], values[:, 0])
    oracle_inf = torch.tensor(ti, dtype=torch.float32, device=device)
    oracle_z = -torch.expm1(-torch.tensor(values[:, 2]/tt, dtype=torch.float32, device=device))
    oracle_state = x[0, :, 1].clone()
    results = [{} for _ in descriptors]
    with torch.no_grad():
        _, inf, tau = model(x)
        z = -torch.expm1(-x[..., 2]/tau)
        state = x[..., 1].clone()
        for step in range(1, max(horizons)+1):
            state = (1-z)*state+z*inf
            oracle_state = (1-oracle_z)*oracle_state+oracle_z*oracle_inf
            if step in horizons:
                target = engine.update(values[:, 1], ti, tt, step*values[:, 2])
                pred = state.cpu().double().numpy()
                floor_error = oracle_state.cpu().double().numpy()-target
                for n in range(len(descriptors)):
                    error = pred[n]-target
                    results[n][str(step)] = dict(gate_rmse=float(np.sqrt(np.mean(error**2))), gate_max_error=float(abs(error).max()),
                        occupancy_violations=int(np.sum((pred[n]<0)|(pred[n]>1))), finite=bool(np.isfinite(pred[n]).all()),
                        persistence_rmse=float(np.sqrt(np.mean((values[:, 1]-target)**2))),
                        formula_f32_rmse=float(np.sqrt(np.mean(floor_error**2))),
                        learned_single_macro_update_rmse=float(np.sqrt(np.mean((engine.update(values[:,1], inf[n].cpu().double().numpy(), tau[n].cpu().double().numpy(), step*values[:,2])-target)**2))))
    return results


def parameter_count(width):
    return width*width+5*width+2


def dense_macs(width):
    return width*width+3*width


def numerical_grid(size):
    return np.linspace(-135, 75, size)


def gpu_benchmark(torch, selected_models, values, output):
    return dict(measured=False, reason='Task21 is a slow-state accuracy/identifiability experiment; no acceleration claim.')


def native_audit(teacher, output):
    from neuron import h, load_mechanisms
    import neuron
    assert neuron.__version__.split('+')[0] == '8.2.7'
    assert subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'], text=True).strip() == TEACHER_REVISION
    root = Path(output)/'native'
    root.mkdir()
    source = Path(teacher)/'L5PC_NEURON_simulation/mods/Nap_Et2.mod'
    raw = source.read_text()
    modified = raw.replace('SUFFIX Nap_Et2', 'SUFFIX Nap_Et2\n RANGE hInf, hTau', 1)
    (root/source.name).write_text(modified)
    command = shutil.which('nrnivmodl') or str(Path(sys.executable).parent/'nrnivmodl')
    with (root/'compile.log').open('w') as log:
        subprocess.run([command,str(root.resolve())],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    load_mechanisms(str(root.resolve()))
    h.CVode().active(0)
    h.secondorder = 0
    sec = h.Section(name='audit_slow_h')
    sec.insert('Nap_Et2')
    mech = sec(.5).Nap_Et2
    mech.gNap_Et2bar = 0
    rate_error = gate_error = drift = 0.
    for voltage in np.r_[np.linspace(-155,95,101), -38., -17., -64.4, -38.00001, -37.99999, -64.40001, -64.39999, -17.00001, -16.99999]:
        inf, tau = rates(CHANNELS[0], voltage)
        h.finitialize(float(voltage))
        rate_error = max(rate_error, float(np.max(abs(np.array([mech.hInf,mech.hTau])-np.array([inf,tau]))/(1+abs(np.array([inf,tau]))))))
        for dt in (.025, 1., 100., 1000., 10000.):
            h.dt = dt
            h.finitialize(float(voltage))
            mech.h = .31
            h.fadvance()
            gate_error = max(gate_error, abs(float(mech.h)-float(engine.update(.31,inf,tau,dt))))
            drift = max(drift, abs(sec(.5).v-voltage))
    h.delete_section(sec=sec)
    report = dict(valid=rate_error<1e-7 and gate_error<1e-7 and drift<1e-8,
                  rate_scaled_max_error=rate_error, gate_max_error=gate_error, unexpected_voltage_drift_mv=drift,
                  canonical_mod_sha256=sha(source), instrumented_mod_sha256=sha(root/source.name),
                  scope='Nap_Et2 h only; m branch preserved; gbar0; canonical qt; RANGE-only instrumentation.')
    write(Path(output)/'native_audit.json',report)
    if not report['valid']:
        raise RuntimeError('Task21 double oracle failed')
    return report


def finalize(report, benchmark_models, torch, device, output, config):
    report.pop('task20_authorized')
    report.pop('single_gate_transfer_passed')
    report['schema_version'] = 'giada-roadmap-task21-v1'
    passed = report['per_channel_transfer']['multiscale_dt']
    report.update(slow_gate_passed=passed, task22_authorized=passed,
                  whole_Nap_channel_tested=False, current_tested=False,
                  rollout_dt_ms=1, duration_support_is_training_factor=True)
    rows = __import__('json').loads((Path(output)/'development_ladder.json').read_text())
    index = {(r['channel'],r['objective'],r['seed'],r['width'],r['step']):r for r in rows}
    contrasts=[]
    for r in rows:
        if r['channel']=='short_dt':
            other=index[('multiscale_dt',r['objective'],r['seed'],r['width'],r['step'])]
            contrasts.append(dict(seed=r['seed'],objective=r['objective'],width=r['width'],step=r['step'],
                score_difference=other['score']-r['score'], metric_differences={d:{k:other['metrics'][d][k]-r['metrics'][d][k] for k in config['gates']} for d in r['metrics']}))
    write(Path(output)/'paired_duration_contrasts.json',contrasts)
    return report


def run(output, config, revision):
    return engine.run(output,config,revision,backend=sys.modules[__name__])
