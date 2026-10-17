"""New roadmap Task37: causal ionic fixtures and stratified current evaluators.

No neuron membrane is learned or simulated here: V and the same-tick calcium
bridge are externally prescribed. Native validation remains separately scoped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from . import ionic_block_teacher_forced as ionic

CONFIG = 'experiments/task37_ionic_playground_preregistration_v1.json'
OFFSETS = tuple(int(x) for x in ionic.OFFSETS)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def readout(voltage, gates, multipliers, reversals):
    """Units S/cm2, mV -> mA/cm2; supports leading time/batch axes."""
    openings = []
    for k, names in enumerate(ionic.STATE_NAMES):
        opening = gates[..., OFFSETS[k]] ** int(ionic.POWERS[k])
        if len(names) == 2:
            opening = opening * gates[..., OFFSETS[k] + 1]
        openings.append(opening)
    conductance = np.stack(openings, -1) * ionic.GBAR * multipliers
    return conductance, conductance * (voltage[..., None] - reversals)


def coefficients(voltage, bridge_ca, dt):
    shape = voltage.shape + (18,)
    inf, fraction = np.empty(shape), np.empty(shape)
    for k, name in enumerate(ionic.CHANNELS):
        a, b = OFFSETS[k:k+2]
        if name == 'SK_E2':
            u = 1 / (1 + (.00043 / np.maximum(bridge_ca, 1e-7)) ** 4.8)
            tau = np.ones_like(u)
        else:
            u, tau = ionic.rates(name, voltage, bridge_ca)
        if b-a == 1:
            u, tau = u[..., None], tau[..., None]
        inf[..., a:b], fraction[..., a:b] = u, -np.expm1(-dt / tau)
    if not np.isfinite(inf).all() or not np.isfinite(fraction).all():
        raise ValueError('Nonfinite canonical coefficients')
    return inf, fraction


def recurrent_reference(voltage, bridge_ca, initial, multipliers, reversals, dt, reset_tick=None):
    """Updated V gates -> source Ica -> external Ca bridge -> updated SK."""
    inf, fraction = coefficients(voltage, bridge_ca, dt)
    gates = np.empty(voltage.shape + (18,), dtype=np.float64)
    current = np.array(initial, dtype=np.float64, copy=True)
    for tick in range(len(voltage)):
        if tick == reset_tick:
            current = np.array(initial, dtype=np.float64, copy=True)
        current = (1-fraction[tick])*current + fraction[tick]*inf[tick]
        gates[tick] = current
    conductances, currents = readout(voltage, gates, multipliers, reversals)
    # SK does not contribute to Ica; no use of the external bridge in its source.
    source_ca = currents[..., :2].sum(-1)
    return gates, currents, source_ca


def make_paths(spec, role, family):
    """Each role has independent history parameters, never a pointwise split."""
    if role not in spec['roles']:
        raise ValueError('Unregistered role; confirmation generation is forbidden')
    fidx = list(spec['families']).index(family)
    cfg = spec['roles'][role]
    rng = np.random.default_rng(cfg['seed'] + fidx * 1009)
    count, dt = cfg['trajectories_per_family'], spec['dt_ms']
    duration = spec['families'][family]['duration_ms']
    ticks = round(duration / dt)
    t = np.arange(ticks) * dt
    v = np.empty((ticks, count)); ca = np.empty_like(v)
    initial = np.empty((count, 18)); mult = np.empty((count, 11)); rev = np.empty_like(mult)
    metadata = []
    for j in range(count):
        rest, high = rng.uniform(-88, -65), rng.uniform(-25, 35)
        prep_v, prep_ca = rng.uniform(-95, 25), 10.**rng.uniform(-7, -2)
        u, _ = coefficients(np.array([prep_v]), np.array([prep_ca]), dt)
        initial[j] = u[0] if j % 2 == 0 else rng.uniform(0, 1, 18)
        mult[j] = rng.uniform(.25, 4, 11)
        rev[j] = ionic.REVERSALS + rng.uniform(-5, 5, 11)
        x = np.full(ticks, rest); calcium = np.full(ticks, 10.**rng.uniform(-7, -2))
        onset, end = .2*duration + rng.uniform(0, .01*duration), .7*duration
        pulse_count = 1
        if family in ('steps', 'channel_block'):
            x[(t >= onset) & (t < end)] = high
            if family == 'channel_block':
                mult[j, (j + fidx) % 11] = 0
        elif family == 'ramps':
            x = rest + (high-rest) * np.maximum(0, 1-np.abs(2*t/duration-1))
        elif family == 'chirp':
            x = (rest+high)/2 + (high-rest)/2*np.sin(2*np.pi*(t/duration + 8*(t/duration)**2))
            pulse_count = 9
        elif family == 'rapid_voltage_train':
            period, width = rng.uniform(2, 6), rng.uniform(.4, 1.2)
            active = (t >= onset) & (t < end)
            x[active & (np.mod(t-onset, period) < width)] = high
            pulse_count = int(np.ceil((end-onset)/period))
        elif family == 'paired_recovery':
            gap = 10.**rng.uniform(.3, 2.1)
            x[(t >= onset) & (t < onset+4)] = high
            x[(t >= onset+4+gap) & (t < onset+8+gap)] = high
            pulse_count = 2
        elif family == 'same_v_history':
            x[t < duration*.4] = rng.uniform(-95, -15)
            x[t >= duration*.4] = -55
            onset, end = 0., duration*.4
        elif family == 'slow_recovery':
            x[t < duration*.2] = high
            onset, end = 0., duration*.2
        elif family == 'calcium_switch':
            x[(t >= onset) & (t < end)] = high
            calcium[t >= onset] = 10.**rng.uniform(-4, -2)
            calcium[t >= end] = 10.**rng.uniform(-7, -5)
        elif family == 'reversal':
            k = j % 11
            x = np.linspace(rev[j, k]-8, rev[j, k]+8, ticks).clip(-100, 60)
        elif family == 'activation_boundary':
            center = rng.choice([-60., -48.8, -38., -27., -10.])
            x = center + rng.uniform(.2, 3)*np.sin(2*np.pi*t/duration*5)
            pulse_count = 5
        elif family == 'seams':
            centers = np.array([-66., -64.4, -60., -38., -32., -27., -17.])
            x = np.resize(np.repeat(centers, 10), ticks)
            if j % 2:
                x = x + 1e-4
            pulse_count = len(centers)
        else:
            raise ValueError(family)
        if role == 'stress_diagnostic':
            rev[j] = ionic.REVERSALS + rng.uniform(-15, 15, 11)
            if family == 'calcium_switch':
                calcium[t >= end] = spec['calcium_stress_mM'][j % 2]
        v[:, j], ca[:, j] = x, calcium
        # Signature includes all inputs/initial state/parameters, not the seed alone.
        h = hashlib.sha256()
        for a in (x, calcium, initial[j], mult[j], rev[j]):
            h.update(np.asarray(a, dtype='<f8').tobytes())
        metadata.append(dict(id=f'{role}/{family}/{j}', role=role, family=family,
            seed=cfg['seed'] + fidx*1009, trajectory=j, duration_ms=duration,
            input_sha256=h.hexdigest(), pulse_count=pulse_count,
            onset_ms=float(onset), end_ms=float(end), preparation_v_mv=float(prep_v),
            preparation_ca_mM=float(prep_ca), initial_kind='equilibrium' if j % 2 == 0 else 'off_equilibrium',
            rare=spec['families'][family]['rare']))
    return dict(voltage=v, bridge_ca=ca, initial_gates=initial, multipliers=mult,
                reversals=rev), metadata


def event_metrics(reference, prediction, dt, floor, plateau_band=.001):
    """One transient/window. Multiple bursts are evaluated in later composition."""
    base = reference[0]
    rr, pp = np.abs(reference-base), np.abs(prediction-base)
    idx = int(np.argmax(rr)); amplitude = float(rr[idx])
    scale = max(float(np.max(np.abs(reference))), floor)
    if amplitude < .25*scale or amplitude < 10*floor or idx in (0, len(rr)-1):
        return dict(status='not_evaluable', reason='insufficient or boundary transient')
    target_active = np.flatnonzero(rr >= .1*amplitude)
    pred_active = np.flatnonzero(pp >= .1*amplitude)
    if not len(pred_active):
        return dict(status='missing_event', peak_error_ms=None, onset_error_ms=None, offset_error_ms=None)
    # A flat peak has an interval, not a unique time; rounding must not create
    # a fictitious delay. Compare the nearest near-maximum samples.
    rpeak = np.flatnonzero(rr >= amplitude*(1-plateau_band))
    ppeak = np.flatnonzero(pp >= float(pp.max())*(1-plateau_band))
    positions = np.searchsorted(rpeak, ppeak)
    left = rpeak[np.maximum(positions-1, 0)]
    right = rpeak[np.minimum(positions, len(rpeak)-1)]
    peak_distance = np.minimum(np.abs(ppeak-left), np.abs(ppeak-right)).min()
    return dict(status='evaluable', peak_error_ms=float(peak_distance*dt),
        onset_error_ms=abs(int(pred_active[0])-int(target_active[0]))*dt,
        offset_error_ms=abs(int(pred_active[-1])-int(target_active[-1]))*dt)


def evaluate_currents(reference, prediction, dt, quality, reference_gates=None, prediction_gates=None, blocked=None):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.ndim != 2 or reference.shape[-1] != 11 or prediction.shape != reference.shape or not len(reference):
        raise ValueError('Require aligned, nonempty (ticks,11) signed current arrays')
    if dt <= 0 or not np.isfinite(reference).all():
        raise ValueError('Reference and clock must be finite/valid')
    if not np.isfinite(prediction).all():
        return dict(passed=False, finite=False, channels=[], gate=None)
    floor = quality['current_scale_floor_ma_cm2']
    scale = np.maximum(np.abs(reference).max(0), floor)
    diff = prediction-reference
    rms = np.sqrt(np.mean(diff**2, 0)); maximum = np.max(np.abs(diff), 0)
    signed = np.abs(diff.sum(0)*dt)/(len(reference)*dt*scale)
    absolute = np.abs((np.abs(prediction)-np.abs(reference)).sum(0)*dt)/(len(reference)*dt*scale)
    rows = []
    for k, name in enumerate(ionic.CHANNELS):
        events = event_metrics(reference[:, k], prediction[:, k], dt, floor, quality.get('peak_plateau_relative_band', .001))
        event_pass = events['status'] == 'not_evaluable' or (
            events['status'] == 'evaluable' and events['peak_error_ms'] <= quality['peak_timing_max_ms'] + 1e-10 and
            max(events['onset_error_ms'], events['offset_error_ms']) <= quality['onset_offset_max_ms'] + 1e-10)
        blocked_ok = blocked is None or not blocked[k] or np.abs(prediction[:, k]).max() <= quality['blocked_current_atol_ma_cm2']
        passed = rms[k]/scale[k] <= quality['current_nrmse_max'] and maximum[k]/scale[k] <= quality['current_nmax_max'] and signed[k] <= quality['signed_charge_error_max'] and absolute[k] <= quality['absolute_charge_error_max'] and event_pass and blocked_ok
        rows.append(dict(channel=name, passed=bool(passed), scale_ma_cm2=float(scale[k]),
            rmse_ma_cm2=float(rms[k]), max_abs_ma_cm2=float(maximum[k]),
            nrmse=float(rms[k]/scale[k]), nmax=float(maximum[k]/scale[k]),
            signed_charge_error_ma_ms_cm2=float(abs(diff[:, k].sum()*dt)),
            absolute_charge_error_ma_ms_cm2=float(abs((np.abs(prediction[:, k])-np.abs(reference[:, k])).sum()*dt)),
            signed_charge_normalized=float(signed[k]), absolute_charge_normalized=float(absolute[k]),
            reference_signed_charge_ma_ms_cm2=float(reference[:, k].sum()*dt),
            event=events, blocked_current_passed=bool(blocked_ok)))
    gate = None
    if (reference_gates is None) != (prediction_gates is None):
        raise ValueError('Both gate trajectories are required')
    if reference_gates is not None:
        a, b = np.asarray(reference_gates), np.asarray(prediction_gates)
        if a.shape != b.shape or a.shape != (len(reference), 18) or not np.isfinite(a).all():
            raise ValueError('Gate shape/reference mismatch')
        finite = bool(np.isfinite(b).all())
        occupancy = int(np.sum((b < -quality['occupancy_atol']) | (b > 1+quality['occupancy_atol'])))
        rmse = float(np.sqrt(np.mean((a-b)**2))) if finite else None
        max_abs = float(np.max(np.abs(a-b))) if finite else None
        gate = dict(finite=finite, occupancy_violations=occupancy, rmse=rmse, max_abs=max_abs,
                    passed=bool(finite and occupancy == 0 and rmse <= quality['gate_rmse_max'] and max_abs <= quality['gate_max_abs_max']))
    horizons = {}
    for horizon in quality['evaluation_horizons_ms']:
        ticks = round(horizon/dt)
        if ticks <= len(reference):
            horizons[str(horizon)] = float(np.sqrt(np.mean((diff[:ticks]/scale)**2)))
    failing = np.flatnonzero(np.any(np.abs(diff)/scale > quality['current_nmax_max'], axis=1))
    return dict(passed=all(r['passed'] for r in rows) and (gate is None or gate['passed']), finite=True,
        channels=rows, gate=gate, horizon_current_nrmse=horizons,
        first_tolerance_exceedance_ms=float(failing[0]*dt) if len(failing) else None,
        nonfailure_right_censored=not len(failing),
        total_current_rmse_ma_cm2=float(np.sqrt(np.mean(diff.sum(-1)**2))),
        causal_behavior_validated=False)


def evaluate_intervention(reference_base, reference_intervention, prediction_base, prediction_intervention, dt, quality):
    return evaluate_currents(np.asarray(reference_intervention)-np.asarray(reference_base),
        np.asarray(prediction_intervention)-np.asarray(prediction_base), dt, quality)


def sensitivity_controls(spec):
    q, dt = spec['quality'], spec['dt_ms']
    t = np.arange(1000)*dt
    signal = np.exp(-.5*((t-30)/2)**2)*1e-3
    reference = np.stack([signal*(1+k*.1)*(-1 if k < 5 else 1) for k in range(11)], -1)
    correct = evaluate_currents(reference, reference.copy(), dt, q)
    assert correct['passed']
    controls = {}
    controls['50pct_peak_attenuation'] = reference*.5
    controls['1ms_delay'] = np.concatenate((np.zeros((10, 11)), reference[:-10]), axis=0)
    cancellation = reference.copy(); cancellation[:, 0] += 3e-4; cancellation[:, 1] -= 3e-4
    controls['opposing_current_cancellation'] = cancellation
    controls['persistent_drift'] = reference + np.linspace(0, 3e-4, len(t))[:, None]
    wrong_memory = reference.copy(); wrong_memory[400:] += 2e-4
    controls['wrong_memory_reset'] = wrong_memory
    bad = reference.copy(); bad[100, 0] = np.nan; controls['nonfinite'] = bad
    controls['missing_event'] = np.zeros_like(reference)
    controls['false_event_on_quiescent_channel'] = reference.copy()
    reports = {}
    for name, prediction in controls.items():
        rr = reference.copy()
        if name == 'false_event_on_quiescent_channel':
            rr[:, 10] = 0
        report = evaluate_currents(rr, prediction, dt, q)
        assert not report['passed'], name
        reports[name] = report
    assert reports['opposing_current_cancellation']['total_current_rmse_ma_cm2'] < 1e-17
    assert set(reports) == set(spec['defective_controls'])
    return dict(valid=True, correct_reference_passed=True, controls_detected=reports,
                scope='Known current-level fixtures. wrong_memory_reset is an output fault proxy; causal state reset tested separately. Not a proof of emergent behavioral sensitivity.')
