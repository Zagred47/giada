"""Immutable Task41 trace audit and post-hoc, non-selectable mechanism probes.

No training, altered quality threshold, or rewriting of the original result.
CPU coefficient/state interventions are diagnostic, not a new confirmation.
"""
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher import task41_independent_confirmation as x
from audit_task40 import reasons
from audit_task40b import qualify, compare

p, t = x.p, x.t
OUT = ROOT / 'experiments/results/task41_audited_v1'
ARCHIVE = ROOT / 'artifacts/task41_retry_external/giada_task41_result.zip'
SHA = '3d02cdb451a73fc033fe896d84d822ca1552d9d3b48343db65db6b1dccc3ee92'
METRIC_REPLAY_DIFFERENCES = []


def compare_portable(a, b):
    """NumPy-version readout rounding tolerance only; pass flags/timing exact."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            if k.endswith('_ms') or k == 'passed':
                assert a[k] == b[k], (k, a[k], b[k])
            else:
                compare_portable(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x_, y in zip(a, b):compare_portable(x_, y)
    elif isinstance(a, float):
        METRIC_REPLAY_DIFFERENCES.append(abs(a-b))
        # Current readout includes FP32 powers whose last bits differ between
        # NumPy1.26 CPU and NumPy2.1 Colab. Timing and verdicts above stay exact.
        assert np.isclose(a, b, rtol=1e-4, atol=1e-6), (a, b)
    else:assert a == b, (a, b)


def save(name, data):
    p.save_json(OUT / name, data)


def segments(mask, dt=.1):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1)
    return [[float(g[0]*dt), float(g[-1]*dt), len(g)] for g in groups]


def event_detail(ref, pred, q):
    rr, pp = np.abs(ref-ref[0]), np.abs(pred-ref[0])
    amp = float(rr.max()); threshold = .1*amp
    ra, pa = np.flatnonzero(rr >= threshold), np.flatnonzero(pp >= threshold)
    end = int(ra[-1]) if len(ra) else 0
    slope = float((rr[min(end+1, len(rr)-1)]-rr[max(end-1, 0)])/.2)
    return dict(metrics=p.event_metrics(ref, pred, .1, q['current_scale_floor_ma_cm2']),
        reference_base=float(ref[0]), reference_transient_amplitude=amp,
        threshold=threshold, reference_active_intervals=segments(rr >= threshold),
        predicted_active_intervals=segments(pp >= threshold),
        reference_offset_ms=float(end*.1), predicted_offset_ms=float(pa[-1]*.1) if len(pa) else None,
        absolute_residual_at_reference_offset=float(pp[end]-rr[end]),
        reference_threshold_slope_per_ms=slope,
        linearized_offset_shift_ms=float(-(pp[end]-rr[end])/slope) if slope else None,
        threshold_near_margin_min=float(np.min(np.abs(rr-threshold))),
        prediction_max_abs_error=float(np.max(np.abs(pred-ref))))


def recurrence(u, z, initial, dtype, style='delta'):
    u, z, state = np.asarray(u, dtype=dtype), np.asarray(z, dtype=dtype), np.asarray(initial, dtype=dtype).copy()
    result = np.empty(u.shape, dtype=dtype)
    for k in range(len(u)):
        state = state+z[k]*(u[k]-state) if style == 'delta' else (1-z[k])*state+z[k]*u[k]
        result[k] = state
    return result


def memory_patterns():
    queries = {
        'precision_and_timing_lessons': '''SELECT "Codice stabile" code,Nome,Lezione,Limiti FROM lessons
        WHERE "Codice stabile" LIKE '%giada%' AND
        (lower(Lezione) LIKE '%precision%' OR lower(Lezione) LIKE '%timing%' OR lower(Lezione) LIKE '%equilibrium%')''',
        'recent_mechanisms': '''SELECT "Codice stabile" code,Nome,Risultato,Limitazioni FROM findings
        WHERE ("Codice stabile" LIKE '%task39j%' OR "Codice stabile" LIKE '%task40b%'
        OR "Codice stabile" LIKE '%task39g%' OR "Codice stabile" LIKE '%task39h%')''',
        'task41_claims': '''SELECT "Codice stabile" code,Nome,Enunciato,Stato FROM claims
        WHERE "Codice stabile" LIKE '%task41%' OR "Codice stabile" LIKE '%task40-result%' '''}
    con = sqlite3.connect('file:'+str(ROOT/'research_memory/data/research_memory.sqlite')+'?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    results = {name: dict(sql=sql, rows=[dict(r) for r in con.execute(sql)]) for name, sql in queries.items()}
    con.close(); save('memory_patterns.json', results)


def main():
    OUT.mkdir(exist_ok=True); torch.set_num_threads(1)
    assert p.digest(ARCHIVE) == SHA
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        read = lambda n: json.loads(archive.read(n))
        arrays = lambda n: dict(np.load(io.BytesIO(archive.read(n))))
        spec = read('preregistration.json'); q = spec['current_quality']
        assert spec == json.loads((ROOT/x.CONFIG).read_text())
        x.contract(ROOT)
        original = read('final_report.json'); reports = read('causal_results.json')
        stress_rows = read('fresh_stress_results.json'); cases = read('cases.json')
        pred, ref = arrays('candidate_traces.npz'), arrays('reference_traces.npz')
        stress = arrays('fresh_stress_traces.npz')
        model = x.load_b6(torch, ROOT, 'cpu')
        frozen = torch.load(io.BytesIO(archive.read('B6_frozen.pt')), map_location='cpu', weights_only=True)
        assert all(torch.equal(v, frozen[k]) for k, v in model.state_dict().items())
        assert all(torch.isfinite(v).all() for v in frozen.values())
        native = read('native_interface_report.json'); floor = read('gpu_formula_floor.json')
        assert native['valid'] and read('native_current_phase_floor.json')['valid']
        assert all(v['valid'] for v in floor.values())
        cost = read('cost.json')
    seeds = spec['model_seeds']; failures = []; summary = {}; metrics_recomputed = 0
    for label, report in reports.items():
        name, mode = label.split('/'); prefix = label.replace('/', '__')+'__'
        for row in report['cases']:
            c = row['case']['id']; s = seeds.index(row['seed'])
            current = p.evaluate_currents(ref[mode+'__current'][:,c], pred[prefix+'current'][:,s,c], .1, q,
                ref[mode+'__gates'][1:,c], pred[prefix+'gates'][1:,s,c],
                np.array(spec['panels'][row['case']['panel']]) == 0)
            # Block flags can differ from panel: source row metrics are authoritative
            # for blocked conductance and are independently requalified below.
            for actual, stored in zip(current['channels'], row['current']['channels']):
                for key in ('nrmse','nmax','signed_charge_normalized','absolute_charge_normalized','event'):
                    compare(actual[key], stored[key])
            qualify(row['current'],q); metrics_recomputed += 1
            if not row['passed']:
                failures.append(dict(label=label,case=c,seed=row['seed'],metadata=row['case'],
                    metrics=row,channel_reasons={r['channel']:reasons(r,q) for r in row['current']['channels'] if not r['passed']}))
        for row in report['interventions']:
            c,s=row['case'],seeds.index(row['seed']);o=cases[c]['source_index']
            actual=p.evaluate_intervention(ref[mode+'__current'][:,o],ref[mode+'__current'][:,c],
                pred[prefix+'current'][:,s,o],pred[prefix+'current'][:,s,c],.1,q)
            compare(actual,row['metrics']);qualify(actual,q);metrics_recomputed += 1
        summary[label]=dict(trajectory_passed=report['passed_cases'],trajectory_total=report['total'],
            response_passed=sum(r['metrics']['passed'] for r in report['interventions']),response_total=len(report['interventions']),
            worst_voltage_rmse_mv=max(r['voltage_rmse_mv'] for r in report['cases']),
            worst_calcium_rmse_mM=max(r['calcium_rmse_mM'] for r in report['cases']),
            spike_count_failures=sum(not r['spike_count_equal'] for r in report['cases']),
            domain_exits=sum(r['voltage_domain_exits']+r['calcium_domain_exits'] for r in report['cases']))
    save('trajectory_failures.json',failures)
    timing=[]; channel=x.ionic.CHANNELS.index('NaTs2_t')
    related=[c['id'] for c in cases if c['source_index']==1]
    for mode in ('imposed','autonomous'):
        for c in related:
            for s,seed in enumerate(seeds):
                rr=ref[mode+'__current'][:,c,channel];pp=pred['B6__'+mode+'__current'][:,s,c,channel]
                timing.append(dict(mode=mode,case=c,seed=seed,intervention=cases[c]['intervention'],
                    **event_detail(rr,pp,q)))
    save('natiming_neighborhood.json',timing)
    print('Immutable trajectory and intervention metrics reproduced',flush=True)

    v,ca,initial=stress['voltage'],stress['calcium'],stress['initial'];n=v.shape[1]
    truth=stress['target_gates'];mult=stress['multipliers'];gold=p.readout(v,truth,mult,x.ionic.REVERSALS)[1]
    source=x.stress_design(spec); all_stress=[]
    for s,seed in enumerate(seeds):
        current=p.readout(v,stress['predicted_gates'][:,s],mult,x.ionic.REVERSALS)[1]
        for k,row in enumerate(source):
            mm=p.evaluate_currents(gold[:,k],current[:,k],.1,q,truth[:,k],stress['predicted_gates'][:,s,k])
            stored=stress_rows[s*n+k];compare_portable(mm,stored['metrics']);qualify(mm,q)
            if not mm['passed']:
                all_stress.append(dict(index=k,**stored,channel_reasons={c['channel']:reasons(c,q) for c in mm['channels'] if not c['passed']}))
    save('stress_failures.json',all_stress)
    # Two constant input regimes per history. Cache the neural coefficient function
    # at unique inputs only, then retain every .1ms serial update.
    points,inverse=np.unique(np.stack((v,ca),-1).reshape(-1,2),axis=0,return_inverse=True)
    with torch.inference_mode():
        vv=torch.tensor(points[:,0])[None].expand(3,-1);cc=torch.tensor(points[:,1])[None].expand(3,-1)
        ru,rz=model.raw(vv,cc)
        lu,lz=torch.sigmoid(ru).numpy(),torch.sigmoid(rz).numpy()
    lu=lu[:,inverse].reshape(3,len(v),n,18).transpose(1,0,2,3)
    lz=lz[:,inverse].reshape(3,len(v),n,18).transpose(1,0,2,3)
    tu,tz=p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    variants={
        'learned32':recurrence(lu,lz,np.broadcast_to(initial,(3,n,18)),np.float32),
        'learned32coeff_state64':recurrence(lu,lz,np.broadcast_to(initial,(3,n,18)),np.float64),
        'oracle_u':recurrence(np.broadcast_to(tu[:,None],lu.shape),lz,np.broadcast_to(initial,(3,n,18)),np.float64),
        'oracle_z':recurrence(lu,np.broadcast_to(tz[:,None],lz.shape),np.broadcast_to(initial,(3,n,18)),np.float64),
        'formula_coeff_cast32':recurrence(tu,tz,initial,np.float32),
        'formula_coeff_cast32_convex':recurrence(tu,tz,initial,np.float32,'convex'),
        'formula64':recurrence(tu,tz,initial,np.float64)}
    targeted_u,targeted_z=lu.astype('float64'),lz.astype('float64')
    targeted_u[...,8]=tu[:,None,:,8]
    for gate in (7,11):targeted_z[...,gate]=tz[:,None,:,gate]
    variants['three_local_coefficient_oracles']=recurrence(targeted_u,targeted_z,np.broadcast_to(initial,(3,n,18)),np.float64)
    # Actual Torch analytical coefficient evaluator, independent from NumPy labels.
    with torch.inference_mode():
        uu=[];zz=[]
        for name in x.ionic.CHANNELS:
            a,b=t._rates(name,torch.tensor(points[:,0]),torch.tensor(points[:,1]),torch)
            if a.ndim==1:a,b=a[:,None],b[:,None]
            uu.append(a);zz.append(-torch.expm1(-.1/b))
        a=torch.cat(uu,-1).numpy()[inverse].reshape(len(v),n,18)
        b=torch.cat(zz,-1).numpy()[inverse].reshape(len(v),n,18)
    variants['formula_torch32']=recurrence(a,b,initial,np.float32)
    replay_error=float(np.max(np.abs(variants['learned32']-stress['predicted_gates'])))
    probes={}; probe_traces={}
    for label,gates in variants.items():
        if gates.ndim==3:gates=np.broadcast_to(gates[:,None],(len(v),3,n,18))
        rows=[]
        for s,seed in enumerate(seeds):
            current=p.readout(v,gates[:,s],mult,x.ionic.REVERSALS)[1]
            for k,row in enumerate(source):
                mm=p.evaluate_currents(gold[:,k],current[:,k],.1,q,truth[:,k],gates[:,s,k])
                rows.append(dict(model_seed=seed,input_seed=row['seed'],family=row['family'],case=row['case'],metrics=mm))
                if any(f['index']==k and f['model_seed']==seed for f in all_stress):
                    for c in all_stress:
                        if c['index']==k and c['model_seed']==seed:
                            ch=next(a['channel'] for a in c['metrics']['channels'] if not a['passed']);ci=x.ionic.CHANNELS.index(ch)
                            probe_traces[f'{label}/{seed}/{k}/{ch}']=dict(**event_detail(gold[:,k,ci],current[:,k,ci],q))
        probes[label]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows),rows=rows)
    save('stress_counterfactuals.json',dict(post_hoc=True,selection_eligible=False,training_performed=False,
        CPU_learned_replay_max_gate_error=replay_error,CPU_version=torch.__version__,
        original_GPU_results_authoritative=True,variants=probes,failed_trace_event_details=probe_traces))
    coeff=[]
    for row in all_stress:
        k=row['index'];s=seeds.index(row['model_seed']);ci=x.ionic.CHANNELS.index(next(iter(row['channel_reasons'])))
        gs=list(range(p.OFFSETS[ci],p.OFFSETS[ci+1]));parts=[]
        for tick in (0,len(v)-1):
            parts.append(dict(tick=tick,v=float(v[tick,k]),ca=float(ca[tick,k]),gates=[dict(gate=int(g),
                true_u=float(tu[tick,k,g]),learned_u=float(lu[tick,s,k,g]),
                true_z=float(tz[tick,k,g]),learned_z=float(lz[tick,s,k,g]),
                z_relative_error=float(lz[tick,s,k,g]/tz[tick,k,g]-1),
                true_tau_ms=float(-.1/np.log1p(-tz[tick,k,g])),
                learned_tau_ms=float(-.1/np.log1p(-float(lz[tick,s,k,g]))),
                original_final_gate=float(stress['predicted_gates'][-1,s,k,g]),true_final_gate=float(truth[-1,k,g])) for g in gs]))
        coeff.append(dict(index=k,seed=row['model_seed'],channel=x.ionic.CHANNELS[ci],regimes=parts))
    save('coefficient_probes.json',coeff)
    cost_summary=[dict(scope=r['scope'],batch=r['batch_per_seed'],actual_sites=r['actual_sites'],
        B6_vs_formula=r['B6_vs_formula'],B6_vs_LUT=r['B6_vs_LUT'],finite=r['finite'],
        median_wall_ms={k:float(np.median(v['wall_ms'])) for k,v in r['samples'].items()},
        warmup_sensitivity_median_after_first5={k:float(np.median(v['wall_ms'][5:])) for k,v in r['samples'].items()}) for r in cost]
    save('cost_audit.json',dict(results=cost_summary,quality_qualified_speedup=False,
        limitation='T4 only; post-hoc warmup sensitivity not a replacement benchmark. All saved raw samples retained in original archive.'))
    memory_patterns()
    assert len(failures)==2 and len(all_stress)==3
    assert all(set(r for v in f['channel_reasons'].values() for r in v)=={'offset_error_ms'} for f in failures+all_stress)
    assert replay_error<=1e-6
    audit=dict(valid=True,archive_sha256=SHA,checkpoint_bitwise_equal_to_frozen_B6=True,
        original_result=original,summary=summary,metrics_recomputed=metrics_recomputed+len(stress_rows),
        parameters_and_saved_outputs_finite=all(np.isfinite(a).all() for a in list(pred.values())+list(stress.values())),
        trajectory_failed_case_seed_count_per_mode=1,distinct_failed_trajectory_case_seed_count=1,
        stress_failed_case_seed_count=len(all_stress),stress_passed=33,stress_total=36,
        all_failures_are_offset_timing=True,CPU_learned_replay_max_gate_error=replay_error,
        stress_metric_recompute_max_scalar_difference=max(METRIC_REPLAY_DIFFERENCES),
        stress_recomputed_pass_flags_and_event_times_identical=True,
        CPU_numpy_version=np.__version__,
        stress_probe_passes={k:v['passed'] for k,v in probes.items()},thresholds_changed=False,
        original_scientific_NO_GO_preserved=True,task42_authorized=False,training_performed=False,
        next_experiment_not_launched=True)
    save('audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('summary','original_result')},indent=2),flush=True)


if __name__=='__main__':main()
