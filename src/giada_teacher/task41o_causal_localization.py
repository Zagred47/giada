"""Paired causal surgery on frozen ionic dynamics; no oracle is a candidate."""
import hashlib,io,json,os,zipfile
from pathlib import Path
import numpy as np
from . import task41n_evaluation_recovery as recovery
n=recovery.n
ARMS=('learned32','learned64','oracle_u','oracle_z','oracle_both','teacher_previous','reset_10ms','reset_100ms')

def recurrence(u,z,initial,truth,arm):
    """Identical causal affine map; only the registered diagnostic factor changes."""
    state=initial.astype(np.float32 if arm=='learned32' else np.float64).copy();values=[]
    for tick in range(len(u)):
        if arm=='teacher_previous':state=initial if tick==0 else truth[tick-1]
        if arm in ('reset_10ms','reset_100ms'):
            period=100 if arm=='reset_10ms' else 1000
            if tick and tick%period==0:state=truth[tick-1].copy()
        state=state+z[tick]*(u[tick]-state);values.append(state.copy())
    return np.stack(values)

def learned_coefficients(torch,net,v,ca,device):
    """Keep original GEMM batch/seed shapes; no flattening-induced change."""
    vv=torch.tensor(v,device=device);cc=torch.tensor(ca,device=device)
    a=vv[0][None].expand(3,-1).contiguous();b=cc[0][None].expand(3,-1).contiguous()
    for _ in range(3):net.raw(a,b)
    if device=='cuda':
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ru,rz=net.raw(a,b);u,z=torch.sigmoid(ru),torch.sigmoid(rz)
    us=[];zs=[]
    for tick in range(len(v)):
        a.copy_(vv[tick][None]);b.copy_(cc[tick][None])
        if device=='cuda':graph.replay()
        else:
            ru,rz=net.raw(a,b);u,z=torch.sigmoid(ru),torch.sigmoid(rz)
        us.append(u.cpu().numpy().copy());zs.append(z.cpu().numpy().copy())
        if tick%2000==0:print(f'[GIADA41o] coefficient replay {tick}/{len(v)}',flush=True)
    return np.stack(us),np.stack(zs)

def run(torch,root,original,archive,out,device='cuda'):
    out.mkdir(exist_ok=False);spec=json.loads((root/'experiments/task41o_preregistration_v1.json').read_text())
    assert recovery.sha(archive)==spec['recovery_sha256']
    _,prior,_,floors=recovery.verify_source(root,original,out.parent/'verified_original_41o')
    assert spec['quality']==prior['quality'] and spec['training_updates']==0
    os.environ['GIADA_TASK41N_FROZEN']=str(out.parent/'verified_original_41o/frozen_source.zip.bin')
    nets,_=n.load_models(torch,root,device);net=nets['dual'];before=n.weights_hash(net)
    assert all(torch.isfinite(value).all() for value in net.parameters())
    hardware=n.parity(torch,nets,device,prior)
    assert hardware['eligible']==['dual','donor_pruned','shared_features']
    n.p.save_json(out/'gpu_preflight.json',hardware)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        inputs=z.read('fresh_stress_inputs.npz');baseline=z.read('stress_dual.npz')
        with np.load(io.BytesIO(inputs)) as a:data=dict(a)
        with np.load(io.BytesIO(baseline)) as a:saved=a['gates']
        archived=json.loads(z.read('fresh_stress_results.json'))['dual']
    v,ca,truth=data['voltage'],data['calcium'],data['target_gates'];initial=data['initial']
    # Reproduce all generated inputs, not just a matching seed label.
    meta=n.stress_data(prior)
    assert np.array_equal(v,np.stack([r['v'] for r in meta],1).astype('float32'))
    assert np.array_equal(ca,np.stack([r['ca'] for r in meta],1).astype('float32'))
    u,z=learned_coefficients(torch,net,v,ca,device)
    tu,tz=n.p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    target=n.p.readout(v,truth,data['multipliers'],n.p.ionic.REVERSALS)[1]
    results={};coefficient_errors=[];baseline_equal=True
    for arm in ARMS:
        measured=[]
        for s,seed in enumerate(prior['seeds']):
            uu=tu if arm in ('oracle_u','oracle_both') else u[:,s]
            zz=tz if arm in ('oracle_z','oracle_both') else z[:,s]
            if arm!='learned32':uu,zz=uu.astype('float64'),zz.astype('float64')
            pred=recurrence(uu,zz,initial,truth,arm)
            assert np.isfinite(pred).all()
            if arm=='learned32':
                equal=n.array_bits_equal(pred,saved[:,s]);baseline_equal &= equal
                assert equal,('original FP32 recurrence parity failed',seed,float(abs(pred-saved[:,s]).max()))
            if arm=='oracle_both':assert np.array_equal(pred,truth)
            currents=n.p.readout(v,pred,data['multipliers'],n.p.ionic.REVERSALS)[1]
            for k,row in enumerate(meta):
                metrics=n.p.evaluate_currents(target[:,k],currents[:,k],.1,spec['quality'],truth[:,k],pred[:,k])
                if arm=='learned32':assert metrics['passed']==archived['rows'][s*len(meta)+k]['metrics']['passed']
                measured.append(dict(model_seed=seed,case=k,**{key:row[key] for key in ('family','input_seed','history','panel')},metrics=metrics))
        results[arm]=dict(passed_count=sum(r['metrics']['passed'] for r in measured),total=len(measured),rows=measured,selection_eligible=False)
        n.p.save_json(out/'causal_results.json',results)
        print(f'[GIADA41o] {arm}: {results[arm]["passed_count"]}/72 (diagnostic)',flush=True)
    for j,name in enumerate(n.p.ionic.CHANNELS):
        lo,hi=n.p.OFFSETS[j:j+2]
        coefficient_errors.append(dict(channel=name,equilibrium_max_abs=float(abs(u[...,lo:hi]-tu[:,None,...,lo:hi]).max()),
            update_max_abs=float(abs(z[...,lo:hi]-tz[:,None,...,lo:hi]).max())))
    np.savez_compressed(out/'learned_coefficients.npz',u=u,z=z)
    baseline_fail={(r['model_seed'],r['case']) for r in results['learned32']['rows'] if not r['metrics']['passed']}
    contrasts={}
    for arm,value in results.items():
        failed={(r['model_seed'],r['case']) for r in value['rows'] if not r['metrics']['passed']}
        contrasts[arm]=dict(passed=value['passed_count'],rescued=sorted(baseline_fail-failed),regressions=sorted(failed-baseline_fail))
    assert n.weights_hash(net)==before
    report=dict(valid=True,complete=True,task='41o',decision='CAUSAL_LOCALIZATION_COMPLETE_NOT_MODEL_REPAIR',
        contrasts=contrasts,coefficient_errors=coefficient_errors,baseline_bitwise_reproduced=baseline_equal,
        original_fp32_floor_failures_retained=[218,234],original_floor_report=floors,
        weights_unchanged=True,all_parameters_finite=True,thresholds_changed=False,abstraction_changed=False,
        training_updates_performed=0,selection_recomputed=False,task42_authorized=False,speedup_claim=False,
        limits=spec['limits'],progress_is_diagnostic_only=True)
    n.p.save_json(out/'final_report.json',report);return report
