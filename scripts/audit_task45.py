"""Read-only45 artifact audit and frozen coefficient probes; no fitting or GPU run."""
import hashlib,io,json,os,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from research_memory.payload_store import open_readonly
from src.giada_teacher import task45_controlled_coverage as a

def main():
    import torch
    torch.set_num_threads(1)
    receipt=json.loads((ROOT/'artifacts/task45_kaggle_external/external_backup_receipt.json').read_text());archive=Path(receipt['local'])
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
    p,cfg,ca,base=a.spec(ROOT);out=ROOT/'experiments/results/task45_audited_v1';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        full=json.loads(z.read('results.json'));prefix=json.loads(z.read('prefix_results.json'));summary=receipt['report']
        horizon=[];geometry=[];failed=[];gate_failures=[];domain=[]
        for h in (100,500,1000):
            rows=[r for r in full+prefix if r['horizon_ms']==h]
            controls={(r['geometry'],r['case'],r['seed'],r['mode']):r for r in rows if r['arm']=='none'}
            for mode in ('imposed','autonomous'):
                pool=[r for r in rows if r['arm']=='all' and r['mode']==mode]
                valid=[r for r in pool if controls[(r['geometry'],r['case'],r['seed'],mode)]['compartment_passed'] and r['judgment_eligible']]
                bad=[r for r in valid if not(r['current_passed'] and r['compartment_passed'])]
                horizon.append(dict(horizon_ms=h,mode=mode,total=len(pool),reference_regime_invalid=len(pool)-len(valid),interpretable=len(valid),passed=len(valid)-len(bad),failed=len(bad),
                    all_interpretable_voltage_calcium_passed=all(r['compartment_passed'] for r in valid),max_voltage_error_mv=max(r['max_abs_voltage_error_mv'] for r in valid)))
                for geo in p['geometries']:
                    rr=[r for r in valid if r['geometry']==geo['id']]
                    geometry.append(dict(geometry=geo['id'],horizon_ms=h,mode=mode,interpretable=len(rr),passed=sum(r['current_passed'] and r['compartment_passed'] for r in rr)))
                if h==1000:
                    for r in bad:
                        failures=[dict(node=j,**c) for j,node in enumerate(r['per_node']) for c in node['channels'] if not c['passed']]
                        failed.append(dict(**{k:r[k] for k in ('geometry','case','seed','protocol','coupling','mode','voltage_rmse_mv','max_abs_voltage_error_mv')},failed_channels=failures))
                        gate_failures.extend(dict(geometry=r['geometry'],case=r['case'],seed=r['seed'],mode=mode,node=j,**node['gate']) for j,node in enumerate(r['per_node']) if not node['gate']['passed'])
        for geo in p['geometries']:
            with np.load(io.BytesIO(z.read('native_'+geo['id']+'.npz'))) as native:
                for i,coupling in enumerate(p['coupling']):
                    v=native['voltage'][:,i*3,0]
                    domain.append(dict(geometry=geo['id'],coupling=coupling,protocol='45_hyper_rebound',root_min_voltage_mv=float(v.min()),fraction_root_samples_below_minus100=float((v<-100).mean())))
        state=torch.load(io.BytesIO(z.read('frozen41r.pt')),map_location='cpu',weights_only=True)
        assert all(bool(torch.isfinite(v).all()) for v in state.values())
        hh=hashlib.sha256()
        for key,value in sorted(state.items()):hh.update(key.encode());hh.update(str(value.dtype).encode());hh.update(str(tuple(value.shape)).encode());hh.update(value.contiguous().numpy().tobytes())
        assert hh.hexdigest()==p['freeze']['logical_weights_sha256']
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
    baseline=a.n.load_models(torch,ROOT,'cpu')[0]['dual']
    assert a.n.weights_hash(baseline)==p['freeze']['baseline_weights_sha256']
    with np.load(ROOT/p['freeze']['heads_path']) as f:net=a.x.repair.materialize(torch,baseline,dict(f))
    assert a.n.weights_hash(net)==hh.hexdigest()
    vv=np.array([-140,-135,-130,-120,-110,-107,-106,-105,-100,-97,-90,-80,-65,-50],dtype='float32')
    v=torch.tensor(vv)[None].expand(3,-1);c=torch.full_like(v,1e-4)
    probes=[];roundoff=[]
    with torch.inference_mode():
        ulog,zlog=net.raw(v,c);ul=torch.sigmoid(ulog).double().numpy();zl=torch.sigmoid(zlog).double().numpy()
        bu,bz=baseline.raw(v,c);base_inf=torch.sigmoid(bu).double().numpy();base_z=torch.sigmoid(bz).double().numpy()
        features=net.parent.features(v).abs().sum(-1).numpy()
        for channel,localgate,index in [('Ih',0,10),('Nap_Et2',1,9)]:
            inf,tau=a.t._rates(channel,v.double(),c.double(),torch)
            initial_inf,_=a.t._rates(channel,torch.tensor([[-65.]],dtype=torch.float64),torch.tensor([[1e-4]],dtype=torch.float64),torch)
            if channel=='Nap_Et2':inf=inf[...,localgate];tau=tau[...,localgate];initial_inf=initial_inf[...,localgate]
            ur=inf.numpy();tr=tau.numpy();zr=-np.expm1(-.1/tr);initial=float(initial_inf[0,0])
            xf=torch.full_like(v,initial)
            snapshots={}
            for tick in range(1,10001):
                xf=xf+torch.tensor(zl[...,index],dtype=torch.float32)*(torch.tensor(ul[...,index],dtype=torch.float32)-xf)
                if tick in (1000,5000,10000):snapshots[tick]=xf.double().numpy().copy()
            for si,seed in enumerate(p['seeds']):
                for j,voltage in enumerate(vv):
                    trajectory={}
                    for ms in (100,500,1000):
                        exact=ur[si,j]+(initial-ur[si,j])*np.exp(-ms/tr[si,j])
                        oracle={}
                        for label,ui,zi in [('learned',ul[si,j,index],zl[si,j,index]),('oracle_inf',ur[si,j],zl[si,j,index]),('oracle_rate',ul[si,j,index],zr[si,j]),('oracle_both',ur[si,j],zr[si,j])]:
                            retention=1. if zi==0 else (0. if zi==1 else np.exp((ms/.1)*np.log1p(-zi)))
                            value=ui+(initial-ui)*retention;oracle[label]=dict(state=float(value),absolute_error=float(abs(value-exact)))
                        error=abs(snapshots[round(ms/.1)][si,j]-oracle['learned']['state'])
                        trajectory[str(ms)]=dict(reference_state=float(exact),oracles=oracle,fp32_vs_same_coeff_exact_error=float(error))
                        roundoff.append(float(error))
                    probability=float(zl[si,j,index]);tau_value=None if probability==0 else (0. if probability==1 else float(-.1/np.log1p(-probability)))
                    tau_status='infinite_zero_update' if probability==0 else ('instantaneous_saturated_update' if probability==1 else 'finite_positive')
                    bp=float(base_z[si,j,index]);base_tau=None if bp==0 else (0. if bp==1 else float(-.1/np.log1p(-bp)))
                    probes.append(dict(channel=channel,gate='m' if channel=='Ih' else 'h',seed=seed,voltage_mv=float(voltage),
                        reference_inf=float(ur[si,j]),learned_inf=float(ul[si,j,index]),reference_tau_ms=float(tr[si,j]),learned_tau_ms=tau_value,
                        learned_update_probability=probability,learned_tau_status=tau_status,learned_raw_inf_logit=float(ulog[si,j,index]),learned_raw_update_logit=float(zlog[si,j,index]),
                        baseline_inf=float(base_inf[si,j,index]),baseline_tau_ms=base_tau,baseline_update_probability=bp,
                        local_hat_feature_sum=float(features[si,j]),constant_voltage_trajectories=trajectory))
    history=[]
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:
        query='SELECT Nome,[Codice stabile],substr(Lezione,1,1800) AS lesson FROM lessons WHERE Nome LIKE ? OR Nome LIKE ? LIMIT 8'
        history=[dict(r) for r in con.execute(query,('%identifi%','%supervision%'))]
    result=dict(valid=True,task='45',artifact_sha256=receipt['remote']['sha256'],artifact_bytes=receipt['remote']['bytes'],all_parameters_and_traces_finite=summary['all_parameters_finite'] and summary['all_traces_finite'],
        weights_sha256=hh.hexdigest(),baseline_weights_sha256=a.n.weights_hash(baseline),thresholds_changed=False,training_performed=False,horizons=horizon,geometries=geometry,full_horizon_failed_cases=failed,
        full_horizon_gate_failures=gate_failures,hyperpolarization_domain=domain,
        frozen_coefficient_probes=probes,max_fp32_vs_same_coeff_exact_constant_clamp_error=max(roundoff),
        relevant_prior_lessons=history,
        cautions=['Coefficient probes are offline diagnostics, not new selected models or repaired45 results.',
            'Nominal local hat support is not the whole MLP support; zero local corrections do not mean the entire model ignores voltage.',
            'A constant-voltage probe is not a replacement for path replay or closed-loop validation.',
            'Historical rate-supervision successes do not establish the present mechanism without current-weight probes.'])
    assert a.n.weights_hash(net)==hh.hexdigest()
    write_json(out/'audit.json',result)
    print(json.dumps(dict(valid=True,full_horizon_cases=len(failed),max_fp32_roundoff=result['max_fp32_vs_same_coeff_exact_constant_clamp_error'],path=str(out/'audit.json'))))
    for r in probes:
        if r['voltage_mv'] in (-106.,-100.) and r['seed']==29:
            print(json.dumps({k:r[k] for k in ('channel','seed','voltage_mv','reference_inf','learned_inf','reference_tau_ms','learned_tau_ms','baseline_tau_ms','local_hat_feature_sum')}))
            print('1000ms',json.dumps(r['constant_voltage_trajectories']['1000']))
if __name__=='__main__':main()
