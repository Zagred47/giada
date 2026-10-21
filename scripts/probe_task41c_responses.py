"""Opened-case ablations of frozen corrections; diagnostic only."""
import copy,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41c_frozen_confirmation as c
p,t,x=c.p,c.t,c.x
OUT=ROOT/'experiments/results/task41c_audited_v1'

def main():
    torch.set_num_threads(1);spec,cfg,calcium,base=c.contract(ROOT);cfg=x.calibration_config(spec,cfg);q=spec['current_quality']
    with zipfile.ZipFile(ROOT/'artifacts/task41c_primary_external/giada_task41c_result.zip') as z:
        data=dict(np.load(io.BytesIO(z.read('native_shadow_traces.npz'))));ref=dict(np.load(io.BytesIO(z.read('reference_traces.npz'))));meta=json.loads(z.read('cases.json'))
    ids=[34,48,108,122];sub={k:(v if k=='area_um2' else v[ids]) for k,v in data.items()};models=c.load_models(torch,ROOT,'cpu')
    variants={'B6':models['B6'],'P110':models['P110']}
    for j,label in enumerate(('u8','z7','z11')):
        m=copy.deepcopy(models['P110'])
        with torch.no_grad():m.precision_correction[...,j]=0
        variants['revert_'+label]=m
    rows=[];saved={}
    with torch.inference_mode():
        formula=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),sub,'cpu')
        for name,model in variants.items():
            pred=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,t.FAMILIES['all']),sub,'cpu')
            saved[name]=pred
            for case in (108,122):
                k,o=ids.index(case),ids.index(meta[case]['source_index'])
                for s,seed in enumerate(spec['model_seeds']):
                    for target,rr in [('saved_GPU',ref['autonomous__current'][:,ids]),('paired_CPU',formula['current'][:,s])]:
                        mm=p.evaluate_intervention(rr[:,o],rr[:,k],pred['current'][:,s,o],pred['current'][:,s,k],.1,q)
                        rows.append(dict(variant=name,case=case,seed=seed,reference=target,passed=mm['passed'],Nap=mm['channels'][4]))
            print('Ablation',name,flush=True)
    # Actual base/intervention voltage pairs: coefficient secants, no teacher endpoints as model input.
    probes=[]
    for case,seed in ((122,17),(108,43)):
        k,o=ids.index(case),ids.index(meta[case]['source_index']);s=spec['model_seeds'].index(seed)
        v=ref['autonomous__voltage'][:-1,ids];ca=ref['autonomous__calcium'][:-1,ids]
        vs=np.r_[v[:,o],v[:,k]].astype('float32');cs=np.r_[ca[:,o],ca[:,k]].astype('float32')
        tu,tz=p.coefficients(vs.astype('float64'),cs.astype('float64'),.1)
        du=tu[400:,8]-tu[:400,8];dv=vs[400:]-vs[:400];mask=abs(dv)>1e-4
        for name in ('B6','P110'):
            model=models[name]
            with torch.no_grad():ru,rz=model.raw(torch.tensor(vs)[None].expand(3,-1),torch.tensor(cs)[None].expand(3,-1))
            u=torch.sigmoid(ru).numpy()[s,:,8];pred=u[400:]-u[:400];err=pred-du
            probes.append(dict(case=case,seed=seed,model=name,voltage_range=[float(vs.min()),float(vs.max())],pair_delta_voltage_range=[float(dv.min()),float(dv.max())],
                u8_abs_rmse=float(np.sqrt(np.mean((u-tu[:,8])**2))),u8_pair_rmse=float(np.sqrt(np.mean(err**2))),
                u8_secant_error_rmse=float(np.sqrt(np.mean((err[mask]/dv[mask])**2))),secant_sample_count=int(mask.sum()),
                largest_pair_error_time_ms=float(np.argmax(abs(err))*.1)))
    result=dict(post_hoc=True,selection_eligible=False,rows=rows,coefficient_secants=probes,
        limits='Only two failed protocol pairs,allthree modelseeds. No global qualification of ablated models. CPU eager diagnostic;original GPU results unchanged.')
    p.save_json(OUT/'response_ablations.json',result)
    print(json.dumps(dict(rows=[{k:v for k,v in r.items() if k!='Nap'}|dict(nrmse=r['Nap']['nrmse']) for r in rows if r['reference']=='paired_CPU'],secants=probes),indent=2))
if __name__=='__main__':main()
