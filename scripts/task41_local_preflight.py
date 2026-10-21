"""Old opened fixtures only: check authentic current phase before fresh access."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41_independent_confirmation as c
def main():
    torch.set_num_threads(1);spec,cfg,calcium,base=c.t.contract(ROOT)
    with np.load(ROOT/'experiments/results/task35_native_shadow_traces.npz') as z:data={k:z[k] for k in z.files}
    phase_checks={}
    for vp in (0,1):
        for cp in (0,1):
            vv=data['voltage'][:,vp:len(data['voltage'][0])-1+vp].T
            cc=data['calcium'][:,cp:len(data['calcium'][0])-1+cp].T
            u,z=c.p.coefficients(vv,cc,.1);g=data['gates'][:,:-1].transpose(1,0,2)
            predicted=g+z*(u-g);truth=data['gates'][:,1:].transpose(1,0,2)
            phase_checks[str((vp,cp))]=dict(rmse=float(np.sqrt(np.mean((predicted-truth)**2))),max_abs=float(np.max(np.abs(predicted-truth))))
    print('NATIVE_ONE_STEP_PHASE '+json.dumps(phase_checks),flush=True)
    imposed={k:data[k].T for k in ('voltage','calcium')}
    with torch.inference_mode():ref=c.t.rollout(torch,c.t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,'cpu',imposed)
    _,target=c.p.readout(data['voltage'][:,:-1].T,data['gates'][:,1:].transpose(1,0,2),data['multipliers'],c.ionic.REVERSALS)
    rows=[c.p.evaluate_currents(target[:,k],ref['current'][:,0,k],.1,spec['current_quality'],data['gates'][k,1:],ref['gates'][1:,0,k],data['multipliers'][k]==0) for k in range(32)]
    aligned=data['gates'][:,:-1].transpose(1,0,2).copy();aligned[...,16]=data['gates'][:,1:,16].T
    _,current=c.p.readout(data['voltage'][:,:-1].T,aligned,data['multipliers'],c.ionic.REVERSALS)
    corrected=[c.p.evaluate_currents(current[:,k],ref['current'][:,0,k],.1,spec['current_quality'],aligned[:,k],ref['gates'][1:,0,k],data['multipliers'][k]==0) for k in range(32)]
    result=dict(valid=all(r['passed'] for r in corrected),passed=sum(r['passed'] for r in corrected),total=32,
        fresh_confirmation_accessed=False,old_opened_fixtures_only=True,one_step_phase_checks=phase_checks,
        unaligned_endpoint_cases=rows,unaligned_endpoint_passed=sum(r['passed'] for r in rows),cases=corrected,
        phase='Native voltage-dependent gates at t; native calcium-dependent SK at t+dt, matching unchanged split operator')
    c.p.save_json(ROOT/'experiments/results/task41_protocol_v1/old_native_phase_preflight.json',result)
    print(json.dumps(dict(valid=result['valid'],passed=result['passed'])),flush=True)
if __name__=='__main__':main()
