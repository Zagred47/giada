"""Before training: opened baseline reproduction and full selection data floor."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,sys
from pathlib import Path
import numpy as np
import torch
pf=json.loads(Path('/content/task41b_gpu_preflight.json').read_text());root=Path(pf['preflight_root']);sys.path.insert(0,str(root))
from src.giada_teacher import task41b_local_precision as c
from src.giada_teacher import task41b_evaluation as e
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec,cfg,calcium,base=c.contract(root);cfg=c.prior.calibration_config(json.loads((root/c.prior.CONFIG).read_text()),cfg)
parent=c.prior.load_b6(torch,root,'cuda');candidate=c.make_model(torch,parent,c.arms()[0],spec['seeds'],'cuda')
data,meta,st,old=e.load_opened(root,spec);result={}
with torch.inference_mode():
    for mode in ('imposed','autonomous'):
        imposed={k:data[k].T for k in ('voltage','calcium')} if mode=='imposed' else None
        refs=c.t.rollout(torch,c.t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,'cuda',imposed,True)
        ref={k:v[:,0] for k,v in refs.items()};ref.update({k:data[k].T for k in ('voltage','calcium')})
        measured={}
        for name,m in [('B6',parent),('zero_P000',candidate)]:
            a=c.t.rollout(torch,c.t.make_core(torch,cfg,calcium,base,float(data['area_um2']),m,c.t.FAMILIES['all']),data,'cuda',imposed,True)
            mm=c.t.metrics(a,ref,data,meta,spec['quality'],cfg['gates'],mode)
            assert [r['passed'] for r in mm['cases']]==[r['passed'] for r in old['B6/'+mode]['cases']],name
            measured[name]=mm['passed_cases']
        result[mode]=measured
    dev,_=c.development(spec['development_seed']);v,ca=dev['voltage'],dev['calcium'];u,z=c.p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    g=dev['initial'].astype('float32');gg=[]
    for tick in range(len(v)):g=g+z[tick].astype('float32')*(u[tick].astype('float32')-g);gg.append(g.copy())
    gg=np.stack(gg);target=c.p.readout(v,dev['target_gates'],dev['multipliers'],c.ionic.REVERSALS)[1];ii=c.p.readout(v,gg,dev['multipliers'],c.ionic.REVERSALS)[1]
    floor=[c.p.evaluate_currents(target[:,k],ii[:,k],.1,spec['quality'],dev['target_gates'][:,k],gg[:,k]) for k in range(v.shape[1])]
    assert all(r['passed'] for r in floor),'New development FP32 floor failed'
    shadow=c.shadow_evaluate(torch,[parent],dev,spec['quality'],'cuda')
    gpu=c.j.i.old.Scanner(torch,[parent],v.shape[1],32,'cuda').run(v,ca,dev['initial'].astype('float32'))[0]
    mismatches=[]
    for s in range(3):
        current=c.p.readout(v,gpu[:,s],dev['multipliers'],c.ionic.REVERSALS)[1]
        for k in range(v.shape[1]):
            actual=c.p.evaluate_currents(target[:,k],current[:,k],.1,spec['quality'],dev['target_gates'][:,k],gpu[:,s,k])
            if actual['passed']!=shadow[s][k]['passed']:mismatches.append([s,k])
    assert not mismatches,mismatches
    sr=e.stress(torch,dict(B6=parent),st,spec,'cuda');assert sum(r['metrics']['passed'] for r in sr['rows'])==33
report=dict(valid=True,opened_baseline=result,new_development_formula_floor=floor,shadow_GPU_pass_flags_equal=True,
    stress_B6_passed=33,stress_formula_floor_passed=True,training_performed=False,source_sha256=c.p.digest(root/'src/giada_teacher/task41b_evaluation.py'))
Path('/content/task41b_evaluation_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='new_development_formula_floor'}),flush=True)
