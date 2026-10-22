"""Exercise new evaluation paths using synthetic fixtures, never opened cases."""
import json
from pathlib import Path
import subprocess
import sys
request=json.loads(Path('/content/task41e_preflight_request.json').read_text())
roots=sorted(Path('/content').glob('giada41e_preflight_*'),key=lambda p:p.stat().st_mtime)
root=roots[-1]
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==request['revision']
code=r'''
import os,sys,json,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import numpy as np
import torch
from src.giada_teacher import task41e_transfer_model as m
from src.giada_teacher import task41e_transfer_data as data
from src.giada_teacher import task41e_transfer_training as training
from src.giada_teacher import task41e_transfer_evaluation as evaluation
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
spec,cfg,calcium,base=m.contract(root)
out=root/('evaluation_smoke_'+str(time.time_ns()));out.mkdir()
print('[GIADA Task41e preflight] full FIT history materialization',flush=True)
fb,supports,raw,labels=data.prepare_feedback(torch,spec,cfg,calcium,base,'cuda')
assert len(raw['v'])==2048 and all(len(v)==1024 for v in supports.values())
assert all(bool(torch.isfinite(v).all()) for v in fb.values())
print('[GIADA Task41e preflight] synthetic evaluator and full precision branches',flush=True)
n=123;steps=9
v=np.full((n,steps),-45.);ca=np.full((n,steps),1e-4)
g,_=m.p.coefficients(v[:,0],ca[:,0],.1)
fixture=dict(voltage=v,calcium=ca,gates=np.repeat(g[:,None],steps,axis=1),
             injection=np.full((n,steps),.004),base_g_us=np.zeros((n,steps,4)),
             multipliers=np.ones((n,11)),area_um2=100*np.pi)
fixture['multipliers'][108,6]=0;fixture['multipliers'][122,9]=0
meta=[dict(source_index=k,intervention='none') for k in range(n)]
meta[108]=dict(source_index=0,intervention='block_K_Pst')
meta[122]=dict(source_index=1,intervention='block_SK_E2')
with torch.inference_mode():
    controls=m.frozen_controls(torch,root,spec,'cuda')
    equivalence=evaluation.fusion_equivalence(torch,controls,fixture,spec,cfg,calcium,base,'cuda')
    coupling=evaluation.coupling_probe(torch,root,out/'coupling',controls,fixture,meta,spec,cfg,calcium,base,'cuda')
    development=m.d.development(901003,2)
    baseline=m.c.shadow_evaluate(torch,[controls['P110']],development,spec['quality'],'cuda')
    candidate=m.c.shadow_evaluate(torch,[controls['D_fused']],development,spec['quality'],'cuda')
    ranks=[training.ranks(a,b,0.,0.,spec['quality']) for a,b in zip(candidate,baseline)]
    k=m.c.ionic.CHANNELS.index('K_Pst')
    gates=list(range(int(m.c.ionic.OFFSETS[k]),int(m.c.ionic.OFFSETS[k+1])))
    oracle=evaluation.GateOracle(torch,evaluation.clone_model(torch,controls['P110']),gates,False)
    oracle_rows=m.c.shadow_evaluate(torch,[oracle],development,spec['quality'],'cuda')
assert len(ranks)==3 and len(oracle_rows)==3
report=dict(valid=True,GPU=torch.cuda.get_device_name(),full_FIT_history_states=len(raw['v']),
            full_FIT_history_ticks=int(raw['time_tick'].max()),equal_support_counts={k:len(v) for k,v in supports.items()},
            history_voltage_range_mv=[float(raw['v'].min()),float(raw['v'].max())],
            history_calcium_range_mM=[float(raw['ca'].min()),float(raw['ca'].max())],
            fusion_equivalence=equivalence,coupling_precision_settings=len(coupling['rows']),
            jacobian_and_oracle_paths_executed=True,selection_interface_valid=True,
            opened_regressions_accessed=False,scientific_training_performed=False,
            protocol_sha256=m.p.digest(root/m.CONFIG),revision=sys.argv[2])
Path('/content/task41e_gpu_evaluation_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
'''
result=subprocess.run([sys.executable,'-u','-c',code,str(root),request['revision']],cwd=root,capture_output=True,text=True)
Path('/content/task41e_evaluation_smoke_process.log').write_text(result.stdout+'\n'+result.stderr)
print(result.stdout,flush=True)
if result.returncode:raise RuntimeError(result.stderr[-18000:])
