"""Standalone CUDA preflight; no scientific candidate is retained."""
import os,json,subprocess,sys,tempfile,copy
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task41j_operator_reset as j

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available()
spec,cfg,calcium,base=j.contract(ROOT)
with tempfile.TemporaryDirectory() as folder:result=j.preflight(torch,ROOT,Path(folder),'cuda',spec)
parent=j.controls(torch,ROOT,'cuda')['H_nominee']
models={a['name']:j.model_class(torch)(parent,a).to('cuda') for a in j.arms()}
data,_=j.atomic_pool(torch,parent,spec['new_seeds']['fit'],16,'cuda');ids=torch.arange(4,device='cuda')
assert float(j.coefficient_loss(torch,models['J_chain_joint'],data,ids,True).abs().max())==0.
opts={k:torch.optim.Adam(j.parameters(m),lr=torch.tensor(spec['lr'],device='cuda'),capturable=True) for k,m in models.items()}
j.step(torch,models,opts,data,ids)
graph=j.capture(torch,models,opts,data,ids)
saved={k:copy.deepcopy(o.state_dict()) for k,o in opts.items()};copies=copy.deepcopy(models)
copyopts={k:torch.optim.Adam(j.parameters(m),lr=torch.tensor(spec['lr'],device='cuda'),capturable=True) for k,m in copies.items()}
for k,o in copyopts.items():o.load_state_dict(saved[k])
graph.replay();j.step(torch,copies,copyopts,data,ids)
resume_error=max(float((a-b).abs().max()) for k in models for a,b in zip(models[k].parameters(),copies[k].parameters()))
assert resume_error<=spec['computational_parity_atol']
# Exercise the actual primary membrane/learned-SK interfaces on every arm.
args=[torch.full((3,2),-65.,device='cuda'),torch.full((3,2),1e-4,device='cuda'),torch.full((3,2,18),.01,device='cuda'),
      torch.ones((3,2,11),device='cuda'),torch.full((3,2),.001,device='cuda'),torch.full((3,2,4),1e-5,device='cuda')]
core_error=0.
for model in models.values():
    core=j.t.make_core(torch,cfg,calcium,base,314.1592653589793,model,j.t.FAMILIES['all'])
    with torch.no_grad():
        expected=core(*args);runner=j.t.Stepper(torch,core,args,True);runner.reset(args);actual=runner.tick(args[4],args[5])
        core_error=max(core_error,max(float((a-b).abs().max()) for a,b in zip(expected,actual)))
assert core_error<=spec['computational_parity_atol']
result.update(resumed_optimizer_max_error=resume_error,coupled_graph_max_error=core_error,
              exact_zero_parent_distillation=True,all_primary_core_interfaces_exercised=True)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert revision==os.environ['GIADA_PREFLIGHT_REVISION']
result.update(revision=revision,gpu=torch.cuda.get_device_name(),torch=torch.__version__,platform='Kaggle',
              peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),scientific_training_performed=False)
Path(os.environ['GIADA_PREFLIGHT_OUTPUT']).write_text(json.dumps(result,indent=2))
print('[GIADA Task41j] GPU preflight '+json.dumps(result),flush=True)
