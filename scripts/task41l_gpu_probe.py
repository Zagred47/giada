"""Exercise every loss and resumed optimizer on CUDA before scientific launch."""
import os,json,subprocess,sys,tempfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.giada_teacher import task41l_regression_attribution as k
from src.giada_teacher import task41l_frozen_attribution as attribution
from src.giada_teacher.task41k_recovery import preflight as scanner_preflight
assert torch.cuda.is_available()
with tempfile.TemporaryDirectory() as tmp:result=k.run(ROOT,Path(tmp),smoke=True)
spec,cfg,calcium,base=k.contract(ROOT);parent=k.j.controls(torch,ROOT,'cuda')['H_nominee']
models={a['name']:k.j.model_class(torch)(parent,a).to('cuda') for a in k.arms()}
args=[torch.full((3,2),-65.,device='cuda'),torch.full((3,2),1e-4,device='cuda'),torch.full((3,2,18),.01,device='cuda'),torch.ones((3,2,11),device='cuda'),torch.full((3,2),.001,device='cuda'),torch.full((3,2,4),1e-5,device='cuda')]
error=0.
for model in models.values():
    core=k.t.make_core(torch,cfg,calcium,base,314.1592653589793,model,k.t.FAMILIES['all'])
    with torch.no_grad():
        expected=core(*args);runner=k.t.Stepper(torch,core,args,True);runner.reset(args);actual=runner.tick(args[4],args[5])
        error=max(error,max(float((a-b).abs().max()) for a,b in zip(expected,actual)))
assert error<=spec['computational_parity_atol']
from src.giada_teacher import task41i_evaluation as ev
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp);ev.install_resume_cache(folder)
    fake=dict(voltage=np.full((64,103),-65.,dtype='float32'),calcium=np.full((64,103),1e-4,dtype='float32'),gates=np.full((64,103,18),.01,dtype='float32'),multipliers=np.ones((64,11),dtype='float32'),injection=np.full((64,103),.001,dtype='float32'),base_g_us=np.full((64,103,4),1e-5,dtype='float32'),area_um2=np.array(314.1592653589793))
    smoke=dict(spec,cost=dict(spec['cost'],batches=[1],ticks=2,repetitions=2))
    with torch.inference_mode():rows=ev.he.ee.e.benchmark(torch,ROOT,folder,dict(H_nominee=parent,**models),fake,smoke,'cuda')
    assert len(rows)==2 and all(all(row['finite'].values()) for row in rows)
    scanner_preflight(torch,models,spec,'cuda',folder)
    swaps,_=attribution.source(torch,ROOT,dict(H_nominee=parent),spec,'cuda')
    v=torch.linspace(-90,40,97,device='cuda')[None].expand(3,-1);ca=torch.full_like(v,1e-4)
    assert all(torch.equal(a,b) for a,b in zip(swaps['swap000'].raw(v,ca),parent.raw(v,ca)))
    with torch.inference_mode():
        for name,model in {'H':parent,**swaps}.items():
            for policy in ((False,False),(True,True),(True,False),(False,True)):
                pred=attribution.rollout(torch,model,fake,cfg_calcium_base:=(cfg,calcium,base),'cuda',policy,(False,False),[0,1])
                assert all(np.isfinite(x).all() for x in pred.values())
        canonical=attribution.rollout(torch,None,fake,(cfg,calcium,base),'cuda',(False,False),keep=[0,1])
        baseline=k.t.rollout(torch,k.t.make_core(torch,cfg,calcium,base,float(fake['area_um2'])),fake,'cuda',graph=True)
        for key in canonical:
            expected=baseline[key][:,:, [0,1]]
            assert np.allclose(canonical[key],expected,atol=spec['computational_parity_atol'],rtol=0),(key,np.max(abs(canonical[key]-expected)))
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert revision==os.environ['GIADA_PREFLIGHT_REVISION']
result.update(revision=revision,gpu=torch.cuda.get_device_name(),torch=torch.__version__,platform='Kaggle',peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),coupled_graph_error=error,actual_benchmark_interface_exercised=True,scanner_lifetime_verified=True,frozen_swaps_verified=True,matched_canonical_primary_parity_verified=True)
Path(os.environ['GIADA_PREFLIGHT_OUTPUT']).write_text(json.dumps(result,indent=2))
print('[GIADA Task41l] GPU preflight '+json.dumps(result),flush=True)


