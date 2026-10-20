"""Cache LEARNED u/z only; identical serial FP32 state recurrence, no solver shortcut."""
import numpy as np

def install(old,torch,models,output,atol):
    Original=getattr(old,'_task39i_original_scanner',old.Scanner)
    old._task39i_original_scanner=Original
    class Cached:
        def __init__(self,torch_,models_,n,chunk,device):
            assert device=='cpu';self.models=models_;self.n=n
        def run(self,v,ca,initial):
            results=[]
            with torch.no_grad():
                for model in self.models:
                    members=model.a.shape[0];state=np.broadcast_to(initial,(members,*initial.shape)).copy()
                    trajectory=np.empty((len(v),*state.shape),dtype=np.float32)
                    for start in range(0,len(v),256):
                        vv=torch.as_tensor(v[start:start+256].reshape(-1))[None].expand(members,-1)
                        cc=torch.as_tensor(ca[start:start+256].reshape(-1))[None].expand(members,-1)
                        ru,rz=model.raw(vv,cc)
                        count=len(v[start:start+256])
                        u=torch.sigmoid(ru).numpy().reshape(members,count,self.n,18).transpose(1,0,2,3)
                        z=torch.sigmoid(rz).numpy().reshape(members,count,self.n,18).transpose(1,0,2,3)
                        for t in range(count):
                            state=state+z[t]*(u[t]-state)
                            trajectory[start+t]=state
                    results.append(trajectory)
            return results
    rng=np.random.default_rng(893152);v=rng.uniform(-100,50,(1025,2)).astype('float32')
    ca=10**rng.uniform(-7,-1,(1025,2)).astype('float32');initial=rng.uniform(0,1,(2,18)).astype('float32')
    expected=Original(torch,models,2,32,'cpu').run(v,ca,initial)
    actual=Cached(torch,models,2,32,'cpu').run(v,ca,initial)
    errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual,expected)]
    assert max(errors)<=atol,errors
    old.p.save_json(output/'cpu_scanner_equivalence.json',dict(valid=True,atol=atol,max_abs_errors=errors,ticks=1025,models=len(models),learned_coefficients_only=True,serial_float32_state_recurrence=True,analytic_kinetics_added=False))
    old.Scanner=Cached
    print('[39i recovery] cached learned-coefficient CPU scanner validated '+str(max(errors)),flush=True)
