"""Task18d: preserve the parent function while exposing a cusp or mTau branches."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from .hh_family_transfer import data,model_factory,sha,write
from .hh_potassium_local_repair import run as refinement

def architecture_factory(torch,width,ds):
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__();base=model_factory(torch,width,ds)
            for k,v in base.named_parameters():
                if k=='w1':v=torch.cat([v.detach(),torch.zeros_like(v)],dim=-1)
                if k=='wo':v=torch.cat([v.detach(),v.detach()[:,2:3].clone()],dim=1)
                if k=='bo':v=torch.cat([v.detach(),v.detach()[:,2:3].clone()],dim=1)
                setattr(self,k,torch.nn.Parameter(v.detach().clone()))
            self.register_buffer('feature_mask',torch.tensor([a.startswith('feature') for _,a,_ in ds],dtype=torch.float32)[:,None,None])
            self.register_buffer('split_mask',torch.tensor([a.startswith('split') for _,a,_ in ds],dtype=torch.bool)[:,None])
        def forward(self,x):
            v=x[...,0];f=torch.stack(((v+30)/100,abs(v+60)/100),-1);f=torch.cat([f[...,:1],f[...,1:]*self.feature_mask],-1)
            f=torch.nn.functional.silu(torch.bmm(f,self.w1.transpose(1,2))+self.b1[:,None]);f=torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None]);raw=torch.bmm(f,self.wo.transpose(1,2))+self.bo[:,None]
            mt=torch.where(self.split_mask&(v>=-60),raw[...,4],raw[...,2]);inf=torch.sigmoid(raw[...,:2]);tau=torch.exp(torch.stack((mt,raw[...,3]),-1).clamp(-12,12));z=-torch.expm1(-x[...,3,None]/tau)
            return (1-z)*x[...,1:3]+z*inf,inf,tau
    return Ensemble()

def initialize(torch,config,source,device):
    if sha(source)!=config['parent_fixture_sha256']:raise RuntimeError('parent fixture SHA')
    ds=[('K_Pst',f'{a}/top0/lr0.003',s) for a in config['sampling'] for s in config['seeds']]
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:raise RuntimeError('parent CRC')
        f=json.loads(z.read('selection_freeze.json'));h=f.pop('freeze_sha256');assert hashlib.sha256(json.dumps(f,sort_keys=True,separators=(',',':')).encode()).hexdigest()==h
        sel=f['selected'];name=f"checkpoint_step{sel['step']}.pt";raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==f['checkpoint_hashes'][name]
        state=torch.load(io.BytesIO(raw),map_location=device,weights_only=True);chosen={k:v[sel['indices']] for k,v in state.items()}
    parent=model_factory(torch,32,[('K_Pst','parent',s) for s in config['seeds']]).to(device);parent.load_state_dict(chosen)
    model=architecture_factory(torch,32,ds).to(device)
    with torch.no_grad():
        for k,p in model.named_parameters():
            v=chosen[k].repeat((3,)+(1,)*(p.ndim-1))
            if k=='w1':v=torch.cat([v,torch.zeros_like(v)],-1)
            if k=='wo':v=torch.cat([v,v[:,2:3].clone()],1)
            if k=='bo':v=torch.cat([v,v[:,2:3].clone()],1)
            p.copy_(v)
    return model,parent,ds

def pools(count):
    x=data(183101,count);x[:count//4,0]=np.random.default_rng(183102).uniform(-135,-115,count//4);x[count//4:count//2,0]=np.random.default_rng(183103).uniform(-65,-55,count//4)
    return {k:x.copy() for k in ('smooth','feature','split')}

def extrema_domains(role):
    # Error is affine in each initial gate; endpoints bound every initial occupancy.
    grid=np.linspace(-65,-55,201 if role=='development' else 401)
    grid=np.unique(np.r_[grid,-60+np.array([-1e-3,-1e-5,0,1e-5,1e-3])])
    v,dt,initial=np.meshgrid(grid,[.025,.1,.5,1,5,25,100],[0.,1.],indexing='ij')
    return {'state_extrema':np.c_[v.ravel(),initial.ravel(),initial.ravel(),dt.ravel()]}

def run(output,config,revision,source):
    report=refinement(output,config,revision,source,model_initializer=initialize,model_builder=architecture_factory,pool_builder=pools,seed_offset=100,extra_domains=extrema_domains)
    report['schema_version']='giada-task18d-v1';report['effective_parameter_counts']={'smooth':1252,'feature':1284,'split':1285};report['parameter_count_limit']='Effective capacities within3%; same width32 backbone. Inactive padding excluded; not exactly equal capacity.'
    report['robust_margin_passed']=all(mm['gate_max_error']<.008 for r in report['results'] for domain,mm in r['metrics'].items() if domain!='ood_negative')
    write(Path(output)/'final_report.json',report);return report
