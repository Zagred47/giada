"""Output-preserving execution variants of the frozen41m Ca-only operator.

All GEMM output widths and accumulation orders are retained. No distillation,
weight approximation or time/state change. Acceptance requires bitwise parity.
"""
import copy
from . import task41m_frozen_contract as m

def variants(torch,dual):
    class Exact(torch.nn.Module):
        def __init__(self,mode):
            super().__init__();self.parent=copy.deepcopy(dual.parent);self.donor=copy.deepcopy(dual.donor)
            self.register_buffer('mask',dual.mask.clone());self.mode=mode
            for net in (self.parent,self.donor.inner):
                assert net.architecture in ('plain','shared') and net.basis=='local' and not net.compact
                assert net.ca_definition['basis']==32 and net.precision_definition['basis']==64
                assert len(net.definitions)==1 and net.definitions[0]['basis'] in (64,128) and net.definitions[0]['kind'] in ('U64','U128')
            for par in self.parameters():par.requires_grad_(False)
        def __getattr__(self,key):
            if key in ('a','sa','sb','sc','sd','gbar'):return getattr(super().__getattr__('parent'),key)
            return super().__getattr__(key)
        def raw(self,v,ca):
            assert v.dtype==torch.float32 and ca.dtype==torch.float32
            cache={}
            def cached(name,fn):
                if name not in cache:cache[name]=fn()
                return cache[name]
            def feature(n,order):
                def make():
                    centers=torch.linspace(-100,60,n,device=v.device,dtype=v.dtype)
                    delta=v[...,None]-centers
                    distance=delta.abs()/(160/(n-1)) if order=='A' else (delta/(160/(n-1))).abs()
                    return (1-distance).clamp_min(0)
                return cached((n,order),make)
            def voltage_raw(net):
                vv=cached('voltage',lambda:(v+40)/80)
                return torch.bmm(torch.nn.functional.silu(torch.bmm(vv[...,None],net.a)+net.b),net.c)+net.d[:,None]
            def ca_extra(net):
                value=torch.bmm(feature(32,'A').double(),net.ca_correction.double()).to(v.dtype)
                return .1*torch.tanh(value/.1)*net.ca_mask
            def transfer(net,head):
                return .1*torch.tanh((feature(64,'B').double()*net.transfer_correction[:,head,None].double()).sum(-1).to(v.dtype)/.1)
            def tail(net):
                weights=net.f_correction*net.active
                n=net.definitions[0]['basis']
                value=torch.bmm(feature(n,'B').double(),weights[:,:3,:n].transpose(1,2).double()).to(v.dtype)
                value=.1*torch.tanh(value/.1)
                return torch.cat((value,torch.zeros_like(value)),-1)
            def full(net):
                raw=voltage_raw(net);cc=cached('calcium',lambda:(ca.clamp_min(1e-9).log10()+4.5)/2.5)
                sk=torch.bmm(torch.nn.functional.silu(torch.bmm(cc[...,None],net.sa)+net.sb),net.sc)+net.sd[:,None]
                u=torch.cat((raw[...,:16],sk[...,0:1],raw[...,16:17]),-1)
                z=torch.cat((raw[...,17:33],sk[...,1:2],raw[...,33:34]),-1)
                extra=torch.bmm(feature(32,'A').double(),net.correction.double()).to(v.dtype)
                extra=.1*torch.tanh(extra/.1);zero=torch.zeros_like(u[...,:1]);indices=(8,9,12,13)
                u=u+torch.cat([extra[...,indices.index(g):indices.index(g)+1] if g in indices else zero for g in range(18)],-1)
                z=z+torch.cat([extra[...,indices.index(g)+4:indices.index(g)+5] if g in indices else zero for g in range(18)],-1)
                extra=ca_extra(net)
                u=torch.cat((u[...,:4]+extra[...,:4],u[...,4:]),-1);z=torch.cat((z[...,:4]+extra[...,4:],z[...,4:]),-1)
                extra=torch.bmm(feature(64,'A').double(),net.precision_correction.double()).to(v.dtype)
                extra=.1*torch.tanh(extra/.1);zero=torch.zeros_like(u[...,:1])
                u=u+torch.cat([extra[...,0:1] if g==8 else zero for g in range(18)],-1)
                z=z+torch.cat([extra[...,1:2] if g==7 else extra[...,2:3] if g==11 else zero for g in range(18)],-1)
                extra=torch.stack([transfer(net,k) for k in range(2)],-1)
                u=torch.cat((u[...,:1]+extra[...,:1],u[...,1:8],u[...,8:9]+extra[...,1:2],u[...,9:]),-1)
                return m.l.f.add(torch,u,z,tail(net))
            def ca_only(net):
                raw=voltage_raw(net);u=raw[...,:2];z=raw[...,17:19]
                # Preserve the original +zero operations (including signed zero).
                zero=torch.zeros_like(u);u=u+zero;z=z+zero
                extra=ca_extra(net);u=u+extra[...,:2];z=z+extra[...,4:6]
                u=u+zero;z=z+zero
                u=torch.cat((u[...,:1]+transfer(net,0)[...,None],u[...,1:2]),-1)
                extra=tail(net)
                return u+extra[...,:2],z+extra[...,3:5]
            pu,pz=self.parent.raw(v,ca) if self.mode=='donor_pruned' else full(self.parent)
            du,dz=ca_only(self.donor.inner)
            return torch.cat((du,pu[...,2:]),-1),torch.cat((dz,pz[...,2:]),-1)
        def forward(self,v,ca,state):
            u,z=self.raw(v,ca);return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)
    return dict(dual=dual,donor_pruned=Exact('donor_pruned').eval(),shared_features=Exact('shared_features').eval())
