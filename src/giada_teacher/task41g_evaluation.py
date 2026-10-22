"""Same acceptance panels as41f; precision, rollback and approximation probes."""
import copy
import itertools
import json
import types
import numpy as np
from . import task41g_preservation as g
from . import task41f_evaluation as fe
p,t,c,d,ee=g.p,g.t,g.c,g.d,g.f.ee


def precision_core(torch,model,cfg,calcium,base,area,voltage64,gates64,readout64):
    """Independent state/readout casts, frozen FP32 voltage coefficients."""
    import math
    vd=torch.float64 if voltage64 else torch.float32
    gd=torch.float64 if gates64 else torch.float32
    rd=torch.float64 if readout64 else torch.float32
    model=fe.MixedModel(torch,model) if gates64 else fe.cast_model(torch,model,torch.float32)
    def read(v,state,mult):return t.readout(torch,v.to(rd),state.to(rd),mult.to(rd))
    def core(v,ca,state,mult,inj,syn):
        v=v.to(vd);vg=v.to(gd);ca=ca.to(gd);state=state.to(gd)
        neural=model(vg,ca,state)
        gates=torch.cat((neural[...,:16],state[...,16:17],neural[...,17:]),-1)
        _,current=read(v,gates,mult)
        persistence=math.exp(-cfg['dt_ms']/calcium['decay_ms'])
        source=-10000*current[...,:2].sum(-1).to(gd)*calcium['gamma']/(2*96485.33212*calcium['depth_um'])
        nc=1e-4+(ca-1e-4)*persistence+source*calcium['decay_ms']*(1-persistence)
        sk=t.learned_sk(torch,model,nc,state[...,16])
        gates=torch.cat((gates[...,:16],sk[...,None],gates[...,17:]),-1)
        conductance,current=read(v,gates,mult)
        nv=t.membrane(torch,v,conductance.to(vd),inj.to(vd),syn.to(vd),area,cfg,base)
        return nv,nc,gates,current
    return core


def policy_rollout(torch,model,data,cfg,calcium,base,device,policy):
    vd,gd,rd=[torch.float64 if q else torch.float32 for q in policy]
    expand=lambda value,dtype:torch.tensor(value,device=device,dtype=dtype)[None].expand(3,*value.shape).contiguous()
    args=[expand(data['voltage'][:,0],vd),expand(data['calcium'][:,0],gd),expand(data['gates'][:,0],gd),
        expand(data['multipliers'],rd),expand(data['injection'][:,0],vd),expand(data['base_g_us'][:,0],vd)]
    core=precision_core(torch,model,cfg,calcium,base,float(data['area_um2']),*policy)
    runner=t.Stepper(torch,core,args,device=='cuda');currents=[]
    for tick in range(data['injection'].shape[1]):
        out=runner.tick(expand(data['injection'][:,tick],vd),expand(data['base_g_us'][:,tick],vd))
        currents.append(out[-1].clone())
    return torch.stack(currents).cpu().numpy()


def frozen_audit(torch,root,output,controls,spec,cfg,calcium,base,device):
    folder=output/'frozen_audit';folder.mkdir()
    data,meta,_=ee.load41c(root,spec)
    reference=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,device,graph=device=='cuda')['current'][:,0]
    rows=[]
    for name in ('P110','E_Q0','F_nominee'):
        for policy in itertools.product((False,True),repeat=3):
            pred=policy_rollout(torch,controls[name],data,cfg,calcium,base,device,policy)
            if policy==(False,False,False):
                original=fe.typed_rollout(torch,controls[name],data,cfg,calcium,base,device)['current']
                error=float(np.max(np.abs(pred-original)));assert error==0.,('FP32 core parity',name,error)
            rows.append(dict(model=name,voltage64=policy[0],gates64=policy[1],readout64=policy[2],
                responses=fe.paired(pred,reference,meta,spec['quality']),selection_eligible=False,
                coefficient_policy='Frozen FP32 voltage trunk; calcium SK follows gate precision',official_reference='Unchanged FP32 canonical full batch'))
            p.save_json(folder/'precision_factorial.json',fe.json_native(rows))
        print('[GIADA Task41g] checkpoint precision '+name,flush=True)
    old=controls['F_nominee'];rollback=[]
    for label,heads in [('all',(0,1,2)),('without_m',(1,2)),('without_h',(0,2)),('without_Nap',(0,1)),('m_only',(0,)),('none',())]:
        model=copy.deepcopy(old)
        with torch.no_grad():
            for h in range(3):
                if h not in heads:model.f_correction[:,h].zero_()
            model.f_correction[:,3:].zero_()  # Explicit separate z rollback below.
        pred=fe.typed_rollout(torch,model,data,cfg,calcium,base,device)['current']
        rollback.append(dict(label=label,z_corrections=False,parent='P110',responses=fe.paired(pred,reference,meta,spec['quality']),selection_eligible=False))
    for label in ('original_uz','z_only'):
        model=copy.deepcopy(old)
        if label=='z_only':
            with torch.no_grad():model.f_correction[:,:3].zero_()
        pred=fe.typed_rollout(torch,model,data,cfg,calcium,base,device)['current']
        rollback.append(dict(label=label,parent='P110',responses=fe.paired(pred,reference,meta,spec['quality']),selection_eligible=False))
    p.save_json(folder/'per_head_rollback.json',fe.json_native(rollback))
    p.save_json(folder/'scope.json',dict(valid=True,thresholds_changed=False,selection_eligible=False,
        numerical_factorial=8,learned_parent='E_Q0',old_correction_parent='P110',official_reference_retained=True))


def precision_cost(torch,model,data,spec,cfg,calcium,base,device,output):
    import time
    from .ie_ionic_cost import paired_summary
    rows=[]
    for batch in (642,4096):
        ix=np.arange(batch)%len(data['voltage'])
        for policy in ((True,False,False),(False,True,False),(False,False,True),(True,True,True)):
            runners={};timings={}
            for name,pol in [('FP32',(False,False,False)),('mixed',policy)]:
                vd,gd,rd=[torch.float64 if q else torch.float32 for q in pol]
                ex=lambda a,dt:torch.tensor(a,device=device,dtype=dt)[None].expand(3,*a.shape).contiguous()
                args=[ex(data['voltage'][ix,100],vd),ex(data['calcium'][ix,100],gd),ex(data['gates'][ix,100],gd),
                    ex(data['multipliers'][ix],rd),ex(data['injection'][ix,100],vd),ex(data['base_g_us'][ix,100],vd)]
                runner=t.Stepper(torch,precision_core(torch,model,cfg,calcium,base,float(data['area_um2']),*pol),args,True)
                injections=[ex(data['injection'][ix,100+k],vd) for k in range(spec['cost']['ticks'])]
                syn=[ex(data['base_g_us'][ix,100+k],vd) for k in range(spec['cost']['ticks'])]
                runners[name]=(runner,args,injections,syn);timings[name]=[]
            for repeat in range(spec['cost']['repetitions']):
                for name in (('FP32','mixed') if repeat%2==0 else ('mixed','FP32')):
                    runner,args,inj,syn=runners[name];runner.reset(args);torch.cuda.synchronize();start=time.perf_counter()
                    for j in range(len(inj)):runner.tick(inj[j],syn[j])
                    torch.cuda.synchronize();timings[name].append((time.perf_counter()-start)*1000)
            rows.append(dict(batch_per_seed=batch,policy=list(policy),samples=timings,
                comparison_vs_same_model_FP32=paired_summary(timings['FP32'],timings['mixed']),quality_qualified=False))
            p.save_json(output/'precision_execution_cost.json',rows)
            del runners;__import__('gc').collect();torch.cuda.empty_cache()
    return rows


def projection_probe(torch,parent,spec,device,output):
    rows=[];raw=d.anchors(spec['new_seeds']['fit']+29001,32768)
    held=d.anchors(spec['new_seeds']['selection']+29001,1024)
    for n in (64,128):
        for shape in ('linear','smoothC2'):
            def feature(v):
                values=d.feature(torch,torch.tensor(v,dtype=torch.float64),n,'linear')
                return values if shape=='linear' else values.pow(3)*(10+values*(-15+6*values))
            fit=feature(raw['v'][:,1]);test=feature(held['v'][:,1])
            with torch.no_grad():
                def residual(data):
                    v=torch.tensor(data['v'][:,1],device=device)[None].expand(3,-1);ca=torch.tensor(data['ca'][:,1],device=device)[None].expand(3,-1)
                    u,_=parent.raw(v,ca)
                    return -g.prior.clipped_or_censored_error(torch,u[...,0],torch.tensor(data['u'][:,1,0],device=device)[None],True).cpu().double()
                y=residual(raw);target=residual(held)
            for count in (1024,4096,16384):
                for s,seed in enumerate(spec['seeds']):
                    w=torch.linalg.lstsq(fit[:count],y[s,:count,None],driver='gelsy').solution
                    error=(test@w).squeeze(-1)-target[s]
                    rows.append(dict(basis=n,shape=shape,FIT_count=count,seed=seed,heldout_logit_mse=float(error.square().mean()),
                        heldout_logit_max=float(error.abs().max()),selection_eligible=False))
    p.save_json(output/'projection_data_resolution_probe.json',dict(rows=rows,
        scope='FIT-only least squares projection; quintic C2 smooth partition, not cubic B-spline; no deployable or SGD claim.'))


def evaluate(torch,root,output,controls,trained,spec,cfg,calcium,base,device):
    # Bind the already audited evaluator to the new definitions without
    # mutating the41f implementation or any historical source checksum.
    fn=types.FunctionType(fe.evaluate.__code__,dict(fe.__dict__,f=g))
    report=fn(torch,root,output,controls,trained,spec,cfg,calcium,base,device)
    if device=='cuda':
        data,_,_=ee.load41c(root,spec)
        precision_cost(torch,trained[report['nominee']],data,spec,cfg,calcium,base,device,output)
    report.update(task='41g',decision='REPAIR_REQUIRES_INDEPENDENT_CONFIRMATION' if report['passing_arms'] else 'PRESERVATION_REPAIR_NO_GO')
    return report
