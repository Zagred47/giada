"""Frozen 41e regressions, coupling interventions and paired execution cost."""
import copy
import contextlib
import gc
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
from . import task41e_transfer_model as m
from . import task41e_transfer_training as training
from . import task41d_evaluation as old
p,t,c,x=m.p,m.t,m.c,m.x
e=m.d.prior.evaluation


def clone_model(torch,model):
    if isinstance(model,GateOracle):
        return GateOracle(torch,clone_model(torch,model.parent),model.gates,model.equilibrium_only)
    if 'ca_head' in getattr(model,'__dict__',{}):
        return old.compose(torch,copy.deepcopy(model.ca_head),copy.deepcopy(model.nap_head),copy.deepcopy(model.parent))
    return copy.deepcopy(model)


def load41c(root,spec):
    import zipfile
    path=Path(os.environ.get('GIADA_TASK41C_INPUTS','/content/task41d_opened41c.zip'))
    assert p.digest(path)==spec['opened41c_sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,sha in spec['opened41c_members'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
        arrays=lambda name:dict(np.load(io.BytesIO(z.read(name))))
        return arrays('native_shadow_traces.npz'),json.loads(z.read('cases.json')),arrays('fresh_stress_traces.npz')


def suite(torch,output,models,data,meta,spec,cfg,calcium,base,device):
    # Reuse the immutable evaluator; only the log label differs, and the exact
    # metric contract is compared to 41d before dispatch.
    return old.suite(torch,output,models,data,meta,spec,cfg,calcium,base,device)


def fusion_equivalence(torch,controls,data,spec,cfg,calcium,base,device):
    legacy,fused=controls['D_combined'],controls['D_fused']
    v=torch.linspace(-100,60,641,device=device)[None].expand(3,-1)
    ca=torch.logspace(-7,-2,641,device=device)[None].expand(3,-1)
    a,b=legacy.raw(v,ca),fused.raw(v,ca)
    raw_error=max(float((aa-bb).abs().max()) for aa,bb in zip(a,b))
    bitwise=all(torch.equal(aa,bb) for aa,bb in zip(a,b))
    subset={k:(a[:4,:81] if k in ('voltage','calcium','gates','injection','base_g_us') else a[:4])
            for k,a in data.items() if k!='area_um2'}
    subset['area_um2']=float(data['area_um2'])
    paths=[]
    for model in (legacy,fused):
        paths.append(t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,t.FAMILIES['all']),subset,device,graph=device=='cuda'))
    errors={k:float(abs(paths[0][k].astype('float64')-paths[1][k].astype('float64')).max()) for k in paths[0]}
    valid=raw_error<=spec['computational_parity_atol'] and max(errors.values())<=spec['computational_parity_atol']
    assert valid,'Fused composition failed numerical equivalence'
    return dict(valid=valid,raw_max_error=raw_error,raw_bitwise_equal=bitwise,rollout_max_errors=errors,
                parent_calls_legacy=3,parent_calls_fused=1,thresholds_changed=False)


def typed_rollout(torch,model,data,cfg,calcium,base,device,dtype,clamp_v=False,clamp_ca=False):
    if model is not None:
        model=clone_model(torch,model)
        # Composition is a plain wrapper; its children must be cast together.
        if hasattr(model,'ca_head'):
            for child in (model.parent,model.ca_head,model.nap_head):child.to(device=device,dtype=dtype)
        else:model.to(device=device,dtype=dtype)
    convert=lambda a:torch.tensor(a,device=device,dtype=dtype)
    expand=lambda a:convert(a)[None].expand(3,*a.shape).contiguous()
    args=[expand(data[k][:,0]) for k in ('voltage','calcium','gates')]
    args += [expand(data['multipliers']),expand(data['injection'][:,0]),expand(data['base_g_us'][:,0])]
    core=t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,t.FAMILIES['all'] if model is not None else ())
    runner=t.Stepper(torch,core,args,device=='cuda')
    inputs={k:convert(data[k]).transpose(0,1) for k in ('voltage','calcium','injection','base_g_us')}
    values=[[a.clone()] for a in args[:3]]+[[]]
    for tick in range(data['injection'].shape[1]):
        if clamp_v:runner.args[0].copy_(inputs['voltage'][tick][None].expand(3,-1))
        if clamp_ca:runner.args[1].copy_(inputs['calcium'][tick][None].expand(3,-1))
        result=runner.tick(inputs['injection'][tick][None].expand(3,-1),inputs['base_g_us'][tick][None].expand(3,-1,-1))
        for target,value in zip(values,result):target.append(value.clone())
    return {k:torch.stack(v).cpu().numpy() for k,v in zip(('voltage','calcium','gates','current'),values)}


class GateOracle:
    """Diagnostic canonical substitution. Never a qualified candidate."""
    def __init__(self,torch,parent,gates,equilibrium_only=True):
        self.torch,self.parent,self.gates,self.equilibrium_only=torch,parent,gates,equilibrium_only
    def __getattr__(self,name):return getattr(self.parent,name)
    def to(self,*args,**kwargs):self.parent.to(*args,**kwargs);return self
    def raw(self,v,ca):
        torch=self.torch;u,z=self.parent.raw(v,ca)
        uu=[];zz=[]
        for gate in range(18):
            if gate in self.gates:
                channel=max(k for k in range(11) if int(c.ionic.OFFSETS[k])<=gate)
                inf,tau=t._rates(c.ionic.CHANNELS[channel],v,ca,torch)
                if len(c.ionic.STATE_NAMES[channel])==2:
                    k=gate-int(c.ionic.OFFSETS[channel]);inf,tau=inf[...,k],tau[...,k]
                uu.append(torch.logit(inf)[...,None])
                zz.append(z[...,gate:gate+1] if self.equilibrium_only else torch.logit(-torch.expm1(-.1/tau))[...,None])
            else:uu.append(u[...,gate:gate+1]);zz.append(z[...,gate:gate+1])
        return torch.cat(uu,-1),torch.cat(zz,-1)
    def __call__(self,v,ca,g):
        u,z=self.raw(v,ca);return g+self.torch.sigmoid(z)*(self.torch.sigmoid(u)-g)


def paired_rows(pred,reference,meta,quality):
    rows=[]
    for seed_index,seed in enumerate((17,29,43)):
        for k,row in enumerate(meta):
            if row['intervention']=='none':continue
            source=row['source_index']
            rows.append(dict(case=row['original_id'],seed=seed,metrics=p.evaluate_intervention(
                reference['current'][:,0,source],reference['current'][:,0,k],
                pred['current'][:,seed_index,source],pred['current'][:,seed_index,k],.1,quality)))
    return rows


def coupling_probe(torch,root,output,controls,data,meta,spec,cfg,calcium,base,device):
    output.mkdir()
    original=[int(meta[k]['source_index']) for k in (108,122)]+[108,122]
    original=list(dict.fromkeys(original));lookup={old:new for new,old in enumerate(original)}
    selected={k:(v[original].copy() if k!='area_um2' else float(v)) for k,v in data.items()}
    rows=[dict(original_id=k,intervention=meta[k]['intervention'],source_index=lookup[int(meta[k]['source_index'])]) for k in original]
    models={k:controls[k] for k in ('P110','D_C01','D_N000','D_combined')}
    result=[];autonomous={}
    for precision,dtype in [('FP32',torch.float32),('FP64',torch.float64)]:
        for clamp_v in (False,True):
            for clamp_ca in (False,True):
                label=f'{precision}_V{int(clamp_v)}_Ca{int(clamp_ca)}'
                reference=typed_rollout(torch,None,selected,cfg,calcium,base,device,torch.float64,clamp_v,clamp_ca)
                predictions={}
                for name,model in models.items():
                    pred=typed_rollout(torch,model,selected,cfg,calcium,base,device,dtype,clamp_v,clamp_ca)
                    predictions[name]=pred
                    np.savez_compressed(output/(label+'_'+name+'.npz'),**pred)
                cross=(predictions['D_combined']['current'].astype('float64')-predictions['D_C01']['current'].astype('float64')
                       -predictions['D_N000']['current'].astype('float64')+predictions['P110']['current'].astype('float64'))
                result.append(dict(label=label,maximum_abs_current_cross=float(abs(cross).max()),
                                   rms_current_cross=float(np.sqrt(np.mean(cross**2))),
                                   responses={name:paired_rows(pred,reference,rows,spec['quality']) for name,pred in predictions.items()}))
                if not clamp_v and not clamp_ca:autonomous[precision]=predictions
                p.save_json(output/'clamp_precision.json',result)
                print(f'[GIADA Task41e] checkpoint coupling {label}',flush=True)
    sweeps=[]
    for amplitude in (0.,.25,.5,1.):
        if amplitude==1.:
            # The full intervention already has an exact saved replay above.
            continue
        dd={k:(v.copy() if hasattr(v,'copy') else v) for k,v in selected.items()}
        for k,row in enumerate(rows):
            if row['intervention']=='none':continue
            source=row['source_index']
            for key in ('multipliers','injection','base_g_us'):
                dd[key][k]=dd[key][source]+amplitude*(selected[key][k]-selected[key][source])
            for key in ('voltage','calcium','gates'):
                dd[key][k,0]=dd[key][source,0]
        ref=typed_rollout(torch,None,dd,cfg,calcium,base,device,torch.float64)
        rr={}
        for name,model in models.items():
            pred=typed_rollout(torch,model,dd,cfg,calcium,base,device,torch.float64)
            rr[name]=paired_rows(pred,ref,rows,spec['quality'])
        sweeps.append(dict(amplitude=amplitude,responses=rr))
    p.save_json(output/'perturbation_sweep.json',dict(rows=sweeps,full_amplitude_in='clamp_precision.json',
                zero_amplitude_is_negative_control=True))
    oracle=[]
    for precision,dtype in [('FP32',torch.float32),('FP64',torch.float64)]:
        ref=typed_rollout(torch,None,selected,cfg,calcium,base,device,torch.float64)
        for name,gates in [('u0',[0]),('u8',[8]),('u0_u8',[0,8])]:
            model=GateOracle(torch,copy.deepcopy(controls['D_fused']),gates)
            pred=typed_rollout(torch,model,selected,cfg,calcium,base,device,dtype)
            oracle.append(dict(precision=precision,oracle=name,selection_eligible=False,
                               responses=paired_rows(pred,ref,rows,spec['quality'])))
    p.save_json(output/'oracles.json',oracle)
    jacobians=[]
    # Local derivative probe on the seed43 case122 trajectory. Dimensionless
    # coordinates avoid a calcium-unit change dominating the reported norm.
    k=lookup[122]
    with torch.inference_mode(False),torch.enable_grad():
        for precision,dtype in [('FP32',torch.float32),('FP64',torch.float64)]:
            for name in ('P110','D_combined'):
                model=clone_model(torch,models[name])
                if hasattr(model,'ca_head'):
                    for child in (model.parent,model.ca_head,model.nap_head):child.to(device=device,dtype=dtype)
                else:model.to(device=device,dtype=dtype)
                core=t.make_core(torch,cfg,calcium,base,float(selected['area_um2']),model,t.FAMILIES['all'])
                pred=autonomous[precision][name]
                for tick in (0,10,50,250):
                    if tick>=selected['injection'].shape[1]:continue
                    initial=np.r_[pred['voltage'][tick,2,k]/100,pred['calcium'][tick,2,k]/1e-4,pred['gates'][tick,2,k]]
                    zz=torch.tensor(initial,device=device,dtype=dtype,requires_grad=True)
                    def local(q):
                        v=(q[0]*100).reshape(1,1).expand(3,1)
                        ca=(q[1]*1e-4).reshape(1,1).expand(3,1)
                        g=q[2:].reshape(1,1,18).expand(3,1,18)
                        mult=torch.tensor(selected['multipliers'][k],device=device,dtype=dtype)[None,None].expand(3,1,11)
                        inj=torch.tensor(selected['injection'][k,tick],device=device,dtype=dtype).expand(3,1)
                        syn=torch.tensor(selected['base_g_us'][k,tick],device=device,dtype=dtype)[None,None].expand(3,1,4)
                        nv,nc,ng,_=core(v,ca,g,mult,inj,syn)
                        return torch.cat((nv[2,0:1]/100,nc[2,0:1]/1e-4,ng[2,0]))
                    jac=torch.autograd.functional.jacobian(local,zz).detach().double().cpu().numpy()
                    assert np.isfinite(jac).all()
                    jacobians.append(dict(precision=precision,model=name,tick=tick,jacobian=jac.tolist(),
                                          largest_singular_value=float(np.linalg.svd(jac,compute_uv=False)[0]),
                                          spectral_radius=float(abs(np.linalg.eigvals(jac)).max())))
    p.save_json(output/'local_jacobians.json',dict(rows=jacobians,scales=dict(voltage_mv=100,calcium_mM=1e-4,gates=1),
                limits='Local derivative diagnostics; not a global stability certificate.'))
    return dict(valid=True,rows=result,oracle_selection_eligible=False,thresholds_changed=False)


def evaluate(torch,root,output,trained,spec,device):
    class Progress:
        def __init__(self,stream):self.stream=stream
        def write(self,text):return self.stream.write(re.sub(r'\[GIADA Task\d+[a-z]*\]','[GIADA Task41e]',text))
        def flush(self):self.stream.flush()
    with contextlib.redirect_stdout(Progress(sys.stdout)):
        return _evaluate(torch,root,output,trained,spec,device)


def _evaluate(torch,root,output,trained,spec,device):
    _,cfg,calcium,base=m.contract(root)
    freeze=json.loads((output/'selection_freeze.json').read_text())
    payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    assert p.digest(output/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    assert not freeze['opened_regressions_accessed']
    controls=m.frozen_controls(torch,root,spec,device)
    models={**controls,**trained}
    oldspec=json.loads((root/c.CONFIG).read_text())
    data,meta,st,expected=e.load_opened(root,oldspec)
    dd,mm,ss=load41c(root,spec)
    equivalence=fusion_equivalence(torch,controls,dd,spec,cfg,calcium,base,device)
    p.save_json(output/'fusion_equivalence.json',equivalence)
    allmodels={**models,'LUT_control':x.lut_model(torch,device,.1)}
    opened41=suite(torch,output/'opened41',allmodels,data,meta,spec,cfg,calcium,base,device)
    stress41=old.stress(torch,models,st,spec,device,x.stress_design(json.loads((root/x.CONFIG).read_text())))
    p.save_json(output/'stress41.json',stress41)
    opened41c=suite(torch,output/'opened41c',allmodels,dd,mm,spec,cfg,calcium,base,device)
    stress41c=old.stress(torch,models,ss,spec,device,m.d.prior.stress_design(json.loads((root/m.d.prior.CONFIG).read_text())))
    p.save_json(output/'stress41c.json',stress41c)
    expected=json.loads((root/spec['task41d_summary']).read_text())
    for label,reports in [('41',opened41),('41c',opened41c)]:
        for mode in ('imposed','autonomous'):
            actual=reports['P110/'+mode];prior=expected['summaries']['P110']['panels'][label][mode]
            assert actual['passed_cases']==prior['trajectory']
            assert sum(r['metrics']['passed'] for r in actual['interventions'])==prior['responses']
    # GPU recurrence verifies the selection shadow, including unmodified K_Pst.
    dev=m.d.development(spec['new_seeds']['development']);devrows={}
    target=p.readout(dev['voltage'],dev['target_gates'],dev['multipliers'],c.ionic.REVERSALS)[1]
    for name,model in models.items():
        gates=c.j.i.old.Scanner(torch,[model],dev['voltage'].shape[1],32,device).run(dev['voltage'],dev['calcium'],dev['initial'].astype('float32'))[0]
        devrows[name]=[dict(seed=seed,case=k,metrics=p.evaluate_currents(target[:,k],p.readout(dev['voltage'],gates[:,s],dev['multipliers'],c.ionic.REVERSALS)[1][:,k],.1,spec['quality'],dev['target_gates'][:,k],gates[:,s,k]))
                       for s,seed in enumerate(spec['seeds']) for k in range(dev['voltage'].shape[1])]
        p.save_json(output/'development_gpu.json',devrows)
    k_channel=c.ionic.CHANNELS.index('K_Pst')
    k_gates=list(range(int(c.ionic.OFFSETS[k_channel]),int(c.ionic.OFFSETS[k_channel+1])))
    oracle=GateOracle(torch,copy.deepcopy(controls['P110']),k_gates,equilibrium_only=False)
    oracle_rows=c.shadow_evaluate(torch,[oracle],dev,spec['quality'],device)
    p.save_json(output/'development_KPst_oracle.json',dict(selection_eligible=False,rows=oracle_rows,
                interpretation='Diagnostic reachability of a function unchanged by the two trained heads; all original gates remain mandatory.'))
    hist=output/'historical_regression';hist.mkdir()
    b0=t.load_models(torch,root,device)['J1']
    historical=c.b.causal_evaluate(torch,root,hist,dict(B0=b0,**models),spec,device)
    oldcfg=json.loads((root/c.j.i.old.CONFIG).read_text())
    evaluation={**json.loads((root/c.j.CONFIG).read_text()),'architectures':list(models)}
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=oldcfg[key]
    recurrence=[];items=list(models.items())
    for start in range(0,len(items),3):
        chunk=items[start:start+3];out=hist/f'recurrence_group_{start//3}';out.mkdir()
        c.j.i.old.evaluate(root,out,{**evaluation,'architectures':[k for k,v in chunk]},[v for k,v in chunk],torch,device,json.loads((root/p.CONFIG).read_text()))
        recurrence+=json.loads((out/'recurrence_evaluation.json').read_text())
        p.save_json(hist/'recurrence_evaluation.json',recurrence)
        gc.collect();torch.cuda.empty_cache()
        print(f'[GIADA Task41e] checkpoint historical recurrence group {start//3} complete',flush=True)
    coupling_probe(torch,root,output/'coupling_probe',controls,dd,mm,spec,cfg,calcium,base,device)
    cost=e.benchmark(torch,root,output,models,dd,spec,device)
    from .ie_ionic_cost import paired_summary
    for row in cost:
        row['fusion_comparison']=paired_summary(row['samples']['D_combined']['wall_ms'],row['samples']['D_fused']['wall_ms'])
    p.save_json(output/'cost.json',cost)
    summaries={};passing=[]
    for name in models:
        summary=dict(development=dict(passed=sum(r['metrics']['passed'] for r in devrows[name]),total=len(devrows[name])),panels={},stress={},historical={},recurrence={})
        okay=all(r['metrics']['passed'] for r in devrows[name])
        for label,reports in [('41',opened41),('41c',opened41c)]:
            summary['panels'][label]={}
            for mode in ('imposed','autonomous'):
                rr=reports[name+'/'+mode];passed=sum(r['metrics']['passed'] for r in rr['interventions'])
                summary['panels'][label][mode]=dict(trajectory=rr['passed_cases'],total=rr['total'],responses=passed,response_total=len(rr['interventions']))
                okay &= rr['passed'] and passed==len(rr['interventions'])
        for label,sr in [('41',stress41),('41c',stress41c)]:
            rows=[r for r in sr['rows'] if r['model']==name]
            summary['stress'][label]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows))
            okay &= all(r['metrics']['passed'] for r in rows)
        for mode in ('imposed','autonomous'):
            rr=historical[name+'/'+mode]
            summary['historical'][mode]=dict(trajectory=rr['passed_cases'],total=rr['total'],responses=sum(r['metrics']['passed'] for r in rr['interventions']))
            okay &= rr['passed'] and all(r['metrics']['passed'] for r in rr['interventions'])
        for role in ('development','stress_diagnostic'):
            rows=[r for r in recurrence if r['model']=='joint_w'+name and r['role']==role]
            assert len(rows)==72
            summary['recurrence'][role]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows))
            okay &= all(r['metrics']['passed'] for r in rows)
        summaries[name]=summary
        if okay and name in {a['name'] for a in m.arms()}:passing.append(name)
    contrasts=[]
    for axis,control,treatment in m.contrasts():
        ai=next(i for i,a in enumerate(m.arms()) if a['name']==control)
        bi=next(i for i,a in enumerate(m.arms()) if a['name']==treatment)
        ranks=freeze['primary_ranks']
        ratios=[ranks[bi*3+s][2]/max(ranks[ai*3+s][2],1e-12) for s in range(3)]
        counts=all(ranks[bi*3+s][1]<=ranks[ai*3+s][1] and ranks[bi*3+s][0]<=ranks[ai*3+s][0] for s in range(3))
        contrasts.append(dict(axis=axis,control=control,treatment=treatment,per_seed_target_severity_ratios=ratios,
                              registered_contrast_passed=bool(all(r<1 for r in ratios) and np.median(ratios)<=.8 and counts),
                              all_global_gates_still_required=True))
    p.save_json(output/'paired_contrasts.json',contrasts)
    result=dict(valid=True,task='41e',decision='TRANSFER_REPAIR_REQUIRES_INDEPENDENT_CONFIRMATION' if passing else 'TRANSFER_DIAGNOSIS_NO_GO',
                passing_arms=passing,nominee=freeze['nominee'],summaries=summaries,baseline_reproduced=True,
                thresholds_changed=False,independent_confirmation=False,task42_authorized=False,speedup_claim=False,
                fusion_equivalence=equivalence,limits=spec['limits'])
    p.save_json(output/'final_report.json',result)
    return result
