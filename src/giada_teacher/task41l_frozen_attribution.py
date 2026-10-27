"""Frozen learned-channel interventions; diagnostic only, never selection."""
import copy
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import types
import time
import zipfile
import numpy as np
from . import task41l_regression_attribution as l
from . import task41h_evaluation as he
from . import task41f_evaluation as fe
p,t,c=l.p,l.t,l.c
CHANNELS=('Ca_HVA','Nap_Et2','K_Tst')


def swap_class(torch):
    class Swap(torch.nn.Module):
        def __init__(self,parent,donor,bits):
            super().__init__();self.parent=copy.deepcopy(parent);self.donor=copy.deepcopy(donor);self.bits=tuple(bits)
            mask=torch.zeros(18,dtype=torch.bool,device=parent.a.device)
            for name,bit in zip(CHANNELS,bits):
                channel=list(c.ionic.CHANNELS).index(name);lo,hi=map(int,c.ionic.OFFSETS[channel:channel+2])
                mask[lo:hi]=bool(bit)
            self.register_buffer('mask',mask)
        def __getattr__(self,name):
            if name in ('a','sa','sb','sc','sd','gbar'):return getattr(super().__getattr__('parent'),name)
            return super().__getattr__(name)
        def raw(self,v,ca):
            return tuple(torch.where(self.mask,b,a) for a,b in zip(self.parent.raw(v,ca),self.donor.raw(v,ca)))
        def forward(self,v,ca,state):
            u,z=self.raw(v,ca);return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)
    return Swap


def source(torch,root,controls,spec,device):
    path=Path(os.environ['GIADA_TASK41L_FROZEN_INPUT'])
    assert p.digest(path)==spec['frozen_diagnostic_input']['sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,sha in spec['frozen_diagnostic_input']['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
        weights=torch.load(io.BytesIO(z.read('frozen_checkpoint.pt')),map_location=device,weights_only=True)
        sidecar=json.loads(z.read('voltage_precision.json'))
    donor=l.j.model_class(torch)(controls['H_nominee'],l.k.arms()[4]).to(device)
    donor.load_state_dict(weights['K_dyn_target']);donor.eval()
    models={'swap'+''.join(map(str,bits)):swap_class(torch)(controls['H_nominee'],donor,bits).eval() for bits in itertools.product((0,1),repeat=3)}
    models['K_full']=donor
    return models,sidecar


def canonical_core(torch,cfg,calcium,base,area,state64,solve64):
    # Identical canonical equations/order, explicit casts matching neural policy.
    sd=torch.float64 if state64 else torch.float32;vd=torch.float64 if solve64 else torch.float32
    def rates(name,v,ca,torch):return t._rates(name,v.float(),ca.float(),torch)
    def read(torch,v,g,m):return t.readout(torch,v.float(),g.float(),m.float())
    def membrane(torch,v,g,i,s,a,c,b):return t.membrane(torch,v.to(vd),g.to(vd),i.to(vd),s.to(vd),a,c,b).to(sd)
    fn=types.FunctionType(t.make_core.__code__,dict(t.make_core.__globals__,_rates=rates,readout=read,membrane=membrane),argdefs=t.make_core.__defaults__)
    return fn(torch,cfg,calcium,base,area)


def rollout(torch,model,data,physics,device,policy,forcing=(False,False),keep=None):
    state64,solve64=policy;dtype=torch.float64 if state64 else torch.float32
    ex=lambda a,dt:torch.tensor(a,device=device,dtype=dt)[None].expand(3,*a.shape).contiguous()
    args=[ex(data['voltage'][:,0],dtype),ex(data['calcium'][:,0],torch.float32),ex(data['gates'][:,0],torch.float32),
          ex(data['multipliers'],torch.float32),ex(data['injection'][:,0],dtype),ex(data['base_g_us'][:,0],dtype)]
    core=canonical_core(torch,*physics,float(data['area_um2']),*policy) if model is None else he.precision_core(torch,model,*physics,float(data['area_um2']),*policy)
    runner=t.Stepper(torch,core,args,device=='cuda')
    keep=np.arange(len(data['voltage'])) if keep is None else np.asarray(keep)
    indices=torch.tensor(keep,device=device);values=[[x.index_select(1,indices).clone()] for x in args[:3]]+[[]]
    for tick in range(data['injection'].shape[1]):
        for index,key in enumerate(('voltage','calcium')):
            if forcing[index]:runner.args[index].copy_(ex(data[key][:,tick],args[index].dtype))
        result=runner.tick(ex(data['injection'][:,tick],dtype),ex(data['base_g_us'][:,tick],dtype))
        for dest,value in zip(values,result):dest.append(value.index_select(1,indices).clone())
    return {key:torch.stack(value).cpu().numpy() for key,value in zip(('voltage','calcium','gates','current'),values)}


def continuous(pred,reference):
    result={}
    for name in ('voltage','calcium','gates','current'):
        error=np.asarray(pred[name],dtype=np.float64)-np.asarray(reference[name],dtype=np.float64)
        flat=error.reshape(len(error),3,-1)
        first=[]
        for seed in range(3):
            times=np.flatnonzero(np.any(flat[:,seed]!=0,axis=1));first.append(None if not len(times) else int(times[0]))
        result[name]=dict(rmse=np.sqrt((flat**2).mean((0,2))).tolist(),max_abs=abs(flat).max((0,2)).tolist(),signed_mean=flat.mean((0,2)).tolist(),first_nonzero_tick=first)
    return result


def factorial(rows):
    """Paired full-factorial effects, not claims of independent replication."""
    out=[]
    keys=sorted({(r['panel'],r['policy'],r['forcing']) for r in rows})
    for key in keys:
        pool={r['model']:r for r in rows if (r['panel'],r['policy'],r['forcing'])==key and r['model'].startswith('swap')}
        if len(pool)!=8:continue
        for mask in itertools.product((0,1),repeat=3):
            if not any(mask):continue
            effect={}
            for variable in ('voltage','calcium','gates','current'):
                effect[variable]=sum(np.asarray(row['continuous'][variable]['rmse'])*np.prod([2*int(name[4+i])-1 for i in range(3) if mask[i]]) for name,row in pool.items())/4
            out.append(dict(panel=key[0],policy=key[1],forcing=key[2],interaction=[CHANNELS[i] for i in range(3) if mask[i]],rmse_effect={n:v.tolist() for n,v in effect.items()}))
    return out


def metric_margins(metric,q):
    rows=[]
    for row in metric['channels']:
        margins={key:q[limit]-row[key] for key,limit in [('nrmse','current_nrmse_max'),('nmax','current_nmax_max'),('signed_charge_normalized','signed_charge_error_max'),('absolute_charge_normalized','absolute_charge_error_max')]}
        if row['event']['status']=='evaluable':
            margins.update({key:q[limit]-row['event'][key] for key,limit in [('peak_error_ms','peak_timing_max_ms'),('onset_error_ms','onset_offset_max_ms'),('offset_error_ms','onset_offset_max_ms')]})
        rows.append(dict(channel=row['channel'],margins=margins,event_status=row['event']['status'],blocked_passed=row['blocked_current_passed']))
    gate=metric.get('gate')
    return dict(channels=rows,gate=None if gate is None else dict(rmse=q['gate_rmse_max']-gate['rmse'],max_abs=q['gate_max_abs_max']-gate['max_abs'],occupancy_violations=gate['occupancy_violations']))


def evaluate(torch,root,out,controls,spec,cfg,calcium,base,device,stage_started=None):
    stage_started=time.monotonic() if stage_started is None else stage_started
    dest=out/'frozen_channel_attribution';dest.mkdir(exist_ok=True)
    models,sidecar=source(torch,root,controls,spec,device)
    p.save_json(dest/'reused_41k_voltage_precision.json',sidecar)
    physics=(cfg,calcium,base);oldspec=json.loads((root/c.CONFIG).read_text())
    aa,am,stress41,_=he.ee.e.load_opened(root,oldspec);bb,bm,stress=he.ee.load41c(root,spec)
    rows=[];floors=[]
    policies=[(False,False),(True,True)]
    for label,data,meta,cases in [('41',aa,am,[120]),('41c',bb,bm,[108,122])]:
        keep=list(dict.fromkeys([int(meta[i]['source_index']) for i in cases]+cases))
        # Entire original batch is computed, preserving GEMM/roundoff semantics.
        native={key:np.asarray(data[key])[keep].swapaxes(0,1)[:,None] for key in ('voltage','calcium','gates')}
        for policy in policies+[(False,True),(True,False)]:
            policy_name='state%d_solve%d'%policy
            for forcing in itertools.product((False,True),repeat=2):
                force_name='V%d_Ca%d'%forcing
                canonical=rollout(torch,None,data,physics,device,policy,forcing,keep)
                # Native records are post/pre-aligned by the canonical rollout contract.
                floor={key:float(np.sqrt(np.mean((canonical[key][:min(len(canonical[key]),len(native[key]))]-native[key][:min(len(canonical[key]),len(native[key]))])**2))) for key in native}
                qualified=(floor['voltage']<=cfg['gates']['max_formula_floor_worst_voltage_rmse_mv'] and floor['calcium']<=cfg['gates']['max_formula_floor_worst_cai_rmse_mM'])
                floors.append(dict(panel=label,policy=policy_name,forcing=force_name,native_rmse=floor,native_floor_qualified=qualified,selection_eligible=False))
                selected=models if policy in policies else {n:models[n] for n in ('swap000','K_full')}
                for name,model in selected.items():
                    cache=dest/f'{label}_{policy_name}_{force_name}_{name}.json'
                    if cache.exists():rows.append(json.loads(cache.read_text()));continue
                    if time.monotonic()-stage_started>5.5*3600:
                        from .task41i_evaluation import EvaluationPause
                        raise EvaluationPause()
                    pred=rollout(torch,model,data,physics,device,policy,forcing,keep)
                    measurements=[];case_metrics=[]
                    for seed_index,seed in enumerate(spec['seeds']):
                        for case in cases:
                            src=keep.index(int(meta[case]['source_index']));idx=keep.index(case)
                            metrics=p.evaluate_intervention(canonical['current'][:,seed_index,src],canonical['current'][:,seed_index,idx],pred['current'][:,seed_index,src],pred['current'][:,seed_index,idx],.1,spec['quality'])
                            measurements.append(dict(case=case,seed=seed,metrics=metrics,margins=metric_margins(metrics,spec['quality'])))
                        for index,case in enumerate(keep):
                            metrics=p.evaluate_currents(canonical['current'][:,seed_index,index],pred['current'][:,seed_index,index],.1,spec['quality'],canonical['gates'][1:,seed_index,index],pred['gates'][1:,seed_index,index],data['multipliers'][case]==0)
                            case_metrics.append(dict(case=case,seed=seed,metrics=metrics,margins=metric_margins(metrics,spec['quality'])))
                    if policy==(False,False) and forcing==(False,False) and name in ('swap000','K_full'):
                        old_name='H_nominee' if name=='swap000' else 'nominee'
                        previous=next(r for r in sidecar if r['panel']==label and r['model']==old_name and not r['state64'] and not r['solve64'] and not r['recentered'])
                        wanted={(r['case'],r['seed']):r['metrics']['passed'] for r in previous['responses']}
                        assert all(r['metrics']['passed']==wanted[(r['case'],r['seed'])] for r in measurements),('Frozen primary reproduction failed',label,name)
                    row=dict(panel=label,policy=policy_name,forcing=force_name,model=name,continuous=continuous(pred,canonical),responses=measurements,cases=case_metrics,selection_eligible=False,full_original_batch=True,reference_policy_matched=True,native_floor_qualified=qualified,causal_interpretation_requires_valid_floor=True)
                    np.savez_compressed(cache.with_suffix('.npz'),case_ids=np.array(keep),**pred)
                    p.save_json(cache,fe.json_native(row));rows.append(row)
                    print('[GIADA Task41l] frozen attribution '+cache.stem,flush=True)
        p.save_json(dest/'paired_numerical_floors.json',floors)
    design=l.d.prior.stress_design(json.loads((root/l.d.prior.CONFIG).read_text()))
    stress_file=dest/'imposed_stress41c.json'
    if not stress_file.exists():p.save_json(stress_file,he.ee.old.stress(torch,models,stress,spec,device,design))
    stress_file=dest/'imposed_stress41.json'
    if not stress_file.exists():p.save_json(stress_file,he.ee.old.stress(torch,models,stress41,spec,device,fe.x.stress_design(json.loads((root/fe.x.CONFIG).read_text()))))
    # Original recurrence panels also cover the K_Tst regression, not only
    # the opened autonomous responses. Pure frozen diagnostics, no selection.
    oldcfg=json.loads((root/c.j.i.old.CONFIG).read_text());evaluation=json.loads((root/c.j.CONFIG).read_text())
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=oldcfg[key]
    recurrence=[];items=list(models.items())
    for first in range(0,len(items),3):
        subset=items[first:first+3];folder=dest/f'recurrence_group_{first//3}';folder.mkdir(exist_ok=True)
        c.j.i.old.evaluate(root,folder,{**evaluation,'architectures':[n for n,m in subset]},[m for n,m in subset],torch,device,json.loads((root/p.CONFIG).read_text()))
        recurrence+=json.loads((folder/'recurrence_evaluation.json').read_text())
    p.save_json(dest/'recurrence_evaluation.json',recurrence)
    p.save_json(dest/'factorial_effects.json',factorial(rows))
    result=dict(valid=True,panels=len(rows),diagnostic_only=True,selection_eligible=False,thresholds_changed=False,primary_policy_changed=False,native_floor_requires_interpretation=True)
    p.save_json(dest/'summary.json',result);return result
