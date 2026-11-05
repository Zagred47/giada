"""Existing-head repair only; paired arms/seeds, no teacher at inference."""
import copy,json,os,time,zipfile,io
from pathlib import Path
import numpy as np
from . import task41n_evaluation_recovery as r
n=r.n
EQ=(8,9,11,12,13,16)
COLS=(8,9,11,12,13,25,26)
ARMS=('equilibrium','joint_nap')

def materialize(torch,base,wv,wc,arm):
    net=copy.deepcopy(base)
    with torch.no_grad():
        for j,col in enumerate(COLS):
            if arm=='equilibrium' and j>=5:continue
            net.parent.c[:,:,col].copy_(wv[:,:-1,j]);net.parent.d[:,col].copy_(wv[:,-1,j])
        net.parent.sc[:,:,0].copy_(wc[:,:-1,0]);net.parent.sd[:,0].copy_(wc[:,-1,0])
    for value in net.parameters():value.requires_grad_(False)
    return net.eval()

def pool(torch,base,spec,device):
    rng=np.random.default_rng(spec['fit_seed']);count=spec['fit_points']
    v=rng.uniform(-100,60,count);v[::2]=rng.uniform(-65,-30,len(v[::2]))
    ca=10**rng.uniform(-7,-2,count);ca[::2]=10**rng.uniform(-4.5,-3.2,len(ca[::2]))
    # Compute offline labels from the actual quantized causal inputs.
    v=v.astype('float32');ca=ca.astype('float32');u,z=n.p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    data={};logs=[];features=[];cfeatures=[]
    for start in range(0,count,1024):
        vv=torch.tensor(v[start:start+1024],device=device)[None].expand(3,-1);cc=torch.tensor(ca[start:start+1024],device=device)[None].expand(3,-1)
        with torch.no_grad():
            ru,rz=base.raw(vv,cc)
            logs.append(torch.stack([ru[...,k] for k in EQ[:-1]]+[rz[...,8],rz[...,9]],-1).double())
            x=torch.nn.functional.silu(torch.bmm(((vv+40)/80)[...,None],base.parent.a)+base.parent.b)
            y=torch.nn.functional.silu(torch.bmm(((cc.clamp_min(1e-9).log10()+4.5)/2.5)[...,None],base.parent.sa)+base.parent.sb)
            features.append(torch.cat((x,torch.ones_like(x[...,:1])),-1).double())
            cfeatures.append(torch.cat((y,torch.ones_like(y[...,:1])),-1).double())
            data.setdefault('sk_logits',[]).append(ru[...,16:17].double())
    data['fv']=torch.cat(features,1);data['fc']=torch.cat(cfeatures,1)
    data['raw']=torch.cat(logs,1);data['sk_logits']=torch.cat(data['sk_logits'],1)
    wv=torch.cat((base.parent.c[:,:,list(COLS)],base.parent.d[:,None,list(COLS)]),1).detach()
    wc=torch.cat((base.parent.sc[:,:,:1],base.parent.sd[:,None,:1]),1).detach()
    data['offset_v']=data['raw']-torch.bmm(data['fv'],wv.double())
    data['offset_c']=data['sk_logits']-torch.bmm(data['fc'],wc.double())
    data['u']=torch.tensor(u[:,list(EQ)],device=device);data['z']=torch.tensor(z[:,list(EQ)],device=device)
    # Other update fractions are frozen at their learned values in both arms.
    zs=[]
    for start in range(0,count,1024):
        vv=torch.tensor(v[start:start+1024],device=device)[None].expand(3,-1);cc=torch.tensor(ca[start:start+1024],device=device)[None].expand(3,-1)
        with torch.no_grad():zs.append(torch.sigmoid(base.raw(vv,cc)[1][...,list(EQ)]).double())
    data['base_z']=torch.cat(zs,1)
    data['initial']=torch.tensor(rng.uniform(0,1,(count,len(EQ))),device=device)
    data['input_v']=v;data['input_ca']=ca
    return data,wv,wc

def loss(torch,wv,wc,data,ids,joint):
    f=data['fv'][:,ids];fc=data['fc'][:,ids]
    raw=torch.einsum('snk,askh->asnh',f,wv.double())+data['offset_v'][:,ids][None]
    sk=torch.einsum('snk,askh->asnh',fc,wc.double())+data['offset_c'][:,ids][None]
    u=torch.sigmoid(torch.cat((raw[...,:5],sk),-1))
    zz=data['base_z'][:,ids][None].expand(2,-1,-1,-1).clone()
    # Actual architecture already has these heads; joint arm updates Nap only.
    zz=torch.cat((torch.sigmoid(raw[...,5:7]),zz[...,2:]),-1)
    wanted_u=data['u'][ids][None,None];wanted_z=data['z'][ids][None,None]
    eq=((u-wanted_u)/1e-4).square().mean((2,3))
    rate=((torch.log(zz[...,:2].clamp_min(1e-15))-torch.log(wanted_z[...,:2].clamp_min(1e-15)))/.01).square().mean((2,3))*joint
    # Multiple held-input horizons train persistent bias without long BPTT.
    initial=data['initial'][ids][None,None];terms=[]
    for ticks in (10,80,400,1000,5000,10000):
        pred=u+(initial-u)*torch.exp(torch.log1p(-zz.clamp_max(1-1e-15))*ticks)
        target=wanted_u+(initial-wanted_u)*torch.exp(torch.log1p(-wanted_z.clamp_max(1-1e-15))*ticks)
        terms.append(((pred-target)/1e-4).square().mean((2,3)))
    return eq+rate+.25*torch.stack(terms).mean(0)

def stress(torch,net,spec,out,label,device):
    folder=out/label;folder.mkdir()
    report=n.evaluate_stress(torch,{'dual':net},spec,folder,device)['dual']
    return dict(passed=sum(x['metrics']['passed'] for x in report['rows']),total=len(report['rows']),
        failed=[(x['model_seed'],x['family'],x['input_seed'],x['history'],x['panel']) for x in report['rows'] if not x['metrics']['passed']])

def run(torch,root,source,recovery_archive,out,device='cuda'):
    out.mkdir(exist_ok=False);spec=json.loads((root/'experiments/task41p_preregistration_v1.json').read_text())
    n.p.save_json(out/'preregistration.json',spec)
    plan,prior,freeze,floors=r.verify_source(root,source,out.parent/'verified_original_41p')
    assert spec['quality']==prior['quality'] and r.sha(recovery_archive)==spec['recovery_sha256']
    os.environ['GIADA_TASK41N_FROZEN']=str(out.parent/'verified_original_41p/frozen_source.zip.bin')
    _,cfg,calcium,base=n.contract(root);cfg=n.old_native.x.calibration_config(prior,cfg)
    nets,parent=n.load_models(torch,root,device);baseline=nets['dual'];baseline_hash=n.weights_hash(baseline)
    pre=n.parity(torch,nets,device,prior);assert pre['eligible']==freeze['eligible'];n.p.save_json(out/'gpu_preflight.json',pre)
    fit,wv0,wc0=pool(torch,baseline,spec,device)
    assert n.weights_hash(materialize(torch,baseline,wv0,wc0,'joint_nap'))==baseline_hash
    np.savez_compressed(out/'fit_inputs.npz',voltage=fit.pop('input_v'),calcium=fit.pop('input_ca'))
    wv=torch.nn.Parameter(wv0[None].repeat(2,1,1,1));wc=torch.nn.Parameter(wc0[None].repeat(2,1,1,1))
    optimizer=torch.optim.Adam([wv,wc],lr=spec['learning_rate'])
    mask=torch.ones_like(wv);mask[0,:,:,5:]=0;joint=torch.tensor([[0.],[1.]],device=device)
    development=dict(prior,stress_seeds=spec['development_seeds']);baseline_dev=stress(torch,baseline,development,out,'baseline_development',device)
    generator=np.random.default_rng(spec['minibatch_seed']);records=[];start=time.time()
    for step in range(1,max(spec['checkpoints'])+1):
        ids=torch.tensor(generator.integers(0,spec['fit_points'],spec['batch_size']),device=device)
        optimizer.zero_grad(set_to_none=True);scores=loss(torch,wv,wc,fit,ids,joint);scores.sum().backward()
        wv.grad.mul_(mask)
        # Independent clipping and Adam moments for each arm and each seed.
        norm=(wv.grad.square().sum((2,3))+wc.grad.square().sum((2,3))).sqrt().clamp_min(1e-30)
        factor=(spec['gradient_clip']/norm).clamp_max(1)
        wv.grad.mul_(factor[:,:,None,None]);wc.grad.mul_(factor[:,:,None,None]);optimizer.step()
        assert torch.isfinite(wv).all() and torch.isfinite(wc).all()
        if step%250==0:print(f'[GIADA41p] fit {step}/{max(spec["checkpoints"])} loss={scores.detach().cpu().tolist()}',flush=True)
        if step in spec['checkpoints']:
            torch.save(dict(step=step,wv=wv.detach().cpu(),wc=wc.detach().cpu(),optimizer=optimizer.state_dict()),out/f'training_{step}.pt')
            for a,name in enumerate(ARMS):
                net=materialize(torch,baseline,wv[a].detach(),wc[a].detach(),name)
                torch.save(net.state_dict(),out/f'{name}_{step}.pt')
                result=stress(torch,net,development,out,f'development_{name}_{step}',device)
                records.append(dict(arm=name,step=step,development=result))
            n.p.save_json(out/'development_results.json',records)
    # All checkpoints reported; selection depends ONLY on new development rows.
    chosen={name:max([x for x in records if x['arm']==name],key=lambda x:(x['development']['passed'],-x['step'])) for name in ARMS}
    n.p.save_json(out/'selection_freeze.json',dict(selection=chosen,checkpoint_sha256={name:r.sha(out/f'{name}_{chosen[name]["step"]}.pt') for name in ARMS},opened41n_used_for_selection=False,rule='highest new-development pass count, then earliest budget; report every checkpoint; no primary qualification'))
    with zipfile.ZipFile(recovery_archive) as z:
        old_results=json.loads(z.read('autonomous_results.json'))
        old_stress=json.loads(z.read('fresh_stress_results.json'))['dual']['rows']
    old_stress_failed={(x['model_seed'],x['family'],x['input_seed'],x['history'],x['panel']) for x in old_stress if not x['metrics']['passed']}
    cache=out.parent/'verified_original_41p'
    with np.load(cache/'native_shadow_traces.npz') as a:data=dict(a)
    refs={}
    for policy in ('primary','voltage_solve64'):
        with np.load(cache/f'reference_autonomous_{policy}.npz') as a:refs[policy]=dict(a)
    meta=json.loads((cache/'cases.json').read_text());final={};trained={}
    imposed={k:data[k].T for k in ('voltage','calcium')}
    imposed_refs={policy:n.m.rollout(torch,None,data,(cfg,calcium,base),device,policy,imposed) for policy in refs}
    for name in ARMS:
        net=copy.deepcopy(baseline);net.load_state_dict(torch.load(out/f'{name}_{chosen[name]["step"]}.pt',map_location=device,weights_only=True));trained[name]=net
        opened=stress(torch,net,prior,out,'opened41n_'+name,device)
        pred=n.m.rollout(torch,net,data,(cfg,calcium,base),device,'voltage_solve64')
        native={}
        for ref,truth in refs.items():
            metrics=n.m.measure(pred,truth,data,meta,prior,cfg,'autonomous','41n');native[ref]=n.m.summary(metrics)
            n.p.save_json(out/f'native_{name}_{ref}.json',metrics)
        imposed_pred=n.m.rollout(torch,net,data,(cfg,calcium,base),device,'voltage_solve64',imposed)
        imposed_summaries={}
        for ref,truth in imposed_refs.items():
            metrics=n.m.measure(imposed_pred,truth,data,meta,prior,cfg,'imposed','41n');imposed_summaries[ref]=n.m.summary(metrics)
            n.p.save_json(out/f'imposed_{name}_{ref}.json',metrics)
        before=set(map(tuple,baseline_dev['failed']));after=set(map(tuple,chosen[name]['development']['failed']))
        # Case-level no-regression protection, never only an average count.
        old_failed={tuple((x['case'],x['seed'])) for x in old_results['dual/autonomous/primary']['summary']['failed_responses']}
        regressions=[x for x in native['primary']['failed_responses'] if (x['case'],x['seed']) not in old_failed]
        opened_regressions=set(map(tuple,opened['failed']))-old_stress_failed
        progress=opened['passed']>61 and not opened_regressions and not (after-before) and not regressions and native['voltage_solve64']['responses_passed']==native['voltage_solve64']['responses_total'] and all(x['trajectory_passed']==x['trajectory_total'] for x in native.values()) and all(x['passed'] for x in imposed_summaries.values())
        final[name]=dict(development=chosen[name],opened_stress=opened,native=native,imposed=imposed_summaries,opened_stress_regressions=sorted(opened_regressions),development_regressions=sorted(after-before),native_response_regressions=regressions,concrete_diagnostic_progress=progress)
    # Same timer, same original formula comparator and batches; no oracle cost.
    costs=n.benchmark(torch,dict(dual=baseline,**trained),parent,data,(cfg,calcium,base),prior,out,device)
    assert n.weights_hash(baseline)==baseline_hash
    report=dict(valid=True,complete=True,task='41p',decision='TARGETED_REPAIR_COMPLETE_REQUIRES_AUDIT',arms=final,baseline_development=baseline_dev,
        checkpoints=records,training_updates_per_arm=max(spec['checkpoints']),baseline_unchanged=True,all_parameters_finite=True,
        thresholds_changed=False,abstraction_changed=False,inference_parameter_count_unchanged=True,teacher_at_inference=False,
        original_fp32_floor_failed_checks=[218,234],original_confirmation_reinstated=False,independent_confirmation=False,task42_authorized=False,speedup_claim=False,
        cost_rows=len(costs),elapsed_seconds=time.time()-start,limits=spec['limits'])
    n.p.save_json(out/'final_report.json',report);return report
