"""FIT-only nested support and stratified paired sampling for Task41l."""
import copy
import numpy as np
from . import task41k_learning_alignment as k
j=k.j


def feedback_pool(torch,spec,physics,device,small=False):
    raw=j.f.ed.history_pool(torch,spec,*physics,device,small)
    labels=j.f.ed.reference_windows(torch,raw,spec,*physics,device)
    count=26 if small else spec['feedback_anchors']
    rng=np.random.default_rng(spec['new_seeds']['support'])
    order=rng.permutation(len(raw['v']))
    # Every action is represented; remainder is sampled without replacement.
    first=[int(next(i for i in order if raw['action'][i]==group)) for group in range(13)]
    chosen=np.array((first+[int(i) for i in order if i not in first])[:count])
    assert len(chosen)==count and set(raw['action'][chosen])==set(range(13))
    data=j.f.ed.tensor_inputs(torch,{key:raw[key][chosen] for key in ('v','ca','g','mult','inj','syn')},device,torch.float64)
    data.update(target=torch.tensor(labels['delta'][:,chosen],device=device),
                family=torch.tensor(raw['action'][chosen],device=device),
                history=torch.tensor(raw['history'][chosen],device=device),
                time_tick=torch.tensor(raw['time_tick'][chosen],device=device))
    data['scale']=data['target'].abs().amax(0).clamp_min(1e-7)
    return data


def path_pool(torch,spec,device,small=False):
    data=j.path_pool(torch,spec,device,count=8 if small else None)
    data['family']=(data['v'].amax(1)!=data['v'].amin(1)).long()
    assert set(data['family'].tolist())=={0,1},'FIT must contain constant and return-ramp families'
    return data


def combine(torch,blocks,feedback=False):
    return {key:torch.cat([block[key] for block in blocks],1 if feedback and key=='target' else 0) for key in blocks[0]}


def nested_pools(torch,spec,physics,device,small=False):
    paths=[];feedback=[]
    for block in range(4):
        local=copy.deepcopy(spec)
        local['new_seeds']={key:value+block*10000 for key,value in spec['new_seeds'].items()}
        paths.append(path_pool(torch,local,device,small))
        feedback.append(feedback_pool(torch,local,physics,device,small))
    return {0:(paths[0],feedback[0]),1:(combine(torch,paths),combine(torch,feedback,True))}


def family_samplers(errors,families,cutoffs=None):
    errors=np.asarray(errors);families=np.asarray(families)
    assert np.isfinite(errors).all()
    result={};thresholds={}
    for family in sorted(set(families.tolist())):
        ids=np.flatnonzero(families==family);values=errors[ids]
        # Stable threshold inclusive of ties. Calibrate once on 1x FIT only.
        threshold=float(np.sort(values)[max(0,len(values)-max(1,len(values)//4))]) if cutoffs is None else cutoffs[str(family)]
        top=values>=threshold
        assert top.any(),('empty high-error stratum',family)
        weights=np.full(len(ids),.5/len(ids));weights[top]+=.5/top.sum()
        cdf=np.cumsum(weights);cdf[-1]=1.
        result[str(family)]=dict(ids=ids.tolist(),cdf=cdf.tolist())
        thresholds[str(family)]=threshold
    return result,thresholds


def draw(torch,sampler,u,device,stratified=False):
    groups=sorted(sampler,key=int)
    if stratified:
        assert len(u)==len(groups)
        choices=list(range(len(groups)));within=np.asarray(u)
    else:
        scaled=np.asarray(u)*len(groups);choices=np.floor(scaled).astype(int);within=scaled-choices
    selected=[]
    for group,q in zip(choices,within):
        row=sampler[groups[group]]
        selected.append(row['ids'][min(np.searchsorted(row['cdf'],q),len(row['ids'])-1)])
    return torch.tensor(selected,device=device,dtype=torch.long)


def minibatches(torch,rng,pools,samplers,spec,device):
    # Reuse exact random variates across all A/P and paired quantiles across D.
    pu=rng.random(spec['tail_batch']);fu=rng.random(spec['feedback_batch'])
    pprotect=rng.random(2);fprotect=rng.random(13)
    return {level:(draw(torch,samplers[level][0],pu,device),draw(torch,samplers[level][1],fu,device),
                   draw(torch,samplers[level][0],pprotect,device,True),draw(torch,samplers[level][1],fprotect,device,True)) for level in pools}


def group_losses(torch,model,paths,fb,pi,fi,physics):
    pp=k.path_prediction(torch,model,paths,pi);fp=k.feedback_prediction(torch,model,fb,fi,physics)
    delta=pp-paths['target'][pi][None]
    pe=(delta/paths['scale'][pi][None,:,None]).square().mean(2)
    timing=(delta[:,:,8:].diagonal(dim1=2,dim2=3)/paths['timing_scale'][pi][None]).square().clamp_max(1e6)
    valid=paths['valid'][pi]
    fe=((fp-fb['target'][:40,fi][None])/fb['scale'][fi][None,None]).square().mean(1)
    rows=[];names=[]
    for group in range(2):
        mask=paths['family'][pi]==group
        assert bool(mask.any()),('empty protection path group',group)
        rows.append(pe[:,mask].mean(1)+.01*(timing[:,mask]*valid[mask][None]).sum(1)/valid[mask].sum(0).clamp_min(1)[None])
        names.extend([f'path{group}/{channel}' for channel in j.c.ionic.CHANNELS])
    for group in range(13):
        mask=fb['family'][fi]==group
        assert bool(mask.any()),('empty protection feedback group',group)
        rows.append(fe[:,mask].mean(1))
        names.extend([f'action{group}/{channel}' for channel in j.c.ionic.CHANNELS])
    result=torch.cat(rows,1)
    assert result.shape==(3,165)
    return result,names


def objective(torch,model,paths,fb,batch,physics,protect):
    pi,fi,ppi,ffi=batch
    base,_=k.dynamic_losses(torch,model,paths,fb,pi,fi,physics)
    if protect:
        groups,names=group_losses(torch,model,paths,fb,ppi,ffi,physics)
        worst,indices=groups.max(1) # first flattened group wins exact ties
        total=base+worst
    else:
        with torch.no_grad():groups,names=group_losses(torch,model,paths,fb,ppi,ffi,physics)
        worst,indices=groups.max(1);total=base
    return total,dict(base=base.detach().tolist(),worst=worst.detach().tolist(),worst_group=[names[i] for i in indices.tolist()])
