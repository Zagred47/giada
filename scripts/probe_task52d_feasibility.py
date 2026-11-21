"""Frozen decoder witnesses; free latent points are not a deployable model."""
import copy,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import itertools
import torch
from src.giada_teacher import task52c_component_repair as c
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'experiments/results/task52d_protocol_v1'
def main():
    torch.set_num_threads(1);archive=ROOT/'artifacts/task52c_kaggle_external/task52c_final.zip'
    receipt=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;p=json.loads(z.read('preregistration.json'));state=torch.load(io.BytesIO(z.read('selected.pt')),map_location='cpu',weights_only=True);cal=torch.load(io.BytesIO(z.read('calibration.pt')),map_location='cpu',weights_only=True)
        bounds=json.loads(z.read('selected_decoder_bounds.json'));ref=dict(np.load(io.BytesIO(z.read('reference.npz'))));cases=json.loads(z.read('cases.json'))
    p=copy.deepcopy(p);p['arms']=[p['arms'][1]];p['source_indices']=[3,4,5]
    net=c.model_class(torch)(p,cal,state,'cpu');net.load_state_dict({k:v[3:6] for k,v in state.items()})
    for par in net.parameters():par.requires_grad_(False)
    before=c.a.n.weights_hash(net);parents=c.parent.plan(ROOT);quality=parents['quality'];bar=np.array(c.q.ionic.GBAR);rev=np.array(c.q.ionic.REVERSALS)
    voltage=[];target=[];scale=[];points=[];openings=[]
    for ci,case in enumerate(cases):
        if case['intervention']!='none':continue
        times=sorted(set(np.linspace(0,len(ref['current'])-1,24).astype(int).tolist()+np.abs(ref['current'][:,0,ci]).argmax(0).tolist()))
        peak=np.maximum(np.abs(ref['current'][:,0,ci]).max(0),quality['current_scale_floor_ma_cm2'])
        for ti in times:
            voltage.append(ref['voltage'][ti,0,ci]);target.append(ref['current'][ti,0,ci]);scale.append(peak);points.append(dict(case=ci,tick=ti));openings.append(ref['conductance'][ti,0,ci]/bar)
    v=torch.tensor(voltage,dtype=torch.float32);y=torch.tensor(np.array(target),dtype=torch.float32);den=torch.tensor(np.array(scale),dtype=torch.float32)
    bar=torch.tensor(bar,dtype=torch.float32);rev=torch.tensor(rev,dtype=torch.float32)
    n=len(v);best=torch.full((3,n),float('inf'));best_errors=torch.full((3,n,11),float('inf'));best_h=torch.zeros(3,n,18)
    logtarget=torch.tensor(np.log10(np.maximum(openings,1e-35)),dtype=torch.float32)
    for seed in range(3):
        gen=torch.Generator().manual_seed(952410+seed);x=torch.nn.Parameter(torch.randn((3,n,18),generator=gen)*.1 if seed else torch.zeros(3,n,18))
        opt=torch.optim.Adam([x],lr=.02)
        for step in range(3200):
            opt.zero_grad();h=(net.hm+net.hs*x)*net.lm;pred=net.opening(h)*bar*(v[None,:,None]-rev);err=(pred-y)/den
            loss=((net.logopening(h)-logtarget)/net.span).square().mean(-1)+.1*err.square().mean(-1);loss.sum().backward();opt.step()
            with torch.no_grad():x.copy_(torch.maximum(torch.minimum(x,(1-net.hm)/net.hs),-net.hm/net.hs));h=(net.hm+net.hs*x)*net.lm
            if step%20==0 or step==3199:
                with torch.no_grad():
                    pred=net.opening(h)*bar*(v[None,:,None]-rev);err=(pred-y)/den;score=err.abs().amax(-1);mask=score<best;best[mask]=score[mask];best_errors[mask]=err.abs()[mask];best_h[mask]=h[mask]
        print('52d decoder witness start',seed,'best max',float(best.max()),flush=True)
    assert c.a.n.weights_hash(net)==before
    adam_pass=int((best<=quality['current_nmax_max']).sum())
    # Independent bounded trust-region solves per family; analytic Jacobian,
    # normalized coordinates and no artificial zero-gradient upper clamp.
    from scipy.optimize import least_squares
    from scipy.special import expit
    candidate=best_h.detach().numpy().copy();offset=0
    for family,channels in p['families'].items():
        width=p['arms'][0]['widths'][list(p['families']).index(family)];sl=list(range(offset,offset+width));offset+=width
        for s in range(3):
            cols=np.flatnonzero(net.wm[s,sl].detach().numpy().any(0));W=net.rw[s].detach().numpy()[np.ix_(sl,cols)].astype(float);O=net.ro[s].detach().numpy()[np.ix_(cols,channels)].astype(float);S=net.skip[s].detach().numpy()[np.ix_(sl,channels)].astype(float)
            bias=net.bias[s,0,channels].detach().numpy().astype(float);bb=net.rb[s,0,cols].detach().numpy().astype(float);hm=net.hm[s,0,sl].numpy();hs=net.hs[s,0,sl].numpy();center=net.center[s,0,channels].numpy();span=net.span[s,0,channels].numpy();latent_bounds=((-hm/hs).astype(float),((1-hm)/hs).astype(float))
            axes=[np.unique(np.clip(np.r_[np.linspace(0,1,1001 if width==1 else 11),hm[j]+hs[j]*np.array([-10,-3,-1,0,1,3,10])],0,1)) for j in range(width)]
            gridh=np.array(list(itertools.product(*axes)));grid=(gridh-hm)/hs;zz=grid@W+bb;gridraw=(zz*expit(zz))@O+grid@S+bias;gridlogs=np.minimum(center+span*gridraw,0)
            for point in range(n):
                goal=logtarget[point,channels].numpy();coef=((v[point]-rev[channels])*bar[channels]).numpy();scale=den[point,channels].numpy();yy=y[point,channels].numpy()
                def calc(xx):
                    z=xx@W+bb;sig=expit(z);raw=(z*sig)@O+xx@S+bias;logs=center+span*raw;jac=(W*(sig+z*sig*(1-sig))[None])@O+S
                    pred=10**np.minimum(logs,0)*coef
                    residual=np.r_[raw-(goal-center)/span,.25*(pred-yy)/scale]
                    cj=(np.log(10)*pred/scale*(logs<0)*span)[None]*jac
                    return residual,np.vstack((jac.T,.25*cj.T))
                griderr=(10**gridlogs*coef-yy)/scale;gridscore=np.max(np.abs(griderr),axis=1);top=np.argsort(gridscore)[:3]
                starts=[grid[j] for j in top]+[np.clip(np.zeros(width),*latent_bounds),np.clip((candidate[s,point,sl]-hm)/hs,*latent_bounds)]
                score=float(gridscore[top[0]]);winner=starts[0]
                for initial in starts:
                    result=least_squares(lambda xx:calc(xx)[0],np.clip(initial,*latent_bounds),jac=lambda xx:calc(xx)[1],bounds=latent_bounds,max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                    value=float(np.max(np.abs(calc(result.x)[0][len(channels):]))/.25)
                    if value<score:score=value;winner=result.x
                candidate[s,point,sl]=np.clip(hm+hs*winner,0,1)
        print('52d bounded family solved',family,flush=True)
    with torch.no_grad():
        h=torch.tensor(candidate,dtype=torch.float32);err=(net.opening(h)*bar*(v[None,:,None]-rev)-y).abs()/den;score=err.amax(-1);mask=score<best;best[mask]=score[mask];best_errors[mask]=err[mask];best_h[mask]=h[mask]
    lower=np.array(bounds['log_lower'])[3:6];upper=np.array(bounds['log_upper'])[3:6]
    # Broad interval test is necessary only. Numerical witnesses cover sampled points only.
    enclosure=bool((lower[:,None]<=logtarget.numpy()[None]).all() and (upper[:,None]>=logtarget.numpy()[None]).all())
    passed=best<=quality['current_nmax_max']
    result=dict(valid=True,source_sha256=receipt['sha256'],weights_unchanged=True,logical_state=10,interval_contains_sample_targets=enclosure,adam_point_witness_passed=adam_pass,point_witness_passed=int(passed.sum()),point_witness_total=passed.numel(),maximum_normalized_error=float(best.max()),per_seed_max=best.amax(1).tolist(),all_sample_witnesses_pass=bool(passed.all()),points=points,point_error=best_errors.tolist(),qualification='Numerical free-latent witnesses at fixed representative points only; not a proof for all states, temporal realizability, recurrent learnability or full threshold acceptance. Failed optimization is inconclusive, not impossibility.')
    DEST.mkdir(exist_ok=True);c.a.n.p.save_json(DEST/'decoder_feasibility.json',result);torch.save(best_h.detach(),ROOT/'artifacts/task52d_decoder_witnesses.pt')
    print(json.dumps({k:v for k,v in result.items() if k not in ('points','point_error')}))
if __name__=='__main__':main()
