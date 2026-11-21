"""Fit-only decoder diagnostic; no model training or fresh data access."""
import copy,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task52d_balance_trajectory as d
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(1)
    archive=ROOT/'artifacts/task52d_kaggle_external/task52d_final.zip'
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest==json.loads((archive.parent/'kaggle_receipt.json').read_text())['sha256']
    with zipfile.ZipFile(archive) as z:
        p=json.loads(z.read('preregistration.json'));state=torch.load(io.BytesIO(z.read('selected.pt')),map_location='cpu',weights_only=True);cal=torch.load(io.BytesIO(z.read('calibration.pt')),map_location='cpu',weights_only=True)
    p=copy.deepcopy(p);p['arms']=[p['arms'][2]];p['source_indices']=[6,7,8]
    net=d.c.model_class(torch)(p,cal,state,'cpu');net.load_state_dict({k:v[6:9].clone() for k,v in state.items()})
    for x in net.parameters():x.requires_grad_(False)
    before=d.a.n.weights_hash(net)
    p['roles']={'fit':p['roles']['fit']}
    with torch.no_grad():data=d.c.b.data(torch,p,'cpu')
    ids=torch.arange(min(12,len(data['meta']))).repeat(3);times=torch.cat([torch.zeros(12,dtype=torch.long),torch.full((12,),1500,dtype=torch.long),torch.full((12,),6500,dtype=torch.long)])
    gates=data['g'][times,ids];v=data['v'][times,ids];target=data['logop'][times,ids]
    with torch.no_grad():h0=net.initialize(gates[None].expand(3,-1,-1));normal=net.logopening(h0)
    best=torch.full_like(normal,1e20);best_h=h0.clone()
    for start in range(2):
        x=torch.nn.Parameter(h0.clone() if start==0 else torch.full_like(h0,.5));opt=torch.optim.Adam([x],lr=.005)
        for step in range(800):
            opt.zero_grad();pred=net.logopening(x);err=(pred-target).square();(err/net.span.square()).mean().backward();opt.step()
            with torch.no_grad():x.clamp_(0,1);x.mul_(net.lm)
            if step%20==0:
                with torch.no_grad():err=(net.logopening(x)-target).abs();improved=err.mean(-1)<best.mean(-1);best[improved]=err[improved];best_h[improved]=x[improved]
    # Ih uses one latent coordinate (8): exhaustive bounded scalar grid diagnostic.
    with torch.no_grad():
        grid=torch.full((3,10001,18),.5);grid[:,:,8]=torch.linspace(0,1,10001);grid*=net.lm
        logs=net.logopening(grid)[:,:,5];dist=(logs[:,:,None]-target[None,None,:,5]).abs();ih=dist.amin(1)
    assert d.a.n.weights_hash(net)==before
    result=dict(valid=True,source_sha256=digest,fit_only=True,fresh_accessed=False,weights_unchanged=True,logical_state=10,points=len(ids),encoded_log10_absolute_error_median=normal.sub(target).abs().median(1).values.tolist(),free_latent_log10_absolute_error_median=best.median(1).values.tolist(),Ih_grid_log10_absolute_error_max=ih.amax(1).tolist(),Ih_encoded_log10_absolute_error_median=normal.sub(target).abs()[:,:,5].median(1).values.tolist(),qualification='Pointwise numerical diagnostic, not a recurrent feasibility certificate. Decoder remains trainable in52e. Failed search is inconclusive; scalar grid resolution is finite.')
    out=ROOT/'experiments/results/task52e_protocol_v1';out.mkdir(exist_ok=True);d.a.n.p.save_json(out/'decoder_preflight.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
