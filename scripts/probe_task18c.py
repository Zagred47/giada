"""Localize consumed Task18c frozen errors without selecting a new model."""
import io,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.giada_teacher.hh_family_transfer import model_factory,rates,update,write
from src.giada_teacher.hh_potassium_local_repair import arms
torch.set_num_threads(1)
with zipfile.ZipFile(sys.argv[1]) as z:
    f=json.loads(z.read('selection_freeze.json'));cfg=f['config'];sel=f['selected'];ds=[('K_Pst',f'{s}/top{k}/lr{lr}',seed) for s,k,lr in arms(cfg) for seed in cfg['seeds']]
    m=model_factory(torch,32,ds);m.load_state_dict(torch.load(io.BytesIO(z.read(f"checkpoint_step{sel['step']}.pt")),map_location='cpu',weights_only=True));fresh=np.load(io.BytesIO(z.read('fresh.npz')));x=fresh['kink'];ii,tt=rates('K_Pst',x[:,0]);y=update(x[:,1:3],ii,tt,x[:,3])
    with torch.no_grad():p,i,t=[a.double().numpy() for a in m(torch.tensor(x,dtype=torch.float32)[None].expand(len(ds),-1,-1))]
    rows=[]
    for n in sel['indices']:
        q,g=np.unravel_index(abs(p[n]-y).argmax(),y.shape)
        rows.append(dict(seed=ds[n][2],voltage_mv=float(x[q,0]),gate=['m','h'][g],dt_ms=float(x[q,3]),error=float(abs(p[n,q,g]-y[q,g])),exact_tau_error=float(abs(update(x[:,1:3],i[n],tt,x[:,3])[q,g]-y[q,g]))))
    out={'valid':True,'confirmation':False,'rows':rows};write(sys.argv[2],out);print(json.dumps(out))
