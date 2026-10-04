"""Read-only diagnostic localization of consumed Task18b frozen checkpoints."""
import io,json,zipfile,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.giada_teacher.hh_family_transfer import model_factory,rates,update,write
from src.giada_teacher.hh_potassium_diagnosis import descriptors

def main():
    torch.set_num_threads(1)
    archive=Path(sys.argv[1]);out=Path(sys.argv[2]);rows=[]
    with zipfile.ZipFile(archive) as z:
        f=json.loads(z.read('selection_freeze.json'));sel=f['selected_common_configuration'];ds=descriptors([17,29,43])
        model=model_factory(torch,sel['width'],ds);model.load_state_dict(torch.load(io.BytesIO(z.read(f"checkpoint_w{sel['width']}_step{sel['step']}.pt")),map_location='cpu',weights_only=True))
        fresh=np.load(io.BytesIO(z.read('fresh.npz')))
        for domain in ('in_support','negative_tail'):
            x=fresh[domain];ti,tt=rates('K_Pst',x[:,0]);truth=update(x[:,1:3],ti,tt,x[:,3])
            with torch.no_grad():p,i,t=[a.double().numpy() for a in model(torch.tensor(x,dtype=torch.float32)[None].expand(len(ds),-1,-1))]
            for n in sel['indices']:
                err=abs(p[n]-truth);q,g=np.unravel_index(err.argmax(),err.shape)
                exact_inf=update(x[:,1:3],ti,t[n],x[:,3]);exact_tau=update(x[:,1:3],i[n],tt,x[:,3])
                rows.append(dict(seed=ds[n][2],domain=domain,worst=dict(voltage_mv=float(x[q,0]),state=x[q,1:3].tolist(),dt_ms=float(x[q,3]),gate=['m','h'][g],error=float(err[q,g]),predicted_inf=float(i[n,q,g]),true_inf=float(ti[q,g]),predicted_tau=float(t[n,q,g]),true_tau=float(tt[q,g]),error_with_exact_inf=float(abs(exact_inf[q,g]-truth[q,g])),error_with_exact_tau=float(abs(exact_tau[q,g]-truth[q,g]))),band_max={f'{lo}:{hi}':float(err[(x[:,0]>=lo)&(x[:,0]<hi)].max(initial=0)) for lo,hi in [(-135,-115),(-115,-80),(-80,-65),(-65,-55),(-55,-20),(-20,75)]}))
    report={'valid':True,'selection_eligible':False,'purpose':'Consumed fresh diagnostic; not independent confirmation','rows':rows};write(out,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
