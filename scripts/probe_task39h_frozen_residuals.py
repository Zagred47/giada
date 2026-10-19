"""Post-hoc CPU localization on frozen H8; no fitting or selection changes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task39h_relaxation_identification as h
from analyze_task39g_result import write
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results/task39h_colab_v1';p=h.p

def main():
    torch.set_num_threads(2)
    spec=json.loads((OUT/'preregistration.json').read_text());quality=json.loads((ROOT/p.CONFIG).read_text())
    freeze=json.loads((OUT/'selection_freeze.json').read_text());weights=torch.load(OUT/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)
    aa=[a for a in freeze['arms'] if a['learning_rate']==.0001];start=[a['name'] for a in aa].index('H8')*3
    model=h.f.model_class(torch)('wide',spec['seeds']);model.load_state_dict({k:v if k=='gbar' else v[start:start+3] for k,v in weights['lr0001'].items()});model.eval()
    source=json.loads((OUT/'recurrence_evaluation.json').read_text());targets=[r for r in source if r['model']=='joint_wH8' and r['role']=='development' and not r['metrics']['passed']]
    rows=[];reproduction=[]
    for row in targets:
        case=row['case'];seed=spec['seeds'].index(row['seed'])
        with np.load(ROOT/spec['dataset']/f'development__{row["family"]}.npz') as zz:a={k:zz[k] for k in zz.files}
        v=a['voltage'][:,case];ca=a['bridge_ca'][:,case];target=a['gate_targets'][:,case];T=len(v)
        with torch.no_grad():
            us=[];zs=[]
            for i in range(0,T,2048):
                # Match original model batch shape and chunking to audit reproduction.
                vv=torch.tensor(a['voltage'][i:i+2048].reshape(-1))[None].expand(3,-1);cc=torch.tensor(a['bridge_ca'][i:i+2048].reshape(-1))[None].expand(3,-1)
                u,z,_=model.coefficients(vv,cc);us.append(u.reshape(3,-1,a['voltage'].shape[1],18)[seed,:,case].numpy());zs.append(z.reshape(3,-1,a['voltage'].shape[1],18)[seed,:,case].numpy())
        u=np.concatenate(us);z=np.concatenate(zs);tu,tz=p.coefficients(v.astype(float),ca.astype(float),.1);tu=tu.astype('float32');tz=tz.astype('float32')
        failed=[j for j,c in enumerate(row['metrics']['channels']) if not c['passed']];gates=sorted({g for c in failed for g in range(p.OFFSETS[c],p.OFFSETS[c+1])})
        modes=['learned','teacher_u','teacher_z','teacher_both','reset8','reset64']+[f'{kind}_gate{gate}' for gate in gates for kind in ('u','z')]
        state=np.broadcast_to(a['initial_gates'][case],(len(modes),18)).astype('float32').copy();pred=np.empty((T,len(modes),18),dtype='float32')
        for t in range(T):
            uu=np.broadcast_to(u[t],state.shape).copy();zz=np.broadcast_to(z[t],state.shape).copy()
            for j,mode in enumerate(modes):
                if mode.startswith('reset') and t and t%int(mode[5:])==0:state[j]=target[t-1]
                if mode in ('teacher_u','teacher_both'):uu[j]=tu[t]
                if mode in ('teacher_z','teacher_both'):zz[j]=tz[t]
                if '_gate' in mode:
                    kind,idx=mode.split('_gate');(uu if kind=='u' else zz)[j,int(idx)]=(tu if kind=='u' else tz)[t,int(idx)]
            state=state+zz*(uu-state);pred[t]=state
        for j,mode in enumerate(modes):
            _,cur=p.readout(v,pred[:,j],a['multipliers'][case],a['reversals'][case])
            metrics=p.evaluate_currents(a['current_targets'][:,case],cur,.1,quality['quality'],target,pred[:,j],a['multipliers'][case]==0)
            if mode=='learned':
                reference=row['metrics'];delta=max(abs(c['nrmse']-r['nrmse']) for c,r in zip(metrics['channels'],reference['channels']))
                events_equal=all(c['event']==r['event'] for c,r in zip(metrics['channels'],reference['channels']))
                reproduction.append(dict(seed=row['seed'],family=row['family'],case=case,max_nrmse_difference=delta,events_equal=events_equal,pass_equal=metrics['passed']==reference['passed']))
                # This is a CPU diagnostic; require unchanged failure classification and tightly matching errors.
                assert events_equal and metrics['passed']==reference['passed'] and delta<1e-5,reproduction[-1]
            rows.append(dict(parent='39h_H8',seed=row['seed'],family=row['family'],case=case,intervention=mode,selection_eligible=False,metrics=metrics))
        print(json.dumps(dict(family=row['family'],case=case,seed=row['seed'],probes=len(modes))),flush=True)
    result=dict(valid=True,post_hoc=True,selection_eligible=False,training_performed=False,device='CPU',checkpoint_sha256=hashlib.sha256((OUT/'frozen_checkpoint.pt').read_bytes()).hexdigest(),reproduction=reproduction,rows=rows,
        limits='Diagnostic H8 interventions restricted to7 failed development cases; no inference deployability or full72 qualification. CPU numerical reproduction tolerance1e-5 in NRMSE is diagnostic,not a quality threshold change.')
    write(OUT/'posthoc_h8_oracles.json',result)
    print(json.dumps(dict(valid=True,rows=len(rows),passes={mode:sum(r['metrics']['passed'] for r in rows if r['intervention']==mode) for mode in modes[:6]})),flush=True)
if __name__=='__main__':main()
