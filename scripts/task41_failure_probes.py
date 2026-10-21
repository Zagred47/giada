"""Targeted post-hoc probes and figures; original gates/verdict remain unchanged."""
import io,json,zipfile,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_task41 import OUT,ARCHIVE,p,t,x,recurrence,event_detail,save

def main():
    torch.set_num_threads(1);spec=json.loads((ROOT/x.CONFIG).read_text());q=spec['current_quality']
    with zipfile.ZipFile(ARCHIVE) as z:
        arrays=lambda name:dict(np.load(io.BytesIO(z.read(name))))
        native=arrays('native_shadow_traces.npz');pred=arrays('candidate_traces.npz');ref=arrays('reference_traces.npz');stress=arrays('fresh_stress_traces.npz')
    model=x.load_b6(torch,ROOT,'cpu');case=161;k=3;ga,gb=p.OFFSETS[k:k+2]
    v=native['voltage'][case,:-1].astype('float32');ca=native['calcium'][case,:-1].astype('float32');initial=native['gates'][case,0]
    with torch.inference_mode():
        ru,rz=model.raw(torch.tensor(v)[None].expand(3,-1),torch.tensor(ca)[None].expand(3,-1))
        u,z=torch.sigmoid(ru).numpy().transpose(1,0,2),torch.sigmoid(rz).numpy().transpose(1,0,2)
    tu,tz=p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    variants={}
    for name,uu,zz,dtype in [('learned32',u,z,np.float32),('learned_state64',u,z,np.float64),
        ('oracle_u',np.broadcast_to(tu[:,None],u.shape),z,np.float64),
        ('oracle_z',u,np.broadcast_to(tz[:,None],z.shape),np.float64)]:
        gates=recurrence(uu,zz,np.broadcast_to(initial,(3,18)),dtype)
        variants[name]=p.readout(v[:,None],gates,native['multipliers'][case],x.ionic.REVERSALS)[1][:,:,k]
    for target in ('u','z'):
        for gate in (ga,ga+1):
            a,b=u.astype('float64'),z.astype('float64')
            (a if target=='u' else b)[:,:,gate]=(tu if target=='u' else tz)[:,gate,None]
            gates=recurrence(a,b,np.broadcast_to(initial,(3,18)),np.float64)
            variants[f'oracle_{target}_gate{gate}']=p.readout(v[:,None],gates,native['multipliers'][case],x.ionic.REVERSALS)[1][:,:,k]
    reference=ref['imposed__current'][:,case,k]
    for label,g in [('oracle_m',ga),('oracle_h',ga+1)]:
        gates=pred['B6__imposed__gates'][1:,:,case].copy()
        gates[:,:,g]=ref['imposed__gates'][1:,case,g][:,None]
        variants[label]=p.readout(v[:,None],gates,native['multipliers'][case],x.ionic.REVERSALS)[1][:,:,k]
    variants['saved_GPU']=pred['B6__imposed__current'][:,:,case,k]
    details={name:{str(seed):event_detail(reference,current[:,s],q) for s,seed in enumerate(spec['model_seeds'])} for name,current in variants.items()}
    pp=variants['saved_GPU'][:,0];rr=np.abs(reference-reference[0]);amplitude=rr.max();threshold=.1*amplitude
    neighborhood=[dict(t_ms=j*.1,reference_current=float(reference[j]),predicted_current=float(pp[j]),
        reference_threshold_margin=float(rr[j]-threshold),predicted_threshold_margin=float(abs(pp[j]-reference[0])-threshold)) for j in range(167,174)]
    save('nats2_coefficient_interventions.json',dict(post_hoc=True,candidate_eligible=False,case=case,
        interventions=details,second_pulse_samples=neighborhood,
        no_waiver='Single-sample secondary threshold crossing remains an official failure; no smoothing, minimum-duration rule or threshold adjustment applied.'))
    # Per-gate coefficient oracle within each affected channel. Other coefficients
    # remain learned; all interventions only diagnose opened histories.
    vv,cc=stress['voltage'],stress['calcium'];points,ix=np.unique(np.stack((vv,cc),-1).reshape(-1,2),axis=0,return_inverse=True)
    with torch.inference_mode():
        ru,rz=model.raw(torch.tensor(points[:,0])[None].expand(3,-1),torch.tensor(points[:,1])[None].expand(3,-1))
        uu=torch.sigmoid(ru).numpy()[:,ix].reshape(3,len(vv),12,18).transpose(1,0,2,3)
        zz=torch.sigmoid(rz).numpy()[:,ix].reshape(3,len(vv),12,18).transpose(1,0,2,3)
    tu,tz=p.coefficients(vv.astype('float64'),cc.astype('float64'),.1);rows=[]
    failures=json.loads((OUT/'stress_failures.json').read_text())
    gold=p.readout(vv,stress['target_gates'],stress['multipliers'],x.ionic.REVERSALS)[1]
    for f in failures:
        c=f['index'];s=spec['model_seeds'].index(f['model_seed']);ch=next(iter(f['channel_reasons']));ci=x.ionic.CHANNELS.index(ch)
        for target in ('u','z'):
            for gate in range(p.OFFSETS[ci],p.OFFSETS[ci+1]):
                a,b=uu[:,s,c].astype('float64'),zz[:,s,c].astype('float64')
                (a if target=='u' else b)[:,gate]=(tu if target=='u' else tz)[:,c,gate]
                g=recurrence(a,b,stress['initial'][c],np.float64)
                curr=p.readout(vv[:,c],g,stress['multipliers'][c],x.ionic.REVERSALS)[1]
                rows.append(dict(case=c,seed=f['model_seed'],channel=ch,coefficient=target,gate=gate,
                    event=event_detail(gold[:,c,ci],curr[:,ci],q)))
    save('stress_single_gate_oracles.json',dict(post_hoc=True,candidate_eligible=False,rows=rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,window in [(axs[0,0],(5,19)),(axs[0,1],(16.6,17.4))]:
        time=np.arange(len(reference))*.1
        ax.plot(time,reference*1e6,label='Formula reference')
        ax.plot(time,pp*1e6,label='B6 seed17',ls='--')
        ax.axhline((reference[0]-threshold)*1e6,color='black',ls=':',label='Registered threshold')
        ax.set(xlim=window,xlabel='Time (ms)',ylabel='NaTs2_t current (nA/cm²)')
        ax.legend(fontsize=8)
    axs[0,1].set(xlim=(16.98,17.02),ylim=(-.146,-.1445),title='Zoom: one sample at 17 ms crosses the threshold')
    axs[0,1].scatter([17,17],[reference[170]*1e6,pp[170]*1e6],color=['C0','C1'],zorder=4)
    for ax,c,s,ci,window in [(axs[1,0],11,2,4,(485,490)),(axs[1,1],7,2,6,(80,85))]:
        curr=p.readout(vv,stress['predicted_gates'][:,s],stress['multipliers'],x.ionic.REVERSALS)[1]
        tr=gold[:,c,ci];pr=curr[:,c,ci];threshold=.1*np.max(abs(tr-tr[0]));tm=np.arange(len(tr))*.1
        ax.plot(tm,abs(tr-tr[0])*1e6,label='Reference |I−I₀|')
        ax.plot(tm,abs(pr-tr[0])*1e6,label='B6 seed43 |I−I₀|',ls='--')
        ax.axhline(threshold*1e6,color='black',ls=':',label='Registered threshold')
        ax.set(xlim=window,xlabel='Time (ms)',ylabel=x.ionic.CHANNELS[ci]+' deviation (nA/cm²)')
        low,high=round(window[0]*10),round(window[1]*10)
        vals=np.r_[abs(tr[low:high]-tr[0]),abs(pr[low:high]-tr[0])]*1e6
        ax.set_ylim(vals.min()*.999,vals.max()*1.001);ax.legend(fontsize=8)
    fig.suptitle('Task41: secondary crossing versus slow offset bias\nSigned inward currents are negative; bottom panels show absolute baseline deviation')
    fig.savefig(OUT/'failure_traces.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(nats2={name:d['17']['metrics'] for name,d in details.items()},
        stress_oracles=[dict(seed=r['seed'],channel=r['channel'],coefficient=r['coefficient'],gate=r['gate'],offset_ms=r['event']['metrics']['offset_error_ms']) for r in rows]),indent=2))

if __name__=='__main__':main()
