"""Audit saved52c evidence; no training, selection or remote execution."""
import gzip,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task52c_component_repair as m
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'experiments/results/task52c_audit_v1'

def main():
    torch.set_num_threads(1);DEST.mkdir(exist_ok=True);folder=ROOT/'artifacts/task52c_kaggle_external';receipt=json.loads((folder/'kaggle_receipt.json').read_text());archive=folder/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    assert archive.stat().st_size==receipt['bytes'];arms=['D','DE','DR','DER'];answer={};traces={};finite={}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'));p=json.loads(z.read('preregistration.json'));cases=json.loads(z.read('cases.json'));ref=dict(np.load(io.BytesIO(z.read('reference.npz'))))
        for name in z.namelist():
            if name.endswith('.pt') and not name.endswith('resumable_training.pt'):
                state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True);finite[name]=all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v))
        assert all(finite.values())
        for name in ['frozen52b']+arms:
            rows=json.loads(gzip.decompress(z.read(name+'/evaluation.json.gz')));traces[name]=dict(np.load(io.BytesIO(z.read(name+'/traces.npz'))));trace=traces[name]
            assert all(np.isfinite(v).all() for v in trace.values())
            modes={};ci=4 if name=='frozen52b' else 0;im=5 if name=='frozen52b' else 1
            for mode in ('compact','compact_imposed','compact_ca_imposed','compact_reset'):
                selected=[r for r in rows if r['arm']==mode and r['horizon_ms']==1000]
                modes[mode]=dict(voltage_pooled=float(np.sqrt(np.mean([r['voltage_rmse_mv']**2 for r in selected]))),passed=sum(r['passed'] for r in selected),families={})
                for family,channels in p['families'].items():
                    cells=[r['current']['channels'][k] for r in selected for k in channels]
                    modes[mode]['families'][family]=dict(median_nrmse=float(np.median([x['nrmse'] for x in cells])),passed=sum(x['passed'] for x in cells),total=len(cells),missing_events=sum(x.get('event',{}).get('status')=='missing_event' for x in cells))
            protocols={}
            for protocol in sorted({r['protocol'] for r in rows}):
                rr=[r for r in rows if r['arm']=='compact' and r['horizon_ms']==1000 and r['protocol']==protocol]
                protocols[protocol]=dict(voltage_pooled=float(np.sqrt(np.mean([r['voltage_rmse_mv']**2 for r in rr]))),per_case=[{k:r[k] for k in ('seed','case','intervention','voltage_rmse_mv','calcium_rmse_mM','physical')} for r in rr])
            # On the common imposed trajectory there is no candidate-voltage feedback.
            err=trace['current'][:,:,im].astype(float)-ref['current'];net=err.sum(-1)
            independent=np.sqrt(np.mean(err**2,axis=0).sum(-1));net_rms=np.sqrt(np.mean(net**2,axis=0))
            cost=json.loads(z.read(name+'/cost.json'))['samples_ms'];latency={k:float(np.median(v)) for k,v in cost.items()}
            answer[name]=dict(modes=modes,protocols=protocols,imposed_net_current_error_rms=float(np.sqrt(np.mean(net**2))),imposed_net_current_error_by_case=net_rms.tolist(),imposed_cancellation_ratio=(net_rms/np.maximum(independent,1e-30)).tolist(),cost_median_ms=latency,atomic_speed_ratio=latency['canonical64']/latency['compact10_step_readout'])
        state=torch.load(io.BytesIO(z.read('selected.pt')),map_location='cpu',weights_only=True);cal=torch.load(io.BytesIO(z.read('calibration.pt')),map_location='cpu',weights_only=True)
        net=m.model_class(torch)(p,cal,state,'cpu');net.load_state_dict(state)
        for v in net.parameters():v.requires_grad_(False)
        # Reconstruct the frozen latent trajectory under saved imposed/autonomous inputs.
        probes={};N=ref['voltage'].shape[2];bar=torch.tensor(m.q.ionic.GBAR)
        _,_,_,rev=m.parent.design(m.parent.plan(ROOT));rev=torch.tensor(rev,dtype=torch.float32);_,_,mult,_=m.parent.design(m.parent.plan(ROOT));mult=torch.tensor(mult,dtype=torch.float32)
        for mode in ('imposed','autonomous'):
            h=net.initialize(torch.tensor(m.q.steady(-65.,1e-4),dtype=torch.float32)[None,None].expand(12,N,-1));moments=np.zeros((12,3));gain=[]
            for k in range(len(ref['current'])):
                vv=np.concatenate([ref['voltage'][k] if mode=='imposed' else traces[name]['voltage'][k,:,0] for name in arms],0)
                cc=np.concatenate([ref['calcium'][k] if mode=='imposed' else traces[name]['calcium'][k,:,0] for name in arms],0)
                v=torch.tensor(vv,dtype=torch.float32);ca=torch.tensor(cc,dtype=torch.float32)
                if k in (0,200,400,600,1000,2500,3500,7000,9999):
                    v.requires_grad_(True);newh=net.step(m.t.features(torch,v,ca),h)
                    current=net.opening(newh)*bar*mult[None]*(v[...,None]-rev[None]);derivative=torch.autograd.grad(current.sum(),v)[0]
                    gain.append(dict(tick=k,derivative=derivative.detach().tolist()));v=v.detach()
                with torch.no_grad():
                    h=net.step(m.t.features(torch,v,ca),h);x=((h-net.hm)/net.hs)[...,:10]
                    moments[:,0]+=(x*x).mean((1,2)).numpy();moments[:,1]+=((x.abs()>3).float().mean((1,2))).numpy();moments[:,2]=np.maximum(moments[:,2],x.abs().amax((1,2)).numpy())
            moments[:,:2]/=len(ref['current']);moments[:,0]=np.sqrt(moments[:,0]);probes[mode]=dict(standardized_latent_rms_fraction_over3_max=moments.tolist(),one_step_current_voltage_derivatives=gain)
        contrasts={};channel_names=['Ca_HVA','Ca_LVAst','NaTa_t','NaTs2_t','Nap_Et2','Ih','Im','K_Pst','K_Tst','SK_E2','SKv3_1']
        for left,right in [('D','DR'),('DE','DER')]:
            e0=traces[left]['current'][:,:,1].astype(float)-ref['current'];e1=traces[right]['current'][:,:,1].astype(float)-ref['current']
            # Exact algebraic allocation of the change in net-current MSE. Not causal.
            attribution=((e0.sum(-1)+e1.sum(-1))[...,None]*(e1-e0)).mean((0,1,2))
            assert np.isclose(attribution.sum(),np.mean(e1.sum(-1)**2-e0.sum(-1)**2),rtol=1e-8,atol=1e-20)
            per_case=((e0.sum(-1)+e1.sum(-1))*(e1[...,0]-e0[...,0])).mean(0)
            top=np.argsort(per_case.ravel())[-6:][::-1]
            contrasts[left+'_'+right]=dict(net_current_mse_change_by_channel=dict(zip(channel_names,attribution.tolist())),Ca_HVA_individual_rmse=[float(np.sqrt(np.mean(e[...,0]**2))) for e in (e0,e1)],top_Ca_HVA_allocation_cases=[dict(seed=[17,29,43][int(i)//len(cases)],**cases[int(i)%len(cases)],contribution=float(per_case.ravel()[i])) for i in top],qualification='Algebraic error covariance allocation, not proof that replacing this channel alone repairs voltage.')
        out=dict(valid=True,archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'],finite_checkpoints=finite,all_traces_finite=True,arms=answer,probes=probes,contrasts=contrasts,learning_curve=json.loads(z.read('learning_curve.json')),h0=json.loads(z.read('h0_diagnostics.json')),representation=json.loads(z.read('representation_diagnostics.json')),limits=['Imposed versus autonomous is a diagnostic contrast, not two equally deployable models.','Derivative is one-step local V sensitivity holding h/Ca fixed, not a full stability theorem.','Standardized latent values use frozen fit mean/std; >3 is descriptive, not an OOD proof.','No training or selection performed.'])
    (DEST/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    for name,row in answer.items():print(name,json.dumps(dict(V=row['modes']['compact']['voltage_pooled'],protocols={k:v['voltage_pooled'] for k,v in row['protocols'].items()},net_current_error=row['imposed_net_current_error_rms'],imposed={k:v['median_nrmse'] for k,v in row['modes']['compact_imposed']['families'].items()})))
    print('LATENT', {mode:np.array(row['standardized_latent_rms_fraction_over3_max']).reshape(4,3,3).mean(1).tolist() for mode,row in probes.items()})
if __name__=='__main__':main()
