"""Offline frozen-checkpoint probes; never change official metrics or selection."""
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41h_path_repair as h

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/task41h_recovery_v1'


def arrays(z,name):
    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as f:return {k:f[k] for k in f.files}


def main():
    torch.set_num_threads(2)
    archive=ROOT/'artifacts/task41h_complete_evaluation_external/recovery_final_merged.zip'
    with zipfile.ZipFile(archive) as z:
        spec=json.loads(z.read('preregistration.json'));freeze=json.loads(z.read('selection_freeze.json'))
        states=torch.load(io.BytesIO(z.read('frozen_checkpoint.pt')),map_location='cpu',weights_only=True)
        nominee=freeze['nominee'];definition=next(a for a in h.arms() if a['name']==nominee)
        model=h.model_class(torch)([definition]).eval();model.load_state_dict(states[nominee])
        parent=copy.deepcopy(model)
        with torch.no_grad():parent.f_correction.zero_()
        dev=arrays(z,'development_inputs.npz');k=8;quality=spec['quality']
        unique,inverse=np.unique(np.stack((dev['voltage'][:,k],dev['calcium'][:,k]),-1),axis=0,return_inverse=True)
        v=torch.tensor(unique[:,0])[None].expand(3,-1);ca=torch.tensor(unique[:,1])[None].expand(3,-1)
        with torch.no_grad():raw=model.raw(v,ca);raw_parent=parent.raw(v,ca)
        nap_equal=all(torch.equal(x[...,8:10],y[...,8:10]) for x,y in zip(raw,raw_parent))
        untouched_equal=all(torch.equal(x[...,1:],y[...,1:]) for x,y in zip(raw,raw_parent))
        assert nap_equal and untouched_equal and torch.equal(raw[1],raw_parent[1])
        coef=[torch.sigmoid(a).numpy()[:,inverse] for a in raw]
        canonical=h.p.coefficients(dev['voltage'][:,k].astype('float64'),dev['calcium'][:,k].astype('float64'),.1)
        reference=h.p.readout(dev['voltage'][:,k],dev['target_gates'][:,k],np.ones((12000,11)),h.c.ionic.REVERSALS)[1]
        all_development=json.loads(z.read('development_gpu.json'));original=all_development[nominee]
        contrast_report=json.loads(z.read('paired_contrasts.json'))
        ca_severity={name:[max(h.f.channel_severity(r['metrics'],quality,(0,)) for r in rows if r['seed']==seed)
            for seed in spec['seeds']] for name,rows in all_development.items()}
        local_contrasts=[dict(axis=r['axis'],control=r['control'],treatment=r['treatment'],
            official_ratios=r['ratios'],posthoc_Ca_only_ratios=[b/max(a,1e-30) for a,b in zip(ca_severity[r['control']],ca_severity[r['treatment']])])
            for r in contrast_report]
        bottleneck=dict(official_contrast_count=len(contrast_report),
            all_official_ratios_one=all(all(v==1 for v in r['ratios']) for r in contrast_report),
            frozen_worst_by_seed=[freeze['ranks'][i][2] for i in range(3)],
            all_arms_share_worst_by_seed=all(freeze['ranks'][i][2]==freeze['ranks'][i%3][2] for i in range(30)),
            attribution='Official contrasts divide the max Ca_HVA/Nap_Et2 severity. Frozen Nap timing 2.1/1.0/1.4ms divided by0.3 dominates all arms. Ratios1 cannot identify the effect on the trainable Ca head.',
            Ca_only_posthoc_contrasts=local_contrasts,official_selection_changed=False)
        probes=[]
        for label,us,zs,dtype in [('learned32',(),(),np.float32),('state64',(),(),np.float64),
            ('nap_u_m32',(8,),(),np.float32),('nap_u_m',(8,),(),np.float64),('nap_u_h',(9,),(),np.float64),
            ('nap_u',(8,9),(),np.float64),('nap_z',(),(8,9),np.float64),('nap_uz',(8,9),(8,9),np.float64)]:
            u,zz=(a.astype(dtype).copy() for a in coef)
            for ix in us:u[:,:,ix]=canonical[0][:,ix]
            for ix in zs:zz[:,:,ix]=canonical[1][:,ix]
            gate=np.broadcast_to(dev['initial'][k],(3,18)).astype(dtype).copy();gg=[]
            for tick in range(12000):
                gate=gate+zz[:,tick]*(u[:,tick]-gate);gg.append(gate.copy())
            gg=np.stack(gg)
            for si,seed in enumerate(spec['seeds']):
                current=h.p.readout(dev['voltage'][:,k],gg[:,si],np.ones((12000,11)),h.c.ionic.REVERSALS)[1]
                metric=h.p.evaluate_currents(reference,current,.1,quality)['channels'][4]
                base=reference[0,4];rr=np.abs(reference[:,4]-base);pp=np.abs(current[:,4]-base)
                threshold=.1*rr.max();ri=np.flatnonzero(rr>=threshold);pi=np.flatnonzero(pp>=threshold)
                crossing=int(ri[-1]);slope=(rr[min(crossing+1,len(rr)-1)]-rr[max(crossing-1,0)])/.2
                probes.append(dict(policy=label,seed=seed,Nap_metric=metric,reference_offset_ms=float(ri[-1]*.1),
                    prediction_offset_ms=float(pi[-1]*.1),reference_crossing_slope=float(slope),
                    crossing_signed_current_error=float(current[crossing,4]-reference[crossing,4]),
                    prediction_within_reference_threshold_band=int(np.sum(np.abs(pp-threshold)<abs(slope)*.3))))
        # Identical recorded traces; promote to float64 before differencing to
        # distinguish reduction arithmetic from already accumulated trajectory error.
        pred=arrays(z,'opened41c/traces_'+nominee+'_autonomous.npz')
        ref=arrays(z,'opened41c/reference_traces.npz')
        with zipfile.ZipFile(ROOT/'artifacts/task41d_opened41c.zip') as zz:meta=json.loads(zz.read('cases.json'))
        source=int(meta[108]['source_index']);rr=ref['autonomous__current'];pp=pred['current'][:,2]
        reductions=[]
        for dtype in (np.float32,np.float64):
            r=rr.astype(dtype);p=pp.astype(dtype)
            metrics=h.p.evaluate_intervention(r[:,source],r[:,108],p[:,source],p[:,108],.1,quality)
            reductions.append(dict(dtype=str(dtype),Nap_metric=metrics['channels'][4]))
        grid=torch.linspace(-100,60,4097)[None].expand(3,-1);cc=torch.full_like(grid,1e-4)
        with torch.no_grad():pu,_=parent.raw(grid,cc);mu,_=model.raw(grid,cc)
        truth=h.p.coefficients(grid[0].double().numpy(),cc[0].double().numpy(),.1)[0][:,0]
        truth=np.log(np.clip(truth,1e-12,1-1e-12)/(1-np.clip(truth,1e-12,1-1e-12)))
        required=truth[None]-pu[...,0].numpy();delta=(mu-pu)[...,0].numpy();local=(grid[0].numpy()>=-50)&(grid[0].numpy()<=-35)
        bounds=dict(local_v_range=[-50,-35],required_correction_max_abs=float(np.abs(required[:,local]).max()),
            allowed_correction_abs_max=.1,observed_delta_max_abs=float(np.abs(delta[:,local]).max()),
            local_required_outside_bound_count=int(np.sum(np.abs(required[:,local])>=.1)),
            observed_local_saturation_fraction=float(np.mean(np.abs(delta[:,local])>.099)),
            caveat='Bound probe only, not an optimized approximation floor or deployment quality proof.')
        out=dict(valid=True,scope='Post hoc CPU/offline diagnostic; original GPU verdicts unchanged',
            source_checkpoint_sha256=hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest(),nominee=nominee,
            training_performed=False,selection_changed=False,thresholds_changed=False,
            imposed_development_nap_structurally_unaffected=nap_equal,
            other17_u_and_all_z_equal_parent=untouched_equal,
            original_development_case8=[r for r in original if r['case']==8],
            numpy_recurrence_probes=probes,response_reduction_probes=reductions,correction_bounds=bounds,
            registered_contrast_bottleneck=bottleneck,
            timing_definition_mismatch=dict(training='10 percent of distance to asymptotic steady current after prefix, with terminal-crossing weighting',
                evaluation='10 percent of max absolute current deviation from first reference sample; first/last crossing on full path',
                causal_attribution='Different objectives by construction; not proof this mismatch alone causes the residual.'),
            limits='CPU coefficient/recurrent arithmetic is not bitwise the original CUDA graph. Oracles are nonselectable. No official result recomputed or replaced.')
    (OUT/'residual_probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(dict(valid=True,nap_unchanged=nap_equal,bounds=bounds,
        nap_offsets=[(r['policy'],r['seed'],r['Nap_metric']['event'].get('offset_error_ms')) for r in probes],
        response_nrmse=[(r['dtype'],r['Nap_metric']['nrmse']) for r in reductions])),flush=True)


if __name__=='__main__':main()
