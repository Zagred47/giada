"""Read-only scientific audit; receipt clarification keeps the original archive."""
import gzip,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task52_closed_loop as q
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'artifacts/task52b_kaggle_external';D=ROOT/'experiments/results/task52b_audit_v1'

def main():
    D.mkdir(exist_ok=True);original=F/'kaggle_receipt.original.json';receipt_path=F/'kaggle_receipt.json'
    if not original.exists():original.write_bytes(receipt_path.read_bytes())
    receipt=json.loads(original.read_text());archive=F/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    assert archive.stat().st_size==receipt['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'));plan=json.loads(z.read('preregistration.json'));diag=json.loads(z.read('initialization_diagnostics.json'))
        assert report['training_updates']==receipt['training_updates_performed']==1200
        assert report['selection_recomputed']==receipt['selection_recomputed'] is True
        params={};finite={}
        for name in z.namelist():
            if name.endswith('.pt') and not name.endswith('resumable_training.pt'):
                state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)
                finite[name]=all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v))
        assert all(finite.values())
        ref=dict(np.load(io.BytesIO(z.read('reference.npz'))));cases=json.loads(z.read('cases.json'));dt=plan['dt_ms'];bar=np.array(q.q.ionic.GBAR)
        parent=q.plan(ROOT);_,injection,mult,rev=q.design(parent);physics=q.a.n.contract(ROOT)[1:];base=physics[2]
        # Counterfactual passive trajectory: no ionic currents, same leak/input/solver.
        v=np.full((3,len(cases)),-65.);zero=[v.copy()];cap=.001*base['cm_uf_cm2']/dt;leak=base['g_pas_s_cm2']
        for inj in injection:
            v=(cap*v+leak*base['e_pas_mv']+inj)/(cap+leak);zero.append(v.copy())
        zero=np.stack(zero)
        arms={};traces_finite=True;minimum_bounds=[]
        lows=np.array(diag['decoder_box']['lower']);highs=np.array(diag['decoder_box']['upper'])
        quality=parent['quality']
        for ai,name in enumerate(['frozen52']+[x['id'] for x in plan['arms']]):
            rows=json.loads(gzip.decompress(z.read(name+'/evaluation.json.gz')));trace=dict(np.load(io.BytesIO(z.read(name+'/traces.npz'))))
            traces_finite &= all(np.isfinite(x).all() for x in trace.values())
            sub=[r for r in rows if r['arm']=='compact' and r['horizon_ms']==1000]
            im=[r for r in rows if r['arm']=='compact_imposed' and r['horizon_ms']==1000]
            families={}
            for family,indices in plan['families'].items():
                cells=[r['current']['channels'][k] for r in sub for k in indices]
                families[family]=dict(passed=sum(c['passed'] for c in cells),total=len(cells),median_nrmse=float(np.median([c['nrmse'] for c in cells])),missing_events=sum(c.get('event',{}).get('status')=='missing_event' for c in cells))
            cost=json.loads(z.read(name+'/cost.json'));sample=cost['samples_ms'];latency={k:float(np.median(x)) for k,x in sample.items()}
            ci=4 if name=='frozen52' else 0;ri=7 if name=='frozen52' else 3
            arm=dict(summary=report['summary'][name],families=families,imposed_current_passed=sum(r['current']['conservative_passed'] for r in im),cost_median_ms=latency,atomic_speed_ratio=latency['canonical64']/latency['compact10_step_readout'],zero_current_baseline_voltage_rmse=float(np.sqrt(np.mean((zero-ref['voltage'])**2))),distance_to_zero_current_voltage_rmse=float(np.sqrt(np.mean((trace['voltage'][:,:,ci]-zero)**2))),reset_voltage_delta_rmse=float(np.sqrt(np.mean((trace['voltage'][:,:,ri]-trace['voltage'][:,:,ci])**2))))
            if ai:
                idx=slice((ai-1)*3,ai*3)
                driving=(ref['voltage'][:-1,...,None]-rev[None,None])*bar*mult[None,None]
                left=driving*lows[idx][None,:,None];right=driving*highs[idx][None,:,None]
                distance=np.maximum(np.minimum(left,right)-ref['current'],0)+np.minimum(np.maximum(left,right)-ref['current'],0)
                scale=np.maximum(np.abs(ref['current']).max(0),quality['current_scale_floor_ma_cm2'])
                nr=np.sqrt((distance**2).mean(0))/scale;nm=np.abs(distance).max(0)/scale
                blocked=(nr>quality['current_nrmse_max'])|(nm>quality['current_nmax_max'])
                arm['fixed_decoder_lower_bound']=dict(blocked_cases=int(blocked.any(-1).sum()),total_cases=int(blocked.shape[0]*blocked.shape[1]),blocked_channel_cases=int(blocked.sum()),total_channel_cases=int(blocked.size),qualification='Any latent in [0,1]^10 with these fixed decoder weights, evaluated on imposed reference V/Ca. Not an architecture impossibility proof.')
                arm['initialization_probe']={mode:float(np.mean(np.array(diag[mode]['nrmse'])[idx])) for mode in ('normal','future_h0_oracle','teacher_reencode')}
                arm['jacobian_max_singular']=[r['max_singular'] for r in diag['local_jacobians'][idx]]
            arms[name]=arm
        assert traces_finite
        controls=all(v['all_quality_passed'] for key in ('canonical64','canonical32','P','P_J','oracle_all') for v in report['summary']['frozen52'][key].values())
        assert controls
        result=dict(valid=True,archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'],finite_checkpoints=finite,all_traces_finite=True,summary=arms,learning_curve=json.loads(z.read('learning_curve.json')),selection=json.loads(z.read('selection_freeze.json')),reference_controls_passed=controls,limitations=['No causal attribution of 52-to52b improvement to a single factor: retraining, initialization and loss changed together.','h0 oracle used only 10ms fitting and finite optimization; reencoding is not an optimal encoder oracle.','Local Jacobians are not global memory theorems.','1000ms pooled RMSE dilutes transient errors: inspect 100ms and current/timing failures.','No whole-neuron speedup or qualified candidate.'])
        (D/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    clarified=dict(receipt,execution_valid=True,scientific_acceptance_passed=False,receipt_clarification=dict(reason='valid denotes technical execution/archive integrity, not scientific threshold acceptance',counts_corrected=False,original_receipt_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),archive_unchanged=True))
    receipt_path.write_text(json.dumps(clarified,indent=2)+'\n');(D/'kaggle_receipt.original.json').write_bytes(original.read_bytes());(D/'kaggle_receipt.clarified.json').write_text(json.dumps(clarified,indent=2)+'\n')
    print(json.dumps({k:{f:v[f] for f in ('families','fixed_decoder_lower_bound','initialization_probe','zero_current_baseline_voltage_rmse','distance_to_zero_current_voltage_rmse','atomic_speed_ratio') if f in v} for k,v in arms.items()},indent=2))
if __name__=='__main__':main()
