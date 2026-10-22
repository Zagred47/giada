"""Post-hoc diagnostics on saved 41d trajectories, gradients and SQLite history."""
import hashlib, io, json, sqlite3, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task41d_audited_v1'

def read(z, name): return json.loads(z.read(name))
def arrays(z, name):
    with np.load(io.BytesIO(z.read(name))) as d: return {k: d[k] for k in d.files}
def rms(v): return float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64)**2)))

def main():
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'artifacts/task41d_primary_external/giada_task41d_result.zip') as z:
        freeze=read(z,'selection_freeze.json'); curves=read(z,'training_curves.json'); cal=read(z,'fit_calibration.json')
        with zipfile.ZipFile(ROOT/'artifacts/task41d_opened41c.zip') as inputs: cases=read(inputs,'cases.json')
        refs=arrays(z,'opened41c/reference_traces.npz'); rows=[]; interaction=[]
        for mode in ('imposed','autonomous'):
            models={name:arrays(z,f'opened41c/traces_{name}_{mode}.npz') for name in ('P110','C01','N000','combined')}
            # Subtract in float64 to avoid introducing extra subtraction roundoff in the audit.
            currents={n:a['current'].astype('float64') for n,a in models.items()}
            cross=currents['combined']-currents['C01']-currents['N000']+currents['P110']
            interaction.append(dict(mode=mode,maximum_abs_current_cross=float(abs(cross).max()),
                                    rmse_current_cross=rms(cross),units='mA/cm2'))
            for case in (108,122):
                source=cases[case]['source_index']; ref=refs[mode+'__current']
                delta=ref[:,case,4].astype('float64')-ref[:,source,4].astype('float64'); scale=max(float(abs(delta).max()),1e-7)
                for si,seed in enumerate((17,29,43)):
                    metrics={}; signed={}
                    for name,a in models.items():
                        curr=currents[name]; eb=curr[:,si,source,4]-ref[:,source,4]; ei=curr[:,si,case,4]-ref[:,case,4]
                        error=ei-eb; signed[name]=error
                        metrics[name]=dict(nap_response_nrmse=rms(error)/scale,base_current_rmse=rms(eb),
                                           intervention_current_rmse=rms(ei),response_error_rmse=rms(error),
                                           voltage_pair_error_rmse_mv=rms((a['voltage'][:,si,case].astype('float64')-a['voltage'][:,si,source])-(refs[mode+'__voltage'][:,case].astype('float64')-refs[mode+'__voltage'][:,source])),
                                           maximum_abs_voltage_error_mv=float(abs(a['voltage'][:,si,[source,case]]-refs[mode+'__voltage'][:,[source,case]]).max()))
                    ix=signed['combined']-signed['C01']-signed['N000']+signed['P110']
                    linear=signed['C01']+signed['N000']-signed['P110']
                    rows.append(dict(mode=mode,case=case,seed=seed,case_metadata=cases[case],reference_peak_response=float(abs(delta).max()),scale=scale,
                                     models=metrics,nonadditive_response_error_nrmse=rms(ix)/scale,
                                     linear_superposition_response_error_nrmse=rms(linear)/scale))
        selection=[]
        for j,a in enumerate(freeze['arms']):
            for si,seed in enumerate((17,29,43)):
                member=3*j+si; step=freeze['chosen_steps'][member]; initial=curves[0]; final=curves[-1]
                weights=np.asarray(cal['weights'][member]); series=[]
                for curve in curves:
                    gn=np.asarray(curve['FIT_gradient_norms'][member]); weighted=gn*weights
                    series.append(dict(step=curve['step'],point=curve['atomic'][member][0],secant=curve['atomic'][member][1],
                                       response=curve['feedback'][member],rank=curve['ranks'][member],
                                       weighted_gradient_norms=weighted.tolist(),
                                       feedback_to_point_gradient=float(weighted[2]/max(weighted[0],1e-30))))
                selection.append(dict(arm=a['name'],seed=seed,selected_step=step,series=series,
                                      feedback_final_over_initial=final['feedback'][member]/max(initial['feedback'][member],1e-30)))
        # Measure objective geometry without optimizing anything. Frozen snapshot contains
        # the complete parent weights; zeroing only the added head restores P110.
        import torch
        from src.giada_teacher import task41d_precision_response as d
        torch.set_num_threads(2)
        checkpoint=torch.load(io.BytesIO(z.read('frozen_checkpoint.pt')),map_location='cpu',weights_only=False)
        objective=[]
        for name,file,gate in [('C01','fit_Ca_uniform.npz',0),('N000','fit_Nap_uniform.npz',8)]:
            a=next(a for a in d.arms() if a['name']==name)
            model=d.model_class(torch)([a],[17,29,43]);model.load_state_dict(checkpoint[name])
            with torch.no_grad():model.local_equilibrium.zero_()
            data=arrays(z,file);v=torch.tensor(data['v'].reshape(1,-1)).expand(3,-1);ca=torch.tensor(data['ca'].reshape(1,-1)).expand(3,-1)
            target=data['u'][...,gate].reshape(-1);cut=np.clip(target,1e-7,1-1e-7);expected=torch.tensor(np.log(cut/(1-cut)))
            mask=(target<1e-7)|(target>1-1e-7)
            raw=model.raw(v,ca)[0][...,gate].double();error=(raw-expected).square()
            grad_clipped=torch.autograd.grad(error[:,mask].mean(-1).sum(),model.local_equilibrium,retain_graph=True)[0].double()
            grad_unclipped=torch.autograd.grad(error[:,~mask].mean(-1).sum(),model.local_equilibrium)[0].double()
            raw=raw.detach().numpy();error=error.detach().numpy()
            gc=grad_clipped*mask.mean();gu=grad_unclipped*(1-mask.mean())
            objective.append(dict(arm=name,gate=gate,clipped_target_fraction=float(mask.mean()),
                                  clipped_point_SSE_fraction=(error[:,mask].sum(1)/error.sum(1)).tolist(),
                                  full_point_loss=error.mean(1).tolist(),unsaturated_point_loss=error[:,~mask].mean(1).tolist(),
                                  lower_bound_MSE_with_pointwise_logit_correction_bound_0p1=(np.maximum(abs(raw-expected.numpy())-.1,0)**2).mean(1).tolist(),
                                  gradient_norm_clipped=gc.norm(dim=1).tolist(),gradient_norm_unsaturated=gu.norm(dim=1).tolist(),
                                  gradient_cosine=((gc*gu).sum(1)/(gc.norm(dim=1)*gu.norm(dim=1)).clamp_min(1e-30)).tolist(),
                                  limitation='Loss concentration is not proof of optimizer harm: local basis supports can decouple regions. Test a target-consistent objective at matched support, budget and deployment.'))
        out=dict(scope='Post-hoc saved-output analysis, no training or selection changes; nonadditivity includes floating-point effects until precision replay isolates them.',
                 thresholds_changed=False,interaction=interaction,paired_trace_analysis=rows,selection_gradient_analysis=selection,
                 objective_geometry=objective,
                 feedback_policy=dict(ticks=4,dt_ms=.1,every=64,batch=8,weights_calibrated_once=True),
                 source_archive_sha256='b27fdfc79da68a5b6afbe3aba44f608d53a84a42594a01183d6d5c6dd4e16311')
    sql='SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings WHERE [Codice stabile] IN (?,?,?,?,?,?,?)'
    codes=['findings-giada-task39j-result-registered_contrasts-v1','findings-giada-task39j-result-freeze_contrast-v1',
           'findings-giada-task41c-audit-ca_mechanism-v1','findings-giada-task41c-audit-nap_mechanism-v1',
           'findings-giada-task41c-audit-secants-v1','findings-giada-task41c-audit-regression-v1','findings-giada-task41bc-41b-result-v1']
    with sqlite3.connect((ROOT/'research_memory/data/research_memory.sqlite').as_uri()+'?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row; history=[dict(r) for r in con.execute(sql,codes)]
    assert len(history)==len(codes)
    out['memory_query']=dict(sql=sql,parameters=codes,rows=history)
    (OUT/'transfer_probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(dict(interaction=interaction,autonomous=[dict(case=r['case'],seed=r['seed'],models={n:m['nap_response_nrmse'] for n,m in r['models'].items()},cross=r['nonadditive_response_error_nrmse'],linear=r['linear_superposition_response_error_nrmse']) for r in rows if r['mode']=='autonomous'],
                         feedback_deterioration={n:[r['feedback_final_over_initial'] for r in selection if r['arm']==n] for n in ('N000','N001','N011','N101')},
                         gradient_ratio_N001=[r['series'][-1]['feedback_to_point_gradient'] for r in selection if r['arm']=='N001'],objective_geometry=objective)))

if __name__=='__main__': main()
