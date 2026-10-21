"""Immutable 41c audit and non-selectable CPU counterfactuals, no training."""
import hashlib, io, json, sqlite3, sys, zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41c_frozen_confirmation as c
from audit_task41 import recurrence,event_detail
p,t,x=c.p,c.t,c.x
OUT=ROOT/'experiments/results/task41c_audited_v1'

def save(name,value):p.save_json(OUT/name,value)
def arrays(z,n):return dict(np.load(io.BytesIO(z.read(n))))
def channel(mm,name):return next(a for a in mm['channels'] if a['channel']==name)
def compact(mm):return dict(passed=mm['passed'],failed=[r for r in mm['channels'] if not r['passed']],gate=mm.get('gate'))

class Oracle:
    def __init__(self,parent,targets):self.parent,self.targets=parent,targets
    def __getattr__(self,k):return getattr(self.parent,k)
    def __call__(self,v,ca,g):
        ru,rz=self.parent.raw(v,ca);u,z=torch.sigmoid(ru),torch.sigmoid(rz)
        tu,tau=t._rates('Nap_Et2',v,ca,torch);tz=-torch.expm1(-.1/tau)
        for kind,gate in self.targets:(u if kind=='u' else z)[...,gate]=(tu if kind=='u' else tz)[...,gate-8]
        return g+z*(u-g)

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    spec,cfg,calcium,base=c.contract(ROOT);q=spec['current_quality'];cfg=x.calibration_config(spec,cfg)
    arc=ROOT/'artifacts/task41c_primary_external/giada_task41c_result.zip'
    assert p.digest(arc)=='3e7a0d6430a0b3f44a517e0f932787b7dc7e42ca05d4765191d85bc752c4f597'
    with zipfile.ZipFile(arc) as zz:
        assert zz.testzip() is None
        read=lambda n:json.loads(zz.read(n))
        results=read('causal_results.json');sr=read('fresh_stress_results.json');meta=read('cases.json')
        data=arrays(zz,'native_shadow_traces.npz');ref=arrays(zz,'reference_traces.npz');sd=arrays(zz,'fresh_stress_traces.npz')
        traces={name:{mode:arrays(zz,f'traces_{name}_{mode}.npz') for mode in ('imposed','autonomous')} for name in ('P110','B6','LUT_control')}
        final=read('final_report.json');cost=read('cost.json')
        assert read('native_interface_report.json')['valid'] and read('gpu_preflight.json')['all_parameters_finite']
        assert hashlib.sha256(zz.read('source_checkpoint.pt')).hexdigest()==spec['sources'][spec['checkpoint']]
        assert all(r['passed'] for m in read('formula_floor.json').values() for rows in m.values() for r in rows)
        assert all(r['passed'] for r in read('native_current_phase_floor.json'))
        assert all(all(r['finite'].values()) for r in cost)
        assert all(r['passed'] for r in sr['formula_floor'])
    models=c.load_models(torch,ROOT,'cpu')
    failures=[r for r in results['P110/autonomous']['interventions'] if not r['metrics']['passed']]
    response=[]
    for f in failures:
        k,s=f['case'],spec['model_seeds'].index(f['seed']);o=meta[k]['source_index'];row=dict(case=meta[k],model_seed=f['seed'],original=compact(f['metrics']),comparisons={})
        for name in traces:
            for mode in ('imposed','autonomous'):
                rr=ref[mode+'__current'];pr=traces[name][mode]['current'][:,s]
                mm=p.evaluate_intervention(rr[:,o],rr[:,k],pr[:,o],pr[:,k],.1,q)
                saved=next(r for r in results[name+'/'+mode]['interventions'] if r['case']==k and r['seed']==f['seed'])['metrics']
                assert mm['passed']==saved['passed']
                assert np.isclose(channel(mm,'Nap_Et2')['nrmse'],channel(saved,'Nap_Et2')['nrmse'],rtol=1e-5)
                row['comparisons'][name+'/'+mode]=channel(mm,'Nap_Et2')
        rr=ref['autonomous__current'];pr=traces['P110']['autonomous']['current'][:,s]
        er0=pr[:,o,4]-rr[:,o,4];er1=pr[:,k,4]-rr[:,k,4];de=er1-er0;truth=rr[:,k,4]-rr[:,o,4]
        row['differential']=dict(base_error_rmse=float(np.sqrt(np.mean(er0**2))),intervention_error_rmse=float(np.sqrt(np.mean(er1**2))),response_error_rmse=float(np.sqrt(np.mean(de**2))),
            response_peak=float(abs(truth).max()),largest_error_time_ms=float(np.argmax(abs(de))*.1),floor_active=bool(abs(truth).max()<q['current_scale_floor_ma_cm2']))
        row['voltage']=dict(base_rmse=float(np.sqrt(np.mean((traces['P110']['autonomous']['voltage'][:,s,o]-ref['autonomous__voltage'][:,o])**2))),intervention_rmse=float(np.sqrt(np.mean((traces['P110']['autonomous']['voltage'][:,s,k]-ref['autonomous__voltage'][:,k])**2))))
        response.append(row)
    save('response_failures.json',response)
    print('Saved two response audits',flush=True)
    # Coupled CPU oracles only on the two failed pairs. Saved GPU results remain authoritative.
    selected=sorted({i for f in failures for i in (f['case'],meta[f['case']]['source_index'])})
    sub={k:(v if k=='area_um2' else v[selected]) for k,v in data.items()}
    oracledefs={'P110':[],'oracle_u8':[('u',8)],'oracle_z8':[('z',8)],'oracle_u9':[('u',9)],'oracle_z9':[('z',9)],'oracle_Nap':[(a,b) for a in ('u','z') for b in (8,9)]}
    probes=[];cpu_predictions={}
    with torch.inference_mode():
        for name,targets in oracledefs.items():
            model=models['P110'] if not targets else Oracle(models['P110'],targets)
            pred=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,t.FAMILIES['all']),sub,'cpu')
            cpu_predictions[name]=pred
            for f in failures:
                k,o=f['case'],meta[f['case']]['source_index'];s=spec['model_seeds'].index(f['seed']);kk,oo=selected.index(k),selected.index(o)
                mm=p.evaluate_intervention(ref['autonomous__current'][:,o],ref['autonomous__current'][:,k],pred['current'][:,s,oo],pred['current'][:,s,kk],.1,q)
                probes.append(dict(variant=name,case=k,seed=f['seed'],metrics=compact(mm),Nap=channel(mm,'Nap_Et2')))
            print('Coupled diagnostic',name,flush=True)
    parity={key:float(np.max(np.abs(cpu_predictions['P110'][key]-traces['P110']['autonomous'][key][:,:,selected]))) for key in cpu_predictions['P110']}
    # CPU/GPU rounding is reported, not silently accepted as experimental identity.
    save('response_counterfactuals.json',dict(post_hoc=True,selection_eligible=False,device='CPU',cpu_gpu_max_abs=parity,rows=probes))
    # Unique-point evaluation reproduces coefficients without repeating the large trunk every tick.
    v,ca=sd['voltage'],sd['calcium'];points,ix=np.unique(np.stack((v,ca),-1).reshape(-1,2),axis=0,return_inverse=True)
    with torch.inference_mode():
        ru,rz=models['P110'].raw(torch.tensor(points[:,0])[None].expand(3,-1),torch.tensor(points[:,1])[None].expand(3,-1))
    uu=torch.sigmoid(ru).numpy()[:,ix].reshape(3,len(v),12,18).transpose(1,0,2,3)
    zz=torch.sigmoid(rz).numpy()[:,ix].reshape(3,len(v),12,18).transpose(1,0,2,3)
    tu,tz=p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    gs=recurrence(uu,zz,np.broadcast_to(sd['initial'],(3,12,18)),np.float32)
    truth=p.readout(v,sd['target_gates'],sd['multipliers'],c.local.ionic.REVERSALS)[1]
    stressmeta=c.stress_design(spec);stress=[];stressprobes=[]
    replay_checks=[]
    for s,seed in enumerate(spec['model_seeds']):
        currents=p.readout(v,gs[:,s],sd['multipliers'],c.local.ionic.REVERSALS)[1]
        for k,m in enumerate(stressmeta):
            original=next(r['metrics'] for r in sr['rows'] if r['model']=='P110' and r['seed']==seed and r['input_seed']==m['seed'] and r['family']==m['family'] and r['case']==m['case'])
            replay=p.evaluate_currents(truth[:,k],currents[:,k],.1,q,sd['target_gates'][:,k],gs[:,s,k])
            assert replay['passed']==original['passed'],(k,seed)
            assert all(a['passed']==b['passed'] and a['event']==b['event'] for a,b in zip(replay['channels'],original['channels'])),(k,seed)
            replay_checks.append(dict(index=k,seed=seed,passed=replay['passed'],same_channel_verdicts_and_events=True,
                max_nrmse_difference=max(abs(a['nrmse']-b['nrmse']) for a,b in zip(replay['channels'],original['channels']))))
    save('stress_replay_validation.json',dict(valid=True,rows=replay_checks,all_coefficients_finite=bool(np.isfinite(uu).all() and np.isfinite(zz).all())))
    for f in sr['rows']:
        if f['model']!='P110' or f['metrics']['passed']:continue
        s=spec['model_seeds'].index(f['seed']);k=next(i for i,r in enumerate(stressmeta) if r['family']==f['family'] and r['seed']==f['input_seed'] and r['case']==f['case'])
        pred=p.readout(v[:,k],gs[:,s,k],sd['multipliers'][k],c.local.ionic.REVERSALS)[1]
        mm=p.evaluate_currents(truth[:,k],pred,.1,q,sd['target_gates'][:,k],gs[:,s,k])
        ev=event_detail(truth[:,k,0],pred[:,0],q)
        b6=next(a for a in sr['rows'] if a['model']=='B6' and all(a[key]==f[key] for key in ('seed','input_seed','family','case')))
        row=dict(index=k,model_seed=f['seed'],input_seed=f['input_seed'],family=f['family'],case=f['case'],panel=stressmeta[k]['panel'],original=compact(f['metrics']),
            B6_Ca_HVA_identical=channel(b6['metrics'],'Ca_HVA')==channel(f['metrics'],'Ca_HVA'),cpu_replay=compact(mm),event=ev,
            rest_voltage=float(v[-1,k]),calcium_mM=float(ca[-1,k]),initial_voltage_mv=(-95 if f['case']==0 else -42))
        # Coefficient/state precision and individual oracles isolate persistent bias versus relaxation.
        for name,targets,dtype in [('learned32',[],np.float32),('learned64',[],np.float64),('oracle_u0',[('u',0)],np.float64),('oracle_z0',[('z',0)],np.float64),('oracle_u1',[('u',1)],np.float64),('oracle_z1',[('z',1)],np.float64),('oracle_CaHVA',[(a,b) for a in ('u','z') for b in (0,1)],np.float64)]:
            a,b=uu[:,s,k].astype(dtype),zz[:,s,k].astype(dtype)
            for kind,gate in targets:(a if kind=='u' else b)[:,gate]=(tu if kind=='u' else tz)[:,k,gate]
            gg=recurrence(a,b,sd['initial'][k],dtype);ii=p.readout(v[:,k],gg,sd['multipliers'][k],c.local.ionic.REVERSALS)[1]
            cm=p.evaluate_currents(truth[:,k],ii,.1,q)
            stressprobes.append(dict(index=k,seed=f['seed'],variant=name,Ca_HVA=channel(cm,'Ca_HVA'),event=event_detail(truth[:,k,0],ii[:,0],q)))
        end=round(ev['reference_offset_ms']*10)
        row['coefficient_at_rest']={str(g):dict(u_reference=float(tu[-1,k,g]),u_learned=float(uu[-1,s,k,g]),z_reference=float(tz[-1,k,g]),z_learned=float(zz[-1,s,k,g]),state_reference_at_offset=float(sd['target_gates'][end,k,g]),state_learned_at_offset=float(gs[end,s,k,g])) for g in (0,1)}
        stress.append(row)
    save('stress_failures.json',stress);save('stress_counterfactuals.json',dict(post_hoc=True,selection_eligible=False,rows=stressprobes))
    print('Saved five stress audits and coefficient interventions',flush=True)
    # Matched controls on 41c versus historical opened repair, no cross-dataset regression claim.
    with zipfile.ZipFile(ROOT/'artifacts/task41b_primary_external/giada_task41b_result.zip') as old:
        assert p.digest(Path(old.filename))=='1d8ae46033437dfec9da72b47db01f8aa78f15fc31a17d8c57c3913c5e199cec'
        oldfinal=json.loads(old.read('final_report.json'));oldstress=json.loads(old.read('stress_results.json'));oldcausal=json.loads(old.read('causal_results.json'))
    comparison=dict(task41b=oldfinal,task41c=final,old_stress_failures=[{k:v for k,v in r.items() if k!='metrics'}|dict(failed=compact(r['metrics'])) for r in oldstress['rows'] if r['model'] in ('B6','P110') and not r['metrics']['passed']],
        same_41c_response_failures={name:[dict(case=r['case'],seed=r['seed']) for r in results[name+'/autonomous']['interventions'] if not r['metrics']['passed']] for name in ('B6','P110','LUT_control')},
        limits='41b opened repair versus41c independent new inputs;different stress distribution,not paired regression. Five stress checks reflect two histories of one input seed across model seeds,notfive independent protocols.')
    save('comparison_41b.json',comparison)
    # Local read-only semantic evidence queries with provenance.
    queries={'related_findings':"SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings WHERE [Codice stabile] LIKE '%task41%' OR [Codice stabile] LIKE '%task40b%' OR [Codice stabile] LIKE '%task39j%'",
        'claims':"SELECT [Codice stabile] code,Nome,Enunciato,Stato FROM claims WHERE [Codice stabile] LIKE '%task41%'",
        'timing_lessons':"SELECT [Codice stabile] code,Nome,Lezione,Limiti FROM lessons WHERE [Codice stabile] LIKE '%giada%' AND (lower(Lezione) LIKE '%timing%' OR lower(Lezione) LIKE '%equilibrium%' OR lower(Lezione) LIKE '%precision%')"}
    con=sqlite3.connect('file:'+str(ROOT/'research_memory/data/research_memory.sqlite')+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    save('memory_patterns.json',{k:dict(sql=sql,rows=[dict(r) for r in con.execute(sql)]) for k,sql in queries.items()});con.close()
    costs=[dict(batch_per_seed=r['batch_per_seed'],scope=r['scope'],P110={k:v['median_saved_fraction'] for k,v in r['comparisons']['P110'].items()}) for r in cost]
    save('audit.json',dict(valid=True,archive_sha256=p.digest(arc),crc_valid=True,source_checkpoint_verified=True,thresholds_changed=False,training_performed=False,
        scientific_verdict_unchanged=True,task42_authorized=False,response_failure_count=len(response),stress_failure_count=len(stress),all_saved_arrays_finite=all(np.isfinite(a).all() for modes in traces.values() for arr in modes.values() for a in arr.values()),
        cpu_gpu_max_abs=parity,costs=costs,limitations=['CPU posthoc oracles are not selectable candidates.','No gradient training,threshold change,or test reopening as fresh confirmation.','Original CUDA metric reports remain authoritative;CPU parity reported separately.']))
    # Scientific plots from saved traces and diagnostic replay, explicit units.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,row in zip(axes[0],response):
        k=row['case']['id'];o=row['case']['source_index'];s=spec['model_seeds'].index(row['model_seed']);clock=np.arange(400)*.1
        rr=ref['autonomous__current'];pr=traces['P110']['autonomous']['current'][:,s]
        ax.plot(clock,(rr[:,k,4]-rr[:,o,4])*1e6,label='Formula response');ax.plot(clock,(pr[:,k,4]-pr[:,o,4])*1e6,label='P110 response',ls='--')
        ax.set(xlabel='Time (ms)',ylabel='Nap response (nA/cm²)',title=f"Case{k}, seed{row['model_seed']}: {row['case']['intervention']}");ax.legend()
    for ax,f in zip(axes[1],[stress[0],stress[1]]):
        k,s=f['index'],spec['model_seeds'].index(f['model_seed']);pr=p.readout(v[:,k],gs[:,s,k],sd['multipliers'][k],c.local.ionic.REVERSALS)[1][:,0];rr=truth[:,k,0];clock=np.arange(len(v))*.1
        center=f['event']['reference_offset_ms'];threshold=f['event']['threshold']
        ax.plot(clock,abs(rr-rr[0])*1e6,label='Formula |I−I₀|');ax.plot(clock,abs(pr-rr[0])*1e6,label='P110 |I−I₀|',ls='--');ax.axhline(threshold*1e6,color='black',ls=':')
        window=(clock>center-6)&(clock<center+6);vals=np.r_[abs(rr[window]-rr[0]),abs(pr[window]-rr[0])]*1e6
        ax.set(xlim=(center-6,center+6),ylim=(vals.min()*.999,vals.max()*1.001),xlabel='Time (ms)',ylabel='Ca_HVA deviation (nA/cm²)',title=f"Stress{k}, seed{f['model_seed']}, offset {f['original']['failed'][0]['event']['offset_error_ms']:.1f} ms");ax.legend()
    fig.suptitle('Task41c: differential response and shallow relaxation crossings\nTop: signed intervention minus base; bottom: absolute deviation from reference initial current')
    fig.savefig(OUT/'failure_diagnostics.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(responses=[dict(case=r['case']['id'],seed=r['model_seed'],nrmse=r['original']['failed'][0]['nrmse']) for r in response],response_oracles=[dict(variant=r['variant'],case=r['case'],nrmse=r['Nap']['nrmse']) for r in probes],stress_oracles=[dict(variant=r['variant'],index=r['index'],seed=r['seed'],offset=r['Ca_HVA']['event'].get('offset_error_ms')) for r in stressprobes],parity=parity),indent=2))

if __name__=='__main__':main()
