"""Independent frozen B6 confirmation: new native cases and imposed-input stress.

NEURON and CUDA run in separate processes. Only observed initial states and
causal presynaptic shadow conductances are inputs, never future teacher V/Ca.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from . import task40b_calcium_precision as b
t,p,j,ionic=b.t,b.p,b.j,b.t.ionic
CONFIG='experiments/task41_preregistration_v1.json'

def contract(root):
    spec=json.loads((root/CONFIG).read_text());_,cfg,calcium,base=t.contract(root)
    assert spec['current_quality']==json.loads((root/p.CONFIG).read_text())['quality']
    assert spec['compartment_gates']==cfg['gates'] and spec['dt_ms']==cfg['dt_ms']==.1
    for path,sha in spec['sources'].items():assert p.digest(root/path)==sha,path
    prior=json.loads((root/'experiments/results/task40b_audited_v1/audit.json').read_text())
    assert prior['valid'] and prior['preferred_arm']=='B6' and prior['task41_preparation_authorized']
    assert spec['model_seeds']==[17,29,43] and spec['candidate']=='B6'
    assert not set(spec['native_seeds'])&{85317,85329,17,29,43,spec['calibration_seed']}
    return spec,cfg,calcium,base

def injection(protocol,clock):
    y=np.zeros_like(clock)
    pulses={'calibration':[(2.,5.,.003)],
        'delayed_doublet':[(6.,9.,.010),(14.,17.,.010)],
        'rebound_burst':[(2.,7.,-.003),(9.,12.,.010),(15.,18.,.010)],
        'staircase':[(3.,6.,.003),(6.,9.,.006),(9.,12.,.010),(12.,15.,.003)],
        'rapid_train':[(x,x+.8,.011) for x in (4.,6.,8.,10.)]}
    for start,end,amp in pulses[protocol]:y[(clock>=start)&(clock<end)]=amp
    return y

def design(spec):
    base=[dict(protocol=protocol,panel=panel,seed=seed,initial_voltage_mv=v,schedule=protocol,
               intervention='none') for protocol in spec['protocols'] for panel in spec['panels']
          for seed in spec['native_seeds'] for v in spec['initial_voltages_mv']]
    rows=[dict(**r,id=k,source_index=k) for k,r in enumerate(base)]
    for intervention in spec['interventions']:
        for k,r in enumerate(base):
            if r['panel']=='canonical':rows.append(dict(**{**r,'intervention':intervention},id=len(rows),source_index=k))
    assert len(base)==64 and len(rows)==176
    return rows

def case_inputs(row,spec):
    clock=np.arange(round(spec['duration_ms']/spec['dt_ms']))*spec['dt_ms']
    inj=injection(row['protocol'],clock);mult=np.array(spec['panels'][row['panel']],dtype=float)
    intervention=row['intervention']
    if intervention.startswith('block_'):mult[ionic.CHANNELS.index(intervention[6:])]=0
    if intervention.startswith('injection_'):inj*=float(intervention.split('_')[1])
    return inj,mult

def load_b6(torch,root,device):
    spec,_,_,_=contract(root)
    parent=t.load_models(torch,root,device)['J1']
    a=next(a for a in b.arms() if a['name']=='B6')
    model=b.make_model(torch,parent,a,spec['model_seeds'],device)
    state=torch.load(root/spec['checkpoint'],map_location=device,weights_only=True)['B6']
    model.load_state_dict(state);model.eval()
    for name,param in model.named_parameters():param.requires_grad_(False);assert torch.isfinite(param).all(),name
    return model

def calibration_config(spec,cfg):
    return {**cfg,'schedules_ms':spec['schedules_ms'],'duration_ms':spec['duration_ms'],
            'receptor_time_semantics':spec['receptor_time_semantics']}

def native(root,teacher,output):
    import neuron
    from neuron import h,load_mechanisms
    import subprocess
    spec,cfg,calcium,base=contract(root);cfg=calibration_config(spec,cfg)
    assert neuron.__version__.split('+')[0]==spec['neuron_version']
    assert subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()==spec['teacher_revision']
    _,_,c3=t.t33.contract(root);t.t33.t32.verify(root,teacher,calcium)
    build=t.t33.compile_combined(root,teacher,output);load_mechanisms(str(build.resolve()))
    cal=dict(protocol='calibration',panel='canonical',seed=spec['calibration_seed'],initial_voltage_mv=-70.,schedule='calibration',intervention='none')
    inj,mult=case_inputs(cal,spec)
    a=t.t33._native_case(h,cfg,base,calcium,c3,cal,injection_density=inj,channel_multipliers=mult)
    shadow=t.t33.shadow_case(h,cfg,c3,cal,a)
    floor=t.t33.metrics(t.t33.coupled_rollout([a],[shadow],cfg,calcium,base),[a])
    calibration=dict(valid=bool(t.t33._shadow_pass(shadow,cfg) and t.t33._floor_pass(floor,cfg)),formula_floor=floor,
        seed=spec['calibration_seed'],confirmation_accessed=False)
    p.save_json(output/'native_calibration.json',calibration)
    if not calibration['valid']:raise RuntimeError('Independent native calibration failed; confirmation not opened')
    natives=[];shadows=[];summaries=[];meta=design(spec);p.save_json(output/'cases.json',meta)
    for index,row in enumerate(meta):
        inj,mult=case_inputs(row,spec)
        a=t.t33._native_case(h,cfg,base,calcium,c3,row,zero_weight=row['intervention']=='zero_synapses',injection_density=inj,channel_multipliers=mult)
        s=t.t33.shadow_case(h,cfg,c3,row,a,zero_weight=row['intervention']=='zero_synapses')
        natives.append(a);shadows.append(s)
        summaries.append(dict(case=row,shadow_passed=t.t33._shadow_pass(s,cfg),release_count=s['release_count'],
            release_mismatch_count=s['release_mismatch_count'],max_rng_sequence_difference=s['max_rng_sequence_difference'],
            max_shadow_current_error_na=s['max_shadow_current_error_na'],max_shadow_receptor_state_error=s['max_shadow_receptor_state_error']))
        if (index+1)%16==0:
            np.savez_compressed(output/'native_partial.npz',
                **{k:np.stack([a[k] for a in natives]) for k in ('voltage','calcium','gates','injection','multipliers')},
                base_g_us=np.stack([s['base_g_us'] for s in shadows]),area_um2=np.array(natives[0]['area_um2']))
            p.save_json(output/'native_progress.json',dict(completed=index+1,total=len(meta),cases=summaries))
            print(f'[GIADA Task41] checkpoint native {index+1}/{len(meta)}',flush=True)
    prediction=t.t33.coupled_rollout(natives,shadows,cfg,calcium,base)
    floors={}
    for panel in spec['panels']:
        ids=[k for k,r in enumerate(meta) if r['panel']==panel]
        floors[panel]=t.t33.metrics({k:v[:,:,ids] for k,v in prediction.items()},[natives[k] for k in ids])
    np.savez_compressed(output/'native_shadow_traces.npz',
        **{k:np.stack([a[k] for a in natives]) for k in ('voltage','calcium','gates','injection','multipliers')},
        base_g_us=np.stack([s['base_g_us'] for s in shadows]),area_um2=np.array(natives[0]['area_um2']))
    report=dict(valid=all(t.t33._floor_pass(r,cfg) for r in floors.values()) and all(r['shadow_passed'] for r in summaries),
        new_native_simulations=True,case_count=len(meta),floor_by_panel=floors,cases=summaries,model_judged=False,
        teacher_revision=spec['teacher_revision'],neuron=neuron.__version__,thresholds_changed=False)
    p.save_json(output/'native_interface_report.json',report)
    if not report['valid']:raise RuntimeError('Fresh native/interface floor failed; frozen model not judged')
    print('[GIADA Task41] checkpoint native complete; interface floors admissible',flush=True)

def stress_design(spec):
    """New causal input histories; no old diagnostic trajectories copied."""
    steps=round(spec['stress_duration_ms']/spec['dt_ms']);clock=np.arange(steps)*spec['dt_ms'];rows=[]
    for family in ('history_recovery','activation_boundary'):
        for seed in spec['stress_seeds']:
            rng=np.random.default_rng(seed+(0 if family=='history_recovery' else 1000))
            common_v=np.full(steps,rng.uniform(-64,-57) if family=='history_recovery' else rng.uniform(-47,-38))
            common_v[clock<rng.uniform(12,24)]=rng.uniform(-30,-15)
            common_ca=np.full(steps,10**rng.uniform(-4.3,-3.7))
            for k in range(2):
                v,ca=common_v.copy(),common_ca.copy()
                initial_v=-95. if k==0 else -42.
                initial,_=p.coefficients(np.array(initial_v),np.array(ca[0]),spec['dt_ms'])
                panel=list(spec['panels'])[(k+spec['stress_seeds'].index(seed))%4]
                rows.append(dict(family=family,seed=seed,case=k,panel=panel,v=v,ca=ca,initial=initial))
    return rows

def stress(torch,root,output,model,spec,device):
    source=stress_design(spec);v=np.stack([r['v'] for r in source],1).astype('float32');ca=np.stack([r['ca'] for r in source],1).astype('float32')
    initial=np.stack([r['initial'] for r in source]);u,z=p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    g=initial.copy();truth=[]
    for step in range(len(v)):g=g+z[step]*(u[step]-g);truth.append(g.copy())
    truth=np.stack(truth);mult=np.array([spec['panels'][r['panel']] for r in source]);_,target=p.readout(v,truth,mult,ionic.REVERSALS)
    scanner=j.i.old.Scanner(torch,[model],len(source),32,device)
    predicted=scanner.run(v,ca,initial.astype('float32'))[0];rows=[]
    for s,seed in enumerate(spec['model_seeds']):
        _,current=p.readout(v,predicted[:,s],mult,ionic.REVERSALS)
        for k,row in enumerate(source):
            mm=p.evaluate_currents(target[:,k],current[:,k],.1,spec['current_quality'],truth[:,k],predicted[:,s,k])
            rows.append(dict(model_seed=seed,input_seed=row['seed'],family=row['family'],case=row['case'],panel=row['panel'],metrics=mm))
    np.savez_compressed(output/'fresh_stress_traces.npz',voltage=v,calcium=ca,initial=initial,target_gates=truth,predicted_gates=predicted,multipliers=mult)
    p.save_json(output/'fresh_stress_results.json',rows)
    print(f'[GIADA Task41] checkpoint fresh stress {sum(r["metrics"]["passed"] for r in rows)}/{len(rows)}',flush=True)
    return rows

def lut_model(torch,device,dt):
    class LUT:
        def __init__(self):
            self.grid=np.linspace(-100,60,16001,dtype='float64');u,z=p.coefficients(self.grid,np.full_like(self.grid,1e-4),dt)
            self.table=torch.tensor(np.stack((u,z),-1),device=device,dtype=torch.float32)
        def __call__(self,v,ca,g):
            x=((v+100)*100).clamp(0,16000);low=x.floor().long().clamp_max(15999);fraction=(x-low)[...,None,None]
            values=self.table[low]*(1-fraction)+self.table[low+1]*fraction
            return g+values[...,1]*(values[...,0]-g)
    return LUT()

def native_phase_targets(data):
    # NEURON one-step identity on opened fixtures: x_next is updated at V_next,
    # Ca_next. The unchanged explicit operator exposes voltage gates one phase
    # earlier while updating SK after its new Ca. This is a staggered interface,
    # not a change to the candidate nor a future teacher input in deployment.
    gate=data['gates'][:,:-1].transpose(1,0,2).copy()
    gate[...,16]=data['gates'][:,1:,16].T
    _,current=p.readout(data['voltage'][:,:-1].T,gate,data['multipliers'],ionic.REVERSALS)
    return gate,current

def numpy_imposed(data,cfg,calcium,base):
    gate=data['gates'][:,0].copy();gg=[gate.copy()];vv=[data['voltage'][:,0].copy()];cc=[data['calcium'][:,0].copy()];ii=[]
    for tick in range(data['injection'].shape[1]):
        old=data['voltage'][:,tick]
        v,ca,gate=t.n1.numpy_step(old,data['calcium'][:,tick],gate,data['multipliers'],data['injection'][:,tick],data['base_g_us'][:,tick],float(data['area_um2']),cfg,calcium,base)
        _,current=p.readout(old,gate,data['multipliers'],ionic.REVERSALS)
        vv.append(v.copy());cc.append(ca.copy());gg.append(gate.copy());ii.append(current)
    return dict(voltage=np.stack(vv),calcium=np.stack(cc),gates=np.stack(gg),current=np.stack(ii))

def evaluate(torch,root,output,device='cuda',smoke=False):
    spec,cfg,calcium,base=contract(root);cfg=calibration_config(spec,cfg)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_b6(torch,root,device)
    if smoke:
        v=torch.full((3,2),-60.,device=device);ca=torch.full_like(v,1e-4);g=torch.full((3,2,18),.5,device=device)
        args=[v,ca,g,torch.ones((3,2,11),device=device),torch.zeros_like(v),torch.zeros((3,2,4),device=device)]
        core=t.make_core(torch,cfg,calcium,base,100*np.pi,model,t.FAMILIES['all'])
        original=t._rates
        try:
            def forbidden(*a,**k):raise AssertionError('Analytical gate fallback')
            t._rates=forbidden
            eager=t.Stepper(torch,core,args);captured=t.Stepper(torch,core,args,device=='cuda');error=0.
            for _ in range(8):
                aa,bb=eager.tick(),captured.tick();error=max(error,max(float((a-b).abs().max()) for a,b in zip(aa,bb)))
        finally:t._rates=original
        assert error<=1e-5
        report=dict(valid=True,candidate='B6',all_learned_no_canonical_fallback=True,graph_eager_max_error=error,
            all_parameters_finite=all(bool(torch.isfinite(x).all()) for x in model.parameters()),fresh_confirmation_accessed=False)
        p.save_json(output/'gpu_preflight.json',report);return report
    assert json.loads((output/'native_interface_report.json').read_text())['valid']
    with np.load(output/'native_shadow_traces.npz') as z:data={k:z[k] for k in z.files}
    meta=design(spec);gold=t.numpy_reference(data,cfg,calcium,base)
    imposed={k:data[k].T for k in ('voltage','calcium')}
    rows={};saved={};references={};floors={};lut=lut_model(torch,device,.1);phase=[]
    torch.save(model.state_dict(),output/'B6_frozen.pt')
    for mode in ('imposed','autonomous'):
        ref_all=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,device,imposed if mode=='imposed' else None,True)
        ref={k:v[:,0] for k,v in ref_all.items()};references[mode]=ref
        floor=[]
        target=gold if mode=='autonomous' else numpy_imposed(data,cfg,calcium,base)
        for c in range(len(meta)):
            mm=p.evaluate_currents(target['current'][:,c],ref['current'][:,c],.1,spec['current_quality'],target['gates'][1:,c],ref['gates'][1:,c],data['multipliers'][c]==0)
            floor.append(dict(case=c,metrics=mm))
        differential=[]
        if mode=='autonomous':
            for c,row in enumerate(meta):
                if row['intervention']=='none':continue
                o=row['source_index'];differential.append(dict(case=c,metrics=p.evaluate_intervention(gold['current'][:,o],gold['current'][:,c],ref['current'][:,o],ref['current'][:,c],.1,spec['current_quality'])))
        floors[mode]=dict(valid=all(r['metrics']['passed'] for r in floor+differential),cases=floor,interventions=differential)
        p.save_json(output/'gpu_formula_floor.json',floors)
        assert floors[mode]['valid'],'Formula32/64 floor failed; model not judged'
        if mode=='imposed':
            native_gate,native_current=native_phase_targets(data)
            phase=[dict(case=c,metrics=p.evaluate_currents(native_current[:,c],ref['current'][:,c],.1,spec['current_quality'],native_gate[:,c],ref['gates'][1:,c],data['multipliers'][c]==0)) for c in range(len(meta))]
            native_ok=all(r['metrics']['passed'] for r in phase)
            p.save_json(output/'native_current_phase_floor.json',dict(valid=native_ok,cases=phase,
                interpretation='Staggered native reference: voltage gates at t,SK at t+dt,old native V. Not endpoint h.ica; no teacher future inputs in autonomous candidate.'))
            if not native_ok:
                report=dict(valid=False,technical_failure=True,scientific_no_go=False,model_judged=False,task='41',
                    decision='NATIVE_CURRENT_INTERFACE_FLOOR_FAILED',thresholds_changed=False,training_performed=False,task42_authorized=False)
                p.save_json(output/'final_report.json',report);return report
        for name,m in [('B6',model),('LUT_control',lut)]:
            channels=t.FAMILIES['all'] if name=='B6' else tuple(k for k in range(11) if k!=9)
            candidate=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),m,channels),data,device,imposed if mode=='imposed' else None,True)
            ref_metric={k:v.copy() for k,v in ref.items()}
            for k in ('voltage','calcium'):ref_metric[k]=data[k].T.copy()
            measured=t.metrics(candidate,ref_metric,data,meta,spec['current_quality'],cfg['gates'],mode)
            inter=[]
            for s,seed in enumerate(spec['model_seeds']):
                for c,row in enumerate(meta):
                    if row['intervention']=='none':continue
                    o=row['source_index'];inter.append(dict(case=c,seed=seed,metrics=p.evaluate_intervention(ref['current'][:,o],ref['current'][:,c],candidate['current'][:,s,o],candidate['current'][:,s,c],.1,spec['current_quality'])))
            measured['interventions']=inter;rows[name+'/'+mode]=measured;saved[name+'__'+mode]=candidate
            p.save_json(output/'causal_results.json',rows)
            print(f'[GIADA Task41] checkpoint {name}/{mode} trajectory={measured["passed_cases"]}/{measured["total"]} response={sum(r["metrics"]["passed"] for r in inter)}/{len(inter)}',flush=True)
    stress_rows=stress(torch,root,output,model,spec,device)
    np.savez_compressed(output/'candidate_traces.npz',**{label+'__'+k:v for label,r in saved.items() for k,v in r.items()})
    np.savez_compressed(output/'reference_traces.npz',**{label+'__'+k:v for label,r in dict(golden=gold,**references).items() for k,v in r.items()})
    cost=benchmark(torch,model,lut,data,spec,cfg,calcium,base);p.save_json(output/'cost.json',cost)
    native_current_ok=all(r['metrics']['passed'] for r in phase)
    passed=native_current_ok and all(rows['B6/'+mode]['passed'] for mode in ('imposed','autonomous')) and all(r['metrics']['passed'] for r in rows['B6/autonomous']['interventions']) and all(r['metrics']['passed'] for r in stress_rows)
    report=dict(valid=True,task='41',quality_passed=bool(passed),model_judged=native_current_ok,
        decision=('FROZEN_B6_INDEPENDENT_QUALITY_CONFIRMED_COST_REQUIRES_AUDIT' if passed else ('NATIVE_CURRENT_INTERFACE_FLOOR_FAILED' if not native_current_ok else 'FROZEN_B6_FRESH_SCIENTIFIC_NO_GO')),
        new_native_simulations=True,native_cases=len(meta),stress_case_seed_count=len(stress_rows),
        summary={k:dict(trajectory_passed=v['passed_cases'],total=v['total'],response_passed=sum(r['metrics']['passed'] for r in v['interventions']),response_total=len(v['interventions'])) for k,v in rows.items()},
        stress_passed=sum(r['metrics']['passed'] for r in stress_rows),native_current_phase_floor_passed=native_current_ok,
        independent_confirmation=True,training_performed=False,model_selection_performed=False,thresholds_changed=False,
        task42_authorized=False,speedup_claim=False,limitations=spec['limits'])
    p.save_json(output/'final_report.json',report);return report

def benchmark(torch,model,lut,data,spec,cfg,calcium,base):
    from .ie_ionic_cost import paired_summary
    results=[];start_tick=100;ticks=spec['cost']['ticks']
    for batch in spec['cost']['batches']:
        ids=np.arange(batch)%64
        def expand(a):return torch.tensor(a,device='cuda',dtype=torch.float32)[None].expand(3,*a.shape).contiguous()
        args=[expand(data[k][ids,start_tick]) for k in ('voltage','calcium','gates')]
        inj=[expand(data['injection'][ids,start_tick+k]) for k in range(ticks)]
        syn=[expand(data['base_g_us'][ids,start_tick+k]) for k in range(ticks)]
        args += [expand(data['multipliers'][ids]),inj[0],syn[0]]
        for scope in ('ionic','integrated'):
            runners={name:t.Stepper(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),m,channels,scope=='ionic'),args,True)
                for name,m,channels in [('formula',None,()),('B6',model,t.FAMILIES['all']),('LUT_control',lut,tuple(k for k in range(11) if k!=9))]}
            samples={k:dict(wall_ms=[],cuda_ms=[]) for k in runners}
            for repetition in range(spec['cost']['repetitions']):
                for name in list(runners) if repetition%2==0 else list(runners)[::-1]:
                    runner=runners[name];runner.reset(args);torch.cuda.synchronize();start=time.perf_counter()
                    aa,bb=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);aa.record()
                    for tick in range(ticks):runner.tick(inj[tick],syn[tick])
                    bb.record();bb.synchronize();samples[name]['wall_ms'].append((time.perf_counter()-start)*1000);samples[name]['cuda_ms'].append(aa.elapsed_time(bb))
            results.append(dict(batch_per_seed=batch,actual_sites=3*batch,scope=scope,samples=samples,
                B6_vs_formula=paired_summary(samples['formula']['wall_ms'],samples['B6']['wall_ms']),
                B6_vs_LUT=paired_summary(samples['LUT_control']['wall_ms'],samples['B6']['wall_ms']),
                finite={name:all(bool(torch.isfinite(x).all()) for x in runner.out) for name,runner in runners.items()},
                dtype='FP32 state;B6 FP64 affine correction',host_transfer_in_timed_block=False,quality_qualification_pending_audit=True))
            print(f'[GIADA Task41] checkpoint cost {scope} batch={batch}',flush=True)
    return results
