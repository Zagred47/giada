"""IE-P7 frozen native confirmation and explicit staged operational timing."""
import base64
import contextlib
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from . import ie_residual_repair as p6

p4=p6.p4
n1=p6.p5.n1
t33=n1.t33
CONFIG='experiments/ie_p7_preregistration_v1.json'

def write(path,value):p6.write(Path(path),value)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def contract(root):
    spec=read(root/CONFIG)
    assert sha(root/spec['checkpoint'])==spec['checkpoint_sha256'] and sha(root/spec['parent_checkpoint'])==spec['parent_sha256']
    _,cfg,ca,c3,base=n1.contract(root)
    assert cfg['dt_ms']==spec['dt_ms'] and cfg['synaptic_state_phase']=='old'
    cfg={**cfg,'schedules_ms':spec['schedules_ms'],'receptor_time_semantics':'ems_fixed_step'}
    return spec,cfg,ca,c3,base

def cases(spec):
    return [dict(panel=p,seed=s,initial_voltage_mv=v,protocol=i,schedule='p7_confirmation') for p,s,v,i in itertools.product(spec['panels'],spec['native_seeds'],spec['initial_voltages_mv'],spec['protocols'])]

def independent_forcing(rows,cfg,c3,constants,ticks,rng_factory):
    """Event-driven exact receptor convolution; no native outcomes consumed."""
    clock=np.asarray(constants['clock'][:ticks],dtype=np.float64)
    output=np.zeros((len(rows),ticks,4),dtype=np.float64);summaries=[]
    for k,row in enumerate(rows):
        total=0;positions=[]
        for j,params in enumerate(c3['synapses']):
            rng=rng_factory(row['seed'],cfg['random123_streams'][j]);state=dict(constants['initial_plastic'][j])
            active=('AMPA','NMDA') if params['mechanism']=='ProbAMPANMDA_EMS' else ('GABAA','GABAB')
            for event in cfg['schedules_ms'][row['schedule']][params['id']]:
                if event>clock[-1]:break
                state,_=t33.shadow_step(state,event,params,rng.repick)
                if not state['released']:continue
                total+=1;age=t33.fixed_step_event_age(clock,event,cfg['dt_ms']);mask=age>=0;elapsed=age[mask]
                for name in active:
                    rise,decay,factor=constants['parameters'][name]
                    ratio=c3['nmda_ratio'] if name=='NMDA' else c3['gabab_ratio'] if name=='GABAB' else 1.
                    response=np.exp(-elapsed/decay)-np.exp(-elapsed/rise)
                    output[k,mask,t33.RECEPTORS.index(name)]+=cfg['synapse_weight_ns']*ratio*factor*c3['gmax_us_per_ns']*response
            positions.append(int(rng.seq()))
        summaries.append({'release_count':total,'rng_positions':positions})
    return output,summaries

def array_metrics(v,ca,g,target):
    dv=v-target['voltage'];dc=ca-target['calcium']
    return {'voltage_rmse_mv':float(np.sqrt(np.mean(dv*dv))),'voltage_max_mv':float(np.max(np.abs(dv))),
        'calcium_max_mm':float(np.max(np.abs(dc))),'gate_rmse':float(np.sqrt(np.mean((g-target['gates'])**2))),
        'finite':bool(all(np.isfinite(x).all() for x in (v,ca,g))),
        'voltage_domain_exits':int(np.count_nonzero((v<-100)|(v>60))),
        'occupancy_violations':int(np.count_nonzero((g<0)|(g>1))), 'negative_calcium':int(np.count_nonzero(ca<0))}

def acceptable(m,q):
    return bool(m['finite'] and not(m['voltage_domain_exits'] or m['occupancy_violations'] or m['negative_calcium']) and m['voltage_rmse_mv']<=q['historical_voltage_rmse_mv'] and m['voltage_max_mv']<=q['historical_voltage_max_abs_mv'] and m['calcium_max_mm']<=q['historical_calcium_max_abs_mm'])

def numpy_floor(native,forcing,cfg,ca,base):
    v=np.array([x['voltage'][0] for x in native]);c=np.array([x['calcium'][0] for x in native]);g=np.stack([x['gates'][0] for x in native]);mult=np.stack([x['multipliers'] for x in native])
    vs=[v.copy()];cs=[c.copy()];gs=[g.copy()]
    for tick in range(len(native[0]['voltage'])-1):
        v,c,g=n1.numpy_step(v,c,g,mult,np.array([x['injection'][tick] for x in native]),forcing[:,tick],native[0]['area_um2'],cfg,ca,base)
        vs.append(v.copy());cs.append(c.copy());gs.append(g.copy())
    return np.stack(vs,1),np.stack(cs,1),np.stack(gs,1)

def native_phase(root,teacher,output):
    import neuron
    from neuron import h,load_mechanisms
    assert 'torch' not in sys.modules,'Native phase must not import CUDA/torch'
    spec,cfg,ca,c3,base=contract(root)
    assert neuron.__version__.split('+')[0]==spec['neuron_version']
    assert subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()==spec['teacher_revision']
    n1.t32.verify(root,teacher,ca);build=t33.compile_combined(root,teacher,output);load_mechanisms(str(build.resolve()))
    calibration=t33._native_case(h,cfg,base,ca,c3,spec['calibration'])
    constants={k:calibration[k] for k in ('parameters','initial_plastic')};constants['clock']=calibration['clock'].tolist()
    write(output/'generator_constants.json',constants)
    rng=lambda seed,stream:t33._rng(h,seed,stream)
    def audit(rows,natives):
        forcing,summaries=independent_forcing(rows,cfg,c3,constants,len(natives[0]['clock']),rng)
        reports=[]
        for i,(row,native) in enumerate(zip(rows,natives)):
            shadow=t33.shadow_case(h,cfg,c3,row,native)
            reports.append({'case':row,'shadow_passed':t33._shadow_pass(shadow,cfg),'shadow_metrics':{k:v for k,v in shadow.items() if not isinstance(v,np.ndarray)},'forcing_max_delta_us':float(np.max(np.abs(forcing[i]-shadow['base_g_us']))),
                'release_count_equal':summaries[i]['release_count']==shadow['release_count'],'final_rng_equal':summaries[i]['rng_positions']==native['rng_seq'][-1].tolist(),
                'initial_state_equal':native['initial_plastic']==constants['initial_plastic']})
        v,c,g=numpy_floor(natives,forcing,cfg,ca,base)
        metrics=[array_metrics(v[i],c[i],g[i],native) for i,native in enumerate(natives)]
        valid=all(x['shadow_passed'] and x['forcing_max_delta_us']<=1e-10 and x['release_count_equal'] and x['final_rng_equal'] and x['initial_state_equal'] for x in reports)
        return forcing,{'interface_valid':valid,'floor_passed':all(acceptable(m,spec['quality']) for m in metrics),'floor_metrics':metrics,'synaptic_audit':reports}
    _,calibration_report=audit([spec['calibration']],[calibration]);write(output/'calibration_report.json',calibration_report)
    if not calibration_report['interface_valid'] or not calibration_report['floor_passed']:
        write(output/'native_report.json',{'valid':False,'model_judged':False,'diagnosis':'CALIBRATION_INTERFACE_OR_FLOOR_FAILED','calibration':calibration_report});return
    rows=cases(spec);natives=[]
    for j,row in enumerate(rows):
        natives.append(t33._native_case(h,cfg,base,ca,c3,row))
        if (j+1)%4==0:print(f'[GIADA IE-P7] native {j+1}/{len(rows)}',flush=True)
    forcing,report=audit(rows,natives)
    zero=t33._native_case(h,cfg,base,ca,c3,rows[0],zero_weight=True);zs=t33.shadow_case(h,cfg,c3,rows[0],zero,zero_weight=True)
    controls={'zero_weight_no_release_no_rng':zs['release_count']==0 and np.array_equal(zero['rng_seq'][0],zero['rng_seq'][-1]) and float(np.max(np.abs(zero['native_i'])))==0}
    duplicate,_=independent_forcing(rows[:1],cfg,c3,constants,len(natives[0]['clock']),rng);controls['independent_duplicate_equal']=bool(np.array_equal(duplicate[0],forcing[0]))
    arrays={key:np.stack([n[key] for n in natives]) for key in ('voltage','calcium','gates','injection','multipliers')}
    arrays.update(base_g_us=forcing,area_um2=np.array(natives[0]['area_um2']))
    np.savez_compressed(output/'fresh_native_traces.npz',**arrays);write(output/'cases.json',rows)
    write(output/'native_report.json',{'valid':bool(report['interface_valid'] and report['floor_passed'] and all(controls.values())),'model_judged':False,
        'case_count':len(rows),'training_performed':False,'calibration':calibration_report,'controls':controls,**report})

def rng_server(root,output):
    from neuron import h
    assert 'torch' not in sys.modules
    _,cfg,_,c3,_=contract(root);constants=read(output/'generator_constants.json');rows=read(output/'cases.json')
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        start=time.perf_counter();selected=[rows[i] for i in request['indices']]
        forcing,_=independent_forcing(selected,cfg,c3,constants,request['ticks'],lambda s,r:t33._rng(h,s,r))
        # FP32 is the declared GPU input; serialization is included in caller timing.
        payload=base64.b64encode(forcing.astype(np.float32).tobytes()).decode('ascii')
        print(json.dumps({'shape':forcing.shape,'payload':payload,'generation_seconds':time.perf_counter()-start}),flush=True)

def hybrid_operator(torch,model):
    def op(v):
        shape=v.shape;v=v.reshape(-1)
        packed=torch.cat((v[:,None],torch.zeros((len(v),10),device=v.device,dtype=v.dtype),torch.full_like(v[:,None],.1)),-1)
        _,u,tau=model(packed[None]);u=u[0].transpose(0,1).reshape(-1,10);tau=tau[0].transpose(0,1).reshape(-1,10)
        us=[u];zs=[-torch.expm1(-.1/tau)]
        for k,name in enumerate(p4.ionic.CHANNELS[5:],5):
            if name=='SK_E2':continue
            a,b=p4._rates(name,v,torch.full_like(v,1e-4),torch)
            if len(p4.ionic.STATE_NAMES[k])==1:a,b=a[...,None],b[...,None]
            us.append(a);zs.append(-torch.expm1(-.1/b))
        return torch.cat(us,-1).reshape(*shape,17),torch.cat(zs,-1).reshape(*shape,17)
    return op

def operators(torch,root,device):
    spec,_,ca,_,_=contract(root)
    parent=torch.load(root/spec['parent_checkpoint'],map_location='cpu',weights_only=True);saved=torch.load(root/spec['checkpoint'],map_location='cpu',weights_only=True)
    ops={'formula_vectorized':None,'lut025':p4.lut_operator(torch,device=device)}
    for knots in spec['knots']:
        model=p6.local(torch,parent,knots,device);model.load_state_dict({k:v.to(device) for k,v in saved[f'local{knots}'].items()})
        for i,s in enumerate(spec['model_seeds']):ops[f'local{knots}_seed{s}']=p6.local_operator(torch,model,i)
    loaded,seeds=p4.ionic.load_frozen(root,p4.ionic.config(root),torch,device)
    from .joint_heterogeneous_comparison import model_factory
    for family in ca['families']:
        full=loaded[family,ca['frozen_arm']];width=full.b2.shape[-1]
        for i,s in enumerate(seeds):
            single=model_factory(torch,width,family,[s]).to(device)
            single.load_state_dict({k:v[i:i+1] for k,v in full.state_dict().items()});single.eval()
            for p in single.parameters():p.requires_grad_(False)
            ops[f'hybrid_{family}_seed{s}']=hybrid_operator(torch,single)
    assert len(ops)==14
    return ops

def native_quality(torch,ops,data,cfg,ca,base,spec):
    def t(x):return torch.tensor(x,device='cuda',dtype=torch.float32)
    mult=t(data['multipliers']);inj=t(data['injection']);syn=t(data['base_g_us']);result={}
    for name,op in ops.items():
        v=t(data['voltage'][:,0]);c=t(data['calcium'][:,0]);g=t(data['gates'][:,0]);vs=[v.clone()];cs=[c.clone()];gs=[g.clone()]
        for tick in range(400):
            v,c,g,*_=p4.core(torch,v,c,g,mult,inj[:,tick],syn[:,tick],float(data['area_um2']),cfg,ca,base,op)
            vs.append(v);cs.append(c);gs.append(g)
        vv=torch.stack(vs,1).cpu().numpy();cc=torch.stack(cs,1).cpu().numpy();gg=torch.stack(gs,1).cpu().numpy();metrics=[]
        for j in range(len(vv)):
            target={key:data[key][j] for key in ('voltage','calcium','gates')};m=array_metrics(vv[j],cc[j],gg[j],target)
            teacher=np.flatnonzero((target['voltage'][:-1]<-20)&(target['voltage'][1:]>=-20));pred=np.flatnonzero((vv[j,:-1]<-20)&(vv[j,1:]>=-20))
            m.update(native_crossings=len(teacher),spike_count_equal=len(teacher)==len(pred),spike_timing_max_ms=float(np.max(np.abs(teacher-pred))*.1) if len(teacher)==len(pred) and len(teacher) else None)
            metrics.append(m)
        result[name]={'episodes':metrics,'absolute_passed':all(acceptable(m,spec['quality']) for m in metrics),'behavior_support':sum(m['native_crossings'] for m in metrics)>=20}
        print(f'[GIADA IE-P7] independent native quality {name}: {result[name]["absolute_passed"]}',flush=True)
    floor=result['formula_vectorized']['episodes']
    for name,r in result.items():
        r['noninferiority_passed']=all(m['voltage_rmse_mv']-f['voltage_rmse_mv']<=.02 and m['calcium_max_mm']-f['calcium_max_mm']<=1e-6 for m,f in zip(r['episodes'],floor))
        r['spike_passed']=all(m['spike_count_equal'] and (m['spike_timing_max_ms'] is None or m['spike_timing_max_ms']<=.2) for m in r['episodes'])
        r['passed']=r['absolute_passed'] and r['noninferiority_passed'] and r['spike_passed']
    return result

def stepper(torch,op,data,indices,ticks,cfg,ca,base,device='cuda'):
    def t(x):return torch.tensor(x,device=device,dtype=torch.float32)
    v=t(data['voltage'][indices,0]);c=t(data['calcium'][indices,0]);g=t(data['gates'][indices,0]);initial=(v.clone(),c.clone(),g.clone())
    mult=t(data['multipliers'][indices]);injection=t(data['injection'][indices,:ticks]);forcing=t(data['base_g_us'][indices,:ticks]);tick=torch.zeros(1,device=device,dtype=torch.int64)
    def reset(from_host=False):
        if from_host:
            v.copy_(t(data['voltage'][indices,0]));c.copy_(t(data['calcium'][indices,0]));g.copy_(t(data['gates'][indices,0]))
            mult.copy_(t(data['multipliers'][indices]));injection.copy_(t(data['injection'][indices,:ticks]))
        else:
            v.copy_(initial[0]);c.copy_(initial[1]);g.copy_(initial[2])
        tick.zero_()
    def step():
        syn=torch.index_select(forcing,1,tick).squeeze(1);inj=torch.index_select(injection,1,tick).squeeze(1)
        nv,nc,ng,*_=p4.core(torch,v,c,g,mult,inj,syn,float(data['area_um2']),cfg,ca,base,op)
        v.copy_(nv);c.copy_(nc);g.copy_(ng);tick.add_(1)
        return v,c,g
    return step,reset,(v,c,g),forcing

def workload(torch,root,output,ops,data,cfg,ca,base,spec,server):
    w=spec['workload'];rows=[];checks=[];setup=[];timing_seed=np.random.default_rng(spec['data_seed'])
    for batch in w['batches']:
        indices=timing_seed.integers(0,len(data['voltage']),batch);ticks=w['ticks'];functions={};retained=[]
        for name,op in ops.items():
            setup_start=time.perf_counter()
            eager,reset,state,forcing=stepper(torch,op,data,indices,ticks,cfg,ca,base)
            reset()
            for _ in range(ticks):eager()
            reference=[x.clone() for x in state];reset();compiled=torch.compile(eager,fullgraph=True)
            for _ in range(ticks):compiled()
            torch.cuda.synchronize();check=p4.p2.differences(torch,state,reference,w['numeric_atol'])
            checks.append({'batch':batch,'name':name,'mode':'compiled',**check})
            if not check['valid']:raise RuntimeError(f'Operational compile mismatch {name}')
            functions['compiled',name]=(compiled,reset,state,forcing)
            reset();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):compiled()
            stream.synchronize();reset();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):compiled()
            reset()
            for _ in range(ticks):graph.replay()
            torch.cuda.synchronize();check=p4.p2.differences(torch,state,reference,w['numeric_atol']);checks.append({'batch':batch,'name':name,'mode':'cuda_graph',**check})
            if not check['valid']:raise RuntimeError(f'Operational graph mismatch {name}')
            functions['cuda_graph',name]=(graph.replay,reset,state,forcing);retained.append(graph)
            setup.append({'batch':batch,'name':name,'compile_warmup_graph_preflight_seconds_excluded':time.perf_counter()-setup_start})
        measurements={k:[] for k in functions}
        for rep in range(w['repetitions']):
            keys=list(functions);keys=keys if rep%2==0 else keys[::-1]
            for key in keys:
                fn,reset,state,forcing=functions[key];torch.cuda.synchronize();start=time.perf_counter()
                server.stdin.write(json.dumps({'indices':indices.tolist(),'ticks':ticks})+'\n');server.stdin.flush()
                line=server.stdout.readline()
                if not line:raise RuntimeError('Independent RNG generator subprocess closed pipe')
                reply=json.loads(line);values=np.frombuffer(base64.b64decode(reply['payload']),dtype=np.float32).reshape(reply['shape']);rpc=time.perf_counter()
                forcing.copy_(torch.from_numpy(values.copy()).to('cuda'));reset(from_host=True);torch.cuda.synchronize();transferred=time.perf_counter()
                for _ in range(ticks):fn()
                torch.cuda.synchronize();integrated=time.perf_counter()
                endpoints=[x.cpu().numpy().copy() for x in state];finished=time.perf_counter();assert all(np.isfinite(x).all() for x in endpoints)
                measurements[key].append({'wall_ms':(finished-start)*1000,'rng_events_rpc_ms':(rpc-start)*1000,'h2d_reset_ms':(transferred-rpc)*1000,
                    'rollout_ms':(integrated-transferred)*1000,'d2h_ms':(finished-integrated)*1000,'generator_reported_ms':reply['generation_seconds']*1000})
        for (mode,name),samples in measurements.items():
            control=[x['wall_ms'] for x in measurements[mode,'formula_vectorized']];times=[x['wall_ms'] for x in samples]
            rows.append({'batch':batch,'mode':mode,'name':name,'ticks':ticks,'samples':samples,'saving_vs_formula':p4.paired_summary(control,times),
                'saving_vs_hybrids':{n:p4.paired_summary([x['wall_ms'] for x in measurements[mode,n]],times) for n in ops if n.startswith('hybrid_')}})
        write(output/'operational_timing_partial.json',{'rows':rows,'preflights':checks})
        print(f'[GIADA IE-P7] operational timing batch={batch} complete',flush=True)
    return {'rows':rows,'preflights':checks,'setup_excluded_from_samples':setup,'scope':spec['operational_scope']}

class Progress:
    def write(self,value):return sys.__stdout__.write(value.replace('[GIADA IE-P4]','[GIADA IE-P7]'))
    def flush(self):sys.__stdout__.flush()

def gpu_phase(root,output,revision,server_command):
    import os
    import torch
    assert 'neuron' not in sys.modules,'CUDA phase must not import NEURON'
    spec,cfg,ca,_,base=contract(root);native=read(output/'native_report.json')
    if not native['valid']:
        report={'valid':True,'model_judged':False,'candidate_promoted':False,'diagnosis':'NATIVE_INTERFACE_OR_FORMULA_FLOOR_BLOCKED','native':native,'code_revision':revision};write(output/'final_report.json',report);return report
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(256,torch._dynamo.config.cache_size_limit)
    ops=operators(torch,root,'cuda');data=dict(np.load(output/'fresh_native_traces.npz'))
    with torch.inference_mode():
        floor=native_quality(torch,{'formula_vectorized':None},data,cfg,ca,base,spec);write(output/'gpu_formula_floor.json',floor)
        if not floor['formula_vectorized']['absolute_passed']:
            report={'valid':True,'model_judged':False,'candidate_promoted':False,'diagnosis':'GPU_FORMULA_NATIVE_FLOOR_BLOCKED','native_quality':floor,'code_revision':revision};write(output/'final_report.json',report);return report
        quality=native_quality(torch,ops,data,cfg,ca,base,spec);write(output/'native_quality.json',quality)
        atomic_spec={**spec,'fresh_count':8192,'path_seed':82002};fresh=p6.fresh(torch,ops,atomic_spec,output);write(output/'fresh_evaluation.json',fresh)
        with contextlib.redirect_stdout(Progress()):core_rows,core_checks,failures=p4.timing(torch,root,ops,spec,cfg,ca,base,output)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'};start=time.perf_counter()
        with (output/'rng_server.log').open('w') as log:
            server=subprocess.Popen(server_command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env,cwd=root)
            try:
                if not json.loads(server.stdout.readline()).get('ready'):raise RuntimeError('RNG server not ready')
                startup=time.perf_counter()-start
                operational=workload(torch,root,output,ops,data,cfg,ca,base,spec,server)
            finally:
                if server.poll() is None:
                    server.stdin.write('{"stop":true}\n');server.stdin.flush();server.wait(timeout=20)
        operational['server_startup_seconds_excluded']=startup;write(output/'operational_timing.json',operational)
    passed={}
    for name,f in fresh.items():
        passed[name]=quality[name]['passed'] and f['fresh_gate']['rmse']<=.001 and f['fresh_gate']['max_abs']<=.01 and not f['occupancy_violations'] and all(m['rmse']<=.005 and m['max_abs']<=.03 for m in f['families'].values())
    def fast(s):return s['median_saved_fraction']>=.1 and s['bootstrap_95pct'][0]>0
    decisions={}
    for knots in spec['knots']:
        per_seed={}
        for s in spec['model_seeds']:
            name=f'local{knots}_seed{s}';core=next(r for r in core_rows if r['batch']==642 and r['mode']=='cuda_graph' and r['name']==name)
            total=next(r for r in operational['rows'] if r['batch']==642 and r['mode']=='cuda_graph' and r['name']==name)
            per_seed[str(s)]={'quality':passed[name],'core_cost':fast(core['saving_vs_formula']),'operational_cost':fast(total['saving_vs_formula']),
                'core_vs_hybrids':all(fast(p4.paired_summary(r['timings']['wall_ms'],core['timings']['wall_ms'])) for r in core_rows if r['batch']==642 and r['mode']=='cuda_graph' and r['name'].startswith('hybrid_')),
                'operational_vs_hybrids':all(fast(v) for v in total['saving_vs_hybrids'].values())}
        decisions[f'local{knots}']={'seeds':per_seed,'all_registered_gates_passed':all(all(v.values()) for v in per_seed.values())}
    report={'valid':not failures,'schema_version':spec['schema_version'],'code_revision':revision,'training_performed':False,'model_judged':True,'device':torch.cuda.get_device_name(),
        'native_report':native,'native_quality':quality,'fresh':fresh,'quality_passed':passed,'core_rows':core_rows,'core_preflights':core_checks,'graph_failures':failures,
        'operational':operational,'family_decisions':decisions,'candidate_promoted':False,'diagnosis':'FROZEN_CONFIRMATION_AND_OPERATIONAL_COST_MEASURED','exclusions':spec['exclusions']}
    write(output/'final_report.json',report);print('[GIADA IE-P7] complete; no automatic promotion',flush=True);return report
