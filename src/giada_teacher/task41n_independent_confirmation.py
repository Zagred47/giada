"""Frozen41m independent native confirmation plus exact execution controls."""
import hashlib,io,json,os,time,zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from . import task41n_exact_execution as execution
from . import task41c_frozen_confirmation as old_native
m=execution.m;p,t=m.p,m.t
CONFIG='experiments/task41n_preregistration_v1.json'

def tensor_bits_equal(torch,a,b):
    if a.dtype!=b.dtype or a.shape!=b.shape:return False
    dtype=torch.int64 if a.element_size()==8 else torch.int32
    return torch.equal(a.contiguous().view(dtype),b.contiguous().view(dtype))

def array_bits_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))

def weights_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def contract(root):
    spec=json.loads((root/CONFIG).read_text());prior,_,cfg,calcium,base=m.contract(root)
    for key in ('quality','compartment_gates','cost'):assert spec[key]==prior[key],key
    assert spec['training_updates']==0 and spec['selection_performed'] is False
    assert not set(spec['native_seeds']+spec['stress_seeds']+[spec['calibration_seed']])&set(spec['excluded_input_seeds'])
    for name,sha in spec['sources'].items():assert p.digest(root/name)==sha,name
    return spec,cfg,calcium,base

def case_inputs(row,spec):
    clock=np.arange(round(spec['duration_ms']/.1))*.1;inj=np.zeros_like(clock)
    pulses=[(2.,5.,.003)] if row['protocol']=='calibration' else spec['pulses'][row['protocol']]
    for start,stop,amplitude in pulses:inj[(clock>=start)&(clock<stop)]=amplitude
    mult=np.array(spec['panels'][row['panel']],dtype=float);action=row['intervention']
    if action.startswith('block_'):mult[old_native.local.ionic.CHANNELS.index(action[6:])]=0
    if action.startswith('injection_'):inj*=float(action.split('_')[1])
    return inj,mult

def native(root,teacher,out):
    spec,cfg,calcium,base=contract(root)
    x=old_native.x;before=(x.contract,x.case_inputs)
    x.contract=lambda unused:(spec,cfg,calcium,base);x.case_inputs=case_inputs
    try:x.native(root,teacher,out)
    finally:x.contract,x.case_inputs=before

def load_models(torch,root,device):
    spec,*_=contract(root);path=Path(os.environ['GIADA_TASK41N_FROZEN'])
    assert p.digest(path)==spec['frozen_input']['sha256']
    with zipfile.ZipFile(path) as z:
        raw=z.read('frozen_models.pt');assert hashlib.sha256(raw).hexdigest()==spec['frozen_input']['checkpoint_sha256']
        states=torch.load(io.BytesIO(raw),map_location=device,weights_only=True)
    parent=m.l.j.controls(torch,root,device)['H_nominee']
    donor=m.l.j.model_class(torch)(parent,m.l.k.arms()[4]).to(device)
    dual=m.a.swap_class(torch)(parent,donor,(1,0,0)).to(device);dual.load_state_dict(states['swap100'])
    for par in dual.parameters():par.requires_grad_(False);assert torch.isfinite(par).all()
    return execution.variants(torch,dual),parent

def parity(torch,nets,device,spec):
    rows=[];eligible=[];torch.manual_seed(spec['parity_seed'])
    with torch.inference_mode():
        for batch in (1,2,64,176,642,4096):
            v=torch.linspace(-110,70,batch,device=device)[None].expand(3,-1)
            ca=torch.logspace(-8,-1,batch,device=device)[None].expand(3,-1)
            # Preserve original GEMM shapes, include random states and endpoints.
            gate=torch.rand((3,batch,18),device=device);reference=nets['dual'].raw(v,ca);step=nets['dual'](v,ca,gate)
            for name,net in nets.items():
                raw=net.raw(v,ca);actual=net(v,ca,gate)
                exact=all(tensor_bits_equal(torch,a,b) for a,b in zip(raw,reference)) and tensor_bits_equal(torch,actual,step)
                rows.append(dict(model=name,batch=batch,bitwise_equal=exact,max_abs_error=max([float((a-b).abs().max()) for a,b in zip(raw,reference)]+[float((actual-step).abs().max())])))
        eligible=[name for name in nets if all(r['bitwise_equal'] for r in rows if r['model']==name)]
    return dict(valid='dual' in eligible,rows=rows,eligible=eligible,atol=0,rtol=0,fresh_accessed=False,selection_on_fresh=False)

def preflight(torch,root,out,device):
    out.mkdir(parents=True,exist_ok=True)
    spec,cfg,ca,base=contract(root);nets,_=load_models(torch,root,device);report=parity(torch,nets,device,spec)
    # Also exercise the exact mixed-precision CUDA graph and inherited metrics.
    fake=dict(voltage=np.full((4,9),-65.,dtype='float32'),calcium=np.full((4,9),1e-4,dtype='float32'),gates=np.full((4,9,18),.01,dtype='float32'),
        injection=np.full((4,8),.001,dtype='float32'),base_g_us=np.full((4,8,4),1e-5,dtype='float32'),multipliers=np.ones((4,11),dtype='float32'),area_um2=np.array(100*np.pi))
    with torch.inference_mode():
        reference=m.rollout(torch,nets['dual'],fake,(cfg,ca,base),device,'voltage_solve64')
        for name in report['eligible']:
            pred=m.rollout(torch,nets[name],fake,(cfg,ca,base),device,'voltage_solve64')
            assert all(array_bits_equal(pred[k],reference[k]) for k in pred),(name,'coupled parity')
        meta=[dict(id=k,source_index=0,intervention='none' if k==0 else 'test') for k in range(4)]
        m.measure(reference,reference,fake,meta,spec,cfg,'autonomous','41')
        if device=='cuda':
            cost_data=dict(fake)
            for key in ('voltage','calcium','gates','injection','base_g_us'):cost_data[key]=np.repeat(fake[key][:1,:1],64,axis=0).repeat(103,axis=1)
            cost_data['multipliers']=np.ones((64,11),dtype='float32')
            small=dict(spec,cost=dict(spec['cost'],batches=[1],ticks=2,repetitions=2))
            folder=out/'preflight_cost';folder.mkdir(exist_ok=True)
            benchmark(torch,nets,nets['dual'].parent,cost_data,(cfg,ca,base),small,folder,device)
    report.update(gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',torch=torch.__version__,protocol_sha256=p.digest(root/CONFIG),all_parameters_finite=True)
    p.save_json(out/'gpu_preflight.json',report);return report

def stress_data(spec):
    clock=np.arange(round(spec['stress_duration_ms']/.1))*.1;rows=[]
    for family_index,family in enumerate(spec['stress_families']):
        for seed in spec['stress_seeds']:
            rng=np.random.default_rng(seed+1000*family_index)
            rest=rng.uniform(-48,-37) if family in ('slow_boundary_return','near_threshold_staircase') else rng.uniform(-65,-48)
            v=np.full(len(clock),rest);ca=np.full(len(clock),10**rng.uniform(-4.4,-3.6))
            stop=rng.uniform(12,25);v[clock<stop]=rng.uniform(-31,-14)
            if family=='doublepulse_recovery':v[(clock>=85)&(clock<85+rng.uniform(8,20))]=rng.uniform(-35,-18)
            elif family=='slow_boundary_return':
                ix=(clock>=stop)&(clock<stop+30);v[ix]=np.linspace(v[np.flatnonzero(ix)[0]-1],rest,ix.sum())
            elif family=='calcium_swing':
                rise=np.clip((clock-100)/100,0,1);fall=np.clip((700-clock)/300,0,1)
                ca=ca+(rng.uniform(2e-4,5e-4)-ca)*np.minimum(rise,fall)
            elif family=='near_threshold_staircase':v[(clock>=40)&(clock<90)]=rest+2;v[(clock>=90)&(clock<140)]=rest-2
            for history,initial_v in enumerate((-95.,-42.)):
                initial,_=p.coefficients(np.array(initial_v),np.array(ca[0]),.1)
                panel=list(spec['panels'])[(history+spec['stress_seeds'].index(seed))%4]
                rows.append(dict(family=family,input_seed=seed,history=history,panel=panel,v=v.copy(),ca=ca.copy(),initial=initial))
    return rows

def scan(torch,model,v,ca,initial,device):
    vv=torch.tensor(v,device=device);cc=torch.tensor(ca,device=device)
    args=[vv[0][None].expand(3,-1).contiguous(),cc[0][None].expand(3,-1).contiguous(),torch.tensor(initial,device=device,dtype=torch.float32)[None].expand(3,-1,-1).contiguous()]
    if device=='cuda':
        for _ in range(3):model(*args)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=model(*args)
    output=[]
    for tick in range(len(v)):
        args[0].copy_(vv[tick][None]);args[1].copy_(cc[tick][None])
        if device=='cuda':graph.replay()
        else:result=model(*args)
        output.append(result.clone());args[2].copy_(result)
    return torch.stack(output).cpu().numpy()

def evaluate_stress(torch,nets,spec,out,device):
    rows=stress_data(spec);v=np.stack([r['v'] for r in rows],1).astype('float32');ca=np.stack([r['ca'] for r in rows],1).astype('float32');initial=np.stack([r['initial'] for r in rows])
    u,z=p.coefficients(v.astype('float64'),ca.astype('float64'),.1);state=initial.copy();truth=[]
    for tick in range(len(v)):state=state+z[tick]*(u[tick]-state);truth.append(state.copy())
    truth=np.stack(truth);mult=np.array([spec['panels'][r['panel']] for r in rows]);target=p.readout(v,truth,mult,old_native.local.ionic.REVERSALS)[1]
    reports={};dual=None
    for name,net in nets.items():
        gates=scan(torch,net,v,ca,initial,device)
        if name=='dual':dual=gates
        exact=array_bits_equal(dual,gates)
        measured=[]
        for s,seed in enumerate(spec['seeds']):
            current=p.readout(v,gates[:,s],mult,old_native.local.ionic.REVERSALS)[1]
            for k,row in enumerate(rows):measured.append(dict(model_seed=seed,**{key:row[key] for key in ('family','input_seed','history','panel')},metrics=p.evaluate_currents(target[:,k],current[:,k],.1,spec['quality'],truth[:,k],gates[:,s,k])))
        reports[name]=dict(rows=measured,passed=all(r['metrics']['passed'] for r in measured),exact_to_dual=exact)
        np.savez_compressed(out/f'stress_{name}.npz',gates=gates)
        p.save_json(out/'fresh_stress_results.json',reports)
        print(f'[GIADA41n] stress {name}: {sum(r["metrics"]["passed"] for r in measured)}/{len(measured)} exact={exact}',flush=True)
    np.savez_compressed(out/'fresh_stress_inputs.npz',voltage=v,calcium=ca,initial=initial,target_gates=truth,multipliers=mult)
    return reports

def benchmark(torch,nets,parent,data,physics,spec,out,device):
    from .ie_ionic_cost import paired_summary
    results=[]
    for batch in spec['cost']['batches']:
        ix=np.arange(batch)%64
        for scope in ('ionic','integrated'):
            runners={};buffers={};streams={};samples={}
            definitions={'formula32':(None,'primary'),'formula64':(None,'voltage_solve64'),'H_native':(parent,'voltage_solve64'),**{k:(v,'voltage_solve64') for k,v in nets.items()}}
            if scope=='ionic':definitions.pop('formula64')
            for name,(net,policy) in definitions.items():
                dtype=torch.float32 if scope=='ionic' or policy=='primary' else torch.float64
                ex=lambda x,dt:torch.tensor(x,device=device,dtype=dt)[None].expand(3,*x.shape).contiguous()
                args=[ex(data['voltage'][ix,100],dtype),ex(data['calcium'][ix,100],torch.float32),ex(data['gates'][ix,100],torch.float32),ex(data['multipliers'][ix],torch.float32),ex(data['injection'][ix,100],dtype),ex(data['base_g_us'][ix,100],dtype)]
                core=t.make_core(torch,*physics,float(data['area_um2']),net,() if net is None else t.FAMILIES['all'],True) if scope=='ionic' else m.core(torch,net,physics,float(data['area_um2']),policy)
                runners[name]=t.Stepper(torch,core,args,True);buffers[name]=args
                streams[name]=[(ex(data['injection'][ix,100+k],dtype),ex(data['base_g_us'][ix,100+k],dtype)) for k in range(spec['cost']['ticks'])]
                samples[name]=dict(wall_ms=[],cuda_ms=[])
            for repeat in range(spec['cost']['repetitions']):
                for name in list(runners) if repeat%2==0 else reversed(runners):
                    run=runners[name];run.reset(buffers[name]);torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record()
                    for inj,syn in streams[name]:run.tick(inj,syn)
                    b.record();b.synchronize();samples[name]['wall_ms'].append(1000*(time.perf_counter()-start));samples[name]['cuda_ms'].append(a.elapsed_time(b))
            results.append(dict(batch_per_seed=batch,actual_sites=3*batch,scope=scope,samples=samples,vs_dual={k:paired_summary(samples['dual']['wall_ms'],v['wall_ms']) for k,v in samples.items()},vs_original_formula={k:paired_summary(samples['formula32']['wall_ms'],v['wall_ms']) for k,v in samples.items()},all_outputs_finite=all(all(torch.isfinite(x).all() for x in run.out) for run in runners.values()),quality_qualified=False))
            p.save_json(out/'cost.json',results);print(f'[GIADA41n] cost {scope}/{batch}',flush=True)
            del runners;__import__('gc').collect();torch.cuda.empty_cache()
    return results

def evaluate(torch,root,out,device='cuda'):
    spec,cfg,calcium,base=contract(root);cfg=old_native.x.calibration_config(spec,cfg)
    assert json.loads((out/'native_interface_report.json').read_text())['valid']
    freeze=json.loads((out/'execution_freeze.json').read_text());assert freeze['protocol_sha256']==p.digest(root/CONFIG) and freeze['fresh_accessed'] is False
    nets,parent=load_models(torch,root,device);nets={k:v for k,v in nets.items() if k in freeze['eligible']}
    initial_weights={name:weights_hash(net) for name,net in nets.items()}
    with np.load(out/'native_shadow_traces.npz') as z:data=dict(z)
    meta=old_native.x.design(spec);reports={};exact=[];floors=[]
    for mode in ('imposed','autonomous'):
        imposed={k:data[k].T for k in ('voltage','calcium')} if mode=='imposed' else None
        refs={};target=old_native.x.numpy_imposed(data,cfg,calcium,base) if mode=='imposed' else t.numpy_reference(data,cfg,calcium,base)
        for policy in ('primary','voltage_solve64'):
            ref=m.rollout(torch,None,data,(cfg,calcium,base),device,policy,imposed);refs[policy]=ref
            checks=[p.evaluate_currents(target['current'][:,k],ref['current'][:,0,k],.1,spec['quality'],target['gates'][1:,k],ref['gates'][1:,0,k],data['multipliers'][k]==0) for k in range(len(meta))]
            checks += [p.evaluate_intervention(target['current'][:,r['source_index']],target['current'][:,k],ref['current'][:,0,r['source_index']],ref['current'][:,0,k],.1,spec['quality']) for k,r in enumerate(meta) if r['intervention']!='none']
            floors.append(dict(mode=mode,policy=policy,rows=checks,passed=all(r['passed'] for r in checks)))
            np.savez_compressed(out/f'reference_{mode}_{policy}.npz',**ref)
        p.save_json(out/'canonical_floors.json',floors)
        assert all(r['passed'] for r in floors),'Fresh canonical floor failed; model not judged'
        if mode=='imposed':
            ng,nc=old_native.x.native_phase_targets(data)
            phase=[p.evaluate_currents(nc[:,k],refs['primary']['current'][:,0,k],.1,spec['quality'],ng[:,k],refs['primary']['gates'][1:,0,k],data['multipliers'][k]==0) for k in range(len(meta))]
            p.save_json(out/'native_phase_floor.json',phase);assert all(r['passed'] for r in phase),'Native staggered-interface floor failed'
        dual=None
        for name,net in {**nets,'H_control':parent,'dual_fp32':nets['dual']}.items():
            policy='primary' if name=='dual_fp32' else 'voltage_solve64'
            pred=m.rollout(torch,net,data,(cfg,calcium,base),device,policy,imposed)
            assert all(np.isfinite(x).all() for x in pred.values())
            if name=='dual':dual=pred
            if name in nets:
                same=all(array_bits_equal(pred[k],dual[k]) for k in pred);exact.append(dict(mode=mode,model=name,bitwise_equal=same))
            for reference,ref in refs.items():
                value=m.measure(pred,ref,data,meta,spec,cfg,mode,'41n')
                reports[name+'/'+mode+'/'+reference]=dict(summary=m.summary(value),metrics=value)
            np.savez_compressed(out/f'traces_{name}_{mode}.npz',**pred)
            p.save_json(out/'causal_results.json',reports);p.save_json(out/'fresh_execution_parity.json',exact)
            print(f'[GIADA41n] {name}/{mode}: '+str(reports[name+'/'+mode+'/primary']['summary']),flush=True)
    stresses=evaluate_stress(torch,nets,spec,out,device)
    costs=benchmark(torch,nets,parent,data,(cfg,calcium,base),spec,out,device)
    quality=all(reports['dual/'+mode+'/'+reference]['summary']['passed'] for mode in ('imposed','autonomous') for reference in ('primary','voltage_solve64')) and stresses['dual']['passed']
    equivalent=[name for name in nets if all(r['bitwise_equal'] for r in exact if r['model']==name) and stresses[name]['exact_to_dual']]
    assert initial_weights=={name:weights_hash(net) for name,net in nets.items()},'Frozen weights changed'
    return dict(valid=True,complete=True,task='41n',decision='INDEPENDENT_QUALITY_CONFIRMED_REQUIRES_AUDIT' if quality else 'INDEPENDENT_CONFIRMATION_NO_GO',
        independent_quality_passed=quality,equivalent_implementations=equivalent,summaries={k:v['summary'] for k,v in reports.items()},fresh_stress={k:dict(passed=v['passed'],passed_count=sum(r['metrics']['passed'] for r in v['rows']),total=len(v['rows']),exact=v['exact_to_dual']) for k,v in stresses.items()},
        training_updates_performed=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,task42_authorized=False,speedup_claim=False,all_parameters_finite=all(all(torch.isfinite(v).all() for v in net.parameters()) for net in nets.values()),weights_unchanged=True,
        scope='Independent new native inputs and stress histories, frozen network seeds17/29/43; mixed64 declared prospectively. Implementation parity must be exact, never waived by quality thresholds.')
