"""FrozenP spatial interventions; same ionic abstraction and coupled physics."""
import copy,hashlib,io,json,os,time,zipfile
from contextlib import contextmanager
import numpy as np
from . import task45e_anchored_repair as e
d=e.d;c=e.c;n=e.n;y=c.y;x=y.x;t=y.t
ARMS=('none','C1','P','R41')
def spec(root):
    p=json.loads((root/'experiments/task46_preregistration_v1.json').read_text());old,cfg,ca,base=n.contract(root)
    assert p['quality']==old['quality'] and p['compartment_gates']==cfg['gates']
    assert tuple(p['arms'])==ARMS and p['dt_ms']==.1 and p['training_updates']==0 and not p['selection_performed']
    assert not p['thresholds_changed'] and not p['abstraction_changed'];return p,cfg,ca,base
def design(geo,p):
    count=len(geo['parents']);clock=np.arange(round(p['duration_ms']/.1))*.1;rows=[];injections=[];multipliers=[]
    for coupling in p['coupling']:
        for site_label,site in (('proximal',0),('distal',count-1)):
            for form in ('burst','hyper_rebound'):
                for density in ('uniform','graded'):
                    blocks=('none','sodium','Ih') if form=='burst' and density=='graded' else ('none',)
                    for block in blocks:
                        inj=np.zeros((len(clock),count))
                        def pulse(start,end,node,amplitude):inj[(clock>=start)&(clock<end),node]+=amplitude
                        for j in range(count):pulse(1+2*(j%4),5+2*(j%4),j,.004*(1+j%3))
                        if form=='burst':
                            for j in range(8):pulse(80+j*14,82+j*14,site,.018)
                        else:pulse(40,200,site,-.003);pulse(300,315,site,.015)
                        mult=np.ones((count,11))
                        if density=='graded':mult=np.array([[(.75,1.,1.25)[(j+2*k)%3] for k in range(11)] for j in range(count)])
                        if block=='sodium':mult[:,[2,3,4]]=0.
                        elif block=='Ih':mult[:,5]=0.
                        label=f'46_{form}_{site_label}_{density}_{block}'
                        rows.append(dict(coupling=coupling,protocol=label,initial_mv=-65.,form=form,stimulus_site=site,site_label=site_label,density=density,block=block,stimulus_onset_ms=80. if form=='burst' else 40.))
                        injections.append(inj);multipliers.append(mult)
    return rows,np.stack(injections,1),np.stack(multipliers)
def masks(count,arms):return np.array([np.full((count,11),arm!='none',dtype=bool) for arm in arms])
@contextmanager
def configured(root):
    before=y.spec,y.ARMS,y.previous.design,y.channel_masks
    y.spec=spec;y.ARMS=ARMS;y.previous.design=design;y.channel_masks=masks
    try:yield
    finally:y.spec,y.ARMS,y.previous.design,y.channel_masks=before
def native(root,teacher,out):
    with configured(root):y.native(root,teacher,out)
def load_frozen(torch,root,source,checkpoint45c,checkpoint45d,out,device):
    p,_,_,_=spec(root);plans=p['freeze']['inputs']
    cache=out.parent/'verified_source46';x.repair.p.r.verify_source(root,source,cache);os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    old_models,parent=n.load_models(torch,root,device)
    with np.load(root/p['reference41r']['heads_path']) as z:template=x.repair.materialize(torch,old_models['dual'],dict(z))
    states=[]
    for path,plan in zip((checkpoint45c,checkpoint45d),plans):
        assert c.sha(path)==plan['sha256']
        with zipfile.ZipFile(path) as z:
            raw=z.read(plan['member']);assert hashlib.sha256(raw).hexdigest()==plan['member_sha256']
            states.append(torch.load(io.BytesIO(raw),map_location=device,weights_only=True))
    anchor=copy.deepcopy(template);anchor.load_state_dict(states[0]);candidate=copy.deepcopy(anchor)
    for key in ('parent.a','parent.b'):assert torch.equal(states[0][key],states[1][key])
    with torch.no_grad():candidate.parent.c[:,:,26].copy_(states[1]['parent.c'][:,:,26]);candidate.parent.d[:,26].copy_(states[1]['parent.d'][:,26])
    assert e.assert_preserved(torch,candidate,anchor,9,False)['valid']
    assert n.weights_hash(candidate)==p['freeze']['logical_weights_sha256']
    models=dict(C1=anchor,P=candidate,R41=template)
    for net in models.values():
        net.eval()
        for parameter in net.parameters():parameter.requires_grad_(False);assert bool(torch.isfinite(parameter).all())
    n.p.save_json(out/'candidate_freeze.json',dict(valid=True,weights_sha256={k:n.weights_hash(v) for k,v in models.items()},recipe=p['freeze'],training_updates=0,selection_performed=False,new_payload_upload=False))
    for name,net in models.items():torch.save(net.state_dict(),out/(name+'.pt'))
    return models,parent,cache
def propagation(pred,ref,geo,rows,mode):
    result=[]
    for case,meta in enumerate(rows):
        start=round(meta['stimulus_onset_ms']/.1);site=meta['stimulus_site']
        if start>=len(ref['voltage']):
            result.append(dict(geometry=geo['id'],case=case,mode=mode,status='not_evaluable',reason='Stimulus has not started within this prefix',diagnostic_only=True));continue
        for si,seed in enumerate((17,29,43)):
            for ai,arm in enumerate(ARMS[1:],1):
                target=ref['voltage'][:,0,0,case];got=pred['voltage'][:,si,ai,case]
                def describe(trace):
                    value=trace[start:];amplitude=np.max(value-trace[start],0);cross=[((np.flatnonzero((value[:-1,j]<0)&(value[1:,j]>=0))+start+1)*.1).tolist() for j in range(value.shape[1])]
                    delay=[q[0]-cross[site][0] if q and cross[site] else None for q in cross]
                    return dict(positive_excursion_mv=amplitude.tolist(),attenuation_to_stimulus_site=(amplitude/max(float(amplitude[site]),1e-7)).tolist(),upcrossings_ms=cross,first_crossing_delay_ms=delay)
                a,b=describe(target),describe(got)
                result.append(dict(geometry=geo['id'],case=case,seed=seed,arm=arm,mode=mode,reference=a,candidate=b,counts_match=[len(u)==len(v) for u,v in zip(a['upcrossings_ms'],b['upcrossings_ms'])],diagnostic_only=True))
    return result
def intervention_pairs(rows):
    indexed={(r['geometry'],r['coupling'],r['site_label'],r['form'],r['density'],r['block'],r['seed'],r['arm'],r['mode'],r['horizon_ms']):r for r in rows};result=[]
    for key,r in indexed.items():
        if r['block']=='none':continue
        control=indexed[(*key[:5],'none',*key[6:])]
        result.append(dict(geometry=r['geometry'],case=r['case'],seed=r['seed'],arm=r['arm'],mode=r['mode'],horizon_ms=r['horizon_ms'],block=r['block'],unblocked_voltage_rmse_mv=control['voltage_rmse_mv'],blocked_voltage_rmse_mv=r['voltage_rmse_mv'],blocked_new_failure=bool(control['current_passed'] and control['compartment_passed'] and not(r['current_passed'] and r['compartment_passed'])),diagnostic_only=True))
    return result
def component_benchmark(torch,models,p,cfg,ca,base,out,device):
    results=[]
    for geo in p['geometries']:
        tests={};parity=[]
        for label,net in [('formula',None),('C1',models['C1']),('P',models['P'])]:
            core,args,inj,info=y.setup(torch,net,geo,p,cfg,ca,base,device,('all',));nv,nc,g,current=core(*args)
            count=info['nodes'];cases=info['cases'];area,_=x.matrix(geo,0);factor=torch.tensor(area*.01,device=device,dtype=torch.float64)
            lap=torch.tensor(np.stack([x.matrix(geo,r['coupling'])[1] for r in design(geo,p)[0]]),device=device,dtype=torch.float64)
            conductance=t.readout(torch,args[0].float(),g,args[3])[0].double();cap=.001*base['cm_uf_cm2']/.1
            diagonal=(cap+base['g_pas_s_cm2']+conductance.sum(-1)).reshape(3,cases,count)*factor
            rev=torch.tensor(t.ionic.REVERSALS,device=device,dtype=torch.float64)
            rhs=(cap*args[0]+base['g_pas_s_cm2']*base['e_pas_mv']+(conductance*rev).sum(-1)+args[4]).reshape(3,cases,count)*factor
            mat=lap[None]+torch.eye(count,device=device,dtype=torch.float64)*diagonal[...,None]
            solve=lambda mat=mat,rhs=rhs:torch.linalg.solve(mat,rhs[...,None])[...,0].reshape(3,-1)
            want=solve();exact=torch.equal(want,nv);assert exact,'Component solve does not reproduce actual full-step voltage'
            parity.append(dict(model=label,axial_component_bitwise_equal=exact))
            mask=torch.full((*args[0].shape,11),net is not None,device=device,dtype=torch.bool);ionic=y.local_core(torch,net,mask,cfg,ca)
            cpu=[a.detach().cpu().clone() for a in args[:4]]
            tests[label]=dict(full_step=lambda core=core,args=args:core(*args),ionic=lambda fn=ionic,args=args:fn(args[0],args[1],args[2],args[3]),axial_solve=solve,host_to_device=lambda cpu=cpu:[a.to(device,copy=True) for a in cpu],device_to_host=lambda args=args:[a.detach().cpu() for a in args[:4]])
            assert all(bool(torch.isfinite(value).all()) for value in (nv,nc,g,current))
        for scopes in tests.values():
            for fn in scopes.values():
                for _ in range(3):fn()
        samples={name:{scope:[] for scope in scopes} for name,scopes in tests.items()}
        for repeat in range(p['cost']['repetitions']):
            for name in list(tests) if repeat%2==0 else reversed(tests):
                for scope,fn in tests[name].items():
                    if device=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter()
                    for _ in range(p['cost']['ticks']):fn()
                    if device=='cuda':torch.cuda.synchronize()
                    samples[name][scope].append(1000*(time.perf_counter()-start))
        results.append(dict(geometry=geo['id'],actual_sites_per_model=3*len(design(geo,p)[0])*len(geo['parents']),ticks=p['cost']['ticks'],samples_wall_ms=samples,parity=parity,non_additive=True,scope='Fixed common state; component costs do not sum to autonomous runtime. Full step is original eager coupled implementation; ionic scope retains its canonical masked-branch overhead. H2D/D2H are blocking copies, not resident rollout traffic.'))
        n.p.save_json(out/'coupled_component_cost.json',results);print('[GIADA'+p['task']+'] component cost '+geo['id']+' saved',flush=True)
    return results
def evaluate(torch,root,source,ck45c,ck45d,out,device='cuda'):
    p,cfg,ca,base=spec(root);models,parent,cache=load_frozen(torch,root,source,ck45c,ck45d,out,device)
    before={k:n.weights_hash(v) for k,v in models.items()};packed=d.remap(d.adapter,ARMS=ARMS)(torch,models['C1'],models)
    n.p.save_json(out/'packed_execution_preflight.json',packed.execution_preflight)
    floor=json.loads((out/'native_floor.json').read_text());rows=[];parity=[];execution={}
    with torch.inference_mode(),configured(root):
        for geo in p['geometries']:
            parity.extend(dict(geometry=geo['id'],**r) for r in y.endpoint_preflight(torch,root,models['P'],geo,p,cfg,ca,base,device))
            count=len(design(geo,p)[0])*len(geo['parents']);parity.extend(dict(geometry=geo['id'],**r) for r in d.remap(c.packed_parity,ARMS=ARMS)(torch,packed,models,count,p,device))
            with np.load(out/f'native_{geo["id"]}.npz') as z:teacher=dict(z)
            repeated,info=y.rollout(torch,None,geo,p,cfg,ca,base,device,ARMS);ref={k:v[:,:,:1].copy() for k,v in repeated.items()};del repeated
            execution[geo['id']+'/formula']=info;physical=design(geo,p)[0]
            for mode,forcing in [('imposed',ref),('autonomous',None)]:
                pred,info=y.rollout(torch,packed,geo,p,cfg,ca,base,device,ARMS,forcing)
                assert all(float(abs(pred[k][:,:,0]-ref[k][:,:,0]).max())<=y.previous.permutation_limit(p,k) for k in ref),'Formula null mismatch'
                for h in p['prefix_horizons_ms']:
                    ticks=round(h/.1);measured=y.score(c.b.a.crop(pred,ticks),c.b.a.crop(ref,ticks),{k:v[:ticks+1] for k,v in teacher.items()},geo,physical,ARMS,mode,p,cfg,floor['valid'])
                    controls={(q['case'],q['seed']):q for q in measured if q['arm']=='none'}
                    for q in measured:
                        null=controls[(q['case'],q['seed'])];q.update(horizon_ms=h,reference_valid=bool(q['judgment_eligible'] and null['current_passed'] and null['compartment_passed']))
                    rows.extend(measured)
                np.savez_compressed(out/f'traces_{geo["id"]}_{mode}.npz',**pred)
                n.p.save_json(out/f'propagation_{geo["id"]}_{mode}.json',propagation(pred,ref,geo,physical,mode));del pred
                execution[geo['id']+'/'+mode]=info;n.p.save_json(out/'execution.json',execution);n.p.save_json(out/'results.json',rows)
                print('[GIADA'+p['task']+'] '+geo['id']+'/'+mode+' all4arms and prefixes saved',flush=True)
            np.savez_compressed(out/f'reference_{geo["id"]}.npz',**ref)
        with np.load(cache/'native_shadow_traces.npz') as z:cost_data=dict(z)
        physics=(n.old_native.x.calibration_config(n.contract(root)[0],cfg),ca,base)
        cost=d.ORIGINAL_BENCHMARK(torch,dict(dual=models['C1'],C1=models['C1'],P=models['P']),parent,cost_data,physics,p,out,device)
        components=component_benchmark(torch,models,p,cfg,ca,base,out,device)
    assert before=={k:n.weights_hash(v) for k,v in models.items()}
    n.p.save_json(out/'endpoint_preflight.json',parity);n.p.save_json(out/'intervention_pairs.json',intervention_pairs(rows))
    summary={str(h):{arm:{mode:dict(total=len(pool),eligible=sum(q['reference_valid'] for q in pool),passed=sum(q['reference_valid'] and q['current_passed'] and q['compartment_passed'] for q in pool)) for mode in ('imposed','autonomous') for pool in [[q for q in rows if q['horizon_ms']==h and q['arm']==arm and q['mode']==mode]]} for arm in ARMS} for h in p['prefix_horizons_ms']}
    report=dict(valid=True,complete=True,task='46',exploratory=True,training_updates=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,candidate_weights_unchanged=True,all_parameters_finite=all(bool(torch.isfinite(v).all()) for net in models.values() for v in net.parameters()),all_traces_finite=all(q['finite'] for q in rows),passive_native_floor_passed=floor['valid'],active_native_floor_passed=all(q['native_floor_passed'] for q in rows),null_substitution_control_passed=True,endpoint_parity_passed=all(q['passed'] for q in parity),summary=summary,cost_rows=len(cost),coupled_component_geometries=len(components),decision='EXPLORATORY_SPATIAL_INTERVENTIONS_COMPLETE_REQUIRES_AUDIT',task45e_passed=False,task47_authorized=False,speedup_claim=False,release_authorized=False,limits=p['limits'])
    n.p.save_json(out/'final_report.json',report);return report
