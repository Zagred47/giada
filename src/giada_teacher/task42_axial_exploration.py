"""Exploratory axial port with frozen41r, native floors and unchanged gates."""
import hashlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np
from . import task41t_direct_timing as repair
from . import iv_ad1_passive_axial as passive
n = repair.n
t = n.m.t


def spec(root):
    value = json.loads((root/'experiments/task42_preregistration_v1.json').read_text())
    old, cfg, ca, base = n.contract(root)
    assert value['quality'] == old['quality']
    assert value['compartment_gates'] == cfg['gates']
    assert value['passive_gates'] == passive.config(root, 'iv_ad1_passive_axial_preregistration_v3.json')['gates']
    return value, cfg, ca, base


def geometry(count, bifurcation=False):
    parents = [-1]+[max(0,(i-1)//2) if bifurcation else i-1 for i in range(1,count)]
    return dict(id='bifurcation5' if bifurcation else f'chain{count}', parents=parents,
                lengths_um=[100.+10*(i%3) for i in range(count)],
                diameters_um=[1.2-.1*(i%3) for i in range(count)], ra_ohm_cm=130.)


def matrix(geo, coupling):
    size=len(geo['parents']); lap=np.zeros((size,size)); area=np.pi*np.array(geo['lengths_um'])*geo['diameters_um']
    # NEURON child(0)->parent(1): children share a massless distal junction.
    # Eliminate that junction (Kron reduction), not independent series edges:
    # at bifurcations the reduced center matrix also links sibling centers.
    for parent in range(size):
        children=[i for i,p in enumerate(geo['parents']) if p==parent]
        if not children:continue
        indices=[parent,*children]
        resistance=np.array([.01*geo['ra_ohm_cm']*geo['lengths_um'][i]/2/(np.pi*(geo['diameters_um'][i]/2)**2) for i in indices])
        conductance=1/resistance
        reduced=(np.diag(conductance)-np.outer(conductance,conductance)/conductance.sum())*coupling
        lap[np.ix_(indices,indices)]+=reduced
    return area,lap


def design(geo, config):
    count=len(geo['parents']);clock=np.arange(round(config['duration_ms']/.1))*.1
    rows=[];injections=[]
    for coupling in config['coupling']:
        for protocol in config['protocols']:
            injection=np.zeros((len(clock),count))
            if protocol!='rest':
                site=count-1 if protocol=='distal' else 0
                injection[(clock>=2)&(clock<8),site]=.008
                if protocol=='paired':injection[(clock>=18)&(clock<24),count-1]=.008
            rows.append(dict(coupling=coupling,protocol=protocol,initial_mv=-58. if protocol=='paired' else -70.))
            injections.append(injection)
    mult=np.ones((len(rows),count,11))
    for site in range(count):mult[:,site,:]=.9+.1*(site%3)
    return rows,np.stack(injections,axis=1),mult


def native_episode(h, geo, row, inj, mult, cfg, calcium, base, active=True):
    count=len(geo['parents']);sections=[];clamps=[]
    area,_=matrix(geo,row['coupling'])
    for i in range(count):
        sec=h.Section(name=f'giada42_{i}');sec.L=geo['lengths_um'][i];sec.diam=geo['diameters_um'][i];sec.nseg=1
        sec.Ra=geo['ra_ohm_cm']/row['coupling'] if row['coupling'] else geo['ra_ohm_cm'];sec.cm=base['cm_uf_cm2']
        sec.insert('pas');sec.g_pas=base['g_pas_s_cm2'];sec.e_pas=base['e_pas_mv']
        if active:
            for name in (*t.ionic.CHANNELS,'CaDynamics_E2'):sec.insert(name)
            h.ion_style('ca_ion',3,1,0,0,0,sec=sec)
        sections.append(sec)
        if i and row['coupling']:sec.connect(sections[geo['parents'][i]](1),0)
        for start,duration,density in t.t33.t30c._pulses(inj[:,i],.1):
            clamp=h.IClamp(sec(.5));clamp.delay=start;clamp.dur=duration;clamp.amp=density*area[i]*.01;clamps.append(clamp)
    h.CVode().active(0);h.secondorder=0;h.dt=.1;h.celsius=cfg['temperature_c'];h.finitialize(row['initial_mv'])
    initial=t.t30.initial_states(np.full(count,row['initial_mv']),np.full(count,1e-4))
    for i,sec in enumerate(sections):
        seg=sec(.5)
        if active:
            seg.eca=cfg['eca_mv'];seg.ena=55.;seg.ek=-85.;seg.cai=1e-4
            cad=seg.CaDynamics_E2;cad.decay=calcium['decay_ms'];cad.gamma=calcium['gamma'];cad.depth=calcium['depth_um'];cad.minCai=1e-4
            for k,name in enumerate(t.ionic.CHANNELS):
                mech=getattr(seg,name);setattr(mech,'g'+name+'bar',float(t.ionic.GBAR[k]*mult[i,k]))
                for q,state in enumerate(t.ionic.STATE_NAMES[k]):setattr(mech,state,float(initial[i,int(t.ionic.OFFSETS[k])+q]))
    h.fcurrent();v=[];ca=[];g=[]
    for tick in range(len(inj)+1):
        v.append([float(sec(.5).v) for sec in sections])
        if active:
            ca.append([float(sec(.5).cai) for sec in sections])
            g.append([[float(getattr(getattr(sec(.5),name),state)) for k,name in enumerate(t.ionic.CHANNELS) for state in t.ionic.STATE_NAMES[k]] for sec in sections])
        if tick<len(inj):h.fadvance()
    # Native objects stay alive until all recording has finished.
    for sec in reversed(sections):h.delete_section(sec=sec)
    return dict(voltage=np.array(v),calcium=np.array(ca),gates=np.array(g))


def native(root, teacher, out):
    import neuron
    from neuron import h,load_mechanisms
    from . import task32_dynamic_calcium_feedback as cad
    config,cfg,calcium,base=spec(root)
    assert neuron.__version__.split('+')[0]=='8.2.7'
    build,hashes=cad.compile_native(teacher,out,calcium);load_mechanisms(str(build.resolve()))
    numerical=[]
    for geo in config['geometries']:
        rows,inj,mult=design(geo,config);native_rows=[]
        # Native passive floor on each new geometry, before active use.
        for coupling in config['coupling']:
            row=dict(coupling=coupling,initial_mv=-70.,protocol='proximal');pulse=inj[:,1]
            ref=native_episode(h,geo,row,pulse,mult[0],cfg,calcium,base,False)['voltage']
            area,lap=matrix(geo,coupling);cap=base['cm_uf_cm2']*area*1e-5;leak=base['g_pas_s_cm2']*area*.01
            mat=lap+np.diag(cap/.1+leak);v=np.full(len(area),-70.);trace=[v.copy()];balance=0.
            for step in range(len(pulse)):
                old=v.copy();v=np.linalg.solve(mat,cap*old/.1+leak*base['e_pas_mv']+pulse[step]*area*.01);trace.append(v.copy())
                balance=max(balance,float(abs(cap*(v-old)/.1+leak*(v-base['e_pas_mv'])+lap@v-pulse[step]*area*.01).max()))
            error=float(abs(ref-np.array(trace)).max());passed=error<=config['passive_gates']['max_paired_native_discrete_error_mv'] and balance<=config['passive_gates']['max_capacitive_balance_error_na']
            numerical.append(dict(geometry=geo['id'],coupling=coupling,max_native_error_mv=error,balance_na=balance,passed=passed))
        for i,row in enumerate(rows):
            native_rows.append(native_episode(h,geo,row,inj[:,i],mult[i],cfg,calcium,base))
        np.savez_compressed(out/f'native_{geo["id"]}.npz',**{key:np.stack([x[key] for x in native_rows],axis=1) for key in native_rows[0]})
        n.p.save_json(out/'native_progress.json',dict(last_geometry=geo['id'],passive_checks=numerical))
        print('[GIADA42] native '+geo['id']+' complete',flush=True)
    n.p.save_json(out/'native_floor.json',dict(valid=all(x['passed'] for x in numerical),checks=numerical,mechanism_hashes=hashes))


def runner(torch,net,geo,config,cfg,calcium,base,device):
    rows,inj,mult=design(geo,config);cases,count=mult.shape[:2];batch=cases*count
    area=np.array(matrix(geo,0)[0]);lap=np.stack([matrix(geo,r['coupling'])[1] for r in rows])
    factor=torch.tensor(area*.01,device=device,dtype=torch.float64)
    lap=torch.tensor(lap,device=device,dtype=torch.float64)
    eye=torch.eye(count,device=device,dtype=torch.float64)
    cap=.001*base['cm_uf_cm2']/.1
    rev=torch.tensor(t.ionic.REVERSALS,device=device,dtype=torch.float64)
    ionic=n.m.core(torch,net,(cfg,calcium,base),float(area[0]),'voltage_solve64')
    def core(v,ca,state,multipliers,injection,syn):
        _,nc,g,current=ionic(v,ca,state,multipliers,injection,syn)
        conductance,_=t.readout(torch,v.float(),g.float(),multipliers.float());conductance=conductance.double()
        diagonal=(cap+base['g_pas_s_cm2']+conductance.sum(-1)).reshape(3,cases,count)*factor
        rhs=(cap*v+base['g_pas_s_cm2']*base['e_pas_mv']+(conductance*rev).sum(-1)+injection).reshape(3,cases,count)*factor
        matrix=lap[None]+eye*diagonal[...,None]
        nv=torch.linalg.solve(matrix,rhs[...,None])[...,0].reshape(3,batch)
        return nv,nc,g,current
    v=np.repeat([r['initial_mv'] for r in rows],count).astype('float64');ca=np.full(batch,1e-4,dtype='float32')
    initial=t.t30.initial_states(v,ca).astype('float32')
    ex=lambda x,dtype:torch.tensor(x,device=device,dtype=dtype)[None].expand(3,*x.shape).contiguous()
    args=[ex(v,torch.float64),ex(ca,torch.float32),ex(initial,torch.float32),ex(mult.reshape(batch,11),torch.float32),ex(inj[0].reshape(batch),torch.float64),ex(np.zeros((batch,4)),torch.float64)]
    graph_used=False;fallback=None
    if device=='cuda':
        try:
            run=t.Stepper(torch,core,args,True);eager=t.Stepper(torch,core,args,False)
            expected=eager.tick();actual=run.tick()
            assert all(torch.equal(a,b) for a,b in zip(expected,actual)), 'Graph/eager parity failed'
            graph_used=True
        except AssertionError:
            raise
        except Exception as error:
            fallback=repr(error);run=t.Stepper(torch,core,args,False)
    else:run=t.Stepper(torch,core,args,False)
    return run,args,inj,core,dict(graph_used=graph_used,fallback=fallback,cases=cases,nodes=count)


def rollout(torch,net,geo,config,cfg,calcium,base,device,forcing=None):
    run,args,inj,core,info=runner(torch,net,geo,config,cfg,calcium,base,device);count=info['nodes'];cases=info['cases']
    values=[[a.clone()] for a in args[:3]]+[[]];run.reset(args)
    start=time.perf_counter()
    for tick in range(len(inj)):
        if forcing is not None:
            for j,key in enumerate(('voltage','calcium')):run.args[j].copy_(torch.tensor(forcing[key][tick].reshape(-1),device=device,dtype=args[j].dtype)[None].expand(3,-1))
        injection=torch.tensor(inj[tick].reshape(-1),device=device,dtype=torch.float64)[None].expand(3,-1)
        result=run.tick(injection,args[5])
        for dest,value in zip(values,result):dest.append(value.clone())
    if device=='cuda':torch.cuda.synchronize()
    info['wall_seconds']=time.perf_counter()-start
    result={key:torch.stack(value).cpu().numpy().reshape(len(value),3,cases,count,*(() if key in ('voltage','calcium') else (18 if key=='gates' else 11,))) for key,value in zip(('voltage','calcium','gates','current'),values)}
    return result,info


def evaluate(torch,root,source,out,device='cuda'):
    config,cfg,calcium,base=spec(root)
    cache=out.parent/'verified_source42';repair.p.r.verify_source(root,source,cache)
    os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    nets,_=n.load_models(torch,root,device);baseline=nets['dual']
    with np.load(root/config['freeze']['heads_path']) as z:incumbent=repair.materialize(torch,baseline,dict(z))
    assert n.weights_hash(incumbent)==config['freeze']['logical_weights_sha256']
    torch.save(incumbent.state_dict(),out/'frozen41r.pt')
    native_floor=json.loads((out/'native_floor.json').read_text());summaries=[]
    with torch.inference_mode():
        for geo in config['geometries']:
            with np.load(out/f'native_{geo["id"]}.npz') as z:native_data=dict(z)
            formula,execution=rollout(torch,None,geo,config,cfg,calcium,base,device)
            modes={'formula':formula};timings={'formula':execution}
            for mode,forcing in [('imposed',formula),('autonomous',None)]:
                modes['41r_'+mode],timings[mode]=rollout(torch,incumbent,geo,config,cfg,calcium,base,device,forcing)
            rows,inj,mult=design(geo,config)
            for case,row in enumerate(rows):
                errors=formula['voltage'][:,0,case]-native_data['voltage'][:,case]
                cai=formula['calcium'][:,0,case]-native_data['calcium'][:,case]
                rmse=np.sqrt((errors**2).mean(0));crmse=np.sqrt((cai**2).mean(0))
                floor=bool(np.sqrt((errors**2).mean())<=cfg['gates']['max_formula_floor_pooled_voltage_rmse_mv'] and max(rmse)<=cfg['gates']['max_formula_floor_worst_voltage_rmse_mv'] and max(crmse)<=cfg['gates']['max_formula_floor_worst_cai_rmse_mM'])
                for name,prediction in modes.items():
                    for si,seed in enumerate(config['seeds']):
                        ve=prediction['voltage'][:,si,case]-formula['voltage'][:,0,case]
                        ce=prediction['calcium'][:,si,case]-formula['calcium'][:,0,case]
                        pernode=[n.p.evaluate_currents(formula['current'][:,0,case,j],prediction['current'][:,si,case,j],.1,config['quality'],formula['gates'][1:,0,case,j],prediction['gates'][1:,si,case,j]) for j in range(len(geo['parents']))]
                        v_rmse=float(np.sqrt((ve**2).mean()));c_rmse=float(np.sqrt((ce**2).mean()))
                        compartment_passed=bool(v_rmse<=cfg['gates']['max_candidate_pooled_voltage_rmse_mv'] and np.sqrt((ve**2).mean(0)).max()<=cfg['gates']['max_candidate_worst_voltage_rmse_mv'] and np.sqrt((ce**2).mean(0)).max()<=cfg['gates']['max_candidate_worst_cai_rmse_mM'] and np.isfinite(ve).all() and np.isfinite(ce).all() and not np.any((prediction['voltage'][:,si,case]<-140)|(prediction['voltage'][:,si,case]>100)))
                        summaries.append(dict(geometry=geo['id'],case=case,**row,mode=name,seed=seed,native_floor_passed=floor,native_voltage_rmse_mv=rmse.tolist(),native_calcium_rmse_mM=crmse.tolist(),compartment_passed=compartment_passed,
                            current_passed=all(x['passed'] for x in pernode),per_node=pernode,voltage_rmse_mv=float(np.sqrt((ve**2).mean())),max_abs_voltage_error_mv=float(abs(ve).max()),
                            calcium_rmse_mM=float(np.sqrt((ce**2).mean())),finite=bool(np.isfinite(ve).all() and np.isfinite(ce).all()),judgment_eligible=bool(floor and native_floor['valid'])))
            np.savez_compressed(out/f'traces_{geo["id"]}.npz',**{name+'_'+key:value for name,data in modes.items() for key,value in data.items()})
            n.p.save_json(out/'results.json',summaries);n.p.save_json(out/f'execution_{geo["id"]}.json',timings)
            print('[GIADA42] paired evaluation '+geo['id']+' complete',flush=True)
    assert n.weights_hash(incumbent)==config['freeze']['logical_weights_sha256']
    torch.save(incumbent.state_dict(),out/'frozen41r.pt')
    report=dict(valid=True,complete=True,task='42',exploratory=True,training_updates=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,
                all_parameters_finite=all(bool(torch.isfinite(x).all()) for x in incumbent.parameters()),candidate_weights_unchanged=True,
                task41_original_gate_passed=False,task43_automatically_authorized=False,speedup_claim=False,
                passive_native_floor_passed=native_floor['valid'],native_active_floor_passed=all(x['native_floor_passed'] for x in summaries),
                models={name:dict(current_passed=sum(x['current_passed'] for x in summaries if x['mode']==name),joint_passed=sum(x['current_passed'] and x['compartment_passed'] for x in summaries if x['mode']==name),total=sum(x['mode']==name for x in summaries)) for name in modes},
                decision='EXPLORATORY_AXIAL_PORT_COMPLETE_REQUIRES_AUDIT',limits=config['limits'])
    n.p.save_json(out/'final_report.json',report)
    return report
