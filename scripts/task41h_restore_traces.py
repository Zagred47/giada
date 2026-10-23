"""Rebuild only lost raw arrays from frozen weights; cached metrics stay fixed."""
import gc,json
import numpy as np

def restore(torch,h,ev,root,output,controls,trained,spec,cfg,calcium,base,device):
    t,p,c=h.t,h.p,h.c;models={**controls,**trained};allmodels={**models,'LUT_control':h.f.x.lut_model(torch,device,.1)}
    # The complete first-panel traces already survive in the original backup.
    data,meta,_=ev.ee.load41c(root,spec);folder=output/'opened41c';refs={};created=[];checks=[]
    cached=json.loads((folder/'causal_results.json').read_text())
    nominee=json.loads((output/'selection_freeze.json').read_text())['nominee']
    for mode in ('imposed','autonomous'):
        imposed={k:data[k].T for k in ('voltage','calcium')} if mode=='imposed' else None
        ref={k:v[:,0] for k,v in t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,device,imposed,True).items()};refs[mode]=ref
        for name,model in allmodels.items():
            path=folder/f'traces_{name}_{mode}.npz'
            if path.exists():continue
            channels=tuple(k for k in range(11) if k!=9) if name=='LUT_control' else t.FAMILIES['all']
            pred=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,channels),data,device,imposed,True)
            if mode=='autonomous' and name in (nominee,'G_nominee'):
                for row in cached[name+'/'+mode]['interventions']:
                    if row['case'] not in (108,122):continue
                    case=row['case'];seed=spec['seeds'].index(row['seed']);origin=int(meta[case]['source_index'])
                    actual=p.evaluate_intervention(ref['current'][:,origin],ref['current'][:,case],pred['current'][:,seed,origin],pred['current'][:,seed,case],.1,spec['quality'])
                    assert actual['passed']==row['metrics']['passed'],('Backend changed critical response',name,case,row['seed'])
                    assert np.allclose([x['nrmse'] for x in actual['channels']],[x['nrmse'] for x in row['metrics']['channels']],rtol=1e-5,atol=1e-12)
                    checks.append(dict(model=name,case=case,seed=row['seed'],critical_response_reproduced=True))
            np.savez_compressed(path,**pred);created.append(path.relative_to(output).as_posix())
            print(f'[GIADA Task41h] restored lost trace {name}/{mode}',flush=True)
            del pred;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(folder/'reference_traces.npz',**{m+'__'+k:v for m,r in refs.items() for k,v in r.items()});created.append('opened41c/reference_traces.npz')
    # Historical evaluator originally saved only autonomous candidate traces.
    old,hcfg,hca,hbase=t.contract(root);hd,hm=t.design(root,old);golden=t.numpy_reference(hd,hcfg,hca,hbase)
    refs={};traces={};hist=output/'historical_regression'
    for mode in ('imposed','autonomous'):
        imposed={k:golden[k].copy() for k in ('voltage','calcium')}
        if mode=='imposed':
            for k in imposed:imposed[k][:,:32]=hd[k][:32].T
        rr=t.rollout(torch,t.make_core(torch,hcfg,hca,hbase,hd['area_um2']),hd,device,imposed if mode=='imposed' else None,True)
        refs[mode]={k:v[:,0] for k,v in rr.items()}
        if mode=='autonomous':
            for name,model in dict(B0=t.load_models(torch,root,device)['J1'],**models).items():
                pred=t.rollout(torch,t.make_core(torch,hcfg,hca,hbase,hd['area_um2'],model,t.FAMILIES['all']),hd,device,None,True)
                traces.update({name+'__'+k:v for k,v in pred.items()})
                print('[GIADA Task41h] restored historical trace '+name,flush=True)
                gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(hist/'autonomous_traces.npz',**traces)
    np.savez_compressed(hist/'reference_traces.npz',**{m+'__'+k:v for m,r in refs.items() for k,v in r.items()})
    created+=['historical_regression/autonomous_traces.npz','historical_regression/reference_traces.npz']
    report=dict(valid=True,training_updates_performed=0,selection_recomputed=False,cached_metrics_unchanged=True,
        method='Deterministic frozen-weight trace regeneration; not claimed byte-identical to lost NPZ containers',
        critical_response_checks=checks,created=created)
    p.save_json(output/'trace_restoration.json',report)
    return report
