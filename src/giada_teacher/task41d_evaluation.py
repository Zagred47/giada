"""After-freeze regression, composition and cost; never selects on opened tests."""
import hashlib,io,json,os,zipfile
from pathlib import Path
import numpy as np
from . import task41d_precision_response as d
c,p,t,x=d.c,d.p,d.t,d.x
e=d.prior.evaluation

def compose(torch,ca,nap,parent):
    class Composition:
        def __init__(self):self.a,self.b,self.parent=ca,nap,parent
        def __getattr__(self,k):return getattr(self.parent,k)
        def raw(self,v,calcium):
            pu,pz=self.parent.raw(v,calcium);au,_=self.a.raw(v,calcium);bu,_=self.b.raw(v,calcium)
            return torch.cat((au[...,:1],pu[...,1:8],bu[...,8:9],pu[...,9:]),-1),pz
        def __call__(self,v,calcium,g):
            u,z=self.raw(v,calcium);return g+torch.sigmoid(z)*(torch.sigmoid(u)-g)
    return Composition()

def suite(torch,output,models,data,meta,spec,cfg,calcium,base,device):
    output.mkdir(exist_ok=True);reports={};refs={};q=spec['quality']
    for mode in ('imposed','autonomous'):
        imposed={k:data[k].T for k in ('voltage','calcium')} if mode=='imposed' else None
        rr=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2'])),data,device,imposed,True);ref={k:v[:,0] for k,v in rr.items()};refs[mode]=ref
        target=x.numpy_imposed(data,cfg,calcium,base) if mode=='imposed' else t.numpy_reference(data,cfg,calcium,base)
        floor=[p.evaluate_currents(target['current'][:,k],ref['current'][:,k],.1,q,target['gates'][1:,k],ref['gates'][1:,k]) for k in range(len(meta))]
        inter_floor=[p.evaluate_intervention(target['current'][:,r['source_index']],target['current'][:,k],ref['current'][:,r['source_index']],ref['current'][:,k],.1,q) for k,r in enumerate(meta) if r['intervention']!='none']
        assert all(r['passed'] for r in floor+inter_floor),'Formula numerical floor failed'
        p.save_json(output/f'floor_{mode}.json',dict(cases=floor,interventions=inter_floor))
        if mode=='imposed':
            ng,nc=x.native_phase_targets(data);phase=[p.evaluate_currents(nc[:,k],ref['current'][:,k],.1,q,ng[:,k],ref['gates'][1:,k]) for k in range(len(meta))]
            assert all(r['passed'] for r in phase);p.save_json(output/'native_phase_floor.json',phase)
        for name,model in models.items():
            channels=tuple(k for k in range(11) if k!=9) if name=='LUT_control' else t.FAMILIES['all']
            pred=t.rollout(torch,t.make_core(torch,cfg,calcium,base,float(data['area_um2']),model,channels),data,device,imposed,True)
            report=t.metrics(pred,{**ref,**{k:data[k].T for k in ('voltage','calcium')}},data,meta,q,cfg['gates'],mode)
            report['interventions']=[dict(case=k,seed=seed,metrics=p.evaluate_intervention(ref['current'][:,r['source_index']],ref['current'][:,k],pred['current'][:,s,r['source_index']],pred['current'][:,s,k],.1,q)) for s,seed in enumerate(spec['seeds']) for k,r in enumerate(meta) if r['intervention']!='none']
            reports[name+'/'+mode]=report;p.save_json(output/'causal_results.json',reports)
            np.savez_compressed(output/f'traces_{name}_{mode}.npz',**pred)
            print(f'[GIADA Task41d] evaluation {output.name} {name}/{mode} {report["passed_cases"]}/{report["total"]} responses={sum(r["metrics"]["passed"] for r in report["interventions"])}',flush=True)
    np.savez_compressed(output/'reference_traces.npz',**{m+'__'+k:v for m,r in refs.items() for k,v in r.items()})
    return reports

def stress(torch,models,data,spec,device,metadata):
    old=x.stress_design;x.stress_design=lambda unused:metadata
    try:return e.stress(torch,models,data,{**spec,'quality':spec['quality']},device)
    finally:x.stress_design=old

def evaluate(torch,root,output,trained,spec,device):
    _,cfg,calcium,base=d.contract(root);parents=d.prior.load_models(torch,root,device)
    freeze=json.loads((output/'selection_freeze.json').read_text());payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
    assert p.digest(output/'frozen_checkpoint.pt')==freeze['checkpoint_sha256'] and not freeze['opened_regressions_accessed']
    ca,nap=(trained[freeze['nominees'][k]] for k in ('Ca','Nap'))
    composition=compose(torch,ca,nap,parents['P110'])
    models={**parents,**trained,'combined':composition};allmodels={**models,'LUT_control':x.lut_model(torch,device,.1)}
    p.save_json(output/'composition_contract.json',dict(ca=freeze['nominees']['Ca'],nap=freeze['nominees']['Nap'],selection_before_opened_regressions=True,controls=['P110',freeze['nominees']['Ca'],freeze['nominees']['Nap'],'combined'],additional_model_selection=False))
    oldspec=json.loads((root/c.CONFIG).read_text());data,meta,st,expected=e.load_opened(root,oldspec)
    oldmeta=x.stress_design(json.loads((root/x.CONFIG).read_text()))
    opened41=suite(torch,output/'opened41',allmodels,data,meta,spec,cfg,calcium,base,device)
    s41=stress(torch,models,st,spec,device,oldmeta);p.save_json(output/'stress41.json',s41)
    path=Path(os.environ.get('GIADA_TASK41C_INPUTS','/content/task41d_opened41c.zip'));assert p.digest(path)==spec['opened41c_sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,sha in spec['opened41c_members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
        dd=dict(np.load(io.BytesIO(z.read('native_shadow_traces.npz'))));mm=json.loads(z.read('cases.json'));ss=dict(np.load(io.BytesIO(z.read('fresh_stress_traces.npz'))))
    opened41c=suite(torch,output/'opened41c',allmodels,dd,mm,spec,cfg,calcium,base,device)
    s41c=stress(torch,models,ss,spec,device,d.prior.stress_design(json.loads((root/d.prior.CONFIG).read_text())));p.save_json(output/'stress41c.json',s41c)
    # GPU validation of the newly used development generator after checkpoint freeze.
    dev=d.development(spec['new_seeds']['development']);devrows={}
    target=p.readout(dev['voltage'],dev['target_gates'],dev['multipliers'],c.ionic.REVERSALS)[1]
    for name,model in models.items():
        gates=c.j.i.old.Scanner(torch,[model],dev['voltage'].shape[1],32,device).run(dev['voltage'],dev['calcium'],dev['initial'].astype('float32'))[0]
        devrows[name]=[dict(seed=seed,case=k,metrics=p.evaluate_currents(target[:,k],p.readout(dev['voltage'],gates[:,s],dev['multipliers'],c.ionic.REVERSALS)[1][:,k],.1,spec['quality'],dev['target_gates'][:,k],gates[:,s,k])) for s,seed in enumerate(spec['seeds']) for k in range(dev['voltage'].shape[1])]
        p.save_json(output/'development_gpu.json',devrows)
    hist=output/'historical_regression';hist.mkdir()
    b0=t.load_models(torch,root,device)['J1'];oldreports=c.b.causal_evaluate(torch,root,hist,dict(B0=b0,**models),spec,device)
    oldcfg=json.loads((root/c.j.i.old.CONFIG).read_text());evaluation={**json.loads((root/c.j.CONFIG).read_text()),'architectures':list(models)}
    for key in ('baseline_parent','baseline_checkpoint'):evaluation[key]=oldcfg[key]
    # Bound CUDA graph memory by evaluating three models per resident scanner.
    import gc
    recurrence=[];items=list(models.items())
    for start in range(0,len(items),3):
        chunk=items[start:start+3];out=hist/f'recurrence_group_{start//3}';out.mkdir()
        c.j.i.old.evaluate(root,out,{**evaluation,'architectures':[k for k,v in chunk]},[v for k,v in chunk],torch,device,json.loads((root/p.CONFIG).read_text()))
        recurrence+=json.loads((out/'recurrence_evaluation.json').read_text());p.save_json(hist/'recurrence_evaluation.json',recurrence)
        gc.collect();torch.cuda.empty_cache()
        print(f'[GIADA Task41d] checkpoint historical recurrence group {start//3} complete',flush=True)
    cost=e.benchmark(torch,root,output,models,dd,spec,device)
    # F32/F64 floors on short tiny causal responses are a separate numerical probe.
    fb,_=d.feedback_data(torch,spec,cfg,calcium,base,device,32);points={k:v.double() for k,v in fb.items() if k not in ('target','scale')};floor=[]
    for dtype in (torch.float32,torch.float64):
        core=t.make_core(torch,cfg,calcium,base,100*np.pi);v,ca,g=(points[k].to(dtype) for k in ('v','ca','g'));curr=[]
        for _ in range(spec['feedback_ticks']):v,ca,g,ii=core(v,ca,g,points['mult'].to(dtype),points['inj'].to(dtype),points['syn'].to(dtype));curr.append(ii.cpu().numpy())
        floor.append(np.stack(curr))
    fp=[p.evaluate_intervention(floor[1][:,k,0],floor[1][:,k,1],floor[0][:,k,0],floor[0][:,k,1],.1,spec['quality']) for k in range(32)]
    p.save_json(output/'short_response_precision_floor.json',dict(rows=fp,passed=all(r['passed'] for r in fp),scope='Independent FIT-only diagnostic;fails are reported without rewriting native regression gates.'))
    summaries={};passing=[]
    for name in models:
        summary=dict(development=dict(passed=sum(r['metrics']['passed'] for r in devrows[name]),total=len(devrows[name])),panels={},stress={},historical={},recurrence={})
        okay=all(r['metrics']['passed'] for r in devrows[name])
        for label,reports in [('41',opened41),('41c',opened41c)]:
            summary['panels'][label]={}
            for mode in ('imposed','autonomous'):
                rr=reports[name+'/'+mode];passed=sum(r['metrics']['passed'] for r in rr['interventions'])
                summary['panels'][label][mode]=dict(trajectory=rr['passed_cases'],total=rr['total'],responses=passed,response_total=len(rr['interventions']))
                okay &= rr['passed'] and passed==len(rr['interventions'])
        for label,sr in [('41',s41),('41c',s41c)]:
            rows=[r for r in sr['rows'] if r['model']==name];summary['stress'][label]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows));okay &= all(r['metrics']['passed'] for r in rows)
        for mode in ('imposed','autonomous'):
            rr=oldreports[name+'/'+mode];summary['historical'][mode]=dict(trajectory=rr['passed_cases'],total=rr['total'],responses=sum(r['metrics']['passed'] for r in rr['interventions']))
            okay &= rr['passed'] and all(r['metrics']['passed'] for r in rr['interventions'])
        for role in ('development','stress_diagnostic'):
            rows=[r for r in recurrence if r['model']=='joint_w'+name and r['role']==role];assert len(rows)==72
            summary['recurrence'][role]=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows));okay &= all(r['metrics']['passed'] for r in rows)
        summaries[name]=summary
        if okay and name not in parents:passing.append(name)
    contrasts=[]
    for panel,axes in [('Ca',('support','basis')),('Nap',('shape','paired','feedback'))]:
        for axis in axes:
            for ai,a in enumerate(d.arms()):
                if a['panel']!=panel:continue
                for bi,b in enumerate(d.arms()):
                    if b['panel']!=panel:continue
                    other=[k for k in axes if k!=axis]
                    order=([32,64] if axis=='basis' else (['uniform','sensitive'] if axis=='support' else (['linear','smooth'] if axis=='shape' else [False,True])))
                    if a[axis]==order[0] and b[axis]==order[1] and all(a[k]==b[k] for k in other):
                        ratios=[freeze['ranks'][bi*3+s][1]/max(freeze['ranks'][ai*3+s][1],1e-12) for s in range(3)]
                        count_ok=all(freeze['ranks'][bi*3+s][0]<=freeze['ranks'][ai*3+s][0] for s in range(3))
                        contrasts.append(dict(panel=panel,axis=axis,control=a['name'],treatment=b['name'],worst_severity_ratios=ratios,registered_contrast_passed=bool(all(r<1 for r in ratios) and np.median(ratios)<=.8 and count_ok)))
    p.save_json(output/'paired_contrasts.json',contrasts)
    result=dict(valid=True,task='41d',decision='TARGETED_PRECISION_REPAIR_REQUIRES_AUDIT' if passing else 'TARGETED_PRECISION_DIAGNOSTIC_NO_GO',passing_arms=passing,summaries=summaries,nominees=freeze['nominees'],thresholds_changed=False,independent_confirmation=False,task42_authorized=False,speedup_claim=False,short_response_precision_floor_passed=all(r['passed'] for r in fp),limits=spec['limits'])
    p.save_json(output/'final_report.json',result);return result
