"""Missing45d evaluations ONLY. No optimizer, fit, nomination or native generation."""
import copy,hashlib,io,json,os,zipfile
from pathlib import Path
import numpy as np
from . import task45d_learnability_execution as d
c=d.c;n=d.n
def load_plan(root):return json.loads((root/'experiments/task45d_evaluation_recovery_v1.json').read_text())
def summarize(rows,p):
    return {panel:{str(h):{arm:{mode:dict(total=len(pool),eligible=sum(r['reference_valid'] for r in pool),passed=sum(r['reference_valid'] and r['current_passed'] and r['compartment_passed'] for r in pool),regressions=sum(r['reference_valid'] and r['regression_vs_f011'] for r in pool)) for mode in ('imposed','autonomous') for pool in [[r for r in rows if r['panel']==panel and r['horizon_ms']==h and r['arm']==arm and r['mode']==mode]]} for arm in d.ARMS[1:]} for h in sorted({r['horizon_ms'] for r in rows if r['panel']==panel})} for panel in p['panels']}

def recover(torch,root,source,native,archive,control,out,device='cuda'):
    p,cfg,ca,base=d.spec(root);plan=load_plan(root);assert c.sha(archive)==plan['sha256']
    assert c.sha(control)==json.loads((root/'experiments/task45c_cost_transport_v2.json').read_text())['sha256']
    cache=out.parent/'verified_source45c';c.x.repair.p.r.verify_source(root,source,cache);os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    fixtures=out.parent/'verified_native45c';c.verify_fixtures(root,native,fixtures)
    old_models,parent=n.load_models(torch,root,device)
    with np.load(root/p['reference41r']['heads_path']) as z:template=c.x.repair.materialize(torch,old_models['dual'],dict(z))
    rows=[];models={}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,expected in plan['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
        freeze_bytes=z.read('nomination_freeze.json');freeze=json.loads(freeze_bytes)
        for arm in d.ARMS[1:]:
            model=copy.deepcopy(template);model.load_state_dict(torch.load(io.BytesIO(z.read(arm+'.pt')),map_location=device,weights_only=True));model.eval()
            for parameter in model.parameters():parameter.requires_grad_(False);assert bool(torch.isfinite(parameter).all())
            if arm in freeze['weights_sha256']:assert n.weights_hash(model)==freeze['weights_sha256'][arm]
            models[arm]=model
        for name in ('nomination_freeze.json','calibration_freeze.json','GPU_training_transition_preflight.json'):(out/name).write_bytes(z.read(name))
        previous_execution=json.loads(z.read('execution.json'));previous_strategy=json.loads(z.read('packed_execution_preflight.json'))
        for panel in p['panels']:
            path=panel+'/results.json'
            if path in z.namelist():rows.extend(json.loads(z.read(path)))
    assert n.weights_hash(models['C0'])==p['baseline']['logical_weights_sha256']
    before={name:n.weights_hash(net) for name,net in models.items()};adapter=d.adapter(torch,models['C0'],models)
    assert adapter.execution_preflight['vectorized']==previous_strategy['vectorized'],'Do not mix a different packed execution policy with saved blocks'
    reused=set(plan['completed_blocks']);finished=set(reused);execution=dict(previous_execution);checks=[]
    n.p.save_json(out/'recovery_progress.json',dict(completed_blocks=sorted(finished),training_performed=False,selection_recomputed=False))
    with torch.inference_mode():
        for panel in c.b.panel_configs(root,p,out):
            label=panel['panel'];dest=out/label;dest.mkdir(exist_ok=True);floor=json.loads((fixtures/label/'native_floor.json').read_text())
            with d.configured(panel):
                for geo in panel['geometries']:
                    missing=[mode for mode in ('imposed','autonomous') if label+'/'+geo['id']+'/'+mode not in reused]
                    if not missing:continue
                    checks.extend(dict(panel=label,geometry=geo['id'],**r) for r in c.y.endpoint_preflight(torch,root,models['C0'],geo,panel,cfg,ca,base,device))
                    count=len(c.b.design(geo,panel)[0])*len(geo['parents']);checks.extend(d.remap(c.packed_parity,ARMS=d.ARMS)(torch,adapter,models,count,p,device))
                    with np.load(fixtures/label/('native_'+geo['id']+'.npz')) as z:teacher=dict(z)
                    repeated,info=c.y.rollout(torch,None,geo,panel,cfg,ca,base,device,d.ARMS);reference={k:v[:,:,:1].copy() for k,v in repeated.items()};del repeated
                    physical=c.b.design(geo,panel)[0]
                    for mode in missing:
                        key=label+'/'+geo['id']+'/'+mode;pred,info=c.y.rollout(torch,adapter,geo,panel,cfg,ca,base,device,d.ARMS,reference if mode=='imposed' else None)
                        assert all(float(abs(pred[k][:,:,0]-reference[k][:,:,0]).max())<=c.y.previous.permutation_limit(p,k) for k in reference),'Formula null mismatch'
                        for h in panel['prefix_horizons_ms']:
                            ticks=round(h/.1);measured=c.y.score(c.b.a.crop(pred,ticks),c.b.a.crop(reference,ticks),{k:v[:ticks+1] for k,v in teacher.items()},geo,physical,d.ARMS,mode,panel,cfg,floor['valid'])
                            controls={(r['case'],r['seed']):r for r in measured if r['arm']=='none'};incumbent={(r['case'],r['seed']):r for r in measured if r['arm']=='C0'}
                            for r in measured:
                                null=controls[(r['case'],r['seed'])];b=incumbent[(r['case'],r['seed'])];r.update(panel=label,horizon_ms=h,reference_valid=bool(r['judgment_eligible'] and null['compartment_passed'] and null['current_passed']),regression_vs_f011=bool(b['current_passed'] and b['compartment_passed'] and not(r['current_passed'] and r['compartment_passed'])))
                            rows.extend(measured)
                        np.savez_compressed(dest/('traces_'+geo['id']+'_'+mode+'.npz'),voltage=pred['voltage'],calcium=pred['calcium'],target_gates=pred['gates'][...,[3,7,9,10]],target_currents=pred['current'][...,[1,4,5]])
                        n.p.save_json(dest/('crossings_'+geo['id']+'_'+mode+'.json'),d.crossing_and_gate_probes(pred,reference,geo,physical,mode));del pred
                        finished.add(key);execution[key]=info;n.p.save_json(dest/'results.json',[r for r in rows if r['panel']==label]);n.p.save_json(out/'execution.json',execution)
                        n.p.save_json(out/'recovery_progress.json',dict(completed_blocks=sorted(finished),reused_blocks=sorted(reused),training_performed=False,selection_recomputed=False));print('[GIADA45d recovery] saved '+key+' '+str(len(finished))+'/18',flush=True)
            n.p.save_json(dest/'results.json',[r for r in rows if r['panel']==label])
        assert len(finished)==plan['total_blocks']
        stress={};prior=n.contract(root)[0]
        for name,net in models.items():stress[name]=c.x.repair.stress(torch,net,prior,out,'stress_'+name,device);n.p.save_json(out/'stress_regressions.json',stress)
        assert stress['C0']['passed']==stress['R41']['passed']==66 and stress['R111']['passed']==64
        for r in stress.values():r['new_failures_vs_f011']=sorted(set(map(tuple,r['failed']))-set(map(tuple,stress['C0']['failed'])))
        with np.load(cache/'native_shadow_traces.npz') as z:cost_data=dict(z)
        # Execute the unchanged registered45d cost closure without calling its
        # evaluate/fit pipeline: closure code is lifted verbatim, not rewritten.
        import ast,inspect
        tree=ast.parse(inspect.getsource(d.evaluate));node=next(q for q in tree.body[0].body if isinstance(q,ast.FunctionDef) and q.name=='cost')
        scope=dict(d.__dict__,torch=torch,device=device,out=out,benchmark=d.ORIGINAL_BENCHMARK)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'registered45d_cost','exec'),scope)
        physics=(n.old_native.x.calibration_config(prior,cfg),ca,base);costs=scope['cost'](torch,dict(dual=models['C0'],**{k:v for k,v in models.items() if k!='C0'}),parent,cost_data,physics,p,out,device)
    assert before=={name:n.weights_hash(net) for name,net in models.items()}
    assert (out/'nomination_freeze.json').read_bytes()==freeze_bytes
    report=dict(valid=True,complete=True,task='45d evaluation recovery',decision='COMPLETE_REQUIRES_COMPARATIVE_AUDIT',training_performed=False,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,weights_unchanged=True,nomination_bytes_unchanged=True,all_parameters_finite=True,all_traces_finite=all(r['finite'] for r in rows),panels=summarize(rows,p),stress=stress,nomination=freeze,reused_blocks=sorted(reused),new_blocks=sorted(finished-reused),cost_rows=len(costs),release_authorized=False,speedup_claim=False)
    report['frozen_control_reproduction']=d.control_check(report,control);assert report['all_traces_finite'] and len(costs)==6
    n.p.save_json(out/'endpoint_preflight.json',checks);n.p.save_json(out/'stress_regressions.json',stress);n.p.save_json(out/'final_report.json',report);return report
