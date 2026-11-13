"""45e: preserve the whole45c model while refining existing ionic heads."""
import copy,io,json,os,zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from . import task45d_learnability_execution as d
c=d.c;n=d.n
ARMS=('none','C0','C1','P','N00','N01','N10','N11','Iz','Iu','J','R41','R111')

def spec(root):
    p=json.loads((root/'experiments/task45e_preregistration_v1.json').read_text());_,cfg,ca,base=n.contract(root)
    assert p['quality']==d.spec(root)[0]['quality'] and p['compartment_gates']==cfg['gates']
    assert tuple(p['coupled_arms'])==ARMS and not p['thresholds_changed'] and not p['abstraction_changed']
    return p,cfg,ca,base

def assert_preserved(torch,model,anchor,gate,joint):
    columns=([gate] if joint else [])+[17+gate];keep=[i for i in range(anchor.parent.d.shape[-1]) if i not in columns]
    a=model.state_dict();b=anchor.state_dict()
    for key in a:
        if key=='parent.c':assert torch.equal(a[key][:,:,keep],b[key][:,:,keep]),key
        elif key=='parent.d':assert torch.equal(a[key][:,keep],b[key][:,keep]),key
        else:assert torch.equal(a[key],b[key]),key
    return dict(valid=True,allowed_columns=columns,untargeted_tensors_bitwise_equal=True)

def fit(torch,baseline,p,out,device):
    d.prepare_training_readout(torch,device)
    with zipfile.ZipFile(os.environ['GIADA_TASK45C_CHECKPOINTS']) as z:
        anchor=copy.deepcopy(baseline);anchor.load_state_dict(torch.load(io.BytesIO(z.read('J.pt')),map_location=device,weights_only=True))
    if device=='cuda':
        with np.load(out.parent/'verified_source45c/native_shadow_traces.npz') as z:probe=dict(z)
        prior,cfg,ca,base=n.contract(Path(__file__).resolve().parents[2]);small=copy.deepcopy(p);small['cost'].update(batches=[1],ticks=2,repetitions=2)
        dest=out/'GPU_cost_preflight';dest.mkdir()
        with torch.inference_mode():checks=d.ORIGINAL_BENCHMARK(torch,dict(dual=anchor),anchor.parent,probe,(n.old_native.x.calibration_config(prior,cfg),ca,base),small,dest,device)
        assert len(checks)==2 and all(r['all_outputs_finite'] for r in checks)
        n.p.save_json(out/'GPU_cost_preflight.json',dict(valid=True,scopes=2,ticks=2,training_budget_consumed=False))
    models={'C1':anchor};records=[];snapshots={};diagnostics=[];lineage=[]
    with zipfile.ZipFile(os.environ['GIADA_TASK45D_SAVED']) as z:
        donor=copy.deepcopy(anchor);donor.load_state_dict(torch.load(io.BytesIO(z.read('Nline.pt')),map_location=device,weights_only=True))
    # The feature trunk must match before head transplantation is meaningful.
    for name in ('a','b'):assert torch.equal(getattr(anchor.parent,name),getattr(donor.parent,name))
    models['P']=c.materialize(torch,anchor,9,False,c.readout_weights(torch,donor,9,False));lineage.append(dict(arm='P',**assert_preserved(torch,models['P'],anchor,9,False)))
    fits=[d.role(torch,anchor,p,'fit',bool(s),out,device) for s in (0,1)]
    dev=d.union(*[d.role(torch,anchor,p,'dev',bool(s),out,device) for s in (0,1)])
    base_rows=c.evaluate_role(torch,anchor,dev,p,device);n.p.save_json(out/'development_C1.json',base_rows)
    prows=c.evaluate_role(torch,models['P'],dev,p,device);records.append(('P',0,prows));snapshots[('P',0)]=models['P'];n.p.save_json(out/'development_P_0.json',prows)
    # Old45dDEV is a diagnostic only, not part of this task's nomination.
    oldp=d.spec(Path(__file__).resolve().parents[2])[0]
    olddev=d.union(c.histories(oldp,'dev'),d.matched(c.histories(oldp,'dev')))
    oldrows=c.evaluate_role(torch,models['P'],olddev,p,device)
    n.p.save_json(out/'opened45d_development_P.json',dict(rows=oldrows,used_for_selection=False))
    pools=[d.pool(torch,anchor,p,fits[s],s,device) for s in (0,1)]
    for label,arm in p['matrix']['arms'].items():
        gate=arm['gate'];joint=arm['joint'];po=pools[arm['support']];data=fits[arm['support']]
        w,_=d.whitening(po,p);width=w.shape[-1];heads=2 if joint else 1;white=np.zeros((3,width*heads,width*heads))
        for h in range(heads):white[:,h*width:(h+1)*width,h*width:(h+1)*width]=w
        theta=np.zeros((3,width*heads));w0=c.readout_weights(torch,anchor,gate,joint);stopped=None
        for step in range(1,max(p['fitting']['checkpoints'])+1):
            if stopped is None:
                theta,info=c.solve_refinement(data,po,gate,joint,arm['path_guard'],False,theta,white,p)
                diagnostics.append(dict(arm=label,step=step,**info))
                if max(info['unclipped_step_norm'])<1e-8:stopped=step
            if step not in p['fitting']['checkpoints']:continue
            delta=np.einsum('sij,sj->si',white,theta).reshape(3,heads,width).transpose(0,2,1)
            model=c.materialize(torch,anchor,gate,joint,(w0+delta).astype('float32'))
            lineage.append(dict(arm=label,step=step,**assert_preserved(torch,model,anchor,gate,joint)))
            assert all(bool(torch.isfinite(q).all()) for q in model.parameters())
            torch.save(model.state_dict(),out/f'{label}_{step}.pt');snapshots[(label,step)]=model
            rows=c.evaluate_role(torch,model,dev,p,device);records.append((label,step,rows));n.p.save_json(out/f'development_{label}_{step}.json',dict(rows=rows,early_stopped_at=stopped))
            n.p.save_json(out/'refinement_diagnostics.json',diagnostics);n.p.save_json(out/'lineage_checks.json',lineage)
            print(f'[GIADA45e] {label} budget{step} saved; untouched heads preserved',flush=True)
    choices={}
    for label in ('P',*p['matrix']['arms']):
        rec=d.choose([r for r in records if r[0]==label],base_rows);models[label]=snapshots[(label,rec[1])]
        choices[label]=dict(checkpoint=rec[1],DEV_nonregressing=d.safe(rec[2],base_rows))
    chosen=d.choose([('C1',0,base_rows)]+[r for r in records if d.safe(r[2],base_rows)],base_rows)
    models['J']=copy.deepcopy(anchor if chosen[0]=='C1' else snapshots[(chosen[0],chosen[1])])
    freeze=dict(valid=True,chosen=dict(arm=chosen[0],checkpoint=chosen[1]),per_arm_choices=choices,selection_used_coupled=False,selection_used_old_stress=False,selection_used_old45dDEV=False,
        weights_sha256={k:n.weights_hash(v) for k,v in models.items()},lineage_checks=lineage,refinement_diagnostics=diagnostics)
    n.p.save_json(out/'nomination_freeze.json',freeze);return models,freeze

@contextmanager
def configured(p):
    before=c.y.spec,c.y.ARMS,c.y.previous.design,c.y.channel_masks
    c.y.spec=lambda root:(p,*n.contract(root)[1:]);c.y.ARMS=ARMS;c.y.previous.design=c.b.design
    c.y.channel_masks=lambda count,arms:np.array([np.full((count,11),arm!='none',dtype=bool) for arm in arms])
    try:yield
    finally:c.y.spec,c.y.ARMS,c.y.previous.design,c.y.channel_masks=before

def evaluate(torch,root,source,fixtures,checkpoints,saved,out,device='cuda'):
    plan=json.loads((root/'experiments/task45d_evaluation_recovery_v1.json').read_text());assert c.sha(saved)==plan['sha256']
    ck=json.loads((root/'experiments/task45c_cost_transport_v2.json').read_text());assert c.sha(checkpoints)==ck['sha256']
    os.environ['GIADA_TASK45D_SAVED']=str(saved);os.environ['GIADA_TASK45C_CHECKPOINTS']=str(checkpoints)
    benchmark=d.ORIGINAL_BENCHMARK
    def cost(*args):
        nets=args[1];measured={k:nets[k] for k in ('dual','C1','P','J','N00','N01','N10','N11')};parity=[]
        for label in ('C1','P','J','N00','N01','N10','N11'):
            for variant,net in n.execution.variants(torch,nets[label]).items():
                exact=True
                for batch in (1,2,64,176,642,4096):
                    v=torch.linspace(-135,60,batch,device=device)[None].expand(3,-1);ca=torch.logspace(-8,-1,batch,device=device)[None].expand(3,-1)
                    exact &= all(n.tensor_bits_equal(torch,a,b) for a,b in zip(nets[label].raw(v,ca),net.raw(v,ca)))
                parity.append(dict(arm=label,variant=variant,bitwise_equal=exact,classification='equivalent_on_preflight' if exact else 'diagnostic_not_eligible'))
                if exact and variant!='dual':measured[label+'_'+variant]=net
        n.p.save_json(out/'execution_variant_parity.json',parity);mutable=list(args);mutable[1]=measured
        return benchmark(*mutable)
    from unittest.mock import patch
    def adapter(*args):
        model=d.remap(d.adapter,ARMS=ARMS)(*args);n.p.save_json(out/'packed_execution_preflight.json',model.execution_preflight);return model
    funcs=dict(spec=spec,fit=fit,batched_model=adapter,configured=configured,ARMS=ARMS,
        packed_parity=d.remap(c.packed_parity,ARMS=ARMS),crossing_audit=d.remap(d.crossing_and_gate_probes,ARMS=ARMS))
    with patch.object(n,'benchmark',side_effect=cost):report=d.remap(c.evaluate,**funcs)(torch,root,source,fixtures,out,device)
    report.update(task='45e',decision='ANCHORED_REPAIR_COMPLETE_REQUIRES_AUDIT',frozen_control_reproduction=d.control_check(report,checkpoints),release_authorized=False,speedup_claim=False)
    n.p.save_json(out/'final_report.json',report);return report
