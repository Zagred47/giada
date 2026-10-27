"""Unchanged complete qualification, with actual whole-operator timing."""
import copy
import json
import types
from . import task41l_regression_attribution as j
from . import task41i_evaluation as previous


def install_cache(out):
    # This installs only completed-panel IO guards and the already repaired H
    # comparator. Source-bound numerical and metric bodies remain unchanged.
    previous.install_resume_cache(out)


def evaluate(torch,root,out,controls,trained,spec,cfg,calcium,base,device):
    fe=previous.fe;he=previous.he
    proxy=types.SimpleNamespace(**{**vars(j),'contrasts':lambda:[]})
    quality=types.FunctionType(fe.evaluate.__code__,dict(fe.__dict__,f=proxy,execution_matrix=lambda *a,**kw:[]))
    benchmark=he.ee.e.benchmark
    # The inherited quality suite also invoked several obsolete execution
    # matrices. Do quality once, then time all actual new operators explicitly.
    he.ee.e.benchmark=lambda *a,**kw:[]
    try:report=quality(torch,root,out,controls,trained,spec,cfg,calcium,base,device)
    finally:he.ee.e.benchmark=benchmark
    data,_,_=he.ee.load41c(root,spec)
    benchmark(torch,root,out,{**controls,**trained},data,spec,device)
    # A frozen numerical-policy sidecar distinguishes roundoff from learning.
    # It is NOT allowed to nominate or waive any failed primary FP32 gate.
    freeze=j.verify_freeze(out)
    he.numerical_panels(torch,root,out/'selected_numerical_audit',
                        {'H_nominee':controls['H_nominee'],'nominee':trained[freeze['nominee']]},spec,cfg,calcium,base,device)
    # Same graph compiler and wall/CUDA timers for parent and all three arms.
    # Compilation is an execution diagnostic; only FP32 full-suite candidates
    # can qualify. The quality equivalence probe is separately mandatory.
    parity=[]
    for name,model in trained.items():
        vv=torch.linspace(-99,59,257,device=device)[None].expand(3,-1)
        ca=torch.logspace(-7,-2,257,device=device)[None].expand(3,-1)
        compiled=previous.RuntimeModel(torch,copy.deepcopy(model),'compiled')
        actual=compiled.raw(vv,ca);expected=model.raw(vv,ca)
        error=max(float((a-b).abs().max()) for a,b in zip(actual,expected))
        parity.append(dict(model=name,max_abs_raw_error=error,valid=error<=spec['computational_parity_atol'],selection_eligible=False))
    j.p.save_json(out/'compiled_parity.json',parity)
    contrasts=[]
    for contrast in spec['registered_comparisons']:
        left,right=contrast['control'],contrast['treatment']
        a,b=freeze['ranks'][left],freeze['ranks'][right]
        compression=left.startswith('K_copy')
        contrasts.append(dict(contrast,control_ranks=a,treatment_ranks=b,
            selection_metric='parent fidelity, not canonical repair' if compression else 'canonical development gates',
            primary_rank_difference=[b[s][0]-a[s][0] for s in range(3)],
            full_suite_control=report['summaries'][left],full_suite_treatment=report['summaries'][right],
            control_globally_qualified=left in report['passing_arms'],treatment_globally_qualified=right in report['passing_arms'],
            new_nominee_selected=False,interpretation='Report all paired effects; global scientific gates remain mandatory. Different parameter counts and initial compression gaps are not controlled away.'))
    j.p.save_json(out/'paired_contrasts.json',contrasts)
    report['compression_canonical_passing_arms']=[name for name in report['passing_arms'] if name.startswith('K_copy')]
    report['passing_arms']=[name for name in report['passing_arms'] if not name.startswith('K_copy')]
    report['compression_parent_fidelity_ranks']={name:freeze['ranks'][name] for name in trained if name.startswith('K_copy')}
    report.update(task='41l',decision='REPAIR_REQUIRES_INDEPENDENT_CONFIRMATION' if freeze['nominee'] in report['passing_arms'] else 'REGRESSION_ATTRIBUTION_NO_GO',
                  nominee_passed=freeze['nominee'] in report['passing_arms'],
                  task42_authorized=False,independent_confirmation=False,speedup_claim=False,
                  scientific_scope='Same 18 learned occupancies, same affine bounded transition, same V/Ca dependencies and dt; no kinetic inference fallback.',
                  parameter_counts={name:sum(x.numel() for x in model.parameters())//3 for name,model in trained.items()},
                  no_reselection_on_opened_panels=True)
    from .task41l_frozen_attribution import evaluate as attribution
    report['frozen_causal_attribution']=attribution(torch,root,out,controls,spec,cfg,calcium,base,device)
    return report


