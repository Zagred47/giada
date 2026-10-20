"""Additional frozen-weight/selection audit and explicit causal limitations."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39j_atomic_relaxation as j
OUT=ROOT/'experiments/results/task39j_audited_v1'
def read(n):return json.loads((OUT/n).read_text())
torch.set_num_threads(1)
curves=read('training_curves.json');freeze=read('selection_freeze.json');spec=read('preregistration.json')
states=torch.load(OUT/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)
weights=j.parent_weights(torch,ROOT,spec,'cpu');rank_checks=[]
for k,chosen in enumerate(freeze['chosen_steps']):
    eligible=[(tuple(c['panel'][k]['rank']),c['step']) for c in curves if c['panel'][k]['selection_eligible']]
    best=min(eligible,key=lambda r:r[0]);assert best[1]==chosen and list(best[0])==freeze['ranks'][k]
    rank_checks.append(dict(member=k,step=chosen,valid=True))
parent=j.i.model_class(torch)('plain',spec['seeds']);parent.load_state_dict(weights)
v=torch.linspace(-100,60,513)[None].expand(3,-1);ca=torch.full_like(v,1e-4)
parameters={};counterfactual={}
with torch.no_grad():
    for arm in j.arms():
        m=j.make_model(torch,weights,arm,spec['seeds'],'cpu');m.load_state_dict(states[arm['name']])
        parameters[arm['name']]=dict(total_per_seed=sum(p.numel() for p in m.parameters())//3,
            trainable_per_seed=sum(p.numel() for p in m.parameters() if p.requires_grad)//3,
            correction_norms=m.correction.flatten(1).norm(dim=1).tolist())
        if not arm['parent_trainable']:
            m.correction.zero_();a=m.raw(v,ca);b=parent.raw(v,ca)
            assert all(torch.equal(x,y) for x,y in zip(a,b));counterfactual[arm['name']]='Zero learned correction exactly restores I3 coefficients on513point diagnostic grid'
initial=curves[0]['atomic_nonselectable'];last=curves[-1]['atomic_nonselectable']
ratios={k:(np.array(last[k])/np.array(initial[k])).tolist() for k in last}
audit=read('audit.json');cpu=json.loads((OUT/'cpu_J1_replay/audit.json').read_text());assert audit['valid'] and cpu['valid']
findings={
 'outcome':'J1-J7 development72/72 versus frozenI3 66/72. J1 stress70/72 versus62/72. J4 stress69/72. J8 selectsstep0 for allseeds and stays66/72,62/72.',
 'sufficient_package':'I3 parent frozen +256 learned bounded residual weights onjoint u/z ofNap_Et2,K_Pst +coefficient supervision on1024independent FITanchors suffices inJ1. No canonical kinetics in inference. Not a proof that each element is necessary.',
 'freeze_contrast':'J2 vsJ8 differs byparenttrainability. Unfreezing harms selection-panel performance under this budget/objective/learningrate; it does not prove unfreezing can never work. J8 lowers coefficient holdout loss dramatically but not relaxation loss;global scalar optimization is not eventtiming qualification.',
 'unnecessary_additions':'Extra relaxationloss, sensitive sampling,fullbatch and channelclipping are not required for72/72onthispanel;not proof they never help. Global FourierJ4 also passes,so locality is not necessary here.',
 'registered_contrasts':'0/8 meet registered thresholds;severity excludes timing and max-over-channel metrics are often dominated by unchangedchannels. Do not call unfreezing harm a positive result for the reverse preregistered improvement test.',
 'remaining_stress':'Ca_LVAst same_v_history case1 seed17 offset0.5ms versus0.3ms limit;Ca_HVA activation_boundary case0 seed29 nRMSE0.01178899 versus0.01. Both unchanged fromI3,not introducedbyJ1.',
 'selection':'Same development-only lexicographic panel;chosencheckpoints independentlyrecomputed. Three seeds share24trajectories/role,not72independentprotocols. Many repeateddevelopmentiterations require Task41independentconfirmation.',
 'numerics':'Actual correction affine multiply accumulatesFP64 evenat inference,then returnsFP32;serialstatesFP32. Historical copied narrative saying purelyFP32deployment is not literal. CPU replay reproducesall144pass/fail outcomes,notbitwiseidentity.',
 'scope':'V/Ca imposed;18explicitgates;knowncheapcurrentreadout retained. No autonomousvoltage,causalcompartment,biologicalgenerality orspeedup established.',
 'decision':'GO to prepareTask40 causal reinsertionandlocalcost withfrozenJ1 three-seed ensemble andI3/formulacontrols. Not GO todeploy,Task41orTask42. Include knownstressfailures and numericalinterfaceaudit;unchangedthresholds. NoTask40runlaunchedbythisaudit.'}
result=dict(valid=True,selection_checks=rank_checks,parameters=parameters,zero_correction_control=counterfactual,
 atomic_holdout_last_over_initial=ratios,atomic_terms=['coefficient','relaxation_plus_shape','preservation'],
 findings=findings,task40_preparation_authorized=True,task40_execution_started=False,independent_confirmation_passed=False)
(OUT/'attribution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(parameters=parameters,J1_ratios=ratios['J1'],J8_ratios=ratios['J8'],decision=findings['decision'])))
