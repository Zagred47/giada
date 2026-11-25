import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'experiments/task52h_preregistration_v1.json').read_text());p['task']='52i';p.pop('feasibility_probe')
p['warmstart']=dict(source_sha256='0cb383d40e30e86d0dee23357954e24b2beed798c6f269152b2b21a31ab5a584',indices=[6,7,8],selected_updates=[300,0,300],description='Same per-seed selected worst_current_guard52h weights for all four arms;seed29 is original fallback. Reset Adam identically for all arms; this is a new experiment, not recovery.')
p['arms']=[dict(copy.deepcopy(p['arms'][2]),id=name,targeted_Ih=False,decay=decay,acceptance_guard=guard) for name,decay,guard in [('continuation',False,False),('lr_decay',True,False),('step_guard',False,True),('combined',True,True)]]
p['step_acceptance']=dict(ordinary_cases=2,support_cases=2,start_ms=350,ticks=128,trial_factors=[1.,.5,.25,.125],minimum_lr_fraction=.05,description='Fixed fit-only panel excluded from all gradient minibatches in all arms. Accept first step preserving each current MSE versus initial warmstart within original1e-6relative+1e-12absolute tolerance, otherwise zero step. Reject restores Adam moments; common Adam bias-correction clock counts attempts. No development/fresh acceptance feedback.')
p['objective']='Paired2x2 late step decay x fit-panel actual-step current guard; warmstart same52h selected perseed. Same worst-current teacher loss and800attempted updates. No extra states or inference mechanism.'
p['selection']='Unchanged original52d per-channel andV/Ca development guards and same validation objective/checkpoints. Initial warmstart retained as fallback; no reset of tolerances to a worse baseline.'
p['limitations']='Training-panel acceptance is not a generalization guarantee. Adam moments reset equally at experiment start; common clock counts attempted steps. Warmstart includes one original fallback seed. Same10states and thresholds; reused evaluation not independent confirmation. Cap52j.'
(ROOT/'experiments/task52i_preregistration_v1.json').write_text(json.dumps(p,indent=2)+'\n')
