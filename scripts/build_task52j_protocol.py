import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'experiments/task52i_preregistration_v1.json').read_text());p['task']='52j';p['updates']=400;p['checkpoints']=[25,50,100,150,200,300,400]
p['warmstart']=dict(source_sha256='a9bf728bb5d567c1abffb35d9293d7e460b04d52dda2a36982e3f6ca2bb61f45',indices=[9,10,11],selected_updates=[150,0,300],description='Same selected52i combined weights for all four arms; one fallback inherited. Common optimizer reset, not recovery or independent confirmation.')
p['arms']=[dict(copy.deepcopy(p['arms'][3]),id=name,long_guard=wide,coupled_guard=coupled) for name,wide,coupled in [('short_current',False,False),('long_current',True,False),('short_coupled',False,True),('long_coupled',True,True)]]
p['step_acceptance'].update(long_ticks=512,coupled_cases=2,coupled_start_ms=200,coupled_ticks=128)
p['objective']='Final2x2 diagnostic:128vs512tick current acceptance panel x additional128tick autonomousV/Ca acceptance panel. Same decaying step, initial52i combined weights,400attempts. All reserved fit panels excluded from gradient sampling in every arm.'
p['limitations']='Last52subtask, no52k. Reused data and warmstart, not independent confirmation.400attempts chosen because52i had no eligible600/800checkpoints. Fit acceptance cannot prove generalization. Proceed53after audit with unresolved limits explicit. Thresholds and10states unchanged.'
(ROOT/'experiments/task52j_preregistration_v1.json').write_text(json.dumps(p,indent=2)+'\n')
