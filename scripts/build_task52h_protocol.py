import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'experiments/task52g_preregistration_v1.json').read_text())
p['task']='52h'
p['arms']=[dict(copy.deepcopy(p['arms'][1]),id=name,project=False,worst_guard=worst,targeted_Ih=target) for name,target,worst in [('guard_control',False,False),('Ih_support',True,False),('worst_current_guard',False,True),('combined',True,True)]]
p['targeted_Ih']=dict(start_ms=[50,150,350,550,700,900],fraction_total_batch=.25,description='Replace two of eight fit windows with existing slow_recovery fit histories, sampling preregistered onset/late hyperpolarization/recovery anchors. No development or fresh targets added. Same total batch and updates.')
p['feasibility_probe']=dict(chunk_ticks=512,batch_size=4,steps=[1e-3,1e-4,1e-5,1e-6],target_channel=5,relative_tolerance=1e-6,absolute_tolerance=1e-12,rule='At least one fit-only probe step per seed must strictly improve Ih MSE and preserve every other channel under unchanged non-regression tolerance. Probe weights never initialize training. Failure stops full training; not proof of infeasibility.')
p['objective']='Paired2x2: targeted Ih fit-window coverage x mean-plus-worst-channel preservation. Same original52d initialization, frozen encoder/readout,10states,800updates,512BPTT. No gradient projection. Identical draws for paired arms; targeted replacement only in designated quarter-batch.'
p['limitations']='Exploratory reused52panel. Worst-channel penalty and support sampling are training heuristics, not guarantees. Preflight is a bounded fit-only local search, not proof of global existence or impossibility. Original selection and quality thresholds unchanged. No extra inference state or cost. Cap52j.'
p['diagnostics']='Original channel/V/Ca eligibility per checkpoint; Ih error and regressions for all11channels; support x worst-guard contrasts and interaction. Same final imposed/autonomous, interventions and cost evaluation. No promotion by mean score.'
(ROOT/'experiments/task52h_preregistration_v1.json').write_text(json.dumps(p,indent=2)+'\n')
print('52h preregistered')
