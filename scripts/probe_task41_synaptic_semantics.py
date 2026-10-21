"""Opened technical controls only: identify event-age omission; no B6 access."""
import json,sys
from pathlib import Path
root=Path('/content/giada_task41_ac24e4a44226/giada');sys.path.insert(0,str(root))
from src.giada_teacher import task41_independent_confirmation as c
from neuron import h,load_mechanisms
spec,cfg,calcium,base=c.contract(root);cfg=c.calibration_config(spec,cfg)
_,_,c3=c.t.t33.contract(root)
load_mechanisms('/content/giada_task41_ac24e4a44226/artifacts/compiled_active_synapses')
rows=[]
for row in c.design(spec):
    if row['panel']!='canonical' or row['seed']!=104117 or row['initial_voltage_mv']!=-77. or row['intervention']!='none':continue
    inj,mult=c.case_inputs(row,spec)
    native=c.t.t33._native_case(h,cfg,base,calcium,c3,row,injection_density=inj,channel_multipliers=mult)
    old=c.t.t33.shadow_case(h,cfg,c3,row,native)
    fixed=c.t.t33.shadow_case(h,{**cfg,'receptor_time_semantics':'ems_fixed_step'},c3,row,native)
    rows.append(dict(protocol=row['protocol'],before_passed=c.t.t33._shadow_pass(old,cfg),after_passed=c.t.t33._shadow_pass(fixed,cfg),
        before={k:old[k] for k in ('max_shadow_current_error_na','max_shadow_receptor_state_error','release_mismatch_count')},
        after={k:fixed[k] for k in ('max_shadow_current_error_na','max_shadow_receptor_state_error','release_mismatch_count')}))
report=dict(valid=all(r['after_passed'] for r in rows),opened_control_cases=rows,B6_accessed=False,thresholds_changed=False)
Path('/content/task41_synaptic_semantics_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
