"""CPU float64 read-only probes of frozen learned candidate/fraction functions."""
import json
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task39b_architecture_diagnosis as task

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/task39b_kaggle_v2'


def main():
    torch.set_num_threads(1)
    execution=json.loads((OUT/'execution.json').read_text())
    freeze=json.loads((OUT/'selection_freeze.json').read_text())
    model=task.model_class(torch)(freeze['arms'],freeze['seeds'],execution['rapid_output_positions'])
    model.load_state_dict(torch.load(OUT/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)); model.double()
    # A fixed held-voltage diagnostic used by the history-tail protocol. Not a new test set.
    v=torch.full((36,1),-55.,dtype=torch.float64); ca=torch.full_like(v,.0001)
    with torch.no_grad():
        inp=torch.cat(((v[...,None]+40)/80,torch.zeros(36,1,17,dtype=v.dtype)),-1)
        raw=torch.bmm(torch.nn.functional.silu(torch.bmm(inp,model.a)+model.b),model.c*model.head_mask)+model.d[:,None]
        u=torch.sigmoid(raw[...,:17]); z=model.fraction(raw[...,17:])
        names=[f'{channel}.{gate}' for channel,gates in zip(task.p.ionic.CHANNELS,task.p.ionic.STATE_NAMES) for gate in gates]
        exact_u,exact_z=task.p.coefficients(np.array([-55.]),np.array([.0001]),.1)
        records=[]
        for member in range(36):
            for local,gate in enumerate(task.old.IDS):
                eu,ez=float(exact_u[0,gate]),float(exact_z[0,gate]); pu,pz=float(u[member,0,local]),float(z[member,0,local])
                records.append(dict(arm=freeze['arms'][member//3]['name'],seed=freeze['seeds'][member%3],gate=names[gate],
                    voltage_mv=-55,reference_u=eu,learned_u=pu,reference_z=ez,learned_z=pz,
                    equilibrium_error=pu-eu,z_ratio=pz/ez,reference_tau_ms=-.1/np.log1p(-ez),
                    learned_tau_ms=-.1/np.log1p(-pz)))
    report=dict(valid=True,records=records,scope='Post-freeze held V=-55mV CPU float64 coefficient probe; no trajectory replay or proof of global cause. SK excluded. Direct head extraction avoids subtractive cancellation in 1-slope.')
    (OUT/'coefficient_probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for arm in ('base_h8_c1_r0','base_h64_c1_r0','multiscale_h64_c1_r1'):
        rs=[r for r in records if r['arm']==arm]
        selected=sorted({r['gate'] for r in sorted(rs,key=lambda r:abs(r['equilibrium_error']),reverse=True)[:6]})
        print(json.dumps(dict(arm=arm,largest_equilibrium_errors=[r for r in rs if r['gate'] in selected]),indent=2))


if __name__=='__main__':main()
