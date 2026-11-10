"""Generate the prospective42 protocol from immutable existing quality gates."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task42_axial_exploration as x
from research_memory.mirror import write_json

def main():
    prior,cfg,ca,base=x.n.contract(ROOT)
    freeze=json.loads((ROOT/'experiments/task42_exploratory_transition_v1.json').read_text())
    protocol=dict(task='42',version='v1',exploratory=True,freeze=freeze,seeds=[17,29,43],dt_ms=.1,duration_ms=40.,
        coupling=[0.,.25,1.],protocols=['rest','proximal','distal','paired'],geometries=[x.geometry(k) for k in (2,4,8,16)]+[x.geometry(5,True)],
        quality=prior['quality'],compartment_gates=cfg['gates'],
        passive_gates=x.passive.config(ROOT,'iv_ad1_passive_axial_preregistration_v3.json')['gates'],
        training_updates=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,
        contrasts=['coupling0 vs weak vs nominal, same protocols/sites and frozen weights',
                   'formula vs41r on identical imposed V/Ca paths vs autonomous local states',
                   'native passive vs independent dense reference on each new topology',
                   'native active reference vs canonical kinetic formula and same classical axial solve'],
        execution='Native NEURON and CUDA evaluation isolated subprocesses. Modelseeds vectorized; cases/sites batched within topology. Graph/eager bitwise preflight; explicit eager fallback only for graph capture unavailable, never parity failure. No timing speedup qualification.',
        evaluation='Per-site original current/gate metrics including event support; native V/Ca floor with original compartment thresholds. Native-floor failure disables model judgment for affected cases, but preserves diagnostics.40ms interface panel is not a long-tail stress confirmation.',
        hypotheses={'port':'Area/current-density and axial reciprocity reproduce native passive references on new chains/tree.',
                    'feedback':'New failures in autonomous vs imposed paths indicate feedback amplification, not automatically missing local state.',
                    'coupling':'Topology/coupling contrasts separate disconnected batching errors from true propagation effects.'},
        backup='Persist native files after each topology, progress JSON and all frozen outputs. Notebook parent archives partial outputs on worker failure. Kaggle persisted ZIP recovered locally and SHA/CRC verified. DB complete backup local only. No intermediate backup claim without external readback.',
        limits=['Task41 remains NO-GO on original gates; known six timing errors not repaired here.',
                'Small current-clamp systems, no synapses/RNG, no full Hay morphology or independent model qualification.',
                'No training, checkpoint/architecture selection, teacher reset in autonomous mode, new abstraction or threshold waiver.',
                'Native floor failure is a reference/semantics issue, not a learned-model NO-GO.',
                'Dense classical axial solve reference is not a custom accelerated kernel; measured wall times descriptive only.',
                'Task43 requires audit and explicit decision; no automatic downstream authorization.'])
    write_json(ROOT/'experiments/task42_preregistration_v1.json',protocol)
    print('Task42 protocol generated; original thresholds copied exactly.')
if __name__=='__main__':main()
