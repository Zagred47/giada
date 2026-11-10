"""Preregister causal spatial/channel substitution without changing gates."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher import task44_composition_localization as y

def main():
    p=json.loads((ROOT/'experiments/task43_preregistration_v1.json').read_text())
    p.update(task='44',geometries=[y.x.geometry(4),y.x.geometry(8),y.x.geometry(5,True)],protocols=['recovery','alternating'],arms=list(y.ARMS),groups=y.GROUPS,
        node_orders=['canonical; permutation equivalence previously audited in43'],
        hypotheses={'spatial':'One-site replacement separates local injected error from spread to formula-only neighbors.',
            'families':'Disjoint calcium/sodium/potassium/Ih substitutions distinguish family-specific and Ca-mediated cross-family error.',
            'feedback':'Same masks imposed versus autonomous and coupling0/.25/1 isolate feedback from inherited local approximation.'},
        contrasts=['none vs proximal-only vs distal-only vs all, same histories and conductances.',
            'none vs each disjoint family vs all: diagnostic interactions, not additive-error assumption.',
            'Each arm imposed versus autonomous on the same formula V/Ca paths.',
            'Each protocol/seed/arm across coupling0/.25/1 (strongest in this declared panel, not universal strong-coupling qualification).'],
        execution='Eight independent substitution arms and physical cases batched on a single resident GPU; three model seeds kept separate. Explicit eager mode, no CUDA Graph claim. Skip discarded local membrane solve only after bitwise legacy-core endpoint preflight. Same FP32 gates/readout and FP64 axial solve. No optimization or model selection.',
        evaluation='Original per-site current/gate, V/Ca and native-floor thresholds. Null-substitution control mandatory. Raw per-node peak voltage/time and0mV upcrossing sequences are propagation diagnostics only, not new acceptance gates. Record timing support and missing events separately.',
        limits=p['limits'][:-3]+['Task44 causal localization is exploratory, not independent full-neuron qualification.',
            'Hybrid masked arms are diagnostic interventions, not new nominated lower-abstraction candidates; all-arm remains unchanged41r.',
            '200ms/two protocols/three small topologies do not resolve original six41 timing failures or qualify10s tails.',
            'Original single-transient current timing evaluator retained; raw voltage crossing sequences do not certify every multiburst computational behavior.',
            'No training, selection, synaptic changes or automatic45 promotion; no speedup qualification.'])
    write_json(ROOT/'experiments/task44_preregistration_v1.json',p)
if __name__=='__main__':main()
