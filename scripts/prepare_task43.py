"""Prospective frozen transfer contract, copying all existing quality thresholds."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    p=json.loads((ROOT/'experiments/task42_preregistration_v1.json').read_text())
    p.update(task='43',duration_ms=200.,protocols=['recovery','staircase','alternating','relaxation'],
        node_orders=['canonical','seeded_permutation_43000_plus_node_count'],
        contrasts=['Matched canonical versus permuted physical node order; invert permutation before scoring.',
            'Channel-specific .75/1/1.25 conductance heterogeneity and site-specific conditioning history.',
            'Frozen41r imposed versus autonomous, each against its own canonical formula and native floor.',
            'Disconnected versus weak versus nominal axial coupling on identical local stimulus streams.'],
        hypotheses={'indexing':'Frozen local operator is equivariant to node reindexing with geometry, forcing and parameters permuted together.',
            'transfer':'Heterogeneous channels and conditioned local states transfer without retraining.',
            'feedback':'Longer200ms challenges distinguish local imposed error from autonomous accumulation and coupling amplification.'},
        limits=p['limits'][:-1]+['Task43 is exploratory accepted-limits transfer; no automatic44 qualification.',
            '200ms is not the full10s tail suite; historical six41 timing failures remain unresolved.',
            'Timing support is reported per channel, never count not_evaluable as timing success.'],
        evaluation='Original per-site quality and compartment/native floor gates. Report event support per channel. Permutation error audited separately using existing numerical tolerances; not a replacement for quality gates.',
        execution='Native and GPU subprocesses isolated. Three frozen model seeds and all12 cases vectorized per topology; two node-order controls sequential on same GPU to preserve matched comparisons. No training or selection; CUDA graph/eager path inherited from42 with explicit fallback and no speedup claim.')
    write_json(ROOT/'experiments/task43_preregistration_v1.json',p)
if __name__=='__main__':main()
