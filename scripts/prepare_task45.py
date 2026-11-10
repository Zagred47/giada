"""Prospective controlled coverage before the roadmap's optional refit."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher import task45_controlled_coverage as a

def main():
    p=json.loads((ROOT/'experiments/task44_preregistration_v1.json').read_text())
    branch=a.x.geometry(7,True);branch.update(id='45_asymmetric_branch7',parents=[-1,0,0,1,1,2,5])
    chain=a.x.geometry(6);chain['id']='45_chain6'
    p.update(task='45',phase='conditional_refit_need_assessment_frozen',arms=list(a.ARMS),duration_ms=1000.,prefix_horizons_ms=[100.,500.,1000.],
        geometries=[a.x.geometry(4),chain,branch],protocols=['45_hyper_rebound','45_burst_recovery','45_slow_staircase'],
        hypotheses={'history':'Frozen coverage transfers to hyperpolarization/rebound, repeated burst recovery and slower drive histories.',
            'geometry':'Anchor chain4 versus diagnostic chain6/asymmetric7 separates history changes from new composition geometry.',
            'exposure':'Matched100/500/1000ms prefixes reveal horizon-dependent gaps without duplicate simulations or training.',
            'refit_need':'Only eligible frozen failures justify proposing the optional controlled refit; success does not justify unnecessary retraining.'},
        contrasts=['Formula-only null vs fully frozen41r, identical batch shape and input histories.',
            'Each arm imposed versus autonomous, coupling0/.25/1, original numerical policy.',
            'Same protocols on opened anchor chain4 versus new diagnostic geometries.',
            '100/500/1000ms from the same simulation trajectory; original gates at each horizon.'],
        execution='Two arms batched with all physical cases, three model seeds separate; explicit eager GPU; unchanged44 core with legacy bitwise preflight at actual batch widths. Native/GPU isolated processes. No training or selection.',
        evaluation='Original current/gate/native/V/Ca gates at full horizon and all preregistered prefixes. Event support explicit. Refit recommendation requires valid references and any eligible frozen full/prefix failure; never starts an optimizer automatically.',
        conditional_refit=dict(performed=False,authorized=False,reason='Task44 all arms pass; user requests frozen41r. First test expanded coverage before deciding if the optional roadmap refit is needed.',training_mini_scaling_deferred=True),
        reserved_future_confirmation=dict(geometry_namespace='47_*',protocol_namespace='47_*',generated_or_accessed=False),
        limits=['Exploratory controlled coverage, not independent Task47 confirmation or full Hay qualification.',
            '41r remains the only frozen nominee; original six41 timing failures remain unresolved and unchanged.',
            '1000ms and declared small geometries/protocols do not qualify10s tails or every computational response.',
            'Reference/floor failures block candidate judgment and do not justify a refit.',
            'No training/selection, no neighbors future input, no new architecture/abstraction/precision or threshold changes.',
            'Optional refit and training-budget mini scaling are deferred, not claimed completed; require separate explicit decision.',
            'Original single-transient current timing evaluator retained; raw voltage crossing sequences descriptive only.',
            'No speedup claim or automatic46/47 authorization.'])
    write_json(ROOT/'experiments/task45_preregistration_v1.json',p)
if __name__=='__main__':main()
