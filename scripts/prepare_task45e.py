"""Prepare bounded, C1-anchored contrasts after the45d identity audit."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    p=copy.deepcopy(json.loads((ROOT/'experiments/task45d_preregistration_v1.json').read_text()))
    p.update(task='45e',schema_version='45e-C1-anchored-v1',coupled_arms=['none','C0','C1','P','N00','N01','N10','N11','Iz','Iu','J','R41','R111'])
    p['fitting'].update(fit_seeds=[9455103,9455117,9455129],dev_seeds=[9455207,9455219,9455231],coefficient_seed=9455001,checkpoints=[1,4,16],max_whitened_step_norm=2.)
    p['matrix']=dict(support_seed=9455031,arms={k:dict(gate=9,joint=False,support=int(k[1]),path_guard=bool(int(k[2]))) for k in ('N00','N01','N10','N11')},
        composition='P: transplant only existing Nap rate-head column26 from archived45dNline onto45cJ. All other learned tensors/columns bitwise preserved. No training of P; opened evidence, not independent confirmation.',
        Ih='Iz/Iu: matched causal FIT histories, same path gate/current loss without crossing-neighbor loss; rate head only versus both existing heads. Initialized atC1, all other columns immutable.',
        nomination='Whole candidate on common fresh DEV union(previous+matched support), no baseline-passing identity may become failing. Rank original target pass count then worst severity, earlier budget and fixed arm order. C1 fallback. Opened coupled/stress never nominate. No mixing candidates after nomination.',
        optimizer='Same original45c whitened causal refinement and ridge, cap2, snapshots1/4/16. No requirement that budget16 proves convergence; record requested step norms. New seeds for fitting and development. No shared-trunk training.')
    p['matrix']['arms'].update(Iz=dict(gate=10,joint=False,support=1,path_guard=True),Iu=dict(gate=10,joint=True,support=1,path_guard=True))
    p['hypotheses']=[dict(id=h,contrast=q) for h,q in [
        ('H1','P versus45dNline andC1:18 Ca_LVAst regressions due to discardedC1 heads, not Nap interference. Check actual tensor lineage and oldDEV case identities.'),
        ('H2','N10-N00 andN11-N01: matched FIT support at fixed4096 coefficient rows; new variable-voltage DEV, not coefficient loss, decides transfer.'),
        ('H3','N01-N00 andN11-N10: actual history gate/current guard versus coefficient-only objective distinguishes objective conflict.'),
        ('H4','Iz-Iu: Ih rate-only versus inf+rate refinement on same histories distinguishes kinetic versus equilibrium interference. No repeated signed-crossing loss.'),
        ('H5','Each selected N arm andP, C1,J dual versus shared_features and donor_pruned: measure actual repaired models, not extrapolate45dC1 timing.')]]
    p['fitting']['nomination']=p['matrix']['nomination'];p['fitting']['objective']='45e matrix definitions; all materializations anchored to frozen45cJ, never f011.'
    p['execution_scope']='Same18 states,64features,FP32 recurrence,dt0.1ms,physical readout,membrane,synapses. Execution variants require bitwise raw parity at1/2/64/176/642/4096; speed descriptive and separate from quality.'
    p['runtime_budget']='One privateT4 worker10800s, outer12000s; same notebook all paired arms, seed solves batched. Persist each checkpoint/DEV, each coupled geometry/mode, stress and cost; no automatic restart.'
    p['limits']+=['45dDEV and coupled opened.45e new seeds are exploratory same-family development, not independent release qualification.','P isolates archived-head composition; cannot erase45d failures or count reused audit as fresh confirmation.','Untargeted columns, donors and trunks must remain bitwise equal toC1. Exact execution tests are bounded preflight evidence.']
    write_json(ROOT/'experiments/task45e_preregistration_v1.json',p)
if __name__=='__main__':main()
