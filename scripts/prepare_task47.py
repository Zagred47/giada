"""Prospective independent level-one confirmation; no fitting or selection."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    p=copy.deepcopy(json.loads((ROOT/'experiments/task46_preregistration_v1.json').read_text()))
    p.update(task='47',input_seeds=[9471103,9471127],coupling=[0.,.35],geometries=[
        dict(id='47_chain5',parents=[-1,0,1,2,3],lengths_um=[87.,123.,96.,141.,108.],diameters_um=[1.35,1.12,.93,.84,1.04],ra_ohm_cm=145.),
        dict(id='47_branch9',parents=[-1,0,0,1,1,2,2,4,6],lengths_um=[91.,117.,133.,82.,126.,104.,147.,98.,112.],diameters_um=[1.4,1.15,1.08,.9,.87,.95,.81,.78,.74],ra_ohm_cm=155.)])
    p['factorial'].update(coupling=p['coupling'],forms=['irregular_burst','long_rebound'],cases_per_geometry=24,blocked_subset='irregular_burst+graded only, both sites/coupling; same waveform for blocked/unblocked counterparts',information='Frozen model seeds17/29/43 are not new training seeds. Disjoint input RNG seeds9471103/9471127 generate new waveforms; two new physical geometries. Same inputs across all arms.')
    p['hypotheses']=[dict(id=k,contrast=v) for k,v in [
        ('H1','FrozenP generalizes to new geometry and input seeds under unchanged per-case gates; no candidate repair or post-hoc selection.'),
        ('H2','Imposed/autonomous and disconnected/coupled paired contrasts separate inherited ionic errors from feedback amplification.'),
        ('H3','Matched sodium/Ih block, input location and density perturbations preserve qualitative responses; report actual reference spiking support, never count absent events as demonstrated spike fidelity.'),
        ('H4','Unchanged resident/component cost benchmark tests reproducibility of46cost frontier; distinguish ionic implementation overhead from axial solve and transfers, no speedup inferred from accuracy.')]]
    p['progression']='User approved proceeding after46. Independent47evaluation only; no automatic48launch or qualification of unfinished45e. Quality and cost verdicts remain separate.'
    p['independence']='Geometry parent arrays, lengths, diameters, resistivity and stochastic input waveforms differ from46. No new model seeds: same immutable trained ensemble. Input seed is fixed per geometry and shared across causal contrasts.'
    p['decision_rule']='Technical validity requires finite weights/traces, frozen hash, endpoint/null parity, valid native floors. Quality confirmation requires every registered eligible P current/compartment case at every horizon and mode to pass; zero eligible cases cannot pass. Failed/unsupported native or event cases stay explicit. Release/speedup not authorized; cost is a separate measured frontier. No thresholds waived by averages.'
    write_json(ROOT/'experiments/task47_preregistration_v1.json',p)
if __name__=='__main__':main()
