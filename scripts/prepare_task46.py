"""Task46 spatial-intervention preregistration; frozenP, no new payload upload."""
import copy,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    from src.giada_teacher import task45e_anchored_repair as e
    import torch
    old=json.loads((ROOT/'experiments/task45e_preregistration_v1.json').read_text());prior,cfg,ca,base=e.n.contract(ROOT)
    paths=[ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin']
    sources=[json.loads((ROOT/name).read_text()) for name in ('experiments/task45c_cost_transport_v2.json','experiments/task45d_evaluation_recovery_v1.json')]
    states=[]
    for path,plan,member in zip(paths,sources,('J.pt','Nline.pt')):
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==plan['sha256']
        with zipfile.ZipFile(path) as z:
            raw=z.read(member);assert hashlib.sha256(raw).hexdigest()==plan['members'][member];states.append(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True))
    joined=copy.deepcopy(states[0])
    for key in ('parent.c','parent.d'):
        if key.endswith('.c'):joined[key][:,:,26].copy_(states[1][key][:,:,26])
        else:joined[key][:,26].copy_(states[1][key][:,26])
    with zipfile.ZipFile(ROOT/'artifacts/task45e_kaggle_external/task45e_final.zip') as z:
        saved=z.read('P.pt');expected=torch.load(io.BytesIO(saved),map_location='cpu',weights_only=True);nomination=z.read('nomination_freeze.json')
    assert joined.keys()==expected.keys() and all(torch.equal(joined[k],expected[k]) for k in joined)
    freeze=dict(candidate='45e/P',logical_weights_sha256='d2c3c77a1d4eff5ee84c2fd200746f98c468d62649823e71bfb94a1d13c9b256',source45e_sha256='16cf035671bdc4fd877a381db3bfdbf3a7a448e6d075bfb811847f41730180a8',saved_P_member_sha256=hashlib.sha256(saved).hexdigest(),nomination_sha256=hashlib.sha256(nomination).hexdigest(),all_saved_tensors_bitwise_equal=True,recipe='Load45cJ; copy ONLY parent.c[:,:,26] and parent.d[:,26] from45dNline. Other tensors immutable. Not training or model selection.',inputs=[dict(filename=plan['filename'],dataset=plan['dataset'],sha256=plan['sha256'],member=member,member_sha256=plan['members'][member]) for plan,member in zip(sources,('J.pt','Nline.pt'))])
    p=dict(task='46',version='v1',exploratory=True,training_updates=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,dt_ms=.1,duration_ms=1000.,prefix_horizons_ms=[100.,500.,1000.],seeds=[17,29,43],coupling=[0.,.25],arms=['none','C1','P','R41'],geometries=[copy.deepcopy(old['geometries'][0]),copy.deepcopy(old['geometries'][-1])],quality=copy.deepcopy(old['quality']),compartment_gates=copy.deepcopy(old['compartment_gates']),passive_gates=copy.deepcopy(old['passive_gates']),freeze=freeze,reference41r=old['reference41r'],cost=dict(ticks=80,repetitions=30,batches=[1,642,4096]),teacher_revision='074c4666300a8ad246601dab179a97a6942f0f29',
        factorial=dict(stimulus_sites=['proximal','distal'],forms=['burst','hyper_rebound'],densities=['uniform','graded'],coupling=[0.,.25],blocks=['none','sodium','Ih'],blocked_subset='burst+graded only, both sites and coupling values; unblocked matched counterparts present',cases_per_geometry=24,information='Same input, initial state, conductance multipliers, physical geometry and injection for canonical/native/C1/P/R41. Only model seed differs across17/29/43.'),
        hypotheses=[dict(id=k,contrast=v) for k,v in [('H1','P vsC1/R41, imposed vsautonomous, disconnected vscoupled isolates inherited ionic approximation from feedback amplification.'),('H2','Proximal vsdistal pulse and sodium/Ih block versus matched unblocked cases: qualitative crossing/attenuation/delay changes, without assuming all protocols spike.'),('H3','Uniform vsgraded density on same physical protocol/site/coupling, no new fit: robustness of parameter-conditioned readout.'),('H4','Resident ionic, axial-solve-only, full coupled step and host/device copies separate execution costs. Same model class/dt, no speed claim from isolated component.')]],
        execution='Four frozen arm lanes ×3seeds; one canonical run pergeometry shared by imposed/autonomous. Native generation in isolated process beforeTorch evaluation. No source/native/DB payload upload; only existing private inputs.',
        cost_scope='Existing single-compartment ionic/integrated benchmark plus coupled component profiler. Component timings at fixed state are non-additive and not whole-model speedups. Local ionic port still computes canonical masked branches; report this explicit overhead, do not disguise it.',
        progression='User authorizes46exploratory despite incomplete45e.45e NOT passed,47not authorized. No47data/protocols accessed.',
        limits=['45e10/18blocks observed, stress/regression44/cost incomplete. P71/72 original,52/54partialhyper, Ih0.9ms and gate max0.04749 at-130mV persist; unchanged gates remain failed.','30saved45echeckpoints finite; P700DEV versus686C1, no newDEVfailures. Not independent confirmation.','Same18states/analytic readout/calcium/membrane/synapses and level-one abstraction. No full morphology/network validation.','Native passive/active floors remain authoritative: invalid references exclude scientific judgments, not silently waived.','No automatic release, biological equivalence, speedup or task47claim; all outcomes including partial/technical failure persisted.'])
    write_json(ROOT/'experiments/task46_preregistration_v1.json',p)
    write_json(ROOT/'experiments/results/task46_protocol_v1/reconstruction_receipt.json',dict(valid=True,freeze=freeze,no_new_payload_upload=True,training_updates=0,selection_recomputed=False))
    print('Task46 frozenP reconstructed bitwise from existing Kaggle inputs',flush=True)
if __name__=='__main__':main()
