"""Preregister new confirmation before fresh teacher outputs are generated."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41c_frozen_confirmation as c

def main():
    folder=ROOT/'experiments/results/task41c_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists()
    archive=ROOT/'artifacts/task41b_primary_external/giada_task41b_result.zip'
    assert c.p.digest(archive)=='1d8ae46033437dfec9da72b47db01f8aa78f15fc31a17d8c57c3913c5e199cec'
    audited=ROOT/'experiments/results/task41b_audited_v1';audited.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        state=torch.load(io.BytesIO(z.read('frozen_checkpoint.pt')),map_location='cpu',weights_only=True)
        torch.save(state['P110'],folder/'P110_frozen.pt')
        for name in ('final_report.json','selection_freeze.json','paired_contrasts.json','cost.json'):
            (audited/name).write_bytes(z.read(name))
    audit=json.loads((ROOT/'artifacts/task41b_completion_audit.json').read_text())
    assert audit['freeze_hash_valid'] and all(r['finite'] for r in audit['checkpoint_finiteness'].values())
    c.p.save_json(audited/'audit.json',audit)
    old=json.loads((ROOT/c.x.CONFIG).read_text());oldlocal=json.loads((ROOT/c.local.CONFIG).read_text())
    names=['asymmetric_doublet','rebound_triplet','reversed_staircase','irregular_train']
    events=dict(exc_1=[3.15,3.85,10.15,19.15],exc_2=[3.35,3.95,10.45,19.45],inh_1=[6.15,8.05,16.15,25.15],inh_2=[6.45,8.35,16.45,25.45])
    schedules={name:{k:[v+shift for v in values] for k,values in events.items()} for name,shift in zip(names,[0.,.3,.9,1.5])}
    schedules['calibration']=old['schedules_ms']['calibration']
    checkpoint='experiments/results/task41c_protocol_v1/P110_frozen.pt'
    sources=[checkpoint,'src/giada_teacher/task41c_frozen_confirmation.py','scripts/run_task41c_frozen_confirmation.py',
        'src/giada_teacher/task41b_local_precision.py','src/giada_teacher/task41b_evaluation.py','src/giada_teacher/task41_independent_confirmation.py',c.local.CONFIG,c.x.CONFIG]
    spec={**old,'task':'41c','version':'v1','candidate':'P110','checkpoint':checkpoint,
        'native_seeds':[241117,241129],'stress_seeds':[241143,241159,241173],'calibration_seed':241003,
        'excluded_input_seeds':old['native_seeds']+old['stress_seeds']+[old['calibration_seed']]+[oldlocal[k] for k in ('fit_seed','selection_seed','development_seed','stream_seed')]+[85317,85329,17,29,43],
        'initial_voltages_mv':[-79.,-54.],'protocols':names,'schedules_ms':schedules,
        'panels':dict(canonical=[1.]*11,calcium_mix=[2.3,2.7,1.1,.9,1.2,.8,1.2,.9,1.1,.8,1.],
            mixed_diagonal=[.7,1.6,1.15,.85,1.3,.8,1.65,.7,1.2,1.45,.95],low_mixed=[.6,.75,.7,.9,.85,.65,.8,.75,.65,.8,.95]),
        'sources':{n:c.p.digest(ROOT/n) for n in sources},'quality':old['current_quality'],'seeds':[17,29,43],
        'training_performed':False,'model_selection_performed':False,
        'freeze':'Only P110 selected from opened Task41b; all original seeds17/29/43. No training, optimizer, calibration fitting, threshold changes or checkpoint selection. B6 and LUT descriptive controls, never fallback candidates.',
        'qualification':'All176x3 fresh native trajectories,112x3 paired responses in BOTH modes,12x3 new1200ms stress histories must pass unchanged gates. Require native/shadow, aligned current phase and FP32/64 floors before interpretation. Quality and speed decisions remain separate; Task42 only after detailed audit.',
        'design':'64fresh native base cases +112paired interventions=176;three original model seeds. Four new pulse families, new synaptic schedules, new RNG seeds, two new initial voltages, three new heterogeneous panels plus canonical. Two new long-history stress shapes,three seeds,two initial gate states=12histories.',
        'hypotheses':['Frozen P110 generalizes beyond opened41b inputs without fitting.','Paired current responses survive autonomous V/Ca feedback.','Nap.m equilibrium and NaTs.h/Im relaxation repairs survive new long histories.','Quality improvement does not imply speed improvement; benchmark frozen P110 vs B6,formula,LUT.'],
        'limits':['Independent prospective test conditional on fixed Hay single-compartment teacher,not independent biological evidence or exhaustive bifurcation coverage.',
            'No model/seed selection after opening;any failed registered case is scientific NO-GO if technical floors pass.',
            'All18learned gates and cheap analytic readout,Ca,membrane,synapses unchanged;no canonical kinetic fallback.',
            'Historical regressions already audited in41b;new41c is not a search over other arms. Task42 requires audit,not automatic authorization.',
            'Cost on one T4;quality and speed qualification separate. DB backup strictly local;run backups off runtime onPC.']}
    c.p.save_json(ROOT/c.CONFIG,spec)
    c.p.save_json(folder/'planned_cases.json',c.design(spec))
    code="import os,sys,subprocess,json\nfrom pathlib import Path\nrepo=Path(os.environ['GIADA_REPO']);output=Path(os.environ['GIADA_OUTPUT'])\nassert not output.exists()\nsubprocess.run([sys.executable,'-u',str(repo/'scripts/run_task41c_frozen_confirmation.py'),str(output)],cwd=repo,check=True)\n"
    cells=[dict(cell_type='markdown',metadata={},source=['# 🧪 GIADA Task41c — conferma indipendente P110\n','P110 congelata; nessun training o selezione.176nuove simulazioni native,36stress lunghi;controlli B6/formula/LUT. Soglie invariate;Task42 subordinata ad audit.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["print(json.loads((output/'final_report.json').read_text()))\n"])]
    c.p.save_json(ROOT/'notebooks/giada_task41c_frozen_confirmation.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    print(json.dumps(dict(valid=True,cases=len(c.design(spec)),candidate_sha256=c.p.digest(ROOT/checkpoint))))
if __name__=='__main__':main()
