"""Freeze Task40 diagnostic protocol; no training or changed qualification gates."""
import json
from pathlib import Path
from src.giada_teacher import task40_causal_reinsertion as t
ROOT=Path(__file__).resolve().parents[1]
_,cfg,calcium,_=t.t35.contract(ROOT)
quality=json.loads((ROOT/t.p.CONFIG).read_text())['quality']
paths=[t.p.CONFIG,t.j.CONFIG,'experiments/task35_full_local_compartment_preregistration.json',
       'experiments/results/task35_native_shadow_traces.npz','experiments/results/task35_final_report.json',
       'experiments/results/task39j_audited_v1/attribution_audit.json',
       'experiments/results/task39j_audited_v1/frozen_checkpoint.pt',
       'experiments/results/task39j_audited_v1/selection_freeze.json']
paths += ['experiments/results/task39j_audited_v1/recurrence_evaluation.json']
oldspec=json.loads((ROOT/t.j.CONFIG).read_text())
paths += [oldspec['dataset']+'/stress_diagnostic__'+family+'.npz' for family in ('same_v_history','activation_boundary')]
spec=dict(task='40',version='v1',dt_ms=.1,seeds=[17,29,43],sources={p:t.p.digest(ROOT/p) for p in paths},
    current_quality=quality,compartment_gates=cfg['gates'],
    interventions=['block_Ca_HVA','block_Nap_Et2','block_K_Pst','block_SK_E2','zero_synapses','injection_0.99','injection_1.01'],
    arms=dict(models=['I3','J1'],families=t.FAMILIES,modes=['imposed','autonomous']),
    hypotheses=['H1: isolated current precision survives autonomous V/Ca feedback',
                'H2: failure localizes to a channel family rather than a common solver floor',
                'H3: learned calcium-only SK head remains valid when Ca is computed causally',
                'H4: removal and small current interventions preserve differential responses',
                'H5: quality improvement does not imply runtime speedup'],
    known_stress=[dict(family='same_v_history',case=1,seed=17,channel='Ca_LVAst',failure='offset',observed_ms=.5,limit_ms=.3),
                  dict(family='activation_boundary',case=0,seed=29,channel='Ca_HVA',failure='current_nrmse',observed=.01178899,limit=.01)],
    stress_policy='Replay both full families for I3/J1/all seeds. These previously opened failures are not waived or used for selection. A causal-domain pass is not an all-domain pass.',
    design='32 historical native Task35 fixtures + 7 paired interventions on 8 canonical fixtures = 88 cases/model/seed/mode. Same fixed initial gates, currents and exogenous presynaptic conductances. Family masks are attribution controls; only all-learned J1 is the candidate.',
    phase='Old V -> learned or canonical voltage gates -> analytical I_Ca -> unchanged CaDynamics -> learned/canonical SK from NEW Ca -> analytical currents -> unchanged semi-implicit membrane/NMDA solver. No future V/Ca in autonomous mode.',
    decisions='Technical floors first. All-learned J1 must satisfy unchanged per-channel and state criteria, existing Task35 pooled/worst V/Ca criteria, paired intervention current criteria and spike count/timing checks. No averaging away failures. Only preparation of independent Task41 can be authorized; no deployment or independent confirmation.',
    spike_metric='Upward zero-mV crossings, exact count matching, ordered timing <= existing 0.3ms onset/offset threshold. Report no-event cases explicitly; not a complete burst or bifurcation validation.',
    cost=dict(batches=[642,1,4096],ticks=80,repetitions=30,minimum_paired_saving=.1,
              baseline='Resident CUDA Graph formula, same FP32 state and readout; J1 retains audited FP64 correction. Local diagnostic, not complete competitive LUT/compiled benchmarking.',
              limits='No general speedup claim until competitive LUT and compilation matched baselines also pass quality. Three seed replicas per batch; report actual sites=3*batch. Wall and device timings paired; no transfer within timed block.'),
    limits=['Historical opened native fixtures; not new independent data.','Intervention references are calibrated formula, not fresh NEURON replays.',
            'Presynaptic shadow conductances are fixed exogenous input; NMDA voltage block recomputed causally.',
            'No training, no model selection, no threshold relaxation.','Full morphology, new seed confirmation and exhaustive bifurcation/burst coverage remain outside this diagnostic.',
            'Both stress failures remain unresolved unless reproduced differently and audited.','No claimed acceleration from this restricted baseline.'])
t.p.save_json(ROOT/t.CONFIG,spec)
cells=[dict(cell_type='markdown',metadata={},source=['# GIADA Task 40 — Reinserimento causale congelato\n',spec['design']+'\n',spec['phase']+'\n','Soglie immutate; nessun training. Stress noti riportati separatamente, non cancellati. Costi diagnostici, nessuna promessa di speedup.\n']),
       dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['import os,sys,subprocess,json\n','from pathlib import Path\n',"repo=Path(os.environ['GIADA_REPO']); output=Path(os.environ['GIADA_OUTPUT'])\n","assert not output.exists()\n","subprocess.run([sys.executable,'-u',str(repo/'scripts/run_task40_causal_reinsertion.py'),str(output)],cwd=repo,check=True)\n"]),
       dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["report=json.loads((output/'final_report.json').read_text())\n","print({k:report.get(k) for k in ('valid','decision','task40_causal_passed','task41_preparation_authorized','training_performed','thresholds_changed')})\n"])]
t.p.save_json(ROOT/'notebooks/giada_task40_causal_reinsertion.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
print('Task40 protocol/notebook generated')
