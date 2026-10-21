"""Preregister fresh data and frozen B6 before any Task41 outcome exists."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41_independent_confirmation as c
t,p=c.t,c.p
folder=ROOT/'experiments/results/task41_protocol_v1';folder.mkdir(exist_ok=True)
assert not (folder/'dispatch_request.json').exists(),'Do not rewrite a dispatched protocol'
old,cfg,calcium,base=t.contract(ROOT);teacher=json.loads((ROOT/'experiments/task35_full_local_compartment_preregistration.json').read_text())
panels=dict(canonical=[1.]*11,
    calcium_mix=[2.5,3.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
    mixed_diagonal=[.65,1.75,1.25,.8,1.4,.75,1.8,.6,1.1,1.6,.9],
    low_mixed=[.55,.7,.65,.85,.9,.6,.75,.8,.6,.7,.9])
events=dict(exc_1=[4.25,4.55,11.25,20.25],exc_2=[4.35,4.65,11.35,20.35],inh_1=[7.25,7.55,15.25,24.25],inh_2=[7.35,7.65,15.35,24.35])
schedules={name:{k:[x+shift for x in vv] for k,vv in events.items()} for name,shift in [('delayed_doublet',0.),('rebound_burst',.4),('staircase',.8),('rapid_train',1.2)]}
schedules['calibration']=dict(exc_1=[3.6,13.6],exc_2=[4.6,14.6],inh_1=[6.6,16.6],inh_2=[7.6,17.6])
paths=['experiments/results/task40b_audited_v1/frozen_checkpoint.pt','experiments/results/task40b_audited_v1/audit.json',
    'experiments/results/task40b_audited_v1/selection_freeze.json',t.CONFIG,p.CONFIG,
    'src/giada_teacher/task41_independent_confirmation.py','src/giada_teacher/task40_causal_reinsertion.py',
    'src/giada_teacher/task40b_calcium_precision.py','src/giada_teacher/task33_observable_synaptic_feedback.py']
paths += ['experiments/results/task41_protocol_v1/old_native_phase_preflight.json']
spec=dict(task='41',version='v1',candidate='B6',checkpoint=paths[0],model_seeds=[17,29,43],native_seeds=[104117,104129],
    calibration_seed=104003,stress_seeds=[104143,104159,104173],initial_voltages_mv=[-77.,-56.],
    dt_ms=.1,duration_ms=40.,stress_duration_ms=1200.,panels=panels,protocols=list(schedules)[:-1],schedules_ms=schedules,
    interventions=old['interventions'],current_quality=old['current_quality'],compartment_gates=old['compartment_gates'],
    neuron_version=teacher['neuron_version'],teacher_revision=teacher['teacher_revision'],
    sources={path:p.digest(ROOT/path) for path in paths},
    cost=dict(batches=[1,642,4096],ticks=80,repetitions=30,minimum_paired_saving=old['cost']['minimum_paired_saving'],
        baselines=['resident formula CUDA Graph','0.01mV voltage LUT plus canonical calcium-dependent SK'],
        timing='Same active native initial states at10ms,80resident future input ticks,no hosttransfer,alternatingorder;CUDAandwalltimes;pairedbootstrap uncertainty'),
    design='64fresh native basecases(4protocolsx4panelsx2inputseedsx2initialV)+112new native pairedinterventions=176cases;3frozenmodelseeds;12fresh1200msimposedinputstresscases=36case-seedchecks.',
    phase=old['phase'],
    hypotheses=['H1: frozen B6 generalizes to fresh native input histories and unseen conductance panels',
        'H2: matched blockade and small-injection response fidelity survives causal V/Ca feedback',
        'H3: equilibrium and slow-relaxation precision survives new long conditioned histories',
        'H4: accuracy qualification and speed qualification are distinct; compare matched formula and LUT costs'],
    qualification='Require admissible native/shadow and FP32/64 formula floors, all unchangedB6 per-case trajectory/current/state/V/Ca/spike gates inbothmodes,allautonomousinterventiongates,andallfreshstressgates.Noaveragingawayfailures.NoTask42withoutaudit.',
    native_current_policy='Staggered operator reference: native voltage gates at t, native SK at t+dt,old native V. Proven old-fixture one-step NMODL identity uses V_next/Ca_next; direct endpoint gates are not the same interface. Derived current is not endpoint h.ica. Require unchanged quality gate on aligned formula/native floor before B6 predictions; otherwise model not judged.',
    freeze='OnlyB6,allthreeoriginalmodelseeds17/29/43;nooptimizer,checkpointselection,calibrationfittingortraining. Newseedsrefer toinputprotocol/RNG,notnewtrainedmodels.',
    limits=['Single compartment,40mscoupled and1200msimposed stress;notwholecell,morphologyorexhaustivebifurcations.',
        'B6 selectedposthoc onopenedTask40bdiagnostics;Task41data notusedforselection.',
        'Analyticalcheapcurrentreadout,calcium,membraneandsynapticshadow retained;all18gateupdateslearnedinB6.',
        'Quality GO doesnotimplyspeedGO;timinguncertaintyononedeviceisnotcross-devicereplication.',
        'LUT iscomparisononly,notalternatecandidate;clampedlookuprequiresnoexitsforcostqualification.',
        'Freshmodeloutcomesmustnottriggerretraining,restartsorthresholdrelaxation.','DBbackupremainslocal;runlogsandartifactsrecoveredoffVMtothePC.'])
p.save_json(ROOT/c.CONFIG,spec)
case_rows=c.design(spec);p.save_json(folder/'planned_cases.json',case_rows)
cells=[dict(cell_type='markdown',metadata={},source=['# GIADA Task41 — conferma indipendente di B6\n',
    'B6 congelata,treseedmodello originali; nuovi seed dei dati e nuovi protocolli NEURON. Nessuntrainingosoglia modificata.\n',
    '176nuove simulazioni native,controlli imposti/autonomi,LUT eformula,stress1200ms,costiappaiati. BackupDBsolo locale.\n']),
    dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['import os,sys,subprocess,json\n','from pathlib import Path\n',
        "repo=Path(os.environ['GIADA_REPO']);output=Path(os.environ['GIADA_OUTPUT'])\n",
        "subprocess.run([sys.executable,'-u',str(repo/'scripts/run_task41_independent_confirmation.py'),str(output)],cwd=repo,check=True)\n"]),
    dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["report=json.loads((output/'final_report.json').read_text())\n",
        "print({k:report.get(k) for k in ('valid','decision','quality_passed','model_judged','native_cases','stress_passed','training_performed','thresholds_changed','task42_authorized')})\n"])]
p.save_json(ROOT/'notebooks/giada_task41_independent_confirmation.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
print(json.dumps(dict(task=41,cases=len(case_rows),checkpoint_sha=spec['sources'][spec['checkpoint']],config_sha=p.digest(ROOT/c.CONFIG))))
