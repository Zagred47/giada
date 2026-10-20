"""Build prospective atomic precision diagnosis, without changing Task37 gates."""
import json
from pathlib import Path
from src.giada_teacher import task39j_atomic_relaxation as j
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/j.i.CONFIG).read_text())
spec.update(task='39j',version='v1',parent='experiments/results/task39i_recovered_v1',
    parent_sha256=j.p.digest(ROOT/'experiments/results/task39i_recovered_v1/frozen_checkpoint.pt'),
    checkpoints=[0,100,300,1000,3000,10000],anchors=1024,batch=128,lr=1e-4,
    fit_seed=390701,holdout_seed=390702,stream_seed=390703,
    matrix=j.arms(),paired_edges=j.contrasts(),
    architecture='Frozen I3 exceptJ8;256zero-initialized correction weights/member for joint u/z ofNap_Et2,K_Pst.32local hat features vs32global Fourier features with same parameter count. Correction bounded0.1in logit space.18states,learned recurrent map,cheap currentreadout retained. No teacher kinetics at inference.',
    objective='Coefficient logit-u/log-z precision plus bounded full held-relaxation shape/slow-slope current-opening loss.65logarithmic nodes from0.1to10000ms,shared first-tick origin,never local chunk origins. Auxiliary gradient calibrated to0.1ofcoefficient norm onFIT only and frozen. All arms have same all-gate coefficient preservation loss. Task37 actual discrete event metrics remain the qualification criteria.',
    support='Independent1024anchors:uniformV[-100,60],logCa[-7,-2],initialgates[0,1];sensitive treatment replaceshalf withV[-20,20] andloguniformgates[1e-4,1].Equalanchorcount,newseed,no evaluation V/state copied. Independent256atomic holdout anchors,diagnostic only,not selection. Same FIT RNG stream acrossarms andseeds.',
    optimization='J1/J2objective;J2/J3fixedfullbatchAdam vsstochastic128batch;J4/J2basis;J2/J5support;J5/J6fixedrefinementundernewcoverage;J2/J7perchannelvspermembergradientclipping;J2/J8parentfrozenvsunfrozen.24members,paired3seeds. Fullbatchhas8xexamples/update:reportexposureandwalltime,not a pureequalcompute optimizerclaim. J7is optimizercoupling control,not proof of representational independence. No LBFGS claim;deterministicAdam isfirst controlled refinement.',
    selection=spec['selection'],quality=spec['quality'],
    limits='39j remains withinTask39;Task40blocked until72/72andaudit. Development-informed hypotheses,not independent confirmation. No timestep/threshold relaxation. Frozenparentprotectsunrelatedchannels exceptexplicitJ8. Atomic held-input loss composes learnedcoefficients onlyduringtraining,finalinferenceusesordinaryFP32serialrecurrence. Same-levelabstraction;no restoredcanonicalrates. No definitive attributionfrom onecontrast;fixedbatchcomputeandbasisconditioningare reported.',
    preflights='Zero-init exactI3;finite gradients inclslownear-zerooccupancy;canonicalpowerschecked;isolated/vectorizedAdamparity;unrelatedchannelsunchanged;actualCUDA2updates;recurrenceparity;input/checkpointhashes. Savebeforeselection,externalconsolebackup. No scientific NO-GO ontechnicalfailure.',
    external_backup='Existing console manager and local supervisor: checkpoint/Adam/RNG beforeselection,immutableSHAverifiedZIPoutsideVM;finalZIPalsoDrive.')
j.p.save_json(ROOT/j.CONFIG,spec)
cells=[dict(cell_type='markdown',metadata={},source=[
    '# GIADA Task 39j — Precisione dei rilassamenti\n',
    'Otto bracci × tre seed, stessa funzione iniziale I3 e soglie Task37 immutate.\n',
    'Confronti: obiettivo, località, copertura, raffinamento deterministico, clipping e interferenza del parent.\n',
    'I dati atomici indipendenti chiariscono la learnability; la suite ricorrente completa decide la qualità.\n',
    'Full-batch e minibatch hanno esposizioni diverse: i confronti includono costo e budget, non soltanto aggiornamenti.\n']),
dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
    'import os,sys,subprocess,json\n','from pathlib import Path\n',
    "repo=Path(os.environ['GIADA_REPO']);output=Path(os.environ['GIADA_OUTPUT'])\n",
    "assert not output.exists(),'Usare una directory nuova'\n",
    "subprocess.run([sys.executable,'-u',str(repo/'scripts/run_task39j_atomic_relaxation.py'),str(output)],cwd=repo,check=True)\n"]),
dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
    "report=json.loads((output/'final_report.json').read_text())\n",
    "print({k:report[k] for k in ('valid','decision','passing_arms','thresholds_changed')})\n"])]
j.p.save_json(ROOT/'notebooks/giada_task39j_atomic_relaxation.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
print('39j prospective protocol and notebook generated')
