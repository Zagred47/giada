"""Generate one concrete preregistered calcium correction notebook."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if (ROOT/'experiments/results/task40b_protocol_v1/dispatch_request.json').exists():
    raise FileExistsError('Executed Task40b preregistration is immutable; create a new protocol revision.')
from src.giada_teacher import task40b_calcium_precision as b
old=json.loads((ROOT/b.t.CONFIG).read_text());parent=json.loads((ROOT/b.j.CONFIG).read_text())
sources=[b.t.CONFIG,b.j.CONFIG,'experiments/results/task40_audited_v1/audit.json',
         'experiments/results/task40_audited_v1/attribution_audit.json','experiments/results/task40_audited_v1/synthesis.json',
         'src/giada_teacher/task40_causal_reinsertion.py','src/giada_teacher/task39j_atomic_relaxation.py']
spec=dict(task='40b',version='v1',seeds=[17,29,43],dt_ms=.1,arms=[dict(name='B0',frozen=True)]+b.arms(),
    sources={name:b.p.digest(ROOT/name) for name in sources},current_quality=old['current_quality'],compartment_gates=old['compartment_gates'],
    fit_seed=40401,selection_seed=40402,stream_seed=40403,batch=128,lr=.001,checkpoints=[0,1000,3000,10000,30000],
    fit='Independent nested uniform voltage anchors: first1024of4096. Shared initial gates and paired V±epsilon; epsilon in0.05,0.2,1mV. No historical failed paths copied into FIT.',
    architecture='J1 frozen; learned bounded local residuals on Ca_HVA/Ca_LVAst u/z logits. Same18 recurrent states,0.1ms clock,cheap current readout,membrane andcalcium update. All promoted kinetics remainlearned.',
    trainable_parameters={a['name']:a['basis']*sum(b.mask(a)) for a in b.arms()},
    capacity_caveat='B6 u+z has twice the trainable outputs of B1; B7 has half. These are package/restriction contrasts, not capacity-matched pure causal attribution. B5 changes basis32to64 only.',
    loss='Coefficient logit-u/log-z MSE plus optional bounded paired-current residual at held ticks1,10,80,400. Paired-loss gradient scale calibrated once on initial B2 FIT and shared with B4. Censored target logit-u at1e-7,log-z floor1e-12 as39j; no metric threshold changes.',
    selection='Per arm/seed choose lowest common mean of coefficient and paired atomic selection losses normalized by frozen-parent values on256independent selection anchors; exact ties keep earliest. Task40 paths,72development traces andstress are evaluated only after freeze. Selection set is not fresh confirmation.',
    contrasts=[dict(axis='paired_objective',control='B1',treatment='B2'),dict(axis='paired_objective_dense',control='B3',treatment='B4'),
               dict(axis='coverage',control='B1',treatment='B3'),dict(axis='coverage_paired',control='B2',treatment='B4'),
               dict(axis='capacity',control='B1',treatment='B5'),dict(axis='joint_u_z_package',control='B1',treatment='B6'),
               dict(axis='m_only_restriction',control='B1',treatment='B7')],
    contrast_rule='Report per-seed paired change, median and worst severity, counts and checkpoint curves. No supported-cause claim from one seed or confounded package. Use unchanged39j severity comparison: median treatment/control<=0.8,each ratio<1,worst ratio<=1.05; report timing separately.',
    qualification='All72development case-seed checks andTask40 trajectory/current/state/V/Ca/spike gates inbothmodes; all168autonomous differential intervention checks. Imposed differential failures reported separately asTask40. Any candidate requires full audit beforeTask41 authorization.',
    stress_policy=old['stress_policy'],known_stress=old['known_stress'],
    execution='21 learned members vectorized with independent Adam moments and per-member norm1clipping; frozen parent coefficients precomputed. Eager training verified after runtime backward capture incompatibility; CUDA Graph retained for inference. Common random minibatch uniforms mapped to supportsize; exposure128/update/member. FP64 affine correction accumulation,FP32 deployed state. No new speedup claim.',
    limits=[x for x in old['limits'] if not x.startswith('No training, no model selection,')]+['Task40b performs training and atomic checkpoint selection as specified above; quality thresholds remain unchanged.','Posthoc-informed repair on opened diagnostic domains; independentTask41 still required.','No training on stress or historical failed trajectories.','Atomic sensitivity objective is a FIT surrogate, not the registered current evaluator.'])
b.p.save_json(ROOT/b.CONFIG,spec)
rows=['# 🧪 GIADA Task 40b — Precisione cinetica del calcio\n',
      'J1 congelato + sette varianti apprese. Soglie Task37/40 invariate.\n',
      '| Braccio | Confronto |\n|---|---|\n',
      '| B0 | J1 congelato |\n| B1 | Correzione z di m+h,32 basi,1024 anchor |\n',
      '| B2 | B1 + loss sulle risposte appaiate |\n| B3 | B1 con4096 anchor |\n| B4 | B2 con4096 anchor |\n',
      '| B5 | B1 con64 basi |\n| B6 | Correzione u+z; capacità maggiore dichiarata |\n| B7 | Correzione z solo m |\n',
      '\nTre seed per variante; checkpoint0/1000/3000/10000/30000. Selezione atomica indipendente, poi replay dei72development,interventi causali e due famiglie stress.\n',
      'I controlli con formule dell’audit40 restano diagnostici. La cinetica del candidato resta appresa.\n']
cells=[dict(cell_type='markdown',metadata={},source=rows),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
    'import os,sys,subprocess,json\nfrom pathlib import Path\n',
    "repo=Path(os.environ['GIADA_REPO']); output=Path(os.environ['GIADA_OUTPUT'])\n",
    "assert not output.exists()\n",
    "subprocess.run([sys.executable,'-u',str(repo/'scripts/run_task40b_calcium_precision.py'),str(output)],cwd=repo,check=True)\n"]),
    dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
    "report=json.loads((output/'final_report.json').read_text())\n",
    "print({k:report.get(k) for k in ('valid','decision','passing_arms','thresholds_changed','task41_authorized')})\n"])]
b.p.save_json(ROOT/'notebooks/giada_task40b_calcium_precision.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
print('Task40b protocol/notebook generated')
