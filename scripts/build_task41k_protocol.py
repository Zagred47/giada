"""Seal repair/compression diagnostics without modifying prior contracts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41k_learning_alignment as k


def main():
    folder=ROOT/'experiments/results/task41k_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'kaggle_active_launch.json').exists()
    old=json.loads((ROOT/k.j.CONFIG).read_text());spec=dict(old)
    spec.update(task='41k',arms=k.arms(),distillation_updates=0,
        new_seeds=dict(fit=701101,selection=701103,development=701107,stream=701109,feedback=701113,feedback_stream=701127,intervention=701137,support=701139,selection_support=701149),
        anchors=8192,selection_anchors=128,batch=64,fit_paths=128,selection_paths=16,tail_batch=4,
        feedback_histories=32,selection_feedback_histories=8,feedback_anchors=128,selection_feedback_anchors=8,feedback_batch=2,
        checkpoints=[0,16,64,256,1024],lr=1e-5,lr_min_ratio=.1,
        registered_comparisons=[dict(name='coefficient_vs_dynamics',control='K_point',treatment='K_dyn_uniform'),
          dict(name='old_weighting_vs_dynamics',control='K_oldmix',treatment='K_dyn_uniform'),
          dict(name='retention_uniform',control='K_dyn_uniform',treatment='K_dyn_uniform_retain'),
          dict(name='retention_targeted',control='K_dyn_target',treatment='K_dyn_target_retain'),
          dict(name='targeting_without_retention',control='K_dyn_uniform',treatment='K_dyn_target'),
          dict(name='targeting_with_retention',control='K_dyn_uniform_retain',treatment='K_dyn_target_retain'),
          dict(name='compression_objective',control='K_copy_coeff',treatment='K_copy_dynamic')],
        training_policy='Six repair arms initialized exactly at frozen H; two compact64 compression arms initialized identically from its base trunk. Compression targets H only, never canonical repair. Same18occupancies,V/Ca dependencies,dt0.1ms and physical readout.',
        support='Disjoint new FIT/selection/development seeds. Shared atomic triples,128 imposed1200ms FIT paths and128 coupled FIT anchors; heldout16paths/8anchors. Targeted arms sample50%uniform+50%topFIT-errorquartile measured with frozen H across seeds, using the same inverse-CDF uniform draws. Targeting never reads opened tests.',
        objective='Repair: coefficient-only; prior sparse coefficient+dynamic ratio0.1 every16; dynamic-only everyupdate in2x2(targeted x retention). Dynamic loss uses identical full-path current/offset margins plus40-tick coupled causal difference. Retention unit-weight parent matching on lower-halfFIT-discrepancy query cases per seed,NOT a claim these trajectories passed complete gates. Compression: parent coefficient-only versus parent dynamic-only at identical initialization/budget; no canonical target gradient. All optimizers Adam,lr1e-5 decays10x,per-seed norm clip1.',
        selection='Repair earliest per-seed best original lexicographic development rank. Compression earliest best heldout parent-dynamics error then parent-coefficient error, not canonical repair rank. Nominee from repair arms only; all8arms fully evaluated after freeze. Complete threshold predicates unchanged.',
        budget='1024updates x8arms x3vectorizedseeds; checkpoints0/16/64/256/1024. Same update count not same compute: point arms cheaper than dynamic. Causal repair2x2 shares identical compute and streams apart from designated objective/mapping. No long-run scaling/exhaustion claim. Phase64/256/1024+evaluation; checkpoint0 retained. No automatic extension.',
        diagnostics='Initial FIT coefficient/dynamic gradient cosine and normalized finite-difference steps1e-5/1e-4/1e-3 with discarded clones. Every checkpoint records canonical and parent-dynamic heldout errors separately. Report snapshot0 selection explicitly; zero learning not inferred from final equality. All originals stress/historical/recurrence/whole-operator costs.',
        hypotheses=[dict(id='objective_conflict',statement='Coefficient improvement can oppose dynamic fidelity; dynamic-first optimization can preserve/improve H where prior weighted objective regresses.',falsifier='No opposing finite-difference effect or no heldout benefit of dynamic-first at matched initialization; gradient bugs,conditioning,budget remain alternatives.'),
          dict(id='forgetting',statement='FIT parent replay can protect already-low-error behaviors during canonical dynamic repair.',falsifier='No reduction of introduced heldout failures versus paired no-retention arm,or unresolved target errors simply remain frozen.'),
          dict(id='coverage',statement='FIT-only targeted sampling can repair sensitive families under fixed data/update budget.',falsifier='No heldout gain versus uniform with and without replay; not evidence that more data can never help.'),
          dict(id='compression_objective',statement='A compact model can better copy H dynamics when trained on those dynamics rather than coefficient agreement.',falsifier='No improved heldout parent fidelity at checkpoints; capacity/budget/conditioning not distinguished by this arm alone.')],
        limits=['Diagnostic repair and compression separated; no guaranteedGO or Task42authorization.',
          'Opened scientific panels are regression diagnostics,not independent confirmation. No threshold/abstraction change.',
          'Retention masks and targeting quantiles are FIT training choices,not altered evaluation tolerances. Lower-half query error does not certify full trajectory correctness.',
          'Compression copying H inherits H failures; no requirement to fix them in this experiment. Report canonical failures and parent fidelity separately.',
          'Equal updates are not equalFLOPs; all walltimes reported.3seeds are not independent biological replicates.',
          '1024updates are a diagnostic budget; negative result not proof of representational impossibility.',
          'DB stays local; phase archives externally SHA/CRC verified before continuation.'])
    paths=list(old['sources'])+[k.j.CONFIG,'src/giada_teacher/task41k_learning_alignment.py','src/giada_teacher/task41k_evaluation.py',
      'scripts/run_task41k_learning_alignment.py','scripts/task41k_gpu_probe.py','scripts/task41k_kaggle_pipeline.py','scripts/build_task41k_protocol.py','tests/test_task41k_learning_alignment.py']
    spec['sources']={name:k.p.digest(ROOT/name) for name in dict.fromkeys(paths)}
    k.p.save_json(ROOT/k.CONFIG,spec)
    intro='# GIADA Task41k — Riparazione e compressione separate\n\n'+spec['training_policy']+'\n\n'+spec['objective']+'\n\n'+spec['selection']+'\n\n'+spec['budget']+'\n\nSoglie e astrazione immutate. Backup DB soltanto locale.\n'
    code="import os,subprocess,sys\nfrom pathlib import Path\nREPO=Path(os.environ.get('GIADA_REPO','/kaggle/working/giada'))\nOUTPUT=Path(os.environ.get('GIADA_OUTPUT','/kaggle/working/artifacts/giada_task41k'))\nassert not OUTPUT.exists()\nsubprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41k_learning_alignment.py'),str(OUTPUT),'--stop-after','1024'],cwd=REPO,check=True)\n# Production console adds immutable revision,verified resume and evaluation-only phase.\n"
    evaluation="subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41k_learning_alignment.py'),str(OUTPUT),'--resume-local','--evaluation-only'],cwd=REPO,check=True)\n"
    k.p.save_json(ROOT/'notebooks/giada_task41k_learning_alignment.ipynb',dict(cells=[dict(cell_type='markdown',id='design',metadata={},source=intro.splitlines(True)),dict(cell_type='code',id='run',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),dict(cell_type='code',id='frozen-evaluation',metadata={},execution_count=None,outputs=[],source=evaluation.splitlines(True))],metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    print(json.dumps(dict(valid=True,arms=8,members=24,thresholds_changed=False)))

if __name__=='__main__':main()
