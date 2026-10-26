"""Seal the operator-reset experiment before observing new results."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41j_operator_reset as j


def main():
    folder=ROOT/'experiments/results/task41j_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'kaggle_active_launch.json').exists()
    old=json.loads((ROOT/j.prior.CONFIG).read_text());spec=dict(old)
    spec.update(task='41j',version='v1',arms=j.arms(),contrasts=[],
        new_seeds=dict(fit=691101,selection=691103,development=691107,stream=691109,feedback=691113,
                       feedback_stream=691127,intervention=691137,support=691139,selection_support=691149),
        anchors=16384,selection_anchors=128,batch=64,fit_paths=512,selection_paths=32,tail_batch=4,
        distillation_updates=4096,aux_every=16,feedback_histories=128,selection_feedback_histories=16,
        feedback_anchors=512,selection_feedback_anchors=32,feedback_batch=2,lr=1e-4,lr_min_ratio=1e-4,
        checkpoints=[0,512,2048,8192,30000,60000],
        training_policy='Three operators x three vectorized seeds. Unfreeze ALL parameters of H chain, versus a single shared voltage MLP at widths64/128 plus the same learned calcium-only SK head. All36 equilibrium/update outputs learnable. No new recurrent state, no extra causal input, no analytic kinetic inference.',
        support='Identical disjoint FIT/selection anchors, +/-voltage probes, constant/ramp 1200ms histories, and coupled perturbation histories for all arms; all11 currents supervised. New development seed; opened tests only after selection freeze.',
        objective='4096 equal-budget parent-distillation updates, then fresh Adam60000 canonical-refinement updates. Censored logit errors with training censor1e-12 on all36 coefficients plus paired probability secants. Full-path current and official offset margin across all11 channels, plus40-step coupled causal differences; auxiliary every16, per-seed gradient norm ratio0.1. No hard frozen coefficients or parent-residual bounds.',
        selection='Before opened tests, select earliest best per seed by (global failed development trajectories, introduced channel/gate regressions versus H, global worst severity, heldout path+coupled error, atomic error). Nominate one arm by worst-seed lexicographic rank. Failed gates are never waived; all arms reported, no posthoc nominee.',
        qualification=old['qualification'],
        budget='Same4096 warmup and60000 refinement updates per arm, same minibatches and counts; log warmup reconstruction gap, params, raw updates and wall time. Independent Adam moments and clipping per seed. Seeds within arms vectorized, heterogeneous arms inside one captured update. Width is a nested capacity probe, NOT a parameter-matched architecture claim.',
        diagnostics='Parallel affine-prefix scan versus sequential recurrence and gradients; chain exact initial map; compact approximation gap before/after warmup. All equilibria AND update coefficients reachable. Complete original scientific panels on all arms, same-panel H control. Frozen joint64 numerical audit for H/nominee only; no selection. Whole operator ionic and integrated paired timing at actual batches1/642/4096, formula/LUT/H baselines.',
        speed='Whole chain versus compact, no final-head-only timing. Same compiled-graph resident execution and FP32 primary reference. CUDA+wall samples and paired uncertainty; no speedup claim without full quality. Compiler parity diagnostic does not waive quality.',
        hypotheses=[dict(id='restricted_repair_space',statement='Jointly relearning the existing chain can repair errors unreachable or poorly conditioned under frozen z/most u.',falsifier='No reproducible reduction in matched development severity or failed cases despite reachable finite gradients; not proof of impossibility.'),
                    dict(id='cumulative_execution',statement='A single compact operator can remove accumulated correction overhead while retaining the same learned dynamics.',falsifier='No paired cost advantage, or compact quality floor remains worse than jointly trained chain.'),
                    dict(id='compact_capacity',statement='If64hidden units underfit the learned coefficient map,128 can reduce heldout reconstruction and trajectory errors at fixed data/update budget.',falsifier='No consistent heldout/coupled improvement; optimization and finite budget remain alternatives.'),
                    dict(id='numerical_seam',statement='Some autonomous response failures remain numerical rather than learned-map errors.',falsifier='Frozen joint64 policy does not resolve those cases; sidecars never nominate the primary arm.')],
        limits=['Post-hoc repair; previous opened panels are regression diagnostics, NOT independent confirmation.',
                'No change in abstraction,18states,dt0.1ms,causal inputs,readout,calcium/membrane/synapse equations or metric thresholds.',
                'Training loss,censoring,trainable subspace and selection rule change prospectively and are explicitly registered.',
                'Compact starts from H base trunk, not the full corrected function; reconstruction gap is measured, not assumed zero.',
                'Three seeds share protocols; no independent biological replicates. New independent confirmation required after repair.',
                'Equal update/example budget is not equal wall time or parameter count; both costs reported.',
                'No claim that a failed constrained repair disproves the abstraction, and no guarantee that this reset will solve all cases.',
                'DB and DB backup stay local; experimental phase archives copied off-runtime and verified before continuation.'])
    # Previous recovery remains independently source-bound by its own manifest.
    paths=list(old['sources'])+[j.prior.CONFIG,'experiments/task41i_evaluation_recovery_v1.json',
        'src/giada_teacher/task41j_operator_reset.py','src/giada_teacher/task41j_evaluation.py',
        'scripts/run_task41j_operator_reset.py','scripts/build_task41j_protocol.py',
        'scripts/task41j_kaggle_pipeline.py','scripts/task41j_gpu_probe.py','tests/test_task41j_operator_reset.py']
    spec['sources']={name:j.p.digest(ROOT/name) for name in dict.fromkeys(paths)}
    j.p.save_json(ROOT/j.CONFIG,spec)
    intro='''# GIADA — Task41j: riottimizzazione congiunta e operatore compatto

Tre candidati x tre seed, stesso livello di astrazione:18 stati, V/Ca causali,0.1ms, aggiornamento convesso. Correnti, calcio, sinapsi e membrana invariati.

Catena H interamente riottimizzata contro MLP unici64/128 con testa SK calcium-only.4096 update di distillazione appaiati, poi60000 update canonici. Tutti gli equilibri e tutti i coefficienti temporali apprendibili. Gap di compressione iniziale misurato; nessuna equivalenza presunta.

Soglie immutate. Selezione su nuovi development, regressioni e stress completi dopo freeze; nessuna promozione retrospettiva. Costi dell'intero operatore, non dell'ultima correzione. Il confronto di capacità non è parameter-matched.

Console Kaggle con fasi512/8192/30000/60000 ed evaluation-only. Ogni fase è conservata su Kaggle e scaricata/verificata esternamente prima di continuare. DB soltanto locale.
'''
    code="import os,subprocess,sys\nfrom pathlib import Path\nREPO=Path(os.environ.get('GIADA_REPO','/kaggle/working/giada'))\nOUTPUT=Path(os.environ.get('GIADA_OUTPUT','/kaggle/working/artifacts/giada_task41j'))\nassert not OUTPUT.exists()\nsubprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41j_operator_reset.py'),str(OUTPUT)],cwd=REPO,check=True)\n"
    cells=[dict(cell_type='markdown',id='task41j-design',metadata={},source=intro.splitlines(True)),dict(cell_type='code',id='task41j-run',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))]
    j.p.save_json(ROOT/'notebooks/giada_task41j_operator_reset.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    print(json.dumps(dict(valid=True,arms=3,members=9,thresholds_changed=False)),flush=True)


if __name__=='__main__':main()
