"""Generate a source-bound Task41i protocol and one notebook, before training."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41i_selective_repair as i

def main():
    out=ROOT/'experiments/results/task41i_protocol_v1';out.mkdir(exist_ok=True)
    assert not (out/'dispatch_request.json').exists()
    archive=ROOT/'artifacts/task41h_complete_evaluation_external/recovery_final_merged.zip'
    receipt=json.loads((ROOT/'experiments/results/task41h_recovery_v1/memory_receipt.json').read_text(encoding='utf-8'))
    assert receipt['valid'] and receipt['DB_backup']['valid']
    with zipfile.ZipFile(archive) as z:
        freeze=json.loads(z.read('selection_freeze.json'));weights=z.read('frozen_checkpoint.pt')
    assert hashlib.sha256(weights).hexdigest()==freeze['checkpoint_sha256']
    checkpoint=out/'task41h_frozen.pt';checkpoint.write_bytes(weights)
    old=json.loads((ROOT/i.h.CONFIG).read_text(encoding='utf-8'));spec=dict(old)
    spec.update(task='41i',version='v1',arms=i.arms(),contrasts=[list(x) for x in i.contrasts()],
        task41h_checkpoint=checkpoint.relative_to(ROOT).as_posix(),task41h_checkpoint_sha256=freeze['checkpoint_sha256'],task41h_nominee=freeze['nominee'],
        new_seeds=dict(fit=671101,selection=671103,development=671107,stream=671109,feedback=671113,feedback_stream=671127,intervention=671137,support=671139,selection_support=671149),
        anchors=8192,selection_anchors=64,fit_paths=512,selection_paths=32,tail_batch=8,
        training_policy='24 vectorized members; frozen H nominee; only new Ca.m/Nap.m equilibrium residuals learnable; all z and other16 u protected; Nap mask is factorial.',
        support='Same paired FIT paths for every arm. 512 independent constant/ramp prefixes, full1200ms targets, same union of fixed times and both crossing definitions. FIT-only adaptive allocation: 50%uniform/50%sensitivity quantiles; 128 coefficients/head in both bases. No opened path/seed fitting.',
        objective='Same censored coefficient,0.01 zero-residual preservation,4ms feedback and common full-path current loss. Factor: auxiliary timing uses asymptotic-tail vs official first-reference/full-path crossing. Shared query union and minibatches; auxiliary every16updates, per-member gradient norm ratio0.1; frozen coefficients have zero gradients.',
        selection='Original global rank and all original thresholds; candidate chosen on newdevelopment only, per-head severity additionally saved for informative contrasts. Earliest ties. No opened regressions for selection.',
        budget='60000 Adam updates with independent moments/clipping; checkpoints0/512/2048/8192/30000/60000. Atomic64, preservation64, paths8, feedback4x40; matched streams. FIT-only projection at1024/4096 anchors is diagnostic and never installed.',
        diagnostics='Causal reachability and positive/negative metric controls before training. Full original panels after freeze. Frozen H/finalist precision: FP32, selective Vstate64+solver64, FP32 increment-coordinate compensated solve. Exact execution/refactor diagnostics compared with full-quality gates; no oracle selectable.',
        speed='Original batches1/642/4096 and30paired repetitions. Profile whole learned chain; compare legacy versus compiled whole graph, retaining separate nonlinearities. Report compile failures explicitly; quality qualification required before any speedup claim.',
        hypothesis_links=[dict(code='claims-giada-task41h-result-'+x+'-v1') for x in ('selective_nap_equilibrium','crossing_objective_alignment','local_resolution_vs_optimization','coupled_voltage_precision','whole_chain_execution')],
        limits=old['limits']+['41i is post-hoc repair of opened diagnostics; not independent confirmation.','All global thresholds/reference preserved; per-head contrasts cannot waive failed global cases.','Numerical and execution sidecars remain nonselectable until full qualification; no automatic promotion.'])
    notebook='notebooks/giada_task41i_selective_repair.ipynb'
    intro='''# GIADA Task 41i

Otto bracci appaiati e tre seed: Nap.m congelato/addestrabile × loss timing attuale/ufficiale × base uniforme/adattata soltanto su FIT.

Genitore 41h congelato, correzioni iniziali nulle, 18 stati e passo 0,1 ms invariati. Tutti gli z e gli altri coefficienti protetti. Il crossing ufficiale usa il primo campione del riferimento e la massima deviazione dell'intero percorso. Nessun cambio di soglie o riferimento.

Controlli prima del training, mini scaling sulla stessa traiettoria, proiezione FIT solo diagnostica. Dopo il freeze: tutti i pannelli originali, regressioni, precisione ed esecuzione. Gli oracle non sono candidati. Backup sperimentali esterni via supervisore console; DB e backup DB soltanto locali.
'''
    code="import os,subprocess,sys,json\nfrom pathlib import Path\nREPO=Path(os.environ.get('GIADA_REPO','/content/giada'))\nOUTPUT=Path(os.environ.get('GIADA_OUTPUT','/content/giada_task41i_result'))\nassert not OUTPUT.exists()\nsubprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41i_selective_repair.py'),str(OUTPUT)],cwd=REPO,check=True)\n"
    cells=[dict(cell_type='markdown',metadata={},source=intro.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))]
    for k,cell in enumerate(cells):cell['id']=f'task41i-{k}'
    i.p.save_json(ROOT/notebook,dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    paths=list(old['sources'])+[i.h.CONFIG,checkpoint.relative_to(ROOT).as_posix(),'src/giada_teacher/task41i_selective_repair.py','src/giada_teacher/task41i_evaluation.py','scripts/run_task41i_selective_repair.py',notebook]
    spec['sources']={n:i.p.digest(ROOT/n) for n in dict.fromkeys(paths)}
    i.p.save_json(ROOT/i.CONFIG,spec)
    print(json.dumps(dict(valid=True,arms=8,members=24,thresholds_changed=False)),flush=True)

if __name__=='__main__':main()
