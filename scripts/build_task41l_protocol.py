"""Seal Task41l before any scientific training; historical predicates copied."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41l_regression_attribution as l


def main():
    folder=ROOT/'experiments/results/task41l_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'kaggle_active_launch.json').exists(),'Protocol immutable after launch'
    source=ROOT/'artifacts/task41k_kaggle_external/v5/task41k_stage.zip'
    assert l.p.digest(source)=='ee629c24da75676e735670b4a0c5d22b223c932f9d394a4cd9a4ca738257e0c4'
    bundle=ROOT/'artifacts/task41l_frozen_input';bundle.mkdir(exist_ok=True)
    target=bundle/'task41k_frozen_diagnostic.zip.bin';members={}
    with zipfile.ZipFile(source) as old,zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as new:
        for member,name in [('frozen_checkpoint.pt','frozen_checkpoint.pt'),('selection_freeze.json','selection_freeze.json'),('selected_numerical_audit/voltage_precision.json','voltage_precision.json')]:
            raw=old.read(member);members[name]=hashlib.sha256(raw).hexdigest()
            info=zipfile.ZipInfo(name,date_time=(2026,10,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            new.writestr(info,raw)
    assert members['frozen_checkpoint.pt']=='4e3b457bfe609964b3480b8bc5b1e73279256cb9b863acd30a0b6974a3c09644'
    l.p.save_json(bundle/'dataset-metadata.json',dict(id='alessandrobelli/giada-task41l-frozen-diagnostics',title='GIADA Task41l Frozen Diagnostics',licenses=[dict(name='other')],description='Private immutable41k frozen coefficients and original numerical sidecar; diagnostic only.'))
    spec=json.loads((ROOT/l.k.CONFIG).read_text())
    comparisons=[]
    for a in l.arms():
        for axis,key in [('A','amplitude'),('D','coverage'),('P','protection')]:
            if a[key]==0:comparisons.append(dict(name=axis+'_'+a['name'],control=a['name'],treatment=a['name'].replace(axis+'0',axis+'1')))
    spec.update(task='41l',arms=l.arms(),checkpoints=[0,16,64,256,1024,4096],distillation_updates=0,
        new_seeds=dict(fit=811101,selection=811103,development=811107,stream=811109,feedback=811113,feedback_stream=811127,intervention=811137,support=811139,selection_support=811149,probe=811157),
        registered_comparisons=comparisons,
        frozen_diagnostic_input=dict(sha256=l.p.digest(target),members=members,source_archive_sha256=l.p.digest(source)),
        training_policy='8 full4004param H-initialized repair arms,3vectorized seeds; same18occupancies/causalV,Ca/bounded-affine transition/dt0.1/readout. Compression parked.',
        support='128paths/128anchors perblock;4x is exactblock0 plus3 independentseedoffset10000blocks. Families2paths and13feedbackinterventions uniformly sampled. Eachfeedbackpool guarantees13actions before seededwithoutreplacementfill. Cutoff inclusive topquartile per-family from1xHerror reused4x. Within each family50%uniform/50%abovefixedcutoff. Same inverse-CDF variates acrossD; exactsame rows acrossA/P. Independentselection unchanged16paths/8anchors. No openedtest targets in FIT.',
        objective='Same41k canonical dynamic base. P1 adds unit-weight maximum across165(channel x2path/13feedbackfamily) normalized losses, protection minibatch1example/family. P0 computes same probe without gradient. First flattenedgroup wins exactmax tie. Emptygroups abort. Same40tickfeedback,1200mspath timingquery,normalizers; no Hretention.',
        selection='Original canonical lexicographic development rank; P-protection not a selection score. Earliest best perseed; minimaxrank nominee acrossseeds. Opened panels only after freeze. No transplant/numerical diagnostic eligible.',
        budget='4096updates x8arms x3seeds; checkpoints0/16/64/256/1024/4096. LRinitial1e-5 or1e-6 times0.1**(min(completedupdates,1024)/1024). Phase64/256/1024/4096+frozen evaluation, externalSHA/CRC verified before continuation. Actual perarmGPUtime and exposure recorded; unequalP compute explicit.',
        diagnostics='8H/K channel swaps+fullK;2openedpanels x4feedbackforcing x2jointprecisionpolicies plusH/K intermediates; matchedcanonical references/nativefloors; old41knumericalsidecar reused. All fulloriginalbatches, saved diagnosticcases120/108/122 andsources. Imposedstress41c. Atcheckpoints realAdamdirectionfraction0,.25,.5,1 discardedclones andchannelgradientcos/blocknorm onFIT; candidate/optimizer unchanged.',
        hypotheses=[dict(id=x) for x in ('step','coverage','protection','feedback','numerical-interaction')],
        limits=['No threshold,abstraction,primaryFP32 or solver change. No new kinetic inference fallback.',
            'No automaticTask42;openedpanels not independentconfirmation;compression not trained.',
            'Numerical policy/control remains diagnostic. Nativefloor comparison must precede scientific attribution.',
            'Uniformfamily-stratified FIT sampling is declared refinement;41k/41l are not data-paired. Comparisons inside41l are paired.',
            'Protectioncompute unequal;3seeds not independentbiologicalreplicas;4096negative does not prove capacityimpossibility.',
            'Externalphasebackups;DBbackup localonly.'])
    paths=list(spec['sources'])+[l.k.CONFIG,'src/giada_teacher/task41k_recovery.py',
        'src/giada_teacher/task41l_support.py','src/giada_teacher/task41l_regression_attribution.py','src/giada_teacher/task41l_frozen_attribution.py','src/giada_teacher/task41l_evaluation.py',
        'scripts/build_task41l_protocol.py','scripts/run_task41l_regression_attribution.py','scripts/task41l_kaggle_pipeline.py','scripts/task41l_gpu_probe.py','tests/test_task41l_regression_attribution.py']
    spec['sources']={name:l.p.digest(ROOT/name) for name in dict.fromkeys(paths)}
    l.p.save_json(ROOT/l.CONFIG,spec)
    intro='# GIADA Task41l — Cause delle regressioni\n\n'+spec['training_policy']+'\n\n'+spec['support']+'\n\n'+spec['objective']+'\n\n'+spec['diagnostics']+'\n\nSoglie e astrazione immutate. Backup DB esclusivamente locale.\n'
    code="import os,subprocess,sys\nfrom pathlib import Path\nREPO=Path(os.environ.get('GIADA_REPO','/kaggle/working/giada'))\nOUTPUT=Path(os.environ.get('GIADA_OUTPUT','/kaggle/working/artifacts/giada_task41l'))\nassert not OUTPUT.exists()\nsubprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41l_regression_attribution.py'),str(OUTPUT),'--stop-after','4096'],cwd=REPO,check=True)\n# Production console pins revision, verified external inputs and staged resume.\n"
    evaluation="subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41l_regression_attribution.py'),str(OUTPUT),'--resume-local','--evaluation-only'],cwd=REPO,check=True)\n"
    cells=[dict(cell_type='markdown',id='design',metadata={},source=intro.splitlines(True))]+[dict(cell_type='code',id=name,metadata={},execution_count=None,outputs=[],source=text.splitlines(True)) for name,text in [('training',code),('frozen-evaluation',evaluation)]]
    l.p.save_json(ROOT/'notebooks/giada_task41l_regression_attribution.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    print(json.dumps(dict(valid=True,arms=8,members=24,comparisons=len(comparisons),protocol_sha256=l.p.digest(ROOT/l.CONFIG))))

if __name__=='__main__':main()
