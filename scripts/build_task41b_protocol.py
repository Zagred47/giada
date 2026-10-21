"""Build the reviewable 2x2x2 protocol and identical notebook/console worker."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41b_local_precision as c

def main():
    folder=ROOT/'experiments/results/task41b_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists(),'Dispatched protocol is immutable'
    prior=json.loads((ROOT/c.prior.CONFIG).read_text());synthesis=json.loads((ROOT/'experiments/results/task41_audited_v1/synthesis.json').read_text())
    archive=ROOT/'artifacts/task41_retry_external/giada_task41_result.zip'
    assert c.p.digest(archive)=='3d02cdb451a73fc033fe896d84d822ca1552d9d3b48343db65db6b1dccc3ee92'
    names=['native_shadow_traces.npz','reference_traces.npz','fresh_stress_traces.npz','cases.json','causal_results.json','fresh_stress_results.json']
    target=ROOT/'artifacts/task41b_opened_inputs.zip';hashes={}
    with zipfile.ZipFile(archive) as source,zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        assert source.testzip() is None
        for name in names:
            raw=source.read(name);hashes[name]=hashlib.sha256(raw).hexdigest()
            info=zipfile.ZipInfo(name,(2000,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
    sources=[c.prior.CONFIG,c.p.CONFIG,'experiments/results/task41_audited_v1/audit.json',
        'experiments/results/task41_audited_v1/synthesis.json','experiments/results/task40b_audited_v1/frozen_checkpoint.pt',
        'src/giada_teacher/task41b_local_precision.py','src/giada_teacher/task41b_evaluation.py',
        'src/giada_teacher/task41_independent_confirmation.py','src/giada_teacher/task40b_calcium_precision.py']
    spec=dict(task='41b',version='v1',seeds=[17,29,43],dt_ms=.1,arms=c.arms(),
        quality=prior['current_quality'],compartment_gates=prior['compartment_gates'],cost=prior['cost'],
        fit_seed=141101,selection_seed=141103,development_seed=141107,stream_seed=141109,
        anchors=4096,selection_anchors=256,batch=64,lr=.001,checkpoints=[0,1000,3000,10000,30000],
        targets=[dict(coefficient=k,gate=g) for k,g in c.TARGETS],
        sources={name:c.p.digest(ROOT/name) for name in sources},opened_inputs_sha256=c.p.digest(target),opened_members=hashes,
        opened_origin_archive_sha256=c.p.digest(archive),hypotheses=synthesis['hypotheses'][4:],
        training='24independent correction tensors vectorized with independent Adam moments and per-member norm1clipping. ParentB6 frozen;96/192newweights perseed,zero initialization;onlyu_gate8,z_gate7,z_gate11canchange. FP64affine accumulation,FP32addedlogits and deployedstates.',
        data='4096anchors for each support. Uniform and canonical-sensitivity stratified supports share pilot/random stream and equal counts. BroadV[-100,60],Ca[1e-7,1e-2];no Task41exactinputs copied. Independent three-stage held/recovery/pulse levels,initialstates anddurations. Nested timehorizons0.1,1,8,40,400,1200ms.',
        objective='Common normalized logit-u/log-z MSE on3targetfunctions. Treatment adds bounded current residual scaled by canonical FIT slope*0.3ms withscale*1e-5floor. These are FIT surrogate weights,not altered evaluation thresholds. Initial uniformFIT gradient ratio0.1calibrated perseed andsharedacrossarms;allgradients clipped independently.',
        selection='At everycheckpoint,new8developmenthistories x3seeds perarm,1200ms at0.1ms,are scored with cached learned GPUcoefficients and serialCPUFP32state. Lexicographic rank:failedcompletecases,worstcompleteerror/threshold includingtiming,commonatomiclossrelative tozero-correction. Exactties keepearliest. Allarmsuseidentical256selectionanchors andnewdevelopment. ActualCUDAdevelopment isreevaluatedafterfreeze;itmustpassforcandidatequalification.',
        negative_controls='FrozenB6 exactzeroinitialization andbitwiseparentpreservation; untouchedcoefficients exact; no-canonical-fallback guard; formulaFP32/64 floors; actualT4optimizerisolation andeager/graphinferencepreflight. No extra learnedarchitecture sidecar without a new preregistration.',
        contrasts='All12conditional contrasts ofsupport,resolution,objective;perseedfailedcounts,worstcompleteincludingtiming,andatomicerror. Mini budgetcurves alongsameoptimizertrajectory. A cause is only supported if matched effect improves all3seeds and medianworstseverityratio<=0.8,withnoincreasedfailedcasecount;reportdescriptiveeffectsotherwise.',
        regression='OpenedTask41 native176cases x3seeds inbothmodes,112interventions x3seeds per mode,and12old41stresshistories x3seeds. ReproduceB6 passflags. Also all old40b native/syntheticcoupled cases,interventions and39j recurrence development/stress. Allopened,notnewconfirmation.',
        qualification='Require all unchanged newdevelopmentCUDA,old39j/40b/41state,current,charge,timing,occupancy,V,Ca,spike,domainandpairedresponsegates. No averaging orexcludedfailedcase. A passingarmonlyauthorizesaudit andfutureindependentconfirmation;Task42remainsunauthorized.',
        execution='One notebook invokes same isolated worker as console. No NEURON regeneration needed: originalnativeinputs memberSHAverified. Checkpoint beforeselectionwithoptimizer+RNG;out-of-VM supervisor verifies SHA/CRC of each recoverybundle. Fullnative/gpu evaluation onlyafterfreeze.',
        limits=['Diagnostic repair informed byopenedTask41;notindependentconfirmation.',
            'All18gates remainlearned;cheapcurrentreadout,Ca,membraneandsynapses unchanged. No newstateorcanonicalkineticsininference.',
            'Morelocalbasis changesparametersandcost;coverageaxis changesdistributionatfixedcount,notanindependenttestofallpossibledatasetsizes.',
            'ShadowselectionCPU andCUDAarithmeticmaydiffer slightly;actualfrozenCUDAqualityauthoritative. No posthoc reselection.',
            'Singlecompartment andtheseprotocols;no universalbifurcation,multicompartment or quality-qualifiedspeedclaim.',
            'SQLitebackup localonly;externalrunbackups toPC;noAirtableorDriveDBaccess.'])
    c.p.save_json(ROOT/c.CONFIG,spec)
    rows=['# 🧪 GIADA Task 41b — Precisione locale delle correnti\n',
        'Tre funzioni: equilibrio Nap_Et2.m, rilassamento NaTs2_t.h e Im. B6 congelata; soglie Task37/40/41 invariate.\n',
        '| Asse | Controllo | Trattamento |\n|---|---|---|\n| Dati | 4096 anchor uniformi | 4096 anchor stratificati per sensibilità FIT |\n| Capacità | 32 basi | 64 basi |\n| Obiettivo | Coefficienti | Coefficienti + correnti condizionate |\n',
        'Otto bracci × tre seed, eseguiti in un unico training vettorizzato. Checkpoint0/1000/3000/10000/30000; selezione solo sui nuovi dati development.\n',
        'Dopo il freeze: valutazione CUDA dei nuovi development, regressioni39j/40b/41, controlli numerici, correnti, timing e costo su tutti i batch. I dati41 sono aperti e non costituiscono una nuova conferma indipendente.\n',
        'Log, checkpoint, ottimizzatore e RNG salvati prima della selezione; il supervisore da console copia e verifica i backup sul PC. DBSQLite esclusivamente locale.\n']
    code="import os,sys,subprocess,json\nfrom pathlib import Path\nrepo=Path(os.environ['GIADA_REPO']);output=Path(os.environ['GIADA_OUTPUT'])\nassert not output.exists()\nsubprocess.run([sys.executable,'-u',str(repo/'scripts/run_task41b_local_precision.py'),str(output)],cwd=repo,check=True)\n"
    cells=[dict(cell_type='markdown',metadata={},source=rows),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["report=json.loads((output/'final_report.json').read_text())\n","print({k:report.get(k) for k in ('valid','decision','passing_arms','thresholds_changed','task42_authorized')})\n"])]
    c.p.save_json(ROOT/'notebooks/giada_task41b_local_precision.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    c.p.save_json(folder/'opened_inputs_manifest.json',dict(valid=True,archive=str(target.relative_to(ROOT)),sha256=c.p.digest(target),bytes=target.stat().st_size,members=hashes,origin_archive_sha256=c.p.digest(archive),opened=True,selection_allowed=False))
    print(json.dumps(dict(valid=True,arms=8,members=24,opened_bundle_bytes=target.stat().st_size)))

if __name__=='__main__':main()
