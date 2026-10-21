"""Create the prospective41d contract and immutable opened41c transport."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41d_precision_response as d

def main():
    folder=ROOT/'experiments/results/task41d_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists()
    old=json.loads((ROOT/d.c.CONFIG).read_text());oldc=json.loads((ROOT/d.prior.CONFIG).read_text())
    archive=ROOT/'artifacts/task41c_primary_external/giada_task41c_result.zip'
    assert d.p.digest(archive)=='3e7a0d6430a0b3f44a517e0f932787b7dc7e42ca05d4765191d85bc752c4f597'
    names=['cases.json','native_shadow_traces.npz','fresh_stress_traces.npz','causal_results.json'];members={}
    bundle=ROOT/'artifacts/task41d_opened41c.zip'
    with zipfile.ZipFile(archive) as src,zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as out:
        for name in names:
            raw=src.read(name);out.writestr(name,raw);members[name]=hashlib.sha256(raw).hexdigest()
    sources=['src/giada_teacher/task41d_precision_response.py','src/giada_teacher/task41d_evaluation.py','scripts/run_task41d_precision_response.py',d.prior.CONFIG,'src/giada_teacher/task41c_frozen_confirmation.py']
    sources+=list(oldc['sources']);sources=list(dict.fromkeys(sources))
    spec={**old,'task':'41d','arms':d.arms(),'seeds':[17,29,43],
        'new_seeds':dict(fit=341101,selection=341103,development=341107,stream=341109,feedback=341113,selection_feedback=341127,feedback_stream=341139),
        'anchors':4096,'selection_anchors':512,'batch':64,'feedback_anchors':1024,'selection_feedback_anchors':64,'feedback_batch':8,'feedback_ticks':4,'feedback_every':64,
        'checkpoints':[0,1000,3000,10000,30000,60000],'lr':.001,'computational_parity_atol':1e-5,
        'sources':{n:d.p.digest(ROOT/n) for n in sources},'opened41c_sha256':d.p.digest(bundle),'opened41c_members':members,
        'quality':oldc['current_quality'],'compartment_gates':oldc['compartment_gates'],'cost':old['cost'],
        'hypotheses':[
            'Ca_HVA.m equilibrium precision improves with sensitivity-stratified FIT at fixed anchor count.',
            'Ca_HVA local32vs64basis distinguishes capacity from exposure;budget checkpoints distinguish optimization.',
            'A smooth learned Nap.m correction protects small-pair response fidelity better than a linear local basis at64parameters.',
            'Paired secant supervision adds information about local response beyond coefficient point error.',
            'Short causal compartment response supervision improves autonomous perturbation fidelity,conditional on shape/secant factors.',
            'Independent Ca/Nap repairs compose without losing prior learned functions;quality and speed are separately evaluated.'],
        'training_policy':'Only one extra learned equilibrium function per arm:Ca u0 orNap u8. P110(z7/z11 included) frozen.36members=12arms x3originalseeds,independentAdamandclipping. Ca supports sharepilotandseparateRNGstreams;Nap sameFIT/minibatches,64activeweights perseed. Updatesmatched,actualfeedbackcomputecostreported.',
        'objectives':'Pointlogiterror plusoptionalpairedsigmoidsecant error. Optional4tick causal compartment response loss,withinitialstates and exogenousinputs only;canonicalteacher labels generated beforetraining. AuxweightsFITgradientcalibrated0.1basegradient,diagnosticgradientsatcheckpoints;noeventthresholdchanges.',
        'selection':'Onlynewdevelopmentandselectionfeedback. Perseed checkpoint:failedcases,worstcompleteseverity,response loss,atomicloss;earliestexactties. Ca/Napnomineesbyworstseedrankbeforeopenedregressions. Parent,Caonly,Naponly,combined composition frozenbeforetest. No choosinga modelby41/41cpasscount.',
        'qualification':'Every unchanged per-channel current,charge,timing,gate,physical,andpairedresponse gate onallopened41/41c,40b,39jregressions plusnewdevelopment. Noaveragingfailedseed. Newindependentconfirmationrequiredafterrepair;Task42neverautomatic.',
        'contrast_rule':'Onefactoratfixedothers;all3developmentworstseverityratios<1,median<=0.8,andnofailedcasecountincrease. Descriptivefullcurvesalsoretained. Trainingeffectsnotclaimedfromopenedpanelnomination.',
        'limits':['Opened repair,not independent confirmation;allprior41/41b/41care nowregressionfixtures.','Level1 retained:all18states learned;canonicalratesonlyFITlabels/formulacontrols,notcandidateinference.','4tickfeedbacktrainingisashortcontrolledtest,notprooflongcreditassignmentadequacy.','SupportallocationchangesFITexamplesbutnototherRNGstreams;shape activeparametercounts matched,compute maydiffer.','12stresshistories x3modelseeds are36checks,not36independentprotocols.','Qualityandcostseparate;oneGPU;noTask42authorization.','ExternalrunbackupsPC;DBbackupstrictlylocal.']}
    d.p.save_json(ROOT/d.CONFIG,spec)
    code="import os,sys,subprocess,json\nfrom pathlib import Path\nrepo=Path(os.environ['GIADA_REPO']);output=Path(os.environ['GIADA_OUTPUT'])\nassert not output.exists()\nsubprocess.run([sys.executable,'-u',str(repo/'scripts/run_task41d_precision_response.py'),str(output)],cwd=repo,check=True)\n"
    cells=[dict(cell_type='markdown',metadata={},source=['# 🧪 GIADA Task41d — precisione e risposta causale\n','4bracci Ca_HVA +8bracci Nap,tre seed;checkpoint0/1k/3k/10k/30k/60k. Nuovi FIT/development;P110 congelata;regressioni dopo freeze. Soglie invariate.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["r=json.loads((output/'final_report.json').read_text());print({k:r.get(k) for k in ['valid','decision','passing_arms','nominees','task42_authorized']})\n"])]
    d.p.save_json(ROOT/'notebooks/giada_task41d_precision_response.ipynb',dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    print(json.dumps(dict(valid=True,arms=12,members=36,opened_bundle_bytes=bundle.stat().st_size)))
if __name__=='__main__':main()
