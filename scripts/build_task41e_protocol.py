"""Create an explicit 41e protocol; no inherited inactive training metadata."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41e_transfer_model as m


def main():
    folder=ROOT/'experiments/results/task41e_protocol_v1'
    folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists(),'Dispatched contracts are immutable'
    old=json.loads((ROOT/m.d.CONFIG).read_text())
    archive=ROOT/'artifacts/task41d_primary_external/giada_task41d_result.zip'
    expected='b27fdfc79da68a5b6afbe3aba44f608d53a84a42594a01183d6d5c6dd4e16311'
    assert m.p.digest(archive)==expected
    with zipfile.ZipFile(archive) as z:
        weights=z.read('frozen_checkpoint.pt');report=z.read('final_report.json')
    assert hashlib.sha256(weights).hexdigest()=='68fff43f9d2793fd30131321125ca8ce567d4beb37ff48c76697f92080103a9a'
    (folder/'task41d_frozen.pt').write_bytes(weights)
    (folder/'task41d_summary.json').write_bytes(report)
    m.p.save_json(folder/'source_receipt.json',dict(valid=True,archive_sha256=expected,
                members={name:hashlib.sha256(raw).hexdigest() for name,raw in [('frozen_checkpoint.pt',weights),('final_report.json',report)]},
                source_technical_valid=json.loads(report)['valid'],source_no_go=True,thresholds_changed=False))
    matrix='''# 🧪 GIADA Task 41e — apprendimento e trasferimento al sistema accoppiato

Un unico worker esegue i confronti in parallelo su GPU, con 36 membri indipendenti.

| Gruppo | Confronti | Configurazioni nuove |
|---|---|---:|
| A | Loss attuale/censurata × bilanciamento iniziale/dinamico, più due controlli puntuali | 6 |
| H | Feedback 0,4/4 ms × supporto ampio/stratificato, con un controllo condiviso | 3 |
| J | Training separato/congiunto × protezione delle code, con un controllo condiviso | 3 |

P110 congelata; due correzioni di equilibrio da 64 pesi; stessi 18 stati e passo 0,1 ms.
I controlli congelati della 41d sono riprodotti dopo il freeze. Tutte le soglie di corrente,
carica, timing, gate, risposte autonome e regressioni sono identiche alla 41d.
I replay V/Ca, la precisione FP32/64, gli oracoli e i Jacobiani sono diagnostici.

Checkpoint lungo la stessa traiettoria; confronto a pari finestre e a pari tick di feedback
(5.760 update a 4 ms contro 57.600 a 0,4 ms). I budget atomici e i tempi effettivi sono
riportati separatamente. Le regressioni aperte sono lette solo dopo il freeze.
Il notebook e la console richiamano lo stesso script della revisione fissata.
'''
    code="""import os, sys, json, subprocess
from pathlib import Path
REPO=Path(os.environ.get('GIADA_REPO','/content/giada'))
OUTPUT=Path(os.environ.get('GIADA_OUTPUT','/content/giada_task41e_result'))
assert not OUTPUT.exists(), f'Output già presente: {OUTPUT}'
protocol=json.loads((REPO/'experiments/task41e_preregistration_v1.json').read_text())
print({'task':protocol['task'],'arms':len(protocol['arms']),'seeds':protocol['seeds'],'thresholds_changed':False})
subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41e_transfer.py'),str(OUTPUT)],cwd=REPO,check=True)
"""
    cells=[dict(cell_type='markdown',metadata={},source=matrix.splitlines(True)),
           dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),
           dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
               "report=json.loads((OUTPUT/'final_report.json').read_text())\n",
               "print({k:report.get(k) for k in ('valid','decision','passing_arms','nominee','baseline_reproduced','thresholds_changed','task42_authorized')})\n"])]
    for k,cell in enumerate(cells):cell['id']='task41e-'+str(k)
    m.p.save_json(ROOT/'notebooks/giada_task41e_transfer_diagnosis.ipynb',dict(cells=cells,
                  metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    source_names=['src/giada_teacher/task41e_transfer_model.py','src/giada_teacher/task41e_transfer_data.py',
                  'src/giada_teacher/task41e_transfer_training.py','src/giada_teacher/task41e_transfer_evaluation.py',
                  'scripts/run_task41e_transfer.py','notebooks/giada_task41e_transfer_diagnosis.ipynb',m.d.CONFIG]
    source_names+=list(old['sources'])
    source_names+=['experiments/results/task41e_protocol_v1/task41d_frozen.pt','experiments/results/task41e_protocol_v1/task41d_summary.json']
    synthesis=json.loads((ROOT/'experiments/results/task41d_audited_v1/synthesis.json').read_text())
    spec=dict(task='41e',version='v1',seeds=[17,29,43],dt_ms=.1,arms=m.arms(),contrasts=[list(a) for a in m.contrasts()],
        quality=old['quality'],compartment_gates=old['compartment_gates'],cost=old['cost'],
        parent='P110',parent_sha256=m.d.prior.contract(ROOT)[0]['sources']['experiments/results/task41c_protocol_v1/P110_frozen.pt'],
        task41d_checkpoint='experiments/results/task41e_protocol_v1/task41d_frozen.pt',task41d_checkpoint_sha256=hashlib.sha256(weights).hexdigest(),
        task41d_summary='experiments/results/task41e_protocol_v1/task41d_summary.json',
        task41d_archive_sha256=expected,
        sources={name:m.p.digest(ROOT/name) for name in dict.fromkeys(source_names)},
        opened_inputs_sha256=old['opened_inputs_sha256'],opened_members=old['opened_members'],
        opened41c_sha256=old['opened41c_sha256'],opened41c_members=old['opened41c_members'],
        new_seeds=dict(fit=441101,selection=441103,development=441107,stream=441109,feedback=441113,
                       feedback_stream=441127,intervention=441137,support=441139,selection_support=441149,
                       protection=441151,protection_stream=441163),
        anchors=4096,selection_anchors=512,batch=64,lr=.001,
        feedback_histories=128,selection_feedback_histories=16,history_ticks=12000,
        feedback_anchors=1024,selection_feedback_anchors=64,feedback_batch=8,feedback_every=64,
        protection_anchors=256,protection_batch=16,protection_ticks=[4,40,400,4000,12000],
        checkpoints=[0,1024,3072,5760,10000,30000,57600,60000],computational_parity_atol=1e-5,
        balance_policy=dict(target_gradient_ratio=.1,ema=.9,gradient_floor=1e-12,maximum_weight=1e6,
                            minimum_weight_ratio=.5,maximum_weight_ratio=2.),
        hypothesis_links=[dict(code='claims-giada-task41d-result-'+h['id']+'-v1',**h) for h in synthesis['hypotheses']],
        training_policy='36members=12arms x3seeds;128activeweights each,zeroinitialized;P110frozen.IndependentAdamcoordinatesandper-head norm1clipping. ResidentCUDAgraph atomicsteps;grouped4/40tickfeedback;no models or seeds trained sequentially.',
        objective='Clipped logit equality versus censored interval logit loss on saturated labels,keeping every FIT point. Paired secants are diagnostic only. Causal response on both Ca_HVA and Nap currents; separate training disables the other correction per rollout, joint training evolves both. Auxiliary gradients scaled perhead;initial/dynamic balance policy explicit.',
        protection='Canonical FIT held-V/Ca gate/current tails at0.4/4/40/400/1200ms from varied initial occupancies. Analytic repeated relaxation supplies differentiable training supervision; all deployed FP32 long recurrences remain evaluated. Parent is a nonregression control,not a replacement teacher.',
        support='Equal1024windows from the same2048FIT states,128independent random compartment histories over1200ms. Broad seeded sample versus50%broad/25%small responses/25%slow tails. Same atomic anchors,minibatches and exogenous streams.',
        selection='Primary perseed:count of newly failed previously passing per-channel and gate checks,target Ca/Nap failed cases,worst target severity,common40tick deployed response loss,atomic score,global failure count/global severity;earliest exact ties. Secondary global rule stored for all checkpoints;only the primary-nominated arm also gets its secondary-selected checkpoint evaluated on opened regressions. No opened-panel nomination.',
        qualification='All original current,charge,event,gate,physical,pairedresponse,compartment andhistoricalregression gates on every case and seed plusnewdevelopment. K_Pst remains included. New independent confirmation required after repair;Task42notautomatic.',
        contrast_rule='At fixed remaining factors,all3target-severity ratios<1,median<=0.8,andnoincreaseintargetfailcountorintroducedregressions. Global gates remain mandatory;diagnostic target contrast is not a scientificGO.',
        budgets='Same60katomicupdates and feedback every64updates;compare causal horizons both at same windows and at3600feedbackticks per sampledwindow-count unit:long5760updates versus short57600updates,with8windows each event. Atomic exposures differ in tick-matched comparison;actual windows,ticks,scenarios,walltime,memory reported. No equal-total-compute claim.',
        frozen_probes='After freeze:originalP110/C01/N000/combined controls x Vclamp yes/no x Caclamp yes/no x FP32/FP64 on opened41c cases108/122. Amplitude0/.25/.5/1 sweep;u0/u8oracles diagnosticonly;dimensionless localJacobian probe. K_Pst canonical oracle onnewdevelopment onlydiagnostic. Phase and state interface unchanged.',
        speed='Benchmark fused and old41dcomposition after raw/rolloutequivalence,plusalltrainedarms,formula,LUT,B6,P110;paired30repetitions,batches1/642/4096,ticks80,ionicandintegrated. Original10%criterionretained;noqualifiedspeedclaimwithoutallqualitygates.',
        execution='One notebook and same pinned console worker;checkpoint before selection with fullAdam,balanceEMA andRNG. PCsupervisor continuously retrieves immutable SHA/CRCchecked archives. DBbackup remainslocal.',
        stop_rules='Nonfinite parameters/gradient,numerical preflight or required reference floor failure:technical stop and preserve outputs. Scientific failures remain NO_GO with full reports. No threshold changes or automatic retraining.',
        limits=['Opened repair,not independent confirmation.','Same single compartment and18states;no abstraction reduction or new kinetic fallback.',
                'Teacher-formula labels are FIT supervision only;diagnostic clamps/oracles are never candidate inputs.',
                'Three modelseeds are not independent biological protocols.','Held-tail supervision doesnotproveautonomous long-horizon adequacy.',
                'Tick-matched feedback comparisons have different atomic update exposure;report both.',
                'All seven hypotheses remain unconfirmed until their registered contrasts.','ExternalexperimentbackupsPC;DBbackupstrictlylocal.'])
    m.p.save_json(ROOT/m.CONFIG,spec)
    print(json.dumps(dict(valid=True,arms=len(m.arms()),members=36,new_weights_per_member=128,
                          paired_contrasts=len(m.contrasts()),thresholds_changed=False)))


if __name__=='__main__':main()
