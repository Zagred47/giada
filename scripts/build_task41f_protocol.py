"""Materialize exact parent checkpoints, explicit factorial and same notebook."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41f_precision_relaxation as f


def main():
    folder=ROOT/'experiments/results/task41f_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists()
    previous=(ROOT/f.CONFIG).read_bytes() if (ROOT/f.CONFIG).exists() else None
    old=json.loads((ROOT/f.prior.CONFIG).read_text())
    receipt=json.loads((ROOT/'artifacts/task41e_reserve_external/final_recovery_receipt.json').read_text())
    archive=Path(receipt['local']);assert receipt['valid'] and f.p.digest(archive)==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        frozen=z.read('frozen_checkpoint.pt');final=json.loads(z.read('final_report.json'))
        causal=json.loads(z.read('opened41c/causal_results.json'))
        selected={}
        for name in ('P110','D_C01','D_combined','D_fused','Q0','H10'):
            selected['E_'+name if name in ('Q0','H10') else name]=[r['metrics']['passed'] for r in causal[name+'/autonomous']['interventions'] if r['case'] in (108,122)]
    (folder/'task41e_frozen.pt').write_bytes(frozen)
    f.p.save_json(folder/'task41e_reproduction.json',selected)
    f.p.save_json(folder/'task41e_summary.json',final)
    f.p.save_json(folder/'source_receipt.json',dict(valid=True,archive_sha256=receipt['sha256'],runtime_independent=True,
        frozen_sha256=hashlib.sha256(frozen).hexdigest(),original_nominee=final['nominee'],source_no_go=True))
    text='''# 🧪 GIADA Task 41f — precisione, rilassamento e trasferimento

Un notebook e lo stesso worker da console. Soglie e riferimento ufficiale della41e invariati.

| Asse | Livelli |
|---|---|
| Correzione appresa | U64: equilibrio,64basi; U128: equilibrio,128basi; UZ64: equilibrio+rilassamento,64basi |
| Dati FIT | ampi / arricchiti di sensibilità e code lente |
| Obiettivo | coefficienti / coefficienti+risposta accoppiata |

12bracci ×3seed =36modelli addestrati insieme. Gate bersaglio Ca_HVA.m/h eNap_Et2.m;
192/384/384pesi attivi perseed. U128/UZ64 controlla capacità totale.18stati invariati.

Prima del training: riproduzione dei fallimenti congelati sul batch originale, matriciFP32/mista/FP64
e riferimenti alternativi diagnostici,clamp V/Ca e oracoli separati u/z,m/h. Questi dati già aperti
non selezionano i nuovi pesi. FIT/selection/development nuovi,freeze prima delle regressioni dei nuovi modelli.
Ranking e valutazione condividono il predicato canonico,compresa la tolleranza numerica già esistente.

Checkpoint a budget crescenti; loss,gradienti e casi individuali salvati. Confronto di risoluzione
su sottocampioniFIT1024/4096: diagnostica di approssimazione,non scaling del trainingSGD.
Benchmark identico di tutte le varianti e matrice graph intero/perpasso × fisica imposta/accoppiata.
Backup esperimento esterno al runtime; DB solo locale.
'''
    code="""import json,os,sys,subprocess
from pathlib import Path
REPO=Path(os.environ.get('GIADA_REPO','/content/giada'))
OUTPUT=Path(os.environ.get('GIADA_OUTPUT','/content/giada_task41f_result'))
assert not OUTPUT.exists(),f'Output già presente: {OUTPUT}'
subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41f_precision_relaxation.py'),str(OUTPUT)],cwd=REPO,check=True)
"""
    cells=[dict(cell_type='markdown',metadata={},source=text.splitlines(True)),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["report=json.loads((OUTPUT/'final_report.json').read_text())\n","print({k:report.get(k) for k in ('valid','decision','nominee','passing_arms','thresholds_changed','task42_authorized')})\n"])]
    for k,cell in enumerate(cells):cell['id']=f'task41f-{k}'
    notebook='notebooks/giada_task41f_precision_relaxation.ipynb'
    f.p.save_json(ROOT/notebook,dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    sources=list(old['sources'])+[f.prior.CONFIG,'src/giada_teacher/task41f_precision_relaxation.py','src/giada_teacher/task41f_evaluation.py',
        'scripts/run_task41f_precision_relaxation.py',notebook,
        'experiments/results/task41f_protocol_v1/task41e_frozen.pt','experiments/results/task41f_protocol_v1/task41e_reproduction.json']
    new={k:old[k] for k in ('quality','compartment_gates','cost','task41d_checkpoint','task41d_checkpoint_sha256','task41d_summary',
        'opened_inputs_sha256','opened_members','opened41c_sha256','opened41c_members','parent_sha256','balance_policy','protection_ticks')}
    synthesis=json.loads((ROOT/'experiments/results/task41e_audited_v1/synthesis.json').read_text(encoding='utf-8'))
    new.update(task='41f',version='v1',seeds=[17,29,43],dt_ms=.1,arms=f.arms(),contrasts=[list(r) for r in f.contrasts()],
        sources={n:f.p.digest(ROOT/n) for n in dict.fromkeys(sources)},
        task41e_checkpoint='experiments/results/task41f_protocol_v1/task41e_frozen.pt',task41e_checkpoint_sha256=hashlib.sha256(frozen).hexdigest(),
        task41e_reproduction='experiments/results/task41f_protocol_v1/task41e_reproduction.json',
        new_seeds=dict(fit=541101,selection=541103,development=541107,stream=541109,feedback=541113,feedback_stream=541127,
            intervention=541137,support=541139,selection_support=541149),
        anchors=4096,selection_anchors=512,batch=64,lr=.001,feedback_histories=128,selection_feedback_histories=16,
        history_ticks=12000,feedback_anchors=1024,selection_feedback_anchors=32,feedback_batch=4,feedback_every=64,feedback_ratio=.1,
        checkpoints=[0,512,2048,8192,30000,60000],computational_parity_atol=1e-5,
        hypothesis_links=[dict(code='claims-giada-task41e-result-'+h['id']+'-v1',**h) for h in synthesis['hypotheses']],
        training_policy='36vectorizedmembers;P110frozen;Ca_HVA.m/h,Nap.m learned bounded logit corrections;activeweights192/384/384, padded storage768each. IndependentAdam andperhead norm1clip. AtomicCUDAgraph,FP64physicalfeedback withFP32trainableweights. DeploymentFP32 remains primary.',
        selection='Canonical per-channel passed predicate;introduced failures,targetfailedcases,targetworstseverity,common40tickresponse,atomicloss,globalfailures;earliestties. Newdevelopmentonly. Opened frozen diagnostics before training are recorded;never used for nomination or training.',
        objective='Censored-logit equilibrium and log-sigmoid relaxation coefficient errors;normalized peractivehead. Point-only versus joint40tick physical causal response,canonicalFP64reference;dynamic gradientnorm balance perhead. Floating reference sensitivity explicitly audited,official FP32 acceptanceunchanged.',
        support='Equal4096anchors from shared8192FITpool;half broad+half sensitive/slow forrare,seeded broad sampleforcontrol. Same causalhistorypool andequal1024windows;separateatomic labels andnewselection/development. Future endpoints never modelinputs.',
        budget='60000atomicupdates,feedbackevery64,4pairedwindows,40ticks each. Point-only controls use same update slots andbatchstream. All arms share seeds,minibatches,initialization andcheckpoints. Diagnostic1024/4096linear coefficientprojection separatesresolution/data support fromSGDbutisnotatrainedcandidate.',
        qualification=old['qualification'],contrast_rule=old['contrast_rule'],
        diagnostics='Reproduce original6response outcomes percontrol atfull528batch beforeprecision/clamp attribution;FP32/mixedstate64/FP64 againstofficialFP32,diagnosticFP64andNumPyreferences. Batchreduction sidecar,matchedclamps,u_m/z_m/u_h/z_h/u_Nap/z_Nap/alloracles onslowreturn,allselectionineligible.',
        speed='Originalcost30pairedrepetitions,80ticks,batches1/642/4096,ionic/integrated forallarms. AdditionalfrozenP110,Q0,H10andpreregisterednominee graphperstep/interoloop x imposed/coupled,exactsameweights/input/dtype;equivalence1e-5beforecost. Original10%criterion,quality gatesstillrequired.',
        stop_rules='Nonfinite gradient/parameter,failedtechnicalparity orfailedrequiredcanonicalfloor:technicalstop,preservealloutputs. ScientificNO_GO retainedifanygatefails. Nochangesofthresholds orposthocnomination.',
        limits=['Opened repair,notindependentconfirmation;Task42notautomatic.','All18states andabstractionlevelunchanged;noinferencekineticfallback.',
            'Alternateprecisionreferencesandcanonicaloraclesare diagnosticandselectionineligible.','H10isafrozencomparison,notaretrospectivelypromotedwinner.',
            'Three modelseeds shareprotocols;notindependentbiologicalreplications.','Fullrawreportsandcheckpointscopiedoffruntime;DBbackuponlylocal.'])
    f.p.save_json(ROOT/f.CONFIG,new)
    if '--record-technical-amendment' in sys.argv:
        assert previous is not None
        before=json.loads(previous);after=dict(new)
        before_sources=before.pop('sources');after_sources=after.pop('sources')
        assert before==after,'Scientific contract must be unchanged'
        f.p.save_json(folder/'protocol_before_technical_fix.json',json.loads(previous))
        f.p.save_json(folder/'pretraining_amendment.json',dict(valid=True,
            reason='CUDAgraph capture cannot materialize a host Boolean. Equivalent explicit censor expression uses existing CUDA tensors.',
            before_protocol_sha256=hashlib.sha256(previous).hexdigest(),after_protocol_sha256=f.p.digest(ROOT/f.CONFIG),
            before_sources=before_sources,after_sources=after_sources,scientific_contract_unchanged=True,
            thresholds_changed=False,scientific_training_performed_before_fix=False))
    print(json.dumps(dict(valid=True,arms=12,members=36,contrasts=len(f.contrasts()),thresholds_changed=False)))


if __name__=='__main__':main()
