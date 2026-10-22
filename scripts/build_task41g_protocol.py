"""Build the source-bound41g preregistration and shared notebook."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41g_preservation as g

def main():
    folder=ROOT/'experiments/results/task41g_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists()
    old=json.loads((ROOT/g.f.CONFIG).read_text())
    receipt=json.loads((ROOT/'artifacts/task41f_reserve_external/final_recovery_receipt.json').read_text())
    archive=Path(receipt['local']);assert receipt['valid'] and g.p.digest(archive)==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        frozen=z.read('frozen_checkpoint.pt');final=json.loads(z.read('final_report.json'))
        assert final['decision']=='PRECISION_RELAXATION_NO_GO'
    (folder/'task41f_frozen.pt').write_bytes(frozen)
    g.p.save_json(folder/'source_receipt.json',dict(valid=True,archive_sha256=receipt['sha256'],frozen_sha256=hashlib.sha256(frozen).hexdigest(),
        nominee=final['nominee'],runtime_independent=True,source_no_go=True))
    text='''# 🧪 GIADA Task41g — precisione e conservazione delle funzioni apprese

12bracci ×3seed, stesso notebook e worker GPU vettorizzato. Il genitore è Q0 della41e;
P110 e il candidato41f restano controlli congelati.18stati, passo0.1ms e soglie canoniche invariati.

| Asse | Livelli |
|---|---|
| Teste correggibili | solo equilibrio Ca_HVA.m / tutti3equilibri Ca_HVA.m,h eNap.m |
| Ottimizzatore | Adam0.001costante / Adam con annealing fino1e-7 |
| Obiettivo | coefficienti / coefficienti + corrente e sensibilità del crossing + feedback4ms |
| Copertura | ampia / arricchita su code (quattro bracci aggiuntivi solo-m) |

Otto bracci core e quattro di supporto. Conservazione su nuovi ancoraggi comune a tutti;
stesso numero di esempi percontrasto. I gate non corretti restano appresi e congelati.
Prima del training:2×2×2 precisione di V/gate/readout su pesi congelati, riferimentoFP32
ufficiale;rollback pertesta delle correzioni41f contro il loro vero genitoreP110.
Proiezioni64/128nodi,lineari/smoothstepC2 e1024/4096/16384dati: diagnostica,non candidati.

La loss delle code usa label canoniche FIT a1/8/40/160ms e intornoal crossing fino400ms.
Le label future non sono input del modello. Tutti i nuovi pesi sono selezionati solo su
nuovo development; pannelli già aperti e precisioni alternative restano diagnostici.
GO richiede tutti i gate originali, incluse regressioni,stress,ricorrenza e risposte causali.
Confrontare sempre casi risolti/introdotti. Nuova conferma indipendente necessaria dopo riparazione.

Log e checkpoint sono copiati fuori dal runtime dal supervisore console. Backup DB locale.
'''
    code="""import json,os,sys,subprocess
from pathlib import Path
REPO=Path(os.environ.get('GIADA_REPO','/content/giada'))
OUTPUT=Path(os.environ.get('GIADA_OUTPUT','/content/giada_task41g_result'))
assert not OUTPUT.exists(),f'Output già presente: {OUTPUT}'
subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41g_preservation.py'),str(OUTPUT)],cwd=REPO,check=True)
"""
    cells=[dict(cell_type='markdown',metadata={},source=text.splitlines(True)),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["report=json.loads((OUTPUT/'final_report.json').read_text())\n","print({k:report.get(k) for k in ('valid','decision','nominee','passing_arms','thresholds_changed','task42_authorized')})\n"])]
    for j,cell in enumerate(cells):cell['id']=f'task41g-{j}'
    notebook='notebooks/giada_task41g_preservation.ipynb'
    g.p.save_json(ROOT/notebook,dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    sources=list(old['sources'])+[g.f.CONFIG,'src/giada_teacher/task41g_preservation.py','src/giada_teacher/task41g_evaluation.py',
        'scripts/run_task41g_preservation.py',notebook,'experiments/results/task41g_protocol_v1/task41f_frozen.pt']
    spec=dict(old)
    audit=json.loads((ROOT/'experiments/results/task41f_audited_v1/synthesis.json').read_text(encoding='utf-8'))
    spec.update(task='41g',version='v1',arms=g.arms(),contrasts=[list(r) for r in g.contrasts()],
        sources={name:g.p.digest(ROOT/name) for name in dict.fromkeys(sources)},
        task41f_checkpoint='experiments/results/task41g_protocol_v1/task41f_frozen.pt',task41f_nominee=final['nominee'],
        task41f_checkpoint_sha256=hashlib.sha256(frozen).hexdigest(),
        new_seeds=dict(fit=641101,selection=641103,development=641107,stream=641109,feedback=641113,feedback_stream=641127,
            intervention=641137,support=641139,selection_support=641149),
        lr_min_ratio=.0001,tail_anchors=4096,tail_batch=32,feedback_every=256,
        hypothesis_links=[dict(code='claims-giada-task41f-result-'+h['id']+'-v1',**h) for h in audit['hypotheses']],
        training_policy='36vectorizedmembers;Q0frozenparent;M64orALL192activeweights;allzand18statesunchanged;independentAdam;pointCUDAgraph;annealedorconstantpermemberLR;commonpreservationanchors.',
        support='4096fromshared8192FITanchors;equalcountbroad/tailsensitivityenrichment;new4096preservationanchors common;new4096held-inputtails withindependentinitialstates;newphysicalfeedbackhistories;selection/developmentdisjoint.',
        objective='Censoredlogitequilibrium;fixedinitialperheadnormalization;common0.01residualpreservation. Sensitivearms addcurrentlossat1/8/40/160msandcrossingupto400ms plus40tickcoupledresponse,gradientnormratio0.1. Crossinglossweight0.01,scaledbycanonical0.3msandmin1e-10. Samelabelsandslotsforpointcontrols;disabledauxiliarygradient.',
        selection='Samecanonicalrankandthresholds;introducedcasesrelativeQ0first;newdevelopmentonly;earliestties;oldopenedpanelsneverselectweights.',
        budget='60000updates;every256replacepointstepwithmatchedpoint+auxiliaryslot;batch64+64preservation,tail32,feedback4x40ticks.Allarmscomputebothobjectives,onlyregisteredgradientenabled. Sameinitialization/seed/minibatch/checkpoints;countalllabelandpreservationexposure.',
        diagnostics='Frozen8precisionpoliciesV/gates/readout,FP32voltagetrunklabels;originalfullbatchofficialreference;perheadandzrollback41f/P110. Projection64/128linearandC2smoothstep,count1024/4096/16384,heldoutindependent,nondeployable. Noanalyticaloracleincandidate.',
        speed='Alloriginalformula/modelresidentbenchmarksunchanged;selectedmodelprecisionpoliciesV64/gates64/readout64/all64versusownFP32,642/4096sites,30pairedrepeats. Original10%qualificationretained.',
        limits=old['limits']+['Q0sourceparentexplicit;noautomatic41fretrospectivepromotion.','QuinticsmoothpartitionisC2,notcubicB-spline;approximationprobeonly.','Preservationiscommon,itsindividualcausaleffectisnotidentified.'])
    g.p.save_json(ROOT/g.CONFIG,spec)
    print(json.dumps(dict(valid=True,arms=12,members=36,contrasts=len(g.contrasts()),thresholds_changed=False)))

if __name__=='__main__':main()
