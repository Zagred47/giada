"""Generate the single notebook and source-bound prospective41h protocol."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41h_path_repair as h

def main():
    folder=ROOT/'experiments/results/task41h_protocol_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists()
    completion=json.loads((ROOT/'experiments/results/task41g_audited_v1/audit_completion.json').read_text())
    recovery=completion['recovery'];assert recovery['valid'] and recovery['byte_identical_to_remote']
    with zipfile.ZipFile(recovery['local']) as archive:checkpoint=archive.read('frozen_checkpoint.pt')
    assert hashlib.sha256(checkpoint).hexdigest()=='5bff26a0ea7d9ba0995e0ec4a332bce9d01add6448b5a55c96de2036c938e6b6'
    cp=folder/'task41g_frozen.pt';cp.write_bytes(checkpoint)
    old=json.loads((ROOT/h.g.CONFIG).read_text());spec=dict(old)
    spec.update(task='41h',version='v1',arms=h.arms(),contrasts=[list(r) for r in h.contrasts()],
        task41g_checkpoint=cp.relative_to(ROOT).as_posix(),task41g_checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        task41g_nominee='M_constant_point_broad',task41g_diagnostic='M_anneal_sensitive_tail',
        new_seeds=dict(fit=651101,selection=651103,development=651107,stream=651109,feedback=651113,feedback_stream=651127,intervention=651137,support=651139,selection_support=651149),
        anchors=16384,selection_anchors=128,tail_anchors=1024,tail_batch=16,path_prefix_ticks=530,feedback_every=16,
        training_policy='30vectorizedmembers;frozenQ0;onlyCaHVA.m equilibrium;independentannealedAdam;18states,dt0.1ms unchanged.',
        support='Nested4096/16384atomicFIT;common4096preservationanchors;1024pairedconstant/ramp histories,matchedrest,Ca,initialstates,multipliers,horizon1200ms. Historytreatmentretains50%constant. Entireprefix retained by exact affine composition;trainingexecutiononly. HistorychangesprefixVdistribution intentionally;atomicdistribution and endpoint draws paired.',
        objective='Same censored coefficient objective and0.01preservation. Auxiliarycurrent+crossing+4msfeedback,gradientnormratio0.1,executedevery16or256updates. Allmemberscalculatecommonslots;gradientmaskchangesdose. Constant/ramp prefixes530ticks;tailprobes1/8/40/160/400/800/1147msafterprefix+labelledtailcrossing;officialmetricsunchanged.',
        selection='Primarystratified:commonnewdevelopmentcanonicalguards,introducedfailures,worsttargetseverity,feedback+max(constanttail,ramptail),atomic. Secondaryaggregateusesmeanofsame twohistorylosses. Earliestties;bothfrozenbeforeopenedregressions;secondarycandidatesdiagnostic.',
        budget='60000updates;checkpoints0/512/2048/8192/30000/60000;pairedRNGstreams;batch64atomic+64preservation;tail16;feedback4x40ticks. Supportsizechangesuniquecoverageatfixedupdates;logexamples,gradients,cost.',
        diagnostics='FrozengenitorQ0 and41gnominee/best-tail:2x2persistentV/solver32/64+recenteredFP32Newton,same12iterationsandreference;fullopened41/41cautonomousmetrics+allinterventions. CaHVAu_m/z_m/horaclesonstress41c. Repeatauditsonfrozen41hnominee. Alloraclesselectionineligible.',
        speed='Originalformula/LUT,costthreshold10%,batches1/642/4096and30pairedrepetitions;legacy/compactexecutionwithsameweights;step/wholeloopinherited;originalandmatchedrecenteredformulabaselines; noqualificationfromcostalone.',
        hypothesis_links=[dict(code='claims-giada-task41g-result-'+x+'-v1') for x in ('voltage_roundoff','crossing_floor','effective_objective_dose','data_vs_capacity_vs_optimizer','composed_execution_cost','selection_coverage')]+[dict(code='claims-giada-task41g-completion-'+x+'-v1') for x in ('history-coverage','voltage-state-versus-solver')],
        limits=old['limits']+['41hprimaryselectionstratified;secondaryruleisadiagnostic,noretrospectiveselection.','Historyandcapacityeffectsregisteredconditionally;128basis2armsnotfullfactorial.','Affinepathcompositionacceleratestrainingonly;deployedrecurrencemodelunchanged.'])
    description='''# GIADA Task41h — copertura delle traiettorie, dose e precisione

Un notebook,10bracci ×3seed. Otto bracci: storia costante/mista conrampa × ausiliaria ogni256/16update ×4096/16384ancore. Due controlli128basi:baseline e trattamento completo. GenitoreQ0 congelato, sola correzioneequilibrio Ca_HVA.m;18stati e0.1ms.

Prima e dopoiltraining: statoV/solver32/64,NewtonFP32ricentrato, oracoli solo diagnostici. RiferimentoFP32 e tutte le soglie originali invariati. Le storie FIT e development sono nuove. La composizione della ricorrenza è verificata suoutput egradienti; accelera soltanto iltraining aVprescritto.

Selezioneprimaria stratificata; controllosecondario aggregato suglistessicheckpoint. Tuttiipannelli storici,correnti,carica,eventi,interventi,ricorrenza e stress dopofreeze. Confronticosto appaiati:formula,LUT,legacy,compatto,step/wholeloop e solver aaritmeticacorrispondente. Nessuna promozione basatasui testaperti. Conferma indipendente successiva necessaria.

ConsoleColab conbackupesterni apezzi di trasporto verificatiSHA,senza cambiare i filescientifici. DB ebackupDB locali.
'''
    code="import os,subprocess,sys,json\nfrom pathlib import Path\nREPO=Path(os.environ.get('GIADA_REPO','/content/giada'))\nOUTPUT=Path(os.environ.get('GIADA_OUTPUT','/content/giada_task41h_result'))\nassert not OUTPUT.exists()\nsubprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41h_path_repair.py'),str(OUTPUT)],cwd=REPO,check=True)\n"
    cells=[dict(cell_type='markdown',metadata={},source=description.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["report=json.loads((OUTPUT/'final_report.json').read_text())\n","print({k:report.get(k) for k in ('valid','decision','nominee','passing_arms','thresholds_changed','task42_authorized')})\n"])]
    for k,cell in enumerate(cells):cell['id']=f'task41h-{k}'
    notebook='notebooks/giada_task41h_path_repair.ipynb'
    h.p.save_json(ROOT/notebook,dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5))
    paths=list(old['sources'])+[h.g.CONFIG,'src/giada_teacher/task41h_path_repair.py','src/giada_teacher/task41h_evaluation.py','scripts/run_task41h_path_repair.py',notebook,cp.relative_to(ROOT).as_posix()]
    spec['sources']={n:h.p.digest(ROOT/n) for n in dict.fromkeys(paths)}
    h.p.save_json(ROOT/h.CONFIG,spec)
    print(json.dumps(dict(valid=True,arms=10,members=30,thresholds_changed=False)))
if __name__=='__main__':main()
