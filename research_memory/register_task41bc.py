"""Full immutable 41b results plus prospective frozen 41c, local SQLite only."""
import hashlib,json,zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41c_protocol_v1'
    spec=json.loads((repo/'experiments/task41c_preregistration_v1.json').read_text())
    audit=json.loads((repo/'experiments/results/task41b_audited_v1/audit.json').read_text())
    assert audit['freeze_hash_valid'] and audit['final_report']['passing_arms']==['P110']
    m=ValidatedBatchMirror();ids={};arts={};obs=[]
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41bc-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def artifact(path):
        path=Path(path);name=path.relative_to(repo).as_posix()
        rid=put('artifacts',name,Nome='Task41bc '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Versione='v1',Percorso=name,
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size});arts[name]=rid;return rid
    archive=repo/'artifacts/task41b_primary_external/giada_task41b_result.zip';archive_id=artifact(archive)
    expb=ref('experiments-giada-task41b-local-suite-v1')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not name.endswith('.json'):continue
            value=json.loads(z.read(name))
            obs.append(put('observations','41b-'+name,Nome='Task41b full '+name,Descrizione=json.dumps(value,ensure_ascii=False,separators=(',',':')),
                **{'Artefatti dettagliati':[archive_id]}))
    for arm,s in audit['final_report']['summaries'].items():
        for seed in (17,29,43):
            m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41b-local-{arm}-{seed}-v1','Stato':'Completata'})
        obs.append(put('observations','41b-summary-'+arm,Nome='Task41b '+arm+' full qualification',Descrizione=json.dumps(s),Valore=s['stress_passed'],Numerosità=s['stress_total'],**{'Artefatti dettagliati':[archive_id]}))
    auditid=artifact(repo/'experiments/results/task41b_audited_v1/audit.json')
    obs.append(put('observations','41b-audit',Nome='Task41b backup,finite,frozen,cost,regression audit',Descrizione=json.dumps(audit),**{'Artefatti dettagliati':[auditid,archive_id]}))
    finding=put('findings','41b-result',Nome='Task41b P110 repairs opened quality, with cost penalty',Esito='Misto',Esperimenti=[expb],Osservazioni=obs,
        Risultato='P110 alone:528/528 bothmodes,336/336responses bothmodes,36/36stress,24/24newdevelopment;allhistorical264/264+168/168bothmodes,72/72recurrenceeachrole. B6reproduced527/528,33/36stress. Allsavedtensorsfinite. Fullrawreports queryable inobservations.',
        Limitazioni='Opened repair,notindependentconfirmation. P010autonomousNapresponse335/336. Conditionedobjective notsystematicallyhelpful. Noformalcontrastimprovesall3seeds. P110stressoffset touches0.3mslimit. P110slowerthanB6andLUT;large4096batchslowerthanformula. NoTask42authorization.')
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41b-local-suite-v1','Stato':'Concluso','Descrizione':'All8arms completed;P110 solepassingarm,openedrepair,costpenalty;fullresults registered.'})
    for name,text in [('coverage-resolution','Sensitive support plus64localbasis P110 passes, but no factorial main cause established acrossallseeds.'),('objective','Extra conditioned-current objective worsens some timing stress;P11036/36vsP11134/36. Preserve no-go for failed variants.'),('precision-cost','Local learnedprecision fixes tiny timing-sensitive failures but adds runtime cost. Separate fidelity,compression and speed qualification.'),('independence','P110 selected onopened41b:freeze all3originalseeds before new confirmation;no retuning aftertest.')]:
        put('lessons',name,Nome='Task41b '+name,Lezione=text,Risultati=[finding],Limiti='Scoped single-compartment evidence;notuniversal.')
    decision=put('decisions','41c',Nome='Freeze P110; independent41c, not42',Esito='Continuare',Motivazione='Userapprovedfreshconfirmation;unchangedthresholds,nooptimizer,no modelselection. Quality GO separatefromspeed;Task42afterauditonly.',Risultati=[finding],**{'Data':'2026-10-21'})
    paths=['experiments/task41c_preregistration_v1.json',spec['checkpoint'],'src/giada_teacher/task41c_frozen_confirmation.py','scripts/run_task41c_frozen_confirmation.py',
        'scripts/build_task41c_protocol.py','scripts/launch_task41c_console.ps1','scripts/colab_task41c_driver.py','scripts/task41b_console_manager.py','scripts/task41b_local_supervisor.py',
        'tests/test_task41c_frozen_confirmation.py','notebooks/giada_task41c_frozen_confirmation.ipynb','experiments/giada_modular_surrogate_roadmap.md','research_memory/register_task41bc.py']
    for name in paths:artifact(repo/name)
    claimids=[]
    for i,h in enumerate(spec['hypotheses']):claimids.append(put('claims','41c-'+str(i),Nome='Task41c H'+str(i+1),Tipo='Ipotesi',Stato='Aperta',Enunciato=h,
        **{'Condizioni di falsificazione':'Any unchanged corresponding registeredcasegate fails, iftechnicalfloorspass;no averagingorposthocselection.'},Limiti='Prospective hypothesis,notyetoutcome.'))
    exp=put('experiments','41c',Nome='Task41c independent frozen P110 confirmation',Stato='Preregistrato',Ipotesi=claimids,Descrizione=spec['design'],**{'Obiettivo informativo':'Testindependentfidelityandpairedresponseof frozenP110;costdescriptive,separate.'})
    gen=put('generators','41c',Nome='Task41c pinned native and conditioned-history generators',Versione='v1',Procedura=spec['design'],**{'Seed e RNG':json.dumps({k:spec[k] for k in ('native_seeds','stress_seeds','calibration_seed')}),'Artefatti codice':list(arts.values())})
    data=put('datasets','41c',Nome='Task41c unopened native and long histories',Versione='v1',Generatore=[gen],Schema='176nativecases;12stresshistories;V,Ca,18states,11currents,causalshadowinputs;3modelseeds',Copertura=spec['design'],**{'Numero esempi':188})
    partition=put('partitions','41c',Nome='Task41c sealed confirmation',Dataset=[data],Ruolo='Test sigillato',**{'Regola di assegnazione':'Allnewprotocolsandseedsfixedbeforeoutcomes','Restrizioni di accesso':spec['freeze']})
    protocol=put('protocols','41c',Nome='Task41c frozen independent confirmation',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
        **{'Partizioni autorizzate':[partition],'Artefatto preregistrazione':[arts['experiments/task41c_preregistration_v1.json']],
           'Hash preregistrazione':hashlib.sha256((repo/'experiments/task41c_preregistration_v1.json').read_bytes()).hexdigest(),
           'Criteri di successo':spec['qualification'],'Regole di arresto':'Failedtechnicalfloor:stop,no scientificjudgment. Scientificfailure:preservealloutputs,no automaticretrain/thresholdchange.','Registrato il':'2026-10-21'})
    impl=put('implementations','41c',Nome='Task41c isolated native/CUDA console worker',**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41c_frozen_confirmation.py','Artefatti codice':list(arts.values()),'Ambiente':'Colab T4;native NEURON8.2.7 andCUDA separateprocesses;externalPCsupervisorSHA/CRC;SQLitebackup localonly.'})
    model=ref('models-giada-task41b-local-P110-v1')
    models={'P110':model,'B6':ref('models-giada-task40b-calcium-B6-v1')}
    for name in ('formula','LUT_control'):models[name]=put('models','41c-'+name,Nome='Task41c '+name,Ruolo='Baseline',Descrizione='Frozen numerical/cost control,notalternatecandidate;deterministicreplicalabelsnotnewtraining.')
    arms=[]
    for name,modelid in models.items():
        arm=put('arms','41c-'+name,Nome='Task41c '+name,Ruolo='Trattamento' if name=='P110' else 'Baseline',Protocollo=[protocol],Modello=[modelid],Implementazione=[impl],Partizioni=[partition],**{'Configurazione residua':spec['freeze']})
        arms.append(arm)
        for seed in spec['model_seeds']:put('runs','41c-'+name+'-'+str(seed),Nome=f'Task41c {name} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[arm],**{'Configurazione effettiva':spec['freeze']})
    for mode in ('imposed','autonomous'):put('contrasts','41c-'+mode,Nome='Task41c paired '+mode,Protocollo=[protocol],Bracci=arms,
        **{'Contrasto e coefficienti':'P110 versus frozenB6,formula,LUT;sameseeds,inputs,initialstatesandcausalstreams;noalternativeselection.','Soglia interpretativa':spec['qualification'],'Confondenti controllati':spec['freeze']})
    for name,value in [('quality',spec['current_quality']),('compartment',spec['compartment_gates']),('stress',dict(duration_ms=1200,quality=spec['current_quality'])),('phase',spec['native_current_policy']),('cost',spec['cost'])]:
        metric=put('metrics','41c-'+name,Nome='Task41c '+name,Formula=json.dumps(value))
        put('evaluations','41c-'+name,Nome='Task41c '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=[partition],Ruolo='Secondaria' if name=='cost' else 'Primaria',Target=json.dumps(value))
    put('actions','41c-launch',Nome='Launch frozen Task41c from console',Stato='Proposta',Descrizione='Userauthorized;actualverifieddispatchreceiptmustfollow. DBbackup localonly.',**{'Decisioni di origine':[decision]})
    print(json.dumps(dict(stage='staged',records=len(ids),full_observations=len(obs))),flush=True)
    m.commit_batch();print('Committed; exporting and verifying complete mirror',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,full_result_observations=len(obs),verification=v,airtable_accessed=False,DBbackup='localonly',launched=False))
    print('Task41b results and41c preregistration verified',flush=True)
if __name__=='__main__':main()
