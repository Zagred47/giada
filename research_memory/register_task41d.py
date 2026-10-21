"""Prospective41d registration, frozen parent, aligned factorials; SQLite only."""
import hashlib,json,sqlite3,sys
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .mirror import Contract

class PreviewComplete(Exception):pass
class Preview:
    def __init__(self):
        self.contract=Contract();self.con=sqlite3.connect((ROOT/'data/research_memory.sqlite').resolve().as_uri()+'?mode=ro',uri=True);self.con.row_factory=sqlite3.Row
    def local_upsert(self,table,fields):
        self.contract.normalize(table,fields,partial=True)
        return dict(record_id='rec'+hashlib.sha256(fields['Codice stabile'].encode()).hexdigest()[:14])
    def query(self,sql):return dict(rows=[dict(r) for r in self.con.execute(sql)])
    def commit_batch(self):self.con.close();raise PreviewComplete()

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41d_protocol_v1';spec=json.loads((repo/'experiments/task41d_preregistration_v1.json').read_text())
    pf=json.loads((folder/'gpu_preflight.json').read_text());assert pf['valid'] and pf['all_parameters_finite'] and pf['members']==36
    assert pf['source_sha256']==hashlib.sha256((repo/'src/giada_teacher/task41d_precision_response.py').read_bytes()).hexdigest()
    assert pf['protocol_sha256']==hashlib.sha256((repo/'experiments/task41d_preregistration_v1.json').read_bytes()).hexdigest()
    m=Preview() if '--validate-only' in sys.argv else ValidatedBatchMirror();ids={};arts={}
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41d-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    names=['experiments/task41d_preregistration_v1.json','src/giada_teacher/task41d_precision_response.py','src/giada_teacher/task41d_evaluation.py','scripts/run_task41d_precision_response.py','scripts/task41d_gpu_probe.py','scripts/build_task41d_protocol.py','scripts/prepare_task41d_dispatch.py','scripts/launch_task41d_console.ps1','scripts/colab_task41d_driver.py','scripts/task41b_console_manager.py','scripts/task41b_local_supervisor.py','tests/test_task41d_precision_response.py','notebooks/giada_task41d_precision_response.ipynb','experiments/giada_modular_surrogate_roadmap.md','experiments/results/task41d_protocol_v1/gpu_preflight.json','experiments/results/task41d_protocol_v1/preflight_history.json','experiments/results/task41d_protocol_v1/preflight_request.json','artifacts/task41d_opened41c.zip','research_memory/register_task41d.py']
    for name in names:
        path=repo/name;arts[name]=put('artifacts',name,Nome='Task41d '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Versione='v1',Percorso=name,**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    claims=[]
    for i,statement in enumerate(spec['hypotheses']):claims.append(put('claims','H'+str(i+1),Nome='Task41d H'+str(i+1),Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,**{'Condizioni di falsificazione':spec['contrast_rule']},Limiti='Prospective controlledrepair,notnewconfirmation;allcase/seedgatesunchanged.'))
    exp=put('experiments','factorial',Nome='Task41d local equilibrium precision and autonomous response factorial',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['training_policy'],**{'Obiettivo informativo':'Separate support,resolution,shape,secant information and short feedback training;protected frozenfunctions;newFIT/development.'})
    decision=put('decisions','authorized',Nome='Authorize41d on audited41c mechanism evidence',Esito='Continuare',Motivazione='Userauthorized12armsinsamenotebook,unchangedthresholds,GPUparallelism,externalrunbackups,localonlyDBbackup. Sourceprotocolversionedbeforetraining.',Risultati=[ref('findings-giada-task41c-audit-ca_mechanism-v1'),ref('findings-giada-task41c-audit-nap_mechanism-v1')],**{'Data':'2026-10-22'})
    gen=put('generators','fit-development',Nome='Task41d independent atomic,paired and causal FIT generators',Versione='v1',Procedura=spec['objectives'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':list(arts.values())})
    data=put('datasets','new-fit-development',Nome='Task41d new FIT/development and short response pairs',Versione='v1',Generatore=[gen],Schema='V,Ca,18initialstates,canonicalu/zlabels,localvoltagepairs,4tickcausalresponses;noopened41coriginaltrajectoriesintraining',Copertura='BroadV[-100,60],Ca[1e-7,1e-2];4096atomicanchorsperarm,1024sharedcausalpairs;newdevelopment12x1200ms. Support pools overlap;do notsumthem asdistinctindependentexamples.')
    parts=[]
    for name,role in [('fit','Fit'),('development','Development')]:
        # Existing mirror uses the original Airtable enum spellings.
        parts.append(put('partitions',name,Nome='Task41d '+name,Dataset=[data],Ruolo=role,**{'Regola di assegnazione':'Disjointseedsubstreamsandmaterializedimmutablearrays','Restrizioni di accesso':'Noopenedregressionbeforecheckpointfreeze;selectiononlynewdevelopment.'}))
    protocol=put('protocols','factorial',Nome='Task41d preregistered12arm factorial',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),**{'Partizioni autorizzate':parts,'Artefatto preregistrazione':[arts['experiments/task41d_preregistration_v1.json']],'Hash preregistrazione':hashlib.sha256((repo/'experiments/task41d_preregistration_v1.json').read_bytes()).hexdigest(),'Criteri di successo':spec['qualification'],'Regole di arresto':'Nonfinitecheckpointorfloorfailure:technicalstop,preservebackup. Scientificgatefailure:NO_GOwithfulloutputs,noautomaticthresholdchange.','Registrato il':'2026-10-22'})
    revision=json.loads((folder/'preflight_request.json').read_text())['revision']
    impl=put('implementations','worker',Nome='Task41d parallel36member GPU console worker',Commit=revision,**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41d_precision_response.py','Artefatti codice':list(arts.values()),'Ambiente':'Colab CUDA T4,deterministic,TF32disabled;atomicCUDAgraph,causalfeedbackeager;independentAdam/clipping;PCsupervisorSHA/CRC.'})
    arms=[];runs=[]
    parent=ref('models-giada-task41b-local-P110-v1')
    for a in spec['arms']:
        model=put('models',a['name'],Nome='Task41d '+a['name'],Ruolo='Surrogate',Descrizione=json.dumps(a)+' FrozenP110 plusone bounded learned equilibrium correction;all18statesretained;additionalactiveparametersperseed='+str(a['basis']),**{'Contratto di stato':'18physicaloccupanciesin[0,1],samefixed0.1msmap','Contratto di input e output':'V,Ca,current18gate states -> next18states;onlyu0or u8newlearnedcorrection;currentreadoutandmembraneunchanged.'})
        arm=put('arms',a['name'],Nome='Task41d '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(a)})
        arms.append(arm)
        for seed in spec['seeds']:runs.append(put('runs',a['name']+'-'+str(seed),Nome=f"Task41d {a['name']} seed{seed}",Stato='Pianificata',Seed=str(seed),Braccio=[arm],**{'Configurazione effettiva':json.dumps(dict(arm=a,seeds=spec['new_seeds'],checkpoints=spec['checkpoints'],parent='P110'))}))
    baseline=put('arms','P110',Nome='Task41d frozenP110 baseline',Ruolo='Baseline',Protocollo=[protocol],Modello=[parent],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':'Frozen source checkpoint f83dd0...;allthreeoriginalmodelseeds;nooptimizer.'})
    axes={'Ca':['support','basis'],'Nap':['shape','paired','feedback']}
    for panel,aa in axes.items():
        for axis in aa:put('contrasts',panel+'-'+axis,Nome='Task41d '+panel+' '+axis,Protocollo=[protocol],Bracci=[arms[i] for i,a in enumerate(spec['arms']) if a['panel']==panel],**{'Contrasto e coefficienti':'Pairsofarmsdifferinthisfactoronly,holdotheraxesfixed;reportallconditionalcontrasts','Soglia interpretativa':spec['contrast_rule'],'Confondenti controllati':spec['training_policy']})
    for name,definition in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),('cost',spec['cost']),('optimizer',dict(members=36,isolated_moments=True,clip=1.,preflight=pf))]:
        metric=put('metrics',name,Nome='Task41d '+name,Formula=json.dumps(definition))
        put('evaluations',name,Nome='Task41d '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,Ruolo='Secondaria' if name=='cost' else 'Primaria',Target=json.dumps(definition))
    observation=put('observations','gpu-preflight',Nome='Task41d actual GPU numerical and optimizer preflight',Descrizione=json.dumps(pf),**{'Artefatti dettagliati':[arts['experiments/results/task41d_protocol_v1/gpu_preflight.json']]})
    put('findings','gpu-preflight',Nome='Task41d technical preflight passed before scientificrun',Esito='Positivo',Esperimenti=[exp],Osservazioni=[observation],Risultato='GPUfinite,36members,isolatedoptimizer,zero-initparity,untargetedweightsfrozen,noanalyticfallback,graph/eagerandmixedoptimizerparity.',Limitazioni='Technicalsyntheticpreflight,noscientificqualityclaim,freshFIT/developmentnotyettrained;thresholdsunchanged.')
    put('actions','launch',Nome='LaunchTask41d pinned console worker',Stato='Proposta',Descrizione='Userauthorized;dispatchrequestwillpinverifiedsourceandworkerSHAafterDBreceipt. ExternalPCbackupsandlocalDBbackup.',**{'Decisioni di origine':[decision]})
    print(json.dumps(dict(stage='staged',records=len(ids),runs=len(runs))),flush=True)
    m.commit_batch();print('Task41d committed;exporting and verifying',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    rows=m.query("SELECT count(*) n FROM runs WHERE [Codice stabile] LIKE 'runs-giada-task41d-%'")['rows'][0];assert rows['n']==36
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,planned_training_runs=36,airtable_accessed=False,backup_policy='DB localonly',launched=False,registration_before_scientific_run=True))
    print('Task41d prospective registration verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Registration fields,enums andexistingreferences validated;no DB writes',flush=True)
