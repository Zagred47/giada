"""Register original Task24 in the exclusive local mirror."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror

def main():
    m=ValidatedBatchMirror()
    p=ROOT.parent/'experiments/task24_sodium_family_composition.json'
    cfg=json.loads(p.read_text(encoding='utf-8'))
    assert m.query("SELECT record_id FROM v_records WHERE stable_code='decisions-giada-roadmap-task23-to24-v1'")['rows']
    def put(table,key,**fields):
        return m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task24-{key}-v1',**fields})['record_id']
    claims=[put('claims',k,**{'Nome':'Task24 '+k,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':v,'Limiti':cfg['limits'],'Condizioni di falsificazione':cfg['promotion']}) for k,v in cfg['hypotheses'].items()]
    exp=put('experiments','matrix',**{'Nome':'GIADA Task24 originale — NaTa/NaTs2/Nap','Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':claims,'Descrizione':cfg['architecture'],'Obiettivo informativo':'Sodium composition, negative transfer, identity, scaling, slow Nap memory, imposed paths and frozen inference cost.'})
    protocol=put('protocols','paired',**{'Nome':'Task24 composizione appaiata','Esperimento':[exp],'Versione':'v1','Modalità':'Componente isolato','Procedura':json.dumps(cfg,ensure_ascii=False),'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(p.read_bytes()).hexdigest(),'Regole di arresto':'Oracle/equivalenza/hash invalidi: errore esecutivo. Freeze prima di fresh; nessuna soglia adattata al test.'})
    evaluations=[]
    for group in ('gates','pair_gates','held_gates','path_gates','path_channel_gates'):
        for key,threshold in cfg[group].items():
            metric=put('metrics',group+'-'+key,**{'Nome':'Task24 '+group+' '+key,'Famiglia':'Fisica' if key=='occupancy_violations' else 'Regressione','Direzione':'Minimizzare','Unità':'adimensionale','Formula':'sodium_family_composition.'+key,'Casi degeneri':'Non finito: fallimento; zero driving: corrente zero.'})
            evaluations.append(put('evaluations',group+'-'+key,**{'Nome':'Task24 '+group+' '+key,'Metrica':[metric],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(threshold),'Popolazione e regioni':'All3 sodiumchannels, eachof3seeds; OODseparate','Orizzonte e finestre':cfg['path_contract'],'Aggregazione e pesi':'Congiunzione per canale/seed/dominio; componenti e somma misurate separatamente.'}))
    arms=[]
    for family in cfg['families']:
        for width in cfg['widths']:
            count={'independent':3*(width**2+7*width+4),'shared_heads':width**2+15*width+12,'conditioned':width**2+10*width+4}[family]
            model=put('models',f'{family}-w{width}',**{'Nome':f'Task24 sodium {family} w{width}','Ruolo':'Surrogate','Versione':'v1','Descrizione':cfg['architecture']+f' Selected configuration candidate: {family}, width{width}, parameters perseed bundle{count}.','Contratto di stato':'Six explicit occupancies m/h for NaTa_t,NaTs2_t,Nap_Et2; [0,1] preserved by convex exponential update.','Contratto di input e output':'RateMLP input V only plus onehotidentity for conditioned; no hidden endpointteacher/current. Initial6gate states anddt only enter solver. Outputs six newgate states, rate probes and analyticindividual/total sodiumcurrent.'})
            arms.append(put('arms',f'{family}-w{width}',**{'Nome':f'Task24 {family} w{width}','Ruolo':'Baseline' if family=='independent' else 'Trattamento','Protocollo':[protocol],'Modello':[model],'Descrizione':cfg['architecture'],'Configurazione residua':cfg['fit_sampling']}))
    for claim,(key,text) in zip(claims,cfg['hypotheses'].items()):
        prediction=put('predictions',key,**{'Nome':'Task24 '+key,'Ipotesi':[claim],'Origine':'Preregistrata','Risultato atteso':text,'Specifica di valutazione':[evaluations[0]],'Soglia o intervallo':cfg['promotion']})
        put('contrasts',key,**{'Nome':'Task24 '+key,'Bracci':arms,'Predizioni':[prediction],'Specifiche di valutazione':evaluations,'Contrasto e coefficienti':text,'Confondenti controllati':cfg['fit_sampling'],'Soglia interpretativa':text,'Correzione confronti multipli':'Soglie ingegneristiche preregistrate, no pvalue.'})
    for role in ('fit','development','fresh'):
        put('blocks',role,**{'Nome':'Task24 '+role,'Protocollo':[protocol],'Seed':json.dumps(cfg['data_seeds'][role]),'Descrizione':cfg['selection'],'Regola di appaiamento':'Stesse tuple V/stato/dt e minibatch; momenti Adam e clipping indipendenti per seed.'})
    put('artifacts','preregistration',**{'Nome':'Task24 preregistrazione','Tipo':'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})
    for path in sorted((ROOT.parent/'experiments/preflight/task24_sodium_family').glob('*.json')):
        put('artifacts','preflight-'+path.stem,**{'Nome':'Task24 preflight '+path.stem,'Tipo':'Report','Percorso':path.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps({'valid':True,'hypotheses':len(claims),'arms':len(arms),'airtable_accessed':False}))

if __name__=='__main__':main()
