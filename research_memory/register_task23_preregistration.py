"""Register original Task23 in the exclusive local mirror."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror

def main():
    m=ValidatedBatchMirror()
    p=ROOT.parent/'experiments/task23_calcium_pair_composition.json'
    cfg=json.loads(p.read_text(encoding='utf-8'))
    assert m.query("SELECT record_id FROM v_records WHERE stable_code='decisions-giada-roadmap-task22-to23-v1'")['rows']
    finding=m.query("SELECT record_id FROM v_records WHERE stable_code='findings-giada-roadmap-task22-independent-v1'")['rows'][0]['record_id']
    claim=m.query("SELECT record_id FROM v_records WHERE stable_code='claims-giada-roadmap-task22-long_rollout-v1'")['rows'][0]['record_id']
    m.local_upsert('evidence',{'Codice stabile':'evidence-giada-roadmap-task22-independent-long-rollout-v1','Nome':'Task22 rollout lungo: riferimento indipendente','Esito':'Sostiene','Affermazione valutata':[claim],'Risultati a sostegno':[finding],'Argomentazione':'Ih/Im/Nap_h indipendenti passano tutti3seed fino10000 passi; supporto limitato al riferimento indipendente e V imposto, non ai modelli condivisi.'})
    m.local_upsert('claims',{'Codice stabile':'claims-giada-roadmap-task22-long_rollout-v1','Stato':'Supportata nel dominio','Limiti':'Supporto al modello indipendente congelato, domini in-support e V imposto. Non sharing generale, non voltaggio autonomo.'})
    def put(table,key,**fields):
        return m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task23-{key}-v1',**fields})['record_id']
    claims=[put('claims',k,**{'Nome':'Task23 '+k,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':v,'Limiti':cfg['limits'],'Condizioni di falsificazione':cfg['promotion']}) for k,v in cfg['hypotheses'].items()]
    exp=put('experiments','matrix',**{'Nome':'GIADA Task23 originale — Ca-HVA + Ca-LVA','Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':claims,'Descrizione':cfg['architecture'],'Obiettivo informativo':'Composizione, negative transfer, identità, scaling e percorsi imposti.'})
    protocol=put('protocols','paired',**{'Nome':'Task23 composizione appaiata','Esperimento':[exp],'Versione':'v1','Modalità':'Componente isolato','Procedura':json.dumps(cfg,ensure_ascii=False),'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(p.read_bytes()).hexdigest(),'Regole di arresto':'Oracle/equivalenza/hash invalidi: errore esecutivo. Freeze prima di fresh; nessuna soglia adattata al test.'})
    evaluations=[]
    for group in ('gates','pair_gates','held_gates','path_gates'):
        for key,threshold in cfg[group].items():
            metric=put('metrics',group+'-'+key,**{'Nome':'Task23 '+group+' '+key,'Famiglia':'Fisica' if key=='occupancy_violations' else 'Regressione','Direzione':'Minimizzare','Unità':'adimensionale','Formula':'calcium_pair_composition.'+key,'Casi degeneri':'Non finito: fallimento; zero driving: corrente zero.'})
            evaluations.append(put('evaluations',group+'-'+key,**{'Nome':'Task23 '+group+' '+key,'Metrica':[metric],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(threshold),'Popolazione e regioni':'Entrambi canali, ciascuno dei3seed; OOD separato','Orizzonte e finestre':cfg['path_contract'],'Aggregazione e pesi':'Congiunzione per canale/seed/dominio; componenti e somma misurate separatamente.'}))
    arms=[]
    for family in cfg['families']:
        for width in cfg['widths']:
            arms.append(put('arms',f'{family}-w{width}',**{'Nome':f'Task23 {family} w{width}','Ruolo':'Baseline' if family=='independent' else 'Trattamento','Protocollo':[protocol],'Descrizione':cfg['architecture'],'Configurazione residua':cfg['fit_sampling']}))
    for claim,(key,text) in zip(claims,cfg['hypotheses'].items()):
        prediction=put('predictions',key,**{'Nome':'Task23 '+key,'Ipotesi':[claim],'Origine':'Preregistrata','Risultato atteso':text,'Specifica di valutazione':[evaluations[0]],'Soglia o intervallo':cfg['promotion']})
        put('contrasts',key,**{'Nome':'Task23 '+key,'Bracci':arms,'Predizioni':[prediction],'Specifiche di valutazione':evaluations,'Contrasto e coefficienti':text,'Confondenti controllati':cfg['fit_sampling'],'Soglia interpretativa':text,'Correzione confronti multipli':'Soglie ingegneristiche preregistrate, no pvalue.'})
    for role in ('fit','development','fresh'):
        put('blocks',role,**{'Nome':'Task23 '+role,'Protocollo':[protocol],'Seed':json.dumps(cfg['data_seeds'][role]),'Descrizione':cfg['selection'],'Regola di appaiamento':'Stesse tuple V/stato/dt e minibatch; momenti Adam e clipping indipendenti per seed.'})
    put('artifacts','preregistration',**{'Nome':'Task23 preregistrazione','Tipo':'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps({'valid':True,'hypotheses':len(claims),'arms':len(arms),'airtable_accessed':False}))

if __name__=='__main__':main()
