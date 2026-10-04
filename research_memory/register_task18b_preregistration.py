"""Register the causal Task18b factorial and future fresh roles, SQLite only."""
import hashlib,json
from .mirror import Mirror,ROOT,write_json

def main():
    mirror=Mirror();path=ROOT.parent/'experiments/task18b_potassium_diagnosis.json';cfg=json.loads(path.read_text())
    def put(t,s,**f):return mirror.local_upsert(t,{'Codice stabile':f'{t}-giada-task18b-{s}-v1',**f})['record_id']
    claims=[]
    for key,enunciato in cfg['hypotheses'].items():
        claims.append(put('claims',key,**{'Nome':'Task18b — '+key,'Enunciato':enunciato,'Tipo':'Ipotesi','Stato':'Aperta','Condizioni di falsificazione':'Contrasto appaiato non migliora gli strati uniformi/coda alla stessa capacità e checkpoint; soglie originali ancora non superate.','Limiti':'Ipotesi esplorative K_Pst; 3seed e domini registrati. Non prova di limite universale.'}))
    ex=put('experiments','matrix',**{'Nome':'GIADA Task18b — K_Pst coda/tau/capacità/budget','Descrizione':'Fattoriale2x2x2;3seed;4checkpoint. Successore dellaTask18 parzialmente negativa.','Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':claims,'Obiettivo informativo':'Separare copertura coda, peso tau, capacità e budget; conservare Ca/Na senza rilanciare.'})
    protocol=put('protocols','paired',**{'Nome':'Task18b matrice K_Pst appaiata','Esperimento':[ex],'Versione':'v1','Modalità':'Componente isolato','Procedura':json.dumps(cfg),'Criteri di successo':cfg['promotion'],'Regole di arresto':'Audit/freeze/equivalenza invalidi: problema esecutivo. Se non passa, non Task19 e non cambiare soglie dopo test.','Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest()})
    arms={}
    for w in cfg['widths']:
        for s in ('uniform','tail_enriched'):
            for t in (.01,.1):
                key=f'w{w}-{s}-tau{t}'
                arms[key]=put('arms',key,**{'Nome':'Task18b '+key,'Ruolo':'Trattamento','Protocollo':[protocol],'Descrizione':'MLP constrained logtau; stesso model_factoryTask18. Sampling='+s+';tauweight='+str(t),'Configurazione residua':'Infweight0.1; Adam0.003; clip1; stessa inizializzazione/seed, tuplecount24k, dt/stati/indici appaiati; nessun peso condiviso.'})
    ev=[]
    for key,threshold in cfg['gates'].items():
        m=put('metrics',key,**{'Nome':'Task18b '+key,'Famiglia':'Fisica' if key=='occupancy_violations' else 'Regressione','Formula':key+' defined by hh_family_transfer.measurements','Direzione':'Minimizzare','Unità':'adimensionale','Parametri':str(threshold),'Casi degeneri':'Non finito o violazione fisica blocca promozione.'})
        ev.append(put('evaluations',key,**{'Nome':'Task18b '+key,'Metrica':[m],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(threshold),'Popolazione e regioni':'K_Pst m/h; tutti3seed; in-support e coda negativa separati','Orizzonte e finestre':'dt0.025/0.1/0.5/1/5/25/100ms; V costante','Aggregazione e pesi':'Tutti i gate, entrambi gli strati, ciascun seed; OOD diagnostico.'}))
    for i,(key,text) in enumerate(cfg['hypotheses'].items()):
        pred=put('predictions',key,**{'Nome':'Task18b '+key,'Origine':'Preregistrata','Ipotesi':[claims[i]],'Risultato atteso':text,'Specifica di valutazione':[ev[0]],'Soglia o intervallo':'Soglie originali e contrasti a stessi step; nuova confermafresh dopo freeze.'})
        put('contrasts',key,**{'Nome':'Task18b '+key,'Bracci':list(arms.values()),'Predizioni':[pred],'Specifiche di valutazione':ev,'Contrasto e coefficienti':text,'Soglia interpretativa':'Effetti entro capacità/seed/step; non confondere migliore configurazione selezionata con puro effetto causale.','Confondenti controllati':'Stati, dt, seed, minibatch, optimizer; coda cambia soloV.','Correzione confronti multipli':'Soglie congiuntive, nessun p-value.'})
    for role in ('fit','development','fresh'):
        put('blocks',role,**{'Nome':'Task18b '+role,'Protocollo':[protocol],'Seed':json.dumps(cfg['data_seeds']),'Descrizione':'Nuovi seed; oldTask18fresh usato solo per diagnosi, non chiamato conferma nuova.','Regola di appaiamento':'Controlli appaiati per modello; una configurazione comune ai3seed.'})
    put('artifacts','preregistration',**{'Nome':'Task18b preregistrazione','Tipo':'Report','Percorso':path.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'})
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'hypotheses':len(claims),'arms':len(arms),'airtable_accessed':False}))

if __name__=='__main__':main()
