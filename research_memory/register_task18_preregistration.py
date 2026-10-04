"""Task18 original roadmap; preregister hypotheses, paired factorial arms and metrics."""
import hashlib
import json
from .mirror import Mirror,ROOT,write_json

def main():
    mirror=Mirror();path=ROOT.parent/'experiments/task18_hh_family_transfer.json'
    cfg=json.loads(path.read_text())
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task18-{suffix}-v1',**fields})['record_id']
    claims=[]
    for key,text in cfg['hypotheses'].items():
        claims.append(put('claims',key,**{'Nome':'Task18 — '+key,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':text,'Condizioni di falsificazione':'Contrasti appaiati per canale/seed; gate registrati non raggiunti al budget massimo. Nessun successo implicito per media aggregata.','Limiti':cfg['limits']}))
    experiment=put('experiments','matrix',**{'Nome':'GIADA Task18 originale — famiglia HH','Descrizione':'Ca_HVA/NaTa_t/K_Pst: due capacità e due obiettivi, tre seed; audit nativo e LUT.','Obiettivo informativo':'Trasferibilità della struttura atomica; identifiabilità dei rate; capacità e budget. Non anticipare Task22.','Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':claims})
    protocol=put('protocols','paired',**{'Nome':'Task18 — matrice appaiata HH','Esperimento':[experiment],'Versione':'v1','Modalità':'Componente isolato','Procedura':json.dumps(cfg,ensure_ascii=False),'Criteri di successo':cfg['selection'],'Regole di arresto':'Audit nativo o equivalenza vettorizzata falliti: errore esecutivo, non scientifico. Nessun test fresco prima del freeze.','Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest()})
    evals=[]
    for key,threshold in cfg['gates'].items():
        metric=put('metrics',key,**{'Nome':'Task18 — '+key,'Famiglia':'Fisica' if key=='occupancy_violations' else 'Regressione','Formula':key+'; definizione eseguibile in hh_family_transfer.measurements','Direzione':'Minimizzare','Unità':'frazione di occupazione/conduzione' if key!='log_tau_rmse' else 'log(ms)','Parametri':str(threshold),'Casi degeneri':'Non finito sempre fallimento; OOD non compensato con dominio interno.'})
        evals.append(put('evaluations',key,**{'Nome':'Task18 — '+key,'Metrica':[metric],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(threshold),'Popolazione e regioni':'Gate m,h isolati; tre canali canonici; tutti i seed','Orizzonte e finestre':'dt0.025/0.1/0.5/1/5/25/100ms, V costante per singola transizione','Aggregazione e pesi':'Ogni modello rate-supervised deve passare; OOD riportato separatamente.'}))
    arms=[]
    for channel in cfg['channels']:
        for width in cfg['widths']:
            for obj in ('transition_only','rate_supervised'):
                arms.append(put('arms',f'{channel}-w{width}-{obj}',**{'Nome':f'Task18 {channel} w{width} {obj}','Ruolo':'Trattamento' if obj=='rate_supervised' else 'Controllo negativo','Protocollo':[protocol],'Descrizione':'MLP SiLU 1→w→w→4; inf sigmoid, tau exp(logtau), update esponenziale convesso; corrente/conduzione analitiche.','Configurazione residua':'Input V,m,h,dt identici; pesi distinti tra canali; inizializzazioni appaiate per seed; Adam/clipping indipendenti.'}))
    put('arms','lut',**{'Nome':'Task18 LUT2049 f64','Ruolo':'Baseline','Protocollo':[protocol],'Descrizione':'Interpolazione lineare separata per canale; solo in-support, nessuna estrapolazione nascosta.'})
    for i,(key,text) in enumerate(cfg['hypotheses'].items()):
        pred=put('predictions',key,**{'Nome':'Task18 — '+key,'Ipotesi':[claims[i]],'Origine':'Preregistrata','Risultato atteso':text,'Soglia o intervallo':json.dumps(cfg['gates']),'Specifica di valutazione':[evals[0]]})
        put('contrasts',key,**{'Nome':'Task18 — '+key,'Bracci':arms,'Predizioni':[pred],'Specifiche di valutazione':evals,'Contrasto e coefficienti':text,'Soglia interpretativa':'Criteri congiuntivi; rate-supervised meno transition-only per identifiabilità; width32 meno width16 e ladder per budget.','Confondenti controllati':'Seed, input, minibatch, optimizer; nessun peso condiviso.','Correzione confronti multipli':'No p-value o promozione post-hoc; soglie ingegneristiche preregistrate.'})
    for role,seeds in cfg['data_seeds'].items():
        if role=='minibatches':continue
        put('blocks',role,**{'Nome':'Task18 — '+role,'Seed':str(seeds),'Protocollo':[protocol],'Descrizione':'Ruoli disgiunti; fresh materializzato soltanto dopo freeze.','Regola di appaiamento':'Stesse tuple numeriche per canale, obiettivo, capacità e seed.'})
    put('artifacts','preregistration',**{'Nome':'Task18 — preregistrazione','Tipo':'Report','Percorso':'experiments/task18_hh_family_transfer.json','SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Versione':'v1','Dimensione byte':path.stat().st_size})
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    print(json.dumps(mirror.verify(),ensure_ascii=False))

if __name__=='__main__':main()
