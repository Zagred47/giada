"""Preregister original Task26 in the exclusive local SQLite mirror."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    from src.giada_teacher.joint_heterogeneous_comparison import config,validate_prerequisite
    cfg=config(repo);validate_prerequisite(repo,cfg)
    path=repo/'experiments/task26_joint_vs_independent.json';m=ValidatedBatchMirror()
    assert m.query("SELECT record_id FROM v_records WHERE stable_code='decisions-giada-roadmap-task25-to26-v1'")['rows']
    def put(table,key,**fields):return m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task26-{key}-v1',**fields})['record_id']
    claims={key:put('claims',key,**{'Nome':'Task26 '+key,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':text,'Limiti':cfg['limits'],'Condizioni di falsificazione':text}) for key,text in cfg['hypotheses'].items()}
    exp=put('experiments','matrix',**{'Nome':'GIADA Task26 — joint vs independent','Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':list(claims.values()),'Descrizione':cfg['training'],'Obiettivo informativo':'Condivisione, capacità, budget, conflitto dei gradienti e attribuzione percanale; non ancora training sulle correnti.'})
    protocol=put('protocols','paired',**{'Nome':'Task26 famiglia xwidth e ladder','Esperimento':[exp],'Versione':'v1','Modalità':'Ricomposizione','Procedura':json.dumps(cfg,ensure_ascii=False),'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),'Regole di arresto':'Errore nativo/hash/equivalenza: fallimento tecnico. Congiunzione fresh non superata: NO-GO scientifico valido. Nessun cambio di soglie o selezione dopo test.'})
    evaluations=[]
    for group in ('gates','held_gates','current_gates','path_gates','path_current_gates'):
        for key,limit in cfg[group].items():
            metric=put('metrics',group+'-'+key,**{'Nome':'Task26 '+group+' '+key,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'adimensionale','Formula':'Errore percanale gate/rate/open o corrente normalizzata al driving pienamente aperto; peggior canale/pannello, non totale cancellante.'})
            evaluations.append(put('evaluations',group+'-'+key,**{'Nome':'Task26 '+group+' '+key,'Metrica':[metric],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(limit),'Aggregazione e pesi':'Congiunzione di cinque canali, tre seed, tutti domini, orizzonti e path; massimi pannelli di corrente.'}))
    arms=[]
    for family in cfg['families']:
        for width in cfg['widths']:
            key=family+'-w'+str(width);count=5*(width*width+7*width+4) if family=='independent' else width*width+23*width+20
            model=put('models',key,**{'Nome':'Task26 '+key,'Ruolo':'Baseline' if family=='independent' else 'Surrogate','Versione':'v1','Descrizione':f'{count} parametri perseed; due layer SiLU, cinque head rate; training da zero.','Contratto di stato':'Dieci occupanze m/h; update esponenziale convesso indipendente pergate.','Contratto di input e output':'Stesso V,dieci stati,dt. MLP usa soltanto V; output inf e tau; formula di corrente analitica post-update.'})
            arms.append(put('arms',key,**{'Nome':'Task26 '+key,'Ruolo':'Baseline' if family=='independent' else 'Trattamento','Protocollo':[protocol],'Modello':[model],'Descrizione':cfg['training'],'Configurazione residua':'Stesso fit/minibatch/LR/loss/clipping; seed e momentiAdam indipendenti; un optimizer foreach senza sharedmoments.'}))
    for key,claim in claims.items():
        pred=put('predictions',key,**{'Nome':'Task26 '+key,'Ipotesi':[claim],'Origine':'Preregistrata','Risultato atteso':cfg['hypotheses'][key],'Specifica di valutazione':[evaluations[0]],'Soglia o intervallo':cfg['hypotheses'][key]})
        put('contrasts',key,**{'Nome':'Task26 '+key,'Bracci':arms,'Predizioni':[pred],'Specifiche di valutazione':evaluations,'Contrasto e coefficienti':cfg['hypotheses'][key],'Confondenti controllati':cfg['training'],'Soglia interpretativa':cfg['hypotheses'][key],'Correzione confronti multipli':'Selezione preregistrata development; nessun probe diagnostico promuove, nessuna selezione fresh.'})
    for role in ('fit','development','fresh'):
        seeds=cfg['fresh_seeds'] if role=='fresh' else cfg['data_seeds'][role]
        put('blocks',role,**{'Nome':'Task26 '+role,'Protocollo':[protocol],'Seed':json.dumps(seeds),'Descrizione':cfg['selection'],'Regola di appaiamento':'Stream nuovi260xxx, identici tra famiglie/width/seedmodello; endpoint/seam fisici intenzionalmente ripetuti.'})
    for p in [path,repo/'experiments/task26_design_notes.md',*list((repo/'experiments/preflight/task26_joint').glob('*.json'))]:
        put('artifacts',p.stem,**{'Nome':'Task26 '+p.name,'Tipo':'Report','Percorso':p.relative_to(repo).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})
    put('decisions','scope',**{'Nome':'Task26 originale e dipendenze','Esito':'Continuare','Motivazione':'Task25 conferma componibilità indipendente. Confronto jointvsindependent dazero, non riparazione retroattiva dei modelliTask25.','Condizioni di revisione':cfg['promotion']})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,arms=4,claims=6,airtable_accessed=False)))


if __name__=='__main__':main()
