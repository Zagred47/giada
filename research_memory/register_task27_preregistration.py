"""Preregister Task27 in the exclusive SQLite research mirror."""
import hashlib
import json

from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    from src.giada_teacher.current_supervision_comparison import config,verify_parent
    cfg=config(repo);widths,hashes=verify_parent(repo,cfg)
    path=repo/'experiments/task27_current_supervision.json'
    mirror=ValidatedBatchMirror()
    assert mirror.query("SELECT record_id FROM v_records WHERE stable_code='experiments-giada-roadmap-task26-matrix-v1'")['rows']
    def put(table,key,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task27-{key}-v1',**fields})['record_id']
    claims={key:put('claims',key,**{'Nome':'Task27 '+key,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':value,
                                   'Limiti':cfg['limits'],'Condizioni di falsificazione':value})
            for key,value in cfg['hypotheses'].items()}
    experiment=put('experiments','matrix',**{'Nome':'GIADA Task27 — individual and total current supervision',
        'Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':list(claims.values()),
        'Descrizione':cfg['loss'],'Obiettivo informativo':'Distinguere benefici di current supervision, masking del totale e dipendenza dalla condivisione dei parametri.'})
    protocol=put('protocols','factorial',**{'Nome':'Task27 2 famiglie x4 loss x3 seed','Esperimento':[experiment],
        'Versione':'v1','Modalità':'Ricomposizione','Procedura':json.dumps(cfg,ensure_ascii=False),
        'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto':'Prerequisiti/native/equivalenza/hash falliti: errore tecnico. Fresh oltre soglia: NO-GO scientifico; nessuna riselezione.'})
    evaluations=[]
    for group in ('gates','held_gates','current_gates','path_gates','path_current_gates'):
        for key,limit in cfg[group].items():
            metric=put('metrics',group+'-'+key,**{'Nome':'Task27 '+group+' '+key,'Famiglia':'Regressione',
                'Direzione':'Minimizzare','Unità':'adimensionale',
                'Formula':'Per-canale gate/rate/open o corrente analitica normalizzata al driving fully-open; massimo pannello.'})
            evaluations.append(put('evaluations',group+'-'+key,**{'Nome':'Task27 '+group+' '+key,
                'Metrica':[metric],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(limit),
                'Aggregazione e pesi':'Congiunzione per seed/canale/dominio/held/path; nessuna compensazione fra canali.'}))
    arms=[]
    for family in cfg['families']:
        width=widths[family]
        for arm in cfg['supervision_arms']:
            key=f'{family}-{arm}'
            model=put('models',key,**{'Nome':'Task27 '+key,'Ruolo':'Baseline' if arm=='none' else 'Surrogate',
                'Versione':'v1','Descrizione':f'Width {width} fissato da Task26. Cinque head rate, due layer SiLU. Corrente analitica.',
                'Contratto di stato':'Dieci gate m/h; update esponenziale convesso per canale.',
                'Contratto di input e output':'V, dieci stati, dt; MLP usa soltanto V; formula di corrente deterministica.'})
            arms.append(put('arms',key,**{'Nome':'Task27 '+key,'Ruolo':'Baseline' if arm=='none' else 'Trattamento',
                'Protocollo':[protocol],'Modello':[model],'Descrizione':cfg['loss'],
                'Configurazione residua':f'Loss arm {arm}; stessa inizializzazione, fit, minibatch e schedule; pesi della current loss preregistrati.'}))
    for key,claim in claims.items():
        prediction=put('predictions',key,**{'Nome':'Task27 '+key,'Ipotesi':[claim],'Origine':'Preregistrata',
            'Risultato atteso':cfg['hypotheses'][key],'Specifica di valutazione':[evaluations[0]],
            'Soglia o intervallo':cfg['comparison']})
        put('contrasts',key,**{'Nome':'Task27 '+key,'Bracci':arms,'Predizioni':[prediction],
            'Specifiche di valutazione':evaluations,'Contrasto e coefficienti':cfg['comparison'],
            'Confondenti controllati':cfg['loss'],'Soglia interpretativa':cfg['comparison'],
            'Correzione confronti multipli':'Endpoint primario preregistrato; altri contrasti diagnostici. Selezione solo development.'})
    for role in ('fit','development','fresh'):
        seeds=cfg['fresh_seeds'] if role=='fresh' else cfg['data_seeds'][role]
        put('blocks',role,**{'Nome':'Task27 '+role,'Protocollo':[protocol],'Seed':json.dumps(seeds),
            'Descrizione':cfg['selection'],'Regola di appaiamento':'Stessi stream per tutte le otto configurazioni; nuovi 270xxx disgiunti da Task26.'})
    for p in (path,repo/'notebooks/27_roadmap_current_supervision.ipynb',
              repo/'scripts/run_roadmap_task27.py'):
        put('artifacts',p.stem,**{'Nome':'Task27 '+p.name,'Tipo':'Report','Percorso':p.relative_to(repo).as_posix(),
            'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})
    put('decisions','scope',**{'Nome':'Task27 attivata dopo GO Task26','Esito':'Continuare',
        'Motivazione':'Task26 3/3 independent e 3/3 shared; 644 vs6260 parametri. Current loss factorial con correnti analitiche.',
        'Condizioni di revisione':cfg['promotion']+' Hash parent: '+json.dumps(hashes,sort_keys=True)})
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'arms':len(arms),'claims':len(claims),'airtable_accessed':False}))


if __name__=='__main__':main()
