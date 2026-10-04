"""Register original Task25; heterogeneous frozen composition, not Task26."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    path=ROOT.parent/'experiments/task25_heterogeneous_mechanisms.json';cfg=json.loads(path.read_text());m=ValidatedBatchMirror()
    assert m.query("SELECT record_id FROM v_records WHERE stable_code='decisions-giada-roadmap-task24b-to25-v1'")['rows']
    def put(table,key,**fields):return m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task25-{key}-v1',**fields})['record_id']
    claims={key:put('claims',key,**{'Nome':'Task25 '+key,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':value,'Limiti':cfg['limits'],'Condizioni di falsificazione':value}) for key,value in cfg['hypotheses'].items()}
    exp=put('experiments','matrix',**{'Nome':'GIADA Task25 — heterogeneous frozen composition','Tipo':'Confermativo','Stato':'Preregistrato','Ipotesi':list(claims.values()),'Descrizione':cfg['matrix'],'Obiettivo informativo':'Composability without retraining; per-channel vs total masking; reuse of validated compression; numerical vs rate error.'})
    protocol=put('protocols','paired',**{'Nome':'Task25 paired frozen2x2','Esperimento':[exp],'Versione':'v1','Modalità':'Ricomposizione','Procedura':json.dumps(cfg),'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),'Regole di arresto':'Native/hash/contract failure: technical error. Valid run with failed gates: scientificNO-GO. No fresh selection.'})
    evaluations=[]
    for group in ('gates','held_gates','current_gates','path_gates','path_current_gates'):
        for key,limit in cfg[group].items():
            metric=put('metrics',group+'-'+key,**{'Nome':'Task25 '+group+' '+key,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'adimensionale','Formula':'Per-channel gate/rate/open error, or worst individual/panel fully-open-driving normalized current error; no division by cancelling total.'})
            evaluations.append(put('evaluations',group+'-'+key,**{'Nome':'Task25 '+group+' '+key,'Metrica':[metric],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(limit),'Aggregazione e pesi':'Conjunction all5mechanisms,3seeds,domains,heldhorizons and each fast/slow path; never pooled-only promotion.'}))
    arms=[]
    for arm,count in [('independent',1860),('calcium_compressed',1556),('sodium_compressed',1164),('both_compressed',860)]:
        model=put('models',arm,**{'Nome':'Task25 '+arm,'Ruolo':'Baseline' if arm=='independent' else 'Surrogate','Versione':'v1','Descrizione':f'Frozen composite {count}parameters; no new weights or optimizer.','Contratto di stato':'Ten m/h occupancies in canonical channel order; independent convex exponential updates.','Contratto di input e output':'Common V,10states,dt. Rate MLP sees V only (plus identity for conditioned). Currents analytic p2Ca/p3Na; supplied reversal/gbar.'})
        arms.append(put('arms',arm,**{'Nome':'Task25 '+arm,'Ruolo':'Baseline' if arm=='independent' else 'Trattamento','Protocollo':[protocol],'Modello':[model],'Descrizione':cfg['matrix'],'Configurazione residua':cfg['freeze']}))
    for key,claim in claims.items():
        pred=put('predictions',key,**{'Nome':'Task25 '+key,'Ipotesi':[claim],'Origine':'Preregistrata','Risultato atteso':cfg['hypotheses'][key],'Specifica di valutazione':[evaluations[0]],'Soglia o intervallo':cfg['hypotheses'][key]})
        put('contrasts',key,**{'Nome':'Task25 '+key,'Bracci':arms,'Predizioni':[pred],'Specifiche di valutazione':evaluations,'Contrasto e coefficienti':cfg['hypotheses'][key],'Confondenti controllati':cfg['matrix'],'Soglia interpretativa':cfg['promotion'],'Correzione confronti multipli':'Primary independent fixed; otherarms/precision/identity contrasts cannot bypass referenceNO-GO.'})
    put('blocks','fresh',**{'Nome':'Task25 matched fresh2504xx','Protocollo':[protocol],'Seed':json.dumps(cfg['fresh_seeds']),'Descrizione':cfg['extrema_support'],'Regola di appaiamento':'Exact same tuples and paths across all frozenarms/seeds. Endpoints/seams intentionally repeated, offgridV new.'})
    for p in [path,ROOT.parent/'experiments/task25_design_notes.md',*list((ROOT.parent/'experiments/preflight/task25_heterogeneous').glob('*.json'))]:
        put('artifacts',p.stem,**{'Nome':'Task25 '+p.name,'Tipo':'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})
    for source in ('calcium_source','sodium_source','sodium_shared_source'):
        folder=ROOT.parent/cfg[source]
        for p in [folder/'selection_freeze.json',*list(folder.glob('checkpoint_*.pt'))]:
            put('artifacts',source+'-'+p.stem,**{'Nome':'Task25 prerequisite '+p.name,'Tipo':'Checkpoint' if p.suffix=='.pt' else 'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})
    text='Held-aware and one-step-only development selections are identical for all4arms at60k. This run provides no checkpoint-choice change attributable to added held-aware selection. Primary rule confirmed3/3; selection mismatch remains an unproven explanation, not a repair attribution.'
    m.local_upsert('findings',{'Codice stabile':'findings-giada-roadmap-task24b-selection-v1','Risultato':text})
    m.local_upsert('evidence',{'Codice stabile':'evidence-giada-roadmap-task24b-selection-v1','Argomentazione':text})
    put('decisions','scope',**{'Nome':'Task25 scope and next dependency','Esito':'Continuare','Motivazione':'Test frozenheterogeneous composition now. Do not collapseTask26joint training orTask27current objectives intoTask25. Calcium-input mechanisms await explicitdependencyvalidation.','Condizioni di revisione':cfg['promotion']})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,arms=4,claims=len(claims),airtable_accessed=False)))


if __name__=='__main__':main()
