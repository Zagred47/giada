"""Task24b preregistration and causal contrasts in the exclusive mirror."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    path=ROOT.parent/'experiments/task24b_sodium_control_diagnosis.json'
    cfg=json.loads(path.read_text());base=json.loads((ROOT.parent/'experiments/task24_sodium_family_composition.json').read_text())
    m=ValidatedBatchMirror()
    assert m.query("SELECT record_id FROM v_records WHERE stable_code='decisions-giada-roadmap-task24-to24b-v1'")['rows']
    def put(table,key,**fields):return m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task24b-{key}-v1',**fields})['record_id']
    claims={k:put('claims',k,**{'Nome':'Task24b '+k,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':v,'Limiti':cfg['limits'],'Condizioni di falsificazione':v}) for k,v in cfg['hypotheses'].items()}
    experiment=put('experiments','matrix',**{'Nome':'GIADA Task24b — independent sodium diagnosis','Tipo':'Esplorativo','Stato':'Preregistrato','Ipotesi':list(claims.values()),'Descrizione':cfg['training'],'Obiettivo informativo':'Factorial optimization/clipping,width,budget; rate vs numerical drift; selection-horizon mismatch.'})
    protocol=put('protocols','paired',**{'Nome':'Task24b paired2x2','Esperimento':[experiment],'Versione':'v1','Modalità':'Componente isolato','Procedura':json.dumps(cfg),'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),'Regole di arresto':'Oracle/hash/equivalence failure => execution error. No test-driven promotion or threshold changes.'})
    evaluations=[]
    for group in ('gates','pair_gates','held_gates','path_gates','path_channel_gates'):
        for key,limit in base[group].items():
            metric=put('metrics',group+'-'+key,**{'Nome':'Task24b '+group+' '+key,'Famiglia':'Regressione','Direzione':'Minimizzare','Unità':'adimensionale','Formula':'Unchanged Task24 metric, new disjoint fresh population.'})
            evaluations.append(put('evaluations',group+'-'+key,**{'Nome':'Task24b '+group+' '+key,'Metrica':[metric],'Protocollo':[protocol],'Versione':'v1','Ruolo':'Primaria','Target':str(limit),'Aggregazione e pesi':'Conjunction perchannel/perseed/domain; developmentcomposed held andfresh iterativeheld explicitlyseparated.'}))
    arms={}
    for arm in cfg['arms']:
        for width in cfg['widths']:
            model=put('models',arm+'-w'+str(width),**{'Nome':f'Task24b independent w{width} '+arm,'Ruolo':'Surrogate','Versione':'v1','Descrizione':f'3 separate1-{width}-{width}-4MLPs, {3*(width**2+7*width+4)} parameters. Onlyoptimizer/clipping changes.','Contratto di stato':'Six sodium m/h occupancies; convex exponential update, analyticcurrent.','Contratto di input e output':'V to inf/tau; startstate/dt enter solver only.'})
            arms[arm,width]=put('arms',arm+'-w'+str(width),**{'Nome':f'Task24b {arm} w{width}','Ruolo':'Baseline' if arm=='constant_bundle' else 'Trattamento','Protocollo':[protocol],'Modello':[model],'Descrizione':cfg['training'],'Configurazione residua':'Matched tuples/weights/Adam; boundedstdout; independentseed moments.'})
    for key,claim in claims.items():
        prediction=put('predictions',key,**{'Nome':'Task24b '+key,'Ipotesi':[claim],'Origine':'Preregistrata','Risultato atteso':cfg['hypotheses'][key],'Specifica di valutazione':[evaluations[0]],'Soglia o intervallo':cfg['hypotheses'][key]})
        put('contrasts',key,**{'Nome':'Task24b '+key,'Bracci':list(arms.values()),'Predizioni':[prediction],'Specifiche di valutazione':evaluations,'Contrasto e coefficienti':cfg['hypotheses'][key],'Confondenti controllati':cfg['training'],'Soglia interpretativa':cfg['hypotheses'][key],'Correzione confronti multipli':'Engineering thresholds, primaryarm fixedbeforefresh; otherarmsdiagnostic.'})
    for role in ('fit','development','fresh'):
        put('blocks',role,**{'Nome':'Task24b '+role,'Protocollo':[protocol],'Seed':json.dumps(cfg['fresh_seeds'] if role=='fresh' else base['data_seeds'][role]),'Descrizione':cfg['selection'],'Regola di appaiamento':'Same streams acrossallarms; no oldfresh training/selection.'})
    put('artifacts','preregistration',**{'Nome':'Task24b preregistration','Tipo':'Report','Percorso':path.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'})
    for p in (ROOT.parent/'experiments/preflight/task24b_sodium_control').glob('*.json'):
        put('artifacts','preflight-'+p.stem,**{'Nome':'Task24b preflight '+p.stem,'Tipo':'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,arms=len(arms),claims=len(claims),airtable_accessed=False)))


if __name__=='__main__':main()
