"""IE-P4 local-only protocol, claims, models and paired arms."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;path=repo/'experiments/ie_p4_discrete_ionic_preregistration_v1.json'
    spec=json.loads(path.read_text());mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p4-{suffix}-v1',**fields})['record_id']
    models={}
    for name in ('formula_vectorized','lut025','neural_w32','neural_w64'):
        models[name]=put('models',name,Nome='IE-P4 '+name,Versione='v1',Ruolo='Surrogate' if 'neural' in name else 'Baseline',
            Descrizione=spec['model'] if 'neural' in name else 'Exact formula coefficients' if 'formula' in name else 'Linear interpolation of discrete u,z at .25mV',**{
                'Contratto di stato':'V,Ca,18 physical gates;17voltage-dependent gates replaced,SK kept canonical afterCa.',
                'Contratto di input e output':spec['input_contract']})
    claims=[]
    for name,statement in spec['hypotheses'].items():
        claims.append(put('claims',name,Nome='IE-P4 '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,
            Descrizione='Prospective, not an achieved result. Same paired seed/minibatches, fixed budget.',**{
                'Modelli oggetto':list(models.values()),'Condizioni di falsificazione':'Evaluate registered quality limits, paired fresh/path errors and matched cost; no scope expansion after fresh.',
                'Limiti':spec['exclusions']}))
    experiment=put('experiments','discrete',Nome='IE-P4 joint discrete ionic coefficient matrix',Stato='Preregistrato',
        Descrizione=spec['scope'],Ipotesi=claims,**{'Obiettivo informativo':spec['decision']})
    protocol=put('protocols','discrete',Nome='IE-P4 paired17gate coefficients and supervision',Versione='v1',
        Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec,ensure_ascii=False),**{
            'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),
            'Criteri di successo':json.dumps(spec['quality'])+'; atb642 >=10% paired saving with positiveCI; all seeds/families, no automatic promotion.',
            'Regole di arresto':spec['validation']})
    artifact=None
    for relative in ('experiments/ie_p4_discrete_ionic_preregistration_v1.json','experiments/ie_p4_design_v1.md',
                     'src/giada_teacher/ie_discrete_ionic.py','scripts/run_ie_p4_discrete_ionic.py',
                     'notebooks/ie_p4_discrete_ionic.ipynb','tests/test_ie_discrete_ionic.py'):
        p=repo/relative
        item=put('artifacts',p.stem,Nome='IE-P4 '+p.name,Tipo='Report',Percorso=relative,Versione='v1',**{
            'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size})
        if relative.endswith('preregistration_v1.json'):artifact=item
    put('protocols','discrete',Nome='IE-P4 paired17gate coefficients and supervision',Versione='v1',**{'Artefatto preregistrazione':[artifact]})
    for name in spec['arms']:
        model=name if name in models else 'neural_'+name.split('_')[-1]
        put('arms',name,Nome='IE-P4 '+name,Ruolo='Baseline' if name in ('formula_vectorized','lut025') else 'Trattamento',Protocollo=[protocol],Modello=[models[model]],
            **{'Configurazione residua':spec['loss']+';3seeds' if name not in ('formula_vectorized','lut025') else 'No training; same exact vectorized readout and downstream phases.'})
    put('actions','execute',Nome='Execute IE-P4 joint coefficient matrix in one notebook',Stato='Pronta',Descrizione=spec['decision']+' '+spec['exclusions'])
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'preregistered':True,'airtable_accessed':False}))

if __name__=='__main__':main()
