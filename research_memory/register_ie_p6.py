"""Preregister IE-P6 hypotheses, contrasts and executable artifacts locally."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;path=repo/'experiments/ie_p6_preregistration_v1.json';spec=json.loads(path.read_text())
    assert hashlib.sha256((repo/spec['parent_checkpoint']).read_bytes()).hexdigest()==spec['parent_sha256']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p6-{suffix}-v1',**fields})['record_id']
    models={}
    for name,description in {
        'mlp64':'Own-seed P5standard parent; same64width across factorial arms;18physical states.',
        'local81':'81knot learned normalized u,z values, convex interpolation, projected bounds. Not fewer parameters by assumption.',
        'local161':'161knot learned local operator, paired initialization/data with81knots; different projection errors disclosed.',
        'formula_vectorized':'Exact17voltage-gate coefficients, fixed physical readout and solver.',
        'lut025':'Analytic coefficient table control; not neural predictions.',
        'neural_table025':'Frozen .25mV deployment of each selected MLP, no analytic coefficient substitution.'}.items():
        models[name]=put('models',name,Nome='IE-P6 '+name,Versione='v1',Ruolo='Baseline' if name in ('formula_vectorized','lut025') else 'Surrogate',Descrizione=description,**{'Contratto di stato':spec['input_contract'],'Contratto di input e output':spec['input_contract']})
    claims=[]
    for key,statement in spec['hypotheses'].items():
        claims.append(put('claims',key,Nome='IE-P6 '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,**{'Modelli oggetto':list(models.values()),'Condizioni di falsificazione':spec['contrasts'],'Limiti':spec['exclusions']}))
    experiment=put('experiments','repair',Nome='IE-P6 coverage recurrence and learned local operator',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['scope'],**{'Obiettivo informativo':spec['contrasts']})
    protocol=put('protocols','repair',Nome='IE-P6 paired factorial and local representation protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec),**{'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),'Criteri di successo':json.dumps(spec['quality'])+' AND >=10%primary saving, positive paired interval; no promotion.','Regole di arresto':spec['validation']})
    for relative in ('experiments/ie_p6_preregistration_v1.json','experiments/ie_p6_design_v1.md','src/giada_teacher/ie_residual_repair.py','scripts/run_ie_p6_residual_repair.py','notebooks/ie_p6_residual_repair.ipynb','tests/test_ie_residual_repair.py'):
        p=repo/relative
        artifact=put('artifacts',p.name,Nome='IE-P6 '+p.name,Tipo='Report',Versione='v1',Percorso=relative,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size})
        if relative.endswith('preregistration_v1.json'):put('protocols','repair',Nome='IE-P6 paired factorial and local representation protocol',**{'Artefatto preregistrazione':[artifact]})
    for name in spec['arms']:
        model=name if name in models else 'mlp64'
        put('arms',name,Nome='IE-P6 '+name,Ruolo='Baseline' if name in ('A_uniform','parent','formula_vectorized','lut025') else 'Trattamento',Protocollo=[protocol],Modello=[models[model]],**{'Configurazione residua':spec['matrix']+' '+spec['local_representation']+' '+spec['deployment']})
    put('arms','oracle',Nome='IE-P6 P5parent oracle interventions',Ruolo='Oracle diagnostico',Protocollo=[protocol],**{'Configurazione residua':spec['attribution']})
    put('actions','execute',Nome='Execute IE-P6 paired repair and local representation notebook',Stato='Pronta',Descrizione=spec['validation']+' '+spec['exclusions'])
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'preregistered':True,'hypotheses':len(claims),'airtable_accessed':False}))

if __name__=='__main__':main()
