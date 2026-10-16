"""Preregister local-only IE-P5 accuracy/deployment contrasts and limitations."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;path=repo/'experiments/ie_p5_accuracy_execution_preregistration_v1.json'
    spec=json.loads(path.read_text());assert hashlib.sha256((repo/spec['parent_checkpoint']).read_bytes()).hexdigest()==spec['parent_sha256']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p5-{suffix}-v1',**fields})['record_id']
    models={}
    for name,description in {
        'formula_vectorized':'Exact reference with vectorized readout, same17V-dependent and canonicalSK gates.',
        'lut025':'Analytic coefficient table, not neural predictions.',
        'mlp64':'Same IE-P4 MLP64 class, per-seed frozen or matched standard/balanced weight refit.',
        'neural_table025':'Linear interpolation of frozen neural outputs at.25mV. Numerical deployment, not a new learned architecture.',
        'neural_table1':'Linear interpolation of frozen neural outputs at1mV. Numerical deployment, not learned sparse knots.'}.items():
        models[name]=put('models',name,Nome='IE-P5 '+name,Versione='v1',Ruolo='Baseline' if name in ('formula_vectorized','lut025') else 'Surrogate',Descrizione=description,**{
            'Contratto di stato':'SameV,Ca,18physical gates, dt.1ms; no memory compression.',
            'Contratto di input e output':spec['input_contract']})
    claims=[]
    for name,statement in spec['hypotheses'].items():
        claims.append(put('claims',name,Nome='IE-P5 '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,
            Descrizione='Prospective paired contrasts, not achieved results.',**{
                'Modelli oggetto':list(models.values()),'Condizioni di falsificazione':'Report all paired seed differences, registered quality gates, conversion budget and matched resident timing; negative swaps retained.',
                'Limiti':spec['exclusions']}))
    experiment=put('experiments','accuracy-execution',Nome='IE-P5 accuracy repair and inference representation',Stato='Preregistrato',Ipotesi=claims,
        Descrizione=spec['scope'],**{'Obiettivo informativo':spec['decision']})
    protocol=put('protocols','accuracy-execution',Nome='IE-P5 matched refit and neural-tabulation protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec,ensure_ascii=False),**{
        'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),'Criteri di successo':json.dumps(spec['quality'])+'; conversion<=5e-5; >=10%saving atb642 with paired positiveCI; no operational promotion.',
        'Regole di arresto':spec['validation']})
    artifact=None
    for relative in ('experiments/ie_p5_accuracy_execution_preregistration_v1.json','experiments/ie_p5_design_v1.md',
                     'src/giada_teacher/ie_accuracy_execution.py','scripts/run_ie_p5_accuracy_execution.py',
                     'notebooks/ie_p5_accuracy_execution.ipynb','tests/test_ie_accuracy_execution.py'):
        p=repo/relative
        item=put('artifacts',p.stem,Nome='IE-P5 '+p.name,Tipo='Report',Percorso=relative,Versione='v1',**{
            'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size})
        if relative.endswith('preregistration_v1.json'):artifact=item
    put('protocols','accuracy-execution',Nome='IE-P5 matched refit and neural-tabulation protocol',**{'Artefatto preregistrazione':[artifact]})
    for name in spec['arms']:
        model=name if name in models else 'mlp64'
        put('arms',name,Nome='IE-P5 '+name,Ruolo='Baseline' if name in ('frozen64','formula_vectorized','lut025') else 'Trattamento',Protocollo=[protocol],Modello=[models[model]],**{
            'Configurazione residua':spec['warm_start']+' '+spec['deployment']})
    put('arms','oracle-coefficients',Nome='IE-P5 diagnostic coefficient swaps',Ruolo='Oracle diagnostico',Protocollo=[protocol],**{'Configurazione residua':spec['attribution']+' Never cost/selection eligible.'})
    put('actions','execute',Nome='Execute IE-P5 matched accuracy and deployment experiment',Stato='Pronta',Descrizione=spec['decision']+' '+spec['exclusions'])
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'preregistered':True,'airtable_accessed':False}))

if __name__=='__main__':main()
