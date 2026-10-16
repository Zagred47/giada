"""Preregister frozen IE-P7 confirmation and operational perimeter locally."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;path=repo/'experiments/ie_p7_preregistration_v1.json';spec=json.loads(path.read_text(encoding='utf-8'))
    assert hashlib.sha256((repo/spec['checkpoint']).read_bytes()).hexdigest()==spec['checkpoint_sha256']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p7-{suffix}-v1',**fields})['record_id']
    models={}
    for name,description in {'local81':'All3 IE-P6frozen learned local81operators; no training or seed selection.','local161':'All3 IE-P6frozen learned local161operators; no training or seed selection.','formula_vectorized':'Exact17coefficients, symmetric optimized readout and membrane solve.','lut025':'Exact-coefficient .25mV LUT frontier, not learned predictions.','hybrid_independent':'All3original Task35frozen independent hybrids, unpadded single-seed inference.','hybrid_shared_heads':'All3original Task35frozen shared-head hybrids, unpadded single-seed inference.'}.items():
        models[name]=put('models',name,Nome='IE-P7 '+name,Versione='v1',Ruolo='Surrogate' if name.startswith('local') else 'Baseline',Descrizione=description,**{'Contratto di stato':'SameV/Ca/18gates and declared canonical stochastic synaptic state, dt.1ms.','Contratto di input e output':spec['operational_scope']})
    claims=[]
    for key,text in spec['hypotheses'].items():
        claims.append(put('claims',key,Nome='IE-P7 '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=text,**{'Modelli oggetto':list(models.values()),'Condizioni di falsificazione':spec['decision'],'Limiti':spec['exclusions']}))
    exp=put('experiments','confirmation',Nome='IE-P7 frozen native confirmation and operational cost',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['scope'],**{'Obiettivo informativo':spec['decision']})
    protocol=put('protocols','confirmation',Nome='IE-P7 calibration native and staged-cost protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),**{'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),'Criteri di successo':spec['native_quality']+' '+spec['decision'],'Regole di arresto':spec['exclusions']})
    for relative in ('experiments/ie_p7_preregistration_v1.json','experiments/ie_p7_design_v1.md','src/giada_teacher/ie_native_operational_confirmation.py','scripts/run_ie_p7_confirmation.py','notebooks/ie_p7_native_operational.ipynb','tests/test_ie_native_operational_confirmation.py'):
        p=repo/relative
        artifact=put('artifacts',p.name,Nome='IE-P7 '+p.name,Tipo='Report',Versione='v1',Percorso=relative,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size})
        if relative.endswith('preregistration_v1.json'):put('protocols','confirmation',Nome='IE-P7 calibration native and staged-cost protocol',**{'Artefatto preregistrazione':[artifact]})
    for name,model in models.items():
        put('arms',name,Nome='IE-P7 '+name,Ruolo='Trattamento' if name.startswith('local') else 'Baseline',Protocollo=[protocol],Modello=[model],**{'Configurazione residua':spec['freeze']+' '+spec['baselines']})
    put('findings','perimeter-clarification',Nome='IE-P7 clarification of existing IE-P6 timing perimeter',Esito='Misto',**{'Risultato':'IE-P6 timing already included gates,calcium,vectorized ionic readout and membrane solver with resident prescribed injection/synaptic conductances. It was not an isolated gate-block latency. New operational experiment adds independent event/RNG/receptor generation, IPC, H2D, causal multi-step state mutation and D2H.','Limitazioni':'No new empirical outcome in this clarification. CPU orchestration is one implementation, not an intrinsic lower bound or universal end-to-end cost.'})
    put('actions','execute',Nome='Execute IE-P7 frozen confirmation and operational cost notebook',Stato='Pronta',Descrizione=spec['decision']+' '+spec['exclusions'])
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'preregistered':True,'hypotheses':len(claims),'airtable_accessed':False}))

if __name__=='__main__':main()
