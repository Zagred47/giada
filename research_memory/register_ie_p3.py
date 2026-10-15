"""Local-only IE-P3 preregistration, before opening fresh atomic outcomes."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;path=repo/'experiments/ie_p3_readout_preregistration_v1.json'
    spec=json.loads(path.read_text());mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p3-{suffix}-v1',**fields})['record_id']
    experiment=put('experiments','readout',Nome='IE-P3 paired exact and learned ionic readout',Stato='Preregistrato',
        Descrizione=spec['scope'],**{'Obiettivo informativo':spec['decision']})
    protocol=put('protocols','readout',Nome='IE-P3 atomic aperture learnability and matched execution',Versione='v1',
        Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec,ensure_ascii=False),**{
        'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),
        'Criteri di successo':json.dumps(spec['quality'])+'; paired saving >=10% with positive CI versus vectorized algebra at b642; no deployment authorization.',
        'Regole di arresto':spec['validation']})
    artifacts=[]
    for relative in ('experiments/ie_p3_readout_preregistration_v1.json','experiments/ie_p3_design_v1.md',
                     'src/giada_teacher/ie_readout_comparison.py','scripts/run_ie_p3_readout.py',
                     'notebooks/ie_p3_readout.ipynb','tests/test_ie_readout_comparison.py'):
        p=repo/relative
        artifacts.append(put('artifacts',p.stem,Nome='IE-P3 '+p.name,Tipo='Report',Percorso=relative,Versione='v1',**{
            'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    for name in spec['arms']:
        put('arms',name,Nome='IE-P3 '+name,Ruolo='Baseline' if 'algebra' in name else 'Trattamento',Protocollo=[protocol])
    put('actions','execute',Nome='Execute IE-P3 single notebook with paired six-member training',Stato='Pronta',
        Descrizione=spec['decision']+' '+spec['exclusions'])
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'preregistered':True,'airtable_accessed':False}))

if __name__=='__main__':main()
