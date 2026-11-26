import json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;F=repo/'experiments/results/task53_exploratory_v1'
    prior=json.loads((repo/'experiments/results/task52j_protocol_v1/preregistration_receipt.json').read_text())
    audit=json.loads((F/'source_audit.json').read_text())
    m=IncrementalBatchMirror(); assert m.baseline_hash==prior['verification']['snapshot_hash']
    ids={}
    def put(table,key,**fields):
        code=f'{table}-giada-task53-{key}-v1'
        ids[key]=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];return ids[key]
    old=prior['records']['experiments-giada-task52i-audit52j-52j-v1']
    obs=put('observations','52j-result',Nome='52j completed; no scientific acceptance',Descrizione=json.dumps(audit))
    finding=put('findings','52j-finding',Nome='512-tick guard modestly improves RMSE, no promotion',Esito='Misto',Esperimenti=[old],Osservazioni=[obs],Risultato='0.716288 vs 0.718383 mV at 1000 ms; 0/36 composite passes at each horizon. V/Ca guard did not improve final metrics.',Limitazioni='Reused panel; coupled quality still fails; no independent confirmation.')
    m.local_upsert('experiments',{'Stato':'Concluso'},record_id=old)
    for code,rid in prior['records'].items():
        if code.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    exp=put('experiments','exploration',Nome='53 exploratory frozen level-2 transfer',Tipo='Esplorativo',Stato='Preregistrato',Descrizione='52j long_current indices 3,4,5 frozen. Matched canonical FP64/FP32, P, autonomous and imposed compact; two seeded stimulus variants within existing families. No training or reselection.',**{'Obiettivo informativo':'Measure transfer quality and atomic cost separately at unchanged thresholds and 10 logical states.'})
    put('protocols','protocol',Nome='53 frozen comparison on new stimulus variants',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura='New timing and amplitude variants, original solver, interventions, quality metrics and thresholds. Freeze before evaluation.',**{'Criteri di successo':'Report original composite criteria unchanged; no automatic promotion.','Regole di arresto':'Stop if provenance, finite parameters or canonical parity fails.'})
    put('decisions','transition',Nome='Close 52; proceed with 53 exploratory only',Esito='Continuare',Motivazione='User-approved transition. No 52k, no threshold or abstraction change. Reused families and control solver limit validation; no whole-neuron speed claim.',Risultati=[finding])
    m.commit_batch();v=m.verify_incremental(prior['verification']);assert v['valid']
    write_json(F/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False))
if __name__=='__main__':main()
