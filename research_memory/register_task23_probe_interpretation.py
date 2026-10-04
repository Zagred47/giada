"""Close Task23 identity/scaling reasoning with explicitly scoped diagnostics."""
import json
import statistics
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    report=json.loads((ROOT.parent/'experiments/results/task23_kaggle_0f79081/final_report.json').read_text())
    scaling=json.loads((ROOT.parent/'experiments/results/task23_kaggle_0f79081/paired_scaling_contrasts.json').read_text())
    m=ValidatedBatchMirror()
    def ref(table,code):return m.query("SELECT record_id FROM v_records WHERE stable_code='"+f'{table}-giada-roadmap-task23-{code}-v1'+"'")['rows'][0]['record_id']
    def put(table,code,**fields):return m.local_upsert(table,{'Codice stabile':f'{table}-giada-roadmap-task23-{code}-v1',**fields})['record_id']
    minimum=min(r['swapped_identity_rmse_ratio'] for r in report['learned'])
    assert minimum>=10
    finding=put('findings','identity',**{'Nome':'Task23 channel identity used by learned model','Esito':'Positivo','Esperimenti':[ref('experiments','matrix')],'Risultato':f'All9seed/family routing swaps exceed registered10x deterioration. Minimumratio={minimum:.6f}.','Limitazioni':'Frozen imposedV probe, not evidence that conditioned architecture passes.'})
    put('evidence','identity',**{'Nome':'Task23 identity negativecontrol','Esito':'Sostiene','Affermazione valutata':[ref('claims','identity')],'Risultati a sostegno':[finding],'Argomentazione':'Swapped-routing probe >=10x in all9 combinations, preregistered diagnostic.'})
    put('claims','identity',**{'Stato':'Supportata nel dominio'})
    medians={f:{factor:statistics.median([r['improvement_fraction'] for r in scaling if r['family']==f and r['factor']==factor]) for factor in ('budget','width')} for f in ('independent','shared_heads','conditioned')}
    finding=put('findings','scaling-identity',**{'Nome':'Task23 mini scaling laws: heterogeneous capacity benefit','Esito':'Misto','Esperimenti':[ref('experiments','matrix')],'Risultato':json.dumps(medians)+'; development medians only, not fresh confirmation. Budget medians positive for allfamilies; conditioned width median ~0, others>10%.','Limitazioni':'Aggregated descriptive contrasts; initialization/architecture and selected checkpoint differ. No universal inference.'})
    put('evidence','scaling',**{'Nome':'Task23 scaling heterogeneous','Esito':'Indeterminata','Affermazione valutata':[ref('claims','scaling')],'Risultati a sostegno':[finding],'Argomentazione':'Budget and width do not show oneuniform pattern acrossfamilies/steps; preserve full paired contrasts for further queries.'})
    put('claims','scaling',**{'Stato':'Indeterminata'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,minimum_identity_ratio=minimum,scaling_medians=medians,airtable_accessed=False)))

if __name__=='__main__':main()
