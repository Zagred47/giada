"""Link post-hoc residual hypotheses to current evidence and scoped precedents."""
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    m=ValidatedBatchMirror()
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    current=ref('findings-giada-task39b-architecture-outcome-v1')
    mapping={
        'dynamic_information':['finding-task2b-h-physical-tau-identifiability-repaired-v1','findings-giada-roadmap-task21-duration-v1','findings-giada-roadmap-task24b-rate_attribution-v1'],
        'optimization_budget':['finding-task1b-budget-limited-engineering-go-v1','finding-task3b-symmetric-rates-stabilize-shared-mh-v1','findings-giada-roadmap-task24b-budget_capacity-v1'],
        'local_support':['finding-task3d-voltage-support-conditional-shape-repair-v1','finding-task3e-full-repair-sealed-generalization-v1'],
        'shared_capacity':['findings-giada-roadmap-task26-gradient_interference-v1','findings-giada-roadmap-task24b-repair-and-sharing-v1'],
        'loss_conditioning':['findings-giada-ie-p5-outcome-v1','findings-giada-ie-p6-outcome-v1'],
        'numerical_timing':['findings-giada-roadmap-task24b-precision-v1','findings-giada-roadmap-task24b-clipping-v1']}
    ids=[]
    for name,codes in mapping.items():
        ids.append(m.local_upsert('evidence',{'Codice stabile':f'evidence-giada-task39b-architecture-residual-{name}-v1',
            'Nome':'After39b exploratory precedent: '+name,'Esito':'Indeterminata',
            'Affermazione valutata':[ref(f'claims-giada-task39b-architecture-residual-{name}-v1')],
            'Risultati a sostegno':[current]+[ref(c) for c in codes],
            'Argomentazione':'Exploratory analogy and current diagnostics motivate a discriminating intervention. See task39b_result_and_causal_next_design.md. '+json.dumps(codes),
            'Limiti e spiegazioni alternative':'Historical domains differ. These links do not assert that the current causal explanation has been demonstrated. Current hypothesis remains open.',
            'Data valutazione':'2026-10-18'})['record_id'])
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(ROOT.parent/'experiments/results/task39b_kaggle_v2/residual_links_receipt.json',dict(valid=True,evidence_ids=ids,verification=v,airtable_accessed=False))
    print(json.dumps(dict(valid=True,evidence_links=len(ids),verification=v)),flush=True)


if __name__=='__main__':main()
