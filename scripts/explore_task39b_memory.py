"""Read-only, reproducible GIADA precedent queries; no Airtable access."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task39b_kaggle_v2'


def main():
    db = sqlite3.connect(f'file:{ROOT / "research_memory/data/research_memory.sqlite"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    schema = {t: [r[1] for r in db.execute(f'PRAGMA table_info({t})')]
              for t in ('claims', 'findings', 'lessons', 'evidence', 'analyses')}
    query = """SELECT table_key,record_id,stable_code,name FROM v_records
        WHERE table_key IN ('findings','lessons') AND stable_code LIKE '%giada%'
        AND (lower(name) LIKE '%identif%' OR lower(name) LIKE '%budget%'
          OR lower(name) LIKE '%generaliz%' OR lower(name) LIKE '%symmetric%'
          OR lower(name) LIKE '%precision%' OR lower(name) LIKE '%repair%'
          OR lower(name) LIKE '%task39%' OR lower(name) LIKE '%ie-p%')"""
    matches = [dict(r) for r in db.execute(query)]
    codes = [
        'finding-task1b-budget-limited-engineering-go-v1',
        'finding-task2b-h-physical-tau-identifiability-repaired-v1',
        'finding-task3b-symmetric-rates-stabilize-shared-mh-v1',
        'finding-task3d-voltage-support-conditional-shape-repair-v1',
        'finding-task3e-full-repair-sealed-generalization-v1',
        'findings-giada-roadmap-task21-duration-v1',
        'findings-giada-roadmap-task21-rates-v1',
        'findings-giada-roadmap-task24b-rate_attribution-v1',
        'findings-giada-roadmap-task24b-budget_capacity-v1',
        'findings-giada-roadmap-task24b-repair-and-sharing-v1',
        'findings-giada-roadmap-task24b-precision-v1',
        'findings-giada-roadmap-task24b-clipping-v1',
        'findings-giada-roadmap-task26-gradient_interference-v1',
        'findings-giada-ie-p5-outcome-v1',
        'findings-giada-ie-p6-outcome-v1']
    queries = {
        'precedent_detail': 'SELECT * FROM findings WHERE "Codice stabile" = ?',
        'evidence_chain': '''SELECT ec.*, c."Codice stabile" claim_code,c.Stato claim_status,
            e.Esito evidence_outcome,e.Argomentazione argument
            FROM v_evidence_chains ec JOIN claims c ON c._record_id=ec.claim_id
            JOIN evidence e ON e._record_id=ec.assessment_id WHERE ec.finding_id=?''',
        'task39b_cases': '''SELECT json_extract(j.value,'$.model') model,
            json_extract(j.value,'$.role') role, COUNT(*) cases,
            SUM(json_extract(j.value,'$.metrics.passed')) passed
            FROM observations o, json_each(o.Descrizione,'$.cases') j
            WHERE o."Codice stabile" LIKE 'observations-giada-task39b-architecture-result-%'
            GROUP BY model,role''',
        'task39b_failed_channels': '''SELECT json_extract(c.value,'$.channel') channel,
            COUNT(*) failures, AVG(json_extract(c.value,'$.nrmse')) mean_nrmse,
            MAX(json_extract(c.value,'$.rmse_ma_cm2')) worst_absolute_rmse_ma_cm2
            FROM observations o, json_each(o.Descrizione,'$.cases') j,
            json_each(j.value,'$.metrics.channels') c
            WHERE o."Codice stabile" LIKE 'observations-giada-task39b-architecture-result-%'
            AND json_extract(j.value,'$.model')='joint_wbase_h8_c1_r0'
            AND json_extract(j.value,'$.role')='development'
            AND json_extract(c.value,'$.passed')=0 GROUP BY channel ORDER BY failures DESC'''}
    details=[]
    for code in codes:
        row=db.execute(queries['precedent_detail'],(code,)).fetchone()
        assert row is not None,code
        item=dict(row)
        item['evidence_chains']=[dict(r) for r in db.execute(queries['evidence_chain'],(row['_record_id'],))]
        details.append(item)
    result = dict(schema=schema, discovery_sql=query, discovery=matches, queries=queries,
        precedent_codes=codes, precedents=details,
        task39b_cases=[dict(r) for r in db.execute(queries['task39b_cases'])],
        task39b_failed_channels=[dict(r) for r in db.execute(queries['task39b_failed_channels'])],
        limits='Read-only exploratory retrieval; selected precedents are analogies, not causal proof for Task39b. Legacy records may lack giada in stable code, so exact reconciled codes are included.')
    (OUT / 'memory_exploration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(precedents=len(details),chains=sum(len(x['evidence_chains']) for x in details),
        task39b_cases=result['task39b_cases'],task39b_failed_channels=result['task39b_failed_channels']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
