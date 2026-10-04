"""Read-only evidence-chain audit after Task25 result registration."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_memory.mirror import Mirror, ROOT


def main():
    m = Mirror()
    counts = m.query("SELECT table_key,COUNT(*) AS count FROM v_records WHERE stable_code LIKE '%-giada-roadmap-task25-%' GROUP BY table_key", limit=100)['rows']
    table = m.contract.tables['findings']['name']
    links = m.query('SELECT "Codice stabile" AS code,json_array_length("Osservazioni") AS count FROM "' + table + '" WHERE "Codice stabile" LIKE \'findings-giada-roadmap-task25-%\'', limit=100)['rows']
    assert len(links) == 6 and all(row['count'] > 0 for row in links)
    claims = m.query("SELECT stable_code,name FROM v_records WHERE table_key='claims' AND stable_code LIKE 'claims-giada-roadmap-task25-%'", limit=100)['rows']
    assert len(claims) == 6
    result = json.loads((ROOT.parent / 'experiments/results/task25_kaggle_bdb75e6/final_report.json').read_text())
    assert result['valid'] and result['composition_passed'] and result['task26_preparation_authorized']
    verification = m.verify()
    assert verification['valid']
    report = dict(valid=True, record_counts=counts, findings_observation_links=links, claims=claims, sqlite_verification=verification, airtable_accessed=False)
    dest = ROOT.parent / 'experiments/results/task25_kaggle_bdb75e6/memory_audit.json'
    dest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(valid=True, record_counts=counts, findings_observation_links=links)))


if __name__ == '__main__':
    main()
