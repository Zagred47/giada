"""Read-only older GIADA records, including pre-GIADA stable-code prefixes."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task41g_audited_v1'


def main():
    con = sqlite3.connect((ROOT / 'research_memory/data/research_memory.sqlite').as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    queries = []
    statements = {
        'early_gate_findings': """SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings
          WHERE [Codice stabile] LIKE 'finding-task1%' OR [Codice stabile] LIKE 'finding-task2b%'
          OR [Codice stabile] LIKE 'finding-task3%'""",
        'early_repair_evidence': """SELECT DISTINCT c.[Codice stabile] claim,c.Enunciato,c.Stato,
          e.Esito,e.Argomentazione,f.[Codice stabile] finding,f.Risultato
          FROM v_evidence_chains v JOIN claims c ON c._record_id=v.claim_id
          JOIN evidence e ON e._record_id=v.assessment_id
          JOIN findings f ON f._record_id=v.finding_id
          WHERE f.[Codice stabile] IN ('finding-task2b-h-physical-tau-identifiability-repaired-v1',
            'finding-task3d-voltage-support-conditional-shape-repair-v1',
            'finding-task3e-full-repair-sealed-generalization-v1',
            'finding-task3b-symmetric-rates-stabilize-shared-mh-v1')""",
        'early_repair_decisions': """SELECT DISTINCT f.[Codice stabile] finding,
          d.[Codice stabile] decision,d.Nome,d.Esito,d.Motivazione
          FROM v_semantic_edges v JOIN findings f ON f._record_id=v.target_id
          JOIN decisions d ON d._record_id=v.source_id
          WHERE v.source_table='decisions' AND v.target_table='findings'
          AND (f.[Codice stabile] LIKE 'finding-task2b%' OR f.[Codice stabile] LIKE 'finding-task3%')"""
    }
    for name, sql in statements.items():
        rows = [dict(r) for r in con.execute(sql)]
        queries.append(dict(name=name, sql=sql, rows=rows))
        print(name, len(rows))
    con.close()
    (OUT / 'early_memory_queries.json').write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'early_memory_queries.sql').write_text('\n\n'.join('-- ' + q['name'] + '\n' + q['sql'] + ';' for q in queries), encoding='utf-8')


if __name__ == '__main__':
    main()
