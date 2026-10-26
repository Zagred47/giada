"""Read-only source-linked GIADA history for the completed Task41h audit."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task41h_recovery_v1'


def main():
    con = sqlite3.connect((ROOT/'research_memory/data/research_memory.sqlite').as_uri()+'?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    queries = []
    statements = {
        'baseline': "SELECT * FROM _mirror_meta WHERE key='snapshot_hash'",
        'mechanisms_and_limits': """SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings
          WHERE length(Risultato)<20000 AND (
            [Codice stabile] LIKE '%giada-task41%' OR [Codice stabile] LIKE '%giada-task39j%'
            OR [Codice stabile] LIKE 'finding-task2b%' OR [Codice stabile] LIKE 'finding-task3%'
            OR [Codice stabile] LIKE '%giada-task17d%' OR [Codice stabile] LIKE '%giada-task40b%')
          ORDER BY [Codice stabile]""",
        'precision_frozen_head_and_speed_lessons': """SELECT [Codice stabile] code,Nome,Lezione,Limiti FROM lessons
          WHERE ([Codice stabile] LIKE '%giada%' OR [Codice stabile] LIKE '%task2b%') AND
          (lower(Lezione) LIKE '%precision%' OR lower(Lezione) LIKE '%frozen%'
           OR lower(Lezione) LIKE '%timing%' OR lower(Lezione) LIKE '%speed%'
           OR lower(Lezione) LIKE '%fusion%' OR lower(Lezione) LIKE '%round%')
          ORDER BY [Codice stabile]""",
        'claims_tested_by_41h': """SELECT [Codice stabile] code,Nome,Enunciato,Stato,
          [Condizioni di falsificazione],Limiti FROM claims
          WHERE [Codice stabile] LIKE '%giada-task41g%' ORDER BY [Codice stabile]""",
        'causal_evidence_chains': """SELECT DISTINCT c.[Codice stabile] claim,c.Enunciato,c.Stato,
          e.Esito,e.Argomentazione,e.[Limiti e spiegazioni alternative] alternatives,
          f.[Codice stabile] finding,f.Risultato
          FROM v_evidence_chains v JOIN claims c ON c._record_id=v.claim_id
          JOIN evidence e ON e._record_id=v.assessment_id JOIN findings f ON f._record_id=v.finding_id
          WHERE f.[Codice stabile] IN (
            'finding-task2b-h-physical-tau-identifiability-repaired-v1',
            'finding-task3d-voltage-support-conditional-shape-repair-v1',
            'finding-task3e-full-repair-sealed-generalization-v1')
          OR f.[Codice stabile] LIKE '%giada-task39j%'
          OR f.[Codice stabile] LIKE '%giada-task41g-completion%'""",
        '41h_registration_state': """SELECT [Codice stabile] code,Nome,Stato FROM experiments
          WHERE [Codice stabile] LIKE '%giada-task41h%'"""
    }
    for name, sql in statements.items():
        rows = [dict(r) for r in con.execute(sql)]
        queries.append(dict(name=name,sql=sql,rows=rows))
        print(json.dumps(dict(query=name,rows=len(rows))),flush=True)
    con.close()
    (OUT/'memory_queries.json').write_text(json.dumps(queries,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'memory_queries.sql').write_text('\n\n'.join('-- '+q['name']+'\n'+q['sql']+';' for q in queries),encoding='utf-8')


if __name__ == '__main__':
    main()
