"""Read-only, reproducible historical GIADA pattern exploration."""
import json, sqlite3
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'experiments/results/task41g_audited_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect((REPO/'research_memory/data/research_memory.sqlite').as_uri()+'?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    queries=[]
    def query(name,sql,params=()):
        rows=[dict(r) for r in con.execute(sql,params)]
        queries.append(dict(name=name,sql=sql,parameters=params,rows=rows))
        print(json.dumps(dict(query=name,count=len(rows)),ensure_ascii=False))
    for t in ('findings','claims','lessons','v_evidence_chains','v_semantic_edges'):
        query('schema_'+t,'PRAGMA table_info('+t+')')
    query('baseline',"SELECT * FROM _mirror_meta WHERE key='snapshot_hash'")
    query('historical_patterns',"""SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings
      WHERE [Codice stabile] LIKE '%giada%' AND
      (lower(Risultato) LIKE '%precision%' OR lower(Risultato) LIKE '%cancell%' OR
       lower(Risultato) LIKE '%interferen%' OR lower(Risultato) LIKE '%fusi%' OR
       lower(Risultato) LIKE '%identifi%' OR lower(Risultato) LIKE '%relax%')
      AND length(Risultato)<15000 ORDER BY [Codice stabile]""")
    query('focused_history',"""SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings
      WHERE [Codice stabile] LIKE '%giada%' AND
      ([Codice stabile] LIKE '%task39j%' OR [Codice stabile] LIKE '%task17d%' OR
       [Codice stabile] LIKE '%task2b%' OR [Codice stabile] LIKE '%task40b%' OR
       [Codice stabile] LIKE '%ie-p%') AND length(Risultato)<20000""")
    query('recent_lessons',"""SELECT [Codice stabile] code,Nome,Lezione,Limiti FROM lessons
      WHERE [Codice stabile] LIKE '%giada%' AND
      ([Codice stabile] LIKE '%task41%' OR [Codice stabile] LIKE '%task39j%' OR
       lower(Lezione) LIKE '%precision%' OR lower(Lezione) LIKE '%fusi%')""")
    query('evidence_chains',"""SELECT DISTINCT c.[Codice stabile] claim_code,c.Enunciato,c.Stato,
      e.Esito,e.Argomentazione,f.[Codice stabile] finding_code,f.Risultato
      FROM v_evidence_chains v JOIN claims c ON c._record_id=v.claim_id
      JOIN evidence e ON e._record_id=v.assessment_id
      JOIN findings f ON f._record_id=v.finding_id
      WHERE c.[Codice stabile] LIKE '%giada%' AND
      (f.[Codice stabile] LIKE '%task41f%' OR f.[Codice stabile] LIKE '%task39j%'
       OR f.[Codice stabile] LIKE '%task17d%' OR f.[Codice stabile] LIKE '%task40b%')""")
    query('findings_to_prior_decisions',"""SELECT DISTINCT f.[Codice stabile] finding_code,
      d.[Codice stabile] decision_code,d.Nome,d.Esito,d.Motivazione
      FROM v_semantic_edges v JOIN findings f ON f._record_id=v.target_id
      JOIN decisions d ON d._record_id=v.source_id
      WHERE v.source_table='decisions' AND v.target_table='findings'
      AND (f.[Codice stabile] LIKE '%task39j%' OR f.[Codice stabile] LIKE '%task40b%'
       OR f.[Codice stabile] LIKE '%task17d%' OR f.[Codice stabile] LIKE '%task41f%')""")
    query('small_gate_identifiability',"""SELECT [Codice stabile] code,Nome,Risultato,Limitazioni
      FROM findings WHERE [Codice stabile] LIKE '%giada%'
      AND (lower(Nome) LIKE '%identifiability%' OR lower(Nome) LIKE '%rate supervision%'
        OR lower(Nome) LIKE '%task 2b%') AND length(Risultato)<15000""")
    (OUT/'memory_queries.json').write_text(json.dumps(queries,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'memory_queries.sql').write_text('\n\n'.join('-- '+q['name']+'\n'+q['sql']+';' for q in queries),encoding='utf-8')

if __name__=='__main__':main()
