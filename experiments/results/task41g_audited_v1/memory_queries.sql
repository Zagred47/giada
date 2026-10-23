-- schema_findings
PRAGMA table_info(findings);

-- schema_claims
PRAGMA table_info(claims);

-- schema_lessons
PRAGMA table_info(lessons);

-- schema_v_evidence_chains
PRAGMA table_info(v_evidence_chains);

-- schema_v_semantic_edges
PRAGMA table_info(v_semantic_edges);

-- baseline
SELECT * FROM _mirror_meta WHERE key='snapshot_hash';

-- historical_patterns
SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings
      WHERE [Codice stabile] LIKE '%giada%' AND
      (lower(Risultato) LIKE '%precision%' OR lower(Risultato) LIKE '%cancell%' OR
       lower(Risultato) LIKE '%interferen%' OR lower(Risultato) LIKE '%fusi%' OR
       lower(Risultato) LIKE '%identifi%' OR lower(Risultato) LIKE '%relax%')
      AND length(Risultato)<15000 ORDER BY [Codice stabile];

-- focused_history
SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings
      WHERE [Codice stabile] LIKE '%giada%' AND
      ([Codice stabile] LIKE '%task39j%' OR [Codice stabile] LIKE '%task17d%' OR
       [Codice stabile] LIKE '%task2b%' OR [Codice stabile] LIKE '%task40b%' OR
       [Codice stabile] LIKE '%ie-p%') AND length(Risultato)<20000;

-- recent_lessons
SELECT [Codice stabile] code,Nome,Lezione,Limiti FROM lessons
      WHERE [Codice stabile] LIKE '%giada%' AND
      ([Codice stabile] LIKE '%task41%' OR [Codice stabile] LIKE '%task39j%' OR
       lower(Lezione) LIKE '%precision%' OR lower(Lezione) LIKE '%fusi%');

-- evidence_chains
SELECT DISTINCT c.[Codice stabile] claim_code,c.Enunciato,c.Stato,
      e.Esito,e.Argomentazione,f.[Codice stabile] finding_code,f.Risultato
      FROM v_evidence_chains v JOIN claims c ON c._record_id=v.claim_id
      JOIN evidence e ON e._record_id=v.assessment_id
      JOIN findings f ON f._record_id=v.finding_id
      WHERE c.[Codice stabile] LIKE '%giada%' AND
      (f.[Codice stabile] LIKE '%task41f%' OR f.[Codice stabile] LIKE '%task39j%'
       OR f.[Codice stabile] LIKE '%task17d%' OR f.[Codice stabile] LIKE '%task40b%');

-- findings_to_prior_decisions
SELECT DISTINCT f.[Codice stabile] finding_code,
      d.[Codice stabile] decision_code,d.Nome,d.Esito,d.Motivazione
      FROM v_semantic_edges v JOIN findings f ON f._record_id=v.target_id
      JOIN decisions d ON d._record_id=v.source_id
      WHERE v.source_table='decisions' AND v.target_table='findings'
      AND (f.[Codice stabile] LIKE '%task39j%' OR f.[Codice stabile] LIKE '%task40b%'
       OR f.[Codice stabile] LIKE '%task17d%' OR f.[Codice stabile] LIKE '%task41f%');

-- small_gate_identifiability
SELECT [Codice stabile] code,Nome,Risultato,Limitazioni
      FROM findings WHERE [Codice stabile] LIKE '%giada%'
      AND (lower(Nome) LIKE '%identifiability%' OR lower(Nome) LIKE '%rate supervision%'
        OR lower(Nome) LIKE '%task 2b%') AND length(Risultato)<15000;