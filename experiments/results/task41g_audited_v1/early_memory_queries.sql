-- early_gate_findings
SELECT [Codice stabile] code,Nome,Risultato,Limitazioni FROM findings
          WHERE [Codice stabile] LIKE 'finding-task1%' OR [Codice stabile] LIKE 'finding-task2b%'
          OR [Codice stabile] LIKE 'finding-task3%';

-- early_repair_evidence
SELECT DISTINCT c.[Codice stabile] claim,c.Enunciato,c.Stato,
          e.Esito,e.Argomentazione,f.[Codice stabile] finding,f.Risultato
          FROM v_evidence_chains v JOIN claims c ON c._record_id=v.claim_id
          JOIN evidence e ON e._record_id=v.assessment_id
          JOIN findings f ON f._record_id=v.finding_id
          WHERE f.[Codice stabile] IN ('finding-task2b-h-physical-tau-identifiability-repaired-v1',
            'finding-task3d-voltage-support-conditional-shape-repair-v1',
            'finding-task3e-full-repair-sealed-generalization-v1',
            'finding-task3b-symmetric-rates-stabilize-shared-mh-v1');

-- early_repair_decisions
SELECT DISTINCT f.[Codice stabile] finding,
          d.[Codice stabile] decision,d.Nome,d.Esito,d.Motivazione
          FROM v_semantic_edges v JOIN findings f ON f._record_id=v.target_id
          JOIN decisions d ON d._record_id=v.source_id
          WHERE v.source_table='decisions' AND v.target_table='findings'
          AND (f.[Codice stabile] LIKE 'finding-task2b%' OR f.[Codice stabile] LIKE 'finding-task3%');