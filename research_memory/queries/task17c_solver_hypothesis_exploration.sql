-- Read-only exploration after the Task17c Kaggle replication.
-- Scope: GIADA. These queries inspect findings as well as hypothesis status,
-- because a status label can lag behind its linked evidence.

-- Q1: relevant precedents, including their scope and unresolved alternatives.
SELECT "Codice stabile", "Nome", "Risultato", "Incertezza", "Limitazioni"
FROM findings
WHERE "Codice stabile" LIKE '%task7b%'
   OR "Codice stabile" LIKE '%task7c%'
   OR "Codice stabile" LIKE '%task9%'
   OR "Codice stabile" LIKE '%task16%'
   OR "Codice stabile" LIKE '%task17%'
   OR "Codice stabile" LIKE '%task5-lut%'
   OR "Codice stabile" LIKE '%task2b%'
ORDER BY "Codice stabile";

-- Q2: follow actual relations rather than infer a decision from a title.
-- NULL means no direct linked decision in this query, not no historical decision.
SELECT f."Codice stabile" AS finding_code, f."Nome" AS finding,
       x."Nome" AS experiment, d."Nome" AS linked_decision,
       d."Motivazione", d."Condizioni di revisione"
FROM findings f
LEFT JOIN rel_fldYAbc64NIZGKzWO fx ON fx.source_id = f._record_id
LEFT JOIN experiments x ON x._record_id = fx.target_id
LEFT JOIN rel_fldpPpy9MTdwYYfvG df ON df.target_id = f._record_id
LEFT JOIN decisions d ON d._record_id = df.source_id
WHERE f."Codice stabile" LIKE '%task7b%'
   OR f."Codice stabile" LIKE '%task7c%'
   OR f."Codice stabile" LIKE '%task9%'
   OR f."Codice stabile" LIKE '%task16%'
   OR f."Codice stabile" LIKE '%task17%'
ORDER BY f."Codice stabile";

-- Q3: distinguish the claim label from the evidence and its source finding.
SELECT c."Codice stabile" AS claim_code, c."Nome" AS claim,
       c."Stato", e."Esito" AS evidence_outcome, e."Argomentazione",
       f."Codice stabile" AS finding_code, f."Risultato", f."Limitazioni"
FROM claims c
JOIN rel_fldzDr7IhMwkcELwG ec ON ec.target_id = c._record_id
JOIN evidence e ON e._record_id = ec.source_id
LEFT JOIN rel_fldzPthPEKWp0B8dE ef ON ef.source_id = e._record_id
LEFT JOIN findings f ON f._record_id = ef.target_id
WHERE c."Codice stabile" LIKE '%task17%'
   OR c."Codice stabile" LIKE '%task2b%'
   OR c."Codice stabile" LIKE '%task5%'
ORDER BY c."Codice stabile";

-- Q4: flag labels worth reviewing, without automatically changing them.
SELECT c."Codice stabile", c."Nome", c."Stato", e."Esito",
       e."Argomentazione", f."Codice stabile" AS supporting_finding
FROM claims c
JOIN rel_fldzDr7IhMwkcELwG ec ON ec.target_id = c._record_id
JOIN evidence e ON e._record_id = ec.source_id
LEFT JOIN rel_fldzPthPEKWp0B8dE ef ON ef.source_id = e._record_id
LEFT JOIN findings f ON f._record_id = ef.target_id
WHERE c."Stato" IN ('Aperta', 'Indeterminata') AND e."Esito" = 'Sostiene'
ORDER BY c."Codice stabile";
