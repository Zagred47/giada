-- Read-only lens: contradicted GIADA hypotheses and the exact recorded contrast.
-- A missing evidence/finding link is surfaced as NULL, not interpreted as proof.
SELECT
  c."Codice stabile" AS hypothesis_code,
  c."Nome" AS hypothesis,
  c."Enunciato" AS registered_prediction,
  c."Condizioni di falsificazione" AS falsification_rule,
  c."Limiti" AS claim_scope,
  q."Nome" AS research_question,
  x."Nome" AS experiment,
  e."Esito" AS evidence_outcome,
  e."Argomentazione" AS observed_contrast,
  e."Limiti e spiegazioni alternative" AS alternative_explanations,
  f."Nome" AS linked_finding,
  f."Risultato" AS finding_result
FROM claims AS c
LEFT JOIN v_links AS cq ON cq.source_table='claims'
  AND cq.source_id=c._record_id AND cq.role='Domande'
LEFT JOIN questions AS q ON q._record_id=cq.target_id
LEFT JOIN v_links AS xc ON xc.source_table='experiments'
  AND xc.target_id=c._record_id AND xc.role='Ipotesi'
LEFT JOIN experiments AS x ON x._record_id=xc.source_id
LEFT JOIN v_links AS ec ON ec.source_table='evidence'
  AND ec.target_id=c._record_id AND ec.role='Affermazione valutata'
LEFT JOIN evidence AS e ON e._record_id=ec.source_id
LEFT JOIN v_links AS ef ON ef.source_table='evidence'
  AND ef.source_id=e._record_id AND ef.role='Risultati a sostegno'
LEFT JOIN findings AS f ON f._record_id=ef.target_id
WHERE c."Stato"='Contraddetta nel dominio'
ORDER BY c."Nome", x."Nome", e."Data valutazione";
