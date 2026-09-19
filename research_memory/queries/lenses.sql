-- Eseguire una query per volta con `query`; il comando è sempre read-only.

-- Catalogo delle tabelle semantiche.
SELECT table_key, name, airtable_id FROM _mirror_tables ORDER BY name;

-- Distribuzione delle metriche per famiglia: regressione, classificazione, ecc.
SELECT "Famiglia", COUNT(*) AS metriche
FROM metrics GROUP BY "Famiglia" ORDER BY metriche DESC;

-- Catena di evidenza, non un ranking automatico della verità.
SELECT * FROM v_evidence_chains ORDER BY claim, assessment;

-- Riclassificare gli esperimenti attraverso le ipotesi collegate.
SELECT target_name AS ipotesi, source_name AS esperimento
FROM v_semantic_edges
WHERE source_table='experiments' AND target_table='claims'
ORDER BY ipotesi, esperimento;

-- Sostituzioni teacher/surrogate e componenti coinvolti.
SELECT source_name AS sostituzione, role, target_name AS componente
FROM v_semantic_edges
WHERE source_table='substitutions' AND target_table='instances'
ORDER BY sostituzione, role;

-- Decomposizione transitiva: 12 livelli massimi, guardia contro cicli.
WITH RECURSIVE parts(parent_id,child_id) AS (
  SELECT p.target_id,c.target_id FROM v_links p JOIN v_links c
    ON p.source_table=c.source_table AND p.source_id=c.source_id
  WHERE p.source_table='composition' AND p.role='Contenitore' AND c.role='Parte'
), tree(root_id,node_id,depth,path) AS (
  SELECT parent_id,child_id,1,'|'||parent_id||'|'||child_id||'|' FROM parts
  UNION ALL
  SELECT t.root_id,p.child_id,t.depth+1,t.path||p.child_id||'|'
  FROM tree t JOIN parts p ON p.parent_id=t.node_id
  WHERE t.depth<12 AND instr(t.path,'|'||p.child_id||'|')=0
)
SELECT root_id,node_id,depth FROM tree ORDER BY root_id,depth,node_id;

-- Operazioni non ancora completate su entrambi i sistemi.
SELECT operation_id,table_key,stable_code,status FROM _outbox WHERE status='pending';

-- Non aggregare RMSE con definizioni diverse: raggruppare per specifica esatta.
SELECT e.target_id AS specifica_id, AVG(o."Valore") AS media, COUNT(*) AS n
FROM observations o JOIN v_links e ON e.source_table='observations'
  AND e.source_id=o._record_id AND e.role='Specifica di valutazione'
GROUP BY e.target_id;
