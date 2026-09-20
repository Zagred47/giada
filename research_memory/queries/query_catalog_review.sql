-- BOZZA PER REVISIONE — nessuna vista o tabella viene creata.
-- 34 query SELECT indipendenti, schema del mirror corrente.
-- Parametri :nome da legare con sqlite3.execute(sql, params).
-- Il CLI `query` attuale NON supporta binding: usare sqlite3 come nella guida.

-- Q01 | Scheda strutturale del componente
-- Tipo: Esatta sui collegamenti registrati
-- Parametri: component_id
-- Limite interpretativo: Mostra una riga per oggetto. Non inventa stati o equazioni non registrati; fonti, evidenze e test si consultano con Q08/Q19/Q23.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
)
SELECT 'componente' AS sezione, i._record_id AS id, i."Nome" AS nome,
       i."Configurazione" AS dettaglio
FROM instances i WHERE i._record_id=:component_id
UNION ALL
SELECT 'porta',p._record_id,p."Nome",
       json_object('direzione',p."Direzione",'shape',p."Shape",
                   'unita',p."Unità",'tempo',p."Semantica temporale")
FROM ports p JOIN one l ON l.source_table='ports'
  AND l.source_id=p._record_id AND l.role='Componente'
WHERE l.target_id=:component_id
UNION ALL
SELECT 'equazione',e._record_id,e."Nome",e."Espressione LaTeX"
FROM equations e WHERE EXISTS (
  SELECT 1 FROM one b JOIN one be ON be.source_table='bindings'
    AND be.source_id=b.source_id AND be.role='Equazione'
  WHERE b.source_table='bindings' AND b.role='Componente'
    AND b.target_id=:component_id AND be.target_id=e._record_id
)
UNION ALL
SELECT 'parametrizzazione',p._record_id,p."Nome",
       json_object('vincoli',p."Vincoli",'trasformazioni',p."Trasformazioni")
FROM parameters p WHERE EXISTS (
  SELECT 1 FROM v_links l WHERE l.source_table='parameters'
    AND l.source_id=p._record_id AND l.role='Componenti associati'
    AND l.target_id=:component_id
)
ORDER BY sezione,id;

-- Q02 | Decomposizione ricorsiva del modello
-- Tipo: Esatta, con limite di profondità
-- Parametri: model_id, component_id, max_depth
-- Limite interpretativo: Conserva percorsi distinti se il grafo ha parti condivise. Evita cicli; un percorso registrato non prova una dipendenza causale.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
im AS (
  SELECT source_id AS component_id, target_id AS model_id
  FROM one WHERE source_table='instances' AND role='Modello'
),
parts AS (
  SELECT p.source_id AS composition_id, m.target_id AS model_id,
         p.target_id AS parent_id, c.target_id AS child_id
  FROM one p JOIN one c ON c.source_table='composition'
    AND c.source_id=p.source_id AND c.role='Parte'
  JOIN one m ON m.source_table='composition'
    AND m.source_id=p.source_id AND m.role='Modello'
  JOIN im pi ON pi.component_id=p.target_id AND pi.model_id=m.target_id
  JOIN im ci ON ci.component_id=c.target_id AND ci.model_id=m.target_id
  WHERE p.source_table='composition' AND p.role='Contenitore'
)
,
tree(node_id,depth,path) AS (
  SELECT :component_id,0,'|'||:component_id||'|'
  WHERE EXISTS (SELECT 1 FROM im
    WHERE component_id=:component_id AND model_id=:model_id)
  UNION ALL
  SELECT p.child_id,t.depth+1,t.path||p.child_id||'|'
  FROM tree t JOIN parts p ON p.parent_id=t.node_id AND p.model_id=:model_id
  WHERE t.depth<:max_depth AND instr(t.path,'|'||p.child_id||'|')=0
)
SELECT t.depth,i._record_id,i."Nome",i."Identificatore locale",t.path,
       CASE WHEN t.depth=:max_depth AND EXISTS (
         SELECT 1 FROM parts p WHERE p.parent_id=t.node_id AND p.model_id=:model_id
       ) THEN 1 ELSE 0 END AS possibile_troncamento
FROM tree t JOIN instances i ON i._record_id=t.node_id
ORDER BY t.path;

-- Q03 | Ingressi, uscite e dipendenze dirette
-- Tipo: Esatta per il grafo dichiarato
-- Parametri: model_id, component_id
-- Limite interpretativo: Legge accoppiamenti dichiarati, non inferisce dipendenze dalla sola correlazione tra variabili.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
im AS (
  SELECT source_id AS component_id, target_id AS model_id
  FROM one WHERE source_table='instances' AND role='Modello'
),
pc AS (
  SELECT source_id AS port_id, target_id AS component_id
  FROM one WHERE source_table='ports' AND role='Componente'
),
wires AS (
  SELECT a.source_id AS connection_id, m.target_id AS model_id,
         a.target_id AS from_port, b.target_id AS to_port,
         pa.component_id AS from_component, pb.component_id AS to_component
  FROM one a JOIN one b ON b.source_table='connections'
    AND b.source_id=a.source_id AND b.role='Porta destinazione'
  JOIN one m ON m.source_table='connections'
    AND m.source_id=a.source_id AND m.role='Modello'
  JOIN pc pa ON pa.port_id=a.target_id JOIN pc pb ON pb.port_id=b.target_id
  JOIN im ia ON ia.component_id=pa.component_id AND ia.model_id=m.target_id
  JOIN im ib ON ib.component_id=pb.component_id AND ib.model_id=m.target_id
  WHERE a.source_table='connections' AND a.role='Porta origine'
)
SELECT w.connection_id,c."Tipo",c."Ritardo",c."Unità ritardo",
       a."Nome" AS origine,b."Nome" AS destinazione,
       pa."Nome" AS porta_origine,pb."Nome" AS porta_destinazione,
       pa."Informazione causalmente disponibile" AS contratto_input
FROM wires w JOIN connections c ON c._record_id=w.connection_id
JOIN instances a ON a._record_id=w.from_component
JOIN instances b ON b._record_id=w.to_component
JOIN ports pa ON pa._record_id=w.from_port
JOIN ports pb ON pb._record_id=w.to_port
WHERE w.model_id=:model_id
  AND :component_id IN (w.from_component,w.to_component)
ORDER BY origine,destinazione;

-- Q04 | Percorsi di informazione tra due componenti
-- Tipo: Topologica, non prova causale
-- Parametri: model_id, from_component, to_component, max_depth
-- Limite interpretativo: Enumera percorsi semplici: non espande nel tempo le ricorrenze. Per il calcio, scegliere componenti collegati alle grandezze appropriate tramite porte/grandezze.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
im AS (
  SELECT source_id AS component_id, target_id AS model_id
  FROM one WHERE source_table='instances' AND role='Modello'
),
pc AS (
  SELECT source_id AS port_id, target_id AS component_id
  FROM one WHERE source_table='ports' AND role='Componente'
),
wires AS (
  SELECT a.source_id AS connection_id, m.target_id AS model_id,
         a.target_id AS from_port, b.target_id AS to_port,
         pa.component_id AS from_component, pb.component_id AS to_component
  FROM one a JOIN one b ON b.source_table='connections'
    AND b.source_id=a.source_id AND b.role='Porta destinazione'
  JOIN one m ON m.source_table='connections'
    AND m.source_id=a.source_id AND m.role='Modello'
  JOIN pc pa ON pa.port_id=a.target_id JOIN pc pb ON pb.port_id=b.target_id
  JOIN im ia ON ia.component_id=pa.component_id AND ia.model_id=m.target_id
  JOIN im ib ON ib.component_id=pb.component_id AND ib.model_id=m.target_id
  WHERE a.source_table='connections' AND a.role='Porta origine'
)
,
paths(node_id,depth,path,connections_path) AS (
  SELECT :from_component,0,'|'||:from_component||'|',''
  UNION ALL
  SELECT w.to_component,p.depth+1,p.path||w.to_component||'|',
         p.connections_path||'/'||w.connection_id
  FROM paths p JOIN wires w ON w.from_component=p.node_id AND w.model_id=:model_id
  WHERE p.depth<:max_depth AND instr(p.path,'|'||w.to_component||'|')=0
)
SELECT depth,path,connections_path FROM paths
WHERE node_id=:to_component AND depth>0 ORDER BY depth,path;

-- Q05 | Differenze tra componenti di due versioni
-- Tipo: Esatta solo con identificatori locali stabili
-- Parametri: model_a, model_b
-- Limite interpretativo: Confronto testuale di metadati, non equivalenza matematica. ID mancanti/duplicati richiedono Q34. Q06 aggiunge l'inventario di equazioni e connessioni.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
im AS (
  SELECT source_id AS component_id, target_id AS model_id
  FROM one WHERE source_table='instances' AND role='Modello'
)
,
components AS (
  SELECT im.model_id,i."Identificatore locale" AS local_id,
         MIN(i._record_id) AS id,MIN(i."Nome") AS nome,
         MIN(i."Configurazione") AS configurazione,MIN(i."Ruolo") AS ruolo,
         MIN(ty.target_id) AS tipo_id
  FROM im JOIN instances i ON i._record_id=im.component_id
  LEFT JOIN one ty ON ty.source_table='instances' AND ty.source_id=i._record_id
    AND ty.role='Tipo componente'
  WHERE im.model_id IN (:model_a,:model_b)
    AND NULLIF(i."Identificatore locale",'') IS NOT NULL
  GROUP BY im.model_id,i."Identificatore locale" HAVING COUNT(*)=1
), keys AS (
  SELECT local_id FROM components WHERE model_id=:model_a
  UNION SELECT local_id FROM components WHERE model_id=:model_b
)
SELECT k.local_id,a.id AS id_a,b.id AS id_b,
       CASE WHEN a.id IS NULL THEN 'aggiunto'
            WHEN b.id IS NULL THEN 'rimosso'
            ELSE 'metadati modificati' END AS differenza,
       a.configurazione AS configurazione_a,b.configurazione AS configurazione_b,
       a.ruolo AS ruolo_a,b.ruolo AS ruolo_b,a.tipo_id AS tipo_a,b.tipo_id AS tipo_b
FROM keys k
LEFT JOIN components a ON a.local_id=k.local_id AND a.model_id=:model_a
LEFT JOIN components b ON b.local_id=k.local_id AND b.model_id=:model_b
WHERE a.id IS NULL OR b.id IS NULL OR a.nome IS NOT b.nome
   OR a.configurazione IS NOT b.configurazione OR a.ruolo IS NOT b.ruolo
   OR a.tipo_id IS NOT b.tipo_id
ORDER BY k.local_id;

-- Q06 | Inventario matematico e connessioni per versione
-- Tipo: Inventario per confronto umano
-- Parametri: model_a, model_b
-- Limite interpretativo: Non dichiara uguali equazioni solo perché i nomi coincidono. La corrispondenza cross-version delle porte va registrata o verificata.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
im AS (
  SELECT source_id AS component_id, target_id AS model_id
  FROM one WHERE source_table='instances' AND role='Modello'
),
pc AS (
  SELECT source_id AS port_id, target_id AS component_id
  FROM one WHERE source_table='ports' AND role='Componente'
),
wires AS (
  SELECT a.source_id AS connection_id, m.target_id AS model_id,
         a.target_id AS from_port, b.target_id AS to_port,
         pa.component_id AS from_component, pb.component_id AS to_component
  FROM one a JOIN one b ON b.source_table='connections'
    AND b.source_id=a.source_id AND b.role='Porta destinazione'
  JOIN one m ON m.source_table='connections'
    AND m.source_id=a.source_id AND m.role='Modello'
  JOIN pc pa ON pa.port_id=a.target_id JOIN pc pb ON pb.port_id=b.target_id
  JOIN im ia ON ia.component_id=pa.component_id AND ia.model_id=m.target_id
  JOIN im ib ON ib.component_id=pb.component_id AND ib.model_id=m.target_id
  WHERE a.source_table='connections' AND a.role='Porta origine'
)
SELECT im.model_id,'equazione' AS tipo,i."Identificatore locale" AS componente,
       e._record_id AS oggetto_id,e."Nome",e."Espressione LaTeX" AS contenuto
FROM im JOIN instances i ON i._record_id=im.component_id
JOIN one b ON b.source_table='bindings' AND b.role='Componente'
  AND b.target_id=i._record_id
JOIN one be ON be.source_table='bindings' AND be.source_id=b.source_id
  AND be.role='Equazione'
JOIN equations e ON e._record_id=be.target_id
WHERE im.model_id IN (:model_a,:model_b)
UNION ALL
SELECT w.model_id,'connessione',i."Identificatore locale",c._record_id,c."Nome",
       json_object('origine',w.from_port,'destinazione',w.to_port,
                   'tipo',c."Tipo",'trasformazione',c."Trasformazione")
FROM wires w JOIN connections c ON c._record_id=w.connection_id
JOIN instances i ON i._record_id=w.from_component
WHERE w.model_id IN (:model_a,:model_b)
ORDER BY model_id,tipo,componente,oggetto_id;

-- Q07 | Corrispondenze teacher-surrogate
-- Tipo: Esatta sui mapping dichiarati
-- Parametri: teacher_model
-- Limite interpretativo: Una corrispondenza registrata non equivale a una sostituzione validata.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
)
SELECT s._record_id,s."Nome",s."Mappatura stati e output",
       s."Informazione mantenuta",s."Informazione omessa",s."Condizioni di equivalenza",
       (SELECT json_group_array(target_id) FROM v_links
        WHERE source_table='substitutions' AND source_id=s._record_id
          AND role='Componenti originali') AS originali,
       (SELECT json_group_array(target_id) FROM v_links
        WHERE source_table='substitutions' AND source_id=s._record_id
          AND role='Componenti sostitutivi') AS sostitutivi
FROM substitutions s JOIN one m ON m.source_table='substitutions'
  AND m.source_id=s._record_id AND m.role='Modello originale'
WHERE m.target_id=:teacher_model ORDER BY s._record_id;

-- Q08 | Catena di evidenze e provenienza di una ipotesi
-- Tipo: Esatta sui collegamenti
-- Parametri: claim_id
-- Limite interpretativo: Esito esplicito, non voto dedotto dal nome della tabella. Il campo storico 'Risultati a sostegno' contiene anche risultati usati per contraddire: conta Esito.
SELECT e._record_id AS valutazione_id,e."Esito",e."Argomentazione",
       e."Limiti e spiegazioni alternative",e."Data valutazione",
       (SELECT json_group_array(target_id) FROM v_links
        WHERE source_table='evidence' AND source_id=e._record_id
          AND role='Risultati a sostegno') AS risultati,
       (SELECT json_group_array(target_id) FROM v_links
        WHERE source_table='evidence' AND source_id=e._record_id
          AND role='Passaggi fonte') AS passaggi,
       (SELECT json_group_array(target_id) FROM v_links
        WHERE source_table='evidence' AND source_id=e._record_id
          AND role='Domini di validità') AS domini
FROM evidence e WHERE EXISTS (
  SELECT 1 FROM v_links l WHERE l.source_table='evidence'
    AND l.source_id=e._record_id AND l.role='Affermazione valutata'
    AND l.target_id=:claim_id
) ORDER BY e."Data valutazione",e._record_id;

-- Q09 | Ipotesi con evidenze contrastanti
-- Tipo: Shortlist da disambiguare per dominio
-- Parametri: nessuno
-- Limite interpretativo: Conteggia valutazioni, non repliche indipendenti. Contrasti tra domini diversi non sono automaticamente contraddizioni.
SELECT c._record_id,c."Nome",
       COUNT(DISTINCT CASE WHEN e."Esito"='Sostiene' THEN e._record_id END) AS sostiene,
       COUNT(DISTINCT CASE WHEN e."Esito"='Contraddice' THEN e._record_id END) AS contraddice
FROM claims c JOIN v_links l ON l.source_table='evidence'
  AND l.role='Affermazione valutata' AND l.target_id=c._record_id
JOIN evidence e ON e._record_id=l.source_id
GROUP BY c._record_id,c."Nome"
HAVING COUNT(DISTINCT CASE WHEN e."Esito"='Sostiene' THEN e._record_id END)>0
   AND COUNT(DISTINCT CASE WHEN e."Esito"='Contraddice' THEN e._record_id END)>0;

-- Q10 | Scelte che dipendono da assunzioni senza supporto registrato
-- Tipo: Lacuna documentale
-- Parametri: model_id
-- Limite interpretativo: Non dice 'falsa' né 'basata soltanto su assunzioni': altre giustificazioni possono esistere. Dice esattamente che manca supporto registrato per quella premessa.
SELECT DISTINCT ch._record_id AS scelta_id,ch."Nome" AS scelta,
       a._record_id AS assunzione_id,a."Premessa"
FROM choices ch JOIN v_links m ON m.source_table='choices'
  AND m.source_id=ch._record_id AND m.role='Modello' AND m.target_id=:model_id
JOIN v_links ca ON ca.source_table='choices' AND ca.source_id=ch._record_id
  AND ca.role='Assunzioni'
JOIN assumptions a ON a._record_id=ca.target_id
WHERE NOT EXISTS (
  SELECT 1 FROM evidence e JOIN v_links el
    ON el.source_table='evidence' AND el.source_id=e._record_id
    AND el.role='Assunzione valutata' AND el.target_id=a._record_id
  WHERE e."Esito"='Sostiene'
);

-- Q11 | Affermazioni senza valutazioni delle evidenze
-- Tipo: Lacuna documentale
-- Parametri: nessuno
-- Limite interpretativo: Assenza nel DB, non assenza nella letteratura.
SELECT c._record_id,c."Nome",c."Tipo",c."Stato",c."Enunciato"
FROM claims c WHERE NOT EXISTS (
  SELECT 1 FROM v_links l WHERE l.source_table='evidence'
    AND l.role='Affermazione valutata' AND l.target_id=c._record_id
) ORDER BY c."Tipo",c."Nome";

-- Q12 | Domini dichiarati contro domini delle evidenze
-- Tipo: Audit di copertura, non prova di trasferimento
-- Parametri: claim_id
-- Limite interpretativo: Identità dei record dominio, non inclusione semantica automatica tra condizioni testuali.
SELECT d._record_id,d."Nome",d."Condizioni",d."Esclusioni",d."Stato",
       EXISTS(SELECT 1 FROM v_links l WHERE l.source_table='claims'
         AND l.source_id=:claim_id AND l.role='Domini di validità'
         AND l.target_id=d._record_id) AS dichiarato_per_ipotesi,
       (SELECT COUNT(DISTINCT e._record_id) FROM evidence e
        JOIN v_links ec ON ec.source_table='evidence' AND ec.source_id=e._record_id
          AND ec.role='Affermazione valutata' AND ec.target_id=:claim_id
        JOIN v_links ed ON ed.source_table='evidence' AND ed.source_id=e._record_id
          AND ed.role='Domini di validità' AND ed.target_id=d._record_id
        WHERE e."Esito"='Sostiene') AS valutazioni_favorevoli
FROM validity d WHERE EXISTS (
  SELECT 1 FROM v_links l WHERE l.source_table='claims'
    AND l.source_id=:claim_id AND l.role='Domini di validità' AND l.target_id=d._record_id
) OR EXISTS (
  SELECT 1 FROM v_links ec JOIN v_links ed ON ed.source_table='evidence'
    AND ed.source_id=ec.source_id AND ed.role='Domini di validità'
  WHERE ec.source_table='evidence' AND ec.role='Affermazione valutata'
    AND ec.target_id=:claim_id AND ed.target_id=d._record_id
) ORDER BY d._record_id;

-- Q13 | Origine documentata di una ipotesi
-- Tipo: Provenienza, non certificazione
-- Parametri: claim_id
-- Limite interpretativo: Una fonte di tipo Paper non è automaticamente una prova. La classificazione come derivazione o proposta si consulta anche in claims.Tipo ed evidence.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
)
SELECT 'passaggio citato' AS relazione,s._record_id AS oggetto_id,
       s."Nome",s."Testo originale" AS contenuto,f."Tipo" AS tipo_fonte,f."URL"
FROM segments s JOIN v_links l ON l.source_table='claims'
  AND l.source_id=:claim_id AND l.role='Passaggi fonte' AND l.target_id=s._record_id
LEFT JOIN one sf ON sf.source_table='segments' AND sf.source_id=s._record_id
  AND sf.role='Fonte'
LEFT JOIN sources f ON f._record_id=sf.target_id
UNION ALL
SELECT 'analogia generatrice',a._record_id,a."Nome",a."Corrispondenza",NULL,NULL
FROM analogies a WHERE EXISTS (
  SELECT 1 FROM v_links l WHERE l.source_table='analogies'
    AND l.source_id=a._record_id AND l.role='Ipotesi generate' AND l.target_id=:claim_id
);

-- Q14 | Decisioni collegate a ipotesi con controevidenza
-- Tipo: Candidati alla revisione
-- Parametri: nessuno
-- Limite interpretativo: Non chiama 'successiva' una valutazione senza date ISO affidabili. Non distingue automaticamente dipendenza necessaria da semplice associazione: è una shortlist.
WITH RECURSIVE
decision_claims AS (
  SELECT DISTINCT d.source_id AS decision_id,ec.target_id AS claim_id
  FROM v_links d JOIN v_links ec ON ec.source_table='evidence'
    AND ec.source_id=d.target_id AND ec.role='Affermazione valutata'
  WHERE d.source_table='decisions' AND d.role='Valutazioni delle evidenze'
  UNION
  SELECT DISTINCT d.source_id,ci.target_id
  FROM v_links d JOIN v_links ci ON ci.source_table='candidates'
    AND ci.source_id=d.target_id AND ci.role='Ipotesi'
  WHERE d.source_table='decisions' AND d.role='Candidati'
)
SELECT DISTINCT d._record_id,d."Nome",d."Data",d."Esito",
       c._record_id AS claim_id,c."Nome" AS ipotesi,
       e._record_id AS controevidenza,e."Data valutazione"
FROM decision_claims dc JOIN decisions d ON d._record_id=dc.decision_id
JOIN claims c ON c._record_id=dc.claim_id
JOIN v_links ec ON ec.source_table='evidence' AND ec.role='Affermazione valutata'
  AND ec.target_id=c._record_id
JOIN evidence e ON e._record_id=ec.source_id
WHERE e."Esito"='Contraddice'
ORDER BY d._record_id,c._record_id;

-- Q15 | Confronti allineati per specifica e blocco
-- Tipo: Candidati con filtri conservativi
-- Parametri: evaluation_id
-- Limite interpretativo: Stessa specifica, protocollo, blocco, seed, budget/unità, numerosità, sottogruppo e insieme di split. Richiede checkpoint con budget esplicito. Nessun controllo fallito ≠ tutti i controlli necessari superati. Hardware, stop rules e deviazioni vanno ancora esaminati.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
ra AS (
  SELECT source_id AS run_id, target_id AS arm_id
  FROM one WHERE source_table='runs' AND role='Braccio'
),
ap AS (
  SELECT source_id AS arm_id, target_id AS protocol_id
  FROM one WHERE source_table='arms' AND role='Protocollo'
),
measurement AS (
  SELECT o._record_id AS observation_id, r._record_id AS run_id,
         ra.arm_id, ap.protocol_id, ev.target_id AS evaluation_id,
         cp.target_id AS checkpoint_id, ck."Budget consumato" AS budget,
         ck."Unità budget" AS budget_unit,
         o."Strato o sottogruppo" AS stratum,
         o."Valore" AS value, o."Numerosità" AS sample_count,
         rb.target_id AS block_id, r."Seed" AS seed
  FROM observations o
  JOIN one ev ON ev.source_table='observations'
    AND ev.source_id=o._record_id AND ev.role='Specifica di valutazione'
  JOIN one rr ON rr.source_table='observations'
    AND rr.source_id=o._record_id AND rr.role='Run'
  JOIN runs r ON r._record_id=rr.target_id
  JOIN ra ON ra.run_id=r._record_id JOIN ap ON ap.arm_id=ra.arm_id
  JOIN one evp ON evp.source_table='evaluations'
    AND evp.source_id=ev.target_id AND evp.role='Protocollo'
    AND evp.target_id=ap.protocol_id
  LEFT JOIN one cp ON cp.source_table='observations'
    AND cp.source_id=o._record_id AND cp.role='Checkpoint'
  LEFT JOIN checkpoints ck ON ck._record_id=cp.target_id
  LEFT JOIN one cpr ON cpr.source_table='checkpoints'
    AND cpr.source_id=ck._record_id AND cpr.role='Run'
  LEFT JOIN one rb ON rb.source_table='runs'
    AND rb.source_id=r._record_id AND rb.role='Blocco'
  WHERE o."Valore" IS NOT NULL
    AND (cp.target_id IS NULL OR cpr.target_id=r._record_id)
),
pairs AS (
  SELECT a.observation_id AS observation_a, b.observation_id AS observation_b,
         a.run_id AS run_a, b.run_id AS run_b,
         a.arm_id AS arm_a, b.arm_id AS arm_b,
         a.protocol_id, a.evaluation_id, a.block_id,
         a.budget, a.budget_unit, a.stratum, a.sample_count,
         a.value AS value_a, b.value AS value_b
  FROM measurement a JOIN measurement b
    ON a.run_id < b.run_id AND a.arm_id <> b.arm_id
    AND a.protocol_id=b.protocol_id AND a.evaluation_id=b.evaluation_id
    AND a.block_id=b.block_id AND a.block_id IS NOT NULL
    AND a.seed=b.seed AND a.seed IS NOT NULL
    AND a.budget=b.budget AND a.budget_unit=b.budget_unit
    AND a.budget IS NOT NULL AND NULLIF(a.budget_unit,'') IS NOT NULL
    AND a.stratum IS b.stratum AND a.sample_count=b.sample_count
  JOIN runs ar ON ar._record_id=a.run_id AND ar."Stato"='Completata'
  JOIN runs br ON br._record_id=b.run_id AND br."Stato"='Completata'
  WHERE EXISTS (
    SELECT 1 FROM v_links x WHERE x.source_table='runs'
      AND x.source_id=a.run_id AND x.role='Partizioni effettive'
  )
  AND NOT EXISTS (
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=a.run_id AND role='Partizioni effettive'
    EXCEPT
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=b.run_id AND role='Partizioni effettive'
  )
  AND NOT EXISTS (
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=b.run_id AND role='Partizioni effettive'
    EXCEPT
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=a.run_id AND role='Partizioni effettive'
  )
  AND NOT EXISTS (
    SELECT 1 FROM quality q JOIN v_links ql
      ON ql.source_table='quality' AND ql.source_id=q._record_id
      AND ql.role='Run'
    WHERE ql.target_id IN (a.run_id,b.run_id) AND q."Esito"='Fallito'
  )
)
SELECT p.*,b."Hash stream",b."Regola di appaiamento",
       a."Validità tecnica" AS validita_a,r."Validità tecnica" AS validita_b
FROM pairs p JOIN blocks b ON b._record_id=p.block_id
JOIN runs a ON a._record_id=p.run_a JOIN runs r ON r._record_id=p.run_b
WHERE p.evaluation_id=:evaluation_id
ORDER BY p.block_id,p.budget,p.run_a,p.run_b;

-- Q16 | Differenza appaiata cambiando un fattore
-- Tipo: Contrasto osservato, non causalità automatica
-- Parametri: evaluation_id, factor_id, contrast_id
-- Limite interpretativo: Non media run, checkpoint o osservazioni duplicate. Il segno è B−A: per una loss negativa è meglio, per F1 positiva è meglio. Variabili non registrate come fattori non risultano automaticamente controllate.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
ra AS (
  SELECT source_id AS run_id, target_id AS arm_id
  FROM one WHERE source_table='runs' AND role='Braccio'
),
ap AS (
  SELECT source_id AS arm_id, target_id AS protocol_id
  FROM one WHERE source_table='arms' AND role='Protocollo'
),
measurement AS (
  SELECT o._record_id AS observation_id, r._record_id AS run_id,
         ra.arm_id, ap.protocol_id, ev.target_id AS evaluation_id,
         cp.target_id AS checkpoint_id, ck."Budget consumato" AS budget,
         ck."Unità budget" AS budget_unit,
         o."Strato o sottogruppo" AS stratum,
         o."Valore" AS value, o."Numerosità" AS sample_count,
         rb.target_id AS block_id, r."Seed" AS seed
  FROM observations o
  JOIN one ev ON ev.source_table='observations'
    AND ev.source_id=o._record_id AND ev.role='Specifica di valutazione'
  JOIN one rr ON rr.source_table='observations'
    AND rr.source_id=o._record_id AND rr.role='Run'
  JOIN runs r ON r._record_id=rr.target_id
  JOIN ra ON ra.run_id=r._record_id JOIN ap ON ap.arm_id=ra.arm_id
  JOIN one evp ON evp.source_table='evaluations'
    AND evp.source_id=ev.target_id AND evp.role='Protocollo'
    AND evp.target_id=ap.protocol_id
  LEFT JOIN one cp ON cp.source_table='observations'
    AND cp.source_id=o._record_id AND cp.role='Checkpoint'
  LEFT JOIN checkpoints ck ON ck._record_id=cp.target_id
  LEFT JOIN one cpr ON cpr.source_table='checkpoints'
    AND cpr.source_id=ck._record_id AND cpr.role='Run'
  LEFT JOIN one rb ON rb.source_table='runs'
    AND rb.source_id=r._record_id AND rb.role='Blocco'
  WHERE o."Valore" IS NOT NULL
    AND (cp.target_id IS NULL OR cpr.target_id=r._record_id)
),
pairs AS (
  SELECT a.observation_id AS observation_a, b.observation_id AS observation_b,
         a.run_id AS run_a, b.run_id AS run_b,
         a.arm_id AS arm_a, b.arm_id AS arm_b,
         a.protocol_id, a.evaluation_id, a.block_id,
         a.budget, a.budget_unit, a.stratum, a.sample_count,
         a.value AS value_a, b.value AS value_b
  FROM measurement a JOIN measurement b
    ON a.run_id < b.run_id AND a.arm_id <> b.arm_id
    AND a.protocol_id=b.protocol_id AND a.evaluation_id=b.evaluation_id
    AND a.block_id=b.block_id AND a.block_id IS NOT NULL
    AND a.seed=b.seed AND a.seed IS NOT NULL
    AND a.budget=b.budget AND a.budget_unit=b.budget_unit
    AND a.budget IS NOT NULL AND NULLIF(a.budget_unit,'') IS NOT NULL
    AND a.stratum IS b.stratum AND a.sample_count=b.sample_count
  JOIN runs ar ON ar._record_id=a.run_id AND ar."Stato"='Completata'
  JOIN runs br ON br._record_id=b.run_id AND br."Stato"='Completata'
  WHERE EXISTS (
    SELECT 1 FROM v_links x WHERE x.source_table='runs'
      AND x.source_id=a.run_id AND x.role='Partizioni effettive'
  )
  AND NOT EXISTS (
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=a.run_id AND role='Partizioni effettive'
    EXCEPT
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=b.run_id AND role='Partizioni effettive'
  )
  AND NOT EXISTS (
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=b.run_id AND role='Partizioni effettive'
    EXCEPT
    SELECT target_id FROM v_links WHERE source_table='runs'
      AND source_id=a.run_id AND role='Partizioni effettive'
  )
  AND NOT EXISTS (
    SELECT 1 FROM quality q JOIN v_links ql
      ON ql.source_table='quality' AND ql.source_id=q._record_id
      AND ql.role='Run'
    WHERE ql.target_id IN (a.run_id,b.run_id) AND q."Esito"='Fallito'
  )
),
af AS (
  SELECT a.target_id AS arm_id, f.target_id AS factor_id,
         l.target_id AS level_id
  FROM one a JOIN one l ON l.source_table='assignments'
    AND l.source_id=a.source_id AND l.role='Livello'
  JOIN one f ON f.source_table='factorlevels'
    AND f.source_id=l.target_id AND f.role='Fattore'
  WHERE a.source_table='assignments' AND a.role='Braccio'
)
SELECT p.block_id,p.budget,p.stratum,p.run_a,p.run_b,
       la."Valore" AS livello_a,lb."Valore" AS livello_b,
       p.value_a,p.value_b,p.value_b-p.value_a AS differenza_b_meno_a
FROM pairs p JOIN af a ON a.arm_id=p.arm_a AND a.factor_id=:factor_id
JOIN af b ON b.arm_id=p.arm_b AND b.factor_id=:factor_id
JOIN factorlevels la ON la._record_id=a.level_id
JOIN factorlevels lb ON lb._record_id=b.level_id
WHERE p.evaluation_id=:evaluation_id AND a.level_id<>b.level_id
  AND (SELECT COUNT(*) FROM af x WHERE x.arm_id=a.arm_id AND x.factor_id=:factor_id)=1
  AND (SELECT COUNT(*) FROM af x WHERE x.arm_id=b.arm_id AND x.factor_id=:factor_id)=1
  AND EXISTS (SELECT 1 FROM v_links l WHERE l.source_table='contrasts'
    AND l.source_id=:contrast_id AND l.role='Bracci' AND l.target_id=p.arm_a)
  AND EXISTS (SELECT 1 FROM v_links l WHERE l.source_table='contrasts'
    AND l.source_id=:contrast_id AND l.role='Bracci' AND l.target_id=p.arm_b)
  AND NOT EXISTS (
    SELECT factor_id,level_id FROM af WHERE arm_id=p.arm_a AND factor_id<>:factor_id
    EXCEPT SELECT factor_id,level_id FROM af WHERE arm_id=p.arm_b AND factor_id<>:factor_id
  )
  AND NOT EXISTS (
    SELECT factor_id,level_id FROM af WHERE arm_id=p.arm_b AND factor_id<>:factor_id
    EXCEPT SELECT factor_id,level_id FROM af WHERE arm_id=p.arm_a AND factor_id<>:factor_id
  )
ORDER BY p.block_id,p.budget;

-- Q17 | Interazione fattoriale 2×2
-- Tipo: Contrasto descrittivo per blocco completo
-- Parametri: evaluation_id, factor_a, factor_b, level_a0, level_a1, level_b0, level_b1
-- Limite interpretativo: Calcola y11−y10−y01+y00; non p-value né significatività. Parametri livelli/fattori devono essere distinti e coerenti. Blocchi incompleti o celle duplicate non producono stime. Non è ancora una prova del meccanismo.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
ra AS (
  SELECT source_id AS run_id, target_id AS arm_id
  FROM one WHERE source_table='runs' AND role='Braccio'
),
ap AS (
  SELECT source_id AS arm_id, target_id AS protocol_id
  FROM one WHERE source_table='arms' AND role='Protocollo'
),
measurement AS (
  SELECT o._record_id AS observation_id, r._record_id AS run_id,
         ra.arm_id, ap.protocol_id, ev.target_id AS evaluation_id,
         cp.target_id AS checkpoint_id, ck."Budget consumato" AS budget,
         ck."Unità budget" AS budget_unit,
         o."Strato o sottogruppo" AS stratum,
         o."Valore" AS value, o."Numerosità" AS sample_count,
         rb.target_id AS block_id, r."Seed" AS seed
  FROM observations o
  JOIN one ev ON ev.source_table='observations'
    AND ev.source_id=o._record_id AND ev.role='Specifica di valutazione'
  JOIN one rr ON rr.source_table='observations'
    AND rr.source_id=o._record_id AND rr.role='Run'
  JOIN runs r ON r._record_id=rr.target_id
  JOIN ra ON ra.run_id=r._record_id JOIN ap ON ap.arm_id=ra.arm_id
  JOIN one evp ON evp.source_table='evaluations'
    AND evp.source_id=ev.target_id AND evp.role='Protocollo'
    AND evp.target_id=ap.protocol_id
  LEFT JOIN one cp ON cp.source_table='observations'
    AND cp.source_id=o._record_id AND cp.role='Checkpoint'
  LEFT JOIN checkpoints ck ON ck._record_id=cp.target_id
  LEFT JOIN one cpr ON cpr.source_table='checkpoints'
    AND cpr.source_id=ck._record_id AND cpr.role='Run'
  LEFT JOIN one rb ON rb.source_table='runs'
    AND rb.source_id=r._record_id AND rb.role='Blocco'
  WHERE o."Valore" IS NOT NULL
    AND (cp.target_id IS NULL OR cpr.target_id=r._record_id)
),
af AS (
  SELECT a.target_id AS arm_id, f.target_id AS factor_id,
         l.target_id AS level_id
  FROM one a JOIN one l ON l.source_table='assignments'
    AND l.source_id=a.source_id AND l.role='Livello'
  JOIN one f ON f.source_table='factorlevels'
    AND f.source_id=l.target_id AND f.role='Fattore'
  WHERE a.source_table='assignments' AND a.role='Braccio'
)
,
cells AS (
  SELECT m.*,a.level_id AS a_level,b.level_id AS b_level,
    (SELECT json_group_array(json_array(factor_id,level_id)) FROM (
      SELECT factor_id,level_id FROM af x WHERE x.arm_id=m.arm_id
        AND factor_id NOT IN (:factor_a,:factor_b) ORDER BY factor_id,level_id
    )) AS other_factors,
    (SELECT json_group_array(target_id) FROM (
      SELECT target_id FROM v_links WHERE source_table='runs'
        AND source_id=m.run_id AND role='Partizioni effettive' ORDER BY target_id
    )) AS partitions_used
  FROM measurement m
  JOIN af a ON a.arm_id=m.arm_id AND a.factor_id=:factor_a
  JOIN af b ON b.arm_id=m.arm_id AND b.factor_id=:factor_b
  JOIN runs r ON r._record_id=m.run_id AND r."Stato"='Completata'
  WHERE m.evaluation_id=:evaluation_id AND m.block_id IS NOT NULL
    AND m.budget IS NOT NULL AND NULLIF(m.budget_unit,'') IS NOT NULL
    AND a.level_id IN (:level_a0,:level_a1)
    AND b.level_id IN (:level_b0,:level_b1)
    AND NOT EXISTS (
      SELECT 1 FROM quality q JOIN v_links ql ON ql.source_table='quality'
        AND ql.source_id=q._record_id AND ql.role='Run'
      WHERE ql.target_id=m.run_id AND q."Esito"='Fallito'
    )
), unique_cells AS (
  SELECT block_id,protocol_id,evaluation_id,budget,budget_unit,stratum,
         seed,sample_count,other_factors,partitions_used,a_level,b_level,
         MIN(value) AS value
  FROM cells WHERE partitions_used<>'[]' AND sample_count IS NOT NULL
  GROUP BY block_id,protocol_id,evaluation_id,budget,budget_unit,stratum,
           seed,sample_count,other_factors,partitions_used,a_level,b_level
  HAVING COUNT(*)=1
)
SELECT block_id,budget,budget_unit,stratum,seed,other_factors,partitions_used,
       SUM(CASE WHEN a_level=:level_a1 AND b_level=:level_b1 THEN value
                WHEN a_level=:level_a0 AND b_level=:level_b0 THEN value
                ELSE -value END) AS interazione
FROM unique_cells
GROUP BY block_id,protocol_id,evaluation_id,budget,budget_unit,stratum,
         seed,sample_count,other_factors,partitions_used
HAVING COUNT(*)=4;

-- Q18 | Curve di apprendimento a budget crescente
-- Tipo: Esatta per osservazioni registrate
-- Parametri: evaluation_id
-- Limite interpretativo: Una curva per run e sottogruppo, nessuna media tra checkpoint. I budget in unità diverse rimangono separati.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
ra AS (
  SELECT source_id AS run_id, target_id AS arm_id
  FROM one WHERE source_table='runs' AND role='Braccio'
),
ap AS (
  SELECT source_id AS arm_id, target_id AS protocol_id
  FROM one WHERE source_table='arms' AND role='Protocollo'
),
measurement AS (
  SELECT o._record_id AS observation_id, r._record_id AS run_id,
         ra.arm_id, ap.protocol_id, ev.target_id AS evaluation_id,
         cp.target_id AS checkpoint_id, ck."Budget consumato" AS budget,
         ck."Unità budget" AS budget_unit,
         o."Strato o sottogruppo" AS stratum,
         o."Valore" AS value, o."Numerosità" AS sample_count,
         rb.target_id AS block_id, r."Seed" AS seed
  FROM observations o
  JOIN one ev ON ev.source_table='observations'
    AND ev.source_id=o._record_id AND ev.role='Specifica di valutazione'
  JOIN one rr ON rr.source_table='observations'
    AND rr.source_id=o._record_id AND rr.role='Run'
  JOIN runs r ON r._record_id=rr.target_id
  JOIN ra ON ra.run_id=r._record_id JOIN ap ON ap.arm_id=ra.arm_id
  JOIN one evp ON evp.source_table='evaluations'
    AND evp.source_id=ev.target_id AND evp.role='Protocollo'
    AND evp.target_id=ap.protocol_id
  LEFT JOIN one cp ON cp.source_table='observations'
    AND cp.source_id=o._record_id AND cp.role='Checkpoint'
  LEFT JOIN checkpoints ck ON ck._record_id=cp.target_id
  LEFT JOIN one cpr ON cpr.source_table='checkpoints'
    AND cpr.source_id=ck._record_id AND cpr.role='Run'
  LEFT JOIN one rb ON rb.source_table='runs'
    AND rb.source_id=r._record_id AND rb.role='Blocco'
  WHERE o."Valore" IS NOT NULL
    AND (cp.target_id IS NULL OR cpr.target_id=r._record_id)
)
SELECT run_id,arm_id,block_id,seed,checkpoint_id,budget,budget_unit,stratum,
       value,sample_count
FROM measurement WHERE evaluation_id=:evaluation_id
ORDER BY run_id,budget,stratum,observation_id;

-- Q19 | Test isolati e di ricomposizione di una sostituzione
-- Tipo: Copertura esplicita e risultati da leggere
-- Parametri: substitution_id
-- Limite interpretativo: Lo schema attuale non ha un booleano strutturato 'isolato superato/ricomposto fallito'. Questa query mostra modalità e risultati; l'esito richiede giudizio esplicito o una futura regola formalizzata.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
ep AS (
  SELECT source_id AS protocol_id, target_id AS experiment_id
  FROM one WHERE source_table='protocols' AND role='Esperimento'
),
subtests AS (
  SELECT DISTINCT s.source_id AS substitution_id,
         e.target_id AS experiment_id, ep.protocol_id
  FROM v_links s JOIN v_links e
    ON e.source_table='substitutions' AND e.source_id=s.source_id
    AND e.role='Esperimenti di verifica'
  JOIN ep ON ep.experiment_id=e.target_id
  WHERE s.source_table='substitutions'
)
SELECT DISTINCT p._record_id AS protocollo_id,p."Modalità",
       e._record_id AS esperimento_id,e."Nome",e."Stato",
       (SELECT json_group_array(f._record_id) FROM findings f
        WHERE EXISTS (SELECT 1 FROM v_links l WHERE l.source_table='findings'
          AND l.source_id=f._record_id AND l.role='Esperimenti'
          AND l.target_id=e._record_id)) AS risultati
FROM subtests s JOIN protocols p ON p._record_id=s.protocol_id
JOIN experiments e ON e._record_id=s.experiment_id
WHERE s.substitution_id=:substitution_id ORDER BY p."Modalità",p._record_id;

-- Q20 | Predizioni preregistrate e misure corrispondenti
-- Tipo: Affiancamento, non verdetto automatico
-- Parametri: experiment_id
-- Limite interpretativo: La soglia è testo: SQL non interpreta frasi come 'nettamente migliore'. Le date testuali non certificano una preregistrazione; il relativo artefatto/hash va verificato.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
ep AS (
  SELECT source_id AS protocol_id, target_id AS experiment_id
  FROM one WHERE source_table='protocols' AND role='Esperimento'
)
SELECT DISTINCT p._record_id AS predizione_id,p."Risultato atteso",
       p."Soglia o intervallo",p."Registrata il",pr."Registrato il",
       o._record_id AS osservazione_id,o."Valore",o."Numerosità",
       o."Strato o sottogruppo",ev."Orizzonte e finestre"
FROM ep JOIN protocols pr ON pr._record_id=ep.protocol_id
JOIN v_links pp ON pp.source_table='protocols' AND pp.source_id=pr._record_id
  AND pp.role='Predizioni'
JOIN predictions p ON p._record_id=pp.target_id AND p."Origine"='Preregistrata'
LEFT JOIN one pe ON pe.source_table='predictions' AND pe.source_id=p._record_id
  AND pe.role='Specifica di valutazione'
LEFT JOIN evaluations ev ON ev._record_id=pe.target_id
LEFT JOIN one oe ON oe.source_table='observations'
  AND oe.role='Specifica di valutazione' AND oe.target_id=ev._record_id
LEFT JOIN observations o ON o._record_id=oe.source_id
WHERE ep.experiment_id=:experiment_id ORDER BY p._record_id,o._record_id;

-- Q21 | Riclassificazione degli esperimenti secondo oggetti diversi
-- Tipo: Lente basata sulle ipotesi collegate
-- Parametri: axis
-- Limite interpretativo: Una lente parametrica per componente, equazione, parametrizzazione, ricetta, meccanismo, modello o dominio. È pertinenza dichiarata, non prova che l'oggetto sia stato manipolato.
SELECT DISTINCT e._record_id AS experiment_id,e."Nome" AS esperimento,
       c._record_id AS claim_id,c."Nome" AS ipotesi,
       ob.target_table AS asse,ob.target_id AS oggetto_id,r.name AS oggetto
FROM experiments e JOIN v_links ec ON ec.source_table='experiments'
  AND ec.source_id=e._record_id AND ec.role='Ipotesi'
JOIN claims c ON c._record_id=ec.target_id
JOIN v_links ob ON ob.source_table='claims' AND ob.source_id=c._record_id
  AND ob.target_table=:axis
JOIN v_records r ON r.table_key=ob.target_table AND r.record_id=ob.target_id
WHERE :axis IN ('instances','equations','parameters','recipes','mechanisms','models','validity')
ORDER BY oggetto,esperimento;

-- Q22 | Ricerca trasversale per informazione, regime o failure mode
-- Tipo: Ricerca lessicale, non classificazione automatica
-- Parametri: needle
-- Limite interpretativo: Esempi di needle: 'calcio', 'storia temporale', 'plateau', 'ottimizzazione'. lower SQLite standard non offre case-folding Unicode completo. Una menzione non equivale a un'etichetta validata.
SELECT 'protocols' AS table_key,_record_id,"Nome",
       "Modalità" AS tipo,"Procedura" AS testo
FROM protocols WHERE instr(lower(COALESCE("Procedura",'')),lower(:needle))>0
UNION ALL
SELECT 'models',_record_id,"Nome","Ruolo","Contratto di input e output"
FROM models WHERE instr(lower(COALESCE("Contratto di input e output",'')),lower(:needle))>0
UNION ALL
SELECT 'observations',_record_id,"Nome",'sottogruppo',"Strato o sottogruppo"
FROM observations WHERE instr(lower(COALESCE("Strato o sottogruppo",'')),lower(:needle))>0
UNION ALL
SELECT 'anomalies',_record_id,"Nome","Stato","Possibili cause"
FROM anomalies WHERE instr(lower(COALESCE("Possibili cause",'')),lower(:needle))>0;

-- Q23 | Componenti senza esperimenti associati nel catalogo
-- Tipo: Lacuna di collegamento
-- Parametri: model_id
-- Limite interpretativo: Non significa mai testati. Include associazioni dirette tramite sostituzione e indirette tramite ipotesi; manca ancora un contratto universale esperimento→componente testato.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
im AS (
  SELECT source_id AS component_id, target_id AS model_id
  FROM one WHERE source_table='instances' AND role='Modello'
),
scope AS (
  SELECT component_id FROM im WHERE model_id=:model_id
),
component_experiments AS (
  -- Associazione esplicita via sostituzione; non prova di esecuzione.
  SELECT DISTINCT c.target_id AS component_id, e.target_id AS experiment_id
  FROM v_links c JOIN v_links e ON e.source_table='substitutions'
    AND e.source_id=c.source_id AND e.role='Esperimenti di verifica'
  WHERE c.source_table='substitutions'
    AND c.role IN ('Componenti originali','Componenti sostitutivi')
  UNION
  -- Associazione indiretta tramite ipotesi sul componente.
  SELECT DISTINCT c.target_id, e.source_id
  FROM v_links c JOIN v_links e ON e.source_table='experiments'
    AND e.role='Ipotesi' AND e.target_id=c.source_id
  WHERE c.source_table='claims' AND c.role='Componenti oggetto'
)
SELECT i._record_id,i."Nome",i."Identificatore locale"
FROM scope s JOIN instances i ON i._record_id=s.component_id
WHERE NOT EXISTS (SELECT 1 FROM component_experiments ce
  WHERE ce.component_id=i._record_id)
ORDER BY i."Nome";

-- Q24 | Componenti con test isolato associato ma nessuna ricomposizione
-- Tipo: Shortlist, senza inferire successo
-- Parametri: model_id
-- Limite interpretativo: La presenza del protocollo non prova che sia stato eseguito né superato. Una query 'validato isolatamente ma non ricomposto' richiede anche una regola esplicita di successo.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
im AS (
  SELECT source_id AS component_id, target_id AS model_id
  FROM one WHERE source_table='instances' AND role='Modello'
),
scope AS (
  SELECT component_id FROM im WHERE model_id=:model_id
),
component_experiments AS (
  -- Associazione esplicita via sostituzione; non prova di esecuzione.
  SELECT DISTINCT c.target_id AS component_id, e.target_id AS experiment_id
  FROM v_links c JOIN v_links e ON e.source_table='substitutions'
    AND e.source_id=c.source_id AND e.role='Esperimenti di verifica'
  WHERE c.source_table='substitutions'
    AND c.role IN ('Componenti originali','Componenti sostitutivi')
  UNION
  -- Associazione indiretta tramite ipotesi sul componente.
  SELECT DISTINCT c.target_id, e.source_id
  FROM v_links c JOIN v_links e ON e.source_table='experiments'
    AND e.role='Ipotesi' AND e.target_id=c.source_id
  WHERE c.source_table='claims' AND c.role='Componenti oggetto'
),
ep AS (
  SELECT source_id AS protocol_id, target_id AS experiment_id
  FROM one WHERE source_table='protocols' AND role='Esperimento'
)
SELECT i._record_id,i."Nome"
FROM scope s JOIN instances i ON i._record_id=s.component_id
WHERE EXISTS (
  SELECT 1 FROM component_experiments ce JOIN ep ON ep.experiment_id=ce.experiment_id
  JOIN protocols p ON p._record_id=ep.protocol_id
  WHERE ce.component_id=i._record_id AND p."Modalità"='Componente isolato'
) AND NOT EXISTS (
  SELECT 1 FROM component_experiments ce JOIN ep ON ep.experiment_id=ce.experiment_id
  JOIN protocols p ON p._record_id=ep.protocol_id
  WHERE ce.component_id=i._record_id AND p."Modalità" IN ('Ricomposizione','Sistema completo')
);

-- Q25 | Protocolli senza controlli registrati
-- Tipo: Audit del disegno
-- Parametri: experiment_id
-- Limite interpretativo: Non stabilisce quale controllo sia necessario e non considera un oracle equivalente a un controllo negativo. Per requisiti specifici serve registrare controllo richiesto→ipotesi.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
),
ep AS (
  SELECT source_id AS protocol_id, target_id AS experiment_id
  FROM one WHERE source_table='protocols' AND role='Esperimento'
)
SELECT p._record_id,p."Nome",p."Modalità"
FROM protocols p JOIN ep ON ep.protocol_id=p._record_id
WHERE ep.experiment_id=:experiment_id AND NOT EXISTS (
  SELECT 1 FROM v_links l JOIN arms a ON a._record_id=l.source_id
  WHERE l.source_table='arms' AND l.role='Protocollo' AND l.target_id=p._record_id
    AND a."Ruolo" IN ('Baseline','Controllo negativo','Oracle diagnostico','Ablazione')
);

-- Q26 | Copertura dei sottogruppi attesi nelle misure
-- Tipo: Esatta solo per etichette concordate
-- Parametri: evaluation_id, expected_strata_json
-- Limite interpretativo: Esempio JSON ['riposo','subthreshold','spike','plateau'] con virgolette JSON doppie. Sono etichette da concordare, non dati già presenti. Conta misure, non episodi indipendenti.
SELECT j.value AS sottogruppo_atteso,
       COUNT(DISTINCT o._record_id) AS misure_registrate,
       COUNT(DISTINCT CASE WHEN o."Valore" IS NOT NULL THEN o._record_id END) AS misure_numeriche
FROM json_each(:expected_strata_json) j
LEFT JOIN observations o ON o."Strato o sottogruppo"=j.value
 AND EXISTS (
   SELECT 1 FROM v_links l WHERE l.source_table='observations'
     AND l.source_id=o._record_id AND l.role='Specifica di valutazione'
     AND l.target_id=:evaluation_id
 )
GROUP BY j.value ORDER BY j.value;

-- Q27 | Risultati positivi privi di una conferma collegata
-- Tipo: Lacuna di conferma registrata
-- Parametri: nessuno
-- Limite interpretativo: Un record di conferma presente non prova che la conferma sia completata o riuscita. Anche questo esito va formalizzato se vogliamo il filtro 'confermato con successo'.
SELECT f._record_id,f."Nome",f."Risultato",f."Limitazioni"
FROM findings f WHERE f."Esito"='Positivo' AND NOT EXISTS (
  SELECT 1 FROM v_links l JOIN confirmations c ON c._record_id=l.source_id
  WHERE l.source_table='confirmations' AND l.role='Risultati da confermare'
    AND l.target_id=f._record_id AND c."Tipo"='Conferma indipendente'
) ORDER BY f._record_id;

-- Q28 | Ipotesi alternative ancora aperte e relative predizioni
-- Tipo: Shortlist di spiegazioni non risolte
-- Parametri: nessuno
-- Limite interpretativo: Non dimostra compatibilità con tutti i dati. Servono predizioni strutturate e valutazioni di ciascun esito per eliminare automaticamente alternative.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
)
SELECT r._record_id AS relazione_id,a._record_id AS ipotesi_a,a."Enunciato" AS enunciato_a,
       b._record_id AS ipotesi_b,b."Enunciato" AS enunciato_b,
       (SELECT json_group_array(source_id) FROM v_links WHERE source_table='predictions'
        AND role='Ipotesi' AND target_id=a._record_id) AS predizioni_a,
       (SELECT json_group_array(source_id) FROM v_links WHERE source_table='predictions'
        AND role='Ipotesi' AND target_id=b._record_id) AS predizioni_b
FROM claimrels r
JOIN one x ON x.source_table='claimrels' AND x.source_id=r._record_id AND x.role='Enunciato origine'
JOIN one y ON y.source_table='claimrels' AND y.source_id=r._record_id AND y.role='Enunciato destinazione'
JOIN claims a ON a._record_id=x.target_id JOIN claims b ON b._record_id=y.target_id
WHERE r."Tipo"='Alternativa'
  AND a."Stato" IN ('Aperta','Indeterminata')
  AND b."Stato" IN ('Aperta','Indeterminata');

-- Q29 | Esperimenti candidati allo stesso notebook
-- Tipo: Compatibilità preliminare
-- Parametri: nessuno
-- Limite interpretativo: Stesso programma/playground non garantisce stessi dati, assenza di leakage o indipendenza. Confermare partizioni, generatori, budget e dipendenze; non autorizza esecuzione parallela automatica.
SELECT DISTINCT a._record_id AS esperimento_a,a."Nome" AS nome_a,
       b._record_id AS esperimento_b,b."Nome" AS nome_b,
       p.target_id AS playground_condiviso
FROM experiments a JOIN v_links p ON p.source_table='experiments'
  AND p.source_id=a._record_id AND p.role='Playground'
JOIN v_links q ON q.source_table='experiments' AND q.role='Playground'
  AND q.target_id=p.target_id AND q.source_id>a._record_id
JOIN experiments b ON b._record_id=q.source_id
WHERE a."Stato" IN ('Progettato','Preregistrato')
  AND b."Stato" IN ('Progettato','Preregistrato')
  AND EXISTS (
    SELECT 1 FROM v_links x JOIN v_links y ON y.source_table='experiments'
      AND y.source_id=b._record_id AND y.role='Programma' AND y.target_id=x.target_id
    WHERE x.source_table='experiments' AND x.source_id=a._record_id AND x.role='Programma'
  );

-- Q30 | Capacità con molte dipendenze a valle
-- Tipo: Priorità strutturale candidata
-- Parametri: nessuno
-- Limite interpretativo: Conta dipendenze dichiarate dirette. Tenere distinto ipotizzato/verificato esaminando i record capdeps; non è utilità scientifica o necessità dimostrata.
WITH RECURSIVE
one AS (
  SELECT source_table, source_id, role, target_table,
         MIN(target_id) AS target_id
  FROM v_links
  GROUP BY source_table, source_id, role, target_table
  HAVING COUNT(*) = 1
)
SELECT c._record_id,c."Nome",
       COUNT(DISTINCT d.target_id) AS capacita_dipendenti
FROM capabilities c JOIN one p ON p.source_table='capdeps'
  AND p.role='Prerequisito' AND p.target_id=c._record_id
JOIN one d ON d.source_table='capdeps' AND d.source_id=p.source_id
  AND d.role='Capacità dipendente'
JOIN capdeps rel ON rel._record_id=p.source_id
WHERE rel."Tipo" IN ('Prerequisito ipotizzato','Prerequisito verificato')
GROUP BY c._record_id,c."Nome" ORDER BY capacita_dipendenti DESC;

-- Q31 | Esperimenti che interrogano più ipotesi aperte
-- Tipo: Shortlist informativa non probabilistica
-- Parametri: nessuno
-- Limite interpretativo: Numero di ipotesi collegate ≠ numero di ipotesi distinguibili e ≠ guadagno informativo atteso. Servono contrasti discriminanti e, per una misura probabilistica, prior e likelihood.
SELECT e._record_id,e."Nome",e."Obiettivo informativo",
       COUNT(DISTINCT c._record_id) AS ipotesi_aperte_collegate
FROM experiments e JOIN v_links l ON l.source_table='experiments'
  AND l.source_id=e._record_id AND l.role='Ipotesi'
JOIN claims c ON c._record_id=l.target_id
WHERE e."Stato" IN ('Progettato','Preregistrato')
  AND c."Stato" IN ('Aperta','Indeterminata')
GROUP BY e._record_id,e."Nome",e."Obiettivo informativo"
ORDER BY ipotesi_aperte_collegate DESC,e._record_id;

-- Q32 | Azioni pronte senza prerequisiti operativi incompleti
-- Tipo: Filtro di workflow
-- Parametri: nessuno
-- Limite interpretativo: Non interpreta i criteri testuali di avvio. 'Pronta' deve essere assegnato consapevolmente.
SELECT a._record_id,a."Nome",a."Criteri di avvio",a."Risultato atteso"
FROM actions a WHERE a."Stato"='Pronta' AND NOT EXISTS (
  SELECT 1 FROM v_links l JOIN actions p ON p._record_id=l.target_id
  WHERE l.source_table='actions' AND l.source_id=a._record_id
    AND l.role='Dipende da' AND COALESCE(p."Stato",'')<>'Conclusa'
) ORDER BY a._record_id;

-- Q33 | Sincronizzazione e identità dei record
-- Tipo: Controllo tecnico
-- Parametri: nessuno
-- Limite interpretativo: Zero righe indica nessuna intenzione pendente registrata, non garantisce che Airtable non sia stato modificato manualmente dopo l'ultimo import.
SELECT operation_id,table_key,stable_code,record_id,status
FROM _outbox WHERE status='pending' ORDER BY table_key,stable_code;

-- Q34 | Ambiguità che rendono inaffidabili le query
-- Tipo: Controllo di qualità strutturale
-- Parametri: nessuno
-- Limite interpretativo: Le query che usano la CTE one escludono relazioni ambigue anziché scegliere arbitrariamente un estremo. Questa query le rende visibili; obbligatorietà e cicli richiedono ulteriori controlli.
SELECT 'cardinalita_singola' AS problema,l.source_table AS tabella,
       l.source_id AS record_id,r.field_name AS campo,COUNT(*) AS n
FROM v_links l JOIN _mirror_relations r ON r.field_id=l.field_id
WHERE r.intended_single=1
GROUP BY l.source_table,l.source_id,r.field_name HAVING COUNT(*)>1
UNION ALL
SELECT 'codice_duplicato',table_key,NULL,stable_code,COUNT(*)
FROM v_records WHERE NULLIF(stable_code,'') IS NOT NULL
GROUP BY table_key,stable_code HAVING COUNT(*)>1
UNION ALL
SELECT 'id_locale_duplicato','instances',m.target_id,i."Identificatore locale",COUNT(*)
FROM instances i JOIN v_links m ON m.source_table='instances'
  AND m.source_id=i._record_id AND m.role='Modello'
WHERE NULLIF(i."Identificatore locale",'') IS NOT NULL
GROUP BY m.target_id,i."Identificatore locale" HAVING COUNT(*)>1
UNION ALL
SELECT 'id_locale_mancante','instances',_record_id,'Identificatore locale',1
FROM instances WHERE NULLIF("Identificatore locale",'') IS NULL;
