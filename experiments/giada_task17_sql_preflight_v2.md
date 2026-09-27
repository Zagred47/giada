# GIADA — audit SQL post-backfill prima della Task 17

Data: 2026-09-27. Questa nota sostituisce per la pianificazione della Task 17
la fotografia precedente `giada_task17_sql_preflight_v1.md`, scattata quando il
mirror conteneva solo 273 record. Le interrogazioni qui riportate sono state
eseguite in **sola lettura** sul mirror SQLite locale dopo il recupero delle
fonti GIADA Ca_HVA: 3.797 record, 78 tabelle, 241 relazioni, snapshot SHA-256
`9c779b9cb63f092b2fed17781dd7ab9b717c1b49898736b196e9d54a542befbf`.
Airtable non è stato consultato o modificato.

Le 34 query del catalogo `research_memory/queries/query_catalog_review.json`
superano la verifica sintattica e i controlli sintetici con
`python -m research_memory.queries.validate_review_catalog`. Sono state
rieseguite sul DB reale le 19 istanze mirate in
`python -m research_memory.queries.run_task17_preflight`; altre query globali
e le sette lenti Q21 sono state conteggiate con parametri espliciti. Non
interpretiamo i test sintetici come copertura dei dati reali.

## Pattern scientifici che orientano la Task 17

1. **La Task 16 è una conferma limitata ma solida.** Q20 collega la predizione
   registrata (massimo RMSE gate per gruppo ≤0,005) all'osservazione 0,000318555
   nel test embedded shadow; il risultato originale ha 16/16 gruppi, 79
   transizioni e 37 traiettorie. Q25 non segnala assenza di controlli per quel
   protocollo dopo il backfill. Tuttavia V midpoint proveniva dal teacher dopo
   l'avanzamento e la LUT non modificava la membrana. Questo non è Gate C.
2. **L'informazione temporale è un fattore causale centrale.** Le Task 7, 9–11
   mostrano che un singolo V iniziale non sostituisce in generale il path
   intra-ms; gli oracle di path non sono input ammissibili all'inizio del passo.
   La Task 17 deve dichiarare tempi di disponibilità degli ingressi e un
   controllo con formula esatta sottoposto alla stessa interfaccia causale.
3. **Misurare delta, non solo RMSE assoluto.** Le Task 14–15 mostrano che un
   rollout con RMSE assoluto accettabile può perdere gli effetti piccoli delle
   perturbazioni Ca_HVA. Servono confronto appaiato nativo/formula/LUT e
   metriche sia assolute sia differenziali, per compartimento e regime.
4. **La semantica del solver va verificata nuovamente.** La Task 13 ha
   identificato una convenzione pre-step per la corrente campionata nel
   playground fixed-step a un compartimento. Non va trasferita per assunzione
   al teacher multicompartimentale con CVode.

## Limiti emersi dalle query: non confondere lacuna del DB con fallimento

- Q09 trova zero ipotesi con evidenze opposte, mentre Q11 trova **9 affermazioni
  senza una valutazione di evidenza collegata**. Alcune sono marcate
  «Supportata nel dominio» o «Contraddetta nel dominio»: il loro stato testuale
  non sostituisce un collegamento esplicito alla prova. Zero in Q09 non
  dimostra consenso.
- Q14 restituisce **4** decisioni storiche collegate a controevidenza. Sono
  candidati da leggere nel loro ordine temporale, non decisioni attuali da
  ribaltare automaticamente.
- Q20 ora trova l'accoppiamento predizione/misura per Task 16, ma per Task 15
  restituisce ancora la predizione composita con osservazione nulla. Le misure
  Task 15 sono presenti per protocollo e ZIP; manca un mapping atomico
  predizione→specifica, che non va inventato.
- Q15–Q18 non sono ancora lenti affidabili sui confronti storici: le 1.154
  osservazioni recuperate sono legate a run, specifica e ZIP, ma **zero** hanno
  relazioni a braccio o checkpoint. Mancano quindi le chiavi necessarie per
  appaiamenti, interazioni 2×2 e curve a budget tramite quelle query. I report
  originali conservano tali confronti, ma non vanno ricostruiti da join vuoti.
- Q21 produce 3 associazioni esperimento→ipotesi→componente, 1 verso equazione
  e 1 verso modello; zero per parametrizzazioni, ricette, meccanismi o domini
  formali. Q28–Q31 sono vuote, salvo le altre lenti globali, per scarsità di
  relazioni semantiche/fattoriali strutturate: non è prova che manchino ipotesi
  alternative o possibilità di esperimenti multifattoriali.
- Q27 segnala **21** finding positivi senza una conferma *collegata*. Task 16
  compare correttamente: il suo test indipendente conferma la selezione Task
  15c entro lo scope shadow, ma non esiste ancora una conferma successiva della
  sostituzione causale Task 17. La query resta un audit dei link, non una stima
  della riproducibilità.
- Q32 propone ancora Task 5 e Task 7 come azioni pronte, benché siano concluse:
  gli stati di workflow di quelle azioni sono da riconciliare prima di usare
  Q32 per decidere il prossimo lavoro.
- Q34 individua **due** protocolli (Task 3c v2 e Task 3d) con due link nel
  campo pensato come singolo «Artefatto preregistrazione». In entrambi uno dei
  due è in realtà un artefatto di *risultato*. È una classificazione relazionale
  errata da correggere con tracciabilità, non un conflitto sperimentale.

## Decisione operativa

Le query e gli artefatti autorizzano a **progettare** la Task 17, non a
promuovere la LUT a sostituto causale. Il protocollo dovrà preregistrare:
intervento effettivo su Ca_HVA; replay nativo e controllo formula con la stessa
semantica temporale; LUT congelata; gbar, ECa, presenza, stato m/h, segni e
unità; input disponibili causalmente; perturbazioni appaiate; analisi per
traiettoria indipendente; criteri di arresto e nuovo set sigillato. Le lacune
relazionali sopra elencate vanno riparate separatamente e non cambiano le
soglie delle task concluse.
