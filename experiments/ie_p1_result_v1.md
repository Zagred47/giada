# ⚡ GIADA — IE-P1: costo residente e margine del blocco gate

**Completato e verificato nella versione Kaggle 3**, codice
`b177d01bb693da05224d961d0ba2719dcc94bd78`, T4/FP32. Nessun training.
[Notebook eseguito](https://www.kaggle.com/code/alessandrobelli/giada-ie-p1-ionic-cost-and-gate-bottleneck).

## 📊 Risultato primario — batch642 per seed, tre seed vettorizzati

| Core | Tempo completo (ms) | Con gate da buffer (ms) | Risparmio appaiato |
|---|---:|---:|---:|
| Formule | 0,464987 | 0,392049 | 15,92% |
| Ibrido independent | 0,771835 | 0,396138 | 48,23% |
| Ibrido shared_heads | 0,885619 | 0,395850 | 55,60% |

Tempi mediani per chiamata dell'ensemble. La frazione è la mediana delle
differenze relative appaiate, non necessariamente il rapporto delle mediane.
Per formule: bootstrap entro questo run 14,62–16,42%. A batch1/4096 il
risparmio diagnostico formula è rispettivamente 18,74%/15,25%.
File completi in `results/ie_p1_v3_kaggle_b177d01/`.

## 🔎 Interpretazione

Il buffer sostituisce il calcolo dei gate su input fissi identici. Non è un
modello causale o un'accelerazione appresa. Il residuo conserva aperture,
conduttanze/correnti, Ca/SK, solver e NMDA. La differenza descrive l'opportunità
osservata in questa composizione; fusione del compilatore e movimento dei
dati impediscono di usarla come costo additivo o limite universale.

Gli ibridi hanno un overhead recuperabile importante, ma confrontare un
futuro candidato soltanto con essi sarebbe insufficiente: contro le formule
competitive il margine gate-only osservato è molto più piccolo. Prima di
promettere guadagni grandi, IE-03 deve esaminare anche il residuo e il lavoro
ionico che un diverso operatore può realmente eliminare. La misura non
dimostra che una rete per conduttanze sarà veloce o accurata.

Eventi/plasticità/RNG host, trasferimenti, caricamento e compilazione sono
esclusi. **IE-02 globale resta aperta**, come il contratto definitivo IE-04 e
il nuovo playground IE-05. Nessuna promozione di candidato o Task36.
Il profiler di supporto riporta zero eventi in alcune celle gate-only:
non significa costo zero e non viene usato per attribuire percentuali di
compute. Le conclusioni quantitative usano i blocchi temporali sincronizzati.

## 🧪 Verifiche e incidenti conservati

- Equivalenza eager della decomposizione verificata prima dei tempi; test
  CPU su stati storici e cattura fullgraph passati.
- Tutti i 27 preflight compilati della v3 passano i limiti FP32 già registrati
  in CC-N1, riadottati prospetticamente nell'emendamento v2.
- Massimo errore V compilato/eager misurato 1,526e-5 mV. I campioni grezzi
  dei tempi sono conservati; le nove mediane delle frazioni sono ricalcolate
  indipendentemente prima della registrazione SQLite.
- v1: arresto tecnico prima dei tempi principali al batch642, usando un
  nuovo limite compilato/eager più stretto di N1. Risultato incompleto,
  conservato senza promozione retroattiva delle misure parziali.
- v2: errore di invio del notebook, con import `subprocess` rimosso durante
  l'inserimento della revisione. Nessun esperimento eseguito. v3 riparte dal
  notebook in repository, preserva gli import e aggiunge la revisione in testa.

## 🧭 Decisione

Usare IE-P1 per selezionare il bersaglio e definire i limiti del confronto
successivo. Non avviare automaticamente training o ricerca evolutiva.
L'obiettivo resta qualità e risparmio insieme, confrontati anche con formule
competitive e verificati infine nel compartimento.
