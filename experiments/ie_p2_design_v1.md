# ⚡ GIADA — IE-P2: distinguere il costo ionico dal costo del solver

**Direzione approvata il 16 ottobre 2026.** IE-02/03/04, fase residente;
nessun nuovo identificativo numerico della roadmap e nessun training.

IE-P1 ha misurato un margine gate-only di circa il 16% sul core formula.
Prima di scegliere una nuova rete occorre capire che cosa resta: aperture e
conduttanze, calcio/SK, assemblaggio sinaptico, iterazioni NMDA e costo di
avvio delle operazioni. P2 conserva le equazioni e cambia solo quali output
sono disponibili da un buffer diagnostico.

## 🧪 Contrasti nella stessa esecuzione

| Variante | Computazione sostituita dal buffer |
|---|---|
| full | nessuna: ricomposizione completa |
| cached_gates | aggiornamento iniziale dei gate, come P1 |
| cached_conductance | soltanto somme finali g e gE; Ca/SK e gate restano |
| cached_prefix | gate, Ca/SK e readout ionico insieme |
| cached_voltage | assemblaggio e soluzione del voltaggio |
| cached_both | entrambi i blocchi: floor di invocazione e restituzione buffer |

Due componenti isolate (prefisso ionico e soluzione V) aiutano la diagnosi;
i loro tempi non vengono sommati. Tutti i controfattuali restituiscono anche
g_sum e gE_sum, affinché il compilatore non cancelli il readout quando il
voltaggio è in cache. Il vecchio ABI a tre output misura quanto questi
strumenti alterino il costo del riferimento. Una differenza ≥10% richiede
di qualificare le conclusioni sul bersaglio.

Ogni variante è provata come funzione Inductor ordinaria e, dove disponibile,
come replay CUDA Graph degli **stessi** calcoli su input fissi. Questo è un
controllo diagnostico dell'overhead di invocazione; non certifica già il
rollout con ingressi variabili o uno speedup appreso. Nessun kernel custom.

## 📏 Disegno e validazione

- Stessi stati storici Task35 e seme di campionamento P1; batch1/642/4096,
  tre seed vettorizzati, T4/FP32, TF32 disabilitato, dt0,1ms.
- Ricomposizione confrontata con il vecchio core; stesso ordine Ca→SK.
- Ogni buffer conserva gli output sul medesimo stato; compiled e graph
  verificati contro eager prima del timing, con limiti compilatore N1/P1v2.
- Tempi wall e CUDA su 30 blocchi da20chiamate; ordine intercalato fra modi
  e varianti e invertito fra ripetizioni, compilazione/capture escluse.
- Sample e scarti salvati prima di applicare gate; nessun fallback eager
  per una compilazione fallita. Graph indisponibile rende inconclusiva la
  sola ipotesi di overhead, mantenendo leggibili le prove compilate.
- Errori tecnici sempre conservati; niente etichetta di NO-GO del modello
  per un processo interrotto.

## 🧭 Regola prospettica di scelta del bersaglio

Al batch642: ≥50% di riduzione full con graph suggerisce una priorità di
esecuzione; ≥30% per prefisso o solver, con intervallo positivo, suggerisce
un blocco materialmente rilevante; ≥20% per readout suggerisce un'opportunità
più circoscritta. Sono criteri diagnostici di priorità, non soglie di qualità
per promuovere reti. Tutti i risultati vengono riportati, anche negativi.

Le ipotesi possono essere vere insieme. Fusione e output interagiscono, perciò
le percentuali non sono parti additive del tempo totale. Se una stima è vicina
alla soglia, ripetere su un nuovo run prima di una decisione importante.

Il risultato serve a completare IE-03 e precisare IE-04/05. Rimangono aperti
eventi/plasticità/RNG host e trasferimenti; nessuna dichiarazione di IE-02
globale conclusa. OpenEvolve/RunPod e il prossimo training restano condizionati
alla scelta del bersaglio e a un proprio protocollo numerico.
