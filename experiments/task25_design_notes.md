# 🔬 Task25 originale: composizione eterogenea

## ✅ Informazione da Task24b

Archivio verificato CRC, freeze, checkpoint e sorgenti contro il commit bde23cc.
18 righe fresh: tutti i sei bracci superano i criteri nei tre seed. Il controllo
primario decay_channel è valido con1116 parametri. Il modello condizionato
congelato420 passa, mentre shared_heads1516 non offre compressione.

I contrasti a60k hanno miglioramento mediano43.75% per schedule,49.88% per
budget60kvs30k,51.08% perwidth32vs16. Queste mediane di confronti appaiati non
dimostrano una legge universale. Anche il controllo aLRcostante passa conwidth32.
Clipping mai attivo: il cambio di scope non causa il recupero.

Nap_h del vecchio modello: a1000ms correggere tau riduce maggiormente l'errore;
a10000ms correggere inf è più efficace. Correggere entrambe lascia un floorFP32
9.38e-6/2.66e-5. Identici rate appresi inFP64 non recuperano l'errore~.002:
non era prevalentemente accumulo numerico. Non è una decomposizione additiva.
Il NO-GO della Task24 resta immutato; Task24b è una nuova conferma dipendente.

## 🧪 Matrice senza nuovo training

Ca_HVA/Ca_LVAst indipendenti o shared_heads (Task23), combinati con
NaTa_t/NaTs2_t/Nap_Et2 indipendenti riparati o conditioned (Task24b/Task24).
Quattro bundle, tre seed appaiati, nuovi V/stati/dt. Il controllo1860parametri
e il bundle massimo-compresso860 sono congelati prima dei nuovi dati.
Nessun branch è scelto in base al nuovo risultato.

Le cinetiche comprendono attivazione/inattivazione rapida, lenta e persistente;
le correnti hanno potenze diverse e driving force di segno opposto in alcuni V.
La somma accurata non può nascondere un singolo canale inaccurato: si controllano
tutti i gate, ogni canale, ciascun path, ogni seed e il peggior pannello corrente.
I pannelli di assenza/dominanza e la cancellazione sono nello stesso run.

## 🔒 Interpretazione e limiti

V imposto, dieci occupanze esplicite, corrente analitica economica mantenuta.
Fast0.025ms e slow1ms sono due percorsi imposti con updateheld-V per substep,
non la dimostrazione di sufficienza degli endpoint a1ms. Held1–10000ms è iterativo.
FormulaFP32/LUT2049 e solverFP64 sono diagnostici non selezionabili.
Nessun SK_E2 o CaDynamics senza il relativo input calcio validato.

Solo il controllo indipendente3/3 può autorizzare la preparazione della Task26.
La Task26 resta il confronto jointvsindependent, la27 i target individuali/total,
la28 il blocco completo. Il presente esperimento non sostituisce queste task
né autorizza il closed-loop del neurone. Nessuna accelerazione di runtime
viene dedotta dalla sola riduzione del numero di parametri.
