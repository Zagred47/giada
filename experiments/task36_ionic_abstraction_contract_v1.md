# 🧭 Task 36 — Contratto e scala di astrazione ionica

Questa è la **nuova Task36** del dropshot del 17 ottobre 2026; l'exTask36
«due compartimenti attivi» è ora Task72. È un audit locale e una specifica
delle interfacce, senza training. La specifica leggibile dalle macchine è
[`task36_ionic_abstraction_contract_v1.json`](task36_ionic_abstraction_contract_v1.json).

## 📚 Ciò che possiamo riusare

| Evidenza | Lezione e limite |
|---|---|
| Task35 | Compartimento ibrido valido nei suoi 32 casi/40 ms; non tutta la fisiologia né speedup |
| CC-N1 | V/Ca correnti non identificano sempre il futuro: conservare memoria e inizializzazione |
| CC-N2 | Fallisce anche il nuovo controllo explicit18; il fallimento non isola la sola compressione |
| IE-P1/P2 | Overhead e lavoro residuo contano; sostituzioni da buffer sono diagnosi, non modelli |
| IE-P3 | Readout esatto vettorizzato batte MLP testate; apprendere sola algebra economica non ha aiutato |
| IE-P4/P5 | Supervisione interna/deployment aiutano in certi contrasti; coefficienti discreti conservano 18 stati |
| IE-P6 | Sei mappe locali superano qualità e costo residente specifico; LUT analitica ancora più veloce |
| IE-P7 v4 | Conferma nativa breve delle mappe locali; criterio operativo del 10% non superato |

I test già aperti diventano evidenza storica/development. Le nuove conferme
richiedono traiettorie disgiunte. Le tabelle apprese IE non sono reti ricorrenti
a memoria compressa. I criteri diversi tra Task35, CC e IE restano distinti.

## 🔌 Porta ionica e fasi

Inventario: **11 canali e 18 stati**, di cui 17 gate V-dipendenti e un gate
SK dipendente dal calcio. Livello1 conserva tutti gli stati fisici; i due
livelli successivi astraggono prima per famiglia, poi condividono la memoria.
Stato e pesi sono concetti diversi; pesi condivisi non significano un unico
stato globale per tutti i compartimenti.

1. Aggiornare i gate V-dipendenti da V corrente e stato precedente.
2. Calcolare la sorgente Ca dai gate aggiornati; CaDynamics invariato aggiorna cai.
3. Aggiornare SK con **cai nuovo** e SK precedente, senza avanzare due volte gli altri gate.
4. Produrre conduttanze/correnti individuali e sum_g/sum_gE per il solver invariato.

Il readout iniziale conserva `I_k=gbar_k*a_k*(V-E_k)`. La componente appresa
è l'evoluzione della memoria; in seguito a_k può essere appresa direttamente,
senza calcolare prodotti/potenze dei gate. La rete deve rispettare la porta
a due fasi Ca/SK: «un modello» non implica «una sola chiamata» se la dipendenza
fisica richiede una seconda fase. Tutte le chiamate entrano nel costo.

Unità: V mV, cai mM, g S/cm², corrente mA/cm², dt0,1ms. Correnti entranti
negative. V inizialmente nel dominio [-100,60]mV delle baseline IE; il supporto
Ca e gli stress vanno fissati in Task37, prima del training. Nessun endpoint
teacher o correzione dello stato entra nell'inferenza. Parametri cinetici e
temperatura diversi dal teacher richiedono un contratto ulteriore.

## 🔬 Ipotesi e valutazione

Le sette ipotesi nel JSON distinguono update congiunto, copertura rara,
sforzo di training, sufficienza della memoria, bias del readout, trasferimento
spaziale e capacità diagnostica dei valutatori. Sono **aperte**; questa task
non dimostra che una rete più astratta sia più veloce.

Le otto specifiche di valutazione coprono correnti, carica, transitori,
interventi, cambi di regime, rollout, revisione qualitativa e costo.
Task37 deve trasformarle in valutatori eseguibili con soglie numeriche
prospettiche, supporto per regime e controlli difettosi riconosciuti.
I valutatori rapid-fire storici sono riferimenti di progettazione, non codice
già validato a dt0,1ms o per le correnti ioniche.

Un fallimento raro apre la matrice **diversità dei dati × budget di training**:
nuove traiettorie indipendenti e ripetizioni non sono equivalenti. Riportare
esposizione e compute per regime, cercare saturazione e regressioni sui regimi
comuni. Nessuna inferenza «basta scaling» senza contrasto controllato.

## ⚡ Bersaglio computazionale e prossima decisione

Il bersaglio è il **lavoro ionico dai gate alle correnti**, inclusi stato e
readout, con solver/CaDynamics/sinapsi fissi. Il costo CPU/RNG/IPC di IE-P7
non diventa la ragione per cambiare questo bersaglio. Il risparmio operativo
resta un esito distinto; non promettiamo che un miglioramento locale lo
domini. Le baseline sono formule competitive, LUT, Task35 e mappe IE congelate.

L'audit Task36 produce inventario, hash, registro dei gap e decisione.
Il suo GO autorizza **la preparazione della Task37**; non dichiara pronti
training, valutatori nuovi, Task38 o prova multicompartmentale. Task37–39
potranno condividere un notebook, con prerequisiti e freeze separati.
