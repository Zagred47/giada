# 🧠 GIADA — Compressione appresa del compartimento

**Documento storico della direzione CC richiesta il 15 ottobre 2026; non è una preregistrazione di run.**

> **Priorità successiva nella stessa giornata:** l'utente ha richiesto di
> riprendere la decomposizione modulare con efficienza ionica misurata prima
> della compressione dell'intero compartimento. Il riferimento operativo è
> ora la **sezione VI-bis, IE-01…IE-16**, nel
> [dropshot originale aggiornato](giada_modular_surrogate_roadmap.md).
> N1 e gli artefatti N2 sono conservati; prosecuzione N3…N6 sospesa.
> Questo approfondimento resta consultabile per ipotesi e confronti futuri,
> senza autorizzare la vecchia sequenza in automatico.

## 🎯 Il cambiamento

Passare da «reti che sostituiscono alcune formule dei canali, mantenendo quasi
tutta la decomposizione originale» a **una piccola dinamica appresa del
compartimento**, che comprima gli effetti dei meccanismi senza doverli riprodurre
tutti uno per uno online.

Il fine non è massimizzare la percentuale di deep learning: è ottenere un punto
migliore nel compromesso **fedeltà computazionale, latenza e memoria**, trasferendo
lavoro dall'inferenza al training. Il modello finale può conservare poche
equazioni economiche se aiutano; non deve conservarle tutte per principio.

Si resta su un compartimento attivo. **Task36 rinviata per priorità di ricerca**:
il prerequisito passivo assiale già passato non viene invalidato. Le nuove task
si chiamano **CC-01…CC-20**, non sostituiscono né rinumerano la roadmap originale.
Le venti task sono domande/decisioni, organizzate in sei notebook previsti, non
venti run manuali e nemmeno una promessa che bastino esattamente venti prove.

## 🧱 Che cosa conserviamo dal lavoro precedente

- Task35: compartimento locale con 11 canali, calcio dinamico, leak e quattro
  sinapsi stocastiche; cinque canali con **tassi dei gate appresi**, sei canonici.
  Non abbiamo appreso direttamente tutte le correnti. I 18 gate e gli stati
  sinaptici restano espliciti; il modello genera autonomamente V e Ca.
- Il candidato shared-heads ha già **644 parametri per seed**; gli independent
  ne hanno 6.260. Pochi pesi non significano pochi stati o poche operazioni.
  Il nuovo modello potrebbe avere più di 644 pesi e tuttavia costare meno per
  passo: dobbiamo misurarlo, non imporre un numero decorativo.
- La conferma Task35 copre 32 episodi indipendenti di 40 ms a 0,1 ms, quattro
  pannelli di conduttanza e sei checkpoint. Peggior RMSE V per episodio circa
  0,02544 mV; peggior pooled per modello/pannello 0,01061 mV. Queste misure non
  sono garanzia su altri orizzonti, protocolli, morfologie o throughput.
- Task10/11: il problema di un canale guidato da un path esterno non va confuso
  con un sistema accoppiato che genera quel path. Il futuro teacher non diventa
  un ingresso lecito solo perché migliora la predizione.
- Task15: la corrente diretta non vincolata non aveva battuto il braccio
  strutturato nel playground Ca-HVA e budget testati. È un precedente utile
  per progettare controlli, non un divieto di nuovi modelli aggregati.
- Il Gate D storico non ha dimostrato il vantaggio materiale richiesto. Non
  lo riclassifichiamo: il nuovo percorso formula una nuova ipotesi di costo.

Fonti interne: [Task35](task35_full_local_compartment_result.md),
[Task10](teacher_roadmap_task10_input_sufficiency_result_v1.md),
[Task11](teacher_roadmap_task11_causal_operator_result_v1.md),
[Task15](teacher_roadmap_task15_current_architecture_comparison_result_v1.md),
[amendment Task29](task29_scientific_track_amendment.json),
[prerequisito assiale](iv_ad1_passive_axial_result.md).

## 🔬 La distinzione che governa gli esperimenti

Tre errori diversi richiedono interventi diversi:

1. **Informazione insufficiente:** stati indistinguibili per il modello hanno
   futuri differenti. Più training non risolve quella specifica ambiguità:
   servono memoria, ingressi o stato migliori.
2. **Rappresentazione inadeguata:** l'informazione c'è, ma la cella scelta non la
   rappresenta bene. Servono capacità o parametrizzazione migliori.
3. **Apprendimento insufficiente:** una cella adeguata non è stata addestrata
   abbastanza bene. Qui dati, budget e supervisione possono compensare il gap.

Non assumiamo che ogni compressione peggiori necessariamente ogni metrica;
un bias migliore può anche aiutare l'apprendimento. Non assumiamo nemmeno che
lo scaling recuperi qualunque perdita. L'obiettivo verificabile è
**non inferiorità entro tolleranze dichiarate su proprietà e dominio**, non
identità biologica universale.

La grana dello **stato** e la grana del **tempo** sono assi separati: inizialmente
manteniamo 0,1 ms, comprimiamo la dinamica, e solo dopo studiamo il passo.

## ⚙️ Cella candidata: una famiglia da testare, non una scelta già vincente

Un possibile stato operativo è `s = (V, Ca, z)`, con `z` piccolo stato di memoria
appreso. Gli ingressi sono corrente esterna, parametri fisici dichiarati ed
eventi presinaptici disponibili causalmente. All'inizio manteniamo il generatore
sinaptico esplicito per isolare la compressione ionica; una prova dedicata
studierà anche l'aggregazione sinaptica.

Una rete condivisa può produrre l'aggiornamento di `z` e pochi effetti aggregati:

\[
(z_{t+\Delta t},\widehat I_{\rm ion},\widehat J_{\rm Ca})
=F_\theta(z_t,V_t,Ca_t,u_{[t,t+\Delta t)},p,\Delta t).
\]

Qui `u` indica solo l'ingresso realmente disponibile all'interfaccia: eventi
processati quando arrivano o stimoli già programmati. Non include il voltaggio,
le correnti o i rilasci futuri del teacher. L'avanzamento del simulatore deve
rispettare questa disponibilità anche quando si cambia il passo temporale.

Il bilancio capacitivo economico può rimanere un layer strutturale, con correnti
ioniche positive uscenti e corrente esterna positiva entrante:

\[
C_m\,\Delta V/\Delta t
=I_{\rm ext}-\widehat I_{\rm ion}-I_{\rm syn}-I_{\rm leak}.
\]

Questa è una **identità di bilancio**, non la prescrizione di Euler esplicito:
discretizzazione, punto di valutazione delle correnti e stabilità vanno
confrontati. Tutte le quantità devono usare unità coerenti, in densità oppure
assolute con area esplicita. La chiusura appresa può incorporare l'effetto
integrato del passo; una testa alternativa può predire direttamente la
transizione. Non dobbiamo ricostruire tutti i gate per definizione.

Tre opzioni sensate da confrontare:

- corrente ionica aggregata + flusso Ca separato;
- poche conduttanze effettive raggruppate per potenziale di inversione, con
  correnti `G_r(V-E_r)` e memoria appresa;
- transizione diretta di V/Ca con memoria, come controllo meno strutturato.

La corrente netta non identifica il flusso di calcio: correnti di segno opposto
possono cancellarsi pur lasciando dinamiche Ca molto diverse. Non eliminiamo
questa distinzione. Non imponiamo nemmeno `z ∈ [0,1]` solo perché i vecchi gate
lo erano: i vincoli devono avere significato nella nuova rappresentazione.

Il teacher può fornire stati e correnti **come target ausiliari durante il
training**. Online niente reset a ogni passo con lo stato vero. Un eventuale
encoder iniziale deve avere un contratto esplicito; il warmup deve essere
causale. La stochasticità non può essere sostituita dalla media senza cambiare
l'obiettivo scientifico: distinguere confronto pathwise a draw appaiati da
confronto distribuzionale con draw indipendenti.

## 🗺️ Venti task, sei blocchi di lavoro

Il [piano strutturato](giada_compartment_compression_plan_v1.json) contiene
dipendenze, contrasti e deliverable per ogni task.

| Notebook previsto | Task | Domande e risultato atteso |
|---|---|---|
| **N1 — Contratto e informazione** | **CC-01** contratto; **CC-02** baseline/costi; **CC-03** collisioni; **CC-04** memoria | Quali proprietà dobbiamo preservare? Quale informazione può essere tolta? Dove costa il modello attuale? |
| **N2 — Cella unificata** | **CC-05** controllo a gate espliciti; **CC-06** scala latente; **CC-07** bias fisico; **CC-08** inizializzazione | Possiamo apprendere una cella compatta? Il limite è condivisione, memoria o testa di uscita? |
| **N3 — Learnability** | **CC-09** dati×budget; **CC-10** distillazione; **CC-11** rollout training; **CC-12** supporto raro | Quale errore si recupera con training? Quanto aiutano target interni ed esposizione ricorsiva? |
| **N4 — Proprietà e generalizzazione** | **CC-13** interventi sui parametri; **CC-14** sinapsi; **CC-15** passo temporale; **CC-16** dinamiche lunghe | Conserviamo risposte causali, memoria e stochasticità? Possiamo aumentare il passo senza ingressi privilegiati? |
| **N5 — Utilità operativa** | **CC-17** porta componibile; **CC-18** Pareto misurato | Il modello è collegabile in futuro? Costa davvero meno a pari fedeltà? |
| **N6 — Conferma** | **CC-19** test indipendente congelato; **CC-20** decisione/rientro | Quali affermazioni possiamo sostenere? Con quale versione possiamo proporre di riprendere Task36? |

Il controllo CC-05 non è la destinazione finale: mantenere per una prova lo
stato esplicito evita di confondere «una sola rete non apprende» con «abbiamo
eliminato troppa memoria». CC-06/07 sono il vero cambio di livello.

## 🧪 Parallelismo informativo, non una griglia senza criterio

N1 può profilare e fare controfattuali indipendenti sugli stessi dati. In N2
alcune celle dipendono dai controlli precedenti: nessuna selezione silenziosa
nello stesso notebook; usare stop/go espliciti e salvare la decisione. N3 usa
una piccola matrice dati×budget e contrasti separati di loss/esposizione/supporto,
con una sola interazione prioritaria preregistrata; non il prodotto cartesiano
di tutte le architetture e tutti gli iperparametri.

N4 raccoglie sfide ortogonali sullo stesso candidato, con derivazioni distinte
per sinapsi o passo. Non unire automaticamente due modifiche che passano da
sole: la combinazione scelta va verificata su development prima del freeze.
N5 riusa i checkpoint, N6 usa nuovi dati e seed di valutazione dopo il freeze.

Condividere seed, stream e minibatch dove il confronto lo consente. Salvare
checkpoint crescenti nella stessa traiettoria. Vettorizzare i modelli solo dopo
equivalenza con il riferimento sequenziale (predizioni, gradienti, Adam e
clipping indipendenti); misurare throughput reale. Più memoria GPU occupata
non dimostra maggiore efficienza.

## 📏 Contratto di valutazione da fissare prima di N1

- **Precisione:** V, Ca e carica/corrente pertinente; medie, quantili e peggiori
  episodi, separati per regime. Floor formula/native calibrato prima di giudicare
  la rete. Costo/accuratezza riportati per ms biologico simulato.
- **Comportamento:** timing e conteggio spike, recupero/refrattarietà,
  adattamento e risposte a perturbazioni, solo dove il teacher li manifesta.
  Episodi senza spike non possono validare la dinamica degli spike.
- **Causalità:** ingressi identici, maschere off, variazioni dei parametri,
  perturbazioni iniziali, reset/permutazione memoria, nessun futuro teacher.
- **Generalizzazione:** split per episodi, famiglie di stimolo, parametri e seed;
  non finestre adiacenti della stessa traiettoria in ruoli differenti. Separare
  interpolazione, extrapolazione e combinazioni inedite.
- **Costo:** pesi, stato persistente e temporanei, kernel/operazioni, latenza
  stabilizzata e throughput a batch rappresentativi, RNG e gestione eventi
  inclusi. Formula e rete entrambe ottimizzate simmetricamente; compilazione e
  warmup separati dal costo stazionario. Kernel custom e Tenstorrent rinviati.
- **Gates:** le vecchie soglie V/Ca restano documentate, ma non autorizzano a
  chiamare equivalente un errore cento volte maggiore del baseline attuale.
  CC-01 fisserà margini di non inferiorità rispetto alla baseline, requisiti
  comportamentali e un minimo vantaggio operativo con incertezza misurata.
  Numeri, budget, seed, capacità massime e stopping rule sono ancora da
  preregistrare; questo documento non li inventa retroattivamente.

Le evidenze sealed precedenti, già conosciute, diventano benchmark storico e
diagnostica: non sono un nuovo test indipendente. I bracci oracle non sono
selezionabili. Una diagnosi tecnicamente valida può avere un esito scientifico
negativo: `valid` non equivale a `promoted`.

## 🚦 Regole per non restare bloccati in sottotask infinite

Prima di ciascun notebook si fissano budget totale, massimo numero di bracci e
una sola tornata di riparazione diagnostica; oltre quel limite si registra una
decisione e si rivede il piano, senza attribuire al problema un'impossibilità
universale. Un guasto tecnico ha registro separato da un fallimento scientifico.

- Collisioni persistenti → correggere memoria/interfaccia, non comprare solo più training.
- Curva in miglioramento → aumento limitato e dichiarato di budget, non cambio simultaneo di tutto.
- Controllo esplicito passa, latente fallisce → problema di compressione/inizializzazione da isolare.
- Accuratezza passa, runtime no → nessuna rivendicazione di accelerazione; rivedere il grafo di calcolo.
- Solo medie buone, comportamenti o regimi falliscono → niente promozione del compartimento.
- Conferma fallita → non riutilizzare il test come sealed dopo il tuning.

CC-20 può chiudere con promozione, utilità scientifica senza vantaggio operativo,
oppure no-go circoscritto. La roadmap non obbliga a far passare il candidato.

## 📚 Metodo e riferimenti

Riletti i tre documenti grezzi forniti dall'utente su Allen-Zhu/design e l'indice
Neuro_AI. Sono materiale metodologico di conversazione, non evidenza sperimentale
GIADA né citazioni primarie automaticamente attribuibili ad Allen-Zhu.
Applichiamo: componenti/difficoltà controllate, priorità alla learnability,
mini scaling laws, probing interno, confronti allineati e verifica dell'uso
effettivo della complessità introdotta. L'unità atomica qui diventa una proprietà
o una memoria del compartimento, non necessariamente un singolo canale.

Il lavoro di Gupta e Lermusiaux sulle
[neural closure con memoria](https://arxiv.org/abs/2012.13869) mostra, nei sistemi
studiati dagli autori, come apprendere effetti dei modi eliminati usando dati
ad alta fedeltà. Motiva le nostre prove sulla memoria; **non dimostra** che una
specifica cella GIADA compatta sarà equivalente o più veloce.

Discussione utente di partenza:
`C:/Users/belli/.codex/attachments/48adf836-5432-43d2-93fd-b5e19b60e929/Testo incollato.txt`.
L'allegato è stato trattato come materiale da valutare, non come istruzioni da
eseguire automaticamente. In questa fase nessun nuovo training viene lanciato.
