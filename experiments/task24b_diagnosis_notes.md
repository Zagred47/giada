# 🔬 Task24 → Task24b: diagnosi circoscritta

## ✅ Evidenza acquisita

Task24 (commit4fad2c6) è tecnicamente valida: CRC, freeze/checkpoint/source SHA,
oracle nativo e preflight vettorizzato verificati. Shared-heads e conditioned
passano3/3; indipendenti0/3. Il risultato originale non viene reinterpretato.

I dieci sforamenti sono tutti Nap_Et2 h, oppure il gate RMSE che lo include.
Seed17 fallisce solo held1000ms (0.00206812 >0.002). Seed29 e43 falliscono anche
one-step; seed43 fallisce held10000ms. Le correnti e i percorsi imposti non
compensano questo fallimento: i criteri sono congiuntivi.

Development indipendente width16: worstscore15k=1.31031,30k=1.42078.
Width32:1.82585 e2.48059. Questo suggerisce oscillazioni ma non dimostra
ancora una causa né incapacità rappresentativa. Anche alcuni modelli condivisi
peggiorano a30k: aumentare il budget aLRcostante non è una soluzione dimostrata.

## 🧪 Contrasti appaiati

| Fattore | Controllo | Intervento | Che cosa distingue |
| --- | --- | --- | --- |
| Schedule | LR0.003 costante | Cosine15k→60k fino0.0001 | Ottimizzazione tardiva |
| Clipping | Norm1 perseed/bundle | Norm1 perseed/canale | Interferenza fra reti indipendenti |
| Capacità | width16 | width32 | Capacità vs trainability |
| Budget | checkpoint30k | checkpoint60k | Mini scaling lungo stessa traiettoria |
| Parametri cinetici | Frozen learned rates | Sostituire inf/tau/separatamente/entrambi | Causa atomica dell'errore Nap_h |
| Numerica | SolverFP32 | SolverFP64, identiche rate apprese | Accumulo numerico vs errore cinetico |

24 sistemi addestrati nella stessa GPU, seed vettorizzati, stream appaiati.
La diversa riduzione del clipping è l'unico cambiamento in quell'asse;
Adam resta indipendente perparametro. Frequenze e norme registrate: se il
clipping non si attiva, il contrasto non può spiegarne il fallimento.

## 🔒 Selezione e decisione

Probe development held-V composto a1000/10000ms su nuovi V: è un update
analiticamente composto con rate congelate, non un rolloutFP32 iterativo.
Il rollout iterativo reale resta criterio fresh. Una width/checkpoint comune
perarm minimizza il worstscore su3seed. Si salva anche la scelta one-step-only
come confronto development, senza aprire altri test o scegliere post-hoc.

Fresh casuale nuovo (2414xx), griglia offset0.71 invece0.37. Seams edendpoint
fisici e percorsi controllati ripetuti sono dichiarati, non spacciati per nuovi
regimi. Nessun training o selezione sul fresh precedente.

Solo il braccio primario preregistrato `decay_channel` può autorizzare Task25
se supera3/3 con le soglie originarie. Gli altri bracci diagnosticano le cause.
I due modelli condivisi vengono solo rivalutati frozen sulla stessa popolazione.
Non si promette che il repair funzionerà: un NO-GO resta un risultato utile.

## 📌 Limiti

V imposto, sei gate espliciti, corrente analitica economica conservata.
Nessuna dinamica autonoma, full-neuron acceleration o nuovo confronto prestazionale
con ELM. Task24b è una sottotask, non una rinumerazione della roadmap.
