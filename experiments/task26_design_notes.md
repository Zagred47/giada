# 🔬 Task26 — Joint vs independent

## ✅ Punto di partenza

Task25 ha confermato la composizione indipendente e quella con calcio condiviso
in 3/3 seed. La compressione del sodio manca due soglie di m nel seed43 sul
supporto activation_boundary più ampio; non è un errore tecnico né prova di
interazione negativa tra canali. Non promuoviamo quel bundle.

## 🧪 Quattro configurazioni, una sola esecuzione

Cinque MLP indipendenti contro un trunk a due layer SiLU condiviso con cinque
head distinti. Ogni head produce mInf, hInf, log(mTau), log(hTau). Update
esponenziale convesso e formula di corrente restano espliciti. Tutti ricevono lo
stesso V/stati/dt; soltanto V entra nella MLP. Non introduciamo dipendenze
artificiali fra stati dei canali.

Width16/32 e checkpoint0/5k/15k/30k/60k coprono capacità e budget. A width16:
independent1860 vs shared644; a width32: independent6260 vs shared1780.
Shared32 vs independent16 è quasi allineato in parametri (-4.30%), non esatto.
Stessi stream appaiati, inizializzazione a widthfisso, seed vettorizzati e Adam
foreach; niente processi GPU concorrenti né risincronizzazioni perstep su CPU.

## 🔍 Probe e interpretazione

Rate e gate percanale, correnti individuali e somma sui20pannelli di Task25,
conflitto dei gradienti del trunk su minibatch tecnico fissato, mini scaling
laws, identità scambiata e prestazioni GPU sincronizzate. Nessun probe decide
il checkpoint. I target di corrente non entrano nella loss: sono Task27.

Freeze prima dei nuovi fresh2604xx. Tre domini, estremi marginali strutturati,
helditerativo1–10000ms, cinque path a0.025/1ms. Le soglie sono quelle di Task25.
Il controllo semigroup development è analitico composto, non viene spacciato per
rollout iterativo. Eventuali sforamenti restano risultati validi NO-GO.

## 🔒 Dipendenze

Passare3/3 sia independent sia shared autorizza soltanto preparare Task27.
Il GateD richiede anche evidenza materiale di compute e il blocco completo;
nessuna promozione a voltaggio autonomo, SK_E2 o CaDynamics.
