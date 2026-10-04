# 🧩 Task23 originale: prima composizione calcio

## 🔬 Domanda

Ca-HVA e Ca-LVA possono essere appresi insieme senza perdere accuratezza dei
gate o mascherare errori delle correnti individuali nella loro somma?
Task22 ha confermato il riferimento indipendente, non il peso condiviso.

## ⚖️ Confronto

Tre famiglie × due width × tre seed indipendenti. Dati, minibatch e loss appaiati;
Adam e clipping separati per seed. Il numero di parametri non è pareggiato:
viene riportato per misurare il tradeoff esplicito. La corrente economica
`gbar*m²*h*(V-Eca)` resta analitica. Nessuna rete del voltaggio.

## 🛡️ Verifiche

Audit nativo delle due formule, incluso shift e qt Ca-LVA, corrente e update.
Quattro test locali passati: trascrizione, equivalenza Adam, separazione delle
correnti, forma e boundedness dei rollout. Smoke end-to-end CPU con un solo
step passato; non è un risultato scientifico o una misura di learnability.
Subprocessi nativo/GPU separati, working directory esplicita, output compatto
e report di errore distinto da NO-GO scientifico.

## 🚧 Limiti

V ed Eca sono imposti. Percorsi a substep .025ms confrontati con la formula
discreta allo stesso passo; non dimostrano sufficienza dell'input a1ms.
Baseline LUT sulle transizioni, non sui rollout in questa matrice.
OOD diagnostico; selezione solo development; freeze prima di fresh.
Sezione IV resta un filone separato con dipendenze esplicite.

## 🗃️ Registrazione

Le metriche scalari development/fresh/rollout della Task22 sono importate con
codici stabili e relazioni verso run, valutazioni e artefatti originali.
L'indice temporaneo di `ValidatedBatchMirror` evita scansioni ripetute per
codice durante il batch; commit e verifica continuano a usare il validatore
originale. Suite mirror:20 test; cache/batch:2 test passati. Nessuna modifica
allo schema e nessun accesso ad Airtable.
