# 🐢 Task 21 originale — dinamica lenta

## 🔬 Componente e ipotesi

Si isola **h di Nap_Et2**, non l'intero canale a due gate. Il teacher canonico
usa hInf(V), hTau(V) e qt=2.3^1.3; tau è circa 350–2194 ms nel dominio
[-135,75] mV. Si preservano anche i rami locali a −38/−17/−64.4 mV.

Un errore piccolo su una transizione di 1 ms potrebbe essere ottenuto anche
con una cinetica sbagliata, perché h cambia poco. La matrice separa:

- **informazione temporale:** dt brevi (≤1 ms) contro multiscala (fino a 10 s);
- **supervisione:** transizione sola contro transizione + hInf/logtau;
- **capacità:** width 16/32, 338/1186 parametri indipendenti;
- **budget:** checkpoint 0/1000/5000/15000/30000 nella stessa traiettoria.

24 modelli, vettorizzati con Adam e clipping indipendenti, seed 17/29/43.
V, h iniziale, inizializzazione e indici minibatch sono appaiati; **dt differisce
intenzionalmente** nel contrasto di informazione. I test/development sono comuni.

## 📏 Valutazione e decisione

La selezione usa solo development, stesso criterio worst-seed/worst-domain
per tutti i bracci, e viene congelata prima di generare il fresh. Il candidato
primario è preregistrato: multiscale/rate, tutti e tre i seed. Nessun fallback
post-hoc. Soglie e contrasti sono nel JSON, il mirror contiene le relazioni.

Rollout a **passo 1 ms**, fino a 10000 ms, contro soluzione f64 a V costante.
Si riportano anche persistence, formula float32 iterata e update macro usando
le rate apprese, per distinguere errore cinetico e floor numerico. I dt lunghi
di training sono soluzioni esatte a V mantenuto costante: non autorizzano
coarse-graining di un neurone con V variabile.

## 🚧 Limiti espliciti

Non si testano m, corrente Nap completa, voltaggio endogeno, condivisione dei
pesi o speedup. `open_rmse` è solo un alias dell'errore h per compatibilità del
report, **non** errore dell'apertura totale m³h. La misura di performance è
deferita: CUDA qui serve per training, non dimostra vantaggio computazionale.

## ✅ Verifica pre-run

11 test Task19/20/21 passati; audit NEURON 8.2.7: rate scaled max
1.4755162e−12, update max 2.2065683e−13, drift V nullo. Smoke CPU completo
con 12 selezioni e rollout fino 10000 passi: pipeline valida, nessun risultato
scientifico o autorizzazione inferiti dallo smoke. Audit nativo e controllo di
equivalenza saranno ripetuti nella sessione Kaggle pin-nata.
