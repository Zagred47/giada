# 🧬 GIADA Task18d — struttura di mTau

## 🔎 Risultato18c e diagnosi

Run tecnicamente valido: oracle nativo, vettorizzazione, codice e checkpoint
verificati. Uniform e coda passano3/3seed; kink passa2/3. Seed17 ha massimo
0.010521 contro0.01. Il probe del test consumato localizza tutti i massimi su
m a−59.931mV, dt25ms; tau esatta riduce errori a0.000034/0.000258/0.000020.
Sul medesimo nuovo test, il parent18b è peggiore in uniform/coda/kink; OOD
negativo peggiora con18c e viene conservato come limite diagnostico.

## ⚗️ Confronto architetturale

Tre famiglie ×treseed, tutte iniziano dalla funzione congelata18c.

| Famiglia | Struttura aggiunta | Parametri effettivi |
|---|---|---:|
| Smooth | controllo MLPwidth32 |1252 |
| Feature | distanza assoluta da−60mV come feature |1284 |
| Split | due head mTau apprese, selezionate daV |1285 |

Extra feature inizializzata a peso zero; secondo head copia il primo. Stessi
stati, tensioni, dt, minibatch, target, loss, Adam azzerato e checkpoint.
Capacità effettive entro3%; storage padded comune non conta come capacità
attiva. Differenza di32/33parametri è un limite del contrasto, dichiarato.
La soglia−60 è conoscenza strutturale del teacher; nessun endpoint futuro è
input. Nessuna formula tau analitica incorporata nei due head appresi.

## 🔒 Griglia e conferma

L'errore di un gate è affine nell'occupazione iniziale aV/dt fissati: basta
testare occupazione0e1 per delimitarne il massimo sullo stato. Questa proprietà
non certifica automaticamente la corrente polinomiale, né tutto il continuumV.
Development usa201voltages in[-65,-55]mV più soglia e vicini; fresh usa401.
Entrambi testano i7dt registrati e stati0/1. Griglia finite, non prova universale.
Nuovi seed183xxx; freeze di una famiglia/checkpoint comune ai3seed primafresh.
Gates originali restano; griglia estremi è un requisito aggiuntivo registrato.
Max<0.008 è un indicatore di margine, senza sostituire la soglia0.01.

Parent18c viene valutato sugli stessi fresh dopo freeze. Tutti i contrasti
matched a budget comune sono salvati. Il scope resta cinetica isolata K_Pst;
Na/Ca conservano successi precedenti. Task19 richiede tutti i gate registrati.

## ⚙️ Esecuzione

Oracle nativo e CUDA in worker distinti;9modelli con parametri indipendenti
vettorizzati. Test di equivalenza include tutti i tipi architetturali.
Checkpoint0/5k/15k/30k; log compatti e report di errore tecnico distinti dagli
esiti scientifici. Pubblicare revisione prima del lancio e verificare readback.
