# 🧬 GIADA — Task 18 originale: famiglia HH

Prerequisito: Task17f `GATE_C_PASS_BOUNDED_CAHVA`. La promozione riguarda
Ca_HVA con LUT m e h analitico; **non** dimostra una rete universale.
Questo è il primo esperimento dell'Atto III del piano originale, non Task18b
e non la condivisione dei pesi della Task22.

## ⚗️ Domanda e matrice

Trasferire il *metodo*, riaddestrando modelli separati per `Ca_HVA`, `NaTa_t`
e `K_Pst`. K_Pst mantiene due gate e introduce cinetiche/tempi diversi; non
si introduce ancora la dipendenza dal calcio di SK_E2.

Per ciascun canale: larghezza16/32 × transizioni sole/rate supervisionati ×
seed17/29/43. Sono36 modelli indipendenti, non36 run seriali. Due ensemble
vettorizzati di18 modelli, con parametri, momenti Adam e clipping separati.
I checkpoint0/1000/5000/15000 fanno parte della stessa traiettoria.
Formula autentica e LUT lineare2049 f64 sono riferimenti numerici.

La rete è `Linear(1,w) → SiLU → Linear(w,w) → SiLU → Linear(w,4)`.
Produce inf_m/inf_h tramite sigmoid e tau_m/tau_h tramite exp(log_tau), con
log_tau limitato a[-12,12]. Questa parametrizzazione positiva comune
permette di coprire i tempi molto diversi dei tre canali. Input numerico
comune(V,m,h,dt); la testa vede soloV e il solver convesso usa stato e dt:

`x_next = (1-z)*x + z*x_inf`, `z = -expm1(-dt/tau)`.

Loss per modello: media dell'errore quadratico su entrambi i gate, più
(nel solo braccio rate-supervised)0.1*MSE_inf+0.01*MSE_log_tau. Stessa
inizializzazione per seed entro capacità; stessi minibatch per tutti.
Conduzione analitica: m²h per Ca/K e m³h per Na. Nessuna rete di corrente;
open_rmse è un errore di frazione conduttiva, **non** RMSE di corrente in mA/cm².
Nessun benchmark di speedup viene rivendicato da questo run.

## 🔎 Cosa distingue l'esperimento

- Transizione buona ma rate errati: non-identifiabilità o supervisione inadeguata.
- Recupero lungo la ladder: evidenza compatibile con limite di ottimizzazione.
- Recupero solo aumentando width: evidenza compatibile con limite di capacità.
- LUT buona e reti insufficienti: rappresentabilità numerica non equivale a learnability.
- Un canale riesce e un altro no: non promuovere per media; registrare il confine di trasferimento.

Queste sono interpretazioni condizionali, non conclusioni già ottenute.
La matrice non identifica causalmente un limite assoluto di capacità da un
singolo budget. Esplorazione di tensione fuori[-135,75]mV è diagnostica;
una LUT non viene estrapolata tramite clipping nascosto.

## 🔒 Controlli e decisione

Audit compilato NEURON8.2.7: modifiche diagnostiche soltanto all'esposizione
RANGE dei quattro rate; equazioni canoniche conservate. Si verificano anche
singolarità e ramo piecewiseK_Pst, qt canonico fisso34°C e update a diversi dt.
Il processo NEURON termina **prima** di aprire Torch/CUDA.

Preflight vettorizzazione: stessa predizione e stesso passo Adam della copia
singola, con clipping indipendente. Freeze per canale/obiettivo/seed usando
soltanto development; hash dei checkpoint prima di generare fresh.
Soglie e seed sono nel JSON preregistrato, fonte eseguibile del contratto.

`valid` significa esecuzione interpretabile; `transfer_passed` richiede tutti
i gate su tutti i seed rate-supervised dei tre canali nel dominio interno.
Solo allora `task19_authorized=true`. Fallimento dell'audit o crash del
processo è un problema esecutivo, non un NO-GO scientifico del modello.

Questo esperimento testa transizioni aVtenuto costante, non la sufficienza
del soloViniziale lungo un millisecondo variabile, non il rollout closed-loop
del neurone intero e non la sostituzione causale Na/K. Queste distinzioni non
vengono annullate dal successo del playground.

## ✅ Validazione preliminare (non risultato scientifico)

Audit nativo locale: massimo errore gate Ca1.52e-13, Na1.32e-12, K2.22e-16;
driftVzero. Smoke CPU con checkpoint0/2: pipeline completa di36 modelli,
equivalenza, freeze e18 selezioni fresche per obiettivo/seed; nessuna inferenza
di prestazione dal budget ridotto. Nove test passano, inclusi quelli Task17f.
Il run registrato resta GPU con budget15000.
