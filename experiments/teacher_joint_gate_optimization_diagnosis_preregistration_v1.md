# 🔬 GIADA Task 3b — diagnosi causale dell’ottimizzazione condivisa

## 🎯 Domanda

La Task 3 ha mostrato che la cella con due trunk indipendenti apprende `m`, `h` e
la composizione `m²h`, mentre il trunk condiviso parameter-matched è instabile fra
seed. La Task 3b distingue cinque spiegazioni senza riaprire il fresh test:

1. obiettivo informativamente sbilanciato, perché soltanto `h` riceveva rate supervision;
2. interferenza fra i gradienti di `m` e `h` nel trunk condiviso;
3. interazione fra i primi due fattori;
4. inizializzazione o traiettoria di ottimizzazione sfavorevole;
5. budget di training insufficiente.

## 🧪 Matrice preregistrata

| Braccio | Unico intervento | Contrasto |
|---|---|---|
| `baseline_extended` | loss originale, fino a 100k | budget |
| `symmetric_rates` | aggiunge `m_inf` e `log(tau_m)` | informazione nell’obiettivo |
| `pcgrad` | proietta solo gradienti condivisi in conflitto | interferenza |
| `symmetric_rates_pcgrad` | entrambi gli interventi | interazione 2×2 |
| `rate_pretrain` | 5k rate-only, poi loss simmetrica | inizializzazione/percorso |

Architettura, tuple, seed, minibatch, ottimizzatore e metriche restano allineati.
I rate di `m` sono etichette analitiche lecite del teacher e non modificano gli
input o i target endpoint.

## 📏 Regole interpretative

Un fattore è considerato materialmente supportato soltanto con un miglioramento
medio `≥20%`. Per attribuire causalmente il beneficio di PCGrad serve inoltre una
frazione di minibatch con coseno negativo `≥0,25` nel baseline iniziale.

Il notebook registra checkpoint crescenti come mini scaling law. Non apre dati
fresh, non promuove retrospettivamente un seed e non autorizza la Task 4. Dopo la
diagnosi verrà preregistrata una sola riparazione minima e, separatamente, una
nuova conferma disgiunta.
