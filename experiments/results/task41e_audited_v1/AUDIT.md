# GIADA — Task 41e: audit e ipotesi residue

Data: 22 ottobre 2026. Audit post hoc; soglie invariate. Il risultato originale resta **TRANSFER_DIAGNOSIS_NO_GO**, candidato nominato Q0. Nessun nuovo esperimento avviato.

## Esito e recupero

Training completato a 60.000 aggiornamenti, exit 0. Tutti gli 11 file Torch recuperati hanno tensori finiti. Backup finale completo sul PC: 206 membri, dimensioni/CRC verificati contro l'inventario finale remoto. Lo ZIP ricomposto contiene gli stessi file finali ma ha SHA diverso dal contenitore remoto; entrambe le provenienze sono conservate. Nessun retraining o riavvio del worker per il recupero.

## Confronti allineati

| Modello | Risposte autonome 41c | Stress 41c | Vecchi stress 41 | Nuovo development 41e |
|---|---:|---:|---:|---:|
| P110 | 334/336 | 31/36 | 36/36 | 34/36 |
| C01 della 41d | 335/336 | 32/36 | 36/36 | 36/36 |
| Composizione della 41d | 334/336 | 32/36 | 34/36 | 34/36 |
| Q0, nominata 41e | 334/336 | 33/36 | 36/36 | 35/36 |
| H10, diagnostica | 335/336 | 33/36 | 36/36 | 35/36 |

Il development 41e non è quello della 41d: evitare confronti diretti fra conteggi su protocolli diversi. H10 non viene promossa usando test ormai aperti. Tutti i nuovi bracci mantengono le regressioni storiche di traiettorie/risposte e 72/72 recurrence; A01 e H01 introducono però specifici fallimenti di stress, registrati per identità nel JSON.

## Cosa resta e cosa abbiamo scoperto

- **Coda Ca_HVA:** Q0 sbaglia l'offset di 0,7–0,8 ms nello stesso protocollo stress sui tre seed, contro 0,3 ms ammessi. L'errore medio di corrente è già molto piccolo. H10 conserva tre fallimenti simili. Un caso development ha offset 0,4 ms. La 41e correggeva solo l'equilibrio dei gate m, non i coefficienti di rilassamento né h: questa limitazione va isolata sperimentalmente.
- **Risposte Nap minuscole:** Q0 ha NRMSE 0,014420 e 0,010064 su soglia 0,01. H10 risolve la prima, non la seconda. Il floor canonico FP32/NumPy sugli stessi due protocolli è già 0,00765–0,00801: non si può trattare tutta la differenza come errore della rete.
- **Riferimento dei probe:** suite ufficiale FP32, probe FP64. Le predizioni FP32 dei probe passano contro FP64 ma tornano a fallire, con gli stessi conteggi, se confrontate con il riferimento ufficiale. Le simulazioni FP64 dei controlli passano questi sei confronti anche contro il riferimento originale. È evidenza di sensibilità numerica, non autorizzazione a cambiare il benchmark o cancellare il NO-GO.
- **Ranking:** la selezione usava `severity > 1`, il valutatore la tolleranza numerica canonica già esistente. `0.30000000000000004 ms` diventava un falso fallimento nel ranking. L'analisi offline coerente cambia 5 checkpoint su 36; H10/H01/J11 avrebbero 36/36 sul development salvato, ma non sono stati rivalutati/promossi sui test. Q0 non cambia. 25/36 selezioni originali restano allo step zero.
- **Ottimizzazione:** il bilanciamento dinamico riduce la deriva delle norme, ma nessuno dei 15 contrasti soddisfa il criterio registrato. A fine training la loss comune di risposta è peggiore dell'iniziale in tutti i membri; il risultato non dimostra un beneficio generale del feedback.

## Velocità: conservare il risultato e il suo perimetro

Il circa **8×** storico riguarda il nucleo residente isolato: 80 passi dentro un unico CUDA graph e V/Ca imposti. La 41e misura un'altra esecuzione: graph per passo, ciclo Python e copie per passo, tre seed insieme, calcio autonomo, aggiornamento del voltaggio nello scope integrato. Entrambe usano CUDA graph. Non attribuire la differenza alla sola astrazione o al solo accoppiamento.

Un risultato netto della 41e è invece la fusione equivalente: tre chiamate al padre diventano una, senza differenze nelle uscite; il tempo integrato cala del 32–53% a seconda del batch. Questo non è ancora un'accelerazione qualificata del neurone completo.

## Proposta 41f — un notebook, cause distinguibili

1. **Preflight congelato:** riprodurre gli errori originali; unificare il predicato delle soglie senza cambiarle; separare precisione del candidato/riferimento e batching. Nessuna correzione retroattiva del verdetto 41e.
2. **Attribuzione atomica:** oracoli diagnostici di equilibrio e rilassamento m/h, FP32/mixed/FP64, clamp V/Ca; registrare crossing, pendenze, errori con segno e floor. Gli oracoli non sono candidati.
3. **Riparazione appresa, solo se necessaria:** matrice mirata di sei bracci: tre parametrizzazioni (u-only a risoluzione attuale, u-only con più capacità, u+z con capacità confrontabile) × supporto FIT ampio/arricchito. Stessi 18 stati, stesso livello di astrazione, solver e readout invariati. Mini scaling lungo le stesse traiettorie; nuovi protocolli e seed. La scelta esatta dei gate e del budget deve essere preregistrata, con probe FIT indipendenti e senza selezione sui casi test aperti.
4. **Trasferimento della velocità:** stessi pesi e carichi, graph intero/per passo × fisica imposta/accoppiata; formule e rete ottimizzate simmetricamente. Equivalenza prima del benchmark; tempi CUDA e wall-time separati.

Precedenti interrogati nel DB: 39j (u/z, learnability, selezione), 17d/17d-b (riferimento e precisione), 41d (loss saturate, gradienti, composizione). Sono analogie utili, non prove della stessa causa.

Fonti complete: `synthesis.json`, `residual_probe.json`, archivio finale e ricevuta di recupero. La registrazione conserva tutti i 63 report JSON originali, non soltanto questa sintesi. Backup DB locale; Airtable e Drive non modificati.
