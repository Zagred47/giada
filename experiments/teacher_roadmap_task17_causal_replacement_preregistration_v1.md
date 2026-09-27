# GIADA — Task 17a: sostituzione causale Ca_HVA, microcanary preregistrato

Stato: **preregistrazione, nessun risultato osservato**. Task 17a è il primo
gate della Task 17 originale, non una promozione automatica della LUT né una
nuova numerazione della roadmap. Il mirror SQLite locale è l'unica memoria
operativa; Airtable non viene consultato.

## Intervento e domanda

Con il teacher canonico a 642 segmenti e CVode, sostituire **in tutti i
compartimenti che possiedono Ca_HVA** il meccanismo nativo con:

1. una copia NMODL della stessa formula, con identico `USEION ca READ eca WRITE
   ica`, `cnexp`, stato m/h e conduttanza; controllo di interfaccia;
2. la medesima copia, ma con `TABLE` lineare a 513 nodi delle quattro funzioni
   m∞, h∞, τm, τh sul dominio [-135, 75] mV, candidato selezionato in 15c e
   confermato solo come shadow in 16.

Le due copie sono compilate fuori dal repository del teacher. Al confine
iniziale di ogni episodio, si ripristina lo stesso SaveState nativo, stesso
stato Random123 e stesso programma di input; si trasferiscono gbar e m/h e si
reinizializza CVode. La nuova corrente ionica è scritta dal meccanismo NMODL
nel sistema accoppiato. Nessun V(t+Δt), stato o corrente del teacher entra nel
braccio sostitutivo. Il rate TABLE va confrontato numericamente con la tabella
congelata della 16 **prima** del rollout. Se non corrisponde, stop.

## Matrice e indipendenza

- Protocolli fissati prima dei risultati: quiete; cluster NMDA tuft n=10,
  due burst; cluster calcio hot zone n=12, tre burst. Durata 40 ms.
- Seed nuovi e indipendenti da selezione/test precedente: 170017, 170029,
  170043. Sono repliche per traiettoria, non per campione o sito.
- Perturbazione fisiologica appaiata di gbar Ca_HVA: ×0,5, ×1, ×1,5. Si
  mantiene invariata la morfologia, i pesi NetCon, ECa e gli altri meccanismi.
- Bracci nativo / copia-formula / LUT, per ogni protocollo × seed × gbar.
  Totale 27 triplette, 81 episodi; il preflight in quiete viene ripetuto prima.
- Campionamento ogni 0,025 ms ai siti preregistrati 0, 387, 460, 469.
  Si conservano i tracciati completi ma si stampa solo un progresso sintetico.

## Soglie e falsificazioni

1. Replay nativo ripetuto: errore massimo V ≤1e-5 mV. Se fallisce, nessuna
   attribuzione alla LUT.
2. Copia-formula: RMSE V ≤0,05 mV e massimo errore m/h ≤0,002 in **ogni**
   sito e episodio. Se fallisce, l'interfaccia/solver è sospetta e la LUT non
   viene valutata come effetto causale.
3. Tabella NMODL vs tabella congelata: differenza assoluta ≤2e-5 sulle
   componenti di rate ai nodi di preflight. Il controllo formula resta decisivo
   anche se questo passa.
4. LUT: RMSE V ≤2 mV e massimo errore m/h ≤0,01 in ogni sito/episodio.
5. Effetto gbar: per siti con effetto nativo RMSE ≥0,05 mV, errore della
   differenza tra ×1,5 e ×0,5 ≤20% della grandezza dell'effetto, per ogni
   protocollo/seed/sito identificabile. Se nessun effetto è identificabile,
   test non informativo.
6. Stato non finito, occupanza m/h fuori [0,1], mutazione dei pesi sinaptici,
   mismatch di struttura o del controllo formula: arresto immediato.

Le soglie sono gate di *microcanary* e non autorizzano da sole Gate C. Anche
un eventuale `CAUSAL_MICROCANARY_PASS` lascia `gate_c_authorized=false` finché
non saranno verificati test sigillati più ampi, eventi/spike e tenuta per
regime/compartimento. La selezione Task 15c resta congelata; non si ritoccano
le soglie dopo avere visto i risultati. La Task 16 non è evidenza di rollout.

## Ipotesi distinguibili

- Formula fallisce: errore di semantica del meccanismo, stato iniziale,
  SaveState o integrazione, non fallimento della LUT.
- Formula passa e LUT fallisce: errore di approssimazione delle rate, possibile
  amplificazione nel feedback voltaggio–canale o discontinuità della tabella.
- V assoluto passa ma contrasto gbar fallisce: perdita della sensibilità
  biologica anche in presenza di RMSE apparentemente buono.
- Entrambi passano: evidenza positiva **limitata** per sostituzione causale
  di Ca_HVA in questa matrice; non dimostra ancora prestazione dell'intero
  neurone su tutti i regimi o accelerazione GPU.
