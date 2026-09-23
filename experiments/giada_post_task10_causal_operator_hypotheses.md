# GIADA — ipotesi dopo la Task 10 originale

Stato: proposte aperte, non preregistrazione e non nuovi risultati. Obiettivo: update accurato dei gate con interfaccia esterna a 1 ms, senza voltaggio futuro teacher in inferenza. Distinguere accuratezza, stabilità e velocità misurata.

## Prima distinzione: canale guidato e membrana autonoma

Un canale isolato non può indovinare un arbitrario comando di clamp futuro non fornito. In quel dominio, il path imposto è un ingresso legittimo. Nel neurone autonomo, invece, il voltaggio è generato dallo stato accoppiato: va predetto insieme ai gate o mediante un sottosolver causale. La Task 11 path-aware deve rispettare questa distinzione; non autorizza un salto implicito al neurone multicompartimentale completo.

## H0 — lo stato/ingresso locale è incompleto

V,m,h non includono necessariamente correnti degli altri canali, stato sinaptico, coupling assiale e tempi degli eventi. Verificare l'interfaccia nel minimo sistema accoppiato, inizialmente CaHVA+pas già calibrato. Contrasto: stesso budget e stessi esempi, interfaccia ridotta contro stato rilevante completo e ingressi temporalizzati. Un eventuale miglioramento deve essere misurato su development indipendente e più seed; un fallimento di training non prova insufficienza informativa. Non inserire rilasci futuri teacher: eventuale casualità va generata dal front-end nel corretto ordine causale.

## H1 — pochi stadi predittore/correttore accoppiati

Stimare V intermedio, aggiornare i gate con formula/LUT, correggere V usando le correnti aggiornate. Confrontare 1/2/4/8 stadi e un riferimento denso, senza cambiare contemporaneamente la rappresentazione appresa. Il successo sarebbe accuratezza sufficiente con meno lavoro del riferimento denso. Il risultato legacy7c offre un riferimento numerico CaHVA+pas calibrato, non una prova di accuratezza a 1 ms.

Motivazione esterna: Börgers e Nectow (2013), https://nectowlab.org/wp-content/uploads/2019/10/Borgers_SIAM_2013.pdf, confrontano metodi ETD e propongono exponential midpoint. Nei loro sistemi la stabilità a passi grandi non implica ricostruzione precisa dello spike. Non trasferire garanzie, soglie o speedup al nostro teacher.

## H2 — prevedere un path compatto, invece di congelare V

Da stato iniziale e ingressi causali stimare pochi punti o coefficienti del voltaggio intra-ms. Alimentare lo stesso aggiornamento dei gate con path oracle e path predetto. Questo distingue limite della rappresentazione del path da errore del predittore. I target intermedi teacher sono supervisione ammessa durante training, mai input futuro durante inferenza. Misurare m/h/m²h, errore V e timing, non solo RMSE medio del path.

## H3 — apprendere l'effetto integrato direttamente

Per un path assegnato, dx/dt=a(t)(x_inf(t)-x) ammette x1=A*x0+B, con A=exp(-integrale a), B=integrale del termine forzante pesato, 0<=A<=1 e 0<=B<=1-A. Una parametrizzazione equivalente è x1=A*x0+(1-A)*q con q in [0,1]. Due coefficienti per gate, quattro per m/h, possono riassumere l'effetto del path sullo stato finale senza ricostruirne tutti i campioni.

Separare due prove: compressione con coefficienti oracle e learnability dei coefficienti da ingressi leciti. Per il sistema accoppiato il path dipende anche dai gate iniziali: A e q possono quindi dipendere da tutto lo stato rilevante. Non assumere una mappa affine globale indipendente da x0. Quando A è vicino a 1, q è poco identificabile; valutare A/B e lo stato finale, non solo i coefficienti grezzi. Questo aggiorna i gate, non risolve automaticamente V1, la carica integrata o il timing dello spike.

## H4 — concentrare lavoro negli intervalli difficili

Un indicatore causale (derivate iniziali, eventi già noti o discrepanza fra un passo e due mezzi passi) potrebbe selezionare più sottopassi solo dove necessario. Comparare costo/accuratezza con il numero fisso di passi. Vietato usare un'etichetta futura teacher per decidere online. Branching e maschere su GPU possono annullare il risparmio: misurare wall-time a batch appaiati, inclusi overhead e warmup. Nel sistema con coupling assiale serve ulteriore controllo di sincronizzazione.

## Priorità e metodo

H0 è il controllo dell'interfaccia. H1 è il riferimento affidabile; H3 è l'ipotesi appresa più interessante per comprimere il lavoro; H2 localizza quanta informazione temporale esplicita serve. H4 è un'ottimizzazione del metodo già accurato, non una sostituzione del controllo di validità.

Mantenere formule economiche di corrente, usare LUT/formula per isolare prima il coupling, e sostituire selettivamente parti costose solo dopo profiling. Playground atomici, contrasti appaiati, probe intermedi e checkpoint a budget crescenti: niente screening indiscriminato. Soglie, supporto, scelte e budget del prossimo esperimento restano da preregistrare; nessuna promozione da questi risultati già aperti a conferma indipendente.
