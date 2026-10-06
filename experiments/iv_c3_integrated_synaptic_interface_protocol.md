# GIADA IV-C3 — interfaccia sinaptica integrata

Stato: preregistrato prima del run. IV-C1 e IV-C2 sono prerequisiti verificati per hash; Task33 resta chiusa finché questo gate non supera.

Quattro sinapsi canoniche EMS (due eccitatorie, due inibitorie) condividono un compartimento con voltaggio imposto. I bracci appaiati variano schedule di impulsi intercalati, burst coincidenti e recupero dopo depressione; ogni schedule è ripetuta su quattro seed e tre voltaggi. Una schedule/seed separata serve solo come calibrazione tecnica iniziale. Nessun esito confermativo può cambiare soglie, fase o scelta del modello.

Lo shadow processa gli eventi **online**, nel loro ordine temporale, con lo stato plastico e il proprio stream Random123. Non riceve rilasci futuri osservati dal teacher. Prevede incrementi e decadimento degli stati recettoriali A/B, conduttanze e correnti per recettore; il confronto nativo include anche stato plastico, sequenza RNG, conteggio dei rilasci, carica integrata e tensione imposta. A metà episodio si salva lo stato completo delle quattro sinapsi, poi si ripristina il teacher e si riproduce il suffisso con i soli eventi programmati futuri.

Controlli: tutti i pesi nulli devono annullare corrente ed estrazioni RNG; gli impulsi programmati non possono essere scambiati per rilasci; inversione del segno della corrente AMPA/GABA e blocco NMDA dipendente dal voltaggio devono essere coerenti. Le soglie esatte e le condizioni di promozione sono nel [contratto preregistrato](iv_c3_integrated_synaptic_interface_preregistration.json). Un errore di ambiente o API produce un report di fallimento tecnico, non un no-go biologico.
