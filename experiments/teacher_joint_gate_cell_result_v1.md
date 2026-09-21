# GIADA Task 3 — risultato

La composizione con due trunk indipendenti è valida, ma il gate preregistrato per la rappresentazione condivisa non è superato. `independent` ottiene score development `0,0695`; `shared_matched` `0,2650` e `shared_compact` `0,4169`.

Sul fresh, `independent` resta entro il gate ingegneristico di `m` (`0,002277`) e risolve `h` (`9,83e-6`) e `m²h` (`9,33e-4`). `shared_matched` fallisce di poco `m` (`0,002791`) e resta fuori dal margine development del 10%, pur mantenendo zero violazioni.

Il risultato non supporta una spiegazione di capacità: `shared_matched` ha 694 parametri contro 676, migliora ancora a 50k e mostra forte dispersione fra seed. Il seed 17 condiviso supera il seed 17 indipendente su `m`; i seed 29/43 peggiorano. La rappresentazione condivisa è quindi espressiva, ma il protocollo attuale non la ottimizza in modo affidabile, compatibilmente con interferenza fra obiettivi o traiettorie di ottimizzazione.

Task 4 non è autorizzata. Il prossimo passo è una Task 3b piccola e causale che distingua gradient conflict, inizializzazione/pretraining e budget, senza aprire un nuovo fresh per selezionare retrospettivamente.
