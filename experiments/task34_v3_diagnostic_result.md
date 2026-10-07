# Task 34 v3 — probe privilegiati e attribuzione corretta

La [v3 Kaggle](results/task34_v3_final_report.json) è stata eseguita a `e14e2ce` sui 16 casi **già aperti** della Task 33. È un audit, non una nuova conferma indipendente. I sei checkpoint sono rimasti congelati; il baseline Task 33 è riprodotto con differenza zero, il braccio oracle v2 è riprodotto con differenza zero e l'identità della scomposizione delle correnti ha residuo massimo 5,42×10⁻²⁰ mA/cm². Nessun oracle ha partecipato a training o selezione.

Il braccio a fase corretta con **stati teacher privilegiati** dà RMSE V 0,000339 mV, vicino al floor formula 0,000366 mV. Tutti i sei baseline modello restano a 0,000885–0,002237 mV. Il rapporto preregistrato 2× a favore degli stati teacher è soddisfatto per tutti e sei; imporre V teacher ai gate del modello non riduce di 2× il loro RMSE. Nel dominio analizzato, dunque, il piccolo errore di V non è il principale amplificatore dell'errore di stato.

Il nuovo contrasto appaiato cambia la lettura dei probe grezzi:

- Formula–stato nativo sotto V teacher: RMSE aggregato dei gate **0,003543**.
- Modello–formula sotto lo stesso V teacher: **0,000104–0,000242** sui sei checkpoint, circa 15–34 volte più piccolo.
- Per Nap_Et2, la corrente formula–nativo ha RMSE **5,79×10⁻⁶ mA/cm²**; il residuo modello–formula è **1,74–3,49×10⁻⁷ mA/cm²**.
- Per Ca_HVA, formula–nativo è **2,98×10⁻⁶ mA/cm²**; modello–formula è **1,02–4,62×10⁻⁷ mA/cm²**.

Nap_Et2 e Ca_HVA restano i principali contributi del *residuo specifico del modello* fra gli undici canali, ma l'ordine di grandezza dei precedenti errori contro gli stati nativi è soprattutto una discrepanza formula–nativo. La fase di campionamento degli stati è una spiegazione importante, confermata per il solver del voltaggio dall'[audit a un passo](../scripts/audit_task34_teacher_state_phase.py); non si attribuisce senza ulteriori test ogni differenza per canale esclusivamente a quella fase.

**Decisione:** Task 34 è diagnosticamente completa nel perimetro riusato. Non c'è base per una riprogettazione globale dei gate o per scegliere un checkpoint usando oracle. Task 35 richiede un contratto separato per la composizione completa e nuovi dati se vuole rivendicare generalizzazione; il voltage updater appreso della matrice originale resta non istanziato.
