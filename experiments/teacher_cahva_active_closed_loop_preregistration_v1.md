# 🔁 GIADA Task 7b — Ca_HVA attivo nel circuito chiuso

La Task 7 ha calibrato il solver, ma nei suoi protocolli la corrente Ca_HVA era
quasi nulla. Questo esperimento cambia **solo il forcing somatico** e aggiunge
un controllo `gbar=0`: domanda se la LUT e il modello congelato restano accurati
quando il canale contribuisce misurabilmente al voltaggio.

🧪 Ventiquattro episodi di 20 ms: due voltaggi iniziali, quattro schedule di
iniezione (0,015/0,025/0,04 nA e coppia 0,025 nA), tre moltiplicatori di
conduttanza (0×, 1× canonico, 4× stress). Il teacher è ancora un singolo
compartimento NEURON `Ca_HVA+pas` con `E_Ca` fisso, non il neurone Hay completo.

🔒 La formula esatta deve prima riprodurre NEURON entro gli stessi limiti della
Task 7. Almeno quattro delle otto coppie a 1× devono inoltre mostrare sia
`max |ica| >= 1e-5 mA/cm²` sia `max |V_1x - V_0x| >= 0,05 mV`.
Tutte le dosi vengono riportate, anche se il gate di esposizione fallisce;
non scegliamo a posteriori lo stimolo che favorisce un candidato.

📏 I bracci formula, LUT-513 e physical-τ (seed 17/29/43) partono dal medesimo
stato teacher, poi iterano ciascuno il proprio voltaggio. Nessun retraining,
nessun uso del voltaggio futuro del teacher e nessuna scelta del seed. NEURON
e CUDA girano in processi distinti per confinare eventuali abort nativi.

⚠️ Anche un esito positivo autorizzerebbe soltanto un'affermazione sul
microcanary Ca_HVA a un compartimento. Non dimostrerebbe la validità nel
neurone multicompartimentale, con calcio dinamico o con updater V appreso.
