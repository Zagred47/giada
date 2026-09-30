"""Record Task 17c local evidence and bounded interpretation in SQLite only."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


EXPECTED = {
    "final_report.json": "1e7746479f28d2a861fce6f5626617346e02be424eb9940cc0e43833814aa7ca",
    "paired_metrics.json": "8d0158d9c48db7c2d1f4b6cd9ed824e1f6c2395d77a73c669c9bc42ed7479643",
    "native_solver_convergence_ladder.json": "8f74430300bfb422d8a4623ae0ca443d23cc44b22c21f1aca6cac93e94d1ba92",
    "process_status.json": "02edf922061292707fed3f6e203f4f28647c82b0ae545181b6e9cd8360efc35c",
}


def main():
    result_dir = ROOT.parent / "experiments/results/task17c_local_precision_solver"
    for name, expected in EXPECTED.items():
        if hashlib.sha256((result_dir / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Task17c source evidence hash changed: {name}")
    report = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    rows = json.loads((result_dir / "paired_metrics.json").read_text(encoding="utf-8"))
    ladder = json.loads((result_dir / "native_solver_convergence_ladder.json").read_text(encoding="utf-8"))
    status = json.loads((result_dir / "process_status.json").read_text(encoding="utf-8"))
    assert report["valid"] and report["decision"] == "DIAGNOSTIC_ONLY"
    assert not report["gate_c_authorized"] and status["returncode"] == 0
    assert len(rows) == 6 and len(ladder) == 9 and report["teacher_segment_count"] == 642
    assert all(max(v["voltage_max_error_mv"] for v in sites.values()) == 0
               for sites in report["native_repeat_preflight"].values())
    assert all(report["summary"][solver]["formula"]["maximum_gate_error"] == 0
               for solver in ("default", "tight"))
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17c-precision-solver-{suffix}-v1",
            **fields,
        })["record_id"]

    interpolation = put("claims", "interpolation-spacing", **{
        "Nome": "Task17c: passo di griglia m come sorgente dominante dell'errore",
        "Enunciato": "Nei casi NMDA testati, aumentare i nodi dell'interpolazione dei rate m riduce fortemente l'errore closed-loop rispetto al nativo dello stesso solver.",
        "Tipo": "Interpretazione causale", "Stato": "Supportata nel dominio",
        "Condizioni di falsificazione": "Su regime comparabile e solver controllato, una griglia più fitta non riduce l'errore o un controllo con identica griglia ma diversa precisione lo riduce altrettanto.",
        "Limiti": "Due casi critici già noti e un controllo debole; nessun test sealed o benchmark costo computazionale.",
    })
    quantization = put("claims", "f32-dominant", **{
        "Nome": "Task17c: quantizzazione float32 dei valori LUT come causa dominante",
        "Enunciato": "A parità di 513 nodi, passare dai valori float32 a float64 elimina gran parte dell'errore della LUT.",
        "Tipo": "Ipotesi", "Stato": "Contraddetta nel dominio",
        "Condizioni di falsificazione": "La variante 513 float64 ha errore quasi uguale alla 513 float32, mentre la griglia più fitta riduce l'errore.",
        "Limiti": "Non implica che float32 sia sempre innocuo in altri canali o griglie.",
    })
    solver_claim = put("claims", "native-solver-sensitivity", **{
        "Nome": "Task17c: il teacher nativo è sensibile alle tolleranze CVode nei casi critici",
        "Enunciato": "Nei due casi critici, la traiettoria nativa cambia sensibilmente riducendo atol/rtol; la convergenza è ancora incerta per il caso seed 170083 gbar 1,5.",
        "Tipo": "Interpretazione causale", "Stato": "Supportata nel dominio",
        "Condizioni di falsificazione": "Ripetizione deterministica con gli stessi input che mostra traiettorie native invarianti lungo il ladder, oppure errore di replay/seed non controllato.",
        "Limiti": "Le diverse tolleranze cambiano la soluzione numerica; non si dimostra qui quale sia il limite convergente né si separano tutte le sorgenti di sensibilità dell'evento sinaptico.",
    })
    experiment = put("experiments", "matrix", **{
        "Nome": "GIADA Task17c — precisione m e tolleranza CVode",
        "Descrizione": "Matrice appaiata 7 bracci × 2 solver × 3 condizioni, con ladder native-only aggiunto dopo grande divergenza del nativo nella prima matrice.",
        "Obiettivo informativo": "Separare quantizzazione, distanza fra nodi e sensibilità numerica dell'oracolo nativo senza selezionare un candidato.",
        "Tipo": "Forensic", "Stato": "Concluso",
        "Ipotesi": [interpolation, quantization, solver_claim],
    })
    prereg = ROOT.parent / "experiments/task17c_m_precision_solver_preregistration.md"
    protocol = put("protocols", "amended", **{
        "Nome": "Task17c — matrice preregistrata più ladder emendato", "Versione": "v1-amended",
        "Modalità": "Sistema completo", "Esperimento": [experiment],
        "Procedura": "7 bracci × default/tight × 3 condizioni NMDA; ladder native-only stricter/ultra dopo osservazione della divergenza. Teacher 642 segmenti, 40 ms, 0,025 ms di campionamento.",
        "Criteri di successo": "Controlli replay nativo e formula validi; contrasti di precisione e solver descrittivi. Nessuna promozione Gate C.",
        "Regole di arresto": "Stop a hash Task15c, compilazione/probe NMODL, replay nativo, formula clone o stati non finiti non validi.",
        "Hash preregistrazione": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "Descrizione": "Ladder emendato dichiarato nel documento dopo la prima matrice locale; non usato per scelta del candidato.",
    })
    block = put("blocks", "paired", **{
        "Nome": "Task17c — condizioni note appaiate",
        "Descrizione": "Tre condizioni (due critiche note e controllo debole), 42 traiettorie di confronto più replay e ladder native-only. Non replicazioni indipendenti.",
        "Seed": "170029,170083",
        "Regola di appaiamento": "Stesso equilibrio, stimoli e Random123 tra bracci; ciascuna approssimazione rispetto al native con lo stesso CVode.",
        "Protocollo": [protocol],
    })
    artifacts = []
    for name, digest in EXPECTED.items():
        path = result_dir / name
        artifacts.append(put("artifacts", name.split(".")[0], **{
            "Nome": f"Task17c locale — {name}", "Tipo": "Report",
            "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
            "SHA-256": digest, "Dimensione byte": path.stat().st_size,
            "Versione": "2026-09-30-v1",
            "Descrizione": "Output autentico Linux locale. Il codice era nel working tree sopra HEAD 3bb2754; hash dei sorgenti eseguiti nel README di risultato.",
        }))
    run = put("runs", "local", **{
        "Nome": "Task17c — matrice locale e ladder nativo",
        "Stato": "Completata", "Blocco": [block], "Artefatti prodotti": artifacts,
        "Seed": "170029,170083",
        "Descrizione": "42 traiettorie appaiate più replay e ladder native-only. Nessun nuovo seed sealed; non è un benchmark prestazionale.",
        "Hardware e ambiente": "Ubuntu WSL, NEURON 8.2.7, CPU; teacher 074c4666300a8ad246601dab179a97a6942f0f29.",
        "Configurazione effettiva": "CVode default atol0.001/rtol0, tight 1e-5/1e-6, ladder 1e-6/1e-7 e 1e-7/1e-8; griglie 513/1025/2049.",
        "Validità tecnica": "Exit 0; report valid=true; probe NMODL, native-repeat e formula-clone validi a tutti i livelli applicabili.",
    })
    observations = []
    for solver, arm in (("default", "m_pair_513_f32"), ("default", "m_pair_513_f64"),
                        ("default", "m_pair_1025_f64"), ("default", "m_pair_2049_f64"),
                        ("tight", "m_pair_513_f32"), ("tight", "m_pair_2049_f64")):
        observations.append(put("observations", f"{solver}-{arm}", **{
            "Nome": f"Task17c massimo errore gate {solver} {arm}",
            "Valore": report["summary"][solver][arm]["maximum_gate_error"],
            "Descrizione": "Massimo su 3 condizioni × 4 siti; confronto col nativo allo stesso solver.",
            "Numerosità": 3, "Strato o sottogruppo": f"{solver}; {arm}",
            "Run": [run], "Artefatti dettagliati": artifacts,
        }))
    put("findings", "bounded-result", **{
        "Nome": "Task17c: griglia m domina f32, ma il nativo è solver-sensibile",
        "Risultato": "Massimo gate default: 513f32 0.018382, 513f64 0.018381, 1025f64 0.003657, 2049f64 0.001760. Tight: 513f32 0.010804, 2049f64 0.000673. Nativo default→tight: fino a 13.526 mV RMSE V; stricter→ultra resta fino a 0.1039 mV nel secondo caso critico.",
        "Descrizione": "La riduzione dell'errore con più nodi è un contrasto all'interno di ogni solver; il riferimento nativo del secondo caso non è ancora dimostrato convergente.",
        "Incertezza": "Due casi critici già osservati e un controllo; ladder aggiunto dopo prima matrice. Nessuna inferenza su nuovi seed o regimi.",
        "Limitazioni": "Non autorizza modifica del candidato congelato, Gate C, né speedup. Serve verifica numerica più ampia prima di claim biologico.",
        "Esito": "Misto", "Osservazioni": observations, "Esperimenti": [experiment],
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "experiment": experiment,
            "run": run, "observations": len(observations)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
