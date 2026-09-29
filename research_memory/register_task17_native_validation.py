"""Record the native Task17 repair and local matrix; SQLite only, idempotent."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    path = ROOT.parent / "experiments/results/task17a_native_validation/final_report.json"
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != "99d9bd64d3b7b657242899195c2266d96674b9f62ddd46e567d762346c6b5666":
        raise RuntimeError("Local Task17 evidence hash changed")
    report = json.loads(payload)
    assert report["valid"] and report["episode_count"] == 27
    assert not report["gate_c_authorized"]
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-task17a-native-repair-{suffix}-v1", **fields
        })["record_id"]

    block = put("blocks", "matrix", **{
        "Nome": "Task17a — matrice appaiata della validazione nativa locale",
        "Descrizione": "27 triplette, tre seed, tre protocolli, tre moltiplicatori gbar. Non 81 repliche indipendenti.",
        "Seed": "170017,170029,170043",
        "Regola di appaiamento": "Stesso equilibrio, seed Random123, stimoli e gbar fra nativo/formula/LUT. Snapshot rigenerato dopo ogni cambio di struttura; bracci in blocchi contigui.",
        "Protocollo": ["rec72823a8118d77e"],
    })
    artifact = put("artifacts", "report", **{
        "Nome": "Task17a — report integrale validazione Linux locale",
        "Tipo": "Report", "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
        "SHA-256": digest, "Dimensione byte": len(payload), "Versione": "2026-09-29-v1",
        "Descrizione": "Report generato, non ricostruito. Provenance: base HEAD 114e4f7, working tree modificato, SHA256 dei tre sorgenti eseguiti nel JSON. Metriche dettagliate e contrasti nella medesima cartella versionata.",
    })
    put("runs", "kaggle-missing-report", **{
        "Nome": "Task17a 114e4f7 — report Kaggle assente",
        "Stato": "Fallita", "Blocco": [block],
        "Descrizione": "Utente: nessun final_report/failure_report; la cella solleva FileNotFoundError. Exit nativo Kaggle non disponibile. Riproduzione locale separata della vecchia versione: SIGSEGV in SaveState.restore al ritorno nativo dopo formula.",
        "Hardware e ambiente": "Kaggle, revisione 114e4f7; traceback del solo lettore del report.",
        "Validità tecnica": "Non interpretabile scientificamente; non registrare come fallimento della LUT.",
    })
    run = put("runs", "local-full-matrix", **{
        "Nome": "Task17a — matrice completa Linux dopo fix SaveState",
        "Stato": "Completata", "Blocco": [block], "Artefatti prodotti": [artifact],
        "Seed": "170017,170029,170043",
        "Descrizione": "Seconda esecuzione locale completa, inclusa nella suite di integrazione. La prima è una replica tecnica sugli stessi seed: non conta come nuova evidenza indipendente. Non è un run Kaggle.",
        "Hardware e ambiente": "Ubuntu WSL, Python 3.12.3, pacchetto NEURON 8.2.7, CPU, teacher 074c4666300a8ad246601dab179a97a6942f0f29. Nessun benchmark GPU/speedup.",
        "Configurazione effettiva": "Matrice preregistrata invariata, 81 episodi + 4 preflight, 642 segmenti. Emendamento: snapshot fresco per generazione del meccanismo, supervisore separato; nessuna selezione o soglia modificata.",
        "Validità tecnica": "Exit 0; report valid=true; formula e replay/ritorno nativo validi. La suite passa 14 test, incluso SIGSEGV figlio. Decisione scientifica CAUSAL_MICROCANARY_NO_GO per massimo errore gate LUT >0.01.",
    })
    observation_ids = []
    for short, name, formula, unit, key, limit, arm in (
        ("formula-voltage", "Peggior RMSE V copia-formula", "max_episode,site sqrt(mean_t((V_arm-V_native)^2))", "mV", "worst_formula_voltage_rmse_mv", 0.05, "formula"),
        ("formula-gate", "Massimo errore gate copia-formula", "max_episode,site,t,gate abs(x_arm-x_native)", "adimensionale", "worst_formula_gate_error", 0.002, "formula"),
        ("lut-voltage", "Peggior RMSE V LUT", "max_episode,site sqrt(mean_t((V_arm-V_native)^2))", "mV", "worst_lut_voltage_rmse_mv", 2.0, "lut"),
        ("lut-gate", "Massimo errore gate LUT", "max_episode,site,t,gate abs(x_arm-x_native)", "adimensionale", "worst_lut_gate_error", 0.01, "lut"),
    ):
        metric = put("metrics", short, **{
            "Nome": name, "Famiglia": "Regressione", "Formula": formula,
            "Unità": unit, "Direzione": "Minimizzare",
        })
        evaluation = put("evaluations", short, **{
            "Nome": "Task17a — " + name, "Versione": "v1",
            "Descrizione": f"Soglia preregistrata <= {limit} {unit}; non modificata dopo il run.",
            "Target": f"Braccio {arm} rispetto al nativo; sorgente {key}",
            "Popolazione e regioni": "27 condizioni protocollo-seed-gbar, 4 siti 0/387/460/469; tre seed indipendenti, campioni temporalmente correlati.",
            "Orizzonte e finestre": "40 ms, campionamento 0.025 ms, 1601 punti per sito/episodio.",
            "Aggregazione e pesi": formula,
            "Ruolo": "Controllo" if arm == "formula" else "Primaria",
            "Metrica": [metric], "Protocollo": ["rec72823a8118d77e"],
        })
        observation_ids.append(put("observations", short, **{
            "Nome": "Task17a locale — " + name, "Valore": report[key],
            "Descrizione": f"Misura autentica da final_report.json/{key}. Soglia {limit}; nessun intervallo di confidenza stimato.",
            "Numerosità": 27, "Strato o sottogruppo": f"{arm}; massimo su 27 condizioni e quattro siti",
            "Run": [run], "Specifica di valutazione": [evaluation], "Artefatti dettagliati": [artifact],
        }))
    put("findings", "scoped-outcome", **{
        "Nome": "Task17a locale: driver riparato, formula esatta, LUT NO-GO sui gate",
        "Risultato": "Formula: errori V e gate massimi zero. LUT: peggior RMSE V 0.5933959189530174 mV <=2, massimo errore m/h 0.01835138564840766 >0.01. Contrasto gbar valido su 11 effetti identificabili. Gate C non autorizzato.",
        "Descrizione": "Distinguere crash del vecchio driver, corretto, da esito scientifico negativo del criterio gate. Nessuna soglia o architettura adattata ai risultati osservati.",
        "Incertezza": "Tre seed e protocolli fissi; ripetere gli stessi seed non aumenta evidenza indipendente.",
        "Limitazioni": "Validazione CPU Linux locale, non ancora replica Kaggle; non prova copertura completa di regimi, eventi/spike o speedup. Il candidato non è promosso.",
        "Esito": "Misto", "Osservazioni": observation_ids,
        "Esperimenti": ["recbe364a1facd791"],
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    verification = mirror.verify()
    assert verification["valid"]
    return {"valid": True, "run": run, "observations": len(observation_ids),
            "airtable_accessed": False, "source_sha256": digest}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
