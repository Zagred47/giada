"""Register the local Task 17b diagnostic matrix in SQLite only."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


EXPECTED_REPORT_SHA256 = "e8bc4f627f40d86ce00f09a28f31d814cebc41767a7515f22c5cda3b4b52386f"
EXPECTED_METRICS_SHA256 = "33e41a4243455b12b83d834740f476d22c49009121931c5c0cd130124aa4e6f7"


def main():
    result_dir = ROOT.parent / "experiments/results/task17b_rate_attribution"
    report_path = result_dir / "final_report.json"
    metrics_path = result_dir / "paired_metrics.json"
    report_bytes, metrics_bytes = report_path.read_bytes(), metrics_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != EXPECTED_REPORT_SHA256:
        raise RuntimeError("Task17b report hash changed")
    if hashlib.sha256(metrics_bytes).hexdigest() != EXPECTED_METRICS_SHA256:
        raise RuntimeError("Task17b paired metrics hash changed")
    report, rows = json.loads(report_bytes), json.loads(metrics_bytes)
    assert report["valid"] and report["decision"] == "DIAGNOSTIC_ONLY"
    assert not report["gate_c_authorized"] and len(rows) == 9
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17b-rate-attribution-{suffix}-v1",
            **fields,
        })["record_id"]

    claim_m = put("claims", "m-pair", **{
        "Nome": "Task17b: i rate m dominano la deviazione LUT Ca_HVA",
        "Enunciato": "Nel protocollo NMDA e nei siti registrati, sostituire mInf e mTau insieme riproduce quasi l'errore della LUT completa; le sostituzioni h isolate no.",
        "Tipo": "Interpretazione causale", "Stato": "Supportata nel dominio",
        "Condizioni di falsificazione": "Un nuovo regime comparabile in cui la LUT h-only riproduce la deviazione della full LUT, oppure m-pair diverge sistematicamente dalla full LUT.",
        "Limiti": "Solo NMDA, 3 seed nella matrice emendata, tre intensità gbar, quattro siti. Non inferire irrilevanza universale di h.",
    })
    claim_uniform = put("claims", "uniform-error", **{
        "Nome": "Task17b: errore LUT uniforme fra seed e regime",
        "Enunciato": "L'errore della LUT è pressoché uniforme al variare di seed, sito e gbar.",
        "Tipo": "Ipotesi", "Stato": "Contraddetta nel dominio",
        "Condizioni di falsificazione": "Errore gate >0,01 concentrato in poche combinazioni, con altre vicine a zero.",
        "Limiti": "Non identifica da sola la geometria precisa della soglia dinamica.",
    })
    experiment = put("experiments", "matrix", **{
        "Nome": "GIADA Task17b — matrice causale dei rate Ca_HVA",
        "Descrizione": "Follow-up diagnostico della Task17a. Bracci single-rate, coppia m, formula, full LUT e native; emendamento dichiarato dopo primo screening, nuovi seed non chiamati sealed.",
        "Obiettivo informativo": "Separare mInf, mTau, hInf, hTau e interazione mInf+mTau sotto feedback nativo senza cambiare soglie o scegliere un candidato.",
        "Tipo": "Forensic", "Stato": "Concluso", "Ipotesi": [claim_m, claim_uniform],
    })
    protocol_text = (ROOT.parent / "experiments/task17b_rate_attribution_preregistration.md").read_bytes()
    protocol = put("protocols", "amended", **{
        "Nome": "Task17b — matrice emendata, diagnostica", "Versione": "v1-amended",
        "Modalità": "Sistema completo",
        "Procedura": "8 bracci × 3 seed × 3 gbar nel teacher nativo 642 segmenti, protocollo NMDA, 40 ms, campione 0,025 ms, SaveState fresco dopo cambio struttura.",
        "Criteri di successo": "Formula clone entro 0,05 mV RMSE e 0,002 gate; contrasti m-pair/full LUT descrittivi. Nessun Gate C.",
        "Regole di arresto": "Stop se verifica Task15c, NMODL, formula clone o replay falliscono.",
        "Hash preregistrazione": hashlib.sha256(protocol_text).hexdigest(),
        "Descrizione": "Emendamento esplicito dopo screening di sviluppo 170059/170071. Seed nuovi della matrice: 170083/170097; seed fallito 17a: 170029.",
        "Esperimento": [experiment],
    })
    block = put("blocks", "paired", **{
        "Nome": "Task17b — matrice NMDA appaiata",
        "Descrizione": "9 condizioni seed × gbar; 8 bracci per condizione, 72 traiettorie. Non 72 repliche indipendenti.",
        "Seed": "170029,170083,170097",
        "Regola di appaiamento": "Stesso equilibrio, protocollo, seed Random123 e gbar nei bracci; snapshot nuovo per generazione meccanismo.",
        "Protocollo": [protocol],
    })
    artifacts = []
    for suffix, path, data in (("report", report_path, report_bytes),
                               ("metrics", metrics_path, metrics_bytes)):
        artifacts.append(put("artifacts", suffix, **{
            "Nome": f"Task17b — {suffix} della matrice Linux locale",
            "Tipo": "Report", "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
            "SHA-256": hashlib.sha256(data).hexdigest(), "Dimensione byte": len(data),
            "Versione": "2026-09-30-v1",
            "Descrizione": "Output nativo locale; la matrice non è una validazione sealed né un run Kaggle.",
        }))
    run = put("runs", "local-amended", **{
        "Nome": "Task17b — matrice emendata nativa locale",
        "Stato": "Completata", "Blocco": [block], "Artefatti prodotti": artifacts,
        "Seed": "170029,170083,170097",
        "Descrizione": "72 traiettorie: 8 bracci, tre seed, tre gbar; 36 confronti sito-condizione per braccio. Primo screening con 170059/170071 solo sviluppo, non contato come conferma.",
        "Hardware e ambiente": "Ubuntu WSL, NEURON 8.2.7, CPU; teacher commit 074c4666300a8ad246601dab179a97a6942f0f29. Non è un benchmark prestazionale.",
        "Configurazione effettiva": "Protocollo Task17b emendato, nessuna soglia 17a modificata; clone formula esatto.",
        "Validità tecnica": "Exit 0, report valid=true, native-repeat e clone formula con errore zero, decision DIAGNOSTIC_ONLY, Gate C false.",
    })
    observations = []
    for suffix, name, arm, site, seed, multiplier, key in (
        ("known-full", "Gate max full LUT nel caso 17a", "lut", "460", 170029, 1.0, "m_max_error"),
        ("known-mpair", "Gate max m-pair nel caso 17a", "m_pair_lut", "460", 170029, 1.0, "m_max_error"),
        ("new-full", "Gate max full LUT in nuovo seed", "lut", "0", 170083, 1.5, "m_max_error"),
        ("new-mpair", "Gate max m-pair in nuovo seed", "m_pair_lut", "0", 170083, 1.5, "m_max_error"),
    ):
        match = next(row for row in rows if row["seed"] == seed and row["gbar_multiplier"] == multiplier)
        observations.append(put("observations", suffix, **{
            "Nome": name, "Valore": match["metrics"][arm][site][key],
            "Descrizione": f"Misurato dal report appaiato, braccio {arm}, sito {site}, seed {seed}, gbar {multiplier}.",
            "Numerosità": 1, "Strato o sottogruppo": f"{arm}; NMDA; sito {site}",
            "Run": [run], "Artefatti dettagliati": artifacts,
        }))
    put("findings", "scoped-outcome", **{
        "Nome": "Task17b: deviazione Ca_HVA attribuibile ai rate m nel dominio testato",
        "Risultato": "Seed noto sito 460 gbar1: full LUT m 0.018351, coppia m 0.018382. Nuovo seed 170083 soma gbar1.5: full LUT m 0.017199, coppia m 0.017211. Massimo scarto fra metriche gate full e coppia 3.024e-5 nei 36 confronti. Formula esatta; h-only piccolo.",
        "Descrizione": "La coppia m spiega quasi tutta la deviazione nel dominio misurato; singoli mInf/mTau parziali e interazione closed-loop. La LUT resta NO-GO per la soglia gate 17a.",
        "Incertezza": "3 seed emendati, dei quali uno già noto; 170059/170071 hanno informato l'emendamento e sono sviluppo. Nessuna inferenza di probabilità di fallimento.",
        "Limitazioni": "Solo protocollo NMDA e quattro siti; non risolve eventuale dipendenza dal solver, non misura speedup, non costituisce conferma Gate C.",
        "Esito": "Misto", "Osservazioni": observations, "Esperimenti": [experiment],
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    verification = mirror.verify()
    assert verification["valid"]
    return {"valid": True, "airtable_accessed": False, "experiment": experiment,
            "observations": len(observations), "report_sha256": EXPECTED_REPORT_SHA256}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
