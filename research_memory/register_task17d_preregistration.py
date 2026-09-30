"""Register the Task17d questions and frozen protocol in the local mirror only."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    protocol_path = ROOT.parent / "experiments/task17d_native_solver_mechanism_preregistration.md"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17d-native-solver-{suffix}-v1",
            **fields,
        })["record_id"]

    calcium = put("claims", "calcium-scale", **{
        "Nome": "Task17d: precisione dello stato calcio governa la sensibilità nativa",
        "Enunciato": "Una scala CVode mirata allo STATE cai avvicina la traiettoria nativa al riferimento ultra più della scala mirata a v.",
        "Tipo": "Ipotesi", "Stato": "Aperta",
        "Condizioni di falsificazione": "A input identici, calcium_scaled non avvicina a ultra o voltage_scaled lo fa maggiormente nei casi critici.",
        "Limiti": "Tre condizioni note; la coincidenza con ultra non dimostra da sola una soluzione biologicamente convergente.",
    })
    threshold = put("claims", "threshold-amplification", **{
        "Nome": "Task17d: amplificazione vicino alla soglia dinamica",
        "Enunciato": "Piccole differenze precoci fra solver producono differenze maggiori nel timing dello spike o nella depolarizzazione dendritica tardiva dei casi critici rispetto al controllo debole.",
        "Tipo": "Ipotesi", "Stato": "Aperta",
        "Condizioni di falsificazione": "Le differenze sono uniformi fin dall'inizio e non crescono in associazione con eventi o stati attivi.",
        "Limiti": "Non equivale a dimostrare una biforcazione matematica o una soglia biologica universale.",
    })
    driver = put("claims", "driver-and-release", **{
        "Nome": "Task17d: driver e rilasci contribuiscono alla divergenza nativa",
        "Enunciato": "A tolleranze ultra identiche, cambiare la frequenza delle chiamate solve o avere rilasci effettivi diversi modifica la traiettoria.",
        "Tipo": "Ipotesi", "Stato": "Aperta",
        "Condizioni di falsificazione": "I rilasci risultano appaiati e i tracciati ultra a cadenze diverse coincidono entro il floor numerico.",
        "Limiti": "Il confronto di cadenza include l'effetto delle chiamate solve; non isola il solo campionamento osservazionale.",
    })
    experiment = put("experiments", "matrix", **{
        "Nome": "GIADA Task17d — attribuzione della sensibilità numerica nativa",
        "Descrizione": "Cinque politiche CVode su tre condizioni note, più controllo cadenza su due casi, 60 ms, rilasci e stati interni monitorati.",
        "Obiettivo informativo": "Distinguere precisione di cai, precisione di v, timing/plateau, driver e differenze dei rilasci senza selezione del surrogate.",
        "Tipo": "Forensic", "Stato": "Preregistrato",
        "Ipotesi": [calcium, threshold, driver],
    })
    protocol = put("protocols", "v1", **{
        "Nome": "Task17d — solver nativo e monitoraggio meccanicistico",
        "Versione": "v1", "Modalità": "Sistema completo", "Esperimento": [experiment],
        "Procedura": "Teacher 642 segmenti; default/tight/ultra/calcium_scaled/voltage_scaled × tre condizioni, ultra_dense × due; 60 ms; replay e rilascio verificati.",
        "Criteri di successo": "Controlli tecnici validi e contrasti descrittivi; Gate C sempre falso.",
        "Regole di arresto": "Stop su teacher, SaveState, scala runtime, rilascio, replay o finitezza non validi.",
        "Hash preregistrazione": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "Descrizione": "Ipotesi e confronti definiti prima dell'apertura dei risultati Task17d; Task17c è sviluppo noto.",
    })
    put("blocks", "paired", **{
        "Nome": "Task17d — condizioni native appaiate",
        "Descrizione": "Due casi critici già noti e un controllo debole, con stessi schedule e Random123 tra politiche.",
        "Seed": "170029,170083", "Protocollo": [protocol],
        "Regola di appaiamento": "Stesso equilibrio, seed, eventi pianificati, rilascio realizzato verificato e gbar per ogni contrasto.",
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False,
            "experiment": experiment, "protocol": protocol, "claim_count": 3}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
