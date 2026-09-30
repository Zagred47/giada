"""Preregister Task17d-b in the local SQLite mirror only."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    path = ROOT.parent / "experiments/task17db_solver_release_confirmation_preregistration.md"
    payload = path.read_bytes()
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17db-solver-release-{suffix}-v1",
            **fields,
        })["record_id"]

    release = put("claims", "discrete-release", **{
        "Nome": "Task17d-b: l'hash completo può cambiare senza flip del rilascio",
        "Enunciato": "Le decisioni discrete e i draw Random123 possono coincidere fra policy CVode anche quando stati/probabilità sinaptici continui differiscono.",
        "Tipo": "Ipotesi", "Stato": "Aperta",
        "Condizioni di falsificazione": "Almeno un evento cambia identità, draw Random123 o release_success fra policy a condizioni appaiate.",
        "Limiti": "Anche decisioni identiche non garantiscono quantità rilasciata identica.",
    })
    convergence = put("claims", "reference-stability", **{
        "Nome": "Task17d-b: riferimento ultra stabile sui casi noti",
        "Enunciato": "Nativo ultra e super_ultra concordano entro 0,05 mV RMSE e 0,002 errore massimo gate ai quattro siti nei tre casi noti.",
        "Tipo": "Ipotesi", "Stato": "Aperta",
        "Condizioni di falsificazione": "Almeno un caso/sito supera una delle due soglie preregistrate.",
        "Limiti": "Stabilità empirica limitata ai casi noti, non convergenza matematica globale.",
    })
    candidate = put("claims", "matched-candidates", **{
        "Nome": "Task17d-b: formula e LUT a solver identico",
        "Enunciato": "Con policy ultra identica, il clone formula passa il controllo e le due LUT possono essere confrontate col nativo senza confondere tolleranze solver differenti.",
        "Tipo": "Ipotesi", "Stato": "Aperta",
        "Condizioni di falsificazione": "Clone formula oltre 0,05 mV RMSE o 0,002 errore gate, oppure policy non applicata come registrato.",
        "Limiti": "Solo casi di sviluppo; confronto non selettivo e nessun Gate C.",
    })
    experiment = put("experiments", "matrix", **{
        "Nome": "GIADA Task17d-b — conferma rilasci, solver e candidati",
        "Descrizione": "Un notebook: quattro policy native, tre sostituzioni Ca_HVA alla stessa policy ultra, tre condizioni note.",
        "Obiettivo informativo": "Separare decisioni discrete di rilascio, stabilità del riferimento ultra e differenze formula/LUT a solver identico.",
        "Tipo": "Forensic", "Stato": "Preregistrato",
        "Ipotesi": [release, convergence, candidate],
    })
    protocol = put("protocols", "v1", **{
        "Nome": "Task17d-b — matrice rilascio e riferimento nativo",
        "Versione": "v1", "Modalità": "Sistema completo", "Esperimento": [experiment],
        "Procedura": "Teacher 642 segmenti; default/voltage_scaled/ultra/super_ultra nativi e formula/LUT513/m-LUT2049 sotto ultra; stessi tre casi Task17d; decisioni e grandezze continue del rilascio salvate separatamente.",
        "Criteri di successo": "Controlli tecnici validi; stabilità e formula passano soltanto alle soglie fissate; Gate C sempre falso.",
        "Regole di arresto": "Stop su teacher, compilazione, solver, snapshot, verifica del rilascio o finitezza non validi.",
        "Hash preregistrazione": hashlib.sha256(payload).hexdigest(),
        "Descrizione": "Protocollo fissato prima del nuovo run. I casi 17d sono già aperti e non formano test indipendente.",
    })
    put("blocks", "paired", **{
        "Nome": "Task17d-b — tre condizioni note appaiate",
        "Descrizione": "Seed/gbar 170029/1.0, 170083/1.5, 170029/0.5; durata 60 ms.",
        "Seed": "170029,170083", "Protocollo": [protocol],
        "Regola di appaiamento": "Stesso snapshot, schedule, Random123 e gbar; snapshot aggiornato quando cambia la struttura del meccanismo.",
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "experiment": experiment,
            "protocol": protocol, "claim_count": 3}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
