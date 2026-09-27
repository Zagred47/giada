"""Backfill exact local GIADA teacher source archives and aggregate notebook runs.

This script does not infer unreported metrics, per-seed runs, or causal claims.
It checks archive integrity before writing and uses only the local SQLite mirror.
"""

from __future__ import annotations

import json

from .audit_teacher_sources import audit
from .mirror import Mirror, ROOT, write_json


def main() -> dict:
    source = audit()
    if source["missing_archives"] or source["missing_notes"] or source["missing_experiment_records"] or source["crc_invalid"]:
        raise RuntimeError("Source audit failed; refusing partial backfill")
    mirror = Mirror()
    with mirror.connect() as conn:
        protocols = {
            row["target_id"]: row["source_id"]
            for row in conn.execute("SELECT source_id,target_id FROM v_links WHERE source_table='protocols' AND role='Esperimento'")
        }
    archives = {}
    runs = {}
    for row in source["entries"]:
        archive = row["archive"]
        if archive not in archives:
            name = archive.replace("\\", "/").rsplit("/", 1)[-1]
            archive_code = "artifact-teacher-source-zip-" + name[:-4].replace("_", "-") + "-v1"
            association = "ZIP SHA-256 presente nel documento di risultato" if row.get("archive_hash_in_note") else (
                "Il documento non riporta lo SHA dello ZIP; associazione tramite nome/struttura, non prova crittografica"
            )
            result = mirror.local_upsert("artifacts", {
                "Nome": name,
                "Codice stabile": archive_code,
                "Descrizione": "Archivio sorgente locale GIADA, verificato ZIP/CRC. " + association + ". Contenuto JSON: " + ", ".join(row["json_reports"]),
                "Tipo": "Report",
                "Percorso": "research_memory/source_archives/" + name if row["archive_versioned_in_repository"] else archive,
                "SHA-256": row["archive_sha256"],
                "Dimensione byte": row["archive_bytes"],
                "Versione": "backfill-2026-09-27",
            })
            archives[archive] = result["record_id"]
        code = row["experiment"]
        note = row["note"] or "nessun documento di risultato separato"
        result = mirror.local_upsert("runs", {
            "Nome": "Esecuzione aggregata " + code,
            "Codice stabile": "run-teacher-source-" + code.removeprefix("experiment-") + "-v1",
            "Descrizione": "Record retrospettivo dell'esecuzione che ha prodotto l'archivio; non rappresenta i singoli seed. Documento di risultato: " + note + ". Associazione documento/ZIP: " + ("SHA-256 verificato" if row.get("archive_hash_in_note") else "hash ZIP non riportato nel documento; associazione da confermare semanticamente") + ".",
            "Stato": "Completata",
            "Validità tecnica": "ZIP CRC e JSON principali verificati; lo stato scientifico e i gate restano quelli del report originale, non dedotti da questo record.",
            "Artefatti prodotti": [archives[archive]],
        })
        runs[code] = result["record_id"]
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    result = mirror.verify()
    if not result["valid"]:
        raise RuntimeError("Mirror verification failed after source backfill")
    return {"archives": len(archives), "runs": len(runs), "protocols_found": sum(x["experiment_id"] in protocols for x in source["entries"]), "verified": result["valid"], "snapshot_hash": result["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
