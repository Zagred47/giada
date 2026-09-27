"""Connect original teacher findings to their archived final analyses and runs."""

from __future__ import annotations

import json

from .audit_teacher_sources import MANIFEST, audit
from .mirror import Mirror, ROOT, write_json


def main() -> dict:
    source = audit()
    if source["missing_archives"] or source["crc_invalid"]:
        raise RuntimeError("Source audit failed")
    mapping = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_by_experiment = {row["experiment"]: row for row in source["entries"]}
    mirror = Mirror()
    with mirror.connect() as conn:
        experiments = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='experiments'")}
        findings = {}
        for row in conn.execute("SELECT source_id,target_id FROM v_links WHERE source_table='findings' AND role='Esperimenti'"):
            findings.setdefault(row["target_id"], []).append(row["source_id"])
        old_analyses = {}
        for row in conn.execute("SELECT source_id,target_id FROM v_links WHERE source_table='findings' AND role='Analisi'"):
            old_analyses.setdefault(row["source_id"], []).append(row["target_id"])
        runs = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='runs'")}
        artifacts = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='artifacts'")}
    analysis_count = finding_link_count = 0
    for code, (archive_name, note_name) in mapping.items():
        row = source_by_experiment[code]
        run_id = runs["run-teacher-source-" + code.removeprefix("experiment-") + "-v1"]
        artifact_id = artifacts["artifact-teacher-source-zip-" + archive_name[:-4].replace("_", "-") + "-v1"]
        analysis = mirror.local_upsert("analyses", {
            "Nome": "Analisi finale archiviata — " + code,
            "Codice stabile": "analysis-teacher-source-" + code.removeprefix("experiment-") + "-v1",
            "Descrizione": "Indice retrospettivo dell'analisi eseguita nel notebook originale. Documento di risultato: " + (note_name or "non presente separatamente") + ". Report JSON nello ZIP: " + ", ".join(row["json_reports"]) + ". SHA-256 ZIP " + row["archive_sha256"] + ". Non si è ricalcolato né riselezionato alcun candidato.",
            "Versione": "backfill-v1",
            "Procedura": "Leggere report originale e relative metriche; preservare i limiti causali e le condizioni di selezione descritti nella fonte. L'analisi scientifica originale non viene ri-classificata come post hoc solo perché la sua indicizzazione nel DB è retrospettiva.",
            "Run analizzati": [run_id],
            "Output artefatti": [artifact_id],
        })
        analysis_count += 1
        for finding_id in findings.get(experiments[code], []):
            links = list(dict.fromkeys(old_analyses.get(finding_id, []) + [analysis["record_id"]]))
            mirror.local_upsert("findings", {"Analisi": links}, finding_id)
            finding_link_count += 1
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    status = mirror.verify()
    if not status["valid"]:
        raise RuntimeError("Mirror verification failed")
    return {"analyses": analysis_count, "finding_links": finding_link_count,
            "valid": status["valid"], "snapshot_hash": status["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
