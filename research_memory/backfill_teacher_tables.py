"""Conservative backfill of explicitly tabulated teacher results and criteria.

Every numeric observation keeps its original Markdown cell, row/column labels,
and source path. No value is inferred from prose or a model checkpoint. Tables
that cannot be parsed unambiguously remain in their source document.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .audit_teacher_sources import EXPERIMENTS, MANIFEST, audit
from .mirror import Mirror, ROOT, write_json


NUMBER = re.compile(r"^[~≈]?\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)\s*(%)?$")
SEPARATOR = re.compile(r"^:?-{2,}:?$")
NON_METRIC_HEADERS = {"posizione", "rank", "seed", "limite", "soglia", "esito", "pass", "step", "checkpoint"}


def token(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def cells(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def tables(markdown: str):
    lines = markdown.splitlines()
    for index in range(len(lines) - 1):
        if not lines[index].lstrip().startswith("|") or not lines[index + 1].lstrip().startswith("|"):
            continue
        headings, divider = cells(lines[index]), cells(lines[index + 1])
        if len(headings) != len(divider) or not all(SEPARATOR.fullmatch(part) for part in divider):
            continue
        rows = []
        cursor = index + 2
        while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
            values = cells(lines[cursor])
            if len(values) == len(headings):
                rows.append(values)
            cursor += 1
        if rows:
            yield index + 1, headings, rows


def number(cell: str):
    clean = cell.replace("**", "").replace("`", "").replace("−", "-").strip()
    match = NUMBER.fullmatch(clean)
    if match is None:
        return None
    try:
        value = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    return (value, bool(match.group(2)), clean.startswith(("~", "≈")))


def main() -> dict:
    source = audit()
    if source["missing_archives"] or source["crc_invalid"] or source["missing_experiment_records"]:
        raise RuntimeError("Source audit failed")
    mapping = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mirror = Mirror()
    with mirror.connect() as conn:
        experiment_ids = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='experiments'")}
        protocols = {r["target_id"]: r["source_id"] for r in conn.execute(
            "SELECT source_id,target_id FROM v_links WHERE source_table='protocols' AND role='Esperimento'")}
        run_ids = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='runs'")}
        artifact_ids = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='artifacts'")}
        protocol_rows = conn.execute('SELECT _record_id,"Nome","Criteri di successo","Registrato il" FROM protocols').fetchall()
        existing_prediction_protocols = {r["source_id"] for r in conn.execute(
            "SELECT source_id FROM v_links WHERE source_table='protocols' AND role='Predizioni'")}

    # A composite prediction is faithful to the recorded protocol criteria,
    # but is deliberately not presented as a set of independently tested gates.
    prediction_count = 0
    for row in protocol_rows:
        criterion = row["Criteri di successo"]
        if not criterion or row["_record_id"] in existing_prediction_protocols:
            continue
        code = "prediction-backfill-protocol-" + row["_record_id"] + "-v1"
        mirror.local_upsert("predictions", {
            "Nome": "Criteri compositi — " + row["Nome"],
            "Codice stabile": code,
            "Descrizione": "Trascrizione retrospettiva, senza scomporre o reinterpretare i gate. Fonte: campo 'Criteri di successo' del protocollo già registrato nel mirror. I singoli esiti vanno controllati sul report originale.",
            "Risultato atteso": criterion,
            "Soglia o intervallo": criterion,
            "Origine": "Preregistrata",
            **({"Registrata il": row["Registrato il"]} if row["Registrato il"] else {}),
        })
        prediction_id = mirror.local_upsert("predictions", {"Codice stabile": code})["record_id"]
        mirror.local_upsert("protocols", {"Predizioni": [prediction_id]}, row["_record_id"])
        prediction_count += 1

    source_by_experiment = {row["experiment"]: row for row in source["entries"]}
    metric_count = evaluation_count = observation_count = 0
    for experiment_code, (archive_name, note_name) in mapping.items():
        if not note_name:
            continue
        note = EXPERIMENTS / note_name
        if not note.is_file():
            continue
        report = source_by_experiment[experiment_code]
        archive_code = "artifact-teacher-source-zip-" + archive_name[:-4].replace("_", "-") + "-v1"
        artifact_id = artifact_ids[archive_code]
        protocol_id = protocols[experiment_ids[experiment_code]]
        run_code = "run-teacher-source-" + experiment_code.removeprefix("experiment-") + "-v1"
        run_id = run_ids[run_code]
        for line, headings, rows in tables(note.read_text(encoding="utf-8")):
            table_id = token(experiment_code, note_name, str(line))
            for column, header in enumerate(headings):
                if column == 0 or header.lower().strip() in NON_METRIC_HEADERS:
                    continue
                parsed = [(row_index, row, number(row[column])) for row_index, row in enumerate(rows)]
                parsed = [(row_index, row, value) for row_index, row, value in parsed if value is not None]
                if not parsed:
                    continue
                metric_code = f"metric-source-table-{table_id}-{column}-v1"
                metric_result = mirror.local_upsert("metrics", {
                    "Nome": header,
                    "Codice stabile": metric_code,
                    "Descrizione": f"Intestazione riportata senza interpretazione aggiunta in {note_name}, tabella alla linea {line}. La formula esatta resta quella del report e del codice originali; questo indice non la ricostruisce.",
                    "Famiglia": "Regressione" if "rmse" in header.lower() else "Altro",
                    **({"Unità": "mV"} if "mv" in header.lower() else {}),
                    "Direzione": "Minimizzare" if any(word in header.lower() for word in ("rmse", "errore", "error", "loss")) else "Descrittiva",
                })
                metric_count += 1
                evaluation_code = f"evaluation-source-table-{table_id}-{column}-v1"
                evaluation_result = mirror.local_upsert("evaluations", {
                    "Nome": f"{experiment_code}: {header}",
                    "Codice stabile": evaluation_code,
                    "Descrizione": f"Trascrizione dei soli valori numerici espliciti in {note_name}, tabella alla linea {line}, colonna originale '{header}'. Precisione limitata agli arrotondamenti pubblicati nel documento; consultare lo ZIP per la precisione nativa. Hash ZIP: {report['archive_sha256']}.",
                    "Versione": "backfill-v1",
                    "Target": header,
                    "Popolazione e regioni": "Come nelle righe della tabella di fonte; non inferire indipendenza o numerosità da questo indice.",
                    "Aggregazione e pesi": "Esattamente il valore tabulato dalla fonte; nessuna riaggregazione eseguita nel backfill.",
                    "Ruolo": "Diagnostica",
                    "Protocollo": [protocol_id],
                    "Metrica": [metric_result["record_id"]],
                })
                evaluation_count += 1
                for row_index, row, (value, percent, approximate) in parsed:
                    cell = row[column]
                    observation_code = f"observation-source-table-{table_id}-{column}-{row_index}-v1"
                    mirror.local_upsert("observations", {
                        "Nome": f"{row[0]} — {header}",
                        "Codice stabile": observation_code,
                        "Descrizione": f"Cella originale '{cell}' in {note_name}:{line + 2 + row_index}. {'Valore approssimato nel documento; ' if approximate else ''}{'Percentuale numerica (non frazione); ' if percent else ''}ZIP {archive_name} SHA-256 {report['archive_sha256']}. Nessuna conversione o calcolo di nuovi valori.",
                        "Valore": value,
                        "Strato o sottogruppo": row[0],
                        "Specifica di valutazione": [evaluation_result["record_id"]],
                        "Run": [run_id],
                        "Artefatti dettagliati": [artifact_id],
                    })
                    observation_count += 1
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    verification = mirror.verify()
    if not verification["valid"]:
        raise RuntimeError("Mirror verification failed after table backfill")
    return {"predictions": prediction_count, "metrics": metric_count,
            "evaluations": evaluation_count, "observations": observation_count,
            "verified": verification["valid"], "snapshot_hash": verification["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
