"""Read-only provenance audit for the GIADA teacher experiment backfill.

This deliberately does not infer a scientific result from a filename. A ZIP is
marked hash-matched only when its SHA-256 occurs in the versioned result note.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from .mirror import Mirror, ROOT


MANIFEST = ROOT / "records" / "teacher_source_backfill_manifest_v1.json"
EXPERIMENTS = ROOT.parent / "experiments"
DOWNLOADS = Path.home() / "Downloads"
VERSIONED_ARCHIVES = ROOT / "source_archives"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict:
    mapping = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with Mirror().connect() as conn:
        known = {
            row["stable_code"]: row["record_id"]
            for row in conn.execute("SELECT stable_code, record_id FROM v_records WHERE table_key='experiments'")
        }
    entries = []
    for code, (archive_name, note_name) in mapping.items():
        versioned_archive = VERSIONED_ARCHIVES / archive_name
        archive = versioned_archive if versioned_archive.is_file() else DOWNLOADS / archive_name
        note = EXPERIMENTS / note_name if note_name else None
        entry = {"experiment": code, "experiment_id": known.get(code), "archive": str(archive),
                 "note": str(note) if note else None, "archive_exists": archive.is_file(),
                 "note_exists": note.is_file() if note else False,
                 "archive_versioned_in_repository": versioned_archive.is_file()}
        if archive.is_file():
            entry["archive_sha256"] = sha256(archive)
            entry["archive_bytes"] = archive.stat().st_size
            with zipfile.ZipFile(archive) as handle:
                entry["zip_crc_valid"] = handle.testzip() is None
                entry["json_reports"] = [name for name in handle.namelist()
                                         if name.endswith(("final_report.json", "task14_parameter_matrix_report.json"))]
                entry["reports_parse"] = all(isinstance(json.loads(handle.read(name)), dict)
                                             for name in entry["json_reports"])
        if note and note.is_file():
            text = note.read_text(encoding="utf-8")
            hashes = set(re.findall(r"[0-9a-f]{64}", text))
            entry["archive_hash_in_note"] = entry.get("archive_sha256") in hashes
            entry["note_has_any_sha256"] = bool(hashes)
        entries.append(entry)
    return {
        "schema_version": "giada-teacher-source-audit-v1",
        "entry_count": len(entries),
        "unique_archives": len({row["archive"] for row in entries}),
        "missing_archives": [row["experiment"] for row in entries if not row["archive_exists"]],
        "missing_notes": [row["experiment"] for row in entries if row["note"] and not row["note_exists"]],
        "missing_experiment_records": [row["experiment"] for row in entries if not row["experiment_id"]],
        "hash_matched": sum(bool(row.get("archive_hash_in_note")) for row in entries),
        "hash_unmatched": [row["experiment"] for row in entries if row["note_exists"] and not row.get("archive_hash_in_note")],
        "crc_invalid": [row["archive"] for row in entries if row["archive_exists"] and not row["zip_crc_valid"]],
        "entries": entries,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
