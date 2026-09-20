"""Dependency-free Airtable/SQLite mirror and verified dual-write outbox.

SQLite holds the last verified remote snapshot, not speculative pending writes.
All user SQL is read-only. Airtable record and field IDs are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "research_memory.sqlite"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Generated export, not source-code editing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def literal(value):
    return "'" + value.replace("'", "''") + "'"


class Contract:
    def __init__(self, directory=ROOT / "schema"):
        directory = Path(directory)
        logical = read_json(directory / "schema.json")
        manifest = read_json(directory / "airtable_manifest.json")
        actual = read_json(directory / "airtable_schema.json")
        configs = read_json(directory / "airtable_field_config.json")
        self.base_id = manifest["baseId"]
        self.tables = {}
        self.by_id = {}
        self.links = manifest["relationships"]
        cfg = {f["id"]: f.get("config", {}) for t in configs["tables"] for f in t["fields"]}
        remote = {t["id"]: t for t in actual["tables"]}
        for item in logical["tables"]:
            key = item["key"]
            table_id = manifest["tableIds"][key]
            table = dict(remote[table_id], key=key)
            table["fields"] = [dict(f, config=cfg[f["id"]]) for f in table["fields"]]
            self.tables[key] = table
            self.by_id[table_id] = key
        self.link_fields = {}
        for link in self.links:
            link["sql_table"] = "rel_" + link["fieldId"]
            self.link_fields[link["fieldId"]] = (link, False)
            self.link_fields[link["inverseFieldId"]] = (link, True)
        actual_links = {f["id"] for t in self.tables.values() for f in t["fields"] if f["type"] == "multipleRecordLinks"}
        if actual_links != set(self.link_fields):
            raise ValueError("Link catalog does not exactly cover Airtable fields")
        for key, t in self.tables.items():
            for f in t["fields"]:
                if f["type"] == "multipleRecordLinks":
                    link, reverse = self.link_fields[f["id"]]
                    expected = self.tables[link["from"] if reverse else link["to"]]["id"]
                    if f["config"].get("linkedTableId") != expected:
                        raise ValueError("Airtable relationship target mismatch")
                elif f["type"] not in {"singleLineText", "multilineText", "url", "number", "singleSelect"}:
                    raise ValueError(f"Unsupported field type: {f['type']}; add an explicit migration")
        self.fingerprint = digest({"tables": self.tables, "links": self.links, "base_id": self.base_id})

    def key(self, table):
        if table in self.tables:
            return table
        if table in self.by_id:
            return self.by_id[table]
        for key, spec in self.tables.items():
            if spec["name"] == table:
                return key
        raise ValueError(f"Unknown table: {table}")

    def fields(self, table):
        return {f["id"]: f for f in self.tables[self.key(table)]["fields"]}

    def normalize(self, table, values, *, partial=False):
        fields = self.fields(table)
        by_name = {f["name"]: f["id"] for f in fields.values()}
        result = {}
        for name, value in values.items():
            field_id = name if name in fields else by_name.get(name)
            if field_id is None:
                raise ValueError(f"Unknown field {name!r} in {table}")
            if field_id in result:
                raise ValueError("Same field supplied by both ID and name")
            field = fields[field_id]
            kind = field["type"]
            if kind == "multipleRecordLinks":
                if value is None:
                    value = []
                if not isinstance(value, list):
                    raise ValueError("Links must be arrays of Airtable record IDs")
                value = [v.get("id") if isinstance(v, dict) else v for v in value]
                if any(not isinstance(v, str) or not re.fullmatch(r"rec[A-Za-z0-9]{14}", v) for v in value):
                    raise ValueError("Invalid linked record ID")
                if len(value) != len(set(value)):
                    raise ValueError("Duplicate linked record")
            elif kind == "number":
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError("Number must be finite or null")
            else:
                if kind == "singleSelect" and isinstance(value, dict):
                    value = value.get("name")
                if value == "":
                    value = None
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{field['name']} must be text or null")
                if kind == "singleSelect" and value is not None:
                    allowed = {c["name"] for c in field["config"].get("choices", [])}
                    if value not in allowed:
                        raise ValueError(f"Unknown choice {value!r} for {field['name']}")
            result[field_id] = value
        if not partial:
            for fid, field in fields.items():
                result.setdefault(fid, [] if field["type"] == "multipleRecordLinks" else None)
        return result


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class Mirror:
    def __init__(self, path=DEFAULT_DB, contract=None):
        self.path = Path(path)
        self.contract = contract or Contract()

    def connect(self):
        if not self.path.is_file():
            raise ValueError("Database missing; run init first")
        conn = sqlite3.connect(self.path, timeout=30, factory=ClosingConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        row = conn.execute("SELECT value FROM _mirror_meta WHERE key='schema_fingerprint'").fetchone()
        if row is None or row[0] != self.contract.fingerprint:
            conn.close()
            raise ValueError("Schema drift: explicit migration required; database left untouched")
        return conn

    def initialize(self):
        if self.path.exists():
            with self.connect() as conn:
                return {"created": False, "path": str(self.path)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive reservation: never overwrite an existing database.
        with self.path.open("xb"):
            pass
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                CREATE TABLE _mirror_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE _mirror_tables (table_key TEXT PRIMARY KEY, airtable_id TEXT UNIQUE NOT NULL, name TEXT UNIQUE NOT NULL);
                CREATE TABLE _mirror_fields (field_id TEXT PRIMARY KEY, table_key TEXT NOT NULL REFERENCES _mirror_tables(table_key), name TEXT NOT NULL, type TEXT NOT NULL, config_json TEXT NOT NULL);
                CREATE TABLE _mirror_relations (field_id TEXT PRIMARY KEY, inverse_field_id TEXT UNIQUE NOT NULL, source_table TEXT NOT NULL, target_table TEXT NOT NULL, field_name TEXT NOT NULL, sql_table TEXT UNIQUE NOT NULL, intended_single INTEGER NOT NULL);
                CREATE TABLE _remote_records (table_key TEXT NOT NULL, record_id TEXT NOT NULL, created_time TEXT, fields_json TEXT NOT NULL, PRIMARY KEY(table_key, record_id));
                CREATE TABLE _outbox (operation_id TEXT PRIMARY KEY, table_key TEXT NOT NULL, stable_code TEXT NOT NULL, record_id TEXT, patch_json TEXT NOT NULL, before_json TEXT, status TEXT NOT NULL CHECK(status IN ('pending','applied')), verified_snapshot_hash TEXT);
                CREATE UNIQUE INDEX _outbox_one_pending ON _outbox(table_key, stable_code) WHERE status='pending';
            """)
            conn.execute("INSERT INTO _mirror_meta VALUES ('schema_fingerprint', ?)", (self.contract.fingerprint,))
            conn.execute("INSERT INTO _mirror_meta VALUES ('base_id', ?)", (self.contract.base_id,))
            for key, t in self.contract.tables.items():
                conn.execute("INSERT INTO _mirror_tables VALUES (?,?,?)", (key, t["id"], t["name"]))
                columns = ['"_record_id" TEXT PRIMARY KEY', '"_created_time" TEXT']
                for f in t["fields"]:
                    conn.execute("INSERT INTO _mirror_fields VALUES (?,?,?,?,?)", (f["id"], key, f["name"], f["type"], dumps(f["config"])))
                    if f["type"] != "multipleRecordLinks":
                        columns.append(quote(f["name"]) + (" REAL" if f["type"] == "number" else " TEXT"))
                conn.execute(f"CREATE TABLE {quote(key)} ({', '.join(columns)})")
            for l in self.contract.links:
                conn.execute("INSERT INTO _mirror_relations VALUES (?,?,?,?,?,?,?)", (l["fieldId"], l["inverseFieldId"], l["from"], l["to"], l["name"], l["sql_table"], int(l["single"])))
                conn.execute(f"CREATE TABLE {quote(l['sql_table'])} (source_id TEXT NOT NULL REFERENCES {quote(l['from'])}(_record_id), target_id TEXT NOT NULL REFERENCES {quote(l['to'])}(_record_id), source_position INTEGER NOT NULL, target_position INTEGER NOT NULL, PRIMARY KEY(source_id,target_id))")
                conn.execute(f"CREATE INDEX {quote('idx_' + l['fieldId'])} ON {quote(l['sql_table'])}(target_id)")
            for key, t in self.contract.tables.items():
                columns = ['r."_record_id"', 'r."_created_time"']
                for f in t["fields"]:
                    if f["type"] != "multipleRecordLinks":
                        columns.append("r." + quote(f["name"]))
                    else:
                        l, reverse = self.contract.link_fields[f["id"]]
                        owner, other, order = ("target_id", "source_id", "target_position") if reverse else ("source_id", "target_id", "source_position")
                        columns.append(f"(SELECT json_group_array(link_id) FROM (SELECT {other} AS link_id FROM {quote(l['sql_table'])} WHERE {owner}=r._record_id ORDER BY {order}, {other})) AS {quote(f['name'])}")
                conn.execute(f"CREATE VIEW {quote(t['name'])} AS SELECT {', '.join(columns)} FROM {quote(key)} r")
            conn.execute("CREATE VIEW v_records AS " + " UNION ALL ".join(f"SELECT {literal(key)} AS table_key, _record_id AS record_id, {quote('Nome')} AS name, {quote('Codice stabile')} AS stable_code FROM {quote(key)}" for key in self.contract.tables))
            conn.execute("CREATE VIEW v_links AS " + " UNION ALL ".join(f"SELECT {literal(l['fieldId'])} AS field_id, {literal(l['name'])} AS role, {literal(l['from'])} AS source_table, source_id, {literal(l['to'])} AS target_table, target_id FROM {quote(l['sql_table'])}" for l in self.contract.links))
            conn.executescript("""
                CREATE VIEW v_semantic_edges AS
                SELECT l.*, s.name AS source_name, t.name AS target_name
                FROM v_links l JOIN v_records s ON s.table_key=l.source_table AND s.record_id=l.source_id
                JOIN v_records t ON t.table_key=l.target_table AND t.record_id=l.target_id;
                CREATE VIEW v_evidence_chains AS
                SELECT e.source_id AS assessment_id, e.source_name AS assessment,
                       e.target_id AS claim_id, e.target_name AS claim,
                       f.target_id AS finding_id, f.target_name AS finding
                FROM v_semantic_edges e LEFT JOIN v_semantic_edges f
                  ON f.source_table='evidence' AND f.source_id=e.source_id AND f.target_table='findings'
                WHERE e.source_table='evidence' AND e.target_table='claims';
            """)
            conn.commit()
        except Exception:
            conn.close()
            # Only the file exclusively created in this invocation is removed.
            self.path.unlink()
            raise
        finally:
            conn.close()
        return {"created": True, "path": str(self.path)}

    def normalize_snapshot(self, snapshot):
        if snapshot.get("format") != "airtable-record-snapshot-v1" or snapshot.get("base_id") != self.contract.base_id or snapshot.get("complete") is not True:
            raise ValueError("Expected a complete snapshot of this exact base")
        if set(snapshot.get("tables", {})) != set(self.contract.by_id):
            raise ValueError("Snapshot must contain all 78 tables, including empty tables")
        data = {}
        for tid, records in snapshot["tables"].items():
            key = self.contract.by_id[tid]
            data[key] = {}
            for record in records:
                rid = record["id"]
                if not re.fullmatch(r"rec[A-Za-z0-9]{14}", rid) or rid in data[key]:
                    raise ValueError("Invalid or duplicate Airtable record ID")
                fields = self.contract.normalize(key, record.get("cellValuesByFieldId", record.get("fields", {})))
                data[key][rid] = {"fields": fields, "createdTime": record.get("createdTime")}
        # Ensure both directions agree; a changing remote snapshot must be re-read.
        for l in self.contract.links:
            forward = {(rid, target) for rid, r in data[l["from"]].items() for target in r["fields"][l["fieldId"]]}
            backward = {(source, rid) for rid, r in data[l["to"]].items() for source in r["fields"][l["inverseFieldId"]]}
            if forward != backward:
                raise ValueError(f"Missing/dangling/inconsistent reciprocal links: {l['fieldId']}")
        return data

    def import_snapshot(self, snapshot):
        data = self.normalize_snapshot(snapshot)
        snapshot_hash = digest(data)
        with self.connect() as conn:
            for l in self.contract.links:
                conn.execute(f"DELETE FROM {quote(l['sql_table'])}")
            for key in self.contract.tables:
                conn.execute(f"DELETE FROM {quote(key)}")
            conn.execute("DELETE FROM _remote_records")
            for key, records in data.items():
                fields = [f for f in self.contract.tables[key]["fields"] if f["type"] != "multipleRecordLinks"]
                columns = ["_record_id", "_created_time"] + [f["name"] for f in fields]
                for rid, record in records.items():
                    values = [rid, record["createdTime"]] + [record["fields"][f["id"]] for f in fields]
                    conn.execute(f"INSERT INTO {quote(key)} ({','.join(map(quote, columns))}) VALUES ({','.join('?' for _ in columns)})", values)
                    conn.execute("INSERT INTO _remote_records VALUES (?,?,?,?)", (key, rid, record["createdTime"], dumps(record["fields"])))
            for l in self.contract.links:
                for rid, record in data[l["from"]].items():
                    for pos, target in enumerate(record["fields"][l["fieldId"]]):
                        inverse = data[l["to"]][target]["fields"][l["inverseFieldId"]]
                        conn.execute(f"INSERT INTO {quote(l['sql_table'])} VALUES (?,?,?,?)", (rid, target, pos, inverse.index(rid)))
            conn.execute("INSERT OR REPLACE INTO _mirror_meta VALUES ('snapshot_hash', ?)", (snapshot_hash,))
            # A pending write is applied only after exact readback, not after an HTTP acknowledgement.
            for op in conn.execute("SELECT * FROM _outbox WHERE status='pending'").fetchall():
                patch = json.loads(op["patch_json"])
                code_fid = self.field_id(op["table_key"], "Codice stabile")
                matches = [(rid, r) for rid, r in data[op["table_key"]].items() if r["fields"][code_fid] == op["stable_code"]]
                if len(matches) == 1:
                    rid, record = matches[0]
                    if (op["record_id"] is None or rid == op["record_id"]) and all(record["fields"][f] == v for f, v in patch.items()):
                        conn.execute("UPDATE _outbox SET status='applied',record_id=?,verified_snapshot_hash=? WHERE operation_id=?", (rid, snapshot_hash, op["operation_id"]))
        return self.status()

    def field_id(self, table, name):
        return next(f["id"] for f in self.contract.tables[self.contract.key(table)]["fields"] if f["name"] == name)

    def export_snapshot(self):
        snapshot = {"format": "airtable-record-snapshot-v1", "base_id": self.contract.base_id, "complete": True, "tables": {t["id"]: [] for t in self.contract.tables.values()}}
        with self.connect() as conn:
            for r in conn.execute("SELECT * FROM _remote_records ORDER BY table_key, record_id"):
                snapshot["tables"][self.contract.tables[r["table_key"]]["id"]].append({"id": r["record_id"], "createdTime": r["created_time"], "cellValuesByFieldId": json.loads(r["fields_json"])})
        return snapshot

    def local_upsert(self, table, values, record_id=None):
        """Publish one semantic record to the local mirror without Airtable.

        Local IDs intentionally retain Airtable's syntactic shape so existing
        relationship tables and validators remain usable.  Stable codes, not
        these provisional IDs, are the identity used during a future explicit
        reconciliation with Airtable.
        """

        key = self.contract.key(table)
        patch = self.contract.normalize(key, values, partial=True)
        code_fid = self.field_id(key, "Codice stabile")
        name_fid = self.field_id(key, "Nome")
        snapshot = self.export_snapshot()
        table_id = self.contract.tables[key]["id"]
        records = snapshot["tables"][table_id]

        by_code = []
        for row in records:
            normalized = self.contract.normalize(key, row.get("cellValuesByFieldId", {}))
            if normalized[code_fid] == patch.get(code_fid):
                by_code.append(row)
        if record_id is not None:
            matches = [row for row in records if row["id"] == record_id]
            if len(matches) != 1:
                raise ValueError("Local update target missing or ambiguous")
            row = matches[0]
            current = self.contract.normalize(key, row.get("cellValuesByFieldId", {}))
            stable_code = current[code_fid]
            if not stable_code or (code_fid in patch and patch[code_fid] != stable_code):
                raise ValueError("Local updates require an immutable stable code")
            patch[code_fid] = stable_code
        elif by_code:
            if len(by_code) != 1:
                raise ValueError("Duplicate local stable code")
            row = by_code[0]
            current = self.contract.normalize(key, row.get("cellValuesByFieldId", {}))
            stable_code = current[code_fid]
        else:
            stable_code = patch.get(code_fid)
            if not stable_code or not patch.get(name_fid):
                raise ValueError("New local records require Nome and Codice stabile")
            provisional = "rec" + hashlib.sha256((key + "\0" + stable_code).encode("utf-8")).hexdigest()[:14]
            all_ids = {item["id"] for rows in snapshot["tables"].values() for item in rows}
            if provisional in all_ids:
                raise ValueError("Deterministic local record ID collision")
            row = {
                "id": provisional,
                "createdTime": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "cellValuesByFieldId": {},
            }
            records.append(row)
            current = self.contract.normalize(key, {})

        before = dict(current)
        merged = dict(current)
        merged.update(patch)
        merged = self.contract.normalize(key, merged)
        row["cellValuesByFieldId"] = merged

        # Maintain reciprocal link fields automatically and deterministically.
        for field_id, value in patch.items():
            if field_id not in self.contract.link_fields:
                continue
            link, reverse = self.contract.link_fields[field_id]
            if link["single"] and not reverse and len(value) > 1:
                raise ValueError("Relationship is intended to have one target")
            target_key = link["from"] if reverse else link["to"]
            inverse_id = link["fieldId"] if reverse else link["inverseFieldId"]
            target_table_id = self.contract.tables[target_key]["id"]
            target_rows = {item["id"]: item for item in snapshot["tables"][target_table_id]}
            old_targets, new_targets = set(before[field_id]), set(merged[field_id])
            if any(target not in target_rows for target in new_targets):
                raise ValueError("Linked local record does not exist")
            for target in old_targets | new_targets:
                target_row = target_rows.get(target)
                if target_row is None:
                    continue
                target_fields = self.contract.normalize(target_key, target_row.get("cellValuesByFieldId", {}))
                inverse = list(target_fields[inverse_id])
                if target in old_targets - new_targets and row["id"] in inverse:
                    inverse.remove(row["id"])
                if target in new_targets and row["id"] not in inverse:
                    inverse.append(row["id"])
                target_fields[inverse_id] = inverse
                target_row["cellValuesByFieldId"] = target_fields

        result = self.import_snapshot(snapshot)
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO _mirror_meta VALUES ('operating_mode','local_sqlite_authoritative_pending_airtable_reconciliation')")
        result.update({"record_id": row["id"], "stable_code": stable_code, "table": key})
        return result

    def stage(self, table, values, record_id=None):
        key = self.contract.key(table)
        patch = self.contract.normalize(key, values, partial=True)
        code_fid = self.field_id(key, "Codice stabile")
        with self.connect() as conn:
            before = None
            if record_id:
                r = conn.execute("SELECT fields_json FROM _remote_records WHERE table_key=? AND record_id=?", (key, record_id)).fetchone()
                if r is None:
                    raise ValueError("Update target missing from verified local snapshot; pull first")
                before = json.loads(r[0])
                code = before[code_fid]
                if not code or (code_fid in patch and patch[code_fid] != code):
                    raise ValueError("Updates require an existing immutable stable code")
                patch[code_fid] = code
            else:
                code = patch.get(code_fid)
                if not code or not patch.get(self.field_id(key, "Nome")):
                    raise ValueError("New records require Nome and Codice stabile")
                if conn.execute(f"SELECT 1 FROM {quote(key)} WHERE {quote('Codice stabile')}=?", (code,)).fetchone():
                    raise ValueError("Stable code already exists; update by record ID")
            for fid, value in patch.items():
                if fid in self.contract.link_fields:
                    l, reverse = self.contract.link_fields[fid]
                    target = l["from"] if reverse else l["to"]
                    if l["single"] and not reverse and len(value) > 1:
                        raise ValueError("Relationship is intended to have one target")
                    for rid in value:
                        if not conn.execute(f"SELECT 1 FROM {quote(target)} WHERE _record_id=?", (rid,)).fetchone():
                            raise ValueError("Linked record must already be in verified snapshot")
            opid = str(uuid.uuid4())
            # Only one unresolved operation per semantic identity. Applied history is unlimited.
            if conn.execute("SELECT 1 FROM _outbox WHERE table_key=? AND stable_code=? AND status='pending'", (key, code)).fetchone():
                raise ValueError("Pending operation already exists for this record")
            conn.execute("INSERT INTO _outbox (operation_id,table_key,stable_code,record_id,patch_json,before_json,status) VALUES (?,?,?,?,?,?,'pending')", (opid, key, code, record_id, dumps(patch), dumps(before) if before is not None else None))
        return {"operation_id": opid, "status": "pending", "message": "Not yet written to either semantic dataset; use payload or sync"}

    def operation(self, operation_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM _outbox WHERE operation_id=?", (operation_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown operation ID")
            return dict(row)

    def payload(self, operation_id):
        op = self.operation(operation_id)
        if op["status"] != "pending":
            raise ValueError("Operation already applied; do not replay it")
        key = op["table_key"]
        record = {"fields": json.loads(op["patch_json"])}
        if op["record_id"]:
            record["id"] = op["record_id"]
        return {"tool": "airtable_update_records_for_table", "arguments": {"baseId": self.contract.base_id, "tableId": self.contract.tables[key]["id"], "performUpsert": {"fieldIdsToMergeOn": [self.field_id(key, "Codice stabile")]}, "typecast": False, "records": [record]}, "preflight": {"record_id": op["record_id"], "expected_before": json.loads(op["before_json"]) if op["before_json"] else None}, "completion": "Re-read the complete base and import-snapshot; a tool receipt alone is insufficient."}

    def preflight(self, operation_id, snapshot):
        op = self.operation(operation_id)
        if op["status"] != "pending":
            raise ValueError("Operation is not pending")
        data = self.normalize_snapshot(snapshot)
        key = op["table_key"]
        code_fid = self.field_id(key, "Codice stabile")
        matches = [(rid, r["fields"]) for rid, r in data[key].items() if r["fields"][code_fid] == op["stable_code"]]
        patch = json.loads(op["patch_json"])
        if len(matches) > 1:
            raise ValueError("Duplicate stable codes on Airtable: conflict")
        if matches and (op["record_id"] is None or matches[0][0] == op["record_id"]) and all(matches[0][1][f] == v for f, v in patch.items()):
            return "already_applied"
        if op["record_id"] is None:
            if matches:
                raise ValueError("Conflicting remote record already uses this stable code")
        else:
            if len(matches) != 1 or matches[0][0] != op["record_id"] or matches[0][1] != json.loads(op["before_json"]):
                raise ValueError("Remote record changed since staging; no overwrite performed")
        for fid, value in patch.items():
            if fid in self.contract.link_fields:
                l, reverse = self.contract.link_fields[fid]
                target = l["from"] if reverse else l["to"]
                if any(rid not in data[target] for rid in value):
                    raise ValueError("A linked record no longer exists remotely")
        return "ready"

    def status(self):
        with self.connect() as conn:
            counts = {k: conn.execute(f"SELECT count(*) FROM {quote(k)}").fetchone()[0] for k in self.contract.tables}
            pending = conn.execute("SELECT count(*) FROM _outbox WHERE status='pending'").fetchone()[0]
            fk = [tuple(r) for r in conn.execute("PRAGMA foreign_key_check")]
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            snap = conn.execute("SELECT value FROM _mirror_meta WHERE key='snapshot_hash'").fetchone()
        return {"base_id": self.contract.base_id, "tables": len(counts), "relationships": len(self.contract.links), "records": sum(counts.values()), "pending_writes": pending, "foreign_key_errors": fk, "integrity": integrity, "snapshot_hash": snap[0] if snap else None, "table_counts": counts}

    def verify(self):
        """Check normalized SQL tables/views against the preserved remote snapshot."""
        actual = {"format": "airtable-record-snapshot-v1", "base_id": self.contract.base_id, "complete": True, "tables": {}}
        with self.connect() as conn:
            for key, t in self.contract.tables.items():
                rows = []
                for r in conn.execute(f"SELECT * FROM {quote(t['name'])} ORDER BY _record_id"):
                    cells = {f["id"]: json.loads(r[f["name"]]) if f["type"] == "multipleRecordLinks" else r[f["name"]] for f in t["fields"]}
                    rows.append({"id": r["_record_id"], "createdTime": r["_created_time"], "fields": cells})
                actual["tables"][t["id"]] = rows
        normalized = self.normalize_snapshot(actual)
        expected = self.normalize_snapshot(self.export_snapshot())
        # SQLite REAL can return 1.0 where Airtable returned 1; numeric equality is intentional.
        status = self.status()
        status["logical_roundtrip_equal"] = normalized == expected
        status["valid"] = normalized == expected and status["integrity"] == "ok" and not status["foreign_key_errors"]
        return status

    def query(self, sql, limit=200):
        if not 1 <= limit <= 10000:
            raise ValueError("Limit must be between 1 and 10000")
        with self.connect() as conn:
            conn.execute("PRAGMA query_only=ON")
            denied = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}
            conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK)
            # Bound runaway recursive/Cartesian queries, not only output size.
            ticks = [0]
            def progress():
                ticks[0] += 1
                return int(ticks[0] > 10000)
            conn.set_progress_handler(progress, 1000)
            cursor = conn.execute(sql)
            if cursor.description is None:
                raise ValueError("Only read queries are permitted")
            rows = cursor.fetchmany(limit + 1)
            return {"rows": [dict(r) for r in rows[:limit]], "truncated": len(rows) > limit}


class AirtableREST:
    """Optional unattended transport. Never prints or persists credentials."""
    def __init__(self, contract, token=None):
        self.contract = contract
        self.token = token or os.environ.get("AIRTABLE_TOKEN")
        if not self.token:
            raise ValueError("AIRTABLE_TOKEN missing. Use the connected Airtable tools plus payload/import-snapshot instead; never commit a token.")

    def request(self, method, path, body=None, params=None):
        url = "https://api.airtable.com/v0/" + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        req = urllib.request.Request(url, data=dumps(body).encode() if body is not None else None, method=method, headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.load(response)
            time.sleep(0.22)  # Stay below five requests per second per base.
            return result
        except urllib.error.HTTPError as error:
            # No blind mutation retries; next invocation checks the remote state first.
            raise RuntimeError(f"Airtable HTTP {error.code}; operation remains pending. Reconcile before retrying.") from None

    def check_schema(self):
        actual = self.request("GET", f"meta/bases/{self.contract.base_id}/tables")
        expected = {t["id"]: {f["id"]: (f["name"], f["type"]) for f in t["fields"]} for t in self.contract.tables.values()}
        got = {t["id"]: {f["id"]: (f["name"], f["type"]) for f in t["fields"]} for t in actual["tables"]}
        if got != expected:
            raise ValueError("Remote schema changed: reconcile the schema before syncing")
        names = {t["id"]: t["name"] for t in self.contract.tables.values()}
        for table in actual["tables"]:
            if table["name"] != names[table["id"]]:
                raise ValueError("Remote table name changed: schema migration required")
            for field in table["fields"]:
                if field["type"] == "multipleRecordLinks":
                    l, reverse = self.contract.link_fields[field["id"]]
                    target = self.contract.tables[l["from"] if reverse else l["to"]]["id"]
                    if field["options"]["linkedTableId"] != target:
                        raise ValueError("Remote linked-table target changed")
                elif field["type"] == "singleSelect":
                    local = self.contract.fields(table["id"])[field["id"]]
                    if {c["name"] for c in field["options"]["choices"]} != {c["name"] for c in local["config"]["choices"]}:
                        raise ValueError("Remote choices changed: schema migration required")

    def snapshot(self):
        self.check_schema()
        out = {"format": "airtable-record-snapshot-v1", "base_id": self.contract.base_id, "complete": True, "tables": {}}
        for t in self.contract.tables.values():
            records = []
            offset = None
            while True:
                params = {"pageSize": 100, "returnFieldsByFieldId": "true"}
                if offset:
                    params["offset"] = offset
                page = self.request("GET", f"{self.contract.base_id}/{t['id']}", params=params)
                records.extend(page["records"])
                offset = page.get("offset")
                if not offset:
                    break
            out["tables"][t["id"]] = records
        return out

    def apply(self, payload):
        args = payload["arguments"]
        return self.request("PATCH", f"{args['baseId']}/{args['tableId']}", {k: args[k] for k in ("records", "performUpsert", "typecast")})


def sync_operation(mirror, operation_id, remote):
    before = remote.snapshot()
    state = mirror.preflight(operation_id, before)
    if state != "already_applied":
        remote.apply(mirror.payload(operation_id))
        after = remote.snapshot()
    else:
        after = before
    mirror.import_snapshot(after)
    op = mirror.operation(operation_id)
    if op["status"] != "applied":
        raise ValueError("Remote readback does not match; write remains pending")
    return {"operation_id": operation_id, "status": "applied", "record_id": op["record_id"], "verified_snapshot_hash": op["verified_snapshot_hash"]}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    for command in ("status", "verify"):
        p = sub.add_parser(command); p.add_argument("--all-counts", action="store_true")
    p = sub.add_parser("query"); p.add_argument("sql"); p.add_argument("--limit", type=int, default=200)
    for command in ("import-snapshot", "export-snapshot"):
        p = sub.add_parser(command); p.add_argument("path", type=Path)
    p = sub.add_parser("stage"); p.add_argument("table"); p.add_argument("fields", type=Path); p.add_argument("--record-id")
    p = sub.add_parser("local-upsert"); p.add_argument("table"); p.add_argument("fields", type=Path); p.add_argument("--record-id"); p.add_argument("--snapshot", type=Path, default=ROOT / "data" / "airtable_snapshot.json")
    p = sub.add_parser("payload"); p.add_argument("operation_id")
    p = sub.add_parser("preflight"); p.add_argument("operation_id"); p.add_argument("snapshot", type=Path)
    p = sub.add_parser("sync"); p.add_argument("operation_id")
    sub.add_parser("pull")
    args = parser.parse_args(argv)
    mirror = Mirror(args.db)
    if args.command == "init":
        result = mirror.initialize()
    elif args.command == "status":
        result = mirror.status()
    elif args.command == "verify":
        result = mirror.verify()
    elif args.command == "query":
        result = mirror.query(args.sql, args.limit)
    elif args.command == "import-snapshot":
        result = mirror.import_snapshot(read_json(args.path))
    elif args.command == "export-snapshot":
        write_json(args.path, mirror.export_snapshot()); result = {"exported": str(args.path)}
    elif args.command == "stage":
        result = mirror.stage(args.table, read_json(args.fields), args.record_id)
    elif args.command == "local-upsert":
        result = mirror.local_upsert(args.table, read_json(args.fields), args.record_id)
        write_json(args.snapshot, mirror.export_snapshot())
        result["snapshot"] = str(args.snapshot)
    elif args.command == "payload":
        result = mirror.payload(args.operation_id)
    elif args.command == "preflight":
        result = {"status": mirror.preflight(args.operation_id, read_json(args.snapshot))}
    elif args.command == "pull":
        result = mirror.import_snapshot(AirtableREST(mirror.contract).snapshot())
    else:
        result = sync_operation(mirror, args.operation_id, AirtableREST(mirror.contract))
    if isinstance(result, dict) and not getattr(args, "all_counts", False):
        result.pop("table_counts", None)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
