"""No Airtable writes: fixtures live only in temporary SQLite databases."""
import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research_memory.mirror import Contract, Mirror, sync_operation


def rid(n):
    return "rec" + str(n).zfill(14)


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = Contract()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mirror = Mirror(Path(self.tmp.name) / "test.sqlite", self.contract)
        self.mirror.initialize()
        self.snapshot = {"format": "airtable-record-snapshot-v1", "base_id": self.contract.base_id, "complete": True, "tables": {t["id"]: [] for t in self.contract.tables.values()}}
        self.mirror.import_snapshot(self.snapshot)

    def add(self, key, n, values):
        record = {"id": rid(n), "createdTime": "2026-09-19T00:00:00Z", "fields": self.contract.normalize(key, values)}
        self.snapshot["tables"][self.contract.tables[key]["id"]].append(record)
        return record

    def link(self, source, target, key, name):
        link = next(l for l in self.contract.links if l["from"] == key and l["name"] == name)
        source["fields"][link["fieldId"]].append(target["id"])
        target["fields"][link["inverseFieldId"]].append(source["id"])
        return link

    def test_schema_and_empty_roundtrip(self):
        report = self.mirror.verify()
        self.assertTrue(report["valid"])
        self.assertEqual((report["tables"], report["relationships"], report["records"]), (78, 241, 0))
        self.assertEqual(self.mirror.query('SELECT * FROM "06 · Specifiche dei modelli"')["rows"], [])
        self.assertFalse(self.mirror.initialize()["created"])

    def test_links_bidirectional_and_independent_order(self):
        s1 = self.add("sources", 1, {"Nome": "A", "Codice stabile": "S1"})
        s2 = self.add("sources", 2, {"Nome": "B", "Codice stabile": "S2"})
        a1 = self.add("actors", 3, {"Nome": "Uno"})
        a2 = self.add("actors", 4, {"Nome": "Due"})
        self.link(s2, a1, "sources", "Autori")
        self.link(s1, a2, "sources", "Autori")
        self.link(s1, a1, "sources", "Autori")
        self.mirror.import_snapshot(self.snapshot)
        self.assertTrue(self.mirror.verify()["valid"])
        row = self.mirror.query('SELECT "Autori" FROM "01 · Fonti" WHERE "Codice stabile"=\'S1\'')["rows"][0]
        import json
        self.assertEqual(json.loads(row["Autori"]), [rid(4), rid(3)])
        self.assertEqual(len(self.mirror.query("SELECT * FROM v_semantic_edges")["rows"]), 3)

    def test_self_link_roundtrip(self):
        a = self.add("artifacts", 1, {"Nome": "Derived"})
        b = self.add("artifacts", 2, {"Nome": "Original"})
        self.link(a, b, "artifacts", "Artefatti di origine")
        self.mirror.import_snapshot(self.snapshot)
        self.assertTrue(self.mirror.verify()["valid"])

    def test_inconsistent_snapshot_rejected_without_damage(self):
        source = self.add("sources", 1, {"Nome": "Source"})
        self.mirror.import_snapshot(self.snapshot)
        original = self.mirror.export_snapshot()
        fid = self.mirror.field_id("sources", "Autori")
        source["fields"][fid] = [rid(99)]
        with self.assertRaises(ValueError):
            self.mirror.import_snapshot(self.snapshot)
        self.assertEqual(self.mirror.export_snapshot(), original)

    def test_partial_or_wrong_base_rejected(self):
        bad = copy.deepcopy(self.snapshot)
        bad["tables"].pop(next(iter(bad["tables"])))
        with self.assertRaises(ValueError):
            self.mirror.import_snapshot(bad)
        bad = copy.deepcopy(self.snapshot); bad["complete"] = False
        with self.assertRaises(ValueError):
            self.mirror.import_snapshot(bad)
        bad = copy.deepcopy(self.snapshot); bad["base_id"] = "wrong"
        with self.assertRaises(ValueError):
            self.mirror.import_snapshot(bad)

    def test_select_objects_and_numeric_roundtrip(self):
        record = self.add("sources", 1, {"Nome": "Paper", "Tipo": {"id": "selDummy", "name": "Paper"}})
        self.add("observations", 2, {"Nome": "Metric", "Valore": 1})
        self.mirror.import_snapshot(self.snapshot)
        self.assertTrue(self.mirror.verify()["valid"])
        self.assertEqual(self.mirror.query('SELECT "Tipo" FROM sources')["rows"][0]["Tipo"], "Paper")

    def test_stage_does_not_publish_speculative_data(self):
        op = self.mirror.stage("sources", {"Nome": "Draft", "Codice stabile": "SRC-1"})
        self.assertEqual(self.mirror.status()["records"], 0)
        self.assertEqual(self.mirror.status()["pending_writes"], 1)
        payload = self.mirror.payload(op["operation_id"])
        self.assertIn("performUpsert", payload["arguments"])
        with self.assertRaises(ValueError):
            self.mirror.stage("sources", {"Nome": "Again", "Codice stabile": "SRC-1"})

    def test_local_upsert_publishes_and_maintains_reciprocal_links(self):
        actor = self.mirror.local_upsert(
            "actors", {"Nome": "Local actor", "Codice stabile": "ACT-LOCAL-1"}
        )
        source = self.mirror.local_upsert(
            "sources",
            {
                "Nome": "Local source",
                "Codice stabile": "SRC-LOCAL-1",
                "Autori": [actor["record_id"]],
            },
        )
        self.assertTrue(self.mirror.verify()["valid"])
        self.assertEqual(self.mirror.status()["records"], 2)
        edge = self.mirror.query(
            "SELECT source_id,target_id FROM v_links "
            f"WHERE source_id='{source['record_id']}' AND target_id='{actor['record_id']}'"
        )["rows"]
        self.assertEqual(len(edge), 1)
        updated = self.mirror.local_upsert(
            "sources", {"Nome": "Renamed", "Codice stabile": "SRC-LOCAL-1"}
        )
        self.assertEqual(updated["record_id"], source["record_id"])
        self.assertEqual(self.mirror.query("SELECT Nome FROM sources")["rows"][0]["Nome"], "Renamed")

    def test_successful_dual_write_and_multiple_updates(self):
        remote = FakeRemote(self)
        op = self.mirror.stage("sources", {"Nome": "A", "Codice stabile": "SRC-1"})
        result = sync_operation(self.mirror, op["operation_id"], remote)
        self.assertEqual(result["status"], "applied")
        for name in ("B", "C"):
            update = self.mirror.stage("sources", {"Nome": name}, result["record_id"])
            sync_operation(self.mirror, update["operation_id"], remote)
        self.assertEqual(self.mirror.status()["records"], 1)
        self.assertEqual(self.mirror.status()["pending_writes"], 0)
        self.assertTrue(self.mirror.verify()["valid"])

    def test_timeout_after_remote_write_is_idempotently_recovered(self):
        remote = FakeRemote(self); remote.fail_after_write = True
        op = self.mirror.stage("sources", {"Nome": "A", "Codice stabile": "SRC-1"})
        with self.assertRaises(RuntimeError):
            sync_operation(self.mirror, op["operation_id"], remote)
        self.assertEqual(self.mirror.status()["pending_writes"], 1)
        self.assertEqual(self.mirror.status()["records"], 0)
        remote.fail_after_write = False
        sync_operation(self.mirror, op["operation_id"], remote)
        self.assertEqual(remote.apply_count, 1)
        self.assertEqual(self.mirror.status()["records"], 1)

    def test_remote_conflict_blocks_overwrite(self):
        record = self.add("sources", 1, {"Nome": "A", "Codice stabile": "SRC-1"})
        self.mirror.import_snapshot(self.snapshot)
        op = self.mirror.stage("sources", {"Nome": "B"}, rid(1))
        record["fields"][self.mirror.field_id("sources", "Descrizione")] = "external edit"
        with self.assertRaises(ValueError):
            self.mirror.preflight(op["operation_id"], self.snapshot)
        self.assertEqual(self.mirror.status()["pending_writes"], 1)

    def test_invalid_fields_links_and_nonfinite_numbers(self):
        for values in ({"Bogus": 3}, {"Tipo": "Bogus"}, {"Autori": [rid(99)]}):
            with self.assertRaises(ValueError):
                self.mirror.stage("sources", {"Nome": "A", "Codice stabile": "S", **values})
        with self.assertRaises(ValueError):
            self.mirror.stage("observations", {"Nome": "A", "Codice stabile": "O", "Valore": float("nan")})

    def test_query_cannot_mutate_or_attach(self):
        for sql in ("DELETE FROM sources", "DROP TABLE sources", "PRAGMA query_only=OFF", "ATTACH DATABASE ':memory:' AS other"):
            with self.assertRaises(sqlite3.Error):
                self.mirror.query(sql)
        self.assertTrue(self.mirror.verify()["valid"])

    def test_detect_direct_sql_corruption(self):
        self.add("sources", 1, {"Nome": "A"})
        self.mirror.import_snapshot(self.snapshot)
        with self.mirror.connect() as conn:
            conn.execute('UPDATE sources SET "Nome"=\'Tampered\'')
        self.assertFalse(self.mirror.verify()["valid"])

    def test_schema_fingerprint_drift_fails_closed(self):
        with self.mirror.connect() as conn:
            conn.execute("UPDATE _mirror_meta SET value='changed' WHERE key='schema_fingerprint'")
        with self.assertRaises(ValueError):
            self.mirror.status()

    def test_all_documented_query_patterns_execute(self):
        path = Path(__file__).resolve().parents[1] / "research_memory/queries/lenses.sql"
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("--")]
        for sql in "\n".join(lines).split(";"):
            if sql.strip():
                self.assertIn("rows", self.mirror.query(sql))

    def test_wrong_readback_stays_pending(self):
        remote = FakeRemote(self)
        op = self.mirror.stage("sources", {"Nome": "Expected", "Codice stabile": "SRC-1"})
        remote.apply = lambda payload: None
        with self.assertRaises(ValueError):
            sync_operation(self.mirror, op["operation_id"], remote)
        self.assertEqual(self.mirror.status()["pending_writes"], 1)

    def test_link_clear_updates_both_views(self):
        source = self.add("sources", 1, {"Nome": "A"})
        actor = self.add("actors", 2, {"Nome": "B"})
        link = self.link(source, actor, "sources", "Autori")
        self.mirror.import_snapshot(self.snapshot)
        source["fields"][link["fieldId"]] = []
        actor["fields"][link["inverseFieldId"]] = []
        self.mirror.import_snapshot(self.snapshot)
        self.assertEqual(self.mirror.query("SELECT * FROM v_links")["rows"], [])
        self.assertTrue(self.mirror.verify()["valid"])

    def test_duplicate_remote_code_blocks_sync(self):
        op = self.mirror.stage("sources", {"Nome": "A", "Codice stabile": "DUP"})
        self.add("sources", 1, {"Nome": "A", "Codice stabile": "DUP"})
        self.add("sources", 2, {"Nome": "A", "Codice stabile": "DUP"})
        with self.assertRaises(ValueError):
            self.mirror.preflight(op["operation_id"], self.snapshot)

    def test_fk_constraint_rejects_dangling_sql_link(self):
        l = self.contract.links[0]
        with self.mirror.connect() as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(f"INSERT INTO {l['sql_table']} VALUES (?,?,0,0)", (rid(1), rid(2)))


class FakeRemote:
    def __init__(self, case):
        self.case = case
        self.fail_after_write = False
        self.apply_count = 0

    def snapshot(self):
        return copy.deepcopy(self.case.snapshot)

    def apply(self, payload):
        self.apply_count += 1
        args = payload["arguments"]
        patch = args["records"][0]
        key = self.case.contract.by_id[args["tableId"]]
        records = self.case.snapshot["tables"][args["tableId"]]
        if "id" in patch:
            record = next(r for r in records if r["id"] == patch["id"])
            record["fields"].update(patch["fields"])
        else:
            self.case.add(key, 100 + self.apply_count, patch["fields"])
        if self.fail_after_write:
            raise RuntimeError("Simulated lost response AFTER successful remote commit")


if __name__ == "__main__":
    unittest.main()
