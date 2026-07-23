import tempfile
import unittest
from pathlib import Path

from src.hayflow_teacher.audit import (
    detect_spikes,
    load_source_functions,
    repository_file_record,
    validate_parent_tree,
)
from src.hayflow_teacher.audit_runtime import TeacherAuditSession
from src.hayflow_schema import MorphologicalRegion


class AuditHelpersTest(unittest.TestCase):
    def test_snapshot_branch_samples_explicitly_after_restore(self):
        calls = []

        class FakeHoc:
            t = 10.0

        class FakeCVode:
            @staticmethod
            def solve(target):
                calls.append(("solve", target))
                FakeHoc.t = target

        class FakeNetCon:
            @staticmethod
            def event(target):
                calls.append(("event", target))

        class FakeNumpy:
            @staticmethod
            def linspace(start, stop, count):
                del count
                return [start, start + 0.025, start + 15.0]

            @staticmethod
            def asarray(values, dtype=None):
                del dtype
                return list(values)

        class FakeSegment:
            v = -76.0

        session = object.__new__(TeacherAuditSession)
        session.h = FakeHoc()
        session.cvode = FakeCVode()
        session.np = FakeNumpy()
        session.representatives = {"soma": 0}
        session.live_segments = {0: FakeSegment()}

        result = session._snapshot_branch(FakeNetCon(), 10.0)

        self.assertEqual(result["time_ms"], [10.0, 10.025, 25.0])
        self.assertEqual(result["soma"], [-76.0, -76.0, -76.0])
        self.assertEqual(
            calls,
            [
                ("event", 11.0),
                ("solve", 10.025),
                ("solve", 25.0),
            ],
        )

    def test_owned_rng_sequences_are_restored(self):
        class FakeRandom:
            def __init__(self, sequence):
                self.sequence = sequence

            def seq(self, value=None):
                if value is None:
                    return self.sequence
                self.sequence = value

        session = object.__new__(TeacherAuditSession)
        session.synapse_rngs = [FakeRandom(3), FakeRandom(7)]

        saved = session._snapshot_rng_sequences()
        session.synapse_rngs[0].seq(99)
        session.synapse_rngs[1].seq(101)
        session._restore_rng_sequences(saved)

        self.assertEqual(session._snapshot_rng_sequences(), [3.0, 7.0])

    def test_hot_zone_is_an_overlay_on_apical_regions(self):
        self.assertTrue(
            TeacherAuditSession._is_hot_zone(
                700.0, MorphologicalRegion.APICAL_TRUNK
            )
        )
        self.assertFalse(
            TeacherAuditSession._is_hot_zone(
                700.0, MorphologicalRegion.BASAL
            )
        )

    def test_owned_rng_uses_random123_and_canonical_distribution(self):
        calls = []

        class FakeRandom:
            def Random123(self, first, second, third):
                calls.append(("Random123", first, second, third))

            def negexp(self, mean):
                calls.append(("negexp", mean))

        class FakeHoc:
            @staticmethod
            def Random():
                return FakeRandom()

        class FakePointProcess:
            @staticmethod
            def setRNG(rng):
                calls.append(("setRNG", rng))

        session = object.__new__(TeacherAuditSession)
        session.h = FakeHoc()
        session.seed = 1729
        session.synapse_rngs = []

        rng = session._bind_owned_rng(FakePointProcess(), 12)

        self.assertEqual(calls[0], ("Random123", 1729, 12, 0))
        self.assertEqual(calls[1], ("negexp", 1.0))
        self.assertEqual(calls[2], ("setRNG", rng))
        self.assertEqual(session.synapse_rngs, [rng])

    def test_selected_source_functions_do_not_execute_top_level(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "generator.py"
            source.write_text(
                "raise RuntimeError('top level executed')\n"
                "def canonical(value):\n"
                "    return OFFSET + value\n",
                encoding="utf-8",
            )
            functions, provenance = load_source_functions(
                source,
                ["canonical"],
                {"OFFSET": 4},
            )

        self.assertEqual(functions["canonical"](3), 7)
        self.assertEqual(provenance["function_names"], ["canonical"])

    def test_parent_tree_requires_one_root_and_no_cycles(self):
        report = validate_parent_tree({0: None, 1: 0, 2: 1})
        self.assertEqual(report["root_id"], 0)
        self.assertTrue(report["acyclic"])

        with self.assertRaisesRegex(ValueError, "exactly one root"):
            validate_parent_tree({0: None, 1: None})
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_parent_tree({0: 1, 1: 0})

    def test_spikes_are_upward_crossings(self):
        self.assertEqual(
            detect_spikes([0, 1, 2, 3], [-70, -10, 10, -70]),
            [1.0],
        )

    def test_repository_record_is_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            source = repository / "sub" / "file.txt"
            source.parent.mkdir()
            source.write_text("teacher", encoding="utf-8")
            record = repository_file_record(source, repository)

        self.assertEqual(record["path"], "sub/file.txt")
        self.assertEqual(len(record["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
