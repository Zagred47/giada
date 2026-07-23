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


class AuditHelpersTest(unittest.TestCase):
    def test_snapshot_branch_samples_explicitly_after_restore(self):
        calls = []

        class FakeHoc:
            t = 10.0

            @classmethod
            def continuerun(cls, target):
                calls.append(("continuerun", target))
                cls.t = target

        class FakeNetCon:
            @staticmethod
            def event(target):
                calls.append(("event", target))

        class FakeNumpy:
            @staticmethod
            def arange(start, stop, step):
                del stop, step
                return [start, start + 0.025, start + 15.0]

            @staticmethod
            def asarray(values, dtype=None):
                del dtype
                return list(values)

        class FakeSegment:
            v = -76.0

        session = object.__new__(TeacherAuditSession)
        session.h = FakeHoc()
        session.np = FakeNumpy()
        session.representatives = {"soma": 0}
        session.live_segments = {0: FakeSegment()}
        session._seed_neuron = lambda: calls.append("seed")

        result = session._snapshot_branch(FakeNetCon(), 10.0)

        self.assertEqual(result["time_ms"], [10.0, 10.025, 25.0])
        self.assertEqual(result["soma"], [-76.0, -76.0, -76.0])
        self.assertEqual(
            calls,
            [
                "seed",
                ("event", 11.0),
                ("continuerun", 10.025),
                ("continuerun", 25.0),
            ],
        )

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
