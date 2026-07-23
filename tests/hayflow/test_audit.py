import tempfile
import unittest
from pathlib import Path

from src.hayflow_teacher.audit import (
    detect_spikes,
    load_source_functions,
    repository_file_record,
    validate_parent_tree,
)


class AuditHelpersTest(unittest.TestCase):
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
