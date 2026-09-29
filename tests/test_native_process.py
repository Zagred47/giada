"""The supervisor must survive failures that Python try/except cannot catch."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from src.giada_teacher.native_process import supervise_native


class NativeProcessTests(unittest.TestCase):
    def run_worker(self, body):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "result"
        command = [sys.executable, "-c", body, str(output)]
        report, code = supervise_native(command, output)
        self.assertTrue((output / "process.log").is_file())
        self.assertTrue((output / "process_status.json").is_file())
        return output, report, code

    def test_silent_nonzero_exit_produces_failure_report(self):
        output, report, code = self.run_worker("import os; os._exit(6)")
        self.assertEqual(code, 6)
        self.assertFalse(report["valid"])
        self.assertFalse(report["gate_c_authorized"])
        self.assertTrue((output / "failure_report.json").is_file())

    @unittest.skipUnless(os.name == "posix", "POSIX native signal semantics")
    def test_sigsegv_produces_parent_report(self):
        _, report, code = self.run_worker(
            "import os,signal,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
            "os.kill(os.getpid(),signal.SIGSEGV)"
        )
        self.assertEqual(code, -11)
        self.assertEqual(report["process"]["signal"], "SIGSEGV")
        self.assertFalse(report["valid"])

    def test_zero_exit_without_report_is_not_success(self):
        _, report, code = self.run_worker("print('finished but report missing')")
        self.assertEqual(code, 0)
        self.assertFalse(report["valid"])
        self.assertEqual(report["error_type"], "MissingFinalReport")

    def test_successful_report_is_preserved_and_output_cannot_be_overwritten(self):
        output, report, code = self.run_worker(
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
            "(p/'final_report.json').write_text('{\"valid\":true,\"gate_c_authorized\":false}')"
        )
        self.assertTrue(report["valid"])
        self.assertEqual(code, 0)
        with self.assertRaises(FileExistsError):
            supervise_native([sys.executable, "-c", "raise Exception()"], output)

    def test_late_abort_never_authorizes_existing_final_report(self):
        _, report, code = self.run_worker(
            "import pathlib,sys,os; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
            "(p/'final_report.json').write_text('{\"valid\":true}'); os._exit(6)"
        )
        self.assertFalse(report["valid"])
        self.assertEqual(code, 6)

    def test_phase_and_python_error_are_retained(self):
        output, report, _ = self.run_worker(
            "import pathlib,sys,os; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
            "pathlib.Path(os.environ['GIADA_NATIVE_PHASE_PATH']).write_text('{\"phase\":\"restore\"}'); "
            "(p/'failure_report.json').write_text('{\"valid\":false,\"error\":\"specific failure\"}'); "
            "os._exit(1)"
        )
        self.assertEqual(report["error"], "specific failure")
        self.assertEqual(report["process"]["last_phase"]["phase"], "restore")
        self.assertTrue((output / "last_phase.json").is_file())


if __name__ == "__main__":
    unittest.main()
