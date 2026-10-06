"""Run IV-C2 native audit in a separate process, preserving failure evidence."""

import argparse
import json
from pathlib import Path
import subprocess
import traceback

from src.giada_teacher.iv_c2_stochastic_release import run
from src.giada_teacher.hh_family_transfer import write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(
        ['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    try:
        report = run(args.repo, args.teacher, args.output, revision)
        print(json.dumps({k: report[k] for k in ('valid', 'case_count', 'iv_c2_passed',
                    'iv_c3_passed', 'task33_authorized')}), flush=True)
    except Exception as error:
        write(args.output / 'failure_report.json', {
            'valid': False, 'failure_kind': 'runtime_or_contract',
            'iv_c2_passed': False, 'task33_authorized': False,
            'error': repr(error), 'traceback': traceback.format_exc(),
            'code_revision': revision})
        raise


if __name__ == '__main__':
    main()
