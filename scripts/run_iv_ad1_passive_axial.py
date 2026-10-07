"""Run the sequential IV-A1/A2 -> IV-D1 prerequisite without active models."""

import argparse
import faulthandler
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.giada_teacher.hh_family_transfer import write
    from src.giada_teacher.iv_ad1_passive_axial import run
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='iv_ad1_passive_axial_preregistration.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f'Existing output cannot be overwritten: {output}')
    output.mkdir(parents=True)
    with (output / 'native_fault.log').open('w') as log:
        faulthandler.enable(file=log, all_threads=True)
        try:
            revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            report = run(ROOT, output / 'experiment', revision, args.config)
            write(output / 'process_status.json', {'returncode': 0})
            print('[GIADA IV-A/D1] '+json.dumps({
                'valid': report['valid'], 'diagnosis': report['diagnosis'],
                'iv_a1_a2_passed': report['iv_a1_a2_passed'],
                'iv_d1_passed': report['iv_d1_passed'],
                'task36_authorized': report['task36_authorized']}), flush=True)
        except Exception as exc:
            write(output / 'failure_report.json', {'valid': False,
                'scientific_no_go': False, 'error': str(exc),
                'traceback': traceback.format_exc()})
            write(output / 'process_status.json', {'returncode': 1})
            raise


if __name__ == '__main__':
    main()
