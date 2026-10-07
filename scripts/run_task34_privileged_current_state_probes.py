"""Execute Task34 frozen privileged probes in an isolated Kaggle process."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.giada_teacher.task34_privileged_current_state_probes import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    try:
        report = run(ROOT, args.output, code_revision=revision)
        print('[GIADA Task34] ' + json.dumps({key: report[key] for key in (
            'valid', 'diagnostic_complete', 'case_count',
            'parent_baseline_max_metric_delta',
            'current_identity_max_abs_error_ma_cm2')}), flush=True)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'failure_report.json').write_text(json.dumps({
            'valid': False, 'diagnosis': 'TECHNICAL_FAILURE',
            'code_revision': revision, 'error': repr(error),
            'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
