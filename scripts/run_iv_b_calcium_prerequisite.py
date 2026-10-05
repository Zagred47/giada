"""Run the IV-B1/B2 prerequisite audit in an isolated Kaggle process."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.giada_teacher.hh_family_transfer import sha, write
    from src.giada_teacher.iv_b_calcium_prerequisite import run
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--teacher', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'experiments/iv_b1_b2_calcium_prerequisite.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    write(output / 'run_contract.json', config)
    files = ('experiments/iv_b1_b2_calcium_prerequisite.json',
             'src/giada_teacher/iv_b_calcium_prerequisite.py',
             'scripts/run_iv_b_calcium_prerequisite.py')
    write(output / 'code_provenance.json', {
        'code_revision': revision,
        'dirty_runtime': bool(subprocess.check_output(
            ['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
        'source_sha256': {name: sha(ROOT / name) for name in files}})
    try:
        report = run(Path(args.teacher), output, config, revision)
        write(output / 'process_status.json', {'returncode': 0})
        print('[GIADA IV-B] ' + json.dumps({key: report[key] for key in (
            'iv_b1_valid', 'iv_b2_valid', 'task32_feedback_authorized')}), flush=True)
    except Exception as error:
        write(output / 'failure_report.json', {
            'valid': False, 'scientific_no_go': False,
            'error': str(error), 'traceback': traceback.format_exc()})
        write(output / 'process_status.json', {'returncode': 1})
        raise


if __name__ == '__main__':
    main()
