"""Isolated Task32 runner: preserve diagnostic failures and compact notebook output."""

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
    from src.giada_teacher.hh_family_transfer import sha, write
    from src.giada_teacher.task32_dynamic_calcium_feedback import (
        evaluate_frozen_models_only, load_v3_contract, run)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--model-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if args.model_only:
        if not output.is_dir() or not (output / 'premodel_floor.json').is_file():
            raise RuntimeError('Task32 isolated model worker requires native premodel floor')
    else:
        output.mkdir(parents=True, exist_ok=False)
    fault_log = (output / 'python_fault.log').open('w', encoding='utf-8')
    faulthandler.enable(file=fault_log, all_threads=True)
    cfg = load_v3_contract(ROOT)
    if args.model_only:
        floor = json.loads((output / 'premodel_floor.json').read_text())
        if not floor['native_floor_admissible']:
            raise RuntimeError('Task32 model worker cannot bypass native floor')
        evaluate_frozen_models_only(ROOT, Path(args.teacher), output, cfg)
        return
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    write(output / 'run_contract.json', cfg)
    sources = ('src/giada_teacher/task32_dynamic_calcium_feedback.py',
               'src/giada_teacher/iv_b_calcium_prerequisite.py',
               'src/giada_teacher/task30_autonomous_voltage.py',
               'src/giada_teacher/task30c_native_active_confirmation.py',
               'experiments/task32_dynamic_calcium_feedback_v2.json',
               'experiments/task32_dynamic_calcium_feedback_v3.json',
               'scripts/run_task32_dynamic_calcium_feedback.py')
    write(output / 'code_provenance.json', {
        'code_revision': revision,
        'dirty_runtime': bool(subprocess.check_output(
            ['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
        'source_sha256': {name: sha(ROOT / name) for name in sources}})
    try:
        report = run(ROOT, Path(args.teacher), output, cfg, revision)
        write(output / 'process_status.json', {'returncode': 0})
        print('[GIADA Task32] ' + json.dumps({
            'valid': report['valid'],
            'native_floor_admissible': report['native_floor_admissible'],
            'scientific_primary_passed': report['scientific_primary_passed'],
            'source_hypothesis': report['source_hypothesis_frozen_before_run'],
            'model_count': sum(len(family) for family in report['models'].values())}), flush=True)
    except Exception as error:
        write(output / 'failure_report.json', {
            'valid': False, 'scientific_no_go': False,
            'error': str(error), 'traceback': traceback.format_exc()})
        write(output / 'process_status.json', {'returncode': 1})
        raise


if __name__ == '__main__':
    main()
