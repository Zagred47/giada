"""Run Task35 native and frozen CUDA phases in isolated processes."""

import argparse
import faulthandler
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.giada_teacher.hh_family_transfer import write
    from src.giada_teacher.task35_full_local_compartment import evaluate_frozen, run_native

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--model-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    teacher = Path(args.teacher)
    if args.model_only:
        if not (output / 'premodel_report.json').is_file():
            raise RuntimeError('Task35 premodel report missing')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ('model_fault.log' if args.model_only else 'native_fault.log')).open('w') as log:
        faulthandler.enable(file=log, all_threads=True)
        try:
            if args.model_only:
                report = evaluate_frozen(ROOT, teacher, output)
            else:
                revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
                report = run_native(ROOT, teacher, output, revision)
                if report['diagnosis'] == 'READY_FOR_FROZEN_MODEL_EVALUATION':
                    subprocess.run([sys.executable, str(Path(__file__)), '--model-only',
                                    '--output', str(output), '--teacher', str(teacher)], check=True)
                    report = json.loads((output / 'final_report.json').read_text())
            write(output / 'process_status.json', {'returncode': 0})
            print('[GIADA Task35] ' + json.dumps({
                'valid': report['valid'], 'diagnosis': report['diagnosis'],
                'floor_passed': report.get('floor_passed'),
                'model_judged': report['model_judged'],
                'task35_passed': report['task35_passed']}), flush=True)
        except Exception as error:
            write(output / 'failure_report.json', {'valid': False,
                'scientific_no_go': False, 'error': str(error),
                'traceback': traceback.format_exc()})
            write(output / 'process_status.json', {'returncode': 1})
            raise


if __name__ == '__main__':
    main()
