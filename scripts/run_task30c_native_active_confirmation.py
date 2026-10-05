"""Run Task30c in an isolated process, retaining failures as artifacts."""
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
    from src.giada_teacher.task30c_native_active_confirmation import run
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--teacher', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg_path = ROOT / 'experiments/task30c_native_active_confirmation.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    write(output / 'run_contract.json', cfg)
    sources = ('src/giada_teacher/task30c_native_active_confirmation.py',
               'src/giada_teacher/task30_autonomous_voltage.py',
               'experiments/task30c_native_active_confirmation.json',
               'scripts/run_task30c_native_active_confirmation.py')
    write(output / 'code_provenance.json', {
        'code_revision': revision,
        'dirty_runtime': bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
        'source_sha256': {name: sha(ROOT / name) for name in sources}})
    try:
        report = run(ROOT, Path(args.teacher), output, cfg, revision)
        write(output / 'process_status.json', {'returncode': 0})
        print('[GIADA Task30c] ' + json.dumps({
            'valid': report['valid'], 'native_floor': report['native_floor'],
            'scientific_primary_passed': report['scientific_primary_passed'],
            'task31_authorized': report['task31_authorized']}), flush=True)
    except Exception as error:
        write(output / 'failure_report.json', {
            'valid': False, 'scientific_no_go': False,
            'error': str(error), 'traceback': traceback.format_exc()})
        write(output / 'process_status.json', {'returncode': 1})
        raise


if __name__ == '__main__':
    main()
