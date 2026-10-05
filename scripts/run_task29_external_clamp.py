"""Run Task29 in a bounded-output process; preserve failure artifacts."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.giada_teacher.hh_family_transfer import sha, write
    from src.giada_teacher.task29_external_clamp import config, run, verify_prerequisites
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = config(ROOT)
    verify_prerequisites(ROOT, cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    write(output / 'run_contract.json', cfg)
    sources = ('src/giada_teacher/task29_external_clamp.py',
               'src/giada_teacher/ionic_block_teacher_forced.py',
               'experiments/task29_scientific_track_amendment.json',
               'experiments/task29_external_clamp_ionic_passive.json',
               'scripts/run_task29_external_clamp.py')
    write(output / 'code_provenance.json', {
        'code_revision': revision,
        'dirty_runtime': bool(subprocess.check_output(
            ['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
        'source_sha256': {name: sha(ROOT / name) for name in sources}})
    try:
        report = run(ROOT, output, cfg, revision)
        write(output / 'process_status.json', {'returncode': 0})
        print('[GIADA Task29] ' + json.dumps({
            'valid': report['valid'],
            'diagnostic_scientific_passed': report['diagnostic_scientific_passed'],
            'gate_d_completed': report['gate_d_completed'],
            'task30_autonomous_voltage_authorized': report['task30_autonomous_voltage_authorized']}), flush=True)
    except Exception as error:
        write(output / 'failure_report.json', {
            'valid': False, 'scientific_no_go': False,
            'error': str(error), 'traceback': traceback.format_exc()})
        write(output / 'process_status.json', {'returncode': 1})
        raise


if __name__ == '__main__':
    main()
