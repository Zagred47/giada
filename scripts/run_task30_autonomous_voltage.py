"""Execute Task30 with bounded stdout and recoverable diagnostic artifacts."""
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
    from src.giada_teacher.task30_autonomous_voltage import config, run, verify_parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config-file', default='task30_autonomous_voltage_microcanary.json',
                        choices=('task30_autonomous_voltage_microcanary.json',
                                 'task30b_active_exposure_confirmation.json'))
    args = parser.parse_args()
    cfg = config(ROOT, args.config_file)
    verify_parent(ROOT, cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    write(output / 'run_contract.json', cfg)
    sources = ('src/giada_teacher/task30_autonomous_voltage.py',
               'src/giada_teacher/ionic_block_teacher_forced.py',
               'experiments/' + args.config_file,
               'scripts/run_task30_autonomous_voltage.py')
    write(output / 'code_provenance.json', {
        'code_revision': revision,
        'dirty_runtime': bool(subprocess.check_output(
            ['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),
        'source_sha256': {name: sha(ROOT / name) for name in sources}})
    try:
        report = run(ROOT, output, cfg, revision)
        write(output / 'process_status.json', {'returncode': 0})
        print('[GIADA Task30] ' + json.dumps({
            'valid': report['valid'],
            'scientific_primary_passed': report['scientific_primary_passed'],
            'gate_d_performance_status': report['gate_d_performance_status'],
            'task31_authorized': report['task31_authorized']}), flush=True)
    except Exception as error:
        write(output / 'failure_report.json', {
            'valid': False, 'scientific_no_go': False,
            'error': str(error), 'traceback': traceback.format_exc()})
        write(output / 'process_status.json', {'returncode': 1})
        raise


if __name__ == '__main__':
    main()
