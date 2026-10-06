"""Isolated Kaggle runner for GIADA prerequisite IV-C1."""

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
    from src.giada_teacher.iv_c1_deterministic_synapse import load_contract, run
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fault = (output / 'python_fault.log').open('w', encoding='utf-8')
    faulthandler.enable(file=fault, all_threads=True)
    cfg = load_contract(ROOT)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    write(output / 'run_contract.json', cfg)
    sources = ('experiments/iv_c1_deterministic_synapse_preregistration.json',
               'src/giada_teacher/iv_c1_deterministic_synapse.py',
               'scripts/run_iv_c1_deterministic_synapse.py')
    write(output / 'code_provenance.json', {'code_revision': revision,
        'dirty_runtime': bool(subprocess.check_output(
            ['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
        'source_sha256': {name: sha(ROOT / name) for name in sources}})
    try:
        report = run(ROOT, Path(args.teacher), output, cfg, revision)
        write(output / 'process_status.json', {'returncode': 0})
        print('[GIADA IV-C1] ' + json.dumps({
            'valid': report['valid'], 'iv_c1_passed': report['iv_c1_passed'],
            'phase': report['selected_phase_steps'],
            'confirmation_count': len(report['confirmation'])}), flush=True)
    except Exception as error:
        write(output / 'failure_report.json', {'valid': False,
            'scientific_no_go': False, 'error': str(error),
            'traceback': traceback.format_exc()})
        write(output / 'process_status.json', {'returncode': 1})
        raise


if __name__ == '__main__':
    main()
