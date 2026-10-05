"""Task28 supervisor: separate native and frozen-GPU workers, bounded output."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.giada_teacher.ionic_block_teacher_forced import config, native_audit, run, verify_parent
    from src.giada_teacher.hh_family_transfer import sha, write
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=['native', 'gpu'])
    args = parser.parse_args()
    cfg = config(ROOT)
    output = Path(args.output)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if args.phase == 'native':
        native_audit(Path(args.teacher), output, cfg)
        return
    if args.phase == 'gpu':
        run(ROOT, output, cfg, revision)
        return
    verify_parent(ROOT, cfg)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'run_contract.json', cfg)
    files = ['src/giada_teacher/ionic_block_teacher_forced.py',
             'src/giada_teacher/current_supervision_comparison.py',
             'src/giada_teacher/heterogeneous_mechanism_composition.py',
             'src/giada_teacher/joint_heterogeneous_comparison.py',
             'src/giada_teacher/hh_family_transfer.py',
             'src/giada_teacher/single_gate_transfer.py',
             'src/giada_teacher/calcium_gate_transfer.py',
             'experiments/task28_ionic_block_teacher_forced.json',
             'experiments/teacher_mechanism_inventory_v1.json',
             'scripts/run_roadmap_task28.py']
    write(output / 'code_provenance.json', {
        'code_revision': revision,
        'dirty_runtime': bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
        'sources': {name: sha(ROOT / name) for name in files},
        'parent_hashes': cfg['parent_sha256'],
    })
    try:
        for phase in ('native', 'gpu'):
            print('[GIADA Task28] phase ' + phase, flush=True)
            with (output / 'process.log').open('a', encoding='utf-8') as log:
                worker = subprocess.Popen([sys.executable, '-u', __file__, '--teacher', args.teacher,
                                           '--output', args.output, '--phase', phase], cwd=ROOT,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in worker.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith('[GIADA Task28]'):
                        print(line.rstrip(), flush=True)
                code = worker.wait()
            write(output / 'process_status.json', {'phase': phase, 'returncode': code})
            if code:
                raise RuntimeError(f'{phase} worker exit={code}; inspect process.log')
    except Exception as error:
        write(output / 'failure_report.json', {
            'valid': False, 'scientific_no_go': False,
            'error': str(error), 'traceback': traceback.format_exc(),
        })
        raise


if __name__ == '__main__':
    main()
