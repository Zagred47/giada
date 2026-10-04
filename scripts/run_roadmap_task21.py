"""Task21 supervisor: independent native/CUDA workers and recoverable diagnostics."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=['native', 'gpu'])
    args = parser.parse_args()
    from src.giada_teacher.hh_family_transfer import write, sha
    from src.giada_teacher.slow_gate_transfer import native_audit, run
    output = Path(args.output)
    config = json.loads((ROOT / 'experiments/task21_slow_gate_transfer.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    prior = json.loads((ROOT / 'experiments/results/task20_kaggle_f94993f/final_report.json').read_text())
    if not (prior['valid'] and prior['task21_authorized'] and not prior['fresh_used_for_selection']):
        raise RuntimeError('Verified Task20 GO required')
    if args.phase == 'native':
        native_audit(args.teacher, output)
        return
    if args.phase == 'gpu':
        run(output, config, revision)
        return
    output.mkdir(parents=True, exist_ok=False)
    files = ('src/giada_teacher/slow_gate_transfer.py', 'src/giada_teacher/single_gate_transfer.py', 'src/giada_teacher/hh_family_transfer.py',
             'src/giada_teacher/gpu_baseline_runtime.py', 'src/giada_teacher/joint_gate_symmetric_confirmation.py',
             'experiments/task21_slow_gate_transfer.json', 'scripts/run_roadmap_task21.py',
             'experiments/results/task20_kaggle_f94993f/final_report.json')
    write(output / 'run_contract.json', config)
    write(output / 'code_provenance.json', dict(code_revision=revision,
        dirty_runtime=bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
        sources={f: sha(ROOT / f) for f in files}))
    try:
        for phase in ('native', 'gpu'):
            print('[GIADA Task21] phase ' + phase, flush=True)
            with (output / 'process.log').open('a') as log:
                worker = subprocess.Popen([sys.executable, '-u', __file__, '--teacher', args.teacher,
                    '--output', str(output), '--phase', phase], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in worker.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith('[GIADA Task21]'):
                        print(line.strip(), flush=True)
                code = worker.wait()
            write(output / 'process_status.json', dict(phase=phase, returncode=code))
            if code:
                raise RuntimeError(f'{phase} worker exit={code}; see process.log')
    except Exception as error:
        write(output / 'failure_report.json', dict(valid=False, scientific_no_go=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
