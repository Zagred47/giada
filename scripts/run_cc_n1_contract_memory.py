"""Run CC-N1 phases with native and CUDA isolated and durable failure reports."""
import argparse
import faulthandler
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.giada_teacher.cc_n1_contract_memory import (
        CONFIG, analyze_memory, contract, finalize, gpu_phase, native_phase, sha)
    from src.giada_teacher.hh_family_transfer import write
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'native', 'gpu', 'memory'), default='all')
    args = parser.parse_args()
    output = args.output
    if args.phase in ('all', 'memory'):
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise RuntimeError('Phase output root missing')
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    with (output / f'{args.phase}_fault.log').open('w') as fault:
        faulthandler.enable(file=fault, all_threads=True)
        try:
            spec, *_ = contract(ROOT)
            if args.phase == 'all':
                if args.teacher is None: raise ValueError('--teacher required')
                shutil.copy2(ROOT / CONFIG, output / 'preregistration.json')
                write(output / 'contract.json', spec['cc01_contract'])
                write(output / 'code_provenance.json', {'git_revision': revision,
                    'preregistration_sha256': sha(ROOT / CONFIG), 'source_sha256': spec['sources']})
                for phase in ('native', 'gpu'):
                    write(output / 'progress.json', {'phase': phase, 'code_revision': revision})
                    command = [sys.executable, str(Path(__file__)), '--phase', phase,
                               '--output', str(output), '--teacher', str(args.teacher)]
                    completed = subprocess.run(command, cwd=ROOT, check=False)
                    if completed.returncode:
                        raise RuntimeError(f'CC-N1 {phase} subprocess exit={completed.returncode}; inspect {phase}_failure.json and fault/log files')
                report = finalize(ROOT, output, revision)
                write(output / 'process_status.json', {'returncode': 0, 'diagnostic_complete': report['valid']})
                print('[GIADA CC-N1] '+json.dumps({k: report[k] for k in
                      ('valid', 'diagnosis', 'n2_preparation_recommended', 'training_performed')}), flush=True)
            elif args.phase == 'native':
                native_phase(ROOT, args.teacher, output, revision)
            elif args.phase == 'gpu':
                gpu_phase(ROOT, output, revision)
            else:
                report = analyze_memory(ROOT, output)
                print(json.dumps({'valid': report['valid'], 'episode_count': report['episode_count'],
                                  'instantaneous_collision_detected': report['instantaneous_collision_detected']}))
        except Exception as error:
            failure = {'valid': False, 'scientific_no_go': False, 'phase': args.phase,
                       'error': str(error), 'traceback': traceback.format_exc(), 'code_revision': revision}
            write(output / f'{args.phase}_failure.json', failure)
            if args.phase == 'all':
                write(output / 'failure_report.json', failure)
                write(output / 'process_status.json', {'returncode': 1})
            raise


if __name__ == '__main__':
    main()
