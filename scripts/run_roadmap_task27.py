"""Task27 supervisor: separate native and GPU workers, bounded notebook log."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from src.giada_teacher.current_supervision_comparison import run,config
    from src.giada_teacher.heterogeneous_mechanism_composition import native_audit
    from src.giada_teacher.hh_family_transfer import write,sha
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['native','gpu'])
    args=parser.parse_args()
    cfg=config(ROOT);output=Path(args.output)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if args.phase=='native':native_audit(args.teacher,output);return
    if args.phase=='gpu':run(ROOT,output,cfg,revision);return
    output.mkdir(parents=True,exist_ok=False)
    files=['src/giada_teacher/'+name+'.py' for name in
           ('current_supervision_comparison','joint_heterogeneous_comparison',
            'heterogeneous_mechanism_composition','calcium_pair_composition',
            'sodium_family_composition','hh_family_transfer',
            'joint_gate_symmetric_confirmation','gpu_baseline_runtime')]
    files += ['scripts/run_roadmap_task27.py','experiments/task27_current_supervision.json',cfg['parent_contract']]
    files += [cfg[key] for key in ('parent_run','parent_freeze','parent_process')]
    write(output/'run_contract.json',cfg)
    write(output/'code_provenance.json',dict(code_revision=revision,
        dirty_runtime=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),
        sources={f:sha(ROOT/f) for f in files}))
    try:
        for phase in ('native','gpu'):
            print('[GIADA Task27] phase '+phase,flush=True)
            with (output/'process.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u',__file__,'--teacher',args.teacher,'--output',args.output,'--phase',phase],
                                        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in worker.stdout:
                    log.write(line);log.flush()
                    if line.startswith('[GIADA Task27]'):print(line.strip(),flush=True)
                code=worker.wait()
            write(output/'process_status.json',dict(phase=phase,returncode=code))
            if code:raise RuntimeError(f'{phase} worker exit={code}; inspect process.log')
    except Exception as error:
        write(output/'failure_report.json',dict(valid=False,scientific_no_go=False,error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
