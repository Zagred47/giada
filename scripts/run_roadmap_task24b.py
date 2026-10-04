"""Task24b supervisor; NEURON and GPU always separate processes."""
import argparse
import subprocess
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from src.giada_teacher.sodium_control_diagnosis import config, run
    from src.giada_teacher.sodium_family_composition import native_audit
    from src.giada_teacher.hh_family_transfer import write,sha
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['native','gpu']);a=p.parse_args()
    cfg=config(ROOT);output=Path(a.output);revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if a.phase=='native':native_audit(a.teacher,output);return
    if a.phase=='gpu':run(ROOT,output,cfg,revision);return
    output.mkdir(parents=True,exist_ok=False)
    files=('src/giada_teacher/sodium_control_diagnosis.py','src/giada_teacher/sodium_family_composition.py','src/giada_teacher/hh_family_transfer.py','src/giada_teacher/joint_gate_symmetric_confirmation.py','src/giada_teacher/gpu_baseline_runtime.py','experiments/task24b_sodium_control_diagnosis.json','experiments/task24_sodium_family_composition.json','scripts/run_roadmap_task24b.py')
    sources={f:sha(ROOT/f) for f in files}
    for f in (ROOT/cfg['source_result']).glob('checkpoint_*.pt'):sources[f.relative_to(ROOT).as_posix()]=sha(f)
    for name in ('selection_freeze.json','final_report.json','result_audit.json'):sources[cfg['source_result']+'/'+name]=sha(ROOT/cfg['source_result']/name)
    write(output/'run_contract.json',cfg);write(output/'code_provenance.json',dict(code_revision=revision,dirty_runtime=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),sources=sources))
    try:
        for phase in ('native','gpu'):
            print('[GIADA Task24b] phase '+phase,flush=True)
            with (output/'process.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u',__file__,'--teacher',a.teacher,'--output',a.output,'--phase',phase],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in worker.stdout:
                    log.write(line);log.flush()
                    if line.startswith('[GIADA Task24b]'):print(line.strip(),flush=True)
                code=worker.wait()
            write(output/'process_status.json',dict(phase=phase,returncode=code))
            if code:raise RuntimeError(f'{phase} worker exit={code}; see process.log')
    except Exception as e:
        write(output/'failure_report.json',dict(valid=False,scientific_no_go=False,error=str(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
