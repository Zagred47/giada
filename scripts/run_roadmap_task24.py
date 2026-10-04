"""Task24 supervisor: isolated native and CUDA workers, bounded output."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['native','gpu']);a=p.parse_args()
    from src.giada_teacher.sodium_family_composition import native_audit,run
    from src.giada_teacher.hh_family_transfer import write,sha
    config=json.loads((ROOT/'experiments/task24_sodium_family_composition.json').read_text());output=Path(a.output)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    prior=json.loads((ROOT/'experiments/results/task23_kaggle_0f79081/final_report.json').read_text())
    if not(prior['valid'] and prior['task24_preparation_authorized'] and not prior['fresh_used_for_selection']):raise RuntimeError('verified Task23 independent reference required')
    if a.phase=='native':native_audit(a.teacher,output);return
    if a.phase=='gpu':run(output,config,revision);return
    output.mkdir(parents=True,exist_ok=False)
    files=('src/giada_teacher/sodium_family_composition.py','src/giada_teacher/hh_family_transfer.py','src/giada_teacher/joint_gate_symmetric_confirmation.py','src/giada_teacher/gpu_baseline_runtime.py','experiments/task24_sodium_family_composition.json','scripts/run_roadmap_task24.py','experiments/results/task23_kaggle_0f79081/final_report.json')
    write(output/'run_contract.json',config)
    write(output/'code_provenance.json',dict(code_revision=revision,dirty_runtime=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),sources={f:sha(ROOT/f) for f in files}))
    try:
        for phase in ('native','gpu'):
            print('[GIADA Task24] phase '+phase,flush=True)
            with (output/'process.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u',__file__,'--teacher',a.teacher,'--output',a.output,'--phase',phase],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in worker.stdout:
                    log.write(line);log.flush()
                    if line.startswith('[GIADA Task24]'):print(line.strip(),flush=True)
                code=worker.wait()
            write(output/'process_status.json',dict(phase=phase,returncode=code))
            if code:raise RuntimeError(f'{phase} worker exit={code}; see process.log')
    except Exception as e:
        write(output/'failure_report.json',dict(valid=False,scientific_no_go=False,error=str(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
