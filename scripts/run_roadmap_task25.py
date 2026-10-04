"""Task25 supervisor: separate native and CUDA interpreters, bounded stdout."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from src.giada_teacher.heterogeneous_mechanism_composition import run,native_audit
    from src.giada_teacher.hh_family_transfer import write,sha
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['native','gpu']);a=p.parse_args()
    cfg=json.loads((ROOT/'experiments/task25_heterogeneous_mechanisms.json').read_text());output=Path(a.output)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if a.phase=='native':native_audit(a.teacher,output);return
    if a.phase=='gpu':run(ROOT,output,cfg,revision);return
    output.mkdir(parents=True,exist_ok=False)
    files=['src/giada_teacher/'+f+'.py' for f in ('heterogeneous_mechanism_composition','calcium_pair_composition','sodium_family_composition','hh_family_transfer','gpu_baseline_runtime')]
    files+=['experiments/task25_heterogeneous_mechanisms.json','scripts/run_roadmap_task25.py']
    for source in ('calcium_source','sodium_source','sodium_shared_source'):
        files += [p.relative_to(ROOT).as_posix() for p in (ROOT/cfg[source]).iterdir() if p.suffix in ('.json','.pt')]
    write(output/'code_provenance.json',dict(code_revision=revision,dirty_runtime=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),sources={f:sha(ROOT/f) for f in files}))
    write(output/'run_contract.json',cfg)
    try:
        for phase in ('native','gpu'):
            print('[GIADA Task25] phase '+phase,flush=True)
            with (output/'process.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u',__file__,'--teacher',a.teacher,'--output',a.output,'--phase',phase],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in worker.stdout:
                    log.write(line);log.flush()
                    if line.startswith('[GIADA Task25]'):print(line.strip(),flush=True)
                code=worker.wait()
            write(output/'process_status.json',dict(phase=phase,returncode=code))
            if code:raise RuntimeError(f'{phase} worker exit={code}; inspect process.log')
    except Exception as error:
        write(output/'failure_report.json',dict(valid=False,scientific_no_go=False,error=str(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
