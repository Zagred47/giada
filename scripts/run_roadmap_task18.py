"""Keep NEURON native process separate from Torch/CUDA and notebook kernel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['native','gpu']);args=p.parse_args()
    from src.giada_teacher.hh_family_transfer import native_audit,run_training,write,sha
    output=Path(args.output)
    config=json.loads((ROOT/'experiments/task18_hh_family_transfer.json').read_text())
    if args.phase=='native':native_audit(args.teacher,output);return
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if args.phase=='gpu':run_training(output,config,revision);return
    output.mkdir(parents=True,exist_ok=False)
    write(output/'run_contract.json',config)
    write(output/'code_provenance.json',{'code_revision':revision,'dirty_runtime':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),'sources':{f:sha(ROOT/f) for f in ('src/giada_teacher/hh_family_transfer.py','experiments/task18_hh_family_transfer.json','scripts/run_roadmap_task18.py')}})
    try:
        for phase in ('native','gpu'):
            print('[GIADA Task18] phase '+phase,flush=True)
            with (output/'process.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u',__file__,'--teacher',args.teacher,'--output',str(output),'--phase',phase],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in worker.stdout:
                    log.write(line);log.flush()
                    if line.startswith('[GIADA Task18]'):print(line.strip(),flush=True)
                code=worker.wait()
            write(output/'process_status.json',{'phase':phase,'returncode':code})
            if code:raise RuntimeError(f'{phase} worker exit={code}; see process.log')
    except Exception as exc:
        write(output/'failure_report.json',{'valid':False,'scientific_no_go':False,'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
