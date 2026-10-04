"""Task18c supervisor: native/CUDA isolation, logs and failure reports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--teacher',required=True);parser.add_argument('--output',required=True);parser.add_argument('--phase',choices=['native','gpu']);args=parser.parse_args()
    from src.giada_teacher.hh_family_transfer import native_audit,write,sha
    from src.giada_teacher.hh_potassium_local_repair import run
    output=Path(args.output);config=json.loads((ROOT/'experiments/task18c_potassium_local_repair.json').read_text())
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    prior=json.loads((ROOT/'experiments/results/task18_kaggle_a094fa9/final_report.json').read_text())
    if not prior['valid'] or prior['per_channel_transfer']!={'Ca_HVA':True,'NaTa_t':True,'K_Pst':False}:raise RuntimeError('Task18 prerequisite mismatch')
    if args.phase=='native':native_audit(args.teacher,output);return
    if args.phase=='gpu':run(output,config,revision,ROOT/'experiments/fixtures/task18b_frozen_parent.zip');return
    output.mkdir(parents=True,exist_ok=False)
    write(output/'run_contract.json',config)
    write(output/'code_provenance.json',{'code_revision':revision,'dirty_runtime':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),'sources':{f:sha(ROOT/f) for f in ('src/giada_teacher/hh_potassium_local_repair.py','src/giada_teacher/hh_family_transfer.py','experiments/task18c_potassium_local_repair.json','scripts/run_roadmap_task18c.py')}})
    try:
        for phase in ('native','gpu'):
            print('[GIADA Task18c] phase '+phase,flush=True)
            with (output/'process.log').open('a') as log:
                worker=subprocess.Popen([sys.executable,'-u',__file__,'--teacher',args.teacher,'--output',str(output),'--phase',phase],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in worker.stdout:
                    log.write(line);log.flush()
                    if line.startswith('[GIADA Task18c]'):print(line.strip(),flush=True)
                code=worker.wait()
            write(output/'process_status.json',{'phase':phase,'returncode':code})
            if code:raise RuntimeError(f'{phase} worker exit={code}; see process.log')
    except Exception as exc:
        write(output/'failure_report.json',{'valid':False,'scientific_no_go':False,'error':str(exc),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
