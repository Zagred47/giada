"""Run bounded-output compiled Gate D bottleneck diagnostic on CUDA."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from src.giada_teacher.gate_d_bottleneck_forensic import config,run,verify_parent
    from src.giada_teacher.hh_family_transfer import sha,write
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    cfg=config(ROOT);verify_parent(ROOT,cfg)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    write(output/'run_contract.json',cfg)
    files=['src/giada_teacher/gate_d_bottleneck_forensic.py','src/giada_teacher/gate_d_compute_audit.py',
           'src/giada_teacher/gate_d_compiled_confirmation.py',
           'experiments/task28d_gate_d_bottleneck_forensic.json','scripts/run_gate_d_bottleneck_forensic.py']
    write(output/'code_provenance.json',{'code_revision':revision,
        'dirty_runtime':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),
        'sources':{name:sha(ROOT/name) for name in files},'parent_hashes':cfg['parent_sha256']})
    try:
        report=run(ROOT,output,cfg,revision)
        write(output/'process_status.json',{'phase':'gpu','returncode':0})
        print('[GIADA Gate D forensic] '+json.dumps({'valid':report['valid'],
              'diagnostic_only':report['diagnostic_only'],
              'task29_authorized':report['task29_authorized']}),flush=True)
    except Exception as error:
        write(output/'failure_report.json',{'valid':False,'scientific_no_go':False,
              'error':str(error),'traceback':traceback.format_exc()})
        write(output/'process_status.json',{'phase':'gpu','returncode':1})
        raise


if __name__=='__main__':main()
