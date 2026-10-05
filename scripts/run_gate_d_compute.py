"""Bounded-output Gate D CUDA benchmark; scientific NO-GO is a valid result."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from src.giada_teacher.gate_d_compute_audit import config,run,verify_parent
    from src.giada_teacher.hh_family_transfer import sha,write
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    cfg=config(ROOT)
    verify_parent(ROOT,cfg)
    output.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    write(output/'run_contract.json',cfg)
    files=['src/giada_teacher/gate_d_compute_audit.py','src/giada_teacher/ionic_block_teacher_forced.py',
           'experiments/task28b_gate_d_compute.json','scripts/run_gate_d_compute.py']
    write(output/'code_provenance.json',{'code_revision':revision,
        'dirty_runtime':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),
        'sources':{name:sha(ROOT/name) for name in files},
        'parent_hashes':cfg['task28_result_sha256']})
    try:
        report=run(ROOT,output,cfg,revision)
        write(output/'process_status.json',{'phase':'gpu','returncode':0})
        print('[GIADA Gate D] '+json.dumps({'valid':report['valid'],
              'gate_d_completed':report['gate_d_completed'],
              'task29_authorized':report['task29_authorized']}),flush=True)
    except Exception as error:
        write(output/'failure_report.json',{'valid':False,'scientific_no_go':False,
              'error':str(error),'traceback':traceback.format_exc()})
        write(output/'process_status.json',{'phase':'gpu','returncode':1})
        raise


if __name__=='__main__':main()
