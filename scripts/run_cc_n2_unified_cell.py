"""Isolated native/CUDA runner with durable technical failures."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from src.giada_teacher.cc_n2_unified_cell import CONFIG,contract,native_phase,gpu_phase
    from src.giada_teacher.hh_family_transfer import write
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('all','native','gpu'),default='all')
    args=parser.parse_args();output=args.output
    if args.phase=='all':output.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    with (output/f'{args.phase}_fault.log').open('w') as fault:
        faulthandler.enable(file=fault)
        try:
            contract(ROOT)
            if args.phase=='all':
                shutil.copy2(ROOT/CONFIG,output/'preregistration.json')
                for phase in ('native','gpu'):
                    write(output/'progress.json',{'phase':phase,'code_revision':revision})
                    result=subprocess.run([sys.executable,'-u',str(Path(__file__)),
                        '--teacher',str(args.teacher),'--output',str(output),'--phase',phase],cwd=ROOT)
                    if result.returncode:raise RuntimeError(f'{phase} exit={result.returncode}; inspect phase failure and fault log')
            elif args.phase=='native':native_phase(ROOT,args.teacher,output)
            else:gpu_phase(ROOT,output,revision)
        except Exception as error:
            failure={'valid':False,'scientific_no_go':False,'phase':args.phase,
                     'error':str(error),'traceback':traceback.format_exc(),'code_revision':revision}
            write(output/f'{args.phase}_failure.json',failure)
            if args.phase=='all':write(output/'failure_report.json',failure)
            raise


if __name__=='__main__':main()
