"""IE-P7 native/GPU isolation, freeze-before-generation and durable failures."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def main():
    from src.giada_teacher import ie_native_operational_confirmation as p7
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--teacher',type=Path);parser.add_argument('--phase',choices=('all','native','gpu','rng'),default='all');args=parser.parse_args()
    output=args.output;revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if args.phase=='rng':p7.rng_server(ROOT,output);return
    if args.phase=='all':
        output.mkdir(parents=True,exist_ok=False);spec,*_=p7.contract(ROOT)
        shutil.copy2(ROOT/p7.CONFIG,output/'preregistration.json');(output/'code_revision.txt').write_text(revision)
        p7.write(output/'freeze.json',{'code_revision':revision,'checkpoint_sha256':spec['checkpoint_sha256'],'preregistration_sha256':p7.sha(ROOT/p7.CONFIG),'fresh_accessed':False,'training_performed':False,'members':[(k,s) for k in spec['knots'] for s in spec['model_seeds']]})
    with (output/f'{args.phase}_fault.log').open('w') as log:
        faulthandler.enable(log,all_threads=True)
        try:
            if args.phase=='all':
                for phase in ('native','gpu'):
                    command=[sys.executable,'-u',str(Path(__file__)),'--phase',phase,'--output',str(output),'--teacher',str(args.teacher)]
                    env={**os.environ}
                    if phase=='native':env['CUDA_VISIBLE_DEVICES']=''
                    completed=subprocess.run(command,cwd=ROOT,env=env,check=False)
                    if completed.returncode:raise RuntimeError(f'IE-P7 {phase} subprocess exit={completed.returncode}; inspect phase failure report')
                print('[GIADA IE-P7] isolated phases complete',flush=True)
            elif args.phase=='native':p7.native_phase(ROOT,args.teacher,output)
            else:
                server=[sys.executable,'-u',str(Path(__file__)),'--phase','rng','--output',str(output)]
                p7.gpu_phase(ROOT,output,revision,server)
        except Exception as error:
            failure={'valid':False,'scientific_no_go':False,'phase':args.phase,'error':str(error),'traceback':traceback.format_exc(),'code_revision':revision}
            p7.write(output/f'{args.phase}_failure.json',failure);p7.write(output/'failure_report.json',failure)
            raise

if __name__=='__main__':main()
