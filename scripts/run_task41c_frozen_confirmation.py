"""Isolated native/CUDA orchestration; immutable P110, no optimizer."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41c_frozen_confirmation as c

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--phase',choices=['native','model','smoke']);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    try:
        spec,_,_,_=c.contract(ROOT)
        if args.phase in ('model','smoke'):
            import torch
            torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            assert torch.cuda.is_available()
            with torch.inference_mode():report=c.smoke(torch,ROOT,output,'cuda') if args.phase=='smoke' else c.evaluate(torch,ROOT,output)
            print('[GIADA Task41c] '+json.dumps({k:report.get(k) for k in ('valid','decision','quality_passed')}),flush=True);return
        teacher=output.parent/'teacher_canonical'
        if args.phase=='native':c.native(ROOT,teacher,output);return
        assert not (output/'candidate_contract.json').exists(),'Confirmation output already exists'
        c.p.save_json(output/'preregistration.json',spec)
        c.p.save_json(output/'revision.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        c.p.save_json(output/'candidate_contract.json',dict(candidate='P110',checkpoint_sha256=c.p.digest(ROOT/spec['checkpoint']),protocol_sha256=c.p.digest(ROOT/c.CONFIG),fresh_accessed=False,training_performed=False,model_selection_performed=False))
        shutil.copy2(ROOT/spec['checkpoint'],output/'source_checkpoint.pt')
        print('[GIADA Task41c] checkpoint candidate frozen before fresh simulation',flush=True)
        if not teacher.exists():
            subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/SelfishGene/neuron_as_deep_net.git',str(teacher)],check=True)
            subprocess.run(['git','sparse-checkout','set','L5PC_NEURON_simulation/mods'],cwd=teacher,check=True)
        subprocess.run(['git','checkout','--detach',spec['teacher_revision']],cwd=teacher,check=True)
        subprocess.run([sys.executable,'-m','pip','install','--quiet','neuron=='+spec['neuron_version']],check=True)
        for phase in ('smoke','native','model'):subprocess.run([sys.executable,'-u',str(Path(__file__)),str(output),'--phase',phase],cwd=ROOT,check=True)
    except BaseException as exc:
        c.p.save_json(output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()))
        traceback.print_exc();raise
if __name__=='__main__':main()
