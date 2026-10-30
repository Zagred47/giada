"""Strict phase isolation: GPU eligibility freeze -> native CPU -> fresh GPU."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41n_independent_confirmation as n

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--phase',choices=['preflight','native','evaluate'],required=True);a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    try:
        spec,*_=n.contract(ROOT)
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        if a.phase=='native':
            freeze=json.loads((out/'execution_freeze.json').read_text());assert freeze['fresh_accessed'] is False
            n.native(ROOT,out.parent/'teacher_canonical',out);return
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        assert torch.cuda.is_available() and torch.cuda.get_device_name()==spec['gpu'] and torch.__version__==spec['torch']
        with torch.inference_mode():
            if a.phase=='preflight':
                assert not (out/'execution_freeze.json').exists() and not (out/'native_shadow_traces.npz').exists()
                report=n.preflight(torch,ROOT,out,'cuda');assert report['valid']
                n.p.save_json(out/'preregistration.json',spec);n.p.save_json(out/'revision.json',dict(revision=revision,dirty=False))
                n.p.save_json(out/'execution_freeze.json',dict(protocol_sha256=n.p.digest(ROOT/n.CONFIG),revision=revision,eligible=report['eligible'],fresh_accessed=False,training_updates=0,selected_on_fresh=False))
                print('[GIADA41n] execution eligibility frozen '+str(report['eligible']),flush=True)
            else:
                assert not (out/'final_report.json').exists(),'Evaluation already complete; do not repeat'
                report=n.evaluate(torch,ROOT,out,'cuda');n.p.save_json(out/'final_report.json',report)
                print('[GIADA41n] '+report['decision'],flush=True)
    except BaseException as error:
        n.p.save_json(out/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,phase=a.phase,error=repr(error),traceback=traceback.format_exc(),training_updates_performed=0));raise

if __name__=='__main__':main()
