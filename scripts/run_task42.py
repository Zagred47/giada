"""Isolated native and GPU worker phases; durable partial results on errors."""
import os,sys,json,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task42_axial_exploration as x

def main():
    source,out=map(Path,sys.argv[1:3]);phase=sys.argv[3] if len(sys.argv)>3 else 'driver'
    out.mkdir(exist_ok=True)
    try:
        if phase=='native':
            revision=json.loads((ROOT/'experiments/task35_full_local_compartment_preregistration.json').read_text())['teacher_revision']
            teacher=ROOT.parent/'neuron_as_deep_net'
            if not teacher.exists():
                raise RuntimeError('Pinned teacher checkout must be staged by notebook: '+str(teacher))
            assert subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()==revision
            x.native(ROOT,teacher,out)
        elif phase=='gpu':
            import torch
            torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
            prior=x.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.cuda.get_device_name()==prior['gpu'] and torch.__version__==prior['torch']
            with torch.inference_mode():x.evaluate(torch,ROOT,source,out)
        else:
            x.n.p.save_json(out/'preregistration.json',x.spec(ROOT)[0])
            for part in ('native','gpu'):subprocess.run([sys.executable,'-u',__file__,str(source),str(out),part],check=True,cwd=ROOT)
    except BaseException as error:
        x.n.p.save_json(out/'failure_report.json',dict(valid=False,technical_failure=True,phase=phase,error=repr(error),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
