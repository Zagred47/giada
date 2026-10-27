"""Pinned Task41k notebook/console worker; bounded diagnostic output."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41k_learning_alignment as i

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume-archive',type=Path);ap.add_argument('--resume-sha256');ap.add_argument('--resume-local',action='store_true');ap.add_argument('--stop-after',type=int,choices=[64,256,1024]);ap.add_argument('--evaluation-only',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        i.p.save_json(args.output/'revision.json',dict(revision=revision,dirty_runtime=False))
        if args.resume_archive:
            import zipfile
            assert i.p.digest(args.resume_archive)==args.resume_sha256
            with zipfile.ZipFile(args.resume_archive) as archive:
                assert archive.testzip() is None
                for name in archive.namelist():assert (args.output/name).resolve().is_relative_to(args.output.resolve())
                archive.extractall(args.output)
        report=i.run(ROOT,args.output,smoke=args.smoke,resume=bool(args.resume_archive) or args.resume_local,stop_after=args.stop_after,evaluation_only=args.evaluation_only)
        i.p.save_json(args.output/'final_report.json',report)
        print('[GIADA Task41k] '+report.get('decision','GPU_PREFLIGHT_ONLY'),flush=True)
    except BaseException as exc:
        from src.giada_teacher.task41i_evaluation import EvaluationPause
        if isinstance(exc,EvaluationPause):
            i.p.save_json(args.output/'final_report.json',dict(valid=True,decision='FROZEN_EVALUATION_STAGE_COMPLETE',training_complete=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False))
            print('[GIADA Task41k] frozen evaluation stage safely paused',flush=True);return
        i.p.save_json(args.output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
