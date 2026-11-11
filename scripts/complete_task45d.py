"""Single45d launch after proofs; supervise, persist external output and local DB."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task45d_protocol_v1'
def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
def main():
    def state(name,**kw):write_json(F/'workflow.json',dict(state=name,**kw))
    try:
        for name in ('preregistration_receipt.json','input_verified.json','local_preflight.json'):assert json.loads((F/name).read_text())['valid']
        for file,digest in json.loads((F/'local_preflight.json').read_text())['source_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
        state('preregistration_local_backup');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','45d finalized preregistration,6linked hypotheses and original quality gates. No45d scientific result yet. SQLite and raw reports local only; two verified generations retained.')
        initial=['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*initial],cwd=ROOT,check=True);assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(initial)
        subprocess.run(['git','commit','-m','Verify local45d preregistration DB backup before launch'],cwd=ROOT,check=True);subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        state('launching_once');run('scripts/task45d_kaggle.py','launch');state('monitoring');run('scripts/task45d_kaggle.py','supervise')
        state('recording_results');run('-m','research_memory.register_task45d','--result','--validate-only');run('-m','research_memory.register_task45d','--result')
        state('local_DB_backup');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','45d factorial and exact/diagnostic execution output with prior limits. SQLite and raw reports local only; external final/partial SHA/CRC verified.')
        files=['experiments/results/task45d_protocol_v1/'+n for n in ('kaggle_launch.json','result_memory_receipt.json')]+['research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())<=set(files)
        subprocess.run(['git','commit','-m','Record45d output and verified local DB backup'],cwd=ROOT,check=True);subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        receipt=json.loads((ROOT/'artifacts/task45d_kaggle_external/external_backup_receipt.json').read_text());state('complete',technical_failure=receipt['technical_failure'],external_backup_verified=True,DB_backup_verified=True)
    except BaseException as e:state('attention_required',error=repr(e),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
