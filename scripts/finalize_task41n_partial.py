"""Finite serialized local finalization after the launch registration/backup."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'experiments/results/task41n_audited_v1'
def main():
    start=time.time()
    def status(state,**kw):
        temp=FOLDER/'finalization.tmp';temp.write_text(json.dumps(dict(state=state,elapsed_seconds=time.time()-start,**kw),indent=2));temp.replace(FOLDER/'finalization.json')
    def command(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    try:
        status('waiting_for_launch_writer_and_backup')
        while True:
            row=json.loads((ROOT/'experiments/results/task41n_protocol_v1/dispatch_workflow.json').read_text())
            if row['state']=='dispatch_complete':break
            if row['state']=='attention_required':raise RuntimeError('Launch workflow needs inspection; no concurrent DB writer started')
            if time.time()-start>1800:raise TimeoutError('Serialized launch writer deadline exceeded; no DB write attempted')
            time.sleep(15)
        status('registering_stopped_confirmation')
        command('-m','research_memory.register_task41n_partial','--validate-only')
        command('-m','research_memory.register_task41n_partial')
        status('local_only_database_backup')
        command('-m','research_memory.prepare_drive_backup')
        command('-m','research_memory.record_local_backup','--raw-sqlite','--scope','Full41m and41n independent partial result: exact GPU/imposed parity,528/528trajectories336/336responses per arm; originalFP32canonicalfloor failed2/288autonomouschecks; matchedmixed64passed; model autonomous/stress/cost not judged. External final165036864bytearchive verified. No threshold change,Airtable orDrive access.')
        receipt=json.loads((FOLDER/'memory_receipt.json').read_text());receipt['DB_backup']=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text())
        (FOLDER/'memory_receipt.json').write_text(json.dumps(receipt,indent=2))
        files=['experiments/results/task41n_audited_v1/memory_receipt.json','research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines());assert staged<=set(files),staged
        subprocess.run(['git','commit','-m','Register stopped41n and verified local-only full database backup'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        status('complete',verification=receipt['verification'],DB_backup=receipt['DB_backup'])
    except BaseException as error:
        status('attention_required',error=repr(error),traceback=traceback.format_exc(),automatic_restart=False);raise
if __name__=='__main__':main()
