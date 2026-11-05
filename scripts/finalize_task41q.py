"""Small lossless41q bundle, one SQLite batch, rotated backup and push."""
import json,hashlib,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
FOLDER=ROOT/'experiments/results/task41q_audit_v1'

def main():
    started=time.time()
    def state(s,**kw):write_json(FOLDER/'workflow.json',dict(state=s,elapsed_seconds=time.time()-started,**kw))
    def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    try:
        report=json.loads((FOLDER/'audit.json').read_text());assert report['valid'] and report['complete']
        loss=report['full_training_objective'];floor=report['fixed_rate_objective_lower_bound'];small=min(x['relative_ridge'] for x in report['direct_fits'])
        direct=next(x for x in report['direct_fits'] if x['relative_ridge']==small and x['arm']=='joint_nap')
        base=np.mean(report['coefficient_errors']['baseline']['validation']['equilibrium_rmse'],0)
        repaired=np.mean(report['coefficient_errors']['joint_nap/4000']['validation']['equilibrium_rmse'],0)
        fitted=np.mean(direct['validation']['equilibrium_rmse'],0)
        floor_fraction=float(np.mean(floor['equilibrium_total_per_seed'])/np.mean(loss['12000'][0]));pg=np.array(floor['per_gate_per_seed'])
        summary=dict(task='41q',valid=True,posthoc=True,
            objective_conflict=dict(baseline_mean_full_objective=float(np.mean(loss['baseline'][0])),adam12000_mean_full_objective=float(np.mean(loss['12000'][0])),fixed_rate_lower_bound=float(np.mean(floor['equilibrium_total_per_seed'])),lower_bound_fraction_of_final_objective=floor_fraction,K_Pst_activation_fraction_of_lower_bound=float(pg[:,3].sum()/pg.sum()),interpretation='Most remaining41p loss cannot be removed by equilibrium heads while z stays frozen. Per-input optimum can bias equilibrium to compensate fixed-rate error; simply extending same training is not justified.'),
            same_feature_room=dict(diagnostic_ridge=small,all_ridges_reported=True,selection_performed=False,Im=dict(baseline_rmse=float(base[2]),adam4000_rmse=float(repaired[2]),direct_rmse=float(fitted[2])),SK_E2=dict(baseline_rmse=float(base[5]),adam4000_rmse=float(repaired[5]),direct_rmse=float(fitted[5])),original_objective_direct_mean=float(np.mean(direct['original_full_training_objective'])),interpretation='Actual float32 materialized heads on independent anchors show substantial unused same-feature accuracy for Im/SK. This refutes a blanket capacity explanation for these coefficients only, not all channels or timing.'),
            recommended_next_step='Do not expand architecture or extend the same Adam loss first. Evaluate saved direct-fit checkpoints diagnostically on GPU, and contrast fixed versus learnable K_Pst update/appropriately matched trajectory objective. Separate equilibrium accuracy from compensation of frozen z; protect old cases. No deployment or selection from41q.',
            limits=report['limits']+['Coefficient RMSE is dimensionless, not voltage error or behavioral pass count.','CPU direct fits have not passed original GPU stress or coupled trajectory gates.','Fixed-rate lower bound is for chosen41p synthetic training objective, not a lower bound on achievable neuron accuracy.'])
        write_json(FOLDER/'decision_summary.json',summary)
        files=[FOLDER/'audit.json',FOLDER/'decision_summary.json',ROOT/'experiments/task41q_preregistration_v1.json',ROOT/'scripts/audit_task41q.py',ROOT/'tests/test_task41q.py',*sorted(FOLDER.glob('direct_weights_*.npz'))]
        archive=FOLDER/'forensic_bundle.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for path in files:z.write(path,path.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for path in files:assert hashlib.sha256(z.read(path.relative_to(ROOT).as_posix())).digest()==hashlib.sha256(path.read_bytes()).digest()
        with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        write_json(FOLDER/'external_backup_receipt.json',dict(valid=True,runtime_independent=True,local=str(archive),sha256=sha,bytes=archive.stat().st_size,scope='Local CPU forensic bundle, no external compute runtime; small report and diagnostic weights versioned in Git. No Drive/Airtable.',members=[p.relative_to(ROOT).as_posix() for p in files]))
        state('recording_memory');run('-m','research_memory.register_task41q','--validate-only');run('-m','research_memory.register_task41q')
        state('local_database_backup');run('-m','research_memory.prepare_drive_backup');run('-m','research_memory.record_local_backup','--raw-sqlite','--scope','Task41q post-hoc CPU checkpoint audit and direct diagnostic fits; no new stochastic training or model release. Full compact SQLite plus compressed reports, two verified local bundles. No Drive/Airtable.')
        paths=['experiments/task41q_preregistration_v1.json','scripts/audit_task41q.py','scripts/finalize_task41q.py','tests/test_task41q.py','research_memory/register_task41q.py','research_memory/data/local_backup_receipt.json','research_memory/data/research_memory.sqlite.manifest.json']+[p.relative_to(ROOT).as_posix() for p in FOLDER.iterdir() if p.name!='workflow.json']
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines());assert staged<=set(paths)
        subprocess.run(['git','commit','-m','Audit41p objective floor and same-feature diagnostic fit capacity (41q)'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','codex/surrogate-validity-audit'],cwd=ROOT,check=True)
        backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());assert backup['valid']
        state('complete',diagnostic_only=True,backup_verified=True,DB_backup_verified=True)
    except BaseException as error:
        state('attention_required',error=repr(error),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
