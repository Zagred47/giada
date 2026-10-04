"""Immutable Task25 archive/source audit, with explicit conjunctive failures."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    archive=ROOT/'artifacts/giada_task25_heterogeneous_mechanisms_bdb75e6_9d7837eb.zip';dest=ROOT/'experiments/results/task25_kaggle_bdb75e6';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze=json.loads(z.read('composition_freeze.json'));claim=freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==claim
        assert not freeze['fresh_accessed'] and not freeze['retraining_performed']
        provenance=json.loads(z.read('code_provenance.json'));assert not provenance['dirty_runtime']
        for name,digest in provenance['sources'].items():assert hashlib.sha256(subprocess.check_output(['git','show',provenance['code_revision']+':'+name],cwd=ROOT)).hexdigest()==digest
        for row in freeze['sources'].values():assert hashlib.sha256(subprocess.check_output(['git','show',provenance['code_revision']+':'+row['path']],cwd=ROOT)).hexdigest()==row['sha256']
        for name in z.namelist():
            if '/' not in name and name.endswith('.json'):
                target=dest/name
                if target.exists() and target.read_bytes()!=z.read(name):raise RuntimeError('Different evidence: '+name)
                target.write_bytes(z.read(name))
        fresh_sha=hashlib.sha256(z.read('fresh.npz')).hexdigest()
    report=json.loads((dest/'final_report.json').read_text());cfg=freeze['config']
    assert report['valid'] and not report['fresh_used_for_selection'] and not report['retraining_performed']
    assert json.loads((dest/'process_status.json').read_text())['returncode']==0 and json.loads((dest/'native_audit.json').read_text())['valid']
    failures=[]
    def check(row,panel,metrics,gates,current_gates):
        for channel in cfg['channels']:
            for metric,limit in cfg[gates].items():
                value=metrics[channel][metric]
                if value>limit:failures.append(dict(arm=row['arm'],seed=row['seed'],panel=panel,channel=channel,metric=metric,value=value,limit=limit))
            assert metrics[channel]['finite']
        for metric,limit in cfg[current_gates].items():
            value=metrics['currents'][metric]
            if value>limit:failures.append(dict(arm=row['arm'],seed=row['seed'],panel=panel,channel='currents',metric=metric,value=value,limit=limit))
    for row in report['learned']:
        for panel,metrics in row['metrics'].items():check(row,panel,metrics,'gates','current_gates')
        for panel,metrics in row['held_rollout'].items():check(row,'held-'+panel,metrics,'held_gates','path_current_gates')
        for speed,paths in row['path_rollout'].items():
            for path,metrics in paths.items():check(row,'path-'+speed+'-'+path,metrics,'path_gates','path_current_gates')
    ratio=min(r['identity_rmse_ratio'] for r in report['learned'])
    audit=dict(valid=True,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),CRC_valid=True,checkpoint_and_source_hashes_valid=True,
        fresh_npz_sha256=fresh_sha,failures=failures,minimum_identity_rmse_ratio=ratio,
        interpretation='Valid frozen composition; independent reference and calcium compression pass3/3. Sodium compression failures do not invalidate the primary reference or retroactively alter Task24b.')
    (dest/'result_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(valid=True,failures=failures,minimum_identity_rmse_ratio=ratio,passing=report['per_arm_passed'])))


if __name__=='__main__':main()
