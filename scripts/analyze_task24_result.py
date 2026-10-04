"""Read-only scientific diagnosis; preserve verified JSON result artifacts."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / 'artifacts/giada_task24_sodium_family_4fad2c6_37e47d97.zip'
    destination = ROOT / 'experiments/results/task24_kaggle_4fad2c6'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze = json.loads(z.read('selection_freeze.json'))
        claimed = freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == claimed
        assert not freeze['fresh_accessed']
        for name, digest in freeze['checkpoint_hashes'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        for selected in freeze['selection'].values():
            name = selected['checkpoint']
            (destination / name).write_bytes(z.read(name))
        provenance = json.loads(z.read('code_provenance.json'))
        assert not provenance['dirty_runtime']
        for name, digest in provenance['sources'].items():
            assert hashlib.sha256(subprocess.check_output(['git', 'show', provenance['code_revision'] + ':' + name], cwd=ROOT)).hexdigest() == digest
        for name in z.namelist():
            if '/' not in name and name.endswith('.json'):
                target = destination / name
                if target.exists() and target.read_bytes() != z.read(name):
                    raise RuntimeError('Refuse to overwrite different evidence: ' + name)
                target.write_bytes(z.read(name))
    r = json.loads((destination / 'final_report.json').read_text())
    cfg = freeze['config']
    assert r['valid'] and not r['fresh_used_for_selection']
    assert json.loads((destination/'process_status.json').read_text())['returncode'] == 0
    assert all(json.loads((destination/name).read_text())['valid'] for name in ('native_audit.json','equivalence_preflight.json'))
    failures = []
    def check(row, domain, metrics, group, channel):
        for key, limit in cfg[group].items():
            if metrics[key] > limit:
                failures.append(dict(seed=row['seed'], family=row['family'], panel=domain,
                    channel=channel, metric=key, value=metrics[key], limit=limit))
        if not metrics.get('finite', True):
            failures.append(dict(seed=row['seed'], family=row['family'], panel=domain, channel=channel, metric='finite', value=False))
    for row in r['learned']:
        for domain in ('in_support','activation_boundary','state_extrema'):
            for ch in cfg['channels']:
                check(row,domain,row['metrics'][domain][ch],'gates',ch)
            check(row,domain,row['metrics'][domain]['pair'],'pair_gates','sum')
        for horizon, metrics in row['held_rollout'].items():
            for ch in cfg['channels']: check(row,'held-'+horizon,metrics[ch],'held_gates',ch)
            check(row,'held-'+horizon,metrics['pair'],'pair_gates','sum')
        for panel, metrics in row['path_rollout'].items():
            check(row,panel,metrics,'path_gates','aggregate')
            for ch in cfg['channels']: check(row,panel,metrics['per_channel'][ch],'path_channel_gates',ch)
    ladder = json.loads((destination/'development_ladder.json').read_text())
    compact = [dict(family=f, width=w, step=t, scores=[round(a['score'],5) for a in ladder if a['family']==f and a['width']==w and a['step']==t])
        for f in cfg['families'] for w in cfg['widths'] for t in cfg['checkpoints'] if t >= 5000]
    audit = dict(valid=True, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), CRC_valid=True,
                 checkpoint_and_source_hashes_valid=True, failures=failures, development_scores=compact,
                 interpretation='Post-hoc diagnosis only; original promotion decision and sealed outcome unchanged.')
    (destination/'result_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(valid=True, failures=failures, development_scores=compact)))


if __name__ == '__main__': main()
