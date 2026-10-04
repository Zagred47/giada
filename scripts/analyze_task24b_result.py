"""Verify immutable Task24b evidence and preserve exact source bytes."""
import hashlib
import json
import statistics
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT/'artifacts/giada_task24b_sodium_control_diagnosis_bde23cc_3803e8e7.zip'
    dest = ROOT/'experiments/results/task24b_kaggle_bde23cc'
    dest.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze = json.loads(z.read('selection_freeze.json'))
        claimed = freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==claimed
        assert not freeze['fresh_accessed']
        for name,digest in freeze['checkpoint_hashes'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
        provenance=json.loads(z.read('code_provenance.json'))
        assert not provenance['dirty_runtime']
        for name,digest in provenance['sources'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',provenance['code_revision']+':'+name],cwd=ROOT)).hexdigest()==digest
        names=[n for n in z.namelist() if '/' not in n and n.endswith('.json')]
        names += [row['checkpoint'] for row in freeze['selection'].values()]
        for name in set(names):
            target=dest/name
            if target.exists() and target.read_bytes()!=z.read(name):raise RuntimeError('Different evidence: '+name)
            target.write_bytes(z.read(name))
    report=json.loads((dest/'final_report.json').read_text())
    assert report['valid'] and not report['fresh_used_for_selection'] and report['original_task24_decision_unchanged']
    assert json.loads((dest/'process_status.json').read_text())['returncode']==0
    assert all(json.loads((dest/n).read_text())['valid'] for n in ('native_audit.json','equivalence_preflight.json'))
    probes=json.loads((dest/'gradient_probes.json').read_text())
    contrasts=json.loads((dest/'paired_contrasts.json').read_text())
    attribution=json.loads((dest/'frozen_rate_precision_attribution.json').read_text())
    summary={factor:statistics.median([r['improvement_fraction'] for r in contrasts if r['factor']==factor]) for factor in ('schedule','clipping','budget','width')}
    audit=dict(valid=True,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),CRC_valid=True,
        checkpoint_and_source_hashes_valid=True,paired_contrast_medians=summary,
        maximum_channel_clipping_frequency=max(x for p in probes for row in p['fraction_above_one'] for x in row),
        maximum_bundle_clipping_frequency=max(x for p in probes for x in p['bundle_fraction_above_one']),
        old_frozen_attribution=attribution,interpretation='Task24 original NO-GO retained; Task24b fresh confirmation distinct. No universal scaling or runtime claim.')
    (dest/'result_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='old_frozen_attribution'}))
    print(json.dumps(attribution))


if __name__=='__main__':main()
