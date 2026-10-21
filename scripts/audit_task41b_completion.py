"""Read-only scientific archive audit; persist compact local evidence."""
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
import zipfile
import torch

def main():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root/'artifacts/task41b_primary_external/external_backup_receipt.json').read_text())
    archive = Path(receipt['local'])
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert sha == receipt['remote']['sha256'] and receipt['remote']['label'] == 'final'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        read = lambda n: json.loads(z.read(n))
        final = read('final_report.json')
        freeze = read('selection_freeze.json')
        claimed = freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest() == claimed
        checkpoint_sha = hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()
        assert checkpoint_sha == freeze['checkpoint_sha256']
        finite = {}
        for name in z.namelist():
            if name.endswith('.pt'):
                obj = torch.load(io.BytesIO(z.read(name)), map_location='cpu', weights_only=False)
                tensors = []
                def walk(x):
                    if torch.is_tensor(x): tensors.append(x)
                    elif isinstance(x, dict):
                        for v in x.values(): walk(v)
                    elif isinstance(x, (list, tuple)):
                        for v in x: walk(v)
                walk(obj)
                finite[name] = {'tensors': len(tensors), 'finite': all(torch.isfinite(t).all().item() for t in tensors)}
        costs = []
        for row in read('cost.json'):
            costs.append({'batch_per_seed': row['batch_per_seed'], 'scope': row['scope'],
                'saved_fractions': {arm: {base: v['median_saved_fraction'] for base,v in comp.items()}
                                    for arm,comp in row['comparisons'].items()}})
        recurrence = defaultdict(lambda: defaultdict(lambda: [0,0]))
        for r in read('historical_regression/recurrence_evaluation.json'):
            count = recurrence[r['model']][r['role']]
            count[0] += int(r['metrics']['passed']); count[1] += 1
        stress_failures = []
        for row in read('stress_results.json')['rows']:
            metric = row.get('metrics', {})
            if not metric.get('passed', True):
                stress_failures.append({**{k:v for k,v in row.items() if k != 'metrics'},
                    'failed_channels': [{k:c[k] for k in ('channel','nrmse','nmax','event')} for c in metric['channels'] if not c['passed']]})
        result = {'archive_sha256': sha, 'bytes': archive.stat().st_size, 'crc_valid': True,
                  'runtime_independent': True, 'checkpoint_sha_valid': True,
                  'freeze_claimed_sha256': claimed, 'freeze_hash_valid': True, 'checkpoint_finiteness': finite,
                  'final_report': final, 'costs': costs, 'historical_recurrence': recurrence,
                  'stress_failures': stress_failures, 'paired_contrasts': read('paired_contrasts.json')}
        target = root/'artifacts/task41b_completion_audit.json'
        target.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'audit':str(target),'finite':finite,'costs':costs,
                          'recurrence':recurrence,'stress_failures':stress_failures}, indent=2))

if __name__ == '__main__': main()
