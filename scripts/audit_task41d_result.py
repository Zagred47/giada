"""Read-only scientific audit of immutable 41d/41c ZIPs; save derived JSON."""
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/task41d_primary_external'


def load(z, name):
    return json.loads(z.read(name))


def failed(rows, model=None):
    result = []
    for row in rows:
        if model is not None and row.get('model') != model:
            continue
        if row['metrics']['passed']:
            continue
        result.append({**{k: row[k] for k in ('case', 'seed', 'family', 'input_seed') if k in row},
                       'failed_channels': [c for c in row['metrics']['channels'] if not c['passed']],
                       'other_metrics': {k: v for k, v in row['metrics'].items() if k != 'channels'}})
    return result


def main():
    receipt = json.loads((FOLDER / 'external_backup_receipt.json').read_text())
    path = FOLDER / 'giada_task41d_result.zip'
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == receipt['remote']['sha256']
    assert path.stat().st_size == receipt['remote']['bytes']
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(ROOT / 'artifacts/task41c_primary_external/giada_task41c_result.zip') as prior:
        bad = z.testzip()
        assert bad is None, bad
        report, old = load(z, 'final_report.json'), load(prior, 'final_report.json')
        assert report['valid'] and not report['thresholds_changed']
        protocol, old_protocol = load(z, 'preregistration.json'), load(prior, 'preregistration.json')
        assert protocol['quality'] == old_protocol['quality']
        summaries = report['summaries']
        parent_reproduction = {mode: summaries['P110']['panels']['41c'][mode] == old['summary']['P110/' + mode]
                               for mode in ('imposed', 'autonomous')}
        parent_reproduction['stress'] = summaries['P110']['stress']['41c'] == old['stress']['P110']
        assert all(parent_reproduction.values())
        stress = {panel: load(z, 'stress' + panel + '.json') for panel in ('41', '41c')}
        causal = {panel: load(z, 'opened' + panel + '/causal_results.json') for panel in ('41', '41c')}
        dev = load(z, 'development_gpu.json')
        details = {}
        for name in summaries:
            details[name] = {
                'development': failed(dev[name]),
                'stress': {p: failed(stress[p]['rows'], name) for p in stress},
                'autonomous_responses': {p: failed(causal[p][name + '/autonomous']['interventions']) for p in causal},
            }
        cost = []
        for row in load(z, 'cost.json'):
            cost.append({k: v for k, v in row.items() if k != 'samples'} | {
                'median_wall_ms': {name: statistics.median(v['wall_ms']) for name, v in row['samples'].items()},
                'median_cuda_ms': {name: statistics.median(v['cuda_ms']) for name, v in row['samples'].items()}})
        floors = {p: all(r['passed'] for r in stress[p]['formula_floor']) for p in stress}
        for p in causal:
            for mode in ('imposed', 'autonomous'):
                floor = load(z, 'opened' + p + '/floor_' + mode + '.json')
                floors[p + '/' + mode] = all(r['passed'] for r in floor['cases'] + floor['interventions'])
            floors[p + '/native'] = all(r['passed'] for r in load(z, 'opened' + p + '/native_phase_floor.json'))
        assert all(floors.values())
        result = dict(backup=dict(sha256=h.hexdigest(), bytes=path.stat().st_size, crc_all_members_valid=True,
                                  runtime_independent=True), decision=report['decision'],
                      thresholds_identical_to_41c=True, parent_reproduction=parent_reproduction,
                      nominees=report['nominees'], summaries=summaries, failure_details=details,
                      numerical_floors=floors, short_response_floor=report['short_response_precision_floor_passed'],
                      factorial_contrasts=load(z, 'paired_contrasts.json'), cost=cost,
                      cost_policy=protocol['cost'], selection=load(z, 'selection_freeze.json'),
                      task42_authorized=False, independent_confirmation=False)
    dest = FOLDER / 'detailed_audit.json'
    dest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(audit=str(dest), backup=result['backup'], parent_reproduction=parent_reproduction,
                         floors=floors, cost_policy=result['cost_policy'])))


if __name__ == '__main__':
    main()
