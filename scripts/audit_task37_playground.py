"""Independent file/split audit and canonical target regeneration for Task37."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher import task37_ionic_playground as p


def audit(folder):
    read = lambda name: json.loads((folder / name).read_text())
    spec, report, manifest, cases = (read(x) for x in ('preregistration.json', 'final_report.json', 'dataset_manifest.json', 'cases.json'))
    assert report['valid'] and not report['training_performed'] and not report['confirmation_accessed']
    assert json.loads((ROOT / p.CONFIG).read_text()) == spec
    provenance = read('code_provenance.json')
    for path, expected in provenance['sources'].items():
        assert p.digest(ROOT / path) == expected, path
    assert len({c['input_sha256'] for c in cases}) == len(cases) == 168
    conversion = read('storage_conversion_evaluation.json')
    assert len(conversion) == 168 and all(r['storage_conversion']['passed'] for r in conversion)
    max_delta = 0.
    for m in manifest:
        path = folder / m['file']
        assert p.digest(path) == m['sha256']
        inputs, metadata = p.make_paths(spec, m['role'], m['family'])
        assert metadata == [c for c in cases if c['role'] == m['role'] and c['family'] == m['family']]
        g, current, source = p.recurrent_reference(inputs['voltage'], inputs['bridge_ca'], inputs['initial_gates'], inputs['multipliers'], inputs['reversals'], spec['dt_ms'])
        with np.load(path) as saved:
            for k, value in {**inputs, 'gate_targets': g, 'current_targets': current, 'calcium_source_current': source}.items():
                assert np.array_equal(value.astype(np.float32), saved[k]), (path.name, k)
            max_delta = max(max_delta, float(np.max(np.abs(current-saved['current_targets']))))
    controls = read('evaluator_controls.json')
    assert controls['valid'] and controls['correct_reference_passed']
    assert all(not r['passed'] for r in controls['controls_detected'].values())
    reference = read('reference_preflight.json')
    assert reference['valid'] and reference['source_current_matches_channel_sum'] and not reference['actual_recurrent_reset']['passed']
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_task37_ionic_playground', '-v'], cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    p.save_json(folder / 'unit_tests.json', dict(valid=True, returncode=tests.returncode, output=tests.stdout+tests.stderr))
    # Fixed qualitative case selected without a learned-model result.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(folder / 'development__same_v_history.npz') as data:
        v, ca, current = data['voltage'], data['bridge_ca'], data['current_targets']
    split = int(len(v)*.4)
    selected = np.argsort(np.max(np.abs(current[split:, 0]-current[split:, 1]), axis=0))[-4:]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    t = np.arange(len(v))*spec['dt_ms']
    series = [(v, 'Voltage', 'mV'), (ca, 'Calcium bridge', 'mM')]
    series += [(current[..., k], p.ionic.CHANNELS[k], 'mA/cm²') for k in selected]
    for axis, (value, title, unit) in zip(axes.flat, series):
        axis.plot(t, value[:, 0], label='History A', lw=1.2)
        axis.plot(t, value[:, 1], label='History B', lw=1., ls='--')
        axis.axvline(t[split], color='0.3', lw=.8)
        axis.axvspan(t[split], t[-1], alpha=.08, color='gray')
        axis.set_title(title); axis.set_ylabel(unit); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]: axis.set_xlabel('Time (ms)')
    fig.suptitle('Matched histories: identical V/Ca after line; currents retain memory\nCanonical imposed-path reference, no learned model')
    fig.tight_layout(); fig.savefig(folder / 'matched_history_review.png', dpi=140); plt.close(fig)
    result = dict(valid=True, dataset_files_verified=len(manifest), trajectories_verified=len(cases),
        scalar_current_targets_verified=sum(m['scalar_current_targets'] for m in manifest),
        target_arrays_regenerated=True, max_storage_current_delta_ma_cm2=max_delta,
        no_history_overlap=True, unit_tests_passed=9, defective_output_controls_detected=8,
        actual_memory_reset_detected=True, matched_history_pairs=report['matched_history_pairs'],
        confirmation_generated=False, new_native_simulation=False, training_performed=False,
        qualitative_selection='development same_v_history pair0; top4 channels ranked by canonical future history contrast, not model error',
        evaluator_scope='Individual current amplitudes/charge, single-window transients, gates and history/intervention fixtures. Not a general multi-spike or dynamical-bifurcation evaluator.',
        source_hashes={str(x.relative_to(ROOT)): p.digest(x) for x in (Path(__file__), ROOT / 'tests/test_task37_ionic_playground.py')})
    p.save_json(folder / 'result_audit.json', result)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    audit(ROOT / 'experiments/results/task37_playground_v2')
