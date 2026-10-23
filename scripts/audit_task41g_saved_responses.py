"""Analyze saved intervention traces only; no model execution or re-selection."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task41g_audited_v1'


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def main():
    spec = json.loads((ROOT / 'experiments/task41g_preregistration_v1.json').read_text())
    archive = ROOT / 'artifacts/task41g_primary_external/recovery_final_reconstructed.zip'
    with zipfile.ZipFile(ROOT / 'artifacts/task41d_opened41c.zip') as source:
        raw = source.read('cases.json')
        assert hashlib.sha256(raw).hexdigest() == spec['opened41c_members']['cases.json']
        meta = json.loads(raw)
    old = json.loads((ROOT / 'artifacts/task41g_primary_external/partial_scientific_audit.json').read_text())
    rows = []
    provenance = []
    with zipfile.ZipFile(archive) as z:
        def arrays(name):
            raw = z.read(name)
            provenance.append(dict(member=name, sha256=hashlib.sha256(raw).hexdigest()))
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        reference = arrays('opened41c/reference_traces.npz')
        predictions = {mode: arrays('opened41c/traces_M_anneal_sensitive_tail_' + mode + '.npz')
                       for mode in ('imposed', 'autonomous')}
        for case, seed in ((122, 17), (108, 43)):
            si = (17, 29, 43).index(seed)
            base = int(meta[case]['source_index'])
            row = dict(case=case, seed=seed, protocol=meta[case]['protocol'], intervention=meta[case]['intervention'],
                       base_index=base, modes={})
            for mode, prediction in predictions.items():
                rr = reference[mode + '__current']
                pp = prediction['current'][:, si]
                response_r = rr[:, case, 4] - rr[:, base, 4]
                response_p = pp[:, case, 4] - pp[:, base, 4]
                diff = response_p - response_r
                scale = np.maximum(np.max(np.abs(response_r)), np.float32(spec['quality']['current_scale_floor_ma_cm2']))
                nrmse = float(np.sqrt(np.mean(diff ** 2)) / scale)
                if mode == 'autonomous':
                    expected = next(r for r in old['failures']['M_anneal_sensitive_tail']['41c/autonomous']
                                    if r['case'] == case and r['seed'] == seed)['failed_channels'][0]['nrmse']
                    assert abs(nrmse - expected) < 1e-8
                vref = reference[mode + '__voltage'].astype(np.float64)
                vpred = prediction['voltage'][:, si].astype(np.float64)
                row['modes'][mode] = dict(
                    Nap_response_nrmse=nrmse,
                    Nap_response_scale_ma_cm2=float(scale),
                    Nap_reference_response_peak_ma_cm2=float(np.max(np.abs(response_r))),
                    Nap_reference_base_current_peak_ma_cm2=float(np.max(np.abs(rr[:, base, 4]))),
                    Nap_reference_intervention_current_peak_ma_cm2=float(np.max(np.abs(rr[:, case, 4]))),
                    base_voltage_rmse_mv=rms(vpred[:, base] - vref[:, base]),
                    intervention_voltage_rmse_mv=rms(vpred[:, case] - vref[:, case]),
                    voltage_response_error_rmse_mv=rms((vpred[:, case] - vpred[:, base]) - (vref[:, case] - vref[:, base])))
            rows.append(row)
    result = dict(valid=True, scope='Offline reduction of existing saved traces, not new simulation or confirmation',
        model='M_anneal_sensitive_tail', channel='Nap_Et2', rows=rows, trace_members=provenance,
        interpretation='The failing metric measures the difference between intervened and baseline currents. The two interventions block different potassium channels, not Nap_Et2. Raw trajectory quality and causal response quality are distinct; small voltage errors can be amplified relatively in a weak response. This supports examining numerical feedback but does not by itself prove roundoff causality.',
        thresholds_changed=False, selection_changed=False, training_performed=False)
    (OUT / 'saved_response_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
