"""Forensic audit of failed IV-C1 v1/v2 state gates; never a promotion run."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.giada_teacher.iv_c1_deterministic_synapse import (
    STATES, analytic_trace, compile_native, load_v2_contract, native_episode)
from src.giada_teacher.hh_family_transfer import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_v2_contract(ROOT)
    teacher = Path(args.teacher)
    revision = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != cfg['teacher_revision']:
        raise RuntimeError('teacher revision changed')
    from neuron import load_mechanisms
    build = compile_native(ROOT, teacher, output, cfg)
    load_mechanisms(str(build.resolve()))
    schedules = {'calibration_v2': cfg['calibration_schedule'],
                 **cfg['confirmation_schedules']}
    rows = []
    summaries = []
    for label, schedule in schedules.items():
        native, meta = native_episode(schedule, -75., cfg, diagnostic=True)
        right_cfg = {**cfg, 'state_observation_phase': 'post_event_at_exact_time'}
        left = analytic_trace(native['time_ms'], native['voltage_mv'], schedule,
                              cfg, 0, meta['parameters'])
        right = analytic_trace(native['time_ms'], native['voltage_mv'], schedule,
                               right_cfg, 0, meta['parameters'])
        errors = {convention: max(float(np.max(np.abs(native[name] - pred[name])))
                                  for name in STATES)
                  for convention, pred in [('left', left), ('right', right)]}
        maxima = sorted(((float(abs(native[name][n] - right[name][n])), n, name)
                         for name in STATES for n in range(len(native['time_ms']))),
                        reverse=True)[:8]
        summaries.append({'schedule': label, 'events_ms': schedule,
                          'max_state_error_by_convention': errors,
                          'maxima_right': [{'error': value, 'index': n,
                                            'state': name,
                                            'grid_time_ms': float(native['time_ms'][n]),
                                            'actual_h_t_ms': float(native['actual_time_ms'][n])}
                                           for value, n, name in maxima]})
        event_times = sorted(set(schedule['exc_ms'] + schedule['inh_ms']))
        for event in event_times:
            center = int(round(event / cfg['dt_ms']))
            for n in range(max(0, center-1), min(len(native['time_ms']), center+3)):
                row = {'schedule': label, 'scheduled_event_ms': event, 'index': n,
                       'grid_time_ms': float(native['time_ms'][n]),
                       'actual_h_t_ms': float(native['actual_time_ms'][n]),
                       'states': {}, 'per_synapse': {}}
                for name in STATES:
                    row['states'][name] = {'native': float(native[name][n]),
                                           'left': float(left[name][n]),
                                           'right': float(right[name][n])}
                for index, values in native['per_synapse'].items():
                    row['per_synapse'][index] = {name: float(data[n])
                                                 for name, data in values.items()}
                rows.append(row)
        print(f'[GIADA IV-C1 forensic] {label}: left={errors["left"]:.6g} '
              f'right={errors["right"]:.6g}', flush=True)
    provenance = {'code_revision': subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'teacher_revision': revision,
        'v2_report_sha256': hashlib.sha256((ROOT /
            'experiments/results/iv_c1_v2_kaggle_f28d3bc/final_report.json').read_bytes()).hexdigest(),
        'diagnostic_only': True, 'promotion_authorized': False}
    write(output / 'state_timing_forensic.json',
          {'schema_version': 'giada-iv-c1-state-timing-forensic-v1',
           'provenance': provenance, 'summaries': summaries, 'event_windows': rows})
    write(output / 'process_status.json', {'returncode': 0})


if __name__ == '__main__':
    main()
