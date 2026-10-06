"""Prospective IV-C2 v2 native release-frequency confirmation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

from .hh_family_transfer import write
from .iv_c1_deterministic_synapse import compile_native


def run(repo: Path, teacher: Path, output: Path, revision: str) -> dict:
    from neuron import h, load_mechanisms

    spec = json.loads((repo / 'experiments/iv_c2_native_distribution_confirmation.json').read_text())
    parent_path = repo / spec['parent_report']
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != spec['parent_report_sha256']:
        raise RuntimeError('IV-C2 v1 parent report changed')
    parent = json.loads(parent_path.read_text())
    if not (parent['valid'] and parent['iv_c2_passed'] and not parent['task33_authorized']):
        raise RuntimeError('IV-C2 v1 parent did not pass its registered gates')
    teacher_revision = subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip()
    if teacher_revision != spec['teacher_revision']:
        raise RuntimeError('canonical teacher changed')
    c1 = json.loads((repo / 'experiments/iv_c1_deterministic_synapse_preregistration.json').read_text())
    build = compile_native(repo, teacher, output, c1)
    load_mechanisms(str(build.resolve()))
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = spec['dt_ms']
    h.celsius = 6.3
    rows = []
    for mechanism in spec['mechanisms']:
        sec = h.Section(name='giada_iv_c2_distribution')
        sec.L = sec.diam = 10
        sec.nseg = 1
        sec.insert('pas')
        sec.e_pas = spec['hold_mv']
        clamp = h.SEClamp(sec(.5))
        clamp.dur1 = spec['event_time_ms'] + 1
        clamp.amp1 = spec['hold_mv']
        clamp.rs = .001
        syn = getattr(h, mechanism)(sec(.5))
        con = h.NetCon(None, syn)
        con.delay = 0
        con.weight[0] = spec['weight_ns']
        primary = 'A_AMPA' if mechanism == 'ProbAMPANMDA_EMS' else 'A_GABAA'
        for use in spec['use_values']:
            for dep in spec['depression_ms']:
                for initial in ('recovered', 'depressed'):
                    releases = recoveries = 0
                    for offset in range(spec['replicate_count_per_cell']):
                        seed = spec['seed_start'] + offset
                        rng = h.Random()
                        rng.Random123(seed, spec['stream'], 0)
                        rng.negexp(1.0)
                        syn.setRNG(rng)
                        syn.Use = use
                        syn.Fac = 0.
                        syn.Dep = dep
                        h.finitialize(spec['hold_mv'])
                        if initial == 'depressed':
                            syn.Rstate = 0
                            con.weight[4] = 0.
                        con.event(spec['event_time_ms'])
                        while float(h.t) <= spec['event_time_ms'] + spec['dt_ms'] * .5:
                            h.fadvance()
                        released = float(getattr(syn, primary)) > 0
                        recovered = released or int(syn.Rstate) == 1
                        releases += int(released)
                        recoveries += int(recovered)
                    n = spec['replicate_count_per_cell']
                    p_release = 1 - math.exp(-use)
                    p_recovery = (1. if initial == 'recovered' else
                                  math.exp(-math.exp(-spec['event_time_ms'] / dep)))
                    row = {'mechanism': mechanism, 'Use': use, 'Dep': dep,
                           'initial': initial, 'n': n,
                           'release_fraction': releases / n,
                           'recovery_fraction': recoveries / n,
                           'predicted_release_fraction': p_release * p_recovery,
                           'predicted_recovery_fraction': p_recovery,
                           'release_absolute_error': abs(releases / n - p_release * p_recovery),
                           'recovery_absolute_error': abs(recoveries / n - p_recovery)}
                    row['passed'] = (row['release_absolute_error'] <= spec['max_absolute_probability_error']
                                     and row['recovery_absolute_error'] <= spec['max_absolute_probability_error'])
                    rows.append(row)
            print(f'[GIADA IV-C2 v2] {mechanism} Use={use} complete', flush=True)
        h.delete_section(sec=sec)
    uniform_misfit = max(abs(row['release_fraction'] - row['Use'])
                         for row in rows if row['initial'] == 'recovered')
    report = {'schema_version': spec['schema_version'], 'valid': True,
              'code_revision': revision, 'teacher_revision': teacher_revision,
              'parent_v1_sha256': spec['parent_report_sha256'],
              'cells': rows, 'cell_count': len(rows),
              'max_release_absolute_error': max(row['release_absolute_error'] for row in rows),
              'max_recovery_absolute_error': max(row['recovery_absolute_error'] for row in rows),
              'uniform_misfit': uniform_misfit,
              'uniform_negative_control_passed': uniform_misfit >= spec['minimum_uniform_misfit'],
              'iv_c2_passed': bool(all(row['passed'] for row in rows)
                                   and uniform_misfit >= spec['minimum_uniform_misfit']),
              'iv_c3_passed': False, 'task33_authorized': False}
    write(output / 'final_report.json', report)
    return report
