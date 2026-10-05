"""Read-only post-hoc timing diagnosis for the preserved first IV-B run."""

import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher.iv_b_calcium_prerequisite import sk_inf

FOLDER = ROOT / 'experiments/results/iv_b_kaggle_b40e07a'


def main():
    archive = FOLDER / 'artifact_bundle.zip'
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError('IV-B first-run ZIP CRC failure')
        source = json.loads(zipped.read('final_report.json'))
        traces = np.load(io.BytesIO(zipped.read('b2_native_traces.npz')))
        if source['iv_b1_valid'] is not True or source['iv_b2_valid'] is not False:
            raise RuntimeError('Unexpected first-run decision')
        step = math.exp(-.1)
        rows = []
        for row in source['b2_rows']:
            prefix = f"{row['cai0']}_{row['protocol']}"
            calcium = traces[prefix+'_cai']
            gate = traces[prefix+'_z']
            old = np.array([step*gate[i] + (1-step)*sk_inf(calcium[i])
                            for i in range(len(gate)-1)])
            new = np.array([step*gate[i] + (1-step)*sk_inf(calcium[i+1])
                            for i in range(len(gate)-1)])
            rows.append({'cai0': row['cai0'], 'protocol': row['protocol'],
                         'old_cai_one_step_rmse': float(np.sqrt(np.mean((gate[1:]-old)**2))),
                         'updated_cai_one_step_rmse': float(np.sqrt(np.mean((gate[1:]-new)**2)))})
    report = {'schema_version': 'giada-iv-b-first-run-posthoc-semantic-audit-v1',
              'source_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'source_decision_unchanged': 'IV-B1 PASS; IV-B2 NOT PASSED under v1 reference',
              'rows': rows,
              'maximum_updated_cai_one_step_rmse': max(r['updated_cai_one_step_rmse'] for r in rows),
              'diagnosis': 'Native fixed-step SK uses current-step updated cai. V1 used old cai.',
              'decision_grade': False,
              'next_action': 'Run separately preregistered v2 on additional pulse schedules.'}
    (FOLDER / 'semantic_audit.json').write_bytes((json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps({key: report[key] for key in ('maximum_updated_cai_one_step_rmse',
                                                   'diagnosis', 'decision_grade')}))


if __name__ == '__main__':
    main()
