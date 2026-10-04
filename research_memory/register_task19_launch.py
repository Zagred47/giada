"""Persist the actual pinned Task19 Kaggle submission in the local mirror."""
import argparse
import hashlib
import json
from pathlib import Path
from .mirror import ROOT, write_json
from .register_task18d_result import StagedMirror


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    root = ROOT.parent / 'experiments/results' / ('task19_kaggle_' + manifest['code_revision'][:7])
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'final_report.json').exists():
        raise RuntimeError('Do not reset completed result')
    write_json(root / 'launch_manifest.json', manifest)
    notebook = json.loads((ROOT.parent / 'notebooks/19_roadmap_single_gate_transfer.ipynb').read_text(encoding='utf-8'))
    notebook['cells'].insert(1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=[
        'import os\n', "os.environ['GIADA_CODE_REVISION'] = " + repr(manifest['code_revision']) + '\n',
        "os.environ['GIADA_MANUAL_DOWNLOAD'] = '0'\n"]))
    write_json(root / 'submitted_notebook.ipynb', notebook)
    mirror = StagedMirror()
    artifacts = []
    for path in root.iterdir():
        artifacts.append(mirror.local_upsert('artifacts', {'Codice stabile': 'artifacts-giada-roadmap-task19-launch-' + path.stem + '-v1',
            'Nome': 'Task19 Kaggle ' + path.name, 'Tipo': 'Report', 'Percorso': path.relative_to(ROOT.parent).as_posix(),
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})['record_id'])
    block = mirror.query("SELECT record_id FROM v_records WHERE stable_code='blocks-giada-roadmap-task19-fit-v1'")['rows'][0]['record_id']
    mirror.local_upsert('runs', {'Codice stabile': 'runs-giada-roadmap-task19-kaggle-orchestration-v1',
        'Nome': 'Task19 Kaggle MCP — 24modelli', 'Stato': 'In corso', 'Blocco': [block], 'Seed': '17,29,43; data190xxx',
        'Hardware e ambiente': 'Kaggle ' + manifest['machine_shape'], 'Configurazione effettiva': 'Commit ' + manifest['code_revision'] + ';24modelli;30ksteps.',
        'Validità tecnica': 'Submission e source pin verificati; esito scientifico ancora da valutare.',
        'Artefatti prodotti': artifacts, 'Descrizione': manifest['url']})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps(dict(valid=True, status=manifest['status'], airtable_accessed=False)))


if __name__ == '__main__':
    main()
