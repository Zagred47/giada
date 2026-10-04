"""Record the actual MCP launch without claiming a completed experiment."""
from __future__ import annotations
import hashlib
import json
from .mirror import Mirror, ROOT, write_json


def main():
    mirror = Mirror()
    root = ROOT.parent / 'experiments/results/task17f_kaggle_1b87bdd'
    manifest = json.loads((root / 'launch_manifest.json').read_text())
    def ref(table, suffix):
        rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code='"
            + f"{table}-giada-task17f-event-support-{suffix}-v1" + "'")['rows']
        if len(rows) != 1: raise RuntimeError('Task17f preregistration missing')
        return rows[0]['record_id']
    artifacts = []
    for path in sorted(root.iterdir()):
        artifacts.append(mirror.local_upsert('artifacts', {
            'Nome': 'Task17f Kaggle v1 — ' + path.name,
            'Codice stabile': 'artifacts-giada-task17f-kaggle-v1-' + path.stem,
            'Tipo': 'Report', 'Percorso': path.relative_to(ROOT.parent).as_posix(),
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})['record_id'])
    result = mirror.local_upsert('runs', {
        'Nome': 'Task17f — Kaggle MCP, versione1, conferma indipendente',
        'Codice stabile': 'runs-giada-task17f-kaggle-v1-orchestration',
        'Stato': 'In corso', 'Blocco': [ref('blocks', 'confirmation')],
        'Seed': 'pilot171501/171519/171543; conferma171601/171619/171643',
        'Artefatti prodotti': artifacts,
        'Hardware e ambiente': 'Kaggle CPU, Internet abilitato, GPU/TPU disabilitate; ' + manifest['docker_image'],
        'Configurazione effettiva': 'Commit ' + manifest['code_revision'] + '; notebook privato; nessun dataset esterno; pilot nativo e quattro bracci confermativi a tre gbar.',
        'Validità tecnica': 'MCP save_notebook accettato; get_notebook_session_status=RUNNING. Esito scientifico ancora non disponibile.',
        'Descrizione': manifest['url'] + '; kernel_id137021889; v1. Stato osservato ' + manifest['observed_at_utc'] + '. Esecuzione aggregata dei quattro bracci; nessun risultato inferito dalla sola partenza.'})
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    return {'valid': True, 'run': result['record_id'], 'status': 'RUNNING', 'airtable_accessed': False}


if __name__ == '__main__': print(json.dumps(main(), indent=2))
