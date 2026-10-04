"""Register the approved IV roadmap reconciliation, exclusively in SQLite."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    path = ROOT.parent / 'experiments/giada_section_iv_plan.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    nodes = plan['tasks'] + plan['integration_gates']
    ids = [node['id'] for node in nodes]
    assert len(ids) == len(set(ids))
    seen = set()
    for node in nodes:
        assert set(node['depends_on']) <= seen, node['id']
        seen.add(node['id'])
    mirror = ValidatedBatchMirror()
    decision = mirror.local_upsert('decisions', {
        'Codice stabile': 'decisions-giada-section-iv-reconciliation-v1',
        'Nome': 'GIADA: rendere operativa la sezione IV senza rinumerare le task',
        'Esito': 'Modificare', 'Data': plan['date'],
        'Motivazione': 'Approvazione utente: risolvere ambiguita tra sezione IV senza task numerate e Task23 nella sezione V.',
        'Descrizione': plan['policy'] + '\n' + plan['task23_entry'] + '\n' + plan['task28_entry'],
        'Condizioni di revisione': 'Rivalutare dipendenze se cambiano componenti, ingressi o dominio. Nessuna equivalenza automatica tra gate isolato e feedback.'
    })['record_id']
    records = {}
    for node in nodes:
        records[node['id']] = mirror.local_upsert('actions', {
            'Codice stabile': 'actions-giada-section-iv-' + node['id'].lower() + '-v1',
            'Nome': node['id'] + ' — ' + node['title'],
            'Stato': 'Proposta',
            'Decisioni di origine': [decision],
            'Dipende da': [records[dep] for dep in node['depends_on']],
            'Descrizione': json.dumps(node, ensure_ascii=False),
            'Criteri di avvio': 'Prerequisiti collegati validati; contratto, soglie, domini, seed e budget preregistrati. ' + (
                'SK canonico validato (Task20) richiesto. ' if node['id'] == 'IV-B2' else ''
            ) + node.get('condition', ''),
            'Risultato atteso': node.get('completion', node.get('condition'))
        })['record_id']
    mirror.local_upsert('artifacts', {
        'Codice stabile': 'artifacts-giada-section-iv-plan-v1',
        'Nome': 'GIADA sezione IV — piano e dipendenze', 'Tipo': 'Report',
        'Percorso': path.relative_to(ROOT.parent).as_posix(), 'Versione': plan['version'],
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size
    })
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    rows = mirror.query("SELECT stable_code FROM v_records WHERE table_key='actions' AND stable_code LIKE 'actions-giada-section-iv-%'")['rows']
    assert len(rows) == len(nodes)
    print(json.dumps({'valid': True, 'planned_tasks': len(plan['tasks']),
                      'integration_gates': len(plan['integration_gates']),
                      'dependency_edges': sum(len(n['depends_on']) for n in nodes),
                      'airtable_accessed': False}))


if __name__ == '__main__':
    main()
