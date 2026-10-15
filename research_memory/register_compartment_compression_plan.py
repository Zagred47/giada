"""Record the requested research redirect and proposed CC tasks, locally only.

No runs, measurements, confirmations or launch authorizations are created.
"""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def validate_plan(plan):
    assert plan['status'] == 'proposal_not_run_preregistration'
    assert [t['id'] for t in plan['tasks']] == [f'CC-{i:02d}' for i in range(1, 21)]
    hypotheses = {h['id'] for h in plan['hypotheses']}
    bundles = {b['id']: b for b in plan['bundles']}
    assert len(hypotheses) == 8 and len(bundles) == 6
    seen = set()
    for task in plan['tasks']:
        assert task['bundle'] in bundles
        assert set(task['depends_on']) <= seen, task['id']
        assert task['contrast'] and task['deliverable']
        seen.add(task['id'])
    for bundle in bundles.values():
        assert set(bundle['hypotheses']) <= hypotheses
        assert any(t['bundle'] == bundle['id'] for t in plan['tasks'])


def main():
    repo = ROOT.parent
    plan_path = repo / 'experiments/giada_compartment_compression_plan_v1.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    validate_plan(plan)
    document = repo / plan['document']
    assert document.is_file()
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **values):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-cc-{suffix}-v1', **values
        })['record_id']

    baseline = mirror.query(
        "SELECT record_id FROM v_records WHERE table_key='findings' "
        "AND stable_code='findings-giada-task35-local-generalization-v1'"
    )['rows']
    assert len(baseline) == 1
    decision = put('decisions', 'direction', **{
        'Nome': 'GIADA: priorità alla compressione appresa del compartimento prima di Task36',
        'Esito': 'Modificare', 'Data': plan['date'],
        'Risultati': [baseline[0]['record_id']],
        'Motivazione': 'Richiesta esplicita utente: aumentare la grana del modello locale e trasferire lavoro al training, preservando bias utili e proprietà computazionali. Non procedere ora ai due compartimenti.',
        'Descrizione': plan['roadmap_policy'] + ' ' + plan['scope'] +
                       ' Il dettaglio CC è proposto: nessun run preregistrato, avviato o completato.',
        'Condizioni di revisione': 'Rivedere con evidenze di sufficienza, learnability, fedeltà e costo. Né scaling universale né speedup sono assunti. Task36 riprendibile solo con decisione esplicita.'})
    claims = {}
    for h in plan['hypotheses']:
        claims[h['id']] = put('claims', h['id'].lower(), **{
            'Nome': h['id'] + ' — ' + h['title'],
            'Tipo': 'Ipotesi', 'Stato': 'Aperta',
            'Enunciato': h['claim'],
            'Condizioni di falsificazione': h['falsification'],
            'Limiti': 'Ipotesi prospettica sul compartimento singolo. Non supportata da nuovi run; dominio, budget e soglie da preregistrare. Non implica equivalenza universale.',
            'Descrizione': 'Piano CC v1; ' + plan['document']})
    experiments = {}
    for bundle in plan['bundles']:
        tasks = [t for t in plan['tasks'] if t['bundle'] == bundle['id']]
        experiments[bundle['id']] = put('experiments', bundle['id'].lower(), **{
            'Nome': bundle['id'] + ' — ' + bundle['title'],
            'Tipo': 'Confermativo' if bundle['id'] == 'N6' else 'Esplorativo',
            'Stato': 'Progettato',
            'Ipotesi': [claims[h] for h in bundle['hypotheses']],
            'Obiettivo informativo': '\n'.join(t['id'] + ': ' + t['contrast'] for t in tasks),
            'Descrizione': 'Proposta di notebook, non implementato né preregistrato. ' +
                           json.dumps(tasks, ensure_ascii=False)})
    actions = {}
    for task in plan['tasks']:
        actions[task['id']] = put('actions', task['id'].lower(), **{
            'Nome': task['id'] + ' — ' + task['title'], 'Stato': 'Proposta',
            'Decisioni di origine': [decision],
            'Esperimenti previsti': [experiments[task['bundle']]],
            'Dipende da': [actions[d] for d in task['depends_on']],
            'Descrizione': json.dumps(task, ensure_ascii=False),
            'Criteri di avvio': 'Prerequisiti diagnosticamente validi; approvazione del protocollo e preregistrazione di ingressi, split, soglie, budget, seed e stopping rule. Non avviata da questa registrazione.',
            'Risultato atteso': task['deliverable']})
    for suffix, path in [('plan', plan_path), ('roadmap', document)]:
        put('artifacts', suffix, **{
            'Nome': 'GIADA CC ' + suffix + ' — proposta v1', 'Tipo': 'Report',
            'Percorso': path.relative_to(repo).as_posix(), 'Versione': plan['version'],
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'Dimensione byte': path.stat().st_size})

    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    counts = mirror.query(
        "SELECT table_key, count(*) AS n FROM v_records "
        "WHERE stable_code LIKE '%-giada-cc-%-v1' GROUP BY table_key"
    )['rows']
    actual = {row['table_key']: row['n'] for row in counts}
    assert actual == {'actions': 20, 'artifacts': 2, 'claims': 8, 'decisions': 1, 'experiments': 6}, actual
    print(json.dumps({'valid': True, 'counts': actual, 'airtable_accessed': False,
                      'runs_created': 0, 'experiments_launched': 0}))


if __name__ == '__main__':
    main()
