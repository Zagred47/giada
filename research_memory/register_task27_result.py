"""Register verified Task27 factorial outcomes in the SQLite-only mirror."""
import hashlib
import json

from .mirror import ROOT, write_json
from .register_task24_result import scalars
from .register_task25_result import summary
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task27_kaggle_c072e69'
    report = json.loads((folder / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((folder / 'result_audit.json').read_text(encoding='utf-8'))
    status = json.loads((folder / 'process_status.json').read_text(encoding='utf-8'))
    cfg = json.loads((folder / 'run_contract.json').read_text(encoding='utf-8'))
    freeze = json.loads((folder / 'selection_freeze.json').read_text(encoding='utf-8'))
    assert status == {'phase': 'gpu', 'returncode': 0}
    assert report['valid'] and audit['valid'] and not report['fresh_used_for_selection']
    assert report['task28_preparation_authorized'] and not report['gate_d_completed']
    assert all(audit['checks'].values())
    assert len(report['learned']) == 24 and len(report['fresh_paired_contrasts']) == 162
    assert all(all(arms.values()) for arms in report['per_arm_passed'].values())
    assert all(report['current_supervision_material_benefit'].values())
    for family, arms in report['selection'].items():
        for arm, selected in arms.items():
            name = selected['checkpoint']
            assert name in freeze['checkpoint_hashes']
            assert family in name and arm in name
    mirror = ValidatedBatchMirror()
    cache = {}

    def put(table, key, **fields):
        code = f'{table}-giada-roadmap-task27-{key}-v1'
        rid = mirror.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        cache[table, key] = rid
        return rid

    def ref(table, key):
        if (table, key) not in cache:
            code = f'{table}-giada-roadmap-task27-{key}-v1'
            rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code='" + code + "'")['rows']
            assert len(rows) == 1, code
            cache[table, key] = rows[0]['record_id']
        return cache[table, key]

    artifacts = {}
    for path in sorted(folder.iterdir()):
        if path.suffix not in ('.json', '.zip'):
            continue
        artifacts[path.name] = put('artifacts', 'result-' + path.stem, **{
            'Nome': 'Task27 ' + path.name,
            'Tipo': 'Checkpoint' if path.suffix == '.zip' else 'Report',
            'Percorso': path.relative_to(repo).as_posix(),
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'Dimensione byte': path.stat().st_size,
            'Versione': 'v1',
        })

    def evaluation(metric, group):
        if group in cfg and isinstance(cfg[group], dict) and metric in cfg[group]:
            return ref('evaluations', group + '-' + metric)
        key = 'diagnostic-' + metric
        if ('evaluations', key) not in cache:
            mid = put('metrics', key, **{
                'Nome': 'Task27 ' + metric, 'Famiglia': 'Regressione',
                'Direzione': 'Minimizzare', 'Unità': 'adimensionale',
                'Formula': 'Scalare diagnostico; definizione e strati nel report e contratto immutabili.',
            })
            put('evaluations', key, **{
                'Nome': 'Task27 ' + metric, 'Metrica': [mid],
                'Protocollo': [ref('protocols', 'factorial')], 'Versione': 'v1',
                'Ruolo': 'Secondaria',
                'Aggregazione e pesi': 'Famiglia, braccio, seed, dominio, canale, orizzonte e path distinti.',
            })
        return ref('evaluations', key)

    count = 0
    by_family = {family: [] for family in cfg['families']}
    diagnostics = []

    def record(run, key, data, artifact, group=None):
        nonlocal count
        ids = []
        for path, value in scalars(summary(data)):
            name = '/'.join(path)
            selected_group = group
            if 'currents' in path and group in ('gates', 'path_gates'):
                selected_group = 'current_gates' if group == 'gates' else 'path_current_gates'
            rid = put('observations', key + '-' + hashlib.sha256(name.encode()).hexdigest()[:20], **{
                'Nome': 'Task27 ' + key + ' ' + name,
                'Valore': value, 'Run': [run],
                'Specifica di valutazione': [evaluation(path[-1], selected_group)],
                'Artefatti dettagliati': [artifacts[artifact]],
                'Strato o sottogruppo': '/'.join(path[:-1]),
                'Descrizione': 'Fresh frozen, never used for selection. Repeated panels and complete tensors are losslessly preserved in artifact_bundle.zip.',
            })
            ids.append(rid)
            count += 1
        return ids

    for row in report['learned']:
        family, arm, seed = row['family'], row['arm'], row['seed']
        key = f'{family}-{arm}-s{seed}'
        run = put('runs', key, **{
            'Nome': 'Task27 ' + key, 'Stato': 'Completata',
            'Braccio': [ref('arms', family + '-' + arm)],
            'Blocco': [ref('blocks', 'fresh')], 'Seed': str(seed),
            'Artefatti prodotti': [artifacts['final_report.json'], artifacts['artifact_bundle.zip']],
            'Validità tecnica': 'Kaggle COMPLETE/exit0; native oracle, equivalence, freeze and all 40 checkpoint hashes verified.',
        })
        ids = []
        for field, group in (('metrics', 'gates'), ('held_rollout', 'held_gates'),
                             ('path_rollout', 'path_gates')):
            ids.extend(record(run, key + '-' + field, row[field], 'final_report.json', group))
        ids.extend(record(run, key + '-outcome', {
            name: row[name] for name in ('passed', 'parameters', 'width', 'step')
        }, 'final_report.json'))
        by_family[family].extend(ids)

    run = put('runs', 'paired-contrasts', **{
        'Nome': 'Task27 fresh paired contrasts', 'Stato': 'Completata',
        'Artefatti prodotti': [artifacts['final_report.json']],
        'Validità tecnica': '162 preregistered paired rows, stratified by family, seed, domain, treatment and metric; diagnostic contrasts cannot reselect checkpoints.',
    })
    diagnostics.extend(record(run, 'paired-contrasts', report['fresh_paired_contrasts'], 'final_report.json'))
    for name in ('clipping_probes.json', 'equivalence_preflight.json', 'native_audit.json'):
        run = put('runs', 'diagnostic-' + name[:-5], **{
            'Nome': 'Task27 ' + name, 'Stato': 'Completata',
            'Artefatti prodotti': [artifacts[name]],
            'Validità tecnica': 'Verified diagnostic only; no fresh selection.',
        })
        value = json.loads((folder / name).read_text(encoding='utf-8')) if (folder / name).exists() else None
        if value is not None:
            diagnostics.extend(record(run, 'diagnostic-' + name[:-5], value, name))

    gains = report['primary_median_worst_current_gain']
    findings = []
    claims = {
        'individual': ('Misto', 'Indeterminata',
            'Individual-only passes all 3 seeds in both families, but this does not establish consistent dominance over the paired both arm. Full seed/domain current contrasts retained.',
            by_family['independent'] + by_family['shared_heads']),
        'total': ('Misto', 'Indeterminata',
            'Total-only passes all 3 seeds in both families; total current alone is not sufficient evidence against individual-channel masking. Individual and cancellation panels are retained.',
            by_family['independent'] + by_family['shared_heads']),
        'both': ('Positivo', 'Supportata nel dominio',
            f'Both arm passes 3/3 seed in each family, all fresh/held/path gates. Paired median worst-current improvement over none: independent={gains["independent"]:.6%}, shared_heads={gains["shared_heads"]:.6%}; registered minimum=10%. Shared margin is narrow.',
            by_family['independent'] + by_family['shared_heads'] + diagnostics),
        'sharing': ('Positivo', 'Supportata nel dominio',
            f'Benefit differs across independently trained architectures: independent={gains["independent"]:.6%}, shared_heads={gains["shared_heads"]:.6%}. This is local to fixed Task26 widths and does not imply a general sharing law.',
            diagnostics),
    }
    for key, (outcome, state, text, ids) in claims.items():
        finding = put('findings', key, **{
            'Nome': 'Task27 ' + key, 'Esito': outcome,
            'Esperimenti': [ref('experiments', 'matrix')], 'Osservazioni': ids,
            'Risultato': text, 'Limitazioni': cfg['limits'],
        })
        findings.append(finding)
        put('evidence', key, **{
            'Nome': 'Task27 ' + key,
            'Esito': 'Sostiene' if outcome == 'Positivo' else 'Indeterminata',
            'Affermazione valutata': [ref('claims', key)],
            'Risultati a sostegno': [finding], 'Argomentazione': text,
        })
        put('claims', key, **{'Stato': state})
    put('decisions', 'to28', **{
        'Nome': 'Prepare original Task28 teacher-forced ionic block', 'Esito': 'Continuare',
        'Risultati': findings,
        'Motivazione': 'All eight factorial arms pass 3/3; paired both-vs-none current gain exceeds 10% in both families with unchanged preregistered gates. Native/freeze/checkpoint audit valid.',
        'Condizioni di revisione': 'Preparation only. Gate D, autonomous voltage, Ca feedback, synapses and full-neuron speed/accuracy remain untested.',
    })
    put('experiments', 'matrix', **{'Stato': 'Concluso'})
    put('runs', 'kaggle-orchestration', **{
        'Stato': 'Completata', 'Artefatti prodotti': list(artifacts.values()),
        'Validità tecnica': 'Kaggle COMPLETE; worker exit0; independent audit valid; 40 checkpoint hashes and full ZIP verified. Eight arms x three seeds all pass.',
    })
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'observations': count, 'findings': len(findings),
                      'gains': gains, 'task28_preparation_authorized': True,
                      'airtable_accessed': False}))


if __name__ == '__main__':
    main()
