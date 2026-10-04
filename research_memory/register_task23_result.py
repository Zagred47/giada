"""Verify Task23 archive and preserve all scalar probes in the local mirror."""
import hashlib
import json
import subprocess
import zipfile
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    archive = repo / 'artifacts/giada_task23_calcium_pair_0f79081_f87cac0a.zip'
    root = repo / 'experiments/results/task23_kaggle_0f79081'
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None
        freeze = json.loads(source.read('selection_freeze.json'))
        claimed = freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == claimed
        assert not freeze['fresh_accessed']
        for path, value in freeze['checkpoint_hashes'].items():
            assert hashlib.sha256(source.read(path)).hexdigest() == value
        provenance = json.loads(source.read('code_provenance.json'))
        assert not provenance['dirty_runtime']
        for path, value in provenance['sources'].items():
            assert hashlib.sha256(subprocess.check_output(['git', 'show', provenance['code_revision'] + ':' + path], cwd=repo)).hexdigest() == value
        for name in source.namelist():
            if '/' not in name and name.endswith('.json'):
                (root / name).write_bytes(source.read(name))
    report = json.loads((root / 'final_report.json').read_text())
    assert report['valid'] and report['composition_passed'] and report['sharing_confirmed']
    assert report['task24_preparation_authorized'] and not report['fresh_used_for_selection']
    assert json.loads((root / 'process_status.json').read_text())['returncode'] == 0
    assert all(json.loads((root / name).read_text())['valid'] for name in ('native_audit.json', 'equivalence_preflight.json'))
    write_json(root / 'archive_audit.json', dict(valid=True, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), CRC_valid=True, checkpoint_and_source_hashes_valid=True))
    m = ValidatedBatchMirror()
    cache = {}

    def put(table, code, **fields):
        rid = m.local_upsert(table, {'Codice stabile': f'{table}-giada-roadmap-task23-{code}-v1', **fields})['record_id']
        cache[table, code] = rid
        return rid

    def ref(table, code):
        if (table, code) not in cache:
            cache[table, code] = m.query("SELECT record_id FROM v_records WHERE stable_code='" + f'{table}-giada-roadmap-task23-{code}-v1' + "'")['rows'][0]['record_id']
        return cache[table, code]

    arts = {}
    for path in root.glob('*.json'):
        arts[path.name] = put('artifacts', 'result-' + path.stem, **{'Nome': 'Task23 ' + path.name, 'Tipo': 'Report', 'Percorso': path.relative_to(repo).as_posix(), 'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    evaluations = {}
    protocol = ref('protocols', 'paired')

    def evaluation(key, group=None):
        code = (group + '-' if group else 'diagnostic-') + key
        if code not in evaluations:
            if group:
                evaluations[code] = ref('evaluations', code)
            else:
                metric = put('metrics', code, **{'Nome': 'Task23 diagnostic ' + key, 'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mA/cm2' if 'ma_cm2' in key else 'adimensionale', 'Formula': 'Scalar probe preserved from verified Task23 artifact; direction descriptive, not a promotion gate.'})
                evaluations[code] = put('evaluations', code, **{'Nome': 'Task23 diagnostic ' + key, 'Metrica': [metric], 'Protocollo': [protocol], 'Versione': 'v1', 'Ruolo': 'Secondaria', 'Aggregazione e pesi': 'Per seed/domain/channel/step; preserve original strata.', 'Popolazione e regioni': report['scope']})
        return evaluations[code]

    cfg = freeze['config']
    observations = []
    subsets = {f: [] for f in ('independent', 'shared_heads', 'conditioned', 'native')}

    def scalar_rows(value, prefix=()):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from scalar_rows(item, prefix + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from scalar_rows(item, prefix + (str(index),))
        elif isinstance(value, (int, float, bool)):
            yield prefix, float(value)

    def record(run, suffix, value, artifact, family='', default_group=None):
        for path, scalar in scalar_rows(value):
            key = path[-1]
            group = default_group
            if group == 'gates' and 'pair' in path:
                group = 'pair_gates'
            if group and key not in cfg[group]:
                group = None
            identity = '-'.join(path)
            rid = put('observations', suffix + '-' + identity, **{'Nome': 'Task23 ' + suffix + ' ' + '/'.join(path), 'Valore': scalar, 'Run': [run], 'Specifica di valutazione': [evaluation(key, group)], 'Artefatti dettagliati': [arts[artifact]], 'Strato o sottogruppo': '/'.join(path[:-1]), 'Descrizione': 'Preserved scalar; OOD, swapped identities and numerical references diagnostic, not used for selection.'})
            observations.append(rid)
            if family in subsets:
                subsets[family].append(rid)

    for role, rows in [('fresh', report['learned']), ('development', json.loads((root / 'development_ladder.json').read_text()))]:
        for row in rows:
            suffix = f'{role}-{row["family"]}-w{row["width"]}-step{row["step"]}-s{row["seed"]}'
            artifact = 'final_report.json' if role == 'fresh' else 'development_ladder.json'
            run = put('runs', suffix, **{'Nome': 'Task23 ' + suffix, 'Stato': 'Completata', 'Braccio': [ref('arms', row['family'] + '-w' + str(row['width']))], 'Blocco': [ref('blocks', role)], 'Seed': str(row['seed']), 'Artefatti prodotti': [arts[artifact]], 'Configurazione effettiva': json.dumps({k: row[k] for k in ('family', 'width', 'step', 'seed')}), 'Validità tecnica': 'Archive CRC, freeze, source/checkpoint hashes, native oracle and vectorized equivalence verified.'})
            record(run, suffix, row['metrics'], artifact, row['family'], 'gates')
            for name, group in [('held_rollout', 'held_gates'), ('path_rollout', 'path_gates'), ('swapped_identity', 'gates')]:
                if name in row:
                    record(run, suffix + '-' + name, row[name], artifact, row['family'], group)
            record(run, suffix + '-summary', {k: row[k] for k in ('score', 'passed', 'parameter_count', 'swapped_identity_rmse_ratio') if k in row}, artifact, row['family'])
    for artifact, rows in [('final_report.json', report['numerical']), ('paired_architecture_contrasts.json', json.loads((root/'paired_architecture_contrasts.json').read_text())), ('paired_scaling_contrasts.json', json.loads((root/'paired_scaling_contrasts.json').read_text())), ('native_audit.json', json.loads((root/'native_audit.json').read_text())['rows']), ('equivalence_preflight.json', json.loads((root/'equivalence_preflight.json').read_text())['rows'])]:
        for index, row in enumerate(rows):
            suffix = artifact.replace('.json', '') + '-' + str(index)
            run = put('runs', suffix, **{'Nome': 'Task23 ' + suffix, 'Stato': 'Completata', 'Artefatti prodotti': [arts[artifact]], 'Validità tecnica': 'Diagnostic artifact verified; no post-hoc model selection.'})
            record(run, suffix, row, artifact, 'native' if artifact == 'native_audit.json' else '')
    finding_specs = [
        ('composition', 'Positivo', 'Both independent calcium mechanisms and their sum pass all required strata/held horizons/paths for 3/3 seeds.', 'independent'),
        ('sharing', 'Positivo', 'Shared heads passes 3/3 with 440 vs 744 parameters (40.86% fewer). This is not a measured runtime speedup.', 'shared_heads'),
        ('conditioned', 'Negativo', 'Conditioned family fails conjunction: seed17 HVA h_rmse exceeds .001 in-support. Not impossibility of identity conditioning.', 'conditioned'),
        ('calva_oracle', 'Positivo', 'Native canonical rate/update/current audit passes; fixed temperature policy and V+10 shift preserved.', 'native')]
    findings = []
    for code, outcome, result, family in finding_specs:
        findings.append(put('findings', code, **{'Nome': 'Task23 ' + code, 'Esito': outcome, 'Esperimenti': [ref('experiments', 'matrix')], 'Osservazioni': subsets[family], 'Risultato': result, 'Limitazioni': report['scope']}))
    for code, finding in [('composition', findings[0]), ('sharing', findings[1]), ('path_transfer', findings[0]), ('calva_oracle', findings[3])]:
        put('evidence', code, **{'Nome': 'Task23 ' + code, 'Esito': 'Sostiene', 'Affermazione valutata': [ref('claims', code)], 'Risultati a sostegno': [finding], 'Argomentazione': 'Conjunction passed in registered imposed-V domain; independent and shared-head controls separated.'})
        put('claims', code, **{'Stato': 'Supportata nel dominio'})
    put('findings', 'scaling-identity', **{'Nome': 'Task23 scaling and identity probes preserved', 'Esito': 'Inconcludente', 'Esperimenti': [ref('experiments','matrix')], 'Risultato': 'All paired contrasts and swapped probes registered; no universal capacity/optimization conclusion inferred.', 'Limitazioni': 'Descriptive diagnostics, no independent confirmation.'})
    put('decisions', 'to24', **{'Nome': 'Task23 permits original Task24 sodium family', 'Esito': 'Continuare', 'Risultati': findings, 'Motivazione': 'Independent composition and shared heads pass. Sodium comparison remains new hypothesis; retain independent per-channel controls and slow Nap horizons.', 'Condizioni di revisione': 'NaTs2 and full Nap native audits mandatory; sectionIV gates still apply before autonomous voltage.'})
    put('experiments', 'matrix', **{'Stato': 'Concluso'})
    put('runs', 'kaggle-orchestration', **{'Stato': 'Completata', 'Artefatti prodotti': list(arts.values()), 'Validità tecnica': 'Kaggle COMPLETE exit0; scientific independent/shared GO, conditioned NO-GO.'})
    m.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', m.export_snapshot())
    assert m.verify()['valid']
    print(json.dumps(dict(valid=True, observations=len(observations), task24_preparation_authorized=True, airtable_accessed=False)))


if __name__ == '__main__':
    main()
