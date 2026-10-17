"""Prospective Task38 registration, SQLite only; no fabricated results."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec_path = repo/'experiments/task38_joint_transition_preregistration_v1.json'
    spec = json.loads(spec_path.read_text())
    mirror = ValidatedBatchMirror(); ids = {}
    def put(table, suffix, **fields):
        code = f'{table}-giada-task38-joint-transition-{suffix}-v1'
        rid = mirror.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        ids[code] = rid
        return rid
    def ref(code):
        rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows) == 1, code
        return rows[0]['record_id']
    artifacts = {}
    for relative in ('experiments/task38_joint_transition_preregistration_v1.json',
        'src/giada_teacher/task38_joint_transition.py', 'scripts/run_task38_joint_transition.py',
        'tests/test_task38_joint_transition.py', 'notebooks/giada_task38_joint_transition.ipynb',
        'experiments/task38_design_v1.md'):
        path = repo/relative
        artifacts[relative] = put('artifacts', relative, Nome='Task38 '+relative, Percorso=relative,
            Tipo='Codice' if path.suffix == '.py' else 'Report', Versione='v1',
            **{'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size})
    claims = []
    for name, statement in spec['hypotheses'].items():
        claims.append(put('claims', name, Nome='Task38 '+name, Tipo='Ipotesi', Stato='Aperta',
            Enunciato=statement, **{'Limiti': spec['limits']}))
    experiment = put('experiments', 'suite', Nome='Task38 joint explicit-state ionic transition',
        Stato='Preregistrato', Ipotesi=claims, Descrizione=spec['scope'],
        **{'Obiettivo informativo': 'Joint state-conditioned learnability, paired width and budget, recurrence gap, phase and memory probes. No fresh confirmation.'})
    partitions = [ref('partitions-giada-task37-ionic-playground-'+r+'-v1') for r in ('fit', 'development', 'stress_diagnostic')]
    protocol = put('protocols', 'suite', Nome='Task38 prospective protocol', Versione='v1',
        Modalità='Ricomposizione', Esperimento=[experiment], Procedura=json.dumps(spec, ensure_ascii=False),
        **{'Partizioni autorizzate': partitions, 'Hash preregistrazione': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
           'Artefatto preregistrazione': [artifacts[spec_path.relative_to(repo).as_posix()]],
           'Criteri di successo': spec['quality'], 'Regole di arresto': spec['preflights']+' '+spec['limits'], 'Registrato il': '2026-10-17'})
    implementation = put('implementations', 'worker', Nome='Task38 member-vectorized bounded joint transition',
        **{'Repository URL': 'https://github.com/Zagred47/giada', 'Commit': 'Full launch commit pinned in Kaggle notebook and worker revision.json; scientific source hashes registered prospectively.',
           'Percorso entrypoint': 'scripts/run_task38_joint_transition.py', 'Ambiente': spec['runtime'],
           'Artefatti codice': list(artifacts.values())})
    for width in spec['architectures']:
        model = put('models', 'w'+str(width), Nome=f'Task38 joint explicit18 width{width}', Ruolo='Surrogate',
            Descrizione=spec['architecture']+f' Width={width}. '+spec['phase'])
        put('arms', 'w'+str(width), Nome=f'Task38 width{width}, three paired seeds', Ruolo='Trattamento',
            Protocollo=[protocol], Modello=[model], Implementazione=[implementation], Partizioni=partitions,
            **{'Configurazione residua': json.dumps(dict(width=width, seeds=spec['seeds'], checkpoints=spec['checkpoints']))})
    for name, description in [('formula', 'Canonical exact ionic kinetics: numeric reference, not a learned candidate.'),
        ('local161', 'Frozen IE-P6 discrete coefficient map, 3 seeds; canonical SK is a partial replacement control.')]:
        baseline = put('models', name, Nome='Task38 baseline '+name, Ruolo='Baseline', Descrizione=description)
        put('arms', name, Nome='Task38 reference '+name, Ruolo='Baseline', Protocollo=[protocol],
            Modello=[baseline], Implementazione=[implementation], Partizioni=partitions,
            **{'Configurazione residua': description})
    for name in ('gate_rmse', 'current_nrmse', 'long_horizon_failure'):
        metric = put('metrics', name, Nome='Task38 '+name, Formula='Task37 evaluator; per channel, case and horizon, thresholds unchanged', Direzione='Minimizzare')
        put('evaluations', name, Nome='Task38 '+name, Versione='v1', Protocollo=[protocol],
            Metrica=[metric], Partizioni=partitions, Ruolo='Diagnostica', Target=spec['quality'],
            **{'Aggregazione e pesi': 'Per seed/case/channel; conjunction for development pass, stress separate. No mean-only behavioral claim.',
               'Gestione classi assenti': 'Not evaluable is not a pass for unsupported events.'})
    launch_path = repo/'experiments/results/task38_protocol_v1/kaggle_launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        assert launch['status_at_verification'] == 'RUNNING' and launch['published_source_full_revision_verified']
        launch_art = put('artifacts', 'kaggle-launch', Nome='Task38 verified Kaggle launch', Tipo='Log',
            Percorso=launch_path.relative_to(repo).as_posix(), Versione='v1',
            **{'SHA-256': hashlib.sha256(launch_path.read_bytes()).hexdigest(), 'Dimensione byte': launch_path.stat().st_size})
        for width in spec['architectures']:
            arm = ref(f'arms-giada-task38-joint-transition-w{width}-v1')
            put('runs', 'kaggle-w'+str(width), Nome=f'Task38 width{width} paired seed group, Kaggle v{launch["version"]}',
                Stato='In corso', Braccio=[arm], **{'Artefatti prodotti': [launch_art],
                'Configurazione effettiva': json.dumps(dict(launch=launch, width=width, seeds=spec['seeds'])),
                'Hardware e ambiente': spec['runtime']+' '+launch['machine_shape'],
                'Validità tecnica': 'Submission/source pin and RUNNING status verified; CUDA preflights and scientific outcome pending.'})
        incident_path = launch_path.parent/'cuda_preflight_incident_v1.json'
        if incident_path.exists():
            incident = json.loads(incident_path.read_text())
            ia = put('artifacts', 'cuda-preflight-incident-v1', Nome='Task38 v1 CUDA preflight incident',
                Tipo='Log', Percorso=incident_path.relative_to(repo).as_posix(), Versione='v1',
                **{'SHA-256': hashlib.sha256(incident_path.read_bytes()).hexdigest(), 'Dimensione byte': incident_path.stat().st_size})
            failed = put('runs', 'v1-cuda-preflight', Nome='Task38 v1 technical stop before all training',
                Stato='Fallita', Braccio=[ref('arms-giada-task38-joint-transition-w32-v1')],
                **{'Artefatti prodotti': [ia], 'Configurazione effettiva': json.dumps(incident),
                   'Validità tecnica': 'Technical failure before either width training started; not scientific NO-GO. Second width training was not reached.'})
            put('anomalies', 'v1-cuda-host-index', Nome='Task38 host-index copy during CUDA graph capture',
                Stato='Spiegata', Run=[failed], Atteso='Device-only recurrence capture',
                Osservato=incident['error'], **{'Possibili cause': incident['attribution']})
    mirror.local_upsert('actions', {'Codice stabile': 'actions-giada-ionic-abstraction-20261017-task38-v1',
        'Nome': 'Task38 — Aggiornamento appreso a stato esplicito', 'Stato': 'Pronta',
        'Descrizione': 'Protocol registered; launch CUDA run after tests and push. Await actual results; no scientific GO claimed.'})
    print(json.dumps(dict(stage='validated_batch_ready', records=len(ids))), flush=True)
    mirror.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json', mirror.export_snapshot())
    verification = mirror.verify(); assert verification['valid']
    folder = repo/'experiments/results/task38_protocol_v1'; folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'memory_receipt.json', dict(valid=True, records=ids, experiment=experiment,
        protocol=protocol, implementation=implementation, verification=verification,
        result_observations_created=0, airtable_accessed=False))
    print(json.dumps(dict(valid=True, records=len(ids), airtable_accessed=False)), flush=True)


if __name__ == '__main__': main()
