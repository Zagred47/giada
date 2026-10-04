"""Register verified Task18d results and all scalar development/fresh metrics locally."""
import hashlib
import json
import subprocess
import zipfile
from .mirror import Mirror, ROOT, write_json

KEYS = ('gate_rmse', 'gate_max_error', 'inf_rmse', 'log_tau_rmse', 'open_rmse', 'occupancy_violations')


class StagedMirror(Mirror):
    """Use local_upsert validation/reciprocal links, import the complete batch once."""
    def __init__(self):
        super().__init__()
        self.staged = super().export_snapshot()

    def export_snapshot(self):
        return self.staged

    def import_snapshot(self, snapshot):
        self.staged = snapshot
        return {}

    def commit_batch(self):
        return super().import_snapshot(self.staged)


def main():
    repo = ROOT.parent
    archive = repo / 'artifacts/giada_task18d_potassium_38e6431_2cff4e72.zip'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == '968a4fc9c6f1c8063bb872be92de7ccf7ea7a38ff35efce0d8dc3e678a948f2c'
    root = repo / 'experiments/results/task18d_kaggle_38e6431'
    root.mkdir(parents=True, exist_ok=True)
    names = ('final_report.json', 'native_audit.json', 'equivalence_preflight.json',
             'selection_freeze.json', 'development_ladder.json', 'paired_development_contrasts.json',
             'code_provenance.json', 'process_status.json', 'run_contract.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze = json.loads(z.read('selection_freeze.json'))
        claimed = freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == claimed
        assert not freeze['fresh_accessed']
        assert all(hashlib.sha256(z.read(k)).hexdigest() == v for k, v in freeze['checkpoint_hashes'].items())
        provenance = json.loads(z.read('code_provenance.json'))
        assert not provenance['dirty_runtime']
        for path, expected in provenance['sources'].items():
            assert hashlib.sha256(subprocess.check_output(['git', 'show', provenance['code_revision'] + ':' + path], cwd=repo)).hexdigest() == expected
        for name in names:
            (root / name).write_bytes(z.read(name))
    report = json.loads((root / 'final_report.json').read_text())
    ladder = json.loads((root / 'development_ladder.json').read_text())
    assert report['valid'] and report['potassium_passed'] and report['task19_authorized']
    assert not report['fresh_used_for_selection']
    assert all(json.loads((root / n).read_text())['valid'] for n in ('native_audit.json', 'equivalence_preflight.json'))
    write_json(root / 'archive_audit.json', dict(valid=True, archive_sha256=digest,
        CRC_valid=True, source_hashes_valid=True, freeze_and_checkpoint_hashes_valid=True))
    mirror = StagedMirror()
    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-{suffix}-v1', **fields})['record_id']
    def ref(table, suffix):
        rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code='" + f'{table}-giada-{suffix}-v1' + "'")['rows']
        assert len(rows) == 1
        return rows[0]['record_id']
    arts = []
    for path in [*(root / n for n in names), root / 'archive_audit.json', archive]:
        arts.append(put('artifacts', 'task18d-result-' + path.stem, **{
            'Nome': 'Task18d ' + path.name, 'Tipo': 'Report', 'Percorso': path.relative_to(repo).as_posix(),
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size, 'Versione': 'v1'}))
    evaluations = {k: ref('evaluations', 'task18d-' + k) for k in KEYS}
    observations = {'fresh': [], 'parent': [], 'development': []}
    def record_metrics(group, suffix, metrics, run):
        for domain, values in metrics.items():
            # Each voltage has seven dt and two initial states. Four off-grid neighbors.
            count = (205 if group == 'development' else 405) * 14 if domain == 'state_extrema' else (2048 if group == 'development' or domain == 'ood_negative' else 4096)
            for key in KEYS:
                observations[group].append(put('observations', f'task18d-{suffix}-{domain}-{key}', **{
                    'Nome': f'Task18d {suffix} {domain} {key}', 'Valore': values[key], 'Numerosità': count,
                    'Run': [run], 'Specifica di valutazione': [evaluations[key]], 'Artefatti dettagliati': arts,
                    'Strato o sottogruppo': domain,
                    'Descrizione': f'{group}; errori adimensionali, non mV; OOD negativo diagnostico. Numerosità in tuple, non repliche indipendenti.'}))
    def run(suffix, seed, arm, block, config):
        return put('runs', 'task18d-' + suffix, **{'Nome': 'Task18d ' + suffix, 'Stato': 'Completata',
            'Braccio': [arm], 'Blocco': [ref('blocks', 'task18d-' + block)], 'Seed': str(seed),
            'Configurazione effettiva': config, 'Artefatti prodotti': arts,
            'Hardware e ambiente': json.dumps(report['environment']),
            'Validità tecnica': 'CRC, oracle, equivalenza, freeze, checkpoint e sorgenti verificati.'})
    selected_arm = ref('arms', 'task18d-feature-top0-lr0.003')
    for index, row in enumerate(report['results']):
        seed = row['seed']
        suffix = f'result-s{seed}'
        rid = run(suffix, seed, selected_arm, 'fresh', 'feature width32,1284 parametri effettivi,step30000; selezione comune solo development')
        record_metrics('fresh', suffix, row['metrics'], rid)
        suffix = f'parent-s{seed}'
        rid = run(suffix, seed, ref('arms', 'task18c-tail_and_kink-top0-lr0.003'), 'fresh', 'Parent18c congelato; medesime tuple fresh18d')
        record_metrics('parent', suffix, {d: v[index] for d, v in report['parent_same_fresh'].items()}, rid)
        print(f'[SQLite18d] fresh e parent seed{seed}', flush=True)
    for index, row in enumerate(ladder):
        family = row['arm'].split('/')[0]
        suffix = f'dev-{family}-step{row["step"]}-s{row["seed"]}'
        rid = run(suffix, row['seed'], ref('arms', f'task18d-{family}-top0-lr0.003'), 'development',
                  f'{family};step{row["step"]};score={row["score"]}; effective_parameters={report["effective_parameter_counts"][family]}')
        record_metrics('development', suffix, row['metrics'], rid)
        print(f'[SQLite18d] development {index + 1}/{len(ladder)}', flush=True)
    experiment = ref('experiments', 'task18d-matrix')
    findings = []
    def finding(suffix, title, outcome, result, groups, limits):
        fid = put('findings', 'task18d-' + suffix, **{'Nome': title, 'Esito': outcome,
            'Esperimenti': [experiment], 'Osservazioni': sum((observations[g] for g in groups), []),
            'Risultato': result, 'Limitazioni': limits})
        findings.append(fid)
        return fid
    success = finding('confirmation', 'Task18d K_Pst confermato 3/3 seed', 'Positivo',
        'Feature |V+60|/100 selezionata a30k. Tutti3seed passano tutti4domini richiesti e tutte6metriche. Peggior gateRMSE .000819319759; massimo .007357536854<.01; margine descrittivo<.008 passato. Training9modelli158.73674242s suT4. Nessuna violazione occupazione.',
        ['fresh'], 'Canale isolato a V costante; griglia finita;3seed. Non prova sostituzione nel neurone completo, accuratezza con V variabile o speedup hardware. .008 è margine descrittivo preregistrato, non nuova soglia.')
    architecture = finding('architecture', 'Task18d struttura locale utile oltre continuazione smooth', 'Positivo',
        'A30k feature e split riducono RMSE e massimo kink/state_extrema rispetto smooth in tutti3seed, con pesi iniziali, dati e minibatch appaiati. Feature vince score peggiore-seed:1.86858 vs split2.13877 e smooth2.68769. Split utile localmente ma non selezionato. Budget crescente non migliora tutte le metriche monotonicamente.',
        ['development'], 'Confronto sul development usato per selezione; fresh conferma solo winner. Capacità1252/1284/1285 entro3%, non uguale. Nessuna prova di superiorità universale o limite assoluto smooth.')
    baseline = finding('matched-parent', 'Task18d confronto appaiato con parent18c e OOD', 'Misto',
        'Kink e state_extrema migliorano RMSE/massimo in tutti3seed. In-support RMSE seed43 peggiora .000380826→.000521909; coda seed43 peggiora .000372188→.000819320 e max .00266746→.00505057, entro soglie. OOD[-155,-135] RMSE migliora tutti3 ma resta .01233–.01381, massimo .14976–.15898: estrapolazione non risolta.',
        ['fresh', 'parent'], 'Parent e winner valutati sulle identiche tuplefresh18d; confronto col parent include trainingaggiuntivo e cambiamento struttura. OOD escluso dalla promozione già in preregistrazione.')
    extrema = finding('state-extrema', 'Task18d controllo degli stati iniziali estremi', 'Positivo',
        'Grigliafresh405tensioni×7dt×2stati=5670tuple;3/3seed passano. Per update HH a V,dt fissati errore di ciascun gate affine nello stato iniziale:0/1 delimitano errore assoluto per x in[0,1].',
        ['fresh'], 'La proprietà riguarda ciascun gate a V/dt fissati; non estende il bound a tensioni non campionate, corrente polinomiale, o rollout accoppiato.')
    evidence = []
    for key, fid, state, outcome, argument in (
        ('cusp_feature', architecture, 'Supportata nel dominio', 'Sostiene', 'Beneficio locale vs smooth a pari budget in3/3; freshwinner passa.'),
        ('branch_heads', architecture, 'Supportata nel dominio', 'Sostiene', 'Beneficio locale development3/3; non vince la selezione globale; nessuna confermafresh separata.'),
        ('budget', architecture, 'Supportata nel dominio', 'Sostiene', 'Checkpoint appaiati separano contributo struttura e continuazione; piùpassi non garantiscono miglioramento monotono.'),
        ('state_extrema', extrema, 'Supportata nel dominio', 'Sostiene', 'Proprietà affine e controllogrid; dominio e limiti espliciti.')):
        put('claims', 'task18d-' + key, **{'Stato': state})
        evidence.append(put('evidence', 'task18d-' + key, **{'Nome': 'Task18d evidenza ' + key,
            'Esito': outcome, 'Affermazione valutata': [ref('claims', 'task18d-' + key)],
            'Risultati a sostegno': [fid], 'Argomentazione': argument,
            'Limiti e spiegazioni alternative': 'Tre seed, capacità entro3% non identica; selection suldevelopment; nessuna universalità.', 'Data valutazione': '2026-10-04'}))
    put('decisions', 'task18d-to19', **{'Nome': 'Task18d autorizza passaggio alla Task19', 'Esito': 'Continuare',
        'Risultati': findings, 'Valutazioni delle evidenze': evidence,
        'Motivazione': 'Criteri preregistrati superati3/3, con margine e testfresh aperto solo dopo freeze.',
        'Condizioni di revisione': 'Mantenere scope K_Pst isolato; OOD aperto. Task19 autorizzata, non eseguita da questa registrazione.', 'Data': '2026-10-04'})
    put('experiments', 'task18d-matrix', **{'Stato': 'Concluso'})
    put('runs', 'task18d-kaggle-orchestration', **{'Stato': 'Completata', 'Artefatti prodotti': arts,
        'Validità tecnica': 'COMPLETE exit0; CRC/oracle/equivalence/freeze/checkpoints/clean pinned provenance validi; scientific GO3/3.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    expected = sum(map(len, observations.values()))
    rows = mirror.query("SELECT COUNT(*) AS n FROM v_records WHERE table_key='observations' AND stable_code LIKE 'observations-giada-task18d-%'")['rows']
    assert rows[0]['n'] == expected == 1044, rows
    print(json.dumps({'valid': True, 'observations': {k: len(v) for k, v in observations.items()}, 'findings': len(findings), 'evidence': len(evidence), 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
