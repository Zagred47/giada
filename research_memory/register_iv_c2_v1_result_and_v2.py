"""Record IV-C2 v1 native evidence and preregister independent v2 frequencies."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    result_path = repo / 'experiments/results/iv_c2_kaggle_21e9908/final_report.json'
    result = json.loads(result_path.read_text())
    spec_path = repo / 'experiments/iv_c2_native_distribution_confirmation.json'
    spec = json.loads(spec_path.read_text())
    if (not result['valid'] or not result['iv_c2_passed'] or result['task33_authorized']
            or hashlib.sha256(result_path.read_bytes()).hexdigest() != spec['parent_report_sha256']):
        raise RuntimeError('IV-C2 v1 evidence or v2 parent hash changed')
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c2-{suffix}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'native-replay-report', **{
        'Nome': 'IV-C2 native event/RNG/restart report', 'Tipo': 'Report',
        'Percorso': result_path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'Dimensione byte': result_path.stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'native-replay', **{
        'Nome': 'IV-C2 v1 native paired replay', 'Stato': 'Completata',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle process exit 0, canonical teacher, 64 cases and all registered gates passed.'})
    put('findings', 'exact-replay', **{
        'Nome': 'Canonical EMS stochastic replay exact over 64 paired cells',
        'Esito': 'Positivo',
        'Risultato': 'Zero release/state/RNG/restart mismatches; negative controls passed.',
        'Limitazioni': 'Native release frequency over independent seeds remains to be confirmed by v2.'})
    claim = put('claims', 'native-frequency', **{
        'Nome': 'Negexp Random123 determines native release and recovery frequencies',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'For a recovered EMS synapse P(release)=1-exp(-Use). For a depressed synapse P(recover)=exp(-exp(-t/Dep)); release requires a second independent draw.',
        'Limiti': 'Isolated canonical EMS synapses, imposed voltage and no autonomous-cell feedback.',
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'native-frequency', **{
        'Nome': 'IV-C2 v2 independent native release-frequency confirmation',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [claim],
        'Descrizione': '256 fresh Random123 seeds per cell; two mechanisms, two Use values, two Dep values, initial recovered/depressed states.',
        'Obiettivo informativo': 'Verify release and recovery frequencies in NEURON itself, not only the RNG marginal.'})
    put('protocols', 'native-frequency', **{
        'Nome': 'IV-C2 v2 native Monte Carlo protocol',
        'Esperimento': [experiment], 'Versione': 'v2',
        'Modalità': 'Componente isolato',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'No IV-C3 or Task33 promotion; API failure is inconclusive.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v1_result_recorded': True,
                      'v2_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
