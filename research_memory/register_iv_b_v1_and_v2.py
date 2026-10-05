"""Preserve first IV-B outcome and preregister the v2 timing correction locally."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/iv_b_kaggle_b40e07a'
    archive = folder / 'artifact_bundle.zip'
    report = json.loads((folder / 'final_report.json').read_text())
    audit = json.loads((folder / 'semantic_audit.json').read_text())
    cfg_path = repo / 'experiments/iv_b1_b2_calcium_prerequisite_v2.json'
    cfg = json.loads(cfg_path.read_text())
    assert report['iv_b1_valid'] and not report['iv_b2_valid']
    assert audit['source_archive_sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert audit['maximum_updated_cai_one_step_rmse'] < 1e-12
    assert 'late_single' in cfg['b2_protocols'] and 'triplet' in cfg['b2_protocols']
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-b-v1-v2-{key}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'first-run', **{
        'Nome': 'IV-B1/B2 first native run and timing discrepancy', 'Tipo': 'Report',
        'Percorso': archive.relative_to(repo).as_posix(),
        'SHA-256': audit['source_archive_sha256'],
        'Dimensione byte': archive.stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'first-run', **{
        'Nome': 'IV-B first native test: B1 pass, B2 old-cai reference mismatch',
        'Stato': 'Completata', 'Seed': 'deterministic 48 B1 + 8 B2 episodes',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Canonical teacher 074c466, NEURON 8.2.7, source hashes, archived ZIP CRC and independently recomputed native one-step timing.'})
    metric = put('metrics', 'updated-cai-one-step', **{
        'Nome': 'IV-B native SK gate one-step identity under updated cai',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'gate occupancy',
        'Formula': 'Maximum across B2 episodes of RMS native z[t+1] minus exact SK update driven by native cai[t+1].'})
    evaluation = put('evaluations', 'updated-cai-one-step', **{
        'Nome': 'IV-B post-hoc timing attribution', 'Metrica': [metric],
        'Versione': 'v1', 'Ruolo': 'Diagnostica', 'Target': 'none—post-hoc',
        'Aggregazione e pesi': 'Worst of eight B2 episodes, all steps.'})
    observation = put('observations', 'updated-cai-one-step', **{
        'Nome': 'IV-B native updated-cai SK identity',
        'Valore': audit['maximum_updated_cai_one_step_rmse'],
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact], 'Strato o sottogruppo': 'post-hoc B2 timing',
        'Descrizione': 'V1 B2 remains failed against its registered old-cai reference; this explains the mechanism but does not retroactively change its decision.'})
    finding = put('findings', 'timing', **{
        'Nome': 'IV-B1 passes and IV-B2 exposes sequential CaDynamics-to-SK timing',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': '48/48 B1 calcium trajectories agree with native. B2 fails v1 gate threshold because SK reads newly updated cai within the same causal step; updated-cai one-step native identity is near machine precision.',
        'Limitazioni': 'Post-hoc attribution; no v2 confirmation yet, no closed electrical feedback or speed evidence.'})
    claim = put('claims', 'v2-causal-order', **{
        'Nome': 'Causal updated-calcium ordering composes CaDynamics and SK',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': cfg['causal_discrete_order'],
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'v2-confirmation', **{
        'Nome': 'GIADA IV-B1/B2 updated-calcium confirmation v2',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Prospectively confirm corrected causal within-step timing on two new independent calcium-current schedules without relaxing numerical thresholds.'})
    put('protocols', 'v2-confirmation', **{
        'Nome': 'IV-B1/B2 v2 timing confirmation', 'Esperimento': [experiment],
        'Versione': 'v2', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'B1 must pass before B2 is interpreted; B2 must pass before Task32 feedback is prepared. Keep first run unchanged.'})
    put('decisions', 'await-v2', **{
        'Nome': 'IV-B v1 discrepancy requires independent v2 confirmation',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'The reference timing, not calcium dynamics, caused the v1 B2 mismatch. Fix the causally ordered reference and add two new pulse schedules.',
        'Condizioni di revisione': 'Review v2 native B1/B2 thresholds and perturbation informativeness; Task32 electrical feedback remains unauthorized until then.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v1_b2_promoted': False,
                      'v2_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
