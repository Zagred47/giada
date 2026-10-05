"""Preregister original Task28 in the exclusive local SQLite mirror."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task28_ionic_block_teacher_forced.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    from src.giada_teacher.ionic_block_teacher_forced import verify_parent
    verify_parent(repo, cfg)
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-roadmap-task28-{key}-v1', **fields,
        })['record_id']

    claim = put('claims', 'composition', **{
        'Nome': 'Task28 complete teacher-forced ionic composition',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'Cinque rate surrogate congelati e sei formule canoniche formano un blocco ionico locale a 11 meccanismi sotto V e cai imposti senza regressioni di gate o corrente.',
        'Limiti': cfg['limits'],
        'Condizioni di falsificazione': cfg['promotion'],
    })
    compute_claim = put('claims', 'compute', **{
        'Nome': 'Task28 Gate D compute materiality', 'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'Il blocco ibrido riduce il compute rispetto alle formule complete su hardware e batch appaiati.',
        'Limiti': 'Non dedurre speedup dal conteggio parametri o dal solo errore. Un benchmark non comparabile mantiene Gate D incompleto.',
        'Condizioni di falsificazione': f"Riduzione < {cfg['material_compute_reduction_minimum']:.0%} o benchmark non confrontabile.",
    })
    experiment = put('experiments', 'hybrid', **{
        'Nome': 'GIADA original Task28 — full ionic teacher-forced hybrid',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [claim, compute_claim],
        'Descrizione': cfg['input_contract'] + '\n' + cfg['current_contract'],
        'Obiettivo informativo': 'Verificare la composizione di tutti gli 11 meccanismi ionici e distinguere il GO teacher-forced da Gate D e dal closed loop.',
    })
    protocol = put('protocols', 'frozen', **{
        'Nome': 'Task28 frozen 11-channel imposed-V/Ca',
        'Esperimento': [experiment], 'Versione': 'v1',
        'Modalità': 'Ricomposizione', 'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['promotion'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Prerequisiti, native oracle o hash falliti: errore tecnico. Fresh oltre soglia: NO-GO scientifico, senza riselezione.',
    })
    metric_specs = [('gate_rmse', cfg['learned_gate_rmse_limit']),
                    ('current_individual_normalized_rmse', cfg['learned_current_normalized_rmse_limit']),
                    ('current_total_normalized_rmse', cfg['learned_total_current_normalized_rmse_limit']),
                    ('path_gate_rmse', cfg['learned_path_gate_rmse_limit']),
                    ('path_current_normalized_rmse', cfg['learned_path_current_normalized_rmse_limit'])]
    evaluations = []
    for name, limit in metric_specs:
        metric = put('metrics', name, **{
            'Nome': 'Task28 ' + name, 'Famiglia': 'Regressione',
            'Direzione': 'Minimizzare', 'Unità': 'adimensionale',
            'Formula': 'Worst per-channel or per-panel RMSE; exact normalization and signed-current convention in preregistered code.',
        })
        evaluations.append(put('evaluations', name, **{
            'Nome': 'Task28 ' + name, 'Metrica': [metric], 'Protocollo': [protocol],
            'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': str(limit),
            'Aggregazione e pesi': 'Conjunction across all model seeds, independent support seeds, channel panels and imposed paths.',
        }))
    arms = []
    for family in cfg['frozen_families']:
        for arm in cfg['frozen_arms']:
            key = family + '-' + arm
            model = put('models', key, **{
                'Nome': 'Task28 frozen hybrid ' + key, 'Ruolo': 'Surrogate',
                'Versione': 'v1',
                'Descrizione': 'Frozen Task27 five-channel neural rate module plus six canonical formula mechanisms; no retraining.',
                'Contratto di stato': '18 explicit gates in fixed order; each updated independently under imposed voltage and calcium.',
                'Contratto di input e output': cfg['input_contract'] + '\n' + cfg['current_contract'],
            })
            arms.append(put('arms', key, **{
                'Nome': 'Task28 ' + key,
                'Ruolo': 'Baseline' if arm == 'none' else 'Trattamento',
                'Protocollo': [protocol], 'Modello': [model],
                'Descrizione': 'Frozen Task27 arm; six formula channels identical across arms.',
                'Configurazione residua': 'No training or fresh selection; all four frozen arms reported.',
            }))
    for role, seeds in [('development', [cfg['development_seed']]), ('fresh', cfg['fresh_seeds'])]:
        put('blocks', role, **{
            'Nome': 'Task28 ' + role, 'Protocollo': [protocol],
            'Seed': json.dumps(seeds),
            'Descrizione': cfg['selection'],
            'Regola di appaiamento': 'Identical imposed V/cai/gate-state samples and panel parameters across every frozen arm.',
        })
    prediction = put('predictions', 'composition', **{
        'Nome': 'Task28 ionic composition prediction', 'Ipotesi': [claim],
        'Origine': 'Preregistrata', 'Risultato atteso': cfg['promotion'],
        'Specifica di valutazione': [evaluations[0]],
        'Soglia o intervallo': cfg['promotion'],
    })
    put('contrasts', 'composition', **{
        'Nome': 'Task28 frozen four-arm composition', 'Bracci': arms,
        'Predizioni': [prediction], 'Specifiche di valutazione': evaluations,
        'Contrasto e coefficienti': 'Both versus none within each Task27 family; all 11 mechanisms jointly, error attributed by channel and panel.',
        'Confondenti controllati': cfg['selection'],
        'Soglia interpretativa': cfg['promotion'],
        'Correzione confronti multipli': 'No fresh-driven selection; conjunction across families and panels.',
    })
    for filename in ('experiments/task28_ionic_block_teacher_forced.json',
                     'src/giada_teacher/ionic_block_teacher_forced.py',
                     'notebooks/28_roadmap_ionic_block_teacher_forced.ipynb',
                     'scripts/run_roadmap_task28.py'):
        artifact = repo / filename
        put('artifacts', artifact.stem, **{
            'Nome': 'Task28 ' + artifact.name, 'Tipo': 'Report',
            'Percorso': filename, 'SHA-256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'Dimensione byte': artifact.stat().st_size, 'Versione': 'v1',
        })
    put('decisions', 'scope', **{
        'Nome': 'Task28 original ionic block prepared after Task27 GO',
        'Esito': 'Continuare',
        'Motivazione': 'Task27 both and none pass 3/3 in both families. Compose the five frozen learned channels with all remaining canonical formula channels under imposed V/cai.',
        'Condizioni di revisione': 'Task28 pass does not itself complete Gate D, IV-A2, CaDynamics or autonomous-voltage authorization.',
    })
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'channels': len(cfg['channels']), 'frozen_arms': len(arms),
                      'airtable_accessed': False}))


if __name__ == '__main__':
    main()
