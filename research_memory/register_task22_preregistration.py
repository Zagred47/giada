"""Task22 Ih/Im/Nap_h single-gate contract, local mirror only."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror as StagedMirror


def main():
    mirror = StagedMirror()
    path = ROOT.parent / 'experiments/task22_controlled_gate_sharing.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    prerequisite = mirror.query("SELECT record_id FROM v_records WHERE stable_code='decisions-giada-roadmap-task21-to22-v1'")['rows']
    assert len(prerequisite) == 1
    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-roadmap-task22-{suffix}-v1', **fields})['record_id']
    claims = []
    for key, text in cfg['hypotheses'].items():
        claims.append(put('claims', key, **{'Nome': 'Task22 ' + key, 'Tipo': 'Ipotesi', 'Stato': 'Aperta',
            'Enunciato': text, 'Condizioni di falsificazione': 'Contrasti appaiati e criteri registrati; promozione congiuntiva per canale/seed/dominio.', 'Limiti': cfg['limits']}))
    experiment = put('experiments', 'matrix', **{'Nome': 'GIADA Task22 originale — Ih/Im/Nap_h', 'Tipo': 'Esplorativo', 'Stato': 'Preregistrato',
        'Ipotesi': claims, 'Descrizione': '18sistemi:3architetture×2width×3seed,3gate ciascuno; audit, estremi stato, rollout e identità scambiata.',
        'Obiettivo informativo': 'Condivisione pesi; tradeoff parametri/accuratezza; negative transfer; identità; scaling.'})
    protocol = put('protocols', 'paired', **{'Nome': 'Task22 matrice appaiata', 'Esperimento': [experiment], 'Versione': 'v1',
        'Modalità': 'Componente isolato', 'Procedura': json.dumps(cfg, ensure_ascii=False), 'Criteri di successo': cfg['promotion'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Oracle/equivalenza/hash invalidi: fallimento esecutivo. Testfresh dopo freeze; soglie non cambiate dopo esito.'})
    evaluations = []
    for key, threshold in cfg['gates'].items():
        metric = put('metrics', key, **{'Nome': 'Task22 ' + key, 'Famiglia': 'Fisica' if key == 'occupancy_violations' else 'Regressione',
            'Direzione': 'Minimizzare', 'Unità': 'adimensionale', 'Formula': 'controlled_gate_sharing.metrics.' + key,
            'Casi degeneri': 'Non finito o tau≤0 sempre fallimento; ogni gate valutato separatamente.'})
        evaluations.append(put('evaluations', key, **{'Nome': 'Task22 ' + key, 'Metrica': [metric], 'Protocollo': [protocol],
            'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': str(threshold), 'Popolazione e regioni': 'Ih/Im/Nap_h uniforme/coda/statoestremo;3seed percanale',
            'Orizzonte e finestre': 'dt.025/.1/.5/1/5/25/100/500/1000/5000/10000ms aVcostante', 'Aggregazione e pesi': 'Criterio congiuntivo; OOD separato.'}))
    for key, threshold in cfg['rollout_gates'].items():
        evaluations.append(put('evaluations', 'rollout-' + key, **{'Nome': 'Task22 rollout ' + key, 'Metrica': [
            next(mirror_row['id'] for mirror_row in mirror.staged['tables'][mirror.contract.tables['metrics']['id']]
                 if mirror_row['cellValuesByFieldId'].get(mirror.field_id('metrics', 'Codice stabile')) == f'metrics-giada-roadmap-task22-{key}-v1')],
            'Protocollo': [protocol], 'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': str(threshold),
            'Popolazione e regioni': '128nuoveV×2statiiniziali; tutti3seed/canale', 'Orizzonte e finestre': '1/10/100/1000/10000 passi da1ms aVcostante',
            'Aggregazione e pesi': 'Ogni orizzonte deve passare.'}))
    arms = []
    for channel in cfg['families']:
        for width in cfg['widths']:
            for objective in cfg['objectives']:
                arms.append(put('arms', f'{channel}-w{width}-{objective}', **{'Nome': f'Task22 {channel} w{width} {objective}',
                    'Ruolo': 'Baseline' if channel == 'independent' else 'Trattamento', 'Protocollo': [protocol],
                    'Descrizione': cfg['architecture'], 'Configurazione residua': 'Seed indipendenti; stesso stream e loss perbundle; condivisione interna secondo famiglia. Parametri perbundle espliciti.'}))
    for index, (key, text) in enumerate(cfg['hypotheses'].items()):
        prediction = put('predictions', key, **{'Nome': 'Task22 ' + key, 'Ipotesi': [claims[index]], 'Origine': 'Preregistrata',
            'Risultato atteso': text, 'Specifica di valutazione': [evaluations[0]], 'Soglia o intervallo': json.dumps(cfg['gates'])})
        put('contrasts', key, **{'Nome': 'Task22 ' + key, 'Bracci': arms, 'Predizioni': [prediction], 'Specifiche di valutazione': evaluations,
            'Contrasto e coefficienti': text, 'Confondenti controllati': 'Stesse V/stato/streamseed/minibatch; dt identico in tutte famiglie; stati indipendenti; commonselection3seed; controllonegativo appaiato.',
            'Soglia interpretativa': cfg['promotion'], 'Correzione confronti multipli': 'No pvalue; soglie ingegneristiche preregistrate.'})
    for role in ('fit', 'development', 'fresh'):
        put('blocks', role, **{'Nome': 'Task22 ' + role, 'Protocollo': [protocol], 'Seed': json.dumps(cfg['data_seeds'][role]),
            'Descrizione': 'Streamdisgiunti; griglie disgiunte; freshmaterializzato dopo freeze.', 'Regola di appaiamento': 'V/stato/dt comuni a tutte famiglie. Routing/head o onehot danno identità nota in tutte architetture.'})
    put('artifacts', 'preregistration', **{'Nome': 'Task22 preregistrazione', 'Tipo': 'Report', 'Percorso': path.relative_to(ROOT.parent).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    from .task21_reference_followup import register
    register(mirror)
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps(dict(valid=True, hypotheses=len(claims), learned_arms=len(arms), airtable_accessed=False)))


if __name__ == '__main__':
    main()
