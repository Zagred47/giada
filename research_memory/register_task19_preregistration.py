"""Task19 Ih/Im single-gate contract, local mirror only."""
import hashlib
import json
from .mirror import ROOT, write_json
from .register_task18d_result import StagedMirror


def main():
    mirror = StagedMirror()
    path = ROOT.parent / 'experiments/task19_single_gate_transfer.json'
    cfg = json.loads(path.read_text())
    prerequisite = mirror.query("SELECT record_id FROM v_records WHERE stable_code='decisions-giada-task18d-to19-v1'")['rows']
    assert len(prerequisite) == 1
    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-roadmap-task19-{suffix}-v1', **fields})['record_id']
    claims = []
    for key, text in cfg['hypotheses'].items():
        claims.append(put('claims', key, **{'Nome': 'Task19 ' + key, 'Tipo': 'Ipotesi', 'Stato': 'Aperta',
            'Enunciato': text, 'Condizioni di falsificazione': 'Contrasti appaiati e criteri registrati; promozione congiuntiva per canale/seed/dominio.', 'Limiti': cfg['limits']}))
    experiment = put('experiments', 'matrix', **{'Nome': 'GIADA Task19 originale — Ih/Im', 'Tipo': 'Esplorativo', 'Stato': 'Preregistrato',
        'Ipotesi': claims, 'Descrizione': '24modelli:2canali×2obiettivi×2width×3seed; audit, LUT, estremi stato, rollout e latenzaCUDA.',
        'Obiettivo informativo': 'Trasferibilità gate singolo; identificabilità rate; mini scaling; stabilità aVcostante.'})
    protocol = put('protocols', 'paired', **{'Nome': 'Task19 matrice appaiata', 'Esperimento': [experiment], 'Versione': 'v1',
        'Modalità': 'Componente isolato', 'Procedura': json.dumps(cfg, ensure_ascii=False), 'Criteri di successo': cfg['promotion'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Oracle/equivalenza/hash invalidi: fallimento esecutivo. Testfresh dopo freeze; soglie non cambiate dopo esito.'})
    evaluations = []
    for key, threshold in cfg['gates'].items():
        metric = put('metrics', key, **{'Nome': 'Task19 ' + key, 'Famiglia': 'Fisica' if key == 'occupancy_violations' else 'Regressione',
            'Direzione': 'Minimizzare', 'Unità': 'adimensionale', 'Formula': 'single_gate_transfer.measurements.' + key,
            'Casi degeneri': 'Non finito o tau≤0 sempre fallimento; openRMSE=gateRMSE per singologate.'})
        evaluations.append(put('evaluations', key, **{'Nome': 'Task19 ' + key, 'Metrica': [metric], 'Protocollo': [protocol],
            'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': str(threshold), 'Popolazione e regioni': 'Ih/Im uniforme/coda/statoestremo;3seed percanale',
            'Orizzonte e finestre': 'dt.025/.1/.5/1/5/25/100ms aVcostante', 'Aggregazione e pesi': 'Criterio congiuntivo; OOD separato.'}))
    for key, threshold in cfg['rollout_gates'].items():
        evaluations.append(put('evaluations', 'rollout-' + key, **{'Nome': 'Task19 rollout ' + key, 'Metrica': [
            next(mirror_row['id'] for mirror_row in mirror.staged['tables'][mirror.contract.tables['metrics']['id']]
                 if mirror_row['cellValuesByFieldId'].get(mirror.field_id('metrics', 'Codice stabile')) == f'metrics-giada-roadmap-task19-{key}-v1')],
            'Protocollo': [protocol], 'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': str(threshold),
            'Popolazione e regioni': '128nuoveV×2statiiniziali; tutti3seed/canale', 'Orizzonte e finestre': '1/10/100/1000 passi da1ms aVcostante',
            'Aggregazione e pesi': 'Ogni orizzonte deve passare.'}))
    current = put('metrics', 'current_rmse_ma_cm2', **{'Nome': 'Task19 errore corrente densità', 'Famiglia': 'Regressione', 'Direzione': 'Minimizzare',
        'Unità': 'mA/cm²', 'Formula': 'RMSE[gbar*(pred-target)*(V-E)],gbar1e-5S/cm²', 'Parametri': 'E_Ih−45mV,E_Im−85mV; correnteanalitica'})
    put('evaluations', 'current_rmse_ma_cm2', **{'Nome': 'Task19 corrente diagnostica', 'Metrica': [current], 'Protocollo': [protocol],
        'Versione': 'v1', 'Ruolo': 'Secondaria', 'Popolazione e regioni': 'Stesse tuplefresh; unità dichiarate', 'Aggregazione e pesi': 'Non usata per selezione o nuova soglia.'})
    arms = []
    for channel in cfg['channels']:
        for width in cfg['widths']:
            for objective in cfg['objectives']:
                arms.append(put('arms', f'{channel}-w{width}-{objective}', **{'Nome': f'Task19 {channel} w{width} {objective}',
                    'Ruolo': 'Trattamento' if objective == 'rate_supervised' else 'Controllo negativo', 'Protocollo': [protocol],
                    'Descrizione': cfg['architecture'], 'Configurazione residua': 'Pesi indipendenti; stessi dati,minibatch,seed,inizializzazioneentrocapacità.'}))
    for size in (513, 2049):
        put('arms', f'lut{size}', **{'Nome': f'Task19 LUT{size} f64', 'Ruolo': 'Baseline', 'Protocollo': [protocol],
            'Descrizione': 'Interpolazione lineare percanale su[-135,75]mV; correntegateanalitica; OODnon estrapolato.'})
    for index, (key, text) in enumerate(cfg['hypotheses'].items()):
        prediction = put('predictions', key, **{'Nome': 'Task19 ' + key, 'Ipotesi': [claims[index]], 'Origine': 'Preregistrata',
            'Risultato atteso': text, 'Specifica di valutazione': [evaluations[0]], 'Soglia o intervallo': json.dumps(cfg['gates'])})
        put('contrasts', key, **{'Nome': 'Task19 ' + key, 'Bracci': arms, 'Predizioni': [prediction], 'Specifiche di valutazione': evaluations,
            'Contrasto e coefficienti': text, 'Confondenti controllati': 'Stessi input/streamseed/minibatch; stati indipendenti; commonselection3seed; controllonegativo appaiato.',
            'Soglia interpretativa': cfg['promotion'], 'Correzione confronti multipli': 'No pvalue; soglie ingegneristiche preregistrate.'})
    for role in ('fit', 'development', 'fresh'):
        put('blocks', role, **{'Nome': 'Task19 ' + role, 'Protocollo': [protocol], 'Seed': json.dumps(cfg['data_seeds'][role]),
            'Descrizione': 'Streamdisgiunti; griglie disgiunte; freshmaterializzato dopo freeze.', 'Regola di appaiamento': 'Tuple numeriche comuni tra canali/obiettivi/width/seed.'})
    put('artifacts', 'preregistration', **{'Nome': 'Task19 preregistrazione', 'Tipo': 'Report', 'Percorso': path.relative_to(ROOT.parent).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps(dict(valid=True, hypotheses=len(claims), learned_arms=len(arms), airtable_accessed=False)))


if __name__ == '__main__':
    main()
