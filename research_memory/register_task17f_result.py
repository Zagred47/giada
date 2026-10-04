"""Register completed independent Task17f, including each paired condition."""
from __future__ import annotations
import hashlib
import json
from .mirror import Mirror, ROOT, write_json


def main():
    mirror = Mirror()
    root = ROOT.parent / 'experiments/results/task17f_kaggle_1b87bdd'
    read = lambda name: json.loads((root / name).read_text())
    report, audit = read('final_report.json'), read('archive_audit.json')
    pairs, effects = read('paired_metrics.json'), read('paired_effects.json')
    assert audit['valid'] and report['gate_c_authorized'] and report['task18_authorized']
    def ref(table, suffix):
        code = f'{table}-giada-task17f-event-support-{suffix}-v1'
        rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code='" + code + "'")['rows']
        if len(rows) != 1: raise RuntimeError('Task17f prerequisite missing: ' + code)
        return rows[0]['record_id']
    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {'Codice stabile':
            f'{table}-giada-task17f-independent-{suffix}-v1', **fields})['record_id']
    artifacts = []
    for path in sorted(root.iterdir()):
        artifacts.append(put('artifacts', path.stem, **{'Nome': 'Task17f conferma — ' + path.name,
            'Tipo': 'Report', 'Percorso': path.relative_to(ROOT.parent).as_posix(),
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(), 'Dimensione byte': path.stat().st_size,
            'Versione': 'v1'}))
    artifacts.append(put('artifacts', 'full-archive', **{'Nome': 'Task17f — archivio completo Kaggle',
        'Tipo': 'Report', 'Percorso': 'artifacts/giada_task17f_event_confirmation_1b87bdd_3a7e4bfc.zip',
        'SHA-256': audit['archive_sha256'], 'Dimensione byte': audit['archive_size_bytes'],
        'Versione': 'v1', 'Descrizione': '171 tracce dense; ZIP recuperato via MCP, non incluso nel commit Git.'}))
    mirror.local_upsert('runs', {'Codice stabile': 'runs-giada-task17f-kaggle-v1-orchestration',
        'Stato': 'Completata', 'Artefatti prodotti': artifacts,
        'Validità tecnica': 'Exit0; codice/freeze/CRC verificati; 144 episodi, replica e108 confronti; 171 tracce dense finite e gate in[0,1].',
        'Descrizione': 'Kaggle v1, codice1b87bdd. GATE_C_PASS_BOUNDED_CAHVA; Task18 autorizzata nel dominio registrato.'})
    runs = {}
    for arm, key in (('native', 'native'), ('native_super', 'native-super'), ('formula', 'formula'), ('m_pair_2049_f64', 'lut')):
        runs[arm] = put('runs', arm, **{'Nome': 'Task17f conferma — ' + arm,
            'Stato': 'Completata', 'Braccio': [ref('arms', key)], 'Blocco': [ref('blocks', 'confirmation')],
            'Seed': '171601,171619,171643', 'Artefatti prodotti': artifacts,
            'Hardware e ambiente': 'Kaggle CPU, NEURON8.2.7; codice1b87bdd; teacher canonico642segmenti.',
            'Configurazione effettiva': '4 schedule congelate x3 seed x3 gbar; 36 episodi; 60ms; campioni0.025ms.',
            'Validità tecnica': 'Tutti i criteri preregistrati passano; nessuna riscelta sul test.'})
    observations = []
    for i, row in enumerate(pairs):
        suffix = f"{row['arm']}-p{row['protocol_index']}-s{row['seed']}-g{row['gbar_multiplier']}"
        values = list(row['metrics'].values())
        scalars = {'voltage': max([v['voltage_rmse_mv'] for v in values] + list(row['event_probe_voltage_rmse_mv'].values())),
                   'gate': max(max(v['m_max_error'], v['h_max_error']) for v in values),
                   'current': max(v['current_rmse_ma_cm2'] for v in values)}
        for metric, value in scalars.items():
            observations.append(put('observations', suffix + '-' + metric, **{
                'Nome': 'Task17f — ' + suffix + ' — ' + metric, 'Valore': value, 'Numerosità': 1,
                'Run': [runs[row['arm']]], 'Specifica di valutazione': [ref('evaluations', metric)],
                'Artefatti dettagliati': artifacts,
                'Strato o sottogruppo': f"protocol={row['protocol_index']};seed={row['seed']};gbar={row['gbar_multiplier']};arm={row['arm']}",
                'Descrizione': 'Massimo sui siti registrati; valori per sito, eventi e rilascio in paired_metrics.json, indice' + str(i)}))
        if (i + 1) % 12 == 0: print(f'[SQLite Task17f] contrasti {i+1}/108 registrati', flush=True)
    for kind, counts in report['support']['uncensored_events_at_gbar1_by_seed'].items():
        for seed, count in counts.items():
            observations.append(put('observations', f'support-{kind}-{seed}', **{
                'Nome': f'Task17f supporto — {kind} seed{seed}', 'Valore': count, 'Numerosità': 1,
                'Run': [runs['native']], 'Specifica di valutazione': [ref('evaluations', 'support')],
                'Artefatti dettagliati': artifacts, 'Strato o sottogruppo': f'{kind};seed={seed};gbar1;schedule designata',
                'Descrizione': 'Eventi non censurati nella schedule congelata che designa questa classe.'}))
    for row in effects:
        for site, value in row['candidate'].items():
            if not value['effect_identifiable']: continue
            suffix = f"gbar-p{row['protocol_index']}-s{row['seed']}-site{site}"
            observations.append(put('observations', suffix, **{
                'Nome': 'Task17f — ' + suffix, 'Valore': value['relative_error'], 'Numerosità': 1,
                'Run': [runs['m_pair_2049_f64']], 'Specifica di valutazione': [ref('evaluations', 'gbar')],
                'Artefatti dettagliati': artifacts, 'Strato o sottogruppo': suffix,
                'Descrizione': 'Errore relativo del contrasto gbar1.5−0.5; effetto nativo>=0.05mV; limite0.2.'}))
    experiment = ref('experiments', 'matrix')
    finding = put('findings', 'gate-c-pass', **{'Nome': 'Task17f — sostituzione Ca_HVA confermata con eventi attivi',
        'Esito': 'Positivo', 'Esperimenti': [experiment], 'Osservazioni': observations,
        'Risultato': '144 episodi + replica; 108 contrasti passati; tre classi non censurate su3/3 seed nuovi; peggior RMSE V LUT0.0138580152mV; conteggi/onset/censura e rilascio passano. GateC positivo.',
        'Incertezza': 'Tre seed, quattro schedule, tre gbar; nativo a due precisioni. Non una garanzia fuori da questo supporto.',
        'Limitazioni': 'LUTm2049f64, h analitico; nessun vantaggio hardware dimostrato. Altri canali da testare inTask18.'})
    evidence = []
    for suffix in ('support', 'solver', 'candidate'):
        evidence.append(put('evidence', suffix, **{'Nome': 'Task17f conferma indipendente — ' + suffix,
            'Esito': 'Sostiene', 'Affermazione valutata': [ref('claims', suffix)],
            'Risultati a sostegno': [finding],
            'Argomentazione': 'Supporto completo e controlli appaiati validi nelle36 condizioni per braccio; seed separati dal pilot.',
            'Limiti e spiegazioni alternative': 'Dominio limitato ai protocolli/siti/gbar e teacher canonico registrati; non universalità.',
            'Data valutazione': '2026-10-04'}))
        mirror.local_upsert('claims', {'Codice stabile': f'claims-giada-task17f-event-support-{suffix}-v1',
            'Stato': 'Supportata nel dominio', 'Limiti': 'Conferma17f positiva: Ca_HVA canonico, 4 schedule, 3 seed, 3 gbar. Nessuna universalità o accelerazione hardware inferita.'})
    origin = mirror.query("SELECT record_id FROM v_records WHERE stable_code='findings-giada-task17f-development-smoke-controls-v1'")['rows']
    assert len(origin) == 1
    put('confirmations', 'gate-c', **{'Nome': 'Task17f — conferma indipendente GateC',
        'Tipo': 'Conferma indipendente', 'Esperimenti di verifica': [experiment], 'Risultati da confermare': [origin[0]['record_id']],
        'Indipendenza': '171601/171619/171643 disgiunti dal pilot; freeze precedente alla consultazione degli outcome.',
        'Condizioni nuove': 'Eventi attivi prequalificati; gbar0.5/1/1.5; sostituzione causale Ca_HVA.',
        'Esito': report['decision']})
    put('decisions', 'task18-authorized', **{'Nome': 'GateC superato — proseguire allaTask18',
        'Esito': 'Continuare', 'Risultati': [finding], 'Valutazioni delle evidenze': evidence,
        'Motivazione': 'Tutti i dieci gate preregistrati passano. Il blocco17e sul supporto evento è superato.',
        'Condizioni di revisione': 'Nuovi meccanismi e nuovi domini richiedono verifiche dedicate; nessuna universalità implicita.',
        'Data': '2026-10-04'})
    mirror.local_upsert('experiments', {'Codice stabile': 'experiments-giada-task17f-event-support-matrix-v1',
        'Stato': 'Concluso'})
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    return {'valid': True, 'observation_count': len(observations), 'task18_authorized': True, 'airtable_accessed': False}


if __name__ == '__main__': print(json.dumps(main(), indent=2))
