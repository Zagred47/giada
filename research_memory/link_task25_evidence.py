"""Link Task25 diagnostic findings to their measured observation records."""
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror

FINDING_TEXT = {
    'heterogeneous_composition': 'La composizione indipendente (1860 parametri) supera le soglie in 3/3 seed, su cinque canali, domini richiesti, traiettorie imposte, orizzonti a tensione costante e pannelli di corrente. Conferma limitata a tensione imposta: non dimostra ancora un neurone autonomo o la sostituzione in circuito chiuso.',
    'compression_reuse': 'La sola compressione del calcio (1556 parametri complessivi) passa 3/3 seed. La compressione del sodio (1164) e quella di entrambi (860) passano 2/3: nel seed 43, sul dominio activation_boundary, m_RMSE vale 0.001118951 per NaTa_t e 0.001078819 per NaTs2_t, contro il limite 0.001. Tutte le soglie di corrente, traiettoria e orizzonte costante passano. Nessuna promozione della compressione del sodio o conclusione di accelerazione hardware.',
    'cancellation': 'Sono registrati pannelli con correnti di segno opposto e strati di cancellazione. La valutazione delle singole correnti e del pannello peggiore impedisce di promuovere un modello dalla sola somma. Il superamento delle soglie di corrente non implica quello dei gate; non è dimostrato che la cancellazione causi i due errori di m.',
    'identity': 'Lo scambio delle identità dei meccanismi peggiora il gate_RMSE di almeno 415.794667 volte nei dodici sistemi, oltre la soglia diagnostica 10. Il controllo negativo verifica la sensibilità all’identità; non è un candidato selezionabile.',
    'precision': 'Sono confrontati solver FP32 e FP64 usando gli stessi rate appresi FP32, con controlli formula FP32 e LUT. I limiti della compressione riguardano errori one-step di m nel dominio di attivazione, non un errore di esecuzione nativa. Non viene rivendicata una soluzione mediante precisione o una nuova selezione post hoc.',
    'path_transfer': 'Tutti e quattro i bundle superano le soglie su ogni traiettoria imposta veloce/lenta, seed e orizzonte a tensione costante. Restano i due limiti one-step di m nei bundle con sodio compresso. Questo non dimostra voltaggio autonomo né sufficienza degli endpoint a 1 ms.',
}


def main():
    m = ValidatedBatchMirror()
    table = m.contract.tables['observations']['name']
    patterns = {
        'identity': ['Task25 %-swapped_identity %', 'Task25 %-outcome identity_rmse_ratio'],
        'precision': ['Task25 %-held_fp64_diagnostic %', 'Task25 %-held_rollout %', 'Task25 numerical %', 'Task25 native_audit %'],
        'cancellation': ['Task25 %cancellation%'],
        'path_transfer': ['Task25 %-path_rollout %', 'Task25 %-held_rollout %'],
    }
    counts = {}
    for key, likes in patterns.items():
        query = 'SELECT _record_id AS id FROM "' + table + '" WHERE ' + ' OR '.join('Nome LIKE ' + repr(p) for p in likes)
        ids, offset = [], 0
        while True:
            rows = m.query(query + ' ORDER BY _record_id LIMIT 10000 OFFSET ' + str(offset), limit=10000)['rows']
            ids.extend(row['id'] for row in rows)
            if len(rows) < 10000:
                break
            offset += 10000
        assert ids, key
        m.local_upsert('findings', {'Codice stabile': f'findings-giada-roadmap-task25-{key}-v1', 'Osservazioni': ids})
        counts[key] = len(ids)
    for key, text in FINDING_TEXT.items():
        m.local_upsert('findings', {'Codice stabile': f'findings-giada-roadmap-task25-{key}-v1', 'Risultato': text})
        m.local_upsert('evidence', {'Codice stabile': f'evidence-giada-roadmap-task25-{key}-v1', 'Argomentazione': text})
    m.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', m.export_snapshot())
    assert m.verify()['valid']
    print(json.dumps({'valid': True, 'linked_observation_counts': counts, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
