"""Register Task 17e prospectively in the local SQLite mirror only."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    path = ROOT.parent / "experiments/task17e_independent_causal_confirmation_preregistration.md"
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17e-independent-causal-{suffix}-v1",
            **fields,
        })["record_id"]

    claim = put("claims", "bounded-gate-c", **{
        "Nome": "Task17e: LUT m 2049 preserva il Ca_HVA accoppiato",
        "Enunciato": "Il candidato congelato a 2049 nodi rispetta gate, corrente, voltaggio, rilascio, eventi e risposta a gbar su seed indipendenti nel teacher canonico.",
        "Tipo": "Ipotesi", "Stato": "Aperta",
        "Condizioni di falsificazione": "Almeno un controllo di Gate C preregistrato fallisce su una condizione indipendente.",
        "Limiti": "Ambito Ca_HVA e morfologia canonica; nessuna inferenza di speedup o generalizzazione ad altri canali.",
    })
    experiment = put("experiments", "matrix", **{
        "Nome": "GIADA Task17e — conferma causale indipendente e Gate C",
        "Descrizione": "Matrice congelata di 135 episodi nativo/formula/LUT2049 su seed, stimoli e gbar appaiati.",
        "Obiettivo informativo": "Decidere se chiudere Task 17 e autorizzare Task 18 senza ritoccare il candidato dopo il test.",
        "Tipo": "Forensic", "Stato": "Preregistrato", "Ipotesi": [claim],
    })
    protocol = put("protocols", "v1", **{
        "Nome": "Task17e — matrice causale indipendente Ca_HVA",
        "Versione": "v1", "Modalità": "Sistema completo", "Esperimento": [experiment],
        "Procedura": "Tre seed nuovi, cinque programmi, tre livelli gbar, tre bracci; 60 ms a 0,025 ms; snapshot e Random123 appaiati; candidato m-LUT2049 f64 congelato per SHA-256.",
        "Criteri di successo": "Tutti i gate preregistrati su formula, V, m/h, corrente, eventi, rilascio, effetto gbar e supporto eventi; solo allora Gate C limitato a Ca_HVA passa.",
        "Regole di arresto": "Stop tecnico su hash tabella, compilazione, teacher, snapshot, native repeat o formula control invalidi; nessuna sostituzione post hoc di seed o soglie.",
        "Hash preregistrazione": hashlib.sha256(path.read_bytes()).hexdigest(),
        "Descrizione": "Pre-registrato prima della conferma indipendente; i risultati 17d-b sono soltanto sviluppo.",
    })
    put("blocks", "independent", **{
        "Nome": "Task17e — seed e gbar indipendenti appaiati",
        "Descrizione": "Seed 171001/171019/171043; quiete, NMDA e calcio originali e varianti; gbar 0,5/1/1,5.",
        "Seed": "171001,171019,171043", "Protocollo": [protocol],
        "Regola di appaiamento": "Stesso snapshot iniziale, schedule, Random123 e gbar nei tre bracci di ogni condizione.",
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "experiment": experiment,
            "protocol": protocol, "claim": claim}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
