"""Register local pilot/integration evidence, never independent confirmation."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from .mirror import Mirror, ROOT, write_json


def main():
    mirror = Mirror()
    root = ROOT.parent / "experiments/results/task17f_development_validation_20261004"
    report = json.loads((root / "final_report.json").read_text())
    pairs = json.loads((root / "paired_metrics.json").read_text())
    pilot = json.loads((root / "pilot_progress.json").read_text())
    assert report["development_smoke"] and not report["confirmation_accessed"]
    assert not report["gate_c_authorized"] and not report["task18_authorized"]

    def ref(table, suffix):
        code = f"{table}-giada-task17f-event-support-{suffix}-v1"
        rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code = '"
                            + code.replace("'", "''") + "'")["rows"]
        if len(rows) != 1:
            raise RuntimeError(f"Missing prerequisite: {table}/{suffix}")
        return rows[0]["record_id"]

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {"Codice stabile":
            f"{table}-giada-task17f-development-{suffix}-v1", **fields})["record_id"]

    artifacts = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        artifacts.append(put("artifacts", path.stem, **{
            "Nome": "Task17f sviluppo — " + path.name, "Tipo": "Report",
            "Percorso": path.relative_to(ROOT.parent).as_posix(),
            "SHA-256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "Dimensione byte": path.stat().st_size, "Versione": "v1"}))
    environment = "WSL Ubuntu CPU, NEURON 8.2.7; working tree su " + report["code_revision"]
    common = {"Stato": "Completata", "Blocco": [ref("blocks", "pilot")],
              "Artefatti prodotti": artifacts, "Hardware e ambiente": environment,
              "Validità tecnica": "Controllo locale di sviluppo; nessun seed confermativo aperto."}
    pilot_run = put("runs", "native-pilot", **common, **{
        "Nome": "Task17f — pilot nativo locale (27 prove)",
        "Braccio": [ref("arms", "native")], "Seed": "171501,171519,171543",
        "Configurazione effettiva": "Ladder preregistrata, pesi canonici; 3/3 positivi per classe prima del freeze."})
    groups = defaultdict(list)
    for row in pilot:
        groups[(row["kind"], row["spec"])].append(row)
    pilot_observations = []
    for (kind, spec), rows in groups.items():
        pilot_observations.append(put("observations", "pilot-" + spec, **{
            "Nome": f"Task17f pilot — {spec}: seed positivi {kind}",
            "Valore": sum(bool(row["hit"]) for row in rows), "Numerosità": len(rows),
            "Strato o sottogruppo": f"{kind}; gbar1; tre seed pilot; non test indipendente",
            "Run": [pilot_run], "Artefatti dettagliati": artifacts,
            "Specifica di valutazione": [ref("evaluations", "support")]}))
    experiment = ref("experiments", "matrix")
    pilot_finding = put("findings", "pilot-support", **{
        "Nome": "Task17f sviluppo: tre classi prequalificate",
        "Esito": "Positivo", "Esperimenti": [experiment], "Osservazioni": pilot_observations,
        "Risultato": "27 prove: soma 4nA, calcio 12 sinapsi/3 burst/4 eventi, NMDA 10 sinapsi/2 burst/2 eventi. Ogni schedule scelta produce la classe richiesta su 3/3 seed pilot.",
        "Incertezza": "Stimoli scelti sul pilot; generalizzazione ai seed confermativi ancora ignota.",
        "Limitazioni": "Non è Gate C; tutti gli insuccessi intermedi restano in pilot_progress.json e nelle osservazioni."})
    observations = []
    for arm in ("native", "native_super", "formula", "candidate"):
        key = {"native_super": "native-super", "candidate": "lut"}.get(arm, arm)
        run = put("runs", "smoke-" + arm, **common, **{
            "Nome": "Task17f smoke locale — " + arm, "Braccio": [ref("arms", key)],
            "Seed": "171501", "Configurazione effettiva": "Quattro schedule congelate; gbar=1; 60ms; campionamento0.025ms; stesso seed pilot, non confermativo.",
            "Descrizione": "Quattro episodi per braccio; un replay nativo aggiuntivo per ripetibilità."})
        recorded_arm = "m_pair_2049_f64" if arm == "candidate" else arm
        for row in (r for r in pairs if r["arm"] == recorded_arm):
            metrics = list(row["metrics"].values())
            values = {
                "voltage": max([v["voltage_rmse_mv"] for v in metrics]
                               + list(row["event_probe_voltage_rmse_mv"].values())),
                "gate": max(max(v["m_max_error"], v["h_max_error"]) for v in metrics),
                "current": max(v["current_rmse_ma_cm2"] for v in metrics),
            }
            for metric, value in values.items():
                observations.append(put("observations", f"{arm}-{row['protocol_index']}-{metric}", **{
                    "Nome": f"Task17f smoke — {arm}, protocollo {row['protocol_index']}, {metric}",
                    "Valore": value, "Numerosità": 1, "Run": [run],
                    "Specifica di valutazione": [ref("evaluations", metric)],
                    "Artefatti dettagliati": artifacts,
                    "Strato o sottogruppo": "seed171501/gbar1; massimo sui siti registrati; sviluppo",
                    "Descrizione": "Conteggi/onset/censura/rilasci e tutti i gate booleani sono conservati nel paired_metrics.json, senza riduzione a una media."}))
    smoke_finding = put("findings", "smoke-controls", **{
        "Nome": "Task17f: integrazione nativa attiva superata in sviluppo",
        "Esito": "Positivo", "Esperimenti": [experiment], "Osservazioni": observations,
        "Risultato": "16 episodi + replay; tutti i 12 contrasti super-ultra/formula/LUT vs nativo passano V, gate, corrente, eventi e rilasci. Peggior RMSE V LUT circa0.000874028mV.",
        "Incertezza": "Un solo seed pilot e gbar1; nessuna stima indipendente della generalizzazione.",
        "Limitazioni": "Non testa contrasti gbar della conferma; Gate C e Task18 restano non autorizzati. Nessuna misura di accelerazione GPU."})
    probe_misses = [r for r in pilot if r["kind"] == "nmda_spike"
                    and any(e["kind"] == "nmda_spike" for e in r["events"])
                    and not any(e["kind"] == "nmda_spike" for e in r["legacy_probe_events"])]
    assert len(probe_misses) == 1 and probe_misses[0]["seed"] == 171519
    probe_observation = put("observations", "legacy-probe-miss", **{
        "Nome": "Task17f — spike NMDA locale non rilevato dal probe legacy",
        "Valore": len(probe_misses), "Numerosità": 6, "Run": [pilot_run],
        "Specifica di valutazione": [ref("evaluations", "events")],
        "Artefatti dettagliati": artifacts,
        "Strato o sottogruppo": "Sei prove pilot NMDA, confronto detector sulla stessa traiettoria",
        "Descrizione": "nmda_spike_r1_soma0 seed171519: local=1, legacy=0. Conteggio dei trial discordanti, non errore del candidato."})
    probe_finding = put("findings", "probe-localization", **{
        "Nome": "Task17f: probe legacy perde un evento locale nel pilot",
        "Esito": "Positivo", "Esperimenti": [experiment], "Osservazioni": [probe_observation],
        "Risultato": "Un caso su sei prove NMDA: local=1, legacy=0 sulla stessa traiettoria; conferma che la posizione del detector può nascondere eventi locali.",
        "Limitazioni": "Non attribuisce retrospettivamente tutta l'assenza di eventi in17e al probe. Nessuna differenza di modello in questo contrasto."})
    for suffix, finding, argument in (
        ("probe", probe_finding, "Il contrasto appaiato sul pilot identifica un evento locale assente al detector legacy."),
        ("support", pilot_finding, "La parte pilot dell'ipotesi è sostenuta; parte confermativa non valutata."),
        ("solver", smoke_finding, "Due precisioni native concordano sul solo smoke attivo di sviluppo."),
        ("candidate", smoke_finding, "Candidato congelato passa sullo smoke, non ancora sui seed/gbar indipendenti."),
    ):
        put("evidence", suffix, **{"Nome": "Task17f sviluppo — " + suffix,
            "Esito": "Sostiene", "Affermazione valutata": [ref("claims", suffix)],
            "Risultati a sostegno": [finding], "Argomentazione": argument,
            "Limiti e spiegazioni alternative": "Sostegno parziale su sviluppo. Ipotesi lasciata Aperta; non sostituisce la conferma preregistrata.",
            "Data valutazione": "2026-10-04"})
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "pilot_observations": len(pilot_observations),
            "smoke_observations": len(observations), "confirmation_accessed": False,
            "airtable_accessed": False}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
