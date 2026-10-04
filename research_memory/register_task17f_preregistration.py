"""Task17f hypotheses, contrasts and stop rules; SQLite only."""
from __future__ import annotations
import hashlib
import json
from .mirror import Mirror, ROOT, write_json


def main():
    mirror = Mirror()
    path = ROOT.parent / "experiments/task17f_event_supported_confirmation_preregistration.md"

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17f-event-support-{suffix}-v1", **fields})["record_id"]

    hypotheses = (
        ("probe", "Localizzazione del probe NMDA", "Il probe fisso nexus perde eventi presenti nel cluster effettivamente stimolato.",
         "Detector legacy e locale producono gli stessi conteggi su tutti i pilot.",
         "Contrasto esplorativo sulla stessa traiettoria; non misura da solo un errore del surrogato."),
        ("support", "Qualificazione degli stimoli", "Una ladder nativa a pesi canonici produce ogni classe richiesta su tutti i seed pilot e confermativi.",
         "Nessuna schedule qualificata o una classe assente su almeno un seed confermativo a gbar=1.",
         "Pilot e conferma disgiunti; cambiare schedule dopo conferma richiede un nuovo protocollo."),
        ("solver", "Affidabilità del riferimento attivo", "Il nativo ultra concorda col super-ultra entro i limiti formula anche durante eventi attivi.",
         "Un contrasto supera i limiti V/gate/corrente/eventi/rilasci.",
         "Convergenza a due precisioni, non dimostrazione della soluzione esatta."),
        ("candidate", "Fedeltà causale con eventi", "La LUT m2049 f64 congelata preserva dinamica, eventi e risposta gbar nelle schedule prequalificate.",
         "Un controllo candidato fallisce dopo supporto, solver e clone formula validati.",
         "Ca_HVA del teacher canonico e supporto registrato; nessun claim GPU o altro canale."),
    )
    claims = []
    for suffix, name, statement, falsification, limits in hypotheses:
        claims.append(put("claims", suffix, **{"Nome": "Task17f: " + name,
            "Enunciato": statement, "Tipo": "Ipotesi", "Stato": "Aperta",
            "Condizioni di falsificazione": falsification, "Limiti": limits}))
    experiment = put("experiments", "matrix", **{
        "Nome": "GIADA Task17f — supporto eventi e conferma causale",
        "Descrizione": "Pilot nativo, freeze delle schedule, matrice ultra/super-ultra/formula/LUT su seed nuovi.",
        "Obiettivo informativo": "Separare probe scorretto, stimoli insufficienti, errore del riferimento e fedeltà del candidato per decidere Gate C.",
        "Tipo": "Forensic", "Stato": "Preregistrato", "Ipotesi": claims})
    protocol = put("protocols", "protocol", **{
        "Nome": "Task17f — pilot qualificato e test indipendente",
        "Versione": "v1", "Modalità": "Sistema completo", "Esperimento": [experiment],
        "Procedura": "Pilot 171501/171519/171543; conferma 171601/171619/171643; tre classi su tutti i seed a gbar1; quattro programmi, tre gbar, quattro bracci. 60 ms a 0,025 ms. LUT invariata.",
        "Criteri di successo": "Tutti i criteri V, gate, corrente, eventi, rilascio, contrasto gbar, supporto e riferimento passano; solo allora Task18 autorizzata.",
        "Regole di arresto": "Supporto insufficiente è inconclusivo. Riferimento/clone invalidi bloccano attribuzione alla LUT. Non cambiare seed, soglie o schedule dopo conferma.",
        "Hash preregistrazione": hashlib.sha256(path.read_bytes()).hexdigest(),
        "Descrizione": "Successore 17e: distingue l'assenza delle tre classi richieste dal plateau NMDA presente e corregge il probe cluster."})
    arms = {}
    for key, role, description in (
        ("native", "Baseline", "Teacher canonico, CVode atol1e-7 rtol1e-8"),
        ("native-super", "Oracle diagnostico", "Stesso teacher, atol1e-8 rtol1e-9, contrasto di convergenza"),
        ("formula", "Baseline", "Clone NMODL formula Ca_HVA, solver ultra identico"),
        ("lut", "Trattamento", "LUT m2049 f64 congelata, h analitico, solver ultra"),
    ):
        arms[key] = put("arms", key, **{"Nome": "Task17f — " + key, "Ruolo": role,
            "Descrizione": description, "Protocollo": [protocol],
            "Configurazione residua": "Stessa schedule, Random123, gbar, stato iniziale, siti e detector."})
    evaluations = []
    for suffix, name, family, formula, unit, threshold in (
        ("voltage", "RMSE V", "Regressione", "sqrt(mean((V_candidate-V_native)^2))", "mV", "candidato<=2; controlli<=0.05"),
        ("gate", "Massimo errore m/h", "Regressione", "max_t,gate abs(x_candidate-x_native)", "frazione", "candidato<=0.01; controlli<=0.002"),
        ("current", "RMSE ica_HVA", "Regressione", "sqrt(mean((I_candidate-I_native)^2))", "mA/cm2", "<=1e-4"),
        ("events", "Conteggi, censura e onset", "Eventi temporali", "count equality AND censor equality AND maximum paired onset error", "ms", "<=0.2 soma/AIS; <=0.5 dendriti; conteggi/censura identici"),
        ("support", "Seed con positivo non censurato", "Eventi temporali", "sum_seed I(exists uncensored target event at gbar1 in designated protocol)", "seed", "3 per ciascuna delle tre classi; quiete silente"),
        ("release", "Identità del rilascio", "Fisica", "identity/draw/success equality AND max abs(quantity difference)", "quantità canonica", "identità discreta e differenza<=1e-10"),
        ("gbar", "Errore del contrasto gbar", "Fisica", "RMSE(deltaV_candidate-deltaV_native)/RMSE(deltaV_native)", "frazione", "<=0.2 per effetti>=0.05mV; almeno uno identificabile"),
    ):
        metric = put("metrics", suffix, **{"Nome": "Task17f — " + name,
            "Famiglia": family, "Formula": formula, "Unità": unit,
            "Direzione": "Target", "Parametri": threshold,
            "Casi degeneri": "Assenza di supporto/effetto identificabile blocca il gate; non vale come successo."})
        evaluations.append(put("evaluations", suffix, **{"Nome": "Task17f — " + name,
            "Metrica": [metric], "Protocollo": [protocol], "Versione": "v1", "Ruolo": "Primaria",
            "Target": threshold, "Popolazione e regioni": "Quattro schedule; seed e gbar preregistrati. V include probe reali; gate/corrente siti0/387/460/469.",
            "Orizzonte e finestre": "60ms, campionamento0.025ms", "Aggregazione e pesi": "Ogni condizione deve passare; nessuna compensazione con media globale.",
            "Matching eventi": "Per classe in ordine onset; stessi detector e siti tra bracci.",
            "Gestione classi assenti": "Copertura richiesta per ogni seed a gbar1; se manca esito inconclusivo."}))
    predictions = []
    for index, (suffix, name, statement, falsification, limits) in enumerate(hypotheses):
        prediction = put("predictions", suffix, **{"Nome": "Task17f — " + name,
            "Risultato atteso": statement, "Soglia o intervallo": falsification,
            "Descrizione": limits, "Origine": "Preregistrata", "Ipotesi": [claims[index]],
            "Specifica di valutazione": [evaluations[4 if suffix == "support" else 3 if suffix == "probe" else 0]]})
        predictions.append(prediction)
        involved = [arms["native"]] if suffix in ("probe", "support") else [arms["native"], arms["native-super"]] if suffix == "solver" else list(arms.values())
        put("contrasts", suffix, **{"Nome": "Task17f — " + name,
            "Bracci": involved, "Predizioni": [prediction], "Specifiche di valutazione": evaluations,
            "Contrasto e coefficienti": statement, "Soglia interpretativa": falsification,
            "Confondenti controllati": "Snapshot, input espliciti, pesi canonici, Random123, gbar e detector; cambia il solo fattore dichiarato.",
            "Correzione confronti multipli": "Criteri ingegneristici congiuntivi preregistrati, non p-value esplorativi."})
    put("protocols", "protocol", **{"Predizioni": predictions})
    for role, seeds in (("pilot", "171501,171519,171543"), ("confirmation", "171601,171619,171643")):
        put("blocks", role, **{"Nome": f"Task17f — {role}", "Seed": seeds,
            "Protocollo": [protocol], "Descrizione": "Ruoli disgiunti congelati prima di aprire la conferma.",
            "Regola di appaiamento": "Identici input espliciti, sinapsi, probe, snapshot, Random123 e gbar per ogni contrasto."})
    put("artifacts", "preregistration", **{"Nome": "Task17f — protocollo preregistrato",
        "Tipo": "Report", "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
        "SHA-256": hashlib.sha256(path.read_bytes()).hexdigest(), "Versione": "v1",
        "Dimensione byte": path.stat().st_size})
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "experiment": experiment, "claims": claims, "airtable_accessed": False}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
