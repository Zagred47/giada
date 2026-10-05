"""Preregister the post-Task28 Gate D compute test in the SQLite mirror only."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    path=repo/'experiments/task28b_gate_d_compute.json'
    cfg=json.loads(path.read_text(encoding='utf-8'))
    from src.giada_teacher.gate_d_compute_audit import verify_parent
    verify_parent(repo,cfg)
    mirror=ValidatedBatchMirror()
    def put(table,key,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-gated-compute-{key}-v1',**fields})['record_id']
    claim=put('claims','materiality',**{
        'Nome':'Gate D material same-GPU compute reduction','Tipo':'Ipotesi','Stato':'Aperta',
        'Enunciato':'The frozen five-learned/six-formula local ionic block reduces resident-GPU complete-step latency by at least 10% versus eleven canonical formulas, on both model families at batch 642.',
        'Limiti':cfg['limits'],'Condizioni di falsificazione':cfg['promotion']})
    exp=put('experiments','paired',**{
        'Nome':'GIADA Gate D paired same-device compute audit','Tipo':'Confermativo','Stato':'Preregistrato',
        'Ipotesi':[claim],'Descrizione':cfg['timing'],'Obiettivo informativo':cfg['promotion']})
    protocol=put('protocols','cuda',**{
        'Nome':'Gate D resident CUDA complete ionic step','Esperimento':[exp],'Versione':'v1',
        'Modalità':'Ricomposizione','Procedura':json.dumps(cfg,ensure_ascii=False),
        'Criteri di successo':cfg['promotion'],'Hash preregistrazione':hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto':'Parent/preflight mismatch is technical error; measured <10% on either family is scientific NO-GO, without threshold change.'})
    metric=put('metrics','paired-reduction',**{
        'Nome':'Median paired CUDA latency reduction','Famiglia':'Regressione',
        'Direzione':'Massimizzare','Unità':'frazione',
        'Formula':'median across alternating paired CUDA events of (T_exact-T_hybrid)/T_exact; batch 642 primary.'})
    evaluation=put('evaluations','paired-reduction',**{
        'Nome':'Gate D material compute threshold','Metrica':[metric],'Protocollo':[protocol],
        'Versione':'v1','Ruolo':'Primaria','Target':'0.10',
        'Aggregazione e pesi':'Conjunction across independent and shared_heads families at batch 642; no cross-family averaging.'})
    arms=[]
    for family in cfg['families']:
        for kind in ('canonical','hybrid'):
            arms.append(put('arms',family+'-'+kind,**{
                'Nome':'Gate D '+family+' '+kind,'Ruolo':'Baseline' if kind=='canonical' else 'Trattamento',
                'Protocollo':[protocol],
                'Descrizione':'All 18 gate endpoints and 11 analytic signed currents on same resident CUDA input.'}))
    put('predictions','materiality',**{
        'Nome':'Gate D 10% speed prediction','Ipotesi':[claim],
        'Origine':'Preregistrata','Risultato atteso':cfg['promotion'],
        'Specifica di valutazione':[evaluation],'Soglia o intervallo':'>=0.10 both families at batch 642'})
    put('contrasts','paired',**{
        'Nome':'Gate D exact vs frozen hybrid','Bracci':arms,
        'Specifiche di valutazione':[evaluation],
        'Contrasto e coefficienti':'Within-family paired exact minus hybrid latency, same GPU/batch/input/dtype/output and 3 model seeds.',
        'Confondenti controllati':cfg['timing'],'Soglia interpretativa':cfg['promotion'],
        'Correzione confronti multipli':'Both families must pass; larger batch sizes diagnostic only.'})
    for filename in ('experiments/task28b_gate_d_compute.json','src/giada_teacher/gate_d_compute_audit.py',
                     'scripts/run_gate_d_compute.py','notebooks/28b_gate_d_compute.ipynb'):
        source=repo/filename
        put('artifacts',source.stem,**{'Nome':'Gate D '+source.name,'Tipo':'Report',
            'Percorso':filename,'SHA-256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'Dimensione byte':source.stat().st_size,'Versione':'v1'})
    put('decisions','scope',**{
        'Nome':'Run Gate D compute test before Task29','Esito':'Continuare',
        'Motivazione':'Task28 composition passed but no same-device compute evidence exists.',
        'Condizioni di revisione':'Task29 remains unauthorized until both families pass 10% material gain with accuracy intact; no end-to-end speed claim.'})
    mirror.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'family_count':len(cfg['families']),
                      'primary_batch':cfg['gate_d_batch_size'],'airtable_accessed':False}))


if __name__=='__main__':main()
