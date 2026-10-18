"""Record actual console dispatch; explicitly not a training-success claim."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39e_tail_support_heads import arms,CONFIG
def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39e_protocol_v1'
    launch=json.loads((folder/'colab_launch.json').read_text());spec=json.loads((repo/CONFIG).read_text())
    assert launch['revision']==launch['job_status']['revision'] and launch['job_status']['state']=='running'
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39e-tail-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    path=folder/'colab_launch.json'
    artifact=put('artifacts','console-launch',Nome='Task39e verified console dispatch',Tipo='Log',Versione='v1',Percorso=path.relative_to(repo).as_posix(),
        **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    for arm in arms():
        for seed in spec['seeds']:
            put('runs',f'{arm["name"]}-seed{seed}',Nome=f'Task39e {arm["name"]} seed{seed}',Stato='In corso',Seed=str(seed),
                **{'Artefatti prodotti':[artifact],'Configurazione effettiva':json.dumps(dict(arm=arm,seed=seed,checkpoints=spec['checkpoints'],launch=launch)),
                'Hardware e ambiente':'Colab namedT4 session via official CLI0.7.4/WSL2, checked kernel errors and durable native worker status.',
                'Validità tecnica':'Pinned checkout/source verified; frozen-attribution GPU phase observed. Training preflights and scientific outcome pending at dispatch receipt.'})
    put('implementations','worker',Nome='Task39e console CUDA worker',Commit=launch['revision'])
    put('actions','execute',Nome='Execute Task39e on Colab console',Stato='In corso',Descrizione='Actual console dispatch verified; frozen oracle audit running. Not a training completion or scientific GO.')
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'launch_memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False,training_success_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
