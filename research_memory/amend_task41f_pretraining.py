"""Record an equivalent CUDA-capture fix before scientific training starts."""
import hashlib,json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41f_protocol_v1'
    assert (folder/'memory_receipt.json').exists() and not (folder/'dispatch_request.json').exists()
    amendment=json.loads((folder/'pretraining_amendment.json').read_text())
    spec=json.loads((repo/'experiments/task41f_preregistration_v1.json').read_text())
    pf=json.loads((folder/'gpu_preflight.json').read_text())
    assert amendment['valid'] and amendment['scientific_contract_unchanged'] and not amendment['thresholds_changed']
    assert pf['valid'] and pf['frozen_failure_reproduction_passed'] and pf['protocol_sha256']==amendment['after_protocol_sha256']
    assert not pf['scientific_training_performed']
    m=ValidatedBatchMirror();ids={};artifacts=[]
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**values):
        code=f'{table}-giada-task41f-technical-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**values})['record_id'];ids[code]=rid;return rid
    changed=[name for name,sha in amendment['before_sources'].items() if amendment['after_sources'][name]!=sha]
    for name in changed+['experiments/task41f_preregistration_v1.json','experiments/results/task41f_protocol_v1/local_verification.json']:
        raw=(repo/name).read_bytes()
        m.local_upsert('artifacts',{'Codice stabile':f'artifacts-giada-task41f-{name}-v1','Versione':'v1-technical-fix1',
            'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
    for path in [folder/name for name in ('gpu_preflight.json','pretraining_amendment.json','protocol_before_technical_fix.json')]+list(folder.glob('gpu_preflight_failure_*.log')):
        raw=path.read_bytes();artifacts.append(put('artifacts',path.name,Nome='Task41f pretraining '+path.name,Tipo='Report',Versione='v1',
            Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)}))
    obs=put('observations','amendment',Nome='Task41f equivalent CUDAcapture repair before training',
        Descrizione=json.dumps(dict(amendment=amendment,GPU_preflight=pf,scientific_training_performed=False)),**{'Artefatti dettagliati':artifacts})
    put('findings','GPU-preflight',Nome='Task41f actualGPU preflight passed after capture compatibility fix',Esito='Positivo',
        Esperimenti=[ref('experiments-giada-task41f-factorial-v1')],Osservazioni=[obs],
        Risultato='Censorlossisequivalent,verifiedbyexactCPUtest. ScalarhostBooleanremovedfromCUDAcapture. GPUoptimizer/graphparity,4096wholeloop,samebatchfrozenfailurecases,precision,clamp,coefficientoracleandobjectiveprobepassbeforetraining.',
        Limitazioni='Technicalverificationonly;allscientificthresholdsandarmsunchanged;openedcasesusedonlyforfrozendiagnostics.')
    m.local_upsert('protocols',{'Codice stabile':'protocols-giada-task41f-factorial-v1','Versione':'v1-technical-fix1',
        'Procedura':json.dumps(spec,ensure_ascii=False),'Hash preregistrazione':pf['protocol_sha256']})
    m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41f-worker-v1','Commit':pf['revision']})
    print('Task41f equivalentpretrainingamendmentstaged',flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'pretraining_memory_receipt.json',dict(valid=True,records=ids,verification=v,
        protocol_sha256=pf['protocol_sha256'],registration_before_scientific_run=True,airtable_accessed=False,thresholds_changed=False))
    print('Task41f pretrainingamendmentandGPUproofverifiedinDB',flush=True)


if __name__=='__main__':main()
