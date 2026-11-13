"""Create compact immutable45e evaluation input; preserve full original archive."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher.task45e_anchored_repair import ARMS
def existing_output():
    receipt=json.loads((ROOT/'artifacts/task45e_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid']
    path=ROOT/'experiments/task45e_evaluation_recovery_v1.json';old=json.loads(path.read_text())
    if old.get('transport')=='existing_private_kernel_output':return
    write_json(ROOT/'experiments/results/task45e_evaluation_v1/unused_compact_transport.json',dict(plan=old,upload_performed=False,reason='New private Dataset upload rejected; not attempted. Reuse already persisted Kaggle output instead.'))
    with zipfile.ZipFile(receipt['local']) as z:members={name:hashlib.sha256(z.read(name)).hexdigest() for name in old['members']}
    old.pop('dataset',None)
    old.update(transport='existing_private_kernel_output',kernel_source='alessandrobelli/giada-task45e-anchored-repair',source_kernel_version=1,filename='task45e_final.zip',sha256=receipt['remote']['sha256'],bytes=receipt['remote']['bytes'],members=members,new_upload=False,native_traces_included='Already present in original Kaggle output; no local upload or regeneration',DB_included=False)
    write_json(path,old);print('Use existing Kaggle45e output, no payload upload',flush=True)

def main():
    receipt=json.loads((ROOT/'artifacts/task45e_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid']
    source=Path(receipt['local'])
    with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
    folder=ROOT/'artifacts/task45e_evaluation_input';folder.mkdir(exist_ok=True);out=folder/'evaluation45e.zip.bin';assert not out.exists(),'Do not overwrite a published frozen input'
    members={};blocks=[]
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dest:
        def add(name,data):dest.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
        for name in [a+'.pt' for a in ARMS[1:]]+['nomination_freeze.json','calibration_freeze.json','execution.json','packed_execution_preflight.json','GPU_cost_preflight.json','preregistration.json']:
            add(name,z.read(name))
        execution=json.loads(z.read('execution.json'))
        for panel in ('original45','safe_hyper','regression44'):
            path=panel+'/results.json'
            if path not in z.namelist():continue
            rows=json.loads(z.read(path));kept=[]
            for geo,mode in sorted({(q['geometry'],q['mode']) for q in rows}):
                pool=[q for q in rows if q['geometry']==geo and q['mode']==mode];key=panel+'/'+geo+'/'+mode
                assert key in execution and panel+'/traces_'+geo+'_'+mode+'.npz' in z.namelist() and panel+'/crossings_'+geo+'_'+mode+'.json' in z.namelist()
                unique={(q['arm'],q['seed'],q['case'],q['horizon_ms']) for q in pool}
                assert len(unique)==len(pool)==len(ARMS)*3*len({q['case'] for q in pool})*len({q['horizon_ms'] for q in pool})
                assert all(q['finite'] for q in pool);blocks.append(key);kept.extend(pool)
            add(path,json.dumps(kept,separators=(',',':')).encode())
    with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert len(blocks)==10
    write_json(ROOT/'experiments/task45e_evaluation_recovery_v1.json',dict(valid=True,task='45e evaluation-only recovery',dataset='alessandrobelli/giada-task45e-evaluation-recovery',filename=out.name,sha256=digest,bytes=out.stat().st_size,source_archive_sha256=receipt['remote']['sha256'],source_archive_bytes=receipt['remote']['bytes'],members=members,completed_blocks=blocks,total_blocks=18,training_performed=False,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,native_traces_included=False,DB_included=False))
    print('Compact45e recovery input:',out.stat().st_size,'bytes;',len(blocks),'completed blocks;8missing',flush=True)
if __name__=='__main__':
    if '--existing-output' in sys.argv:existing_output()
    else:main()
