"""Compact saved45d checkpoints and completed metrics; never export native traces."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
ARMS=('none','C0','C1','A00','A01','A10','A11','Ncap','Nline','T0','T1','J','R41','R111')
def main():
    receipt=json.loads((ROOT/'artifacts/task45d_retry_external/external_backup_receipt.json').read_text());source=Path(receipt['local']);assert receipt['valid']
    with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
    folder=ROOT/'artifacts/task45d_evaluation_input';folder.mkdir(exist_ok=True);out=folder/'evaluation45d.zip.bin';members={};blocks=[]
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dest:
        def add(name,data):dest.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
        for name in [a+'.pt' for a in ARMS[1:]]+['nomination_freeze.json','calibration_freeze.json','execution.json','packed_execution_preflight.json','GPU_training_transition_preflight.json','preregistration.json']:
            add(name,z.read(name))
        execution=json.loads(z.read('execution.json'))
        for panel in ('original45','safe_hyper','regression44'):
            name=panel+'/results.json'
            if name not in z.namelist():continue
            rows=json.loads(z.read(name));kept=[]
            for geo,mode in sorted({(r['geometry'],r['mode']) for r in rows}):
                pool=[r for r in rows if r['geometry']==geo and r['mode']==mode]
                key=panel+'/'+geo+'/'+mode;trace=panel+'/traces_'+geo+'_'+mode+'.npz';cross=panel+'/crossings_'+geo+'_'+mode+'.json'
                assert key in execution and trace in z.namelist() and cross in z.namelist(),key
                unique={(r['arm'],r['seed'],r['case'],r['horizon_ms']) for r in pool}
                assert len(unique)==len(pool)==len(ARMS)*3*len({r['case'] for r in pool})*len({r['horizon_ms'] for r in pool})
                assert all(r['finite'] for r in pool);blocks.append(key);kept.extend(pool)
            add(name,json.dumps(kept,separators=(',',':')).encode())
    with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    write_json(ROOT/'experiments/task45d_evaluation_recovery_v1.json',dict(valid=True,task='45d evaluation-only recovery',dataset='alessandrobelli/giada-task45d-evaluation-recovery',filename=out.name,sha256=digest,bytes=out.stat().st_size,source_archive_sha256=receipt['remote']['sha256'],source_archive_bytes=receipt['remote']['bytes'],members=members,completed_blocks=blocks,total_blocks=18,training_performed=False,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,native_traces_included=False,DB_included=False))
    print('Prepared evaluation-only input',out.stat().st_size,'bytes; completed blocks',len(blocks),flush=True)
if __name__=='__main__':main()
