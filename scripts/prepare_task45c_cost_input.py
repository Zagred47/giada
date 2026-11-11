"""Compact immutable checkpoints only; no native traces or database export."""
import hashlib,json,zipfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    folder=ROOT/'artifacts/task45c_cost_input';folder.mkdir(exist_ok=True)
    source=ROOT/'artifacts/task45c_kaggle_external/task45c_final.zip'
    receipt=json.loads((source.parent/'external_backup_receipt.json').read_text());assert receipt['valid']
    with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
    dest=folder/'checkpoint45c.zip.bin';members={};summary={};finite=True
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as target:
        for arm in ('C0','N00','N01','N10','N11','I0','I1','L0','L1','J','R41','R111'):
            name=arm+'.pt';data=z.read(name);target.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
        target.writestr('nomination_freeze.json',z.read('nomination_freeze.json'))
        for panel in ('original45','safe_hyper','regression44'):
            rows=json.loads(z.read(panel+'/results.json'));finite &= all(r['finite'] for r in rows);summary[panel]={}
            for h in sorted({r['horizon_ms'] for r in rows}):
                summary[panel][str(h)]={arm:{mode:dict(eligible=sum(r['reference_valid'] for r in pool),passed=sum(r['reference_valid'] and r['current_passed'] and r['compartment_passed'] for r in pool),regressions=sum(r['reference_valid'] and r['regression_vs_f011'] for r in pool)) for mode in ('imposed','autonomous') for pool in [[r for r in rows if r['horizon_ms']==h and r['arm']==arm and r['mode']==mode]]} for arm in ('C0','N00','N01','N10','N11','I0','I1','L0','L1','J','R41','R111')}
        target.writestr('accuracy_summary.json',json.dumps(dict(panels=summary,all_traces_finite=finite,source_archive_sha256=receipt['remote']['sha256'])))
    with dest.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    write_json(ROOT/'experiments/task45c_cost_transport_v2.json',dict(valid=True,dataset='alessandrobelli/giada-task45c-cost-checkpoints',filename=dest.name,sha256=digest,bytes=dest.stat().st_size,members=members,source_archive_sha256=receipt['remote']['sha256'],native_traces_included=False,DB_included=False,training_performed=False,selection_recomputed=False))
    print('Compact checkpoints prepared',dest.stat().st_size,digest)
if __name__=='__main__':main()
