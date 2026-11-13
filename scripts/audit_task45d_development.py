"""Read-only case-identity audit; never reinterpret archived selection."""
import collections, hashlib, json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def regressions(rows, baseline):
    index={(r['seed'],r['case'],r['horizon_ms']):r for r in baseline}
    return [dict(candidate=r,baseline=index[(r['seed'],r['case'],r['horizon_ms'])]) for r in rows
            if index[(r['seed'],r['case'],r['horizon_ms'])]['passed'] and not r['passed']]

def main():
    receipt=json.loads((ROOT/'artifacts/task45d_retry_external/external_backup_receipt.json').read_text())
    with zipfile.ZipFile(receipt['local']) as z:
        base=json.loads(z.read('development_C1.json'));freeze=json.loads(z.read('nomination_freeze.json'));arms={}
        for arm,choice in freeze['per_arm_choices'].items():
            obj=json.loads(z.read(f'development_{arm}_{choice["checkpoint"]}.json'));rows=obj if isinstance(obj,list) else obj['rows'];lost=regressions(rows,base)
            arms[arm]=dict(choice=choice,total=len(rows),passed=sum(r['passed'] for r in rows),lost=lost,
                lost_families=dict(collections.Counter(r['candidate']['family'] for r in lost)),
                failed_target_counts=dict(collections.Counter(g for r in lost for g,q in r['candidate']['targets'].items() if not q['passed'])))
    external=json.loads((ROOT/'artifacts/task45d_evaluation_external/external_backup_receipt.json').read_text());crossings=[];members={}
    for rec in (receipt,external):
        with zipfile.ZipFile(rec['local']) as z:
            for name in z.namelist():
                if '/crossings_' in name and name.endswith('.json'):members[name]=json.loads(z.read(name))
    for name,obj in members.items():
        rows=obj.get('crossings',[]) if isinstance(obj,dict) else obj
        crossings.extend(dict(source=name,**r) for r in rows if r['arm'] in ('C1','Ncap','Nline','T0','T1') and r['channel']=='Ih')
    from research_memory.payload_store import open_readonly
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:
        patterns=[dict(r) for r in con.execute("SELECT stable_code, name FROM v_records WHERE stable_code LIKE '%45c%' OR stable_code LIKE '%45d%' LIMIT 80")]
    report=dict(valid=True,baseline_rows=len(base),baseline_passed=sum(r['passed'] for r in base),arms=arms,Ih_crossings=crossings,
        source_sha256=receipt['remote']['sha256'],evaluation_sha256=external['remote']['sha256'],memory_patterns=patterns,
        provenance='45d starts N/T repairs from f011 whereas C1 is45cJ. Static source confirms materialize(baseline), not materialize(previous).18/19 N regressions targetCa_LVAst.h; this is not evidence that Nap training damaged that head. Controlled C1-anchored composition remains to be tested.',
        limits='DEV and coupled outputs already opened: diagnostic/development evidence only. Do not relabel as independent confirmation or rewrite original selection.')
    write_json(ROOT/'experiments/results/task45e_protocol_v1/development_audit.json',report)
    print(json.dumps(dict(valid=True,baseline_passed=report['baseline_passed'],lost={k:len(v['lost']) for k,v in arms.items()},Ih_crossings=len(crossings))))
if __name__=='__main__':main()
