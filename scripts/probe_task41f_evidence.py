"""Summarize frozen diagnostic evidence without changing any decision."""
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'artifacts/task41f_reserve_external'

def compact(value):
    if isinstance(value,dict):
        return {k:compact(v) for k,v in value.items() if k not in ('original_case','v','ca','g','currents','gate','traces')}
    if isinstance(value,list):
        if len(value)>100:return {'array_length':len(value)}
        return [compact(v) for v in value]
    return value

def main():
    source=EXT/'recovery_final.zip'
    if not source.exists():source=EXT/'recovery_1792688245799.zip'
    with zipfile.ZipFile(source) as z:
        out={}
        for name in ('precision','reproduction','coefficient_oracles','batch_sensitivity','clamps'):
            data=json.loads(z.read('frozen_audit/'+name+'.json'))
            out[name]=compact(data)
            if isinstance(data,list):
                print(name,'row keys',list(data[0]))
                if name=='precision':
                    for r in data:
                        print(r['model'],r['policy'],{k:sum(v['metrics']['passed'] for v in rows) for k,rows in r['reference_comparisons'].items()})
                elif name=='coefficient_oracles':
                    for r in data:
                        print(r['model'],r['oracle'],'sample keys',list(r['rows'][0]))
                        print(json.dumps(compact(r['rows'][0]))[:700])
                elif name in ('clamps','batch_sensitivity'):
                    for r in data:
                        print({k:v for k,v in r.items() if k!='responses'},'passes',sum(v['metrics']['passed'] for v in r['responses']), '/',len(r['responses']))
            else:print(name,json.dumps(data))
    (EXT/'frozen_probe_summary.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
