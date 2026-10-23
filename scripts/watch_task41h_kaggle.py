"""Compact local view of the official read-only Kaggle SSE log stream."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--log',type=Path,default=ROOT/'artifacts/task41h_kaggle_live.log');args=parser.parse_args();path=args.log
messages=[]
for line in path.read_text(errors='replace').splitlines():
    if not line.startswith('data: '):continue
    try:row=json.loads(line[6:])
    except json.JSONDecodeError:continue
    if row.get('stream_name')=='stdout' and row.get('data','').startswith('[GIADA'):messages.append(row)
text=[r['data'] for r in messages]
print(json.dumps(dict(last_progress=messages[-1] if messages else None,
    newly_computed_cost_rows=sum('checkpoint cost ' in s for s in text),
    execution_rows=sum('checkpoint execution ' in s for s in text),
    numerical_panels=sum('checkpoint numerical ' in s for s in text),
    regenerated_opened_traces=sum('restored lost trace ' in s for s in text),
    regenerated_historical_traces=sum('restored historical trace ' in s for s in text),
    scientific_completed=any('[GIADA Task41h] completed ' in s for s in text),
    durable_output_published=any('durable output ' in s for s in text))),flush=True)
