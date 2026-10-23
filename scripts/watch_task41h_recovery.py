"""Compact read-only progress, never queues work in the active kernel."""
import json,re,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'artifacts/task41h_evaluation_primary_external'
lines=(root/'console.log').read_text(errors='replace').splitlines();panels={}
for line in lines:
    m=re.search(r'evaluation (\w+) (\S+)/(imposed|autonomous) (\d+)/(\d+) responses=(\d+)',line)
    if m:
        label,name,mode,*_=m.groups();panels.setdefault(label+'/'+mode,set()).add(name)
last=next((x for x in reversed(lines) if '[GIADA Task' in x or 'Traceback' in x or '[GIADA recovery] final' in x),'')
status=json.loads((root/'job_status.json').read_text()) if (root/'job_status.json').exists() else {}
backup=json.loads((root/'external_backup_receipt.json').read_text()) if (root/'external_backup_receipt.json').exists() else {}
print(json.dumps(dict(state=status.get('state'),phase=status.get('phase'),completed_model_evaluations={k:len(v) for k,v in panels.items()},last_scientific_progress=last,
    console_age_seconds=round(time.time()-(root/'console.log').stat().st_mtime,1),backup_label=backup.get('remote',{}).get('label'),backup_bytes=backup.get('remote',{}).get('bytes'),
    backup_verified=backup.get('valid',False),final_report_complete=status.get('state')=='finished' and status.get('worker_exit_code')==0)),flush=True)
