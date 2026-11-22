import json,subprocess
from pathlib import Path
root=Path('/content/giada_task52f_resume600')
print(json.dumps(dict(root_exists=root.exists(),status=json.loads((root/'job_status.json').read_text()) if (root/'job_status.json').exists() else None)))
print(subprocess.check_output(['ps','-eo','pid,args'],text=True))
