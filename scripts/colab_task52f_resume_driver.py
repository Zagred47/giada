import hashlib,json,subprocess,sys
from pathlib import Path
request=json.loads(Path('/content/task52f_resume_request.json').read_text())
manager=Path('/content/task52f_colab_resume_manager.py')
assert hashlib.sha256(manager.read_bytes()).hexdigest()==request['manager_sha256']
raise SystemExit(subprocess.call([sys.executable,'-u',str(manager)]))
