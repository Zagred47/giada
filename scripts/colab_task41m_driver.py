import hashlib,json,subprocess,sys
from pathlib import Path
r=json.loads(Path('/content/giada_task41m_launch.json').read_text())
manager=Path('/content/task41m_colab_manager.py')
assert hashlib.sha256(manager.read_bytes()).hexdigest()==r['manager_sha256']
raise SystemExit(subprocess.call([sys.executable,'-u',str(manager)]))
