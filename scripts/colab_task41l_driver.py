"""Keep a real active console execution while the independent manager runs."""
import hashlib,json,subprocess,sys
from pathlib import Path
request=Path('/content/giada_task41l_launch.json');r=json.loads(request.read_text())
manager=Path('/content/task41l_colab_manager.py')
assert hashlib.sha256(manager.read_bytes()).hexdigest()==r['manager_sha256']
raise SystemExit(subprocess.call([sys.executable,'-u',str(manager),str(request)]))
