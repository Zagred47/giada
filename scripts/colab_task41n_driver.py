"""Maintain active console execution throughout setup/native/GPU phases."""
import hashlib,json,subprocess,sys
from pathlib import Path
request=json.loads(Path('/content/giada_task41n_launch.json').read_text());manager=Path('/content/task41n_colab_manager.py')
assert hashlib.sha256(manager.read_bytes()).hexdigest()==request['manager_sha256']
raise SystemExit(subprocess.call([sys.executable,'-u',str(manager)]))
