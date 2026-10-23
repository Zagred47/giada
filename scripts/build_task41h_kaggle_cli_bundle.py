"""Materialize a notebook artifact and explicit dependency metadata for CLI."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'scripts/kaggle_task41h_remaining_notebook.py').read_text()
dest=ROOT/'artifacts/task41h_kaggle_stage5_notebook';dest.mkdir(exist_ok=True)
notebook=dict(cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[source])],
    metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
(dest/'notebook_source.ipynb').write_text(json.dumps(notebook,indent=2))
metadata=json.loads((ROOT/'artifacts/task41h_kaggle_metadata_verified/kernel-metadata.json').read_text())
metadata.update(code_file='notebook_source.ipynb',dataset_sources=['alessandrobelli/giada-task41h-frozen-evaluation-inputs'],is_private=True,enable_gpu=True,enable_internet=True)
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(dict(bundle=str(dest),source_sha256=hashlib.sha256(source.encode()).hexdigest(),private=True,dependencies=metadata['dataset_sources'])),flush=True)
