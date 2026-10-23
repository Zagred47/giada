"""Keep original ZIP bytes opaque to Kaggle's automatic ZIP extraction."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'artifacts/task41h_kaggle_inputs';dest=ROOT/'artifacts/task41h_kaggle_binary_inputs';dest.mkdir(exist_ok=True)
for path in source.glob('*.zip'):shutil.copy2(path,dest/(path.name+'.bin'))
shutil.copy2(source/'dataset-metadata.json',dest/'dataset-metadata.json')
print(json.dumps(dict(binary_transport_ready=True,files=[p.name for p in dest.glob('*.bin')])),flush=True)
