"""Keep the prepared runtime connected until local DB registration is verified."""
import json,time
from pathlib import Path
ready=Path('/content/task41f_db_ready.json')
for attempt in range(180):
    if ready.exists():
        receipt=json.loads(ready.read_text());assert receipt['valid'] and receipt['registration_before_scientific_run']
        print('[GIADA Task41f] DBregistrationverified;runtime ready for scientific dispatch',flush=True)
        break
    print('[GIADA Task41f] awaiting local DBregistration',flush=True)
    time.sleep(20)
else:raise RuntimeError('Local DBregistration not acknowledged within one hour')
