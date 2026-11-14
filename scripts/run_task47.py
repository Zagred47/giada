"""Same audited phase isolation and exclusive-cache fix as successful46retry."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import run_task46 as worker
from src.giada_teacher import task47_independent_confirmation as q
if __name__=='__main__':
    with q.configured():
        try:worker.main()
        finally:
            out=Path(sys.argv[4])
            if (out/'final_report.json').exists():q.finalize(out)
