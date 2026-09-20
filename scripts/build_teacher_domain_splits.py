from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.giada_teacher import build_atomic_domain_splits, render_atomic_domain_splits_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GIADA Task 0.5 atomic domain split contract")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_atomic_domain_splits()
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_atomic_domain_splits_markdown(report), encoding="utf-8")
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
