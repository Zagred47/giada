from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.giada_teacher import build_causal_classification, write_causal_classification


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify GIADA teacher mechanisms causally")
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    result = build_causal_classification(inventory)
    write_causal_classification(result, args.json, args.markdown)
    print(json.dumps({"summary": result["summary"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
