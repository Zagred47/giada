from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.giada_teacher import build_atomic_data_contract, write_atomic_data_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GIADA atomic teacher data contracts")
    parser.add_argument("inventory", type=Path)
    parser.add_argument("classification", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    contract = build_atomic_data_contract(inventory, classification)
    write_atomic_data_contract(contract, args.json, args.markdown)
    print(json.dumps(contract["validation"], indent=2))


if __name__ == "__main__":
    main()
