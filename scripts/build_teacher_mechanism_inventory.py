from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.giada_teacher import build_inventory, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GIADA NMODL teacher inventory")
    parser.add_argument("teacher_root", type=Path)
    parser.add_argument("--mod-dir", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    mod_dir = args.mod_dir or args.teacher_root / "L5PC_NEURON_simulation" / "mods"
    inventory = build_inventory(mod_dir, teacher_root=args.teacher_root)
    write_inventory(inventory, args.json, args.markdown)
    print(inventory["summary"])


if __name__ == "__main__":
    main()
