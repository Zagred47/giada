#!/usr/bin/env python3
"""Execute GIADA Task 1 on a CUDA host (CPU is accepted for smoke runs)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.giada_teacher.atomic_gate_playground import (
    AtomicGateTaskConfig,
    evaluate_frozen_atomic_gate,
    materialize_atomic_gate_dataset,
    train_and_select_atomic_gate,
)
from src.giada_teacher.double_oracle import ExtractedGateFormula


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mod_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = AtomicGateTaskConfig(
        seeds=(17,) if args.smoke else (17, 29, 43),
        learning_rates=(0.003,) if args.smoke else (0.01, 0.003, 0.001),
        checkpoints=(0, 2, 5) if args.smoke else (0, 100, 300, 1000, 3000, 10000),
        batch_size=64 if args.smoke else 1024,
    )
    formula = ExtractedGateFormula.from_mod(args.mod_file)
    dataset = materialize_atomic_gate_dataset(formula)
    train_and_select_atomic_gate(dataset, args.output_dir, config)
    report = evaluate_frozen_atomic_gate(dataset, args.output_dir, config, formula=formula)
    print(json.dumps({"valid": report["valid"], "rmse": report["aggregate_sealed_rmse"]}, indent=2))


if __name__ == "__main__":
    main()
