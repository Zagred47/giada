#!/usr/bin/env python3
"""Execute GIADA Task 2 for the Ca_HVA h gate on a CUDA host."""

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
        gate="h",
        seeds=(17,) if args.smoke else (17, 29, 43),
        learning_rates=(0.003,),
        checkpoints=(0, 2, 5) if args.smoke else (0, 1000, 3000, 10000, 30000, 50000),
        batch_size=64 if args.smoke else 1024,
    )
    formula = ExtractedGateFormula.from_mod(args.mod_file)
    dataset = materialize_atomic_gate_dataset(formula, gate="h")
    train_and_select_atomic_gate(dataset, args.output_dir, config)
    report = evaluate_frozen_atomic_gate(
        dataset, args.output_dir, config, formula=formula
    )
    print(json.dumps({
        "valid": report["valid"],
        "rmse": report["aggregate_sealed_rmse"],
        "rollout": report["constant_voltage_rollout_rmse"],
    }, indent=2))


if __name__ == "__main__":
    main()
