"""Generate Task 7 NEURON reference traces without importing or using CUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.giada_teacher.cahva_closed_loop_microcanary import (  # noqa: E402
    ClosedLoopCaHVAConfig,
    _teacher_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from neuron import load_mechanisms

    config = ClosedLoopCaHVAConfig()
    config.validate()
    if not load_mechanisms(str(args.mechanism_root)):
        raise RuntimeError("Compiled Ca_HVA mechanism could not be loaded")
    traces = {}
    count = len(config.initial_voltage_mv) * len(config.gbar_multipliers) * len(config.protocol_names)
    index = 0
    for initial in config.initial_voltage_mv:
        for multiplier in config.gbar_multipliers:
            for protocol in config.protocol_names:
                index += 1
                print(f"[GIADA Task 7 teacher] {index}/{count}: {initial:g}, {multiplier:g}, {protocol}", flush=True)
                rows, area = _teacher_episode(args.mechanism_root, config, initial, multiplier, protocol)
                key = f"v{initial:g}-g{multiplier:g}-{protocol}"
                traces[f"{key}_rows"] = rows
                traces[f"{key}_area"] = np.asarray(area, dtype=np.float64)
    np.savez_compressed(args.output, **traces)
    print(f"[GIADA Task 7 teacher] reference complete: {count} episodes", flush=True)


if __name__ == "__main__":
    main()
