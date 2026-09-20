from pathlib import Path
import unittest

import numpy as np

from src.giada_teacher.atomic_gate_diagnosis import (
    GateMDiagnosisConfig,
    _paired_batch_generator,
    prepare_gate_m_diagnosis,
)
from src.giada_teacher.double_oracle import ExtractedGateFormula


TEACHER_MOD = (
    Path(__file__).resolve().parents[2]
    / "neuron_as_deep_net/L5PC_NEURON_simulation/mods/Ca_HVA.mod"
)


class GateMDiagnosisTests(unittest.TestCase):
    def test_fresh_confirmation_is_disjoint_and_dense_fit_is_larger(self) -> None:
        bundle = prepare_gate_m_diagnosis(ExtractedGateFormula.from_mod(TEACHER_MOD))
        contract = bundle["contract"]
        self.assertFalse(contract["old_task1_sealed_rows_reused"])
        self.assertFalse(contract["fit_or_development_rows_reused"])
        self.assertFalse(contract["fresh_used_for_selection"])
        self.assertGreater(contract["dense_fit_count"], contract["base_fit_count"] * 5)
        self.assertTrue(all(count > 0 for count in contract["fresh_confirmation_counts"].values()))

    def test_compact_batch_stream_is_replayable_without_materialization(self) -> None:
        first = _paired_batch_generator(101, 32, 200017)
        replay = _paired_batch_generator(101, 32, 200017)
        different = _paired_batch_generator(101, 32, 200029)
        left = [next(first) for _ in range(10)]
        right = [next(replay) for _ in range(10)]
        other = [next(different) for _ in range(10)]
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(left, right)))
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(left, other)))
        self.assertTrue(all(len(row) == 32 for row in left))

    def test_registered_factorial_is_valid(self) -> None:
        GateMDiagnosisConfig().validate()
