from dataclasses import replace

import numpy as np
import pytest

from src.giada_teacher.primitive_scaling import (
    PrimitiveScalingConfig,
    prepare_scaling_data,
)


class FakeFormula:
    def rates(self, voltage):
        m_inf = 1/(1+np.exp(-(voltage+35)/8))
        h_inf = 1/(1+np.exp((voltage+55)/7))
        return {"m_inf":m_inf,"h_inf":h_inf,"m_tau_ms":1.0,"h_tau_ms":10.0}

    def step(self, gate, state, voltage, dt):
        rates = self.rates(voltage)
        inf, tau = rates[f"{gate}_inf"], rates[f"{gate}_tau_ms"]
        return inf+(state-inf)*np.exp(-dt/tau)


def test_task5_data_is_deterministic_and_has_new_sealed_spec():
    config = PrimitiveScalingConfig()
    first = prepare_scaling_data(FakeFormula(), config)
    second = prepare_scaling_data(FakeFormula(), config)
    np.testing.assert_array_equal(first["fit"]["inputs"], second["fit"]["inputs"])
    assert first["contract"]["task4_sealed_accessed"] is False
    assert sum(spec["count"] for spec in first["sealed_spec"].values()) == 8192
    assert {spec["seed"] for spec in first["sealed_spec"].values()} == {75301, 75302, 75303}


def test_task5_registered_axes_are_fixed():
    config = PrimitiveScalingConfig()
    config.validate()
    assert config.widths == (16, 23, 32)
    assert config.checkpoints[-1] == 75000
    with pytest.raises(ValueError):
        replace(config, widths=(16, 23)).validate()
