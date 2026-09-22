import numpy as np

from src.giada_teacher.primitive_matrix_playground import (
    PrimitiveMatrixConfig,
    _evaluate_numpy,
    _numerical_predictor,
    prepare_primitive_matrix,
)


class FakeFormula:
    def rates(self, voltage):
        m_inf=1/(1+np.exp(-(voltage+35)/8)); h_inf=1/(1+np.exp((voltage+55)/7))
        return {"m_inf":m_inf,"h_inf":h_inf,"m_tau_ms":0.2+2*np.exp(-((voltage+40)/30)**2),"h_tau_ms":2+15*np.exp(-((voltage+55)/35)**2)}
    def step(self,gate,state,voltage,dt):
        rates=self.rates(voltage); inf=rates[f"{gate}_inf"]; tau=rates[f"{gate}_tau_ms"]
        return inf+(state-inf)*np.exp(-dt/tau)


def test_task4_contract_is_deterministic_and_sealed_is_not_materialized():
    config=PrimitiveMatrixConfig(pool_size=128)
    first=prepare_primitive_matrix(FakeFormula(),config); second=prepare_primitive_matrix(FakeFormula(),config)
    assert first["contract"]["task3e_sealed_accessed"] is False
    assert first["contract"]["fit_count"]==128
    np.testing.assert_array_equal(first["fit"]["inputs"],second["fit"]["inputs"])
    assert set(first["sealed_spec"])=={"central","voltage_tail","long_horizon"}


def test_formula_is_exact_accuracy_oracle():
    formula=FakeFormula(); config=PrimitiveMatrixConfig(pool_size=64)
    bundle=prepare_primitive_matrix(formula,config)
    metrics=_evaluate_numpy(_numerical_predictor(formula,"formula",config),bundle["development"])
    assert max(row["m_rmse"] for row in metrics.values())<1e-12
    assert max(row["h_rmse"] for row in metrics.values())<1e-12
    assert max(row["open_rmse"] for row in metrics.values())<1e-12
    assert sum(row["occupancy_violation_count"] for row in metrics.values())==0


def test_registered_matrix_has_all_orthogonal_families():
    config=PrimitiveMatrixConfig()
    assert set(config.learned_families)=={"direct_mlp","gru","physical_tau","direct_z"}
    assert set(config.numerical_families)=={"formula","lut_nearest","lut_linear","chebyshev"}
    assert config.checkpoints==(0,1000,3000,10000,30000,50000)
