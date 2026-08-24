import ast
import inspect
import json
from pathlib import Path

import numpy as np

from src.hayflow_model.analytic_causal_gain_identifiability import (
    CAUSAL_GAIN_SCHEMES,
    EXPECTED_06BI_ARCHIVE_SHA256,
    EXPECTED_06BI_FINAL_SHA256,
    EXPECTED_06BI_INDEX_SHA256,
    FALLBACK_SCHEME,
    ORACLE_SCHEME,
    PRIMARY_SCHEME,
    AnalyticCausalGainConfig,
    AnalyticCausalGainIdentifiability,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_I = (
    ROOT / "experiments" / "hayflow" / "06b_i_voltage_objective_recalibration_playground"
)
EXPERIMENT_J = (
    ROOT / "experiments" / "hayflow" / "06b_j_analytic_causal_gain_identifiability"
)
NOTEBOOK = ROOT / "notebooks" / "06b_j_analytic_causal_gain_identifiability.ipynb"


def test_06bj_registers_exact_negative_06bi_result():
    result = json.loads((EXPERIMENT_I / "result.json").read_text())
    assert result["archive_sha256"] == EXPECTED_06BI_ARCHIVE_SHA256
    assert result["artifact_index_sha256"] == EXPECTED_06BI_INDEX_SHA256
    assert result["final_report_sha256"] == EXPECTED_06BI_FINAL_SHA256
    assert result["formal_diagnosis"] == "VOLTAGE_OBJECTIVE_RECALIBRATION_NOT_LEARNABLE"
    assert not result["decision"]["coupled_06c_canary_authorized"]


def test_06bj_preregisters_causal_hierarchy_and_sealed_oracle():
    prereg = json.loads((EXPERIMENT_J / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["causal_schemes"] == list(CAUSAL_GAIN_SCHEMES)
    assert prereg["primary_scheme"] == PRIMARY_SCHEME
    assert prereg["fallback_scheme"] == FALLBACK_SCHEME
    assert prereg["oracle"]["scheme"] == ORACLE_SCHEME
    assert prereg["oracle"]["uses_future_teacher_delta"]
    assert not prereg["oracle"]["eligible_for_selection"]
    assert not prereg["roles"]["new_independent_confirmation_claimed"]


def test_06bj_configuration_is_bounded_and_optimizer_free():
    config = AnalyticCausalGainConfig()
    config.validate()
    assert config.analytic_gain_minimum < config.analytic_reference_gain
    assert config.analytic_reference_gain < config.analytic_gain_maximum
    assert config.minimum_recursive_gain_over_global_fraction == 0.02
    fit = inspect.getsource(AnalyticCausalGainIdentifiability._fit_lookup)
    matrix = inspect.getsource(
        AnalyticCausalGainIdentifiability.fit_and_calibrate_analytic_lookups
    )
    assert "np.bincount" in fit
    assert "np.clip" in fit
    assert "optimizer" not in matrix.lower()
    assert "backward" not in matrix.lower()


def test_06bj_causal_cells_do_not_use_teacher_delta_but_oracle_does():
    config = AnalyticCausalGainConfig()
    instance = object.__new__(AnalyticCausalGainIdentifiability)
    instance.config = config
    instance.layout = type("Layout", (), {"region_names": ("soma", "dendrite")})()
    base = {
        "raw_delta": np.asarray([0.2, 3.0]),
        "voltage": np.asarray([-75.0, -30.0]),
        "target_delta": np.asarray([0.1, 8.0]),
        "region": np.asarray([0, 1]),
    }
    changed = {**base, "target_delta": np.asarray([20.0, 0.0])}
    for scheme in CAUSAL_GAIN_SCHEMES:
        assert np.array_equal(instance._cell_ids(base, scheme)[0], instance._cell_ids(changed, scheme)[0])
    assert not np.array_equal(
        instance._cell_ids(base, ORACLE_SCHEME)[0],
        instance._cell_ids(changed, ORACLE_SCHEME)[0],
    )


def test_06bj_finalization_keeps_oracle_out_of_selection_and_blocks_06c():
    source = inspect.getsource(
        AnalyticCausalGainIdentifiability.finalize_analytic_identifiability
    )
    assert source.index("if primary_pass") < source.index("elif fallback_pass")
    assert '"oracle_eligible_for_selection": False' in source
    assert '"coupled_06c_canary_authorized": False' in source
    assert '"full_training_authorized": False' in source
    assert '"neural_training_performed": False' in source


def test_06bj_notebook_is_compact_and_uses_stable_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 12
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "EXPECTED_06BI_INDEX_SHA256" in code
    assert "materialize_nested_indexed_artifact_source" in code
    assert "fit_and_calibrate_analytic_lookups" in code
    assert "evaluate_analytic_lookups" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
