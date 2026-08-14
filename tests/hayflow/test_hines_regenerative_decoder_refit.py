import ast
import hashlib
import json
from pathlib import Path

import pytest

import src.hayflow_model.hines_regenerative_decoder_refit as refit_module
from src.hayflow_model.hines_regenerative_decoder_refit import (
    HinesRegenerativeDecoderRefitConfig,
    deterministic_pair_partition,
)


ROOT = Path(__file__).resolve().parents[2]


def test_05jn_pair_partition_is_deterministic_disjoint_and_outcome_blind():
    pair_ids = [f"pair-{index:03d}" for index in range(20)]
    first = deterministic_pair_partition(pair_ids, 5, salt="registered")
    second = deterministic_pair_partition(pair_ids, 5, salt="registered")
    assert first == second
    fit = set(first["fit_pair_positions"])
    calibration = set(first["calibration_pair_positions"])
    assert not fit & calibration
    assert fit | calibration == set(range(20))
    assert len(fit) == 15 and len(calibration) == 5
    # Only identities and the preregistered salt enter the split function.
    assert "target" not in deterministic_pair_partition.__code__.co_varnames
    assert "outcome" not in deterministic_pair_partition.__code__.co_varnames


def test_05jn_registered_config_has_exact_roles_and_three_seed_gate():
    values = json.loads(json.dumps({
        "old_fit_calibration_pair_count": 11,
        "new_train_calibration_pair_count": 19,
        "expected_old_fit_pair_count": 54,
        "expected_new_train_pair_count": 96,
        "seeds": [17, 29, 43],
    }))
    config = HinesRegenerativeDecoderRefitConfig.from_mapping(values)
    assert config.expected_old_fit_pair_count == 54
    assert config.expected_new_train_pair_count == 96
    assert 54 + 96 - 11 - 19 == 120
    assert 11 + 19 == 30
    assert config.minimum_passing_seeds == 2
    with pytest.raises(ValueError, match="preregistered"):
        HinesRegenerativeDecoderRefitConfig(expected_new_train_pair_count=95).validate()


def test_05jm_registered_hashes_match_05jn_provenance_contract():
    result = json.loads(
        (ROOT / "experiments/hayflow/05j_m_regenerative_training_support/result.json")
        .read_text(encoding="utf-8")
    )
    assert result["archive"]["sha256"] == refit_module.EXPECTED_05JM_ARCHIVE_SHA256
    assert result["archive"]["artifact_index_sha256"] == refit_module.EXPECTED_05JM_INDEX_SHA256
    assert result["archive"]["final_report_sha256"] == refit_module.EXPECTED_05JM_FINAL_SHA256
    assert result["training_shard"]["transition_store_sha256"] == refit_module.EXPECTED_05JM_TRANSITION_SHA256
    assert result["training_shard"]["retained_pair_count"] == 96
    assert result["fresh_test"]["outcomes_generated"] is False


def test_05jn_verifies_but_does_not_materialize_sealed_fresh_test(
    tmp_path, monkeypatch
):
    source = tmp_path / "mounted" / "artifact"
    source.mkdir(parents=True)
    for name in refit_module._05JM_TRAINING_MEMBERS - {"artifact_index.json"}:
        payload = b'{}' if name.endswith(".json") else name.encode()
        (source / name).write_bytes(payload)
    sealed = source / "sealed_fresh_test_plan.json"
    sealed.write_text('{"secret_seed": 123}', encoding="utf-8")
    artifacts = []
    for path in sorted(source.iterdir()):
        if path.name == "artifact_index.json":
            continue
        artifacts.append({
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    index_bytes = json.dumps({"artifacts": artifacts}, sort_keys=True).encode()
    (source / "artifact_index.json").write_bytes(index_bytes)
    monkeypatch.setattr(
        refit_module, "EXPECTED_05JM_INDEX_SHA256", hashlib.sha256(index_bytes).hexdigest()
    )
    monkeypatch.setattr(
        refit_module,
        "EXPECTED_05JM_FINAL_SHA256",
        hashlib.sha256((source / "final_report.json").read_bytes()).hexdigest(),
    )
    root, _, contract = refit_module._verified_training_artifact_root(
        source.parent, tmp_path / "filtered"
    )
    assert (root / "transition_dataset.h5").is_file()
    assert not (root / "sealed_fresh_test_plan.json").exists()
    assert contract["fresh_test_member_extracted"] is False
    assert contract["fresh_test_member_parsed"] is False


def test_05jn_notebook_refits_only_registered_decoder_and_keeps_test_sealed():
    notebook = json.loads(
        (ROOT / "notebooks/05j_n_regenerative_decoder_refit.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "torch.cuda.is_available" in source
    assert "prepare_refit_roles" in source and "run_decoder_refit" in source
    assert "combined_internal_fit_pair_count']==120" in source
    assert "combined_internal_calibration_pair_count']==30" in source
    assert "not role_report['development_used_to_fit_representation']" in source
    assert "not refit_report['development_used_for_checkpoint_selection']" in source
    assert "not refit_report['fresh_test_inputs_extracted']" in source
    assert "not refit_report['fresh_test_outcomes_generated']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "FileLink" not in source and "rglob('/kaggle" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))
