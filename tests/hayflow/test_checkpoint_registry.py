import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads(
    (ROOT / "experiments" / "hayflow" / "checkpoint_registry.json").read_text()
)


def _result(experiment: str):
    return json.loads(
        (ROOT / "experiments" / "hayflow" / experiment / "result.json").read_text()
    )


def test_canonical_one_step_registry_matches_fresh_test_record():
    fresh = _result("05j_o_regenerative_fresh_test")
    canonical = REGISTRY["canonical_one_step_model"]
    assert REGISTRY["decision"]["canonical_report_model_id"] == canonical["id"]
    assert canonical["fresh_test"]["ensemble_mean_rmse_mv"] == fresh[
        "frozen_model_evaluation"
    ]["ensemble_mean_rmse_mv"]
    assert canonical["artifact_chain"]["fresh_test_archive"][
        "archive_sha256"
    ] == fresh["archive"]["sha256"]
    assert fresh["candidate_authorization_scope"] == "one_step_only"
    assert fresh["checkpoint_selection_performed"] is False


def test_decoder_dimensions_give_registered_parameter_count():
    architecture = REGISTRY["canonical_one_step_model"]["architecture"]
    segment_count = architecture["segment_count"]
    embedding = architecture["segment_embedding_dim"]
    feature_width = architecture["feature_width"]
    hidden = architecture["hidden_width"]
    expected = (
        segment_count * embedding
        + (feature_width + embedding) * hidden
        + hidden
        + hidden * hidden
        + hidden
        + hidden
        + 1
    )
    assert expected == architecture["trainable_decoder_parameter_count_derived"]


def test_seed_checkpoints_match_refit_and_fresh_test_records():
    refit = _result("05j_n_regenerative_decoder_refit")
    fresh = _result("05j_o_regenerative_fresh_test")
    registered = {
        row["seed"]: row for row in REGISTRY["canonical_one_step_model"]["checkpoints"]
    }
    for refit_row, fresh_row in zip(refit["runs"], fresh["frozen_model_evaluation"]["runs"]):
        row = registered[refit_row["seed"]]
        assert row["sha256"] == refit_row["checkpoint_sha256"]
        assert row["best_epoch"] == refit_row["best_epoch"]
        assert row["fresh_test_rmse_mv"] == fresh_row["rmse_mv"]


def test_recursive_frontier_is_not_misregistered_as_promoted():
    recursive = _result("06b_r_recursive_event_exposure_playground")
    frontier = REGISTRY["recursive_frontier"]["best_learned_diagnostic"]
    assert frontier["development_median_8ms_rmse_mv"] == recursive[
        "development_diagnostic"
    ]["median_8ms_rmse_mv"]
    assert frontier["physical_voltage_violation_count"] == 0
    assert recursive["calibration_selection"]["selected_candidate"] is None
    assert recursive["registered_interpretation"]["candidate_promoted"] is False
    assert REGISTRY["decision"]["canonical_free_running_model_id"] is None
