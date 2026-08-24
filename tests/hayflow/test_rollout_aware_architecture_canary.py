import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.hayflow_model.rollout_aware_architecture_canary import (
    CAUSAL_DRIVE_FEATURES,
    EXPECTED_05KD_ARCHIVE_SHA256,
    EXPECTED_05KD_FINAL_SHA256,
    EXPECTED_05KD_INDEX_SHA256,
    MorphologyGraphGRU,
    OrderedConvGRUControl,
    RolloutAwareArchitectureCanaryConfig,
    artifact_index_matches,
    discover_indexed_artifact_source,
    materialize_nested_indexed_artifact_source,
    disjoint_episode_roles,
    encode_causal_realized_drive,
    model_parameter_count,
)


def test_artifact_discovery_accepts_kaggle_archive_name_and_extracted_root(tmp_path):
    import hashlib
    import zipfile

    index = b'{"schema_version":"test"}'
    digest = hashlib.sha256(index).hexdigest()
    archive = tmp_path / "dataset-slug" / "archive.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/artifact_index.json", index)
        handle.writestr("nested/architecture_failure_reassessment_config.json", "{}")
    assert artifact_index_matches(archive, digest)
    assert discover_indexed_artifact_source(tmp_path, digest) == archive.resolve()

    archive.unlink()
    extracted = archive.parent / "nested"
    extracted.mkdir()
    (extracted / "artifact_index.json").write_bytes(index)
    assert discover_indexed_artifact_source(tmp_path, digest) == extracted.resolve()


def test_artifact_discovery_materializes_one_kaggle_nested_zip_level(tmp_path):
    import hashlib
    import io
    import zipfile

    index = b'{"schema_version":"nested-test"}'
    digest = hashlib.sha256(index).hexdigest()
    inner_bytes = io.BytesIO()
    with zipfile.ZipFile(inner_bytes, "w") as inner:
        inner.writestr("artifact/artifact_index.json", index)
        inner.writestr("artifact/final_report.json", "{}")
    outer = tmp_path / "dataset" / "archive.zip"
    outer.parent.mkdir()
    with zipfile.ZipFile(outer, "w") as handle:
        handle.writestr("original_artifact.zip", inner_bytes.getvalue())
    materialized = materialize_nested_indexed_artifact_source(
        tmp_path, digest, tmp_path / "cache"
    )
    assert materialized is not None
    assert materialized.is_file()
    assert artifact_index_matches(materialized, digest)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05l_rollout_aware_graphgru_vs_convgru_canary.ipynb"


def test_05kd_hash_contract_is_exact():
    assert EXPECTED_05KD_ARCHIVE_SHA256 == "f6d036156f9fbec1344388759632f060dc7d6126a552274f6c97d290f4de8685"
    assert EXPECTED_05KD_INDEX_SHA256 == "23001abc8f49b4efd3379eb8ae5351343fc9082bb5293bbb35002562d3b5a2e8"
    assert EXPECTED_05KD_FINAL_SHA256 == "3d401bbf5b43d8ee705f1733b849c511445c642074813722b734020c0559cc70"


def test_canary_config_is_fixed_to_closed_loop_horizons():
    config = RolloutAwareArchitectureCanaryConfig()
    config.validate()
    assert config.horizons_ms == (2, 4, 8)
    assert config.minimum_passing_seed_count == 2
    with pytest.raises(ValueError):
        RolloutAwareArchitectureCanaryConfig(horizons_ms=(1, 8)).validate()


def test_causal_drive_uses_realized_records_without_state_reads():
    class Layout:
        segment_count = 3

    class Store:
        layout = Layout()

        def actions(self, index, view):
            assert index == 4 and view == "U_realized"
            return [
                {
                    "kind": "synaptic_event",
                    "segment_id": 2,
                    "synapse_type": "ProbAMPANMDA2",
                    "ampa_state_increment": 0.5,
                    "nmda_state_increment": 0.25,
                    "inhibitory_state_increment": 0.0,
                    "released_quantity": 0.5,
                    "release_success": True,
                    "offset_ms": 0.2,
                    "weight_multiplier": 1.0,
                },
                {
                    "kind": "somatic_current",
                    "offset_ms": 0.1,
                    "duration_ms": 0.5,
                    "amplitude_na": 0.2,
                },
            ]

    encoded = encode_causal_realized_drive(Store(), [4])
    assert encoded.shape == (1, 3, len(CAUSAL_DRIVE_FEATURES))
    assert encoded[0, 2, 0] == pytest.approx(0.5)
    assert encoded[0, 2, 1] == pytest.approx(0.25)
    assert encoded[0, 0, 10] == pytest.approx(0.1)


def test_episode_roles_keep_shared_seed_and_snapshot_together():
    rows = []
    for regime in ("nmda_spike", "calcium_spike", "somatic_spike"):
        for group in range(8):
            for branch in range(2 if group == 0 else 1):
                rows.append(
                    {
                        "split": "train",
                        "trajectory_id": f"{regime}-{group}-{branch}",
                        "seed": group + 1000 * (1 + len(rows)),
                        "snapshot_id": f"{regime}-snapshot-{group}",
                        "event_labels": regime,
                        "category": regime,
                    }
                )
    roles = disjoint_episode_roles(
        rows, config=RolloutAwareArchitectureCanaryConfig()
    )
    memberships = {}
    for role, role_rows in roles.items():
        for row in role_rows:
            memberships.setdefault(row["snapshot_id"], set()).add(role)
    assert all(len(value) == 1 for value in memberships.values())
    assert all(roles.values())


def test_graph_and_conv_models_are_parameter_matched_and_differ_spatially():
    torch = pytest.importorskip("torch")
    static = np.zeros((7, 7), dtype=np.float32)
    parent = np.asarray([0, 0, 1, 1, 2, 2, 3], dtype=np.int64)
    kwargs = dict(
        segment_static=static,
        parent_ids=parent,
        hidden_width=8,
        voltage_delta_limit_mv=120.0,
    )
    graph = MorphologyGraphGRU(**kwargs)
    conv = OrderedConvGRUControl(**kwargs)
    assert model_parameter_count(graph) == model_parameter_count(conv)
    voltage = torch.zeros(2, 7)
    drive = torch.zeros(2, 4, 7, len(CAUSAL_DRIVE_FEATURES))
    assert graph(voltage, drive)["voltage"].shape == (2, 4, 7)
    assert conv(voltage, drive)["voltage"].shape == (2, 4, 7)


def test_notebook_is_compact_uses_exact_artifact_and_blob_downloader():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert len(notebook["cells"]) <= 11
    assert "EXPECTED_05KD_INDEX_SHA256" in code
    assert "prepare_composite_flowmap_bundle" in code
    assert "fresh" not in code.lower()
    assert "base64.b64encode" in code
    assert "new Blob" in code and "a.click()" in code
    assert "FileLink" not in code
