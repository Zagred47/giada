import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.hayflow_model.topology_controlled_recurrence_expansion import (
    EXPECTED_05L_ARCHIVE_SHA256,
    EXPECTED_05L_FINAL_SHA256,
    EXPECTED_05L_INDEX_SHA256,
    TopologyControlledRecurrenceConfig,
    expanded_train_episode_roles,
    topology_relabelled_parent_ids,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "05m_topology_controlled_recurrence_expansion.ipynb"


def _episode_rows():
    rows = []
    for regime, label in enumerate(("nmda_plateau", "somatic_spike")):
        for component in range(8):
            rows.append(
                {
                    "trajectory_id": f"trajectory-{regime}-{component}",
                    "split": "train",
                    "seed": 1000 * regime + component,
                    "snapshot_id": f"snapshot-{regime}-{component}",
                    "snapshot_source": f"snapshot-{regime}-{component}",
                    "event_labels": [label],
                    "category": "event",
                }
            )
    rows.append(
        {
            "trajectory_id": "excluded-validation",
            "split": "validation",
            "seed": 99999,
            "snapshot_id": "excluded",
            "event_labels": ["nmda_plateau"],
        }
    )
    return rows


def test_registered_05l_hashes_are_exact():
    assert EXPECTED_05L_ARCHIVE_SHA256 == "9a5c423aba80da51830a57c6f97808a3366069c6e42807193143d49de13be634"
    assert EXPECTED_05L_INDEX_SHA256 == "03e77d8a7104a25b2636f05f406031b01b81604c9e15d1dcb4d12a859ac0d757"
    assert EXPECTED_05L_FINAL_SHA256 == "a43227c545088b7bc1f9dbf2bafbac9f73534504033cbd365b9e6fe805cca930"


def test_expansion_preserves_roles_and_adds_unused_train_components():
    config = TopologyControlledRecurrenceConfig(
        maximum_extra_fit_components_per_regime=8
    )
    roles, report = expanded_train_episode_roles(_episode_rows(), config=config)
    assert {name: len(rows) for name, rows in roles.items()} == {
        "fit": 12,
        "calibration": 2,
        "development": 2,
    }
    assert sum(row["extra_fit_component_count"] for row in report.values()) == 6
    memberships = {
        role: {row["trajectory_id"] for row in rows}
        for role, rows in roles.items()
    }
    assert not memberships["fit"] & memberships["calibration"]
    assert not memberships["fit"] & memberships["development"]
    assert not memberships["calibration"] & memberships["development"]
    assert all(row["split"] == "train" for rows in roles.values() for row in rows)


def test_relabelled_tree_is_isomorphic_but_misaligned():
    parent = np.asarray([0, 0, 0, 1, 1, 2, 2, 4], dtype=np.int64)
    relabelled = topology_relabelled_parent_ids(parent, seed=905117)
    assert relabelled[0] == 0
    assert not np.array_equal(parent, relabelled)
    authentic_degree = sorted(np.bincount(parent, minlength=len(parent)).tolist())
    relabelled_degree = sorted(
        np.bincount(relabelled, minlength=len(parent)).tolist()
    )
    assert authentic_degree == relabelled_degree
    for node in range(1, len(parent)):
        visited = set()
        current = node
        while relabelled[current] != current:
            assert current not in visited
            visited.add(current)
            current = int(relabelled[current])
        assert current == 0


def test_three_candidates_are_parameter_matched():
    torch = pytest.importorskip("torch")
    from src.hayflow_model.rollout_aware_architecture_canary import (
        MorphologyGraphGRU,
        OrderedConvGRUControl,
        model_parameter_count,
    )

    static = np.zeros((8, 5), dtype=np.float32)
    parent = np.asarray([0, 0, 0, 1, 1, 2, 2, 4], dtype=np.int64)
    relabelled = topology_relabelled_parent_ids(parent, seed=905117)
    kwargs = dict(segment_static=static, hidden_width=8, voltage_delta_limit_mv=120)
    models = [
        MorphologyGraphGRU(parent_ids=parent, **kwargs),
        MorphologyGraphGRU(parent_ids=relabelled, **kwargs),
        OrderedConvGRUControl(parent_ids=parent, **kwargs),
    ]
    assert len({model_parameter_count(model) for model in models}) == 1
    initial = torch.full((2, 8), -70.0)
    drive = torch.zeros((2, 8, 8, 12))
    assert all(model(initial, drive)["voltage"].shape == (2, 8, 8) for model in models)


def test_graph_pair_can_share_parameters_without_overwriting_topology():
    torch = pytest.importorskip("torch")
    from src.hayflow_model.rollout_aware_architecture_canary import MorphologyGraphGRU

    static = np.zeros((8, 5), dtype=np.float32)
    parent = np.asarray([0, 0, 0, 1, 1, 2, 2, 4], dtype=np.int64)
    relabelled = topology_relabelled_parent_ids(parent, seed=905117)
    kwargs = dict(segment_static=static, hidden_width=8, voltage_delta_limit_mv=120)
    authentic = MorphologyGraphGRU(parent_ids=parent, **kwargs)
    control = MorphologyGraphGRU(parent_ids=relabelled, **kwargs)
    with torch.no_grad():
        for name, value in authentic.named_parameters():
            dict(control.named_parameters())[name].copy_(value)
    assert all(
        torch.equal(value, dict(control.named_parameters())[name])
        for name, value in authentic.named_parameters()
    )
    assert not torch.equal(authentic.parent_ids, control.parent_ids)


def test_05m_notebook_is_compact_causal_and_uses_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 11
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "EXPECTED_05L_INDEX_SHA256" in code
    assert "discover_indexed_artifact_source" in code
    assert "prepare_composite_flowmap_bundle" in code
    assert "TopologyControlledRecurrenceExpansion" in code
    assert "base64.b64encode" in code
    assert "new Blob" in code
    assert "FileLink" not in code
    assert "fresh_05jo" not in code.lower()
