"""RunPod-only scaling utilities for the GIADA paper-scale data track.

This package is deliberately separate from the exploratory ``hayflow_*``
notebooks.  It reuses their validated teacher instrumentation but does not
change their datasets, configurations, or experimental registries.
"""

from .config import ScaleConfig, load_scale_config
from .planning import ShardPlan, build_shard_plan, load_shard_plan
from .neuronio_inputs import NeuronIOInputConfig, sample_neuronio_actions
from .store import LeanShardWriter, validate_lean_shard

__all__ = [
    "LeanShardWriter",
    "NeuronIOInputConfig",
    "ScaleConfig",
    "ShardPlan",
    "build_shard_plan",
    "load_scale_config",
    "load_shard_plan",
    "sample_neuronio_actions",
    "validate_lean_shard",
]
