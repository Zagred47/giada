"""Common GPU baseline contract for GIADA atomic-model comparisons."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def build_gpu_baseline_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema_version": "giada-common-gpu-baseline-v1",
        "task": "0.6",
        "scope": "Ca_HVA atomic held-voltage playground",
        "hardware_policy": {
            "accelerator": "single NVIDIA CUDA GPU",
            "same_device_for_paired_arms": True,
            "required_environment_record": [
                "GPU model", "GPU UUID when available", "driver", "CUDA runtime",
                "PyTorch version", "Python version", "OS", "git revision",
            ],
            "tenstorrent_in_scope": False,
        },
        "paired_randomness": {
            "model_seeds": [17, 29, 43],
            "data_order_seed_offset": 100000,
            "same_minibatch_index_stream_per_seed_across_arms": True,
            "same_role_membership_across_arms": True,
            "rng_states_saved_in_checkpoints": ["python", "numpy", "torch_cpu", "torch_cuda"],
        },
        "determinism": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
            "allow_tf32": False,
            "failure_policy": "fail closed if an operation has no deterministic CUDA implementation",
        },
        "numeric_policy": {
            "teacher_and_dataset_dtype": "float64",
            "primary_training_dtype": "float32",
            "automatic_mixed_precision_primary": False,
            "float64_evaluation_accumulator": True,
            "amp_allowed_only_as_separate_optimization_arm": True,
        },
        "data_contract": {
            "split_artifact": "experiments/teacher_atomic_domain_splits_v1.json",
            "split_sha256": "e278a501a77fecc759e8818124344dd2e7bd85ce830b49854a37b8f259f2180e",
            "batch_size": 1024,
            "drop_last": False,
            "shuffle": "one persisted permutation stream per data-order seed",
            "normalization_fit_role_only": True,
            "sealed_test_access_during_training": False,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate_grid": [0.01, 0.003, 0.001],
            "weight_decay": 0.0,
            "gradient_clipping_norm": 1.0,
            "maximum_steps": 10000,
            "checkpoint_steps": [0, 100, 300, 1000, 3000, 10000],
            "same_hyperparameter_trial_count_per_arm": True,
            "selection_role": "development only",
            "early_stopping_changes_exposure": False,
            "primary_loss": "mean squared error in next gate occupancy",
        },
        "comparability": {
            "paired_axes": [
                "dataset rows", "row order", "batch boundaries", "seeds", "optimizer family",
                "learning-rate grid", "step budgets", "precision", "metrics", "hardware",
            ],
            "reported_capacity": ["trainable parameters", "serialized bytes", "estimated MACs per case"],
            "capacity_matching_is_separate_from_equal_budget_comparison": True,
            "formula_lut_polynomial_have_no_optimizer_budget": True,
            "analytic_baselines_report_accuracy_and_cost_on_identical evaluation rows": True,
        },
        "metrics": {
            "primary": [
                "gate next-state RMSE", "gate next-state maximum absolute error",
                "per-domain RMSE", "long-rollout gate RMSE and bound violations",
            ],
            "secondary": [
                "derived conductance/current error", "x_inf error when applicable",
                "tau error when applicable", "monotonic relaxation violations",
            ],
            "aggregation": "report m and h separately before macro averages",
            "uncertainty": "paired per-seed values plus median and full range; no pseudo-replication over rows",
        },
        "latency_benchmark": {
            "modes": ["eager", "torch_compile"],
            "modes_reported_separately": True,
            "batch_sizes": [1, 64, 1024],
            "warmup_iterations": 100,
            "timed_iterations": 1000,
            "timer": "torch.cuda.Event with torch.cuda.synchronize after end-event recording",
            "statistics": ["median", "p05", "p95", "minimum"],
            "compile_time_reported_separately": True,
            "host_to_device_transfer_excluded_from_kernel_latency": True,
            "end_to_end_throughput_reported_separately": True,
            "peak_cuda_memory_reported": True,
        },
        "model_selection_firewall": {
            "development_can_select": ["learning rate", "checkpoint", "architecture variant"],
            "sealed_test_can_select": [],
            "sealed_test_open_condition": "candidate code, weights, normalization and thresholds frozen",
            "new_candidate_after_test": "requires a new sealed confirmation version",
            "embedded_confirmation": "never used for atomic architecture selection",
        },
        "required_outputs": [
            "environment.json", "run_manifest.json", "paired_stream_hashes.json",
            "checkpoint_registry.json", "development_metrics.parquet",
            "sealed_test_metrics.parquet", "latency_report.json", "final_report.json",
        ],
    }
    contract["validation"] = validate_gpu_baseline_contract(contract)
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return contract


def validate_gpu_baseline_contract(contract: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    seeds = contract["paired_randomness"]["model_seeds"]
    checkpoints = contract["optimization"]["checkpoint_steps"]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        blockers.append("at least three unique paired model seeds are required")
    if checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0:
        blockers.append("checkpoint steps must be unique, sorted and include the initialization")
    if checkpoints[-1] != contract["optimization"]["maximum_steps"]:
        blockers.append("final checkpoint must equal maximum step budget")
    if contract["model_selection_firewall"]["sealed_test_can_select"]:
        blockers.append("sealed test must not select any component")
    if contract["numeric_policy"]["automatic_mixed_precision_primary"]:
        blockers.append("AMP cannot be part of the primary numerical comparison")
    if not contract["paired_randomness"]["same_minibatch_index_stream_per_seed_across_arms"]:
        blockers.append("minibatch streams must be paired across learned arms")
    if not contract["latency_benchmark"]["modes_reported_separately"]:
        blockers.append("eager and compiled latency must not be pooled")
    if contract["hardware_policy"]["tenstorrent_in_scope"]:
        blockers.append("Tenstorrent is outside the current GPU phase")
    return {"valid": not blockers, "blockers": blockers, "check_count": 8}


def render_gpu_baseline_markdown(contract: dict[str, Any]) -> str:
    return "\n".join([
        "# GIADA Task 0.6 — common GPU baseline", "",
        f"- Seeds: `{contract['paired_randomness']['model_seeds']}`",
        f"- Batch size: **{contract['data_contract']['batch_size']}**",
        f"- Maximum optimizer steps: **{contract['optimization']['maximum_steps']:,}**",
        f"- Checkpoints: `{contract['optimization']['checkpoint_steps']}`",
        f"- Learning-rate grid: `{contract['optimization']['learning_rate_grid']}`",
        "- Primary training precision: **float32 without AMP**",
        "- Teacher/reference precision: **float64**", "",
        "## Fairness firewall", "",
        "Learned arms receive identical split membership, minibatch indices, batch boundaries, seeds, optimizer family, learning-rate trial count, checkpoint budgets and metrics. Analytic baselines receive the identical evaluation rows but no fictitious optimizer budget.", "",
        "Development data may select hyperparameters and checkpoints. Sealed tests are opened only after code, weights, normalization and thresholds are frozen; they select nothing.", "",
        "## GPU timing", "",
        "Eager and `torch.compile` results are reported separately. CUDA events and synchronization are mandatory; compile time, steady-state kernel latency, end-to-end throughput and peak memory are distinct measurements.", "",
        f"Contract valid: **{contract['validation']['valid']}**", "",
    ])
