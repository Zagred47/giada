"""Command line entry points for the isolated GIADA RunPod workflow."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from .config import load_scale_config
from .planning import ShardPlan, TrajectoryPlan, build_shard_plan, load_shard_plan, write_shard_plan
from .store import validate_lean_shard


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def command_plan(args: argparse.Namespace) -> None:
    config = load_scale_config(args.config)
    shards = build_shard_plan(config)
    destination = Path(args.output) / "plan.json"
    write_shard_plan(destination, config, shards)
    split_counts: Dict[str, int] = {}
    protocol_split_counts: Dict[str, Dict[str, int]] = {}
    for shard in shards:
        for row in shard.trajectories:
            split_counts[row.split] = split_counts.get(row.split, 0) + 1
            by_split = protocol_split_counts.setdefault(row.protocol, {})
            by_split[row.split] = by_split.get(row.split, 0) + 1
    report = {
        "schema_version": "giada-runpod-run-manifest-v1",
        "project": "GIADA",
        "track": "paper_scale_data_validation",
        "config": config.to_dict(),
        "plan": str(destination.resolve()),
        "shard_count": len(shards),
        "trajectory_count": config.trajectory_count,
        "transition_count": config.target_transitions,
        "trajectory_split_counts": split_counts,
        "protocol_split_counts": protocol_split_counts,
        "architecture_experiments_modified": False,
    }
    _write_json(Path(args.output) / "run_manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)


def _new_generator(args: argparse.Namespace, work_suffix: str):
    from .teacher import ScaleTeacherGenerator

    return ScaleTeacherGenerator(
        Path(args.elm_repo),
        Path(args.teacher_repo),
        Path(args.output) / "worker_runtime" / work_suffix,
        seed=int(args.worker_seed),
    )


def command_worker(args: argparse.Namespace) -> None:
    config, shards = load_shard_plan(args.plan)
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index must lie in [0, worker_count)")
    assigned = [row for row in shards if row.shard_index % args.worker_count == args.worker_index]
    print(
        f"[GIADA RunPod][worker {args.worker_index}/{args.worker_count}] "
        f"assigned {len(assigned)} shards",
        flush=True,
    )
    generator = _new_generator(args, f"worker-{args.worker_index:03d}")
    prepare = generator.prepare()
    print(
        f"[GIADA RunPod][worker {args.worker_index}] teacher ready; "
        f"burn-in={prepare['burnin_duration_ms']:.0f} ms",
        flush=True,
    )
    started = time.perf_counter()
    for completed, shard in enumerate(assigned, 1):
        try:
            report = generator.generate_shard(config, shard, Path(args.output))
        except Exception as error:
            failure = {
                "schema_version": "giada-runpod-shard-failure-v1",
                "shard_id": shard.shard_id,
                "plan_sha256": shard.plan_sha256,
                "worker_index": args.worker_index,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
            }
            _write_json(Path(args.output) / "status" / f"{shard.shard_id}.failed.json", failure)
            raise
        elapsed = max(time.perf_counter() - started, 1e-9)
        eta = elapsed / completed * (len(assigned) - completed)
        print(
            f"[GIADA RunPod][worker {args.worker_index}] {completed}/{len(assigned)} "
            f"{shard.shard_id} {'resume' if report['resumed'] else 'done'}; "
            f"{report['transitions_per_second']:.2f} transition/s; ETA {eta/60:.1f} min",
            flush=True,
        )


def command_benchmark(args: argparse.Namespace) -> None:
    config = load_scale_config(args.config)
    duration = int(args.duration_ms)
    protocol = config.input_protocols[0]
    protocol_family = "neuronio_background"
    protocol_arm = "canonical"
    if config.purpose in {
        "giada_hybrid_pilot",
        "giada_protocol_repair_pilot",
        "giada_hybrid_production_targeted",
    }:
        from .hybrid_inputs import hybrid_protocol_spec

        spec = hybrid_protocol_spec(protocol)
        protocol_family = spec.family
        protocol_arm = spec.arm
    trajectory = TrajectoryPlan(
        trajectory_id=f"benchmark-{protocol}",
        trajectory_index=0,
        seed=config.root_seed,
        split="train",
        duration_ms=duration,
        protocol=protocol,
        protocol_family=protocol_family,
        protocol_arm=protocol_arm,
        pair_id=f"benchmark-{protocol_family}",
    )
    identity = f"benchmark|{config.stage}|{protocol}|{duration}|{config.root_seed}"
    import hashlib

    shard = ShardPlan(
        shard_id="benchmark",
        shard_index=0,
        trajectories=(trajectory,),
        expected_transition_count=duration,
        plan_sha256=hashlib.sha256(identity.encode()).hexdigest(),
    )
    generator = _new_generator(args, "benchmark")
    prepare = generator.prepare()
    report = generator.generate_shard(config, shard, Path(args.output) / "benchmark")
    rate = report["transitions_per_second"]
    resource_usage = {}
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        resource_usage = {
            "peak_process_rss_mib": float(usage.ru_maxrss) / 1024.0,
            "process_user_cpu_seconds": float(usage.ru_utime),
            "process_system_cpu_seconds": float(usage.ru_stime),
        }
    except (ImportError, AttributeError):
        resource_usage = {
            "peak_process_rss_mib": None,
            "process_user_cpu_seconds": None,
            "process_system_cpu_seconds": None,
        }
    estimate = {
        "schema_version": "giada-runpod-teacher-benchmark-v1",
        "host": socket.gethostname(),
        "code_revision": _revision(Path(args.elm_repo)),
        "teacher_prepare": prepare,
        "benchmark": report,
        "resource_usage": resource_usage,
        "projected_stage_wall_hours_one_worker": config.target_transitions / rate / 3600.0,
        "projected_stage_storage_gib": report["size_bytes"] / duration * config.target_transitions / (1024**3),
        "note": "measure at least 6000 ms before provisioning the full CPU fleet",
    }
    _write_json(Path(args.output) / "benchmark_report.json", estimate)
    print(json.dumps(estimate, indent=2), flush=True)


def command_validate(args: argparse.Namespace) -> None:
    config, shards = load_shard_plan(args.plan)
    output = Path(args.output)
    blockers = []
    rows = []
    for index, shard in enumerate(shards, 1):
        path = output / "shards" / f"{shard.shard_id}.h5"
        done_path = output / "status" / f"{shard.shard_id}.done.json"
        if not path.is_file() or not done_path.is_file():
            blockers.append(f"missing {shard.shard_id}")
            continue
        report = validate_lean_shard(path, expected_transition_count=shard.expected_transition_count)
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if not report["valid"]:
            blockers.append(f"invalid {shard.shard_id}: {report['blockers']}")
        if done.get("plan_sha256") != shard.plan_sha256 or done.get("sha256") != report["sha256"]:
            blockers.append(f"identity mismatch {shard.shard_id}")
        rows.append(report)
        if index == 1 or index == len(shards) or index % max(1, len(shards) // 20) == 0:
            print(f"[GIADA RunPod][validation] {index}/{len(shards)}", flush=True)
    validation = {
        "schema_version": "giada-runpod-validation-v1",
        "valid": not blockers,
        "blockers": blockers,
        "expected_shard_count": len(shards),
        "validated_shard_count": len(rows),
        "expected_transition_count": config.target_transitions,
        "validated_transition_count": sum(row["transition_count"] for row in rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
    }
    _write_json(output / "validation_report.json", validation)
    terminal_summary = {
        key: value
        for key, value in validation.items()
        if key != "blockers"
    }
    terminal_summary["blocker_count"] = len(blockers)
    terminal_summary["blocker_examples"] = blockers[:10]
    print(json.dumps(terminal_summary, indent=2), flush=True)
    if not validation["valid"]:
        raise SystemExit(2)


def command_audit_corpus(args: argparse.Namespace) -> None:
    from .corpus_audit import audit_soma_corpus

    def progress(index: int, total: int) -> None:
        if index == 1 or index == total or index % 10 == 0:
            print(
                f"[GIADA RunPod][corpus audit] {index}/{total} shards",
                flush=True,
            )

    report = audit_soma_corpus(
        Path(args.corpus), plan_path=args.plan, progress=progress
    )
    destination = Path(args.output)
    _write_json(destination, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["valid"]:
        raise SystemExit(2)


def command_audit_hybrid(args: argparse.Namespace) -> None:
    from .hybrid_audit import audit_hybrid_corpus

    def progress(index: int, total: int) -> None:
        if index == 1 or index == total or index % 20 == 0:
            print(f"[GIADA RunPod][hybrid audit] {index}/{total} shards", flush=True)

    report = audit_hybrid_corpus(
        Path(args.corpus), plan_path=Path(args.plan), progress=progress
    )
    _write_json(Path(args.output), report)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "blockers": report["blockers"],
                "trajectory_count": report["trajectory_count"],
                "protocol_summaries": report["protocol_summaries"],
                "scientific_outcome_assessment": report[
                    "scientific_outcome_assessment"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["valid"]:
        raise SystemExit(2)


def command_train(args: argparse.Namespace) -> None:
    import yaml
    from .training import MatchedTrainingConfig, PaperScaleMatchedTrainer

    values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = MatchedTrainingConfig.from_mapping(values.get("giada_matched_training", values))
    trainer = PaperScaleMatchedTrainer(
        Path(args.corpus), Path(args.output), config, code_revision=_revision(Path(args.elm_repo))
    )
    try:
        report = trainer.train()
    finally:
        trainer.corpus.close()
    print(json.dumps({
        "valid": report["valid"],
        "median_rmse_mv": report["final_median_soma_rmse_mv"],
        "median_active_rmse_mv": report["final_median_active_soma_rmse_mv"],
        "median_spike_transition_rmse_mv": report[
            "final_median_spike_transition_rmse_mv"
        ],
        "median_component_rmse_mv": report["final_median_component_rmse_mv"],
        "seed_dispersion": report["seed_dispersion"],
        "paired_statistical_inference": report["paired_statistical_inference"],
        "giada_reduction_vs_branch_elm": report["giada_relative_rmse_reduction_vs_branch_elm"],
        "decision": report["registered_decision"],
    }, indent=2), flush=True)


def command_compose_s1e(args: argparse.Namespace) -> None:
    from .production_corpus import build_and_audit_s1e_composite

    def progress(index: int, total: int) -> None:
        if index == 1 or index == total or index % 20 == 0:
            print(f"[GIADA RunPod][S1e composite audit] {index}/{total} shards", flush=True)

    report = build_and_audit_s1e_composite(
        Path(args.background),
        Path(args.targeted),
        Path(args.output),
        progress=progress,
    )
    print(json.dumps({
        "valid": report["valid"],
        "blockers": report["blockers"],
        "composition": report["composition"],
        "support_checks": report["support_checks"],
    }, indent=2), flush=True)
    if not report["valid"]:
        raise SystemExit(2)


def command_compose_s2(args: argparse.Namespace) -> None:
    from .production_corpus import build_and_audit_s2_composite

    def progress(index: int, total: int) -> None:
        if index == 1 or index == total or index % 50 == 0:
            print(f"[GIADA RunPod][S2 composite audit] {index}/{total} shards", flush=True)

    report = build_and_audit_s2_composite(
        Path(args.background),
        Path(args.targeted),
        Path(args.output),
        progress=progress,
    )
    print(json.dumps({
        "valid": report["valid"],
        "blockers": report["blockers"],
        "composition": report["composition"],
        "support_checks": report["support_checks"],
    }, indent=2), flush=True)
    if not report["valid"]:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="GIADA paper-scale RunPod workflow")
    sub = result.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write an immutable staged shard plan")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.set_defaults(func=command_plan)
    for name, function in (("worker", command_worker), ("benchmark", command_benchmark)):
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path)
        item.add_argument("--plan", type=Path)
        item.add_argument("--output", required=True, type=Path)
        item.add_argument("--elm-repo", default=Path.cwd(), type=Path)
        item.add_argument("--teacher-repo", required=True, type=Path)
        item.add_argument("--worker-seed", default=7_000_001, type=int)
        if name == "worker":
            item.add_argument("--worker-index", required=True, type=int)
            item.add_argument("--worker-count", required=True, type=int)
        else:
            item.add_argument("--duration-ms", default=6000, type=int)
        item.set_defaults(func=function)
    validate = sub.add_parser("validate", help="hash and validate every completed shard")
    validate.add_argument("--plan", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    validate.set_defaults(func=command_validate)
    audit = sub.add_parser(
        "audit-corpus", help="summarize voltage-target support in every soma shard"
    )
    audit.add_argument("--corpus", required=True, type=Path)
    audit.add_argument("--plan", type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.set_defaults(func=command_audit_corpus)
    hybrid = sub.add_parser(
        "audit-hybrid", help="verify the paired GIADA hybrid pilot and summarize outcomes"
    )
    hybrid.add_argument("--corpus", required=True, type=Path)
    hybrid.add_argument("--plan", required=True, type=Path)
    hybrid.add_argument("--output", required=True, type=Path)
    hybrid.set_defaults(func=command_audit_hybrid)
    train = sub.add_parser("train", help="run the paired GPU comparison on validated shards")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--corpus", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--elm-repo", default=Path.cwd(), type=Path)
    train.set_defaults(func=command_train)
    compose = sub.add_parser(
        "compose-s1e", help="seal and audit the two-part S1e hybrid corpus"
    )
    compose.add_argument("--background", required=True, type=Path)
    compose.add_argument("--targeted", required=True, type=Path)
    compose.add_argument("--output", required=True, type=Path)
    compose.set_defaults(func=command_compose_s1e)
    compose_s2 = sub.add_parser(
        "compose-s2", help="seal and audit the two-part S2 hybrid corpus"
    )
    compose_s2.add_argument("--background", required=True, type=Path)
    compose_s2.add_argument("--targeted", required=True, type=Path)
    compose_s2.add_argument("--output", required=True, type=Path)
    compose_s2.set_defaults(func=command_compose_s2)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "benchmark" and args.config is None:
        raise SystemExit("benchmark requires --config")
    if args.command == "worker" and args.plan is None:
        raise SystemExit("worker requires --plan")
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
