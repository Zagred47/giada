"""S4 coordinator and hostname-bound CPU launch; no automatic claim recovery.

Run with ``python -m src.giada_runpod.s4_production --help``. Generation still
uses the qualified CLI worker and its immutable modulo partition/seed rule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from .config import load_scale_config
from .distributed_generation import (
    _atomic_json, _sha256_file, acquire_exclusive_claim,
    load_distributed_manifest, load_worker_partition, write_distributed_plan,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path('/workspace/giada-data/s4-hybrid-production-v1')
TEACHER_COMMIT = '074c4666300a8ad246601dab179a97a6942f0f29'
COMPONENTS = ('background', 'targeted')
WORKERS = 128


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def revision(repo):
    if git(repo, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError(f'tracked edits in {repo}; deploy a clean pinned revision')
    return git(repo, 'rev-parse', 'HEAD')


def run_manifest(root):
    manifest = read(root / 's4_run.json')
    if manifest.get('schema_version') != 'giada-s4-production-run-v1':
        raise ValueError('not an S4 production root')
    if manifest['global_worker_count'] != WORKERS:
        raise ValueError('S4 worker namespace changed')
    if manifest['code_revision'] != revision(REPO):
        raise ValueError('repository revision differs from frozen production run')
    for component in COMPONENTS:
        _, planned = load_distributed_manifest(root / component / 'distributed_plan')
        if planned['plan_identity_sha256'] != manifest['plans'][component]:
            raise ValueError(f'{component}: frozen plan identity changed')
    return manifest


@contextmanager
def coordinator(root):
    claim = acquire_exclusive_claim(root / 'coordinator.claim.json', kind='coordinator',
                                    identity='s4-production', worker_index=-1,
                                    global_worker_count=WORKERS)
    try:
        yield
    finally:
        claim.release()


def prepare(root):
    code = revision(REPO)
    root.mkdir(parents=True, exist_ok=False)
    with coordinator(root):
        plans = {}
        for component in COMPONENTS:
            config = load_scale_config(REPO / 'runpod_scale' / 'configs' /
                                       f's4_hybrid_{component}_candidate.yml')
            print(f'[S4 prepare] {component}: writing immutable partitions', flush=True)
            manifest = write_distributed_plan(root / component / 'distributed_plan', config, WORKERS)
            if any(row['shard_count'] != 90 for row in manifest['partitions']):
                raise ValueError('S4 requires exactly 90 shards per component per worker')
            plans[component] = manifest['plan_identity_sha256']
            print(f'[S4 prepare] {component}: {manifest["shard_count"]} shards', flush=True)
        _atomic_json(root / 'fleet.json', {'pods': []})
        _atomic_json(root / 's4_run.json', dict(
            schema_version='giada-s4-production-run-v1', created_at_utc=now(),
            code_revision=code, teacher_commit=TEACHER_COMMIT,
            global_worker_count=WORKERS, plans=plans,
            target_cpu_wall_hours=24, production_started_at_epoch=None,
            support_gate_basis='eight times the preregistered S3 support density bounds',
            config_packing={'background': 2, 'targeted': 100},
        ))
    print(f'[S4 prepare] ready: {root}; no generation launched', flush=True)


def validate_assignment(pods, hostname, start, count, cpus):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', hostname):
        raise ValueError('invalid hostname')
    if not 0 <= start < WORKERS or count <= 0 or start + count > WORKERS:
        raise ValueError('worker range outside 0..127')
    if cpus <= 0 or count > cpus:
        raise ValueError('worker count exceeds declared CPU capacity')
    requested = set(range(start, start + count))
    for pod in pods:
        if pod['hostname'] == hostname:
            raise ValueError('hostname already registered; no implicit reassignment')
        if requested.intersection(range(pod['start'], pod['start'] + pod['count'])):
            raise ValueError('worker range overlaps a registered pod')


def register(root, hostname, start, count, cpus):
    run_manifest(root)
    with coordinator(root):
        if (root / 'SEALED.json').exists():
            raise ValueError('corpus already sealed')
        fleet = read(root / 'fleet.json')
        validate_assignment(fleet['pods'], hostname, start, count, cpus)
        fleet['pods'].append(dict(hostname=hostname, start=start, count=count,
                                  declared_cpus=cpus, registered_at_utc=now()))
        _atomic_json(root / 'fleet.json', fleet)
    print(f'[S4 register] {hostname}: indices {start}..{start + count - 1}', flush=True)


def assignment(root, hostname):
    fleet = read(root / 'fleet.json')
    verified = []
    for pod in fleet['pods']:
        validate_assignment(verified, pod['hostname'], pod['start'], pod['count'], pod['declared_cpus'])
        verified.append(pod)
    matches = [pod for pod in verified if pod['hostname'] == hostname]
    if len(matches) != 1:
        raise ValueError(f'hostname {hostname} is not registered; nothing started')
    pod = matches[0]
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    if available is None or available < pod['count']:
        raise ValueError('insufficient visible CPUs for this assignment')
    # Check known container limits as well as affinity; host-wide free/nproc
    # alone need not represent a container's allocation. Missing limits remain
    # unknown and are not presented as unlimited physical capacity.
    quotas = []
    cpu_max = Path('/sys/fs/cgroup/cpu.max')
    if cpu_max.is_file():
        quota, period = cpu_max.read_text().split()
        if quota != 'max':
            quotas.append(int(quota) / int(period))
    for base in ('/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct'):
        quota_path = Path(base) / 'cpu.cfs_quota_us'
        period_path = Path(base) / 'cpu.cfs_period_us'
        if quota_path.is_file() and period_path.is_file():
            quota = int(quota_path.read_text())
            if quota > 0:
                quotas.append(quota / int(period_path.read_text()))
    if quotas and min(quotas) < pod['count']:
        raise ValueError('CPU cgroup quota smaller than registered worker count')
    for limit_name, usage_name in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        limit_path, usage_path = Path(limit_name), Path(usage_name)
        if limit_path.is_file() and usage_path.is_file():
            limit = limit_path.read_text().strip()
            if limit != 'max':
                spare = int(limit) - int(usage_path.read_text())
                if spare < (pod['count'] * 700 + 1024) * 1024**2:
                    raise ValueError('insufficient cgroup memory headroom (700 MiB/worker plus 1 GiB)')
    return pod


def start(root, teacher, *, finalizer=False):
    run_manifest(root)
    host = socket.gethostname()
    if not finalizer:
        assignment(root, host)
    if not finalizer and revision(teacher) != TEACHER_COMMIT:
        raise ValueError('wrong teacher revision')
    log_root = root / 'logs' / host
    log_root.mkdir(parents=True, exist_ok=True)
    command = 'finalize' if finalizer else 'run'
    log_path = log_root / f'{command}-{uuid.uuid4().hex}.log'
    with log_path.open('xb') as log:
        child = subprocess.Popen(
            [sys.executable, '-u', '-m', 'src.giada_runpod.s4_production', '--root', str(root),
             command, '--teacher', str(teacher)], cwd=REPO,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    time.sleep(2)
    if child.poll() is not None and child.returncode != 0:
        raise RuntimeError(f'launcher refused/failed; inspect {log_path}')
    print(json.dumps(dict(hostname=host, pid=child.pid, log=str(log_path)), indent=2), flush=True)


def finalize(root, teacher=None):
    """Wait for exact complete coverage, then seal once, independent of console."""
    run_manifest(root)
    claim = acquire_exclusive_claim(root / 'finalizer.claim.json', kind='finalizer',
        identity='s4-finalizer', worker_index=-1, global_worker_count=WORKERS)
    try:
        while True:
            complete = True
            summary = {}
            for component in COMPONENTS:
                output = root / component
                count = len(list((output / 'status').glob('*.done.json')))
                failures = len(list((output / 'status').glob('*.failed.json')))
                summary[component] = f'{count}/11520'
                if failures:
                    raise RuntimeError(f'{component}: failed markers; operator recovery needed')
                complete &= count == 11520
            latest = {}
            for path in (root / 'logs').glob('*/*/completion.json'):
                completion = read(path)
                host = completion['hostname']
                if host not in latest or completion['finished_at_utc'] > latest[host][0]['finished_at_utc']:
                    latest[host] = (completion, path)
            for completion, path in latest.values():
                if not completion['valid']:
                    raise RuntimeError(f'failed pod attempt: {path}; operator recovery needed')
            pod_claims = len(list((root / 'pod_claims').glob('*.claim.json')))
            print(f'[S4 finalizer] {summary}; pod claims {pod_claims}', flush=True)
            if complete and pod_claims == 0:
                seal(root)
                return
            started = read(root / 's4_run.json').get('production_started_at_epoch')
            if started and time.time() - started >= 24 * 3600:
                print('[S4 WARNING] 24-hour target exceeded; writers are NOT killed automatically', flush=True)
            time.sleep(30)
    finally:
        claim.release()


def run(root, teacher):
    manifest = run_manifest(root)
    host = socket.gethostname()
    pod = assignment(root, host)
    if revision(teacher) != TEACHER_COMMIT:
        raise ValueError('wrong teacher revision')
    with coordinator(root):
        if (root / 'SEALED.json').exists():
            raise ValueError('sealed corpus: no writes allowed')
        claim = acquire_exclusive_claim(root / 'pod_claims' / f'{host}.claim.json',
            kind='pod', identity=manifest['code_revision'], worker_index=pod['start'],
            global_worker_count=WORKERS)
        if manifest['production_started_at_epoch'] is None:
            manifest['production_started_at_epoch'] = time.time()
            _atomic_json(root / 's4_run.json', manifest)
    stopped = threading.Event()
    children = set()
    mutex = threading.Lock()
    attempt = uuid.uuid4().hex
    log_root = root / 'logs' / host / attempt
    log_root.mkdir(parents=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1')
    def stop(signum, frame):
        stopped.set()
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def slot(worker):
        for component in COMPONENTS:
            if stopped.is_set():
                raise RuntimeError('stop requested')
            print(f'[S4 {host}] worker {worker}: starting {component}', flush=True)
            command = [sys.executable, '-u', '-m', 'src.giada_runpod.cli', 'worker',
                '--distributed-plan', str(root / component / 'distributed_plan'),
                '--output', str(root / component), '--elm-repo', str(REPO),
                '--teacher-repo', str(teacher), '--worker-index', str(worker),
                '--worker-count', str(WORKERS), '--worker-seed', str(7000001 + worker)]
            log_path = log_root / f'{component}-worker-{worker:05d}.log'
            with log_path.open('xb') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                with mutex:
                    children.add(child)
                terminated_at = None
                while child.poll() is None:
                    if stopped.is_set():
                        if terminated_at is None:
                            child.terminate()
                            terminated_at = time.monotonic()
                        elif time.monotonic() - terminated_at > 10:
                            child.kill()
                    time.sleep(0.5)
                with mutex:
                    children.discard(child)
                if child.returncode:
                    stopped.set()
                    raise RuntimeError(f'{component} worker {worker} exit {child.returncode}; {log_path}')
            print(f'[S4 {host}] worker {worker}: {component} complete', flush=True)
    succeeded = False
    try:
        with ThreadPoolExecutor(max_workers=pod['count']) as pool:
            futures = [pool.submit(slot, worker) for worker in range(pod['start'], pod['start'] + pod['count'])]
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                print(f'[S4 {host}] {len(pending)}/{pod["count"]} worker pipelines pending; logs {log_root}', flush=True)
                for future in finished:
                    try:
                        future.result()
                    except BaseException:
                        stopped.set()
                        raise
        if stopped.is_set():
            raise RuntimeError('supervisor interrupted')
        succeeded = True
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _atomic_json(log_root / 'completion.json', dict(valid=succeeded, finished_at_utc=now(), **pod))
        # Leave the pod claim on failure, just like worker claims: a human must
        # establish that old owners cannot write before a recovery/handoff.
        if succeeded:
            claim.release()
    print(f'[S4 {host}] assigned background and targeted ranges complete', flush=True)


def status(root):
    if not (root / 's4_run.json').is_file():
        raise FileNotFoundError(f'S4 not prepared: {root}')
    manifest = read(root / 's4_run.json')
    fleet = read(root / 'fleet.json')
    covered = {i for p in fleet['pods'] for i in range(p['start'], p['start'] + p['count'])}
    result = dict(root=str(root), registered_workers=len(covered),
                  unassigned_workers=sorted(set(range(WORKERS)) - covered), components={})
    for component in COMPONENTS:
        output = root / component
        _, plan = load_distributed_manifest(output / 'distributed_plan')
        markers = list((output / 'status').glob('*.done.json'))
        by_worker = {}
        for path in markers:
            row = read(path)
            index = int(row['shard_id'].split('-')[-1]) % WORKERS
            by_worker[index] = by_worker.get(index, 0) + 1
        result['components'][component] = dict(
            completed_shards=len(markers), expected_shards=plan['shard_count'],
            failed_markers=len(list((output / 'status').glob('*.failed.json'))),
            worker_claims=len(list((output / 'claims' / 'workers').glob('*.claim.json'))),
            completed_by_pod={p['hostname']: sum(by_worker.get(i, 0) for i in
                range(p['start'], p['start'] + p['count'])) for p in fleet['pods']})
    result['pod_claims'] = len(list((root / 'pod_claims').glob('*.claim.json')))
    started = manifest.get('production_started_at_epoch')
    result['elapsed_hours'] = None if started is None else round((time.time() - started) / 3600, 3)
    result['sealed'] = (root / 'SEALED.json').exists()
    print(json.dumps(result, indent=2), flush=True)


def seal(root):
    from .distributed_audit import add_counts, audit_distributed_component, empty_counts
    from .production_corpus import PRODUCTION_PROFILES

    manifest = run_manifest(root)
    with coordinator(root):
        if any((root / 'pod_claims').glob('*.claim.json')):
            raise RuntimeError('pod claims remain: sealing refuses possible live writers')
        if (root / 'SEALED.json').exists():
            raise FileExistsError('already sealed; do not rewrite a frozen corpus')
        splits = {name: empty_counts() for name in ('train', 'validation')}
        protocols, reports, blockers = {}, {}, []
        aggregate = hashlib.sha256()
        for component in COMPONENTS:
            def progress(done, total):
                if done == 1 or done == total or done % 200 == 0:
                    print(f'[S4 audit {component}] {done}/{total}', flush=True)
            report = audit_distributed_component(root / component,
                root / component / 'distributed_plan', teacher_commit=TEACHER_COMMIT, progress=progress)
            hashes = report.pop('marker_hashes')
            for name, digest in sorted(hashes.items()):
                aggregate.update(f'{digest}  {component}/status/{name}\n'.encode())
            reports[component] = report
            _atomic_json(root / component / 'validation_report.json', report)
            blockers.extend(report['blockers'])
            for split, values in report['splits'].items():
                add_counts(splits[split], values)
            for protocol, by_split in report['protocol_splits'].items():
                for split, values in by_split.items():
                    add_counts(protocols.setdefault(protocol, {}).setdefault(split, empty_counts()), values)
        checks = []
        if reports['background'].get('feature_schema_sha256') != reports['targeted'].get('feature_schema_sha256'):
            blockers.append('background/targeted feature schemas differ')
        s3 = PRODUCTION_PROFILES['s3']
        for split, count in s3['splits'].items():
            observed = splits[split]['transition_count']
            checks.append(dict(gate='split_count', split=split, observed=observed,
                               required=count * 8, passed=observed == count * 8))
        if set(protocols) != set(s3['protocol_splits']):
            blockers.append('unexpected or missing protocol')
        for protocol, by_split in s3['protocol_splits'].items():
            for split, count in by_split.items():
                observed = protocols.get(protocol, {}).get(split, {}).get('transition_count', 0)
                checks.append(dict(gate='protocol_split_count', protocol=protocol, split=split,
                                   observed=observed, required=count * 8, passed=observed == count * 8))
        for (split, metric), (minimum, maximum) in s3['support_ranges'].items():
            observed = splits[split][metric]
            checks.append(dict(gate=metric, split=split, observed=observed,
                minimum=minimum * 8, maximum=maximum * 8,
                passed=minimum * 8 <= observed <= maximum * 8))
        blockers.extend(f'failed gate: {row}' for row in checks if not row['passed'])
        report = dict(schema_version='giada-runpod-s4-production-audit-v1',
            valid=not blockers, blockers=blockers, splits=splits, protocol_splits=protocols,
            support_checks=checks, quantiles_computed=False,
            composition=dict(total_transition_count=230400000,
                long_stochastic_background_fraction=0.6, confirmed_targeted_fraction=0.4,
                train_fraction=0.8, validation_fraction=0.2))
        composite = root / 'composite'
        _atomic_json(composite / 'production_audit.json', report)
        if blockers:
            raise RuntimeError(f'S4 NOT sealed: {len(blockers)} blockers; inspect production_audit.json')
        corpus = dict(schema_version='giada-runpod-composite-corpus-v1', project='GIADA',
            stage='s4_hybrid_production', valid=True, total_transition_count=230400000,
            split_transition_counts=dict(train=184320000, validation=46080000),
            production_audit='production_audit.json', selection_role='development_validation',
            physical_merge_performed=False, paper_test_claimed=False,
            components=[dict(component_id=c, root=f'../{c}', plan='distributed_plan',
                             distributed_plan_identity_sha256=manifest['plans'][c]) for c in COMPONENTS])
        _atomic_json(composite / 'composite_manifest.json', corpus)
        fingerprint = dict(schema_version='giada-runpod-corpus-fingerprint-v1', valid=True,
            marker_fingerprint_sha256=aggregate.hexdigest(),
            shard_count=sum(r['validated_shard_count'] for r in reports.values()),
            total_size_bytes=sum(r['total_size_bytes'] for r in reports.values()),
            physical_mismatch_count=0, physical_mismatch_examples=[])
        _atomic_json(composite / 'corpus_fingerprint.json', fingerprint)
        files = [root / 's4_run.json'] + [root / c / name for c in COMPONENTS
            for name in ('distributed_plan/manifest.json', 'validation_report.json')]
        files += [composite / name for name in ('composite_manifest.json', 'production_audit.json', 'corpus_fingerprint.json')]
        _atomic_json(root / 'SEALED.json', dict(valid=True, sealed_at_utc=now(),
            files={str(p.relative_to(root)): _sha256_file(p) for p in files}))
    print(f'[S4] complete and sealed: {root / "composite"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    item = commands.add_parser('register')
    item.add_argument('--hostname', required=True)
    item.add_argument('--start', type=int, required=True)
    item.add_argument('--count', type=int, required=True)
    item.add_argument('--cpus', type=int, required=True)
    for name in ('start', 'run', 'start-finalizer', 'finalize'):
        item = commands.add_parser(name)
        item.add_argument('--teacher', type=Path, default=Path('/workspace/neuron_as_deep_net'))
    commands.add_parser('status')
    commands.add_parser('seal')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == 'register':
        register(root, args.hostname, args.start, args.count, args.cpus)
    elif args.command == 'start-finalizer':
        start(root, args.teacher.resolve(), finalizer=True)
    elif args.command in ('start', 'run', 'finalize'):
        globals()[args.command](root, args.teacher.resolve())
    else:
        globals()[args.command](root)


if __name__ == '__main__':
    main()
