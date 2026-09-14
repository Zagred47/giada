"""Exact bounded-memory audits of authenticated distributed generation plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .distributed_generation import load_distributed_manifest, load_worker_partition
from .store import validate_lean_shard


def empty_counts():
    return dict(transition_count=0, absolute_delta_ge_1mv_count=0,
                absolute_delta_ge_2mv_count=0, absolute_delta_ge_5mv_count=0,
                somatic_upcrossings_minus55mv=0, scheduled_event_count=0,
                realized_event_count=0)


def add_counts(target, source):
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def summarize(v0, v1, scheduled, realized):
    absolute = np.abs(v1 - v0)
    return dict(transition_count=len(v0),
                absolute_delta_ge_1mv_count=int(np.count_nonzero(absolute >= 1)),
                absolute_delta_ge_2mv_count=int(np.count_nonzero(absolute >= 2)),
                absolute_delta_ge_5mv_count=int(np.count_nonzero(absolute >= 5)),
                somatic_upcrossings_minus55mv=int(np.count_nonzero((v0 < -55) & (v1 >= -55))),
                scheduled_event_count=int(scheduled.sum()),
                realized_event_count=int(realized.sum()))


def audit_planned_shard(path, marker_path, shard, teacher_commit=None):
    """Verify physical identity AND every planned trajectory's row/seed/split order."""
    import h5py

    report = validate_lean_shard(path, expected_transition_count=shard.expected_transition_count)
    if not report['valid']:
        raise ValueError(f"invalid {shard.shard_id}: {report['blockers']}")
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    for key, expected in (("shard_id", shard.shard_id),
                          ("plan_sha256", shard.plan_sha256),
                          ("transition_count", shard.expected_transition_count),
                          ("trajectory_count", len(shard.trajectories)),
                          ("size_bytes", report['size_bytes']), ("sha256", report['sha256'])):
        if marker.get(key) != expected:
            raise ValueError(f"{shard.shard_id}: marker {key} mismatch")
    if report['plan_sha256'] != shard.plan_sha256:
        raise ValueError(f"{shard.shard_id}: embedded plan mismatch")
    splits = {name: empty_counts() for name in ('train', 'validation')}
    protocols = {}
    with h5py.File(path, 'r') as h:
        metadata = json.loads(h.attrs['schema_metadata_json'])
        if teacher_commit and metadata.get('teacher_commit') != teacher_commit:
            raise ValueError(f"{shard.shard_id}: teacher commit mismatch")
        schema_keys = ('storage_profile', 'mechanism_group_names', 'ion_names',
                       'causal_drive_features', 'segment_ids', 'mechanism_presence',
                       'segment_static', 'region_names', 'segment_region_ids')
        schema = {key: metadata[key] for key in schema_keys}
        if schema['storage_profile'] != 'soma_paper' or schema['segment_ids'] != [0]:
            raise ValueError(f'{shard.shard_id}: not canonical soma storage')
        report['feature_schema_sha256'] = hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()).hexdigest()
        # Every transition-aligned dataset must be finite and complete, not
        # just the smaller mandatory subset checked by the legacy validator.
        for name, dataset in h.items():
            if isinstance(dataset, h5py.Dataset):
                if dataset.shape[0] != shard.expected_transition_count:
                    raise ValueError(f"{shard.shard_id}: {name} length mismatch")
                if not np.isfinite(dataset[...]).all():
                    raise ValueError(f"{shard.shard_id}: {name} non-finite")
        event_count = int(h.attrs.get('causal_release_outcome_count', -1))
        required_events = ('transition_row', 'synapse_id', 'segment_id', 'offset_ms',
                           'release_success', 'released_quantity', 'ampa_state_increment',
                           'nmda_state_increment', 'inhibitory_state_increment')
        if marker.get('causal_release_outcome_count') != event_count or event_count < 0:
            raise ValueError(f'{shard.shard_id}: release outcome count mismatch')
        for name in required_events:
            values = h[f'events/{name}'][...]
            if len(values) != event_count or not np.isfinite(values).all():
                raise ValueError(f'{shard.shard_id}: invalid release table {name}')
        arrays = {name: h[name][...] for name in (
            'trajectory_index', 'seed', 'split_code', 'step_index',
            'voltage_t_mv', 'voltage_t_plus_1_mv',
            'scheduled_event_count', 'realized_event_count')}
        offset = 0
        for trajectory in shard.trajectories:
            end = offset + trajectory.duration_ms
            for name, expected in (
                ('trajectory_index', trajectory.trajectory_index),
                ('seed', trajectory.seed),
                ('split_code', int(trajectory.split == 'validation')),
            ):
                if not np.all(arrays[name][offset:end] == expected):
                    raise ValueError(f"{shard.shard_id}: trajectory {name} mismatch")
            if not np.array_equal(arrays['step_index'][offset:end], np.arange(trajectory.duration_ms)):
                raise ValueError(f"{shard.shard_id}: step order mismatch")
            design = metadata.get('trajectory_design', {}).get(str(trajectory.trajectory_index), {})
            if design.get('protocol') != trajectory.protocol:
                raise ValueError(f"{shard.shard_id}: trajectory protocol mismatch")
            counts = summarize(
                arrays['voltage_t_mv'][offset:end, 0].astype(np.float64),
                arrays['voltage_t_plus_1_mv'][offset:end, 0].astype(np.float64),
                arrays['scheduled_event_count'][offset:end], arrays['realized_event_count'][offset:end])
            add_counts(splits[trajectory.split], counts)
            by_split = protocols.setdefault(trajectory.protocol, {})
            add_counts(by_split.setdefault(trajectory.split, empty_counts()), counts)
            offset = end
    return report, splits, protocols


def audit_distributed_component(root, plan_root, *, teacher_commit=None, progress=None):
    """Read each partition once, each HDF5 one at a time; no voltage accumulation."""
    root, plan_root = Path(root), Path(plan_root)
    config, manifest = load_distributed_manifest(plan_root)
    blockers, expected_names, marker_hashes = [], set(), {}
    splits = {name: empty_counts() for name in ('train', 'validation')}
    protocols = {}
    count = size = scanned = 0
    feature_schema = None
    for worker in range(manifest['global_worker_count']):
        _, shards, _ = load_worker_partition(plan_root, worker, manifest['global_worker_count'])
        for shard in shards:
            if shard.shard_id in expected_names:
                raise ValueError('duplicate distributed shard')
            expected_names.add(shard.shard_id)
            path = root / 'shards' / f'{shard.shard_id}.h5'
            marker = root / 'status' / f'{shard.shard_id}.done.json'
            try:
                report, shard_splits, shard_protocols = audit_planned_shard(
                    path, marker, shard, teacher_commit)
                if feature_schema is None:
                    feature_schema = report['feature_schema_sha256']
                if feature_schema != report['feature_schema_sha256']:
                    raise ValueError('feature schema differs across shards')
                count += 1
                size += report['size_bytes']
                marker_hashes[marker.name] = hashlib.sha256(marker.read_bytes()).hexdigest()
                for split, values in shard_splits.items():
                    add_counts(splits[split], values)
                for protocol, by_split in shard_protocols.items():
                    for split, values in by_split.items():
                        add_counts(protocols.setdefault(protocol, {}).setdefault(split, empty_counts()), values)
            except (OSError, ValueError, KeyError, TypeError) as error:
                blockers.append(f'{shard.shard_id}: {error}')
            scanned += 1
            if progress:
                progress(scanned, manifest['shard_count'])
    actual = {p.stem for p in (root / 'shards').glob('*.h5')}
    markers = {p.name.removesuffix('.done.json') for p in (root / 'status').glob('*.done.json')}
    if actual != expected_names or markers != expected_names:
        blockers.append('unexpected/missing physical shard or completion marker')
    if any((root / 'shards').glob('*.partial')):
        blockers.append('partial shards remain')
    if any((root / 'status').glob('*.failed.json')):
        blockers.append('failed markers remain')
    if any((root / 'claims').rglob('*.claim.json')):
        blockers.append('worker/shard claims remain')
    total = sum(row['transition_count'] for row in splits.values())
    if total != config.target_transitions or count != config.shard_count:
        blockers.append('coverage count mismatch')
    return dict(schema_version='giada-runpod-validation-v1', valid=not blockers,
                blockers=blockers, expected_shard_count=config.shard_count,
                validated_shard_count=count, expected_transition_count=config.target_transitions,
                validated_transition_count=total, total_size_bytes=size,
                plan_identity_sha256=manifest['plan_identity_sha256'],
                feature_schema_sha256=feature_schema,
                splits=splits, protocol_splits=protocols, marker_hashes=marker_hashes,
                quantiles_computed=False)
