import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.giada_runpod import distributed_audit as audit
from src.giada_runpod import s4_production as s4
from src.giada_runpod.distributed_generation import acquire_exclusive_claim
from src.giada_runpod.planning import ShardPlan, TrajectoryPlan
from src.giada_runpod.store import LeanShardWriter, sha256_file


def specimen(root, index=0):
    trajectories = (
        TrajectoryPlan('a', index * 2, 123 + index, 'train', 2, 'p0'),
        TrajectoryPlan('b', index * 2 + 1, 456 + index, 'validation', 2, 'p1'),
    )
    shard = ShardPlan(f'shard-{index:05d}', index, trajectories, 4, f'plan-{index}')
    path = root / 'shards' / f'{shard.shard_id}.h5'
    marker_path = root / 'status' / f'{shard.shard_id}.done.json'
    metadata = dict(plan_sha256=shard.plan_sha256, teacher_commit=s4.TEACHER_COMMIT,
                    storage_profile='soma_paper', mechanism_group_names=['g0'], ion_names=['i0'],
                    causal_drive_features=[f'u{i}' for i in range(12)], segment_ids=[0],
                    mechanism_presence=[[1]], segment_static=[[0.]], region_names=['soma'],
                    segment_region_ids=[0],
                    trajectory_design={str(t.trajectory_index): {'protocol': t.protocol} for t in trajectories})
    writer = LeanShardWriter(path, segment_count_per_transition=1,
        mechanism_group_count=1, ion_count=1, schema_metadata=metadata)
    for trajectory in trajectories:
        for step, voltage in enumerate((-54, -69)):
            writer.append(dict(segment_id=[0], voltage_t_mv=[-70.],
                voltage_t_plus_1_mv=[voltage], parent_delta_t_mv=[0.],
                mean_child_delta_t_mv=[0.], mechanism_state_t=[[0.1]],
                ion_state_t=[[0.2]], causal_drive=[[0.] * 12],
                trajectory_index=trajectory.trajectory_index, step_index=step,
                seed=trajectory.seed, split_code=int(trajectory.split == 'validation'),
                scheduled_event_count=2, realized_event_count=1), [])
    marker = writer.close(expected_transition_count=4)
    marker.update(shard_id=shard.shard_id, plan_sha256=shard.plan_sha256, trajectory_count=2)
    marker_path.parent.mkdir(exist_ok=True)
    marker_path.write_text(json.dumps(marker), encoding='utf-8')
    return shard, path, marker_path


def test_shard_audit_exact_support_and_marker_identity(tmp_path):
    shard, path, marker = specimen(tmp_path)
    report, splits, protocols = audit.audit_planned_shard(path, marker, shard, s4.TEACHER_COMMIT)
    assert report['valid']
    for split in ('train', 'validation'):
        assert splits[split]['transition_count'] == 2
        assert splits[split]['absolute_delta_ge_5mv_count'] == 1
        assert splits[split]['somatic_upcrossings_minus55mv'] == 1
    assert protocols['p1']['validation']['scheduled_event_count'] == 4


@pytest.mark.parametrize('field,value,message', [
    ('seed', 999, 'seed mismatch'), ('step_index', 999, 'step order'),
    ('split_code', 1, 'split_code mismatch'), ('trajectory_index', 999, 'trajectory_index mismatch'),
    ('parent_delta_t_mv', float('nan'), 'non-finite'),
])
def test_semantic_tamper_rejected_even_with_refreshed_hash(tmp_path, field, value, message):
    shard, path, marker_path = specimen(tmp_path)
    with h5py.File(path, 'r+') as handle:
        handle[field][0] = value
    marker = json.loads(marker_path.read_text())
    marker.update(sha256=sha256_file(path), size_bytes=path.stat().st_size)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match=message):
        audit.audit_planned_shard(path, marker_path, shard)


@pytest.mark.parametrize('key,value', [('plan_sha256', 'wrong'), ('sha256', 'wrong'),
                                      ('trajectory_count', 9), ('size_bytes', 0)])
def test_bad_marker_rejected(tmp_path, key, value):
    shard, path, marker_path = specimen(tmp_path)
    marker = json.loads(marker_path.read_text())
    marker[key] = value
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='marker'):
        audit.audit_planned_shard(path, marker_path, shard)


def test_component_streams_and_checks_unexpected_files(tmp_path, monkeypatch):
    shard, _, _ = specimen(tmp_path)
    monkeypatch.setattr(audit, 'load_distributed_manifest', lambda root: (
        SimpleNamespace(target_transitions=4, shard_count=1),
        {'global_worker_count': 1, 'shard_count': 1, 'plan_identity_sha256': 'x'}))
    monkeypatch.setattr(audit, 'load_worker_partition', lambda *a: (None, [shard], None))
    report = audit.audit_distributed_component(tmp_path, tmp_path)
    assert report['valid'] and report['validated_transition_count'] == 4
    (tmp_path / 'shards' / 'other.h5').touch()
    assert not audit.audit_distributed_component(tmp_path, tmp_path)['valid']


@pytest.mark.parametrize('relative', ['claims/workers/worker-00000.claim.json',
    'status/shard-00000.failed.json', 'shards/shard-00000.h5.partial'])
def test_residue_blocks_sealing(tmp_path, monkeypatch, relative):
    shard, _, _ = specimen(tmp_path)
    monkeypatch.setattr(audit, 'load_distributed_manifest', lambda root: (
        SimpleNamespace(target_transitions=4, shard_count=1),
        {'global_worker_count': 1, 'shard_count': 1, 'plan_identity_sha256': 'x'}))
    monkeypatch.setattr(audit, 'load_worker_partition', lambda *a: (None, [shard], None))
    residue = tmp_path / relative
    residue.parent.mkdir(parents=True, exist_ok=True)
    residue.touch()
    assert not audit.audit_distributed_component(tmp_path, tmp_path)['valid']


@pytest.mark.parametrize('host,start,count,cpus', [
    ('../escape', 0, 8, 8), ('pod', -1, 8, 8), ('pod', 127, 8, 8),
    ('pod', 0, 9, 8), ('pod', 0, 0, 8),
])
def test_invalid_fleet_assignment(host, start, count, cpus):
    with pytest.raises(ValueError):
        s4.validate_assignment([], host, start, count, cpus)


def test_fleet_ranges_cover_namespace_without_overlap():
    pods = []
    for name, start, count, cpus in [('a', 0, 30, 32), ('b', 30, 30, 32),
        ('c', 60, 30, 32), ('d', 90, 30, 32), ('e', 120, 8, 8)]:
        s4.validate_assignment(pods, name, start, count, cpus)
        pods.append(dict(hostname=name, start=start, count=count, declared_cpus=cpus))
    with pytest.raises(ValueError, match='overlap'):
        s4.validate_assignment(pods, 'f', 15, 8, 8)
    with pytest.raises(ValueError, match='already registered'):
        s4.validate_assignment(pods, 'a', 0, 30, 32)
    assert sum(p['count'] for p in pods) == 128


def test_assignment_refuses_wrong_host(tmp_path):
    (tmp_path / 'fleet.json').write_text(json.dumps({'pods': [dict(
        hostname='correct-pod', start=0, count=1, declared_cpus=1)]}))
    with pytest.raises(ValueError, match='not registered'):
        s4.assignment(tmp_path, 'wrong-pod')


def test_status_missing_root_is_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='not prepared'):
        s4.status(tmp_path / 'missing')


def test_coordinator_excludes_second_writer(tmp_path):
    with s4.coordinator(tmp_path):
        with pytest.raises(RuntimeError, match='refusing concurrent write'):
            with s4.coordinator(tmp_path):
                pytest.fail('second writer acquired coordinator')
    assert not (tmp_path / 'coordinator.claim.json').exists()


def test_seal_refuses_pod_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(s4, 'run_manifest', lambda root: {})
    claim = acquire_exclusive_claim(tmp_path / 'pod_claims' / 'pod.claim.json',
        kind='pod', identity='x', worker_index=0, global_worker_count=128)
    with pytest.raises(RuntimeError, match='claims remain'):
        s4.seal(tmp_path)
    assert claim.path.exists()


def test_start_uses_real_module_name_and_detaches(tmp_path, monkeypatch):
    monkeypatch.setattr(s4, 'run_manifest', lambda root: {})
    monkeypatch.setattr(s4, 'assignment', lambda *a: {})
    monkeypatch.setattr(s4, 'revision', lambda root: s4.TEACHER_COMMIT)
    monkeypatch.setattr(s4.time, 'sleep', lambda n: None)
    def spawn(command, **kwargs):
        assert command[3] == 'src.giada_runpod.s4_production'
        assert kwargs['start_new_session']
        assert kwargs['stdin'] == s4.subprocess.DEVNULL
        return SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(s4.subprocess, 'Popen', spawn)
    s4.start(tmp_path, tmp_path)


def test_run_manifest_rejects_changed_code(tmp_path, monkeypatch):
    (tmp_path / 's4_run.json').write_text(json.dumps(dict(
        schema_version='giada-s4-production-run-v1', global_worker_count=128, code_revision='old')))
    monkeypatch.setattr(s4, 'revision', lambda root: 'new')
    with pytest.raises(ValueError, match='revision differs'):
        s4.run_manifest(tmp_path)


def test_exact_count_summary_matches_numpy():
    v0 = np.array([-70., -55., -54., -60.])
    v1 = np.array([-54., -54., -70., -59.])
    actual = audit.summarize(v0, v1, np.ones(4), np.zeros(4))
    assert actual['somatic_upcrossings_minus55mv'] == 1
    assert actual['absolute_delta_ge_5mv_count'] == 2


@pytest.mark.parametrize('exit_code', [0, 2])
def test_supervisor_worker_arguments_and_claim_lifecycle(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(s4, 'run_manifest', lambda root: dict(
        code_revision='test', production_started_at_epoch=1))
    monkeypatch.setattr(s4.socket, 'gethostname', lambda: 'pod')
    monkeypatch.setattr(s4, 'assignment', lambda *a: dict(hostname='pod', start=22, count=1, declared_cpus=8))
    monkeypatch.setattr(s4, 'revision', lambda root: s4.TEACHER_COMMIT)
    calls = []
    def spawn(command, **kwargs):
        assert command[command.index('--worker-index') + 1] == '22'
        assert command[command.index('--worker-seed') + 1] == '7000023'
        assert '--recover-stale-claims' not in command
        assert kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
        calls.append(command)
        # Need a hashable child object for the supervisor's live-process set.
        class Child:
            returncode = exit_code
            def poll(self):
                return self.returncode
        return Child()
    monkeypatch.setattr(s4.subprocess, 'Popen', spawn)
    if exit_code:
        with pytest.raises(RuntimeError, match='exit 2'):
            s4.run(tmp_path, tmp_path)
        assert (tmp_path / 'pod_claims' / 'pod.claim.json').exists()
        assert len(calls) == 1  # Never start targeted after a background failure.
    else:
        s4.run(tmp_path, tmp_path)
        assert len(calls) == 2
        assert not (tmp_path / 'pod_claims' / 'pod.claim.json').exists()


def test_full_s4_plan_streams_exact_protocol_splits(tmp_path, monkeypatch):
    """Build production-sized plans, not any teacher data or GPU experiment."""
    from collections import Counter
    monkeypatch.setattr(s4, 'revision', lambda repo: 'offline-test')
    root = tmp_path / 's4'
    s4.prepare(root)
    expected = {
        'background': {'train': 55296000, 'validation': 13824000},
        'targeted': {'train': 6144000, 'validation': 1536000},
    }
    for component in s4.COMPONENTS:
        plan = root / component / 'distributed_plan'
        config, manifest = s4.load_distributed_manifest(plan)
        assert manifest['shard_count'] == 11520
        assert all(p['shard_count'] == 90 for p in manifest['partitions'])
        counts = Counter()
        seed_sets = {'train': set(), 'validation': set()}
        for index in range(128):
            _, shards, _ = s4.load_worker_partition(plan, index, 128)
            for shard in shards:
                for trajectory in shard.trajectories:
                    counts[trajectory.protocol, trajectory.split] += trajectory.duration_ms
                    seed_sets[trajectory.split].add(trajectory.seed)
        assert not seed_sets['train'].intersection(seed_sets['validation'])
        for protocol in config.input_protocols:
            for split in ('train', 'validation'):
                assert counts[protocol, split] == expected[component][split]
        assert sum(counts.values()) == config.target_transitions
    with pytest.raises(FileExistsError):
        s4.prepare(root)


def test_distributed_composite_reader_preserves_protocol_labels(tmp_path, monkeypatch):
    from src.giada_runpod import distributed_generation as distributed
    from src.giada_runpod.training import LeanSomaCorpus
    root = tmp_path / 'background'
    shard, _, _ = specimen(root)
    (root / 'distributed_plan').mkdir()
    composite = tmp_path / 'composite'
    composite.mkdir()
    (composite / 'composite_manifest.json').write_text(json.dumps(dict(
        schema_version='giada-runpod-composite-corpus-v1', valid=True,
        components=[dict(component_id='background', root='../background', plan='distributed_plan')]
    )))
    monkeypatch.setattr(distributed, 'load_distributed_manifest', lambda root: (None, {'global_worker_count': 1}))
    monkeypatch.setattr(distributed, 'load_worker_partition', lambda *args: (None, [shard], None))
    corpus = LeanSomaCorpus(composite)
    try:
        assert corpus.train_count == corpus.validation_count == 2
        labels = list(corpus.iter_raw(1, 2, include_labels=True))[0]['_protocol_label']
        assert set(labels) == {'p1'}
    finally:
        corpus.close()


def test_seal_success_writes_hash_locked_artifacts(tmp_path, monkeypatch):
    from src.giada_runpod.production_corpus import PRODUCTION_PROFILES
    monkeypatch.setattr(s4, 'run_manifest', lambda root: {'plans': dict(background='a', targeted='b')})
    (tmp_path / 's4_run.json').write_text('{}')
    profile = PRODUCTION_PROFILES['s3']
    def fake_audit(root, plan, **kwargs):
        component = root.name
        root.mkdir(exist_ok=True)
        plan.mkdir()
        (plan / 'manifest.json').write_text('{}')
        protocols = {}
        split_counts = {s: audit.empty_counts() for s in ('train', 'validation')}
        for protocol, counts in profile['protocol_splits'].items():
            is_background = protocol.startswith('neuronio')
            if is_background != (component == 'background'):
                continue
            protocols[protocol] = {}
            for split, n in counts.items():
                row = audit.empty_counts()
                row['transition_count'] = n * 8
                protocols[protocol][split] = row
                audit.add_counts(split_counts[split], row)
        if component == 'targeted':
            for (split, metric), (low, high) in profile['support_ranges'].items():
                split_counts[split][metric] = (low + high) * 4
        return dict(valid=True, blockers=[], validated_shard_count=11520, total_size_bytes=1,
            splits=split_counts, protocol_splits=protocols, marker_hashes={'shard-00000.done.json': 'abc'})
    monkeypatch.setattr(audit, 'audit_distributed_component', fake_audit)
    s4.seal(tmp_path)
    sealed = json.loads((tmp_path / 'SEALED.json').read_text())
    assert sealed['valid']
    for relative, digest in sealed['files'].items():
        assert sha256_file(tmp_path / relative) == digest
    with pytest.raises(FileExistsError, match='already sealed'):
        s4.seal(tmp_path)


def test_failed_audit_never_writes_seal_or_valid_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(s4, 'run_manifest', lambda root: {'plans': {}})
    monkeypatch.setattr(audit, 'audit_distributed_component', lambda *args, **kwargs: dict(
        valid=False, blockers=['corrupt'], splits={s: audit.empty_counts() for s in ('train', 'validation')},
        protocol_splits={}, marker_hashes={}))
    with pytest.raises(RuntimeError, match='NOT sealed'):
        s4.seal(tmp_path)
    assert not (tmp_path / 'SEALED.json').exists()
    assert not (tmp_path / 'composite' / 'composite_manifest.json').exists()
    assert not json.loads((tmp_path / 'composite' / 'production_audit.json').read_text())['valid']
