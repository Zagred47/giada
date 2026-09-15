import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from src.giada_runpod import s4_training as gpu
from src.giada_runpod.training import LeanSomaCorpus, MatchedTrainingConfig, PaperScaleMatchedTrainer
from src.giada_runpod.production_corpus import _atomic_json, _sha256_file


def hashes():
    return {k: 'a' * 64 for k in ('background_plan_sha256', 'targeted_plan_sha256',
        'background_validation_sha256', 'targeted_validation_sha256',
        'composite_manifest_sha256', 'production_audit_sha256', 'shard_marker_fingerprint_sha256')}


def test_s4_budget_contract_requires_real_hashes():
    with pytest.raises(ValueError, match='complete frozen'):
        MatchedTrainingConfig(required_composite_stage='s4_hybrid_production').validate()
    c = gpu.config_for(hashes())
    assert c.training_steps == 1_152_000 and c.batch_size == 4096
    assert c.training_steps * c.batch_size / gpu.TRAIN_ROWS == 25.6
    assert c.seeds == gpu.SEEDS and c.minimum_seed_wins == 4
    assert c.evaluation_sample_limit == gpu.VALIDATION_ROWS
    assert gpu.BOUNDARY * 2 == gpu.STEPS
    prereg = gpu.read(gpu.REPO / 'experiments/giada_runpod_paper_scale/s4_matched_training_preregistration.json')
    assert prereg['schedule'] == gpu.SCHEDULE
    assert prereg['paired_updates_per_seed'] == gpu.STEPS
    assert prereg['seeds'] == list(gpu.SEEDS)
    assert prereg['diagnostic_checkpoints'] + [prereg['primary_endpoint']] == list(gpu.CHECKPOINTS)


def test_claim_duplicate_rejected_and_owner_preserved(tmp_path):
    with gpu.claim(tmp_path, 'seed61017', 'run', 0):
        path = tmp_path / 'claims' / 'seed61017.claim.json'
        before = path.read_bytes()
        with pytest.raises(RuntimeError, match='refusing concurrent'):
            with gpu.claim(tmp_path, 'seed61017', 'run', 0):
                pass
        assert path.read_bytes() == before
    assert not path.exists()


def test_claim_released_on_handled_failure(tmp_path):
    with pytest.raises(ValueError):
        with gpu.claim(tmp_path, 'seed61017', 'run', 0):
            raise ValueError('intentional')
    assert not list((tmp_path / 'claims').glob('*.claim.json'))


def seal_fixture(root, monkeypatch):
    corpus = root / 'composite'
    for name in gpu.SEAL_FILES:
        _atomic_json(root / name, {})
    _atomic_json(root / 's4_run.json', {'plans': dict(background='b', targeted='t'), 'teacher_commit': gpu.TEACHER_COMMIT})
    _atomic_json(corpus / 'composite_manifest.json', dict(valid=True, stage='s4_hybrid_production',
        total_transition_count=gpu.TRAIN_ROWS + gpu.VALIDATION_ROWS,
        split_transition_counts=dict(train=gpu.TRAIN_ROWS, validation=gpu.VALIDATION_ROWS),
        components=[dict(component_id=c, root=f'../{c}', plan='distributed_plan',
                         distributed_plan_identity_sha256=c[0]) for c in ('background', 'targeted')]))
    for c, count in [('background', 138240000), ('targeted', 92160000)]:
        _atomic_json(root / c / 'validation_report.json', dict(valid=True, blockers=[],
            validated_shard_count=11520, validated_transition_count=count))
    _atomic_json(corpus / 'production_audit.json', dict(valid=True, blockers=[], splits={
        'train': dict(absolute_delta_ge_5mv_count=3173040),
        'validation': dict(absolute_delta_ge_5mv_count=795264, somatic_upcrossings_minus55mv=292512)}))
    _atomic_json(corpus / 'corpus_fingerprint.json', dict(valid=True, physical_mismatch_count=0,
        shard_count=23040, marker_fingerprint_sha256='a' * 64))
    _atomic_json(root / 'SEALED.json', dict(valid=True, files={p: _sha256_file(root / p) for p in gpu.SEAL_FILES}))
    monkeypatch.setattr(gpu, 'load_distributed_manifest', lambda p: (None, dict(
        global_worker_count=128, shard_count=11520, plan_identity_sha256=p.parent.name[0])))
    monkeypatch.setattr(gpu, 'load_worker_partition', lambda *args: None)
    return corpus


def test_seal_verification_derives_complete_hash_contract(tmp_path, monkeypatch):
    corpus = seal_fixture(tmp_path, monkeypatch)
    c, _, seal = gpu.verify_seal(corpus)
    assert set(c.expected_corpus_hashes) == set(hashes())
    assert len(seal) == 64


@pytest.mark.parametrize('name', sorted(gpu.SEAL_FILES))
def test_sealed_metadata_tampering_rejected(tmp_path, monkeypatch, name):
    corpus = seal_fixture(tmp_path, monkeypatch)
    (tmp_path / name).write_text('{}\n')
    with pytest.raises(RuntimeError, match='sealed metadata'):
        gpu.verify_seal(corpus)


def test_incomplete_seal_rejected(tmp_path, monkeypatch):
    corpus = seal_fixture(tmp_path, monkeypatch)
    seal = gpu.read(tmp_path / 'SEALED.json')
    seal['files'].pop('s4_run.json')
    _atomic_json(tmp_path / 'SEALED.json', seal)
    with pytest.raises(RuntimeError, match='incomplete'):
        gpu.verify_seal(corpus)


def test_live_cpu_owner_rejected(tmp_path, monkeypatch):
    corpus = seal_fixture(tmp_path, monkeypatch)
    _atomic_json(tmp_path / 'pod_claims' / 'host.claim.json', {})
    with pytest.raises(RuntimeError, match='claims remain'):
        gpu.verify_seal(corpus)


def test_bounded_handles_preserve_sampling(tmp_path):
    from tests.test_giada_s4_production import specimen
    for index in range(4):
        specimen(tmp_path, index=index)
    # The synthetic shards have valid row data without production plans.
    a, b = LeanSomaCorpus(tmp_path, max_open_handles=1), LeanSomaCorpus(tmp_path, max_open_handles=16)
    try:
        ra, rb = np.random.default_rng(42), np.random.default_rng(42)
        for _ in range(30):
            x, y = a.sample_raw(0, 7, ra), b.sample_raw(0, 7, rb)
            for key in x:
                np.testing.assert_array_equal(x[key], y[key])
            assert len(a._handles) <= 1
        assert ra.bit_generator.state == rb.bit_generator.state
        updates = []
        x = a.sample_raw_global(1, 30, ra, include_labels=True, progress=lambda *v: updates.append(v))
        y = b.sample_raw_global(1, 30, rb, include_labels=True)
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])
        assert updates[-1] == (4, 4)
        assert ra.bit_generator.state == rb.bit_generator.state
    finally:
        a.close()
        b.close()


def torch_or_skip():
    return pytest.importorskip('torch')


def toy_models(torch):
    return {name: torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(),
            torch.nn.Linear(5, 1), torch.nn.Flatten(0)) for name in gpu.MODELS}


def test_optimizer_rng_roundtrip_and_atomic_pointer(tmp_path):
    torch = torch_or_skip()
    torch.manual_seed(15)
    models = toy_models(torch)
    config = MatchedTrainingConfig()
    optimizers = gpu.optimizers_for(torch, models, config, 1)
    rng = np.random.default_rng(19)
    x, y = torch.tensor(rng.normal(size=(8, 3)), dtype=torch.float32), torch.randn(8)
    gpu.paired_update(torch, models, optimizers, x, y, config)
    snapshot = gpu.capture_state(torch, models, optimizers, rng,
        step=1, phase=1, run_id='run', seed=19, rows=[])
    old = gpu.save_state(torch, tmp_path, snapshot)
    pointer = (tmp_path / 'latest.json').read_bytes()
    bad = copy.deepcopy(snapshot)
    next(iter(bad['models']['giada_voltage_bridge'].values())).fill_(float('nan'))
    with pytest.raises(RuntimeError, match='non-finite'):
        gpu.save_state(torch, tmp_path, bad)
    assert (tmp_path / 'latest.json').read_bytes() == pointer
    new = gpu.save_state(torch, tmp_path, snapshot)
    assert old != new and old.exists() and new.exists()
    expected = copy.deepcopy(snapshot)
    loaded = gpu.load_state(torch, tmp_path, 'run')
    models2 = toy_models(torch)
    optimizers2 = gpu.optimizers_for(torch, models2, config, 1)
    rng2 = np.random.default_rng(999)
    gpu.restore_state(torch, loaded, models2, optimizers2, rng2, run_id='run', seed=19)
    assert rng2.bit_generator.state == expected['numpy_rng_state']
    gpu.paired_update(torch, models, optimizers, x, y, config)
    gpu.paired_update(torch, models2, optimizers2, x, y, config)
    for name in gpu.MODELS:
        for left, right in zip(models[name].parameters(), models2[name].parameters()):
            assert torch.equal(left, right)
    new.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='SHA-256'):
        gpu.load_state(torch, tmp_path, 'run')


@pytest.mark.parametrize('wrong', ['seed', 'phase', 'identity'])
def test_invalid_checkpoint_contract_rejected(wrong):
    torch = torch_or_skip()
    models = toy_models(torch)
    opts = gpu.optimizers_for(torch, models, MatchedTrainingConfig(), 1)
    rng = np.random.default_rng(1)
    state = gpu.capture_state(torch, models, opts, rng, step=1, phase=1, run_id='run', seed=1, rows=[])
    if wrong == 'seed':
        state['seed'] = 2
    elif wrong == 'phase':
        state['phase'] = 2
    else:
        state['run_identity'] = 'other'
    with pytest.raises(RuntimeError, match='checkpoint'):
        gpu.restore_state(torch, state, models, opts, rng, run_id='run', seed=1)


def small_trainer(torch, tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, 'STEPS', 6)
    monkeypatch.setattr(gpu, 'BOUNDARY', 3)
    monkeypatch.setattr(gpu, 'CHECKPOINTS', (2, 3, 6))
    monkeypatch.setattr(gpu, 'SCHEDULE', {**gpu.SCHEDULE, 'save_interval': 1, 'progress_interval': 1})
    t = gpu.S4Trainer.__new__(gpu.S4Trainer)
    t.torch, t.device = torch, torch.device('cpu')
    t.run = dict(identity='run', configuration={}, schedule={}, files={'normalization.json': 'n'})
    t.config = MatchedTrainingConfig(batch_size=8)
    t._models = lambda: toy_models(torch)
    t.corpus = SimpleNamespace(sample_raw=lambda split, size, rng: dict(
        x=rng.normal(size=(size, 3)).astype(np.float32), y=rng.normal(size=size).astype(np.float32)))
    t.transform = SimpleNamespace(apply=lambda raw: (raw['x'], raw['y']))
    t.evaluate_pair = lambda models, full: {n: {'soma_rmse_mv': 1.0} for n in gpu.MODELS}
    # CPU tests do not impersonate GPU runtime; only the scheduler uses this name.
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'CPU test fixture')
    return t


@pytest.mark.parametrize('interrupt_step', [2, 3, 4, 6])
def test_scheduler_interruption_matches_uninterrupted(tmp_path, monkeypatch, interrupt_step):
    torch = torch_or_skip()
    trainer = small_trainer(torch, tmp_path, monkeypatch)
    baseline, resumed = tmp_path / 'baseline', tmp_path / 'resumed'
    baseline.mkdir()
    resumed.mkdir()
    trainer.train_seed(61017, baseline, threading.Event())
    expected = gpu.load_state(torch, baseline, 'run')
    stop = threading.Event()
    original = gpu.paired_update
    calls = []
    def interrupted(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == interrupt_step:
            stop.set()
        return result
    monkeypatch.setattr(gpu, 'paired_update', interrupted)
    assert trainer.train_seed(61017, resumed, stop) is False
    monkeypatch.setattr(gpu, 'paired_update', original)
    # Restore defaults are bound at definition, so explicitly use the toy boundary.
    restore = gpu.restore_state
    monkeypatch.setattr(gpu, 'restore_state', lambda *a, **kw: restore(*a, **kw, boundary=3))
    assert trainer.train_seed(61017, resumed, threading.Event())
    actual = gpu.load_state(torch, resumed, 'run')
    assert actual['phase'] == 2
    assert actual['numpy_rng_state'] == expected['numpy_rng_state']
    assert actual['rows'] == expected['rows']
    for name in gpu.MODELS:
        for key, tensor in expected['models'][name].items():
            assert torch.equal(tensor, actual['models'][name][key])
        for p, state in expected['optimizers'][name]['state'].items():
            for key, value in state.items():
                assert torch.equal(value, actual['optimizers'][name]['state'][p][key])


def test_aggregate_refuses_missing_seed_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, 'load_run', lambda *a, **kw: ({'identity': 'run'}, gpu.config_for(hashes())))
    with pytest.raises(RuntimeError, match='0/5 seeds'):
        gpu.finalize(tmp_path)
    assert not (tmp_path / 'final_report.json').exists()


def test_nonexistent_status_is_not_zero_progress(tmp_path):
    with pytest.raises(FileNotFoundError):
        gpu.status(tmp_path)


def test_finalizer_reports_failed_seed_instead_of_waiting_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, 'load_run', lambda *a, **kw: ({'identity': 'run'}, gpu.config_for(hashes())))
    _atomic_json(tmp_path / 'seeds' / 'seed61017' / 'attempt.json', {'status': 'failed'})
    with pytest.raises(RuntimeError, match='needs attention'):
        gpu.finalize(tmp_path, wait=True)
    assert not (tmp_path / 'final_report.json').exists()


def test_prepare_detached_does_not_launch_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, 'revision', lambda: 'c')
    monkeypatch.setattr(gpu, 'runtime', lambda: {})
    calls = []
    monkeypatch.setattr(gpu, 'launch_process', lambda command, log: calls.append((command, log)))
    gpu.start_prepare(tmp_path / 'new', tmp_path / 'corpus')
    command, log = calls[0]
    assert 'prepare' in command and '--seed' not in command
    assert log.parent.name == 'setup'
    with pytest.raises(FileExistsError):
        gpu.start_prepare(tmp_path, tmp_path / 'corpus')


def real_trainer(torch):
    # Widths and slices checked against the S3 matched-exposure normalization.
    from src.giada_runpod.training import FeatureTransform
    t = gpu.S4Trainer.__new__(gpu.S4Trainer)
    t.torch, t.device, t.config = torch, torch.device('cpu'), MatchedTrainingConfig(evaluation_sample_limit=70)
    rng = np.random.default_rng(8)
    raw = dict(voltage_t_mv=np.full(70, -70, dtype=np.float32),
        voltage_t_plus_1_mv=np.tile([-69.5, -68., -50., -75., -80.], 14).astype(np.float32),
        parent_delta_t_mv=np.zeros(70, np.float32), mean_child_delta_t_mv=np.zeros(70, np.float32),
        mechanism_state_t=rng.uniform(.1, .9, (70, 18)).astype(np.float32),
        ion_state_t=rng.uniform(0, 1, (70, 6)).astype(np.float32),
        causal_drive=rng.uniform(0, 1, (70, 12)).astype(np.float32),
        _component_label=np.array(['background', 'targeted'] * 35),
        _protocol_label=np.array([f'p{i%14}' for i in range(70)]),
        _family_label=np.array([f'f{i%5}' for i in range(70)]))
    meta = dict(mechanism_group_names=list(range(18)), ion_names=list(range(6)),
        causal_drive_features=list(range(12)), mechanism_presence=[[1] * 18],
        segment_static=[[0] * 7], region_names=list(range(11)), segment_region_ids=[0])
    t.corpus = SimpleNamespace(metadata=meta, train_count=70,
        iter_raw=lambda *a, **kw: iter([raw]))
    t.transform = FeatureTransform(t.corpus, t.config)
    t.transform.fit()
    t.diagnostic = raw
    return t, raw


def test_real_models_restart_and_parameter_contract(tmp_path):
    torch = torch_or_skip()
    trainer, raw = real_trainer(torch)
    gpu.verify_model_restart(trainer, raw, tmp_path)


def test_shared_evaluation_matches_qualified_per_model_metrics():
    torch = torch_or_skip()
    trainer, _ = real_trainer(torch)
    models = trainer._models()
    paired = trainer.evaluate_pair(models, full=True)
    for name in gpu.MODELS:
        assert paired[name] == trainer._evaluate(models[name])
    assert trainer.evaluate_pair(models, full=False) == paired


def completion_fixture(directory, seed, run):
    directory.mkdir(parents=True)
    final = directory / f'state_step{gpu.STEPS}-{"a"*32}.pt'
    final.write_bytes(b'generated checkpoint fixture')
    def metric(count, error):
        return dict(soma_rmse_mv=error, persistence_soma_rmse_mv=3.,
            improvement_vs_persistence_fraction=1-error/3, example_count=count)
    rows = []
    for step in gpu.CHECKPOINTS:
        for name in gpu.MODELS:
            error = (1. if name == 'giada_voltage_bridge' else 2.) + seed/1e7
            rows.append(dict(seed=seed, step=step, model=name, soma_rmse_mv=error,
                active_soma_rmse_mv=error, example_count=gpu.VALIDATION_ROWS,
                evaluation_scope='complete_development_validation' if step == gpu.STEPS else 'fixed_global_diagnostic',
                component_metrics={c: metric(gpu.VALIDATION_ROWS//2, error) for c in ('background', 'targeted')},
                protocol_metrics={f'p{i}': metric(gpu.VALIDATION_ROWS//14, error) for i in range(14)},
                protocol_family_metrics={f'f{i}': metric(gpu.VALIDATION_ROWS//5, error) for i in range(5)},
                activity_regime_metrics={'somatic_upcrossing_minus55mv': metric(14, error)}))
    row = dict(valid=True, seed=seed, run_identity=run['identity'], configuration=run['configuration'],
        schedule=run['schedule'], runs=rows, normalization_sha256=run['files']['normalization.json'],
        state_file=final.name, state_sha256=_sha256_file(final))
    _atomic_json(directory / 'completed.json', row)
    return row


def test_five_seed_aggregate_and_contract_rejection(tmp_path, monkeypatch):
    from dataclasses import asdict
    monkeypatch.setattr(gpu, 'VALIDATION_ROWS', 70)
    config = gpu.config_for(hashes())
    corpus = tmp_path / 'corpus'
    _atomic_json(corpus / 'production_audit.json', {'valid': True})
    run = dict(identity='run', configuration=asdict(config), schedule=gpu.SCHEDULE,
        files={'normalization.json': 'n'}, code_revision='c', corpus=str(corpus), seal_sha256='s')
    monkeypatch.setattr(gpu, 'load_run', lambda *a, **kw: (run, config))
    monkeypatch.setattr(gpu, 'verify_seal', lambda *a, **kw: None)
    for seed in gpu.SEEDS:
        directory = tmp_path / 'seeds' / f'seed{seed}'
        row = completion_fixture(directory, seed, run)
        gpu.validate_completed(row, seed, run, directory)
        bad = copy.deepcopy(row)
        bad['run_identity'] = 'other'
        with pytest.raises(RuntimeError, match='contract mismatch'):
            gpu.validate_completed(bad, seed, run, directory)
        bad = copy.deepcopy(row)
        bad['runs'].pop()
        with pytest.raises(RuntimeError, match='incomplete/duplicate'):
            gpu.validate_completed(bad, seed, run, directory)
    gpu.finalize(tmp_path)
    report = gpu.read(tmp_path / 'final_report.json')
    assert report['registered_decision']['s4_development_gates_passed']
    assert report['registered_decision']['observed_seed_wins'] == 5
    assert report['fresh_independent_confirmation'] is False
    with pytest.raises(FileExistsError):
        gpu.finalize(tmp_path)
