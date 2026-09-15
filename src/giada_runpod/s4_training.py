"""Frozen S4 GPU run: independent paired seeds, exact checkpoints, no pod rental."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np

from .distributed_generation import acquire_exclusive_claim, load_distributed_manifest, load_worker_partition
from .production_corpus import _atomic_json, _sha256_file
from .s4_production import TEACHER_COMMIT
from .training import FeatureTransform, LeanSomaCorpus, MatchedTrainingConfig, PaperScaleMatchedTrainer

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = Path('/workspace/giada-data/s4-hybrid-production-v1/composite')
DEFAULT_OUTPUT = Path('/workspace/giada-results/s4-matched-v1')
SEEDS = (61017, 61029, 61043, 61071, 61103)
MODELS = ('branch_elm_core', 'giada_voltage_bridge')
TRAIN_ROWS, VALIDATION_ROWS = 184_320_000, 46_080_000
STEPS, BOUNDARY = 1_152_000, 576_000
CHECKPOINTS = (144_000, 288_000, 576_000, 864_000, 1_152_000)
SCHEDULE = dict(boundary=BOUNDARY, first_learning_rate=0.001,
    second_learning_rate=0.0003, reset_adamw=True, continuation_stream_seed_offset=74_000_000,
    diagnostic_samples=262_144, diagnostic_seed=73_100_001,
    save_interval=10_000, progress_interval=500)
SEAL_FILES = {'s4_run.json', 'background/distributed_plan/manifest.json',
    'targeted/distributed_plan/manifest.json', 'background/validation_report.json',
    'targeted/validation_report.json', 'composite/composite_manifest.json',
    'composite/production_audit.json', 'composite/corpus_fingerprint.json'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def revision():
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO).strip():
        raise RuntimeError('tracked source changes: use one clean published revision')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def runtime():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no silent CPU fallback')
    if not torch.__version__.startswith('2.4.') or torch.version.cuda != '12.4':
        raise RuntimeError('use the qualified PyTorch 2.4 / CUDA 12.4 environment')
    if not np.__version__.startswith('1.26.'):
        raise RuntimeError('use NumPy 1.26.x with the qualified PyTorch ABI')
    return dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                python=f'{sys.version_info.major}.{sys.version_info.minor}')


def config_for(hashes):
    result = MatchedTrainingConfig(seeds=SEEDS, scaling_reference_seeds=SEEDS[:3],
        training_steps=STEPS, checkpoints=CHECKPOINTS, batch_size=4096,
        learning_rate=0.001, progress_interval=500,
        evaluation_sample_limit=VALIDATION_ROWS,
        minimum_train_active_transitions=3_173_040,
        minimum_validation_active_transitions=795_264,
        minimum_validation_somatic_upcrossings=292_512,
        required_composite_stage='s4_hybrid_production', minimum_seed_wins=4,
        minimum_family_wins=5, minimum_protocol_wins=12,
        require_spike_transition_advantage=True, expected_corpus_hashes=hashes)
    result.validate()
    return result


def verify_seal(corpus, *, expected_seal=None):
    """Verify the sealed metadata and authenticated partitions, not physical HDF5."""
    corpus = Path(corpus).resolve()
    root = corpus.parent
    seal_path = root / 'SEALED.json'
    seal = read(seal_path)
    if expected_seal and _sha256_file(seal_path) != expected_seal:
        raise RuntimeError('CPU seal identity changed')
    if seal.get('valid') is not True or set(seal.get('files', {})) != SEAL_FILES:
        raise RuntimeError('incomplete or invalid CPU seal')
    for relative, digest in seal['files'].items():
        if _sha256_file(root / relative) != digest:
            raise RuntimeError(f'sealed metadata changed: {relative}')
    if any(root.glob('pod_claims/*.claim.json')):
        raise RuntimeError('CPU pod claims remain')
    manifest = read(corpus / 'composite_manifest.json')
    if (manifest.get('valid') is not True or manifest.get('stage') != 's4_hybrid_production'
        or manifest.get('total_transition_count') != TRAIN_ROWS + VALIDATION_ROWS
        or manifest.get('split_transition_counts') != dict(train=TRAIN_ROWS, validation=VALIDATION_ROWS)):
        raise RuntimeError('wrong S4 composite counts/stage')
    components = manifest.get('components', [])
    if len(components) != 2 or {c['component_id'] for c in components} != {'background', 'targeted'}:
        raise RuntimeError('wrong S4 components')
    hashes = {f'{c}_{suffix}': seal['files'][f'{c}/{file}']
        for c in ('background', 'targeted') for suffix, file in (
            ('plan_sha256', 'distributed_plan/manifest.json'),
            ('validation_sha256', 'validation_report.json'))}
    cpu_run = read(root / 's4_run.json')
    if cpu_run.get('teacher_commit') != TEACHER_COMMIT:
        raise RuntimeError('S4 teacher identity changed')
    for c in components:
        name = c['component_id']
        if c['root'] != f'../{name}' or c.get('plan') != 'distributed_plan':
            raise RuntimeError('unexpected S4 component path')
        plan_root = root / name / 'distributed_plan'
        _, plan = load_distributed_manifest(plan_root)
        if (plan['global_worker_count'] != 128 or plan['shard_count'] != 11520
            or plan['plan_identity_sha256'] != c['distributed_plan_identity_sha256']
            or plan['plan_identity_sha256'] != cpu_run['plans'][name]):
            raise RuntimeError('distributed plan identity mismatch')
        print(f'[S4 GPU prepare] checking {name} partitions', flush=True)
        for worker in range(128):
            load_worker_partition(plan_root, worker, 128)
        if any((root / name / 'claims').rglob('*.claim.json')) or any((root / name / 'status').glob('*.failed.json')):
            raise RuntimeError('CPU component claims/failures remain')
        report = read(root / name / 'validation_report.json')
        expected_count = 138_240_000 if name == 'background' else 92_160_000
        if (report.get('valid') is not True or report.get('blockers')
            or report.get('validated_shard_count') != 11520
            or report.get('validated_transition_count') != expected_count):
            raise RuntimeError('invalid sealed component validation')
    audit = read(corpus / 'production_audit.json')
    if audit.get('valid') is not True or audit.get('blockers'):
        raise RuntimeError('S4 production audit not passed')
    config = config_for({**hashes,
        'composite_manifest_sha256': seal['files']['composite/composite_manifest.json'],
        'production_audit_sha256': seal['files']['composite/production_audit.json'],
        'shard_marker_fingerprint_sha256': read(corpus / 'corpus_fingerprint.json')['marker_fingerprint_sha256']})
    for split, field, minimum in (
        ('train', 'absolute_delta_ge_5mv_count', config.minimum_train_active_transitions),
        ('validation', 'absolute_delta_ge_5mv_count', config.minimum_validation_active_transitions),
        ('validation', 'somatic_upcrossings_minus55mv', config.minimum_validation_somatic_upcrossings)):
        if audit['splits'][split][field] < minimum:
            raise RuntimeError(f'insufficient {split} {field}')
    fingerprint = read(corpus / 'corpus_fingerprint.json')
    if (fingerprint.get('valid') is not True or fingerprint.get('physical_mismatch_count') != 0
        or fingerprint.get('shard_count') != 23040):
        raise RuntimeError('invalid sealed fingerprint')
    return config, audit, _sha256_file(seal_path)


def progress(label):
    def report(done, total):
        if done == 1 or done == total or done % 200 == 0:
            print(f'[S4 GPU {label}] {done}/{total}', flush=True)
    return report


def prepare(output, corpus):
    if output.exists():
        raise FileExistsError(f'output already exists: {output}; inspect, never overwrite')
    code, environment = revision(), runtime()
    config, audit, seal_hash = verify_seal(corpus)
    output.mkdir(parents=True, exist_ok=False)
    try:
        data = LeanSomaCorpus(corpus, progress=progress('index'))
        try:
            if (data.train_count, data.validation_count) != (TRAIN_ROWS, VALIDATION_ROWS):
                raise RuntimeError('physical split counts differ')
            transform = FeatureTransform(data, config)
            print('[S4 GPU prepare] fitting train-only normalization', flush=True)
            transform.fit()
            _atomic_json(output / 'normalization.json', transform.to_dict())
            print('[S4 GPU prepare] frozen global diagnostic sample (not first rows)', flush=True)
            diagnostic = data.sample_raw_global(1, SCHEDULE['diagnostic_samples'],
                np.random.default_rng(SCHEDULE['diagnostic_seed']), include_labels=True,
                progress=progress('diagnostic sampling'))
            for key in diagnostic:
                if diagnostic[key].dtype == object:
                    diagnostic[key] = diagnostic[key].astype(str)
            np.savez(output / 'diagnostic.npz', **diagnostic)
        finally:
            data.close()
        body = dict(schema_version='giada-s4-gpu-run-v1', code_revision=code,
            runtime=environment, corpus=str(Path(corpus).resolve()),
            seal_sha256=seal_hash, configuration=asdict(config), schedule=SCHEDULE,
            files={name: _sha256_file(output / name) for name in ('normalization.json', 'diagnostic.npz')},
            scientific_role='development_validation', fresh_test_claimed=False)
        body['identity'] = identity(body)
        _atomic_json(output / 'run.json', body)
        _atomic_json(output / 'support.json', audit)
    except BaseException:
        print(f'[S4 GPU prepare] incomplete: inspect {output}; no seed launched', flush=True)
        raise
    print(f'[S4 GPU prepare] ready: {output}; no training launched', flush=True)


def start_prepare(output, corpus):
    if output.exists():
        raise FileExistsError(f'output already exists: {output}; inspect first')
    revision()
    runtime()
    logs = output.parent / 'setup'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f's4-prepare-{uuid.uuid4().hex}.log'
    command = [sys.executable, '-u', '-m', 'src.giada_runpod.s4_training',
               '--output', str(output), 'prepare', '--corpus', str(corpus.resolve())]
    launch_process(command, log)


def load_run(output, *, check_runtime=True):
    run = read(output / 'run.json')
    digest = run.pop('identity')
    if identity(run) != digest:
        raise RuntimeError('GPU run identity mismatch')
    run['identity'] = digest
    if run['schema_version'] != 'giada-s4-gpu-run-v1' or run['code_revision'] != revision():
        raise RuntimeError('GPU run requires the same clean source revision')
    config = config_for(run['configuration']['expected_corpus_hashes'])
    if identity(asdict(config)) != identity(run['configuration']) or run['schedule'] != SCHEDULE:
        raise RuntimeError('S4 frozen numeric contract changed')
    if set(run['files']) != {'normalization.json', 'diagnostic.npz'}:
        raise RuntimeError('GPU preparation artifacts incomplete')
    for name, digest in run['files'].items():
        if _sha256_file(output / name) != digest:
            raise RuntimeError(f'GPU preparation artifact changed: {name}')
    if check_runtime and run['runtime'] != runtime():
        raise RuntimeError('GPU software environment differs from preparation')
    return run, config


@contextmanager
def claim(output, name, run_id, index):
    owned = acquire_exclusive_claim(output / 'claims' / f'{name}.claim.json',
        kind='s4-gpu', identity=run_id, worker_index=index, global_worker_count=5)
    # A normal exit/error releases only this owner's claim. SIGKILL/pod loss
    # leaves it in place; no automatic cross-host claim stealing is permitted.
    try:
        yield
    finally:
        owned.release()


def capture_state(torch, models, optimizers, rng, *, step, phase, run_id, seed, rows):
    return dict(schema_version='giada-s4-gpu-state-v1', run_identity=run_id,
        seed=seed, step=step, phase=phase, rows=rows,
        models={k: v.state_dict() for k, v in models.items()},
        optimizers={k: v.state_dict() for k, v in optimizers.items()},
        numpy_rng_state=rng.bit_generator.state, numpy_global_state=np.random.get_state(),
        python_rng_state=random.getstate(), torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_state(torch, state, models, optimizers, rng, *, run_id, seed, boundary=BOUNDARY):
    if (state.get('schema_version') != 'giada-s4-gpu-state-v1'
        or state.get('run_identity') != run_id or state.get('seed') != seed
        or set(state['models']) != set(MODELS) or set(state['optimizers']) != set(MODELS)):
        raise RuntimeError('checkpoint identity/model mismatch')
    if state['phase'] != (1 if state['step'] <= boundary else 2):
        raise RuntimeError('checkpoint phase inconsistent with step')
    for name in MODELS:
        models[name].load_state_dict(state['models'][name])
        optimizers[name].load_state_dict(state['optimizers'][name])
    rng.bit_generator.state = state['numpy_rng_state']
    np.random.set_state(state['numpy_global_state'])
    random.setstate(state['python_rng_state'])
    torch.set_rng_state(state['torch_rng_state'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng_state']])


def save_state(torch, directory, state):
    for name, values in state['models'].items():
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise RuntimeError(f'non-finite {name} checkpoint; previous pointer preserved')
    # Unique immutable file even if an evaluation augments the SAME step.
    # Thus replacing latest.json is the only publication point.
    destination = directory / f'state_step{state["step"]}-{uuid.uuid4().hex}.pt'
    temporary = destination.with_suffix('.pt.tmp')
    with temporary.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    # Pointer published last; interruption leaves the previous pointer usable.
    _atomic_json(directory / 'latest.json', dict(file=destination.name,
        sha256=_sha256_file(destination), step=state['step'], run_identity=state['run_identity']))
    return destination


def load_state(torch, directory, run_id):
    pointer = read(directory / 'latest.json')
    if (pointer.get('run_identity') != run_id
        or not re.fullmatch(rf'state_step{pointer["step"]}-[0-9a-f]{{32}}\.pt', pointer['file'])):
        raise RuntimeError('checkpoint pointer mismatch')
    path = directory / pointer['file']
    if _sha256_file(path) != pointer['sha256']:
        raise RuntimeError('checkpoint SHA-256 mismatch; refusing silent restart')
    # These are our own hash-checked generated checkpoints, not downloaded models.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['step'] != pointer['step']:
        raise RuntimeError('checkpoint step mismatch')
    return state


def optimizers_for(torch, models, config, phase):
    lr = SCHEDULE['first_learning_rate'] if phase == 1 else SCHEDULE['second_learning_rate']
    return {name: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
            for name, model in models.items()}


def paired_update(torch, models, optimizers, x, y, config):
    active = torch.abs(y * config.voltage_scale_mv) >= config.active_delta_threshold_mv
    weight = torch.where(active, config.active_weight, 1.0)
    losses = {}
    for name in MODELS:
        model = models[name]
        model.train()
        optimizers[name].zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.mean(weight * (prediction - y) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizers[name].step()
        losses[name] = loss.detach()
    return losses


def verify_model_restart(trainer, raw, directory):
    """Three tiny synthetic/diagnostic updates, never part of production training."""
    torch = trainer.torch
    torch.manual_seed(90210)
    models = trainer._models()
    opts = optimizers_for(torch, models, trainer.config, 1)
    features, targets = trainer.transform.apply(raw)
    x, y = torch.as_tensor(features, device=trainer.device), torch.as_tensor(targets, device=trainer.device)
    rng = np.random.default_rng(90210)
    paired_update(torch, models, opts, x, y, trainer.config)
    save_state(torch, directory, capture_state(torch, models, opts, rng,
        step=1, phase=1, run_id='operational-smoke', seed=90210, rows=[]))
    recovered = trainer._models()
    recovered_opts = optimizers_for(torch, recovered, trainer.config, 1)
    restore_state(torch, load_state(torch, directory, 'operational-smoke'), recovered,
        recovered_opts, rng, run_id='operational-smoke', seed=90210)
    for group, optimizers in ((models, opts), (recovered, recovered_opts)):
        losses = paired_update(torch, group, optimizers, x, y, trainer.config)
        if not all(math.isfinite(float(v.cpu())) for v in losses.values()):
            raise RuntimeError('non-finite operational smoke loss')
    for name in MODELS:
        for key, value in models[name].state_dict().items():
            if not torch.equal(value, recovered[name].state_dict()[key]):
                raise RuntimeError(f'operational checkpoint mismatch: {name}/{key}')


def smoke(output):
    import h5py
    import torch
    run, config = load_run(output)
    host = socket.gethostname()
    with claim(output, f'device-{host}-cuda0', run['identity'], -1):
        directory = output / 'checks' / f'{host}-{uuid.uuid4().hex}'
        directory.mkdir(parents=True)
        path = Path(run['corpus']).parent / 'background' / 'shards' / 'shard-00000.h5'
        with h5py.File(path, 'r') as handle:
            metadata = json.loads(handle.attrs['schema_metadata_json'])
        trainer = S4Trainer.__new__(S4Trainer)
        trainer.torch, trainer.device, trainer.config = torch, torch.device('cuda:0'), config
        trainer.corpus = SimpleNamespace(metadata=metadata)
        trainer.transform = FeatureTransform(trainer.corpus, config)
        trainer.transform.load_dict(read(output / 'normalization.json'))
        with np.load(output / 'diagnostic.npz', allow_pickle=False) as source:
            raw = {key: source[key][:64] for key in source.files}
        verify_model_restart(trainer, raw, directory)
        _atomic_json(output / 'checks' / f'{host}.json', dict(valid=True,
            run_identity=run['identity'], runtime=runtime(), gpu=torch.cuda.get_device_name(0)))
        print('[S4 GPU check] both real models: finite updates and exact checkpoint restart passed', flush=True)


class S4Trainer(PaperScaleMatchedTrainer):
    """Reuse qualified transforms/models/metrics, with a separate frozen scheduler."""
    def __init__(self, output, run, config):
        import torch
        self.torch, self.device = torch, torch.device('cuda:0')
        self.config, self.run, self.output_dir = config, run, output
        self.code_revision = run['code_revision']
        corpus_root = Path(run['corpus'])
        verify_seal(corpus_root, expected_seal=run['seal_sha256'])
        print('[S4 GPU] physical corpus verification before training', flush=True)
        self.corpus_contract = self._validate_corpus_contract(corpus_root, config,
            fingerprint_progress=progress('physical verification'))
        fingerprint = self.corpus_contract['shard_fingerprint_report']
        if fingerprint['shard_count'] != 23040:
            raise RuntimeError('physical shard set incomplete')
        self.corpus = LeanSomaCorpus(corpus_root, progress=progress('index'))
        if (self.corpus.train_count, self.corpus.validation_count) != (TRAIN_ROWS, VALIDATION_ROWS):
            raise RuntimeError('physical split counts differ')
        self.transform = FeatureTransform(self.corpus, config)
        self.transform.load_dict(read(output / 'normalization.json'))
        with np.load(output / 'diagnostic.npz', allow_pickle=False) as source:
            self.diagnostic = {key: source[key] for key in source.files}

    def evaluate_pair(self, models, *, full):
        # One HDF5 traversal and one feature transformation for BOTH models.
        metric = self._empty_metric_state
        states = {name: dict(overall=metric(), active=metric(), spike=metric(),
            quiescent=metric(), moderate=metric(), component={}, protocol={}, family={}) for name in MODELS}
        if full:
            iterator = self.corpus.iter_raw(1, self.config.evaluation_sample_limit, include_labels=True)
        else:
            iterator = ({key: value[start:start + 65536] for key, value in self.diagnostic.items()}
                        for start in range(0, len(self.diagnostic['voltage_t_mv']), 65536))
        for model in models.values():
            model.eval()
        started, last, count = time.monotonic(), time.monotonic(), 0
        with self.torch.no_grad():
            for raw in iterator:
                features, y = self.transform.apply(raw)
                x = self.torch.as_tensor(features, device=self.device)
                target_mv = y * self.config.voltage_scale_mv
                absolute = np.abs(target_mv)
                masks = dict(overall=np.ones(len(y), dtype=bool),
                    active=absolute >= 5, quiescent=absolute < 1,
                    moderate=(absolute >= 1) & (absolute < 5),
                    spike=(raw['voltage_t_mv'] < -55) & (raw['voltage_t_plus_1_mv'] >= -55))
                label_masks = {kind: {label: raw[key] == label for label in np.unique(raw[key])}
                    for kind, key in (('component', '_component_label'), ('protocol', '_protocol_label'),
                                      ('family', '_family_label'))}
                for name in MODELS:
                    prediction = models[name](x).cpu().numpy() * self.config.voltage_scale_mv
                    error = prediction - target_mv
                    if not np.isfinite(error).all():
                        raise RuntimeError(f'non-finite {name} evaluation')
                    for key, mask in masks.items():
                        self._update_metric_state(states[name][key], error, target_mv, mask)
                    for kind, by_label in label_masks.items():
                        for label, mask in by_label.items():
                            self._update_metric_state(states[name][kind].setdefault(str(label), metric()),
                                                      error, target_mv, mask)
                count += len(y)
                if time.monotonic() - last >= 30:
                    print(f'[S4 GPU evaluation] {count:,} rows; {time.monotonic()-started:.0f}s', flush=True)
                    last = time.monotonic()
        expected = self.config.evaluation_sample_limit if full else len(self.diagnostic['voltage_t_mv'])
        if count != expected:
            raise RuntimeError(f'evaluation coverage {count} != {expected}')
        result = {}
        for name, values in states.items():
            finished = {k: self._finish_metric_state(values[k]) for k in masks}
            result[name] = dict(soma_rmse_mv=finished['overall']['soma_rmse_mv'],
                persistence_soma_rmse_mv=finished['overall']['persistence_soma_rmse_mv'],
                improvement_vs_persistence_fraction=finished['overall']['improvement_vs_persistence_fraction'],
                active_soma_rmse_mv=finished['active']['soma_rmse_mv'],
                active_count=finished['active']['example_count'], example_count=count,
                activity_regime_metrics={label: finished[key] for label, key in (
                    ('quiescent_abs_delta_lt_1mv', 'quiescent'), ('moderate_abs_delta_1_to_5mv', 'moderate'),
                    ('active_abs_delta_ge_5mv', 'active'), ('somatic_upcrossing_minus55mv', 'spike'))},
                **{section: {label: self._finish_metric_state(state) for label, state in values[kind].items()}
                   for kind, section in (('component', 'component_metrics'), ('protocol', 'protocol_metrics'),
                                         ('family', 'protocol_family_metrics'))})
        return result

    def train_seed(self, seed, directory, stop):
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        models = self._models()
        phase, step, rows = 1, 0, []
        rng = np.random.default_rng(seed)
        optimizers = optimizers_for(torch, models, self.config, phase)
        if (directory / 'latest.json').exists():
            state = load_state(torch, directory, self.run['identity'])
            restore_state(torch, state, models, optimizers, rng, run_id=self.run['identity'], seed=seed)
            phase, step, rows = state['phase'], state['step'], state['rows']
            if not 0 <= step <= STEPS:
                raise RuntimeError('checkpoint step outside frozen budget')
            print(f'[S4 GPU seed={seed}] resumed step {step}; phase {phase}', flush=True)
            # The pre-evaluation checkpoint is intentional. If evaluation was
            # interrupted, complete that checkpoint's metrics before advancing.
            if step in CHECKPOINTS and step != STEPS and not any(r['step'] == step for r in rows):
                values = self.evaluate_pair(models, full=False)
                rows.extend(dict(seed=seed, step=step, model=name, evaluation_scope='fixed_global_diagnostic',
                                 **value) for name, value in values.items())
                save_state(torch, directory, capture_state(torch, models, optimizers, rng,
                    step=step, phase=phase, run_id=self.run['identity'], seed=seed, rows=rows))
        initial, started = step, time.monotonic()
        last_progress = started
        evaluation_seconds = 0.0
        for step in range(step + 1, STEPS + 1):
            if step == BOUNDARY + 1:
                phase = 2
                optimizers = optimizers_for(torch, models, self.config, phase)
                rng = np.random.default_rng(seed + SCHEDULE['continuation_stream_seed_offset'])
                print(f'[S4 GPU seed={seed}] phase 2: fresh AdamW and registered RNG stream', flush=True)
            raw = self.corpus.sample_raw(0, self.config.batch_size, rng)
            features, target = self.transform.apply(raw)
            losses = paired_update(torch, models, optimizers,
                torch.as_tensor(features, device=self.device), torch.as_tensor(target, device=self.device), self.config)
            if (step % SCHEDULE['progress_interval'] == 0 or step == initial + 1 or step == STEPS
                or time.monotonic() - last_progress >= 30):
                numbers = {name: float(value.cpu()) for name, value in losses.items()}
                if not all(math.isfinite(value) for value in numbers.values()):
                    raise RuntimeError('non-finite training loss')
                elapsed = time.monotonic() - started
                rate = (step - initial) / max(elapsed - evaluation_seconds, 1e-9)
                remaining = (STEPS - step) / rate / 3600
                message = dict(seed=seed, step=step, total=STEPS, phase=phase,
                    paired_updates_per_second=round(rate, 2), estimated_compute_hours_remaining=round(remaining, 2),
                    evaluation_seconds=round(evaluation_seconds, 1), training_loss=numbers,
                    note='ETA excludes future evaluation; not a validation metric')
                print('[S4 GPU progress] ' + json.dumps(message), flush=True)
                _atomic_json(directory / 'progress.json', message)
                last_progress = time.monotonic()
            save_due = step % SCHEDULE['save_interval'] == 0 or step in CHECKPOINTS or stop.is_set()
            if save_due:
                save_state(torch, directory, capture_state(torch, models, optimizers, rng,
                    step=step, phase=phase, run_id=self.run['identity'], seed=seed, rows=rows))
            if stop.is_set():
                print(f'[S4 GPU seed={seed}] stopped after saving step {step}', flush=True)
                return False
            if step in CHECKPOINTS and step != STEPS:
                before = time.monotonic()
                values = self.evaluate_pair(models, full=False)
                rows.extend(dict(seed=seed, step=step, model=name, evaluation_scope='fixed_global_diagnostic',
                                 **value) for name, value in values.items())
                evaluation_seconds += time.monotonic() - before
                save_state(torch, directory, capture_state(torch, models, optimizers, rng,
                    step=step, phase=phase, run_id=self.run['identity'], seed=seed, rows=rows))
        # A saved final step can resume straight into evaluation, without another update.
        if stop.is_set():
            return False
        print(f'[S4 GPU seed={seed}] full validation: {VALIDATION_ROWS:,} rows, both models', flush=True)
        values = self.evaluate_pair(models, full=True)
        rows = [r for r in rows if r['step'] != STEPS]
        rows.extend(dict(seed=seed, step=STEPS, model=name, evaluation_scope='complete_development_validation',
                         **value) for name, value in values.items())
        final_state = save_state(torch, directory, capture_state(torch, models, optimizers, rng,
            step=STEPS, phase=2, run_id=self.run['identity'], seed=seed, rows=rows))
        if stop.is_set():
            return False
        _atomic_json(directory / 'completed.json', dict(valid=True, seed=seed, run_identity=self.run['identity'],
            configuration=self.run['configuration'], schedule=self.run['schedule'], runs=rows,
            normalization_sha256=self.run['files']['normalization.json'],
            state_file=final_state.name, state_sha256=_sha256_file(final_state),
            finished_at_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
            gpu=self.torch.cuda.get_device_name(0)))
        print(f'[S4 GPU seed={seed}] complete and checkpointed', flush=True)
        return True


def worker(output, seed):
    import threading
    if seed not in SEEDS:
        raise ValueError('unregistered seed')
    run, config = load_run(output)
    checked = read(output / 'checks' / f'{socket.gethostname()}.json')
    if checked.get('valid') is not True or checked.get('run_identity') != run['identity'] or checked.get('runtime') != runtime():
        raise RuntimeError('this pod needs a successful S4 GPU check before training')
    directory = output / 'seeds' / f'seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    with claim(output, f'device-{socket.gethostname()}-cuda0', run['identity'], SEEDS.index(seed)), \
         claim(output, f'seed{seed}', run['identity'], SEEDS.index(seed)):
        if (directory / 'completed.json').exists():
            raise FileExistsError('seed already completed; do not retrain')
        stop = threading.Event()
        previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
        trainer = None
        _atomic_json(directory / 'attempt.json', dict(status='running', hostname=socket.gethostname(), pid=os.getpid()))
        try:
            trainer = S4Trainer(output, run, config)
            completed = not stop.is_set() and trainer.train_seed(seed, directory, stop)
            _atomic_json(directory / 'attempt.json', dict(status='completed' if completed else 'stopped',
                hostname=socket.gethostname(), pid=os.getpid()))
        except BaseException as error:
            _atomic_json(directory / 'attempt.json', dict(status='failed', error=str(error),
                hostname=socket.gethostname(), pid=os.getpid()))
            raise
        finally:
            if trainer is not None:
                trainer.corpus.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def validate_completed(row, seed, run, directory):
    if (row.get('valid') is not True or row.get('seed') != seed or row.get('run_identity') != run['identity']
        or row.get('normalization_sha256') != run['files']['normalization.json']
        or identity(row.get('configuration')) != identity(run['configuration'])
        or row.get('schedule') != run['schedule']):
        raise RuntimeError(f'seed {seed}: completion contract mismatch')
    expected = {(step, name) for step in CHECKPOINTS for name in MODELS}
    actual = [(r['step'], r['model']) for r in row['runs']]
    if len(actual) != len(expected) or set(actual) != expected or any(r['seed'] != seed for r in row['runs']):
        raise RuntimeError(f'seed {seed}: incomplete/duplicate checkpoint metrics')
    for r in row['runs']:
        if r['step'] == STEPS:
            if r.get('evaluation_scope') != 'complete_development_validation' or r['example_count'] != VALIDATION_ROWS:
                raise RuntimeError('final validation coverage mismatch')
            if len(r['protocol_family_metrics']) != 5 or len(r['protocol_metrics']) != 14:
                raise RuntimeError('final protocol/family coverage mismatch')
            for section in ('component_metrics', 'protocol_family_metrics', 'protocol_metrics'):
                for values in r[section].values():
                    if values['example_count'] <= 0 or not math.isfinite(values['soma_rmse_mv']):
                        raise RuntimeError('invalid final stratum metric')
                if sum(v['example_count'] for v in r[section].values()) != VALIDATION_ROWS:
                    raise RuntimeError('final stratum counts do not cover validation')
            for value in (r['active_soma_rmse_mv'], r['activity_regime_metrics']['somatic_upcrossing_minus55mv']['soma_rmse_mv']):
                if value is None or not math.isfinite(value):
                    raise RuntimeError('missing/non-finite final activity metric')
        if not math.isfinite(r['soma_rmse_mv']):
            raise RuntimeError('non-finite completion metric')
    if (not re.fullmatch(rf'state_step{STEPS}-[0-9a-f]{{32}}\.pt', row.get('state_file', ''))
        or _sha256_file(directory / row['state_file']) != row['state_sha256']):
        raise RuntimeError('completed checkpoint hash mismatch')


def finalize(output, *, wait=False):
    run, config = load_run(output, check_runtime=False)
    with claim(output, 'finalizer', run['identity'], -1):
        while True:
            paths = [output / 'seeds' / f'seed{s}' / 'completed.json' for s in SEEDS]
            finished = sum(p.is_file() for p in paths)
            active = list((output / 'claims').glob('seed*.claim.json'))
            if finished == len(SEEDS) and not active:
                break
            for seed, path in zip(SEEDS, paths):
                attempt = path.parent / 'attempt.json'
                if (not path.exists() and attempt.exists()
                    and not (output / 'claims' / f'seed{seed}.claim.json').exists()
                    and read(attempt)['status'] in ('failed', 'stopped')):
                    raise RuntimeError(f'seed {seed} needs attention; inspect attempt.json and its log; restart finalizer after recovery')
            if not wait:
                raise RuntimeError(f'{finished}/5 seeds complete; {len(active)} claims; no aggregate written')
            print(f'[S4 GPU finalizer] {finished}/5 complete; {len(active)} seed claims', flush=True)
            time.sleep(30)
        if (output / 'final_report.json').exists():
            raise FileExistsError('final report already exists; never overwrite')
        rows = []
        for seed, path in zip(SEEDS, paths):
            row = read(path)
            validate_completed(row, seed, run, path.parent)
            rows.extend(row['runs'])
        trainer = PaperScaleMatchedTrainer.__new__(PaperScaleMatchedTrainer)
        trainer.config, trainer.output_dir, trainer.code_revision = config, output / 'aggregate', run['code_revision']
        trainer.output_dir.mkdir(exist_ok=True)
        trainer.device = 'cuda (independent paired seeds)'
        trainer.corpus = SimpleNamespace(train_count=TRAIN_ROWS, validation_count=VALIDATION_ROWS)
        trainer.support_preflight = read(Path(run['corpus']) / 'production_audit.json')
        trainer.corpus_contract = dict(verified=True, seal_sha256=run['seal_sha256'],
            verified_corpus_hashes=config.expected_corpus_hashes)
        verify_seal(Path(run['corpus']), expected_seal=run['seal_sha256'])
        report = trainer._write_report(rows, [])
        report.update(run_identity=run['identity'], schedule=run['schedule'],
            learning_rate_schedule='576k at 1e-3; fresh AdamW/RNG; 576k at 3e-4',
            checkpoint_metric_scope='intermediate: fixed diagnostic sample; final: full development validation',
            fresh_independent_confirmation=False, paper_confirmation_claimed=False,
            seed_report_sha256={str(s): _sha256_file(p) for s, p in zip(SEEDS, paths)})
        report['registered_decision']['s4_development_gates_passed'] = report['registered_decision']['all_registered_gates_passed']
        _atomic_json(output / 'final_report.json', report)
        print('[S4 GPU] complete: ' + str(output / 'final_report.json'), flush=True)


def start(output, seed=None, *, finalizer=False):
    run, _ = load_run(output, check_runtime=not finalizer)
    if not finalizer and seed not in SEEDS:
        raise ValueError('unregistered seed')
    name = 'finalizer' if finalizer else f'seed{seed}'
    if (output / 'claims' / f'{name}.claim.json').exists():
        raise RuntimeError('claim exists; inspect the old owner, do not launch a duplicate')
    if not finalizer and (output / 'seeds' / name / 'completed.json').exists():
        raise FileExistsError('seed already completed')
    if finalizer and (output / 'final_report.json').exists():
        raise FileExistsError('final report already exists')
    if not finalizer:
        smoke(output)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / f'{name}-{socket.gethostname()}-{uuid.uuid4().hex}.log'
    command = [sys.executable, '-u', '-m', 'src.giada_runpod.s4_training', '--output', str(output),
               'finalize' if finalizer else 'run']
    command += ['--wait'] if finalizer else ['--seed', str(seed)]
    launch_process(command, log, seed=seed)


def launch_process(command, log, *, seed=None):
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)
    if process.poll() not in (None, 0):
        raise RuntimeError(f'launch failed; inspect {log}')
    print(json.dumps(dict(pid=process.pid, seed=seed, hostname=socket.gethostname(), log=str(log)), indent=2))


def status(output):
    run = read(output / 'run.json')
    result = dict(run_identity=run['identity'], seeds={}, final_report=(output / 'final_report.json').is_file())
    for seed in SEEDS:
        directory = output / 'seeds' / f'seed{seed}'
        result['seeds'][str(seed)] = dict(completed=(directory / 'completed.json').is_file(),
            attempt=read(directory / 'attempt.json') if (directory / 'attempt.json').exists() else None,
            claim=read(output / 'claims' / f'seed{seed}.claim.json') if (output / 'claims' / f'seed{seed}.claim.json').exists() else None,
            progress=read(directory / 'progress.json') if (directory / 'progress.json').exists() else None)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'start-prepare'):
        item = commands.add_parser(name)
        item.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    for name in ('start', 'run'):
        item = commands.add_parser(name)
        item.add_argument('--seed', type=int, choices=SEEDS, required=True)
    commands.add_parser('start-finalizer')
    item = commands.add_parser('finalize')
    item.add_argument('--wait', action='store_true')
    commands.add_parser('status')
    commands.add_parser('check')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.command == 'prepare':
        prepare(output, args.corpus)
    elif args.command == 'start-prepare':
        start_prepare(output, args.corpus)
    elif args.command == 'check':
        smoke(output)
    elif args.command == 'start':
        start(output, args.seed)
    elif args.command == 'run':
        worker(output, args.seed)
    elif args.command == 'start-finalizer':
        start(output, finalizer=True)
    elif args.command == 'finalize':
        finalize(output, wait=args.wait)
    else:
        status(output)


if __name__ == '__main__':
    main()
