"""Fixed train-stream 64-batch, ratio-of-medians classification-dose calibration."""
import argparse
import copy
import itertools
import json
import math
import statistics
import time
import traceback
from pathlib import Path

import torch

import checkpoint as full
from common import append, check_binding, config, read, runtime, scientific_hash, sha, source_binding, write

SOURCES = ('G', 'R', 'F', 'E3B', 'GU', 'FU')
BATCHES = 64
TARGET_RATIO = .10
FORMULA = 'lambda_X = 0.10 * median_b ||grad_theta L_native_cls,b||_2 / median_b ||grad_theta (B_b * L_aux,b^X)||_2'


def json_value(value):
    """Preserve every receiver and low-quality diagnostic, without filtering."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def gradients(value, parameters):
    if value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Calibration objective must be a finite scalar')
    values = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    values = tuple(None if g is None else g.detach().float() for g in values)
    if any(not bool(torch.isfinite(g).all()) for g in values if g is not None):
        raise FloatingPointError('Nonfinite calibration gradient: engineering failure')
    return values


def gradient_norm(values):
    terms = [g.square().sum() for g in values if g is not None]
    value = float(torch.stack(terms).sum().sqrt()) if terms else 0.
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite calibration norm: engineering failure')
    return value


def coefficients_from_rows(rows):
    """Do not use median of per-batch ratios, caps, or a common RGB coefficient."""
    if len(rows) != BATCHES:
        raise ValueError('Exactly 64 calibration batches required')
    native = statistics.median(row['native_cls_gradient_l2'] for row in rows)
    auxiliary = {arm: statistics.median([row['batch_aux_gradient_l2'][arm] for row in rows if arm in row['batch_aux_gradient_l2']]) for arm in SOURCES if any(arm in row['batch_aux_gradient_l2'] for row in rows)}
    if not math.isfinite(native) or native <= 0:
        raise FloatingPointError('Zero/nonfinite native classification median: engineering failure')
    if any(not math.isfinite(value) or value <= 0 for value in auxiliary.values()):
        raise FloatingPointError('Zero/nonfinite auxiliary median: engineering failure')
    coefficients = {arm: TARGET_RATIO * native / auxiliary[arm] for arm in auxiliary}
    if any(not math.isfinite(value) or value <= 0 for value in coefficients.values()):
        raise FloatingPointError('Invalid calibrated coefficient: engineering failure')
    return native, auxiliary, coefficients


def close_loaders(trainer):
    for name in ('train_loader', 'test_loader'):
        loader = getattr(trainer, name, None)
        if loader is None:
            continue
        if hasattr(loader, 'close'):
            loader.close()
        else:
            iterator = getattr(loader, 'iterator', None)
            if hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()


def frozen_models(criterion):
    values = []
    for name in ('reference', 'rgb_teacher', 'fusion_teacher', 'teacher', 'rgb_target_teacher', 'fusion_target_teacher'):
        model = getattr(criterion, name, None)
        if isinstance(model, torch.nn.Module) and all(id(model) != id(previous) for previous in values):
            values.append(model)
    for model in getattr(criterion, 'target_models', {}).values():
        if isinstance(model, torch.nn.Module) and all(id(model) != id(previous) for previous in values):
            values.append(model)
    return values


def run(args):
    from train import build
    cfg = config(args.config, 'calibration')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trainer = None
    try:
        write(args.output / 'protocol_config.json', cfg)
        sources = source_binding()
        write(args.output / 'source_binding.json', sources)
        rt = runtime()
        rt.legacy.require_bound_lease_from_environment()
        torch.set_num_threads(4)
        rt, trainer = build(cfg, args.config, args.output, None)
        trainer._setup_train()
        trainer.epoch = 0
        trainer.model.float().train()
        criterion = trainer.criterion_ref
        named = tuple((name, p) for name, p in trainer.model.named_parameters() if p.requires_grad)
        parameters = tuple(p for _, p in named)
        if not parameters:
            raise ValueError('No trainable student parameters')
        state = full.cpu(trainer.model.state_dict())
        initial_rng = full.rng_state()
        optimizer_before = full.digest(trainer.optimizer.state_dict())
        scaler_before = full.digest(trainer.scaler.state_dict())
        scheduler_before = full.digest(trainer.scheduler.state_dict())
        ema_before = full.digest(trainer.ema.ema.state_dict())
        ema_updates = trainer.ema.updates
        init_digest = full.digest(state)
        write(args.output / 'initialization.json', dict(model=cfg['model'], model_sha256=sha(cfg['model']),
              state=init_digest, optimizer_empty=not bool(trainer.optimizer.state), ema_updates=ema_updates))
        torch.cuda.reset_peak_memory_stats()
        rows = []
        try:
            full.restart_epoch_loader(trainer.train_loader, cfg['seed'], 0)
            for batch_index, raw in enumerate(itertools.islice(trainer.train_loader, BATCHES), 1):
                batch = trainer.preprocess_batch(raw)
                batch_rng = full.rng_state()
                trainer.model.load_state_dict(state, strict=True)
                trainer.model.train()
                trainer.model.zero_grad(set_to_none=True)
                try:
                    # FP32 unscaled gradients are the initial mathematical doses;
                    # production retains the inherited AMP/optimizer policy.
                    with torch.autocast(device_type=trainer.device.type, enabled=False):
                        prediction = trainer.model(batch['img'])
                        parts = criterion.components(prediction, batch, all_sources=(cfg["method_variant"] in ("G", "R", "F")), diagnostics=True)
                    native_gradient = gradients(parts['native_cls'], parameters)
                    native_norm = gradient_norm(native_gradient)
                    aux_norms = {}
                    aux_used = {}
                    for arm in parts["batch_aux"]:
                        values = gradients(parts['batch_aux'][arm], parameters)
                        aux_norms[arm] = gradient_norm(values)
                        aux_used[arm] = sum(g is not None for g in values)
                        del values
                    models = frozen_models(criterion)
                    if len(models) not in (1, 2, 3) or any(p.requires_grad or p.grad is not None for model in models for p in model.parameters()):
                        raise AssertionError('Frozen source/reference models must remain gradient-free')
                    b = int(batch['img'].shape[0])
                    if b != cfg['batch']:
                        raise AssertionError('First 64 training batches must retain actual B32')
                    row = dict(batch=batch_index, actual_B=b, finite=True,
                        native_cls_loss=float(parts['native_cls'].detach()),
                        batch_aux_loss={arm: float(parts['batch_aux'][arm].detach()) for arm in parts["batch_aux"]},
                        native_cls_gradient_l2=native_norm, batch_aux_gradient_l2=aux_norms,
                        native_used_parameter_tensors=sum(g is not None for g in native_gradient),
                        auxiliary_used_parameter_tensors=aux_used, files=list(batch['im_file']),
                        input_digests={key: full.digest(batch[key]) for key in ('img', 'strong_img', 'cls', 'bboxes', 'batch_idx')},
                        support=json_value({key: value for key, value in parts['support'].items() if key not in ('foreground_mask', 'target_gt_indices', 'anchor_points', 'stride_tensor')}),
                        support_digests=full.digest(parts['support']), diagnostics=json_value(parts['diagnostics']),
                        common_student_state_restored=True, buffers_restored_after_batch=True,
                        teacher_and_reference_detached=True)
                    append(args.output / 'calibration_batches.jsonl', row)
                    rows.append(row)
                finally:
                    trainer.model.load_state_dict(state, strict=True)
                    trainer.model.zero_grad(set_to_none=True)
                    full.restore_rng(batch_rng)
                del prediction, parts, native_gradient, batch, raw
            native_median, aux_medians, coefficients = coefficients_from_rows(rows)
        finally:
            trainer.model.load_state_dict(state, strict=True)
            trainer.model.zero_grad(set_to_none=True)
            full.restore_rng(initial_rng)
        checks = dict(student_parameters_and_buffers_restored=full.digest(trainer.model.state_dict()) == init_digest,
            rng_restored=full.digest(full.rng_state()) == full.digest(initial_rng),
            optimizer_unchanged=full.digest(trainer.optimizer.state_dict()) == optimizer_before,
            scaler_unchanged=full.digest(trainer.scaler.state_dict()) == scaler_before,
            scheduler_unchanged=full.digest(trainer.scheduler.state_dict()) == scheduler_before,
            ema_unchanged=full.digest(trainer.ema.ema.state_dict()) == ema_before and trainer.ema.updates == ema_updates,
            optimizer_updates_zero=trainer.real_updates == 0 and trainer.update_attempts == 0,
            all_64_batches=len(rows) == BATCHES)
        if not all(checks.values()):
            raise AssertionError('Calibration failed to restore common state: ' + repr(checks))
        check_binding(sources)
        receipt = dict(status='PASS', coefficients=coefficients, formula=FORMULA, target_gradient_ratio=TARGET_RATIO,
            total_batches=BATCHES, native_cls_gradient_median=native_median, batch_aux_gradient_medians=aux_medians,
            achieved_median_gradient_ratios={arm: coefficients[arm] * aux_medians[arm] / native_median for arm in coefficients},
            per_batch_gradient_ratios={arm: [coefficients[arm] * row['batch_aux_gradient_l2'][arm] / row['native_cls_gradient_l2'] if row['native_cls_gradient_l2'] > 0 else None for row in rows] for arm in coefficients},
            nonzero_support_batches={arm: sum(row['batch_aux_gradient_l2'][arm] > 0 for row in rows) for arm in coefficients},
            parameter_names=[name for name, _ in named], parameter_tensors=len(parameters), parameter_count=sum(p.numel() for p in parameters),
            parameter_scope='all_trainable_student_parameters', train_mode=True, forward_precision='float32',
            common_initialization=read(args.output / 'initialization.json'), common_augmented_batch_stream=True,
            loader_policy=cfg['loader_policy'], calibration_epoch=0, calibration_seed=cfg['seed'],
            reset_parameters_and_buffers_each_batch=True, optimizer_updates=0, no_dev_selection=True,
            receiver_filtering=False, checks=checks, config_path=str(args.config), config_sha256=sha(args.config),
            scientific_config_sha256=scientific_hash(cfg), population_sha256=cfg['population_sha256'],
            source_binding_sha256=sha(args.output / 'source_binding.json'), sources=sources,
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20,
            seconds=time.monotonic() - started, official_test_accessed=False,
            interpretation='Separate coefficients match only initial FP32 classification-gradient median dose; training-time doses may diverge.')
        write(args.output / 'calibration_receipt.json', receipt)
        write(args.output / 'completion_receipt.json', receipt)
        print(json.dumps({k: receipt[k] for k in ('status', 'coefficients', 'total_batches', 'native_cls_gradient_median', 'batch_aux_gradient_medians', 'seconds')}, allow_nan=False), flush=True)
    except BaseException as error:
        write(args.output / 'failure_receipt.json', dict(status='ENGINEERING_FAILURE', error=repr(error), traceback=traceback.format_exc(), seconds=time.monotonic() - started, official_test_accessed=False))
        raise
    finally:
        if trainer is not None:
            close_loaders(trainer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
