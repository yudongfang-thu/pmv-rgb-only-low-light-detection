"""One supervised RGB view per step, matched to PVC_AUG_NOKD's 1:0.5 views.

Exactly one out of each three consecutive batches uses the *same* photometric
view as PVC_AUG. The other two use clean RGB. An augmented batch restores BN
buffers after its supervised forward and updates them with one no-grad clean
forward, so the deployed model's running statistics always see clean RGB once
per batch. The clean forward has no native loss or parameter gradient.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from pvc_criterion import privileged_images


CONTRACT = dict(version='rgb_single_view_mix_v1', view='pvc_augmented_pixelmean',
                clean_batches_per_cycle=2, augmented_batches_per_cycle=1,
                one_supervised_forward_per_batch=True, bn_running_statistics='clean_rgb_once_per_batch',
                deployed_view='rgb', gt='union')


def augmented_step(seed, step):
    """One reproducible augmented slot per 3-step block, independent of global RNG."""
    if step < 0:
        raise ValueError('Step must be nonnegative')
    block, slot = divmod(int(step), 3)
    digest = hashlib.sha256(f'{int(seed)}:{block}'.encode('ascii')).digest()
    return slot == int.from_bytes(digest[:8], 'big') % 3


def bn_modules(model):
    modules = [m for m in model.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    if not modules:
        raise ValueError('Expected tracked BN modules')
    return modules


def snapshot_bn(model):
    return [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
            for m in bn_modules(model)]


def restore_bn(snapshot):
    # BN backward may retain buffer references. As in the accepted PVC release,
    # .data.copy_ avoids incrementing their version counters before backward.
    for m, mean, var, tracked in snapshot:
        m.running_mean.data.copy_(mean)
        m.running_var.data.copy_(var)
        m.num_batches_tracked.data.copy_(tracked)


def prepare_batch(batch, model, seed, step):
    """Use one full-batch RGB view; never read IR pixel values into the view."""
    clean = batch['img']
    use_aug = augmented_step(seed, step)
    batch['_single_view_step'] = int(step)
    batch['_single_view_augmented'] = use_aug
    if use_aug:
        batch['_single_view_clean_img'] = clean
        batch['_single_view_bn_before'] = snapshot_bn(model)
        batch['img'] = privileged_images(batch, 'augmented')
    return batch


class SingleViewCriterion:
    def __init__(self, native, teacher, reference, config, arm, trainer, sanity):
        if any(model is not None and any(True for _ in model.parameters())
               for model in (teacher, reference)):
            raise ValueError('Single-view control binds no frozen model')
        self.native, self.trainer, self.cfg = native, trainer, config
        self.calls = self.selected_total = self.fallback_batches = 0
        self.clean_batches = self.augmented_batches = 0
        self.bn_checks = []
        self.gradient_checks = []
        self.last_stats = {}
        self.method_variant, self.target_source = config['method_variant'], config['target_source']

    def __deepcopy__(self, memo):
        return None

    def __call__(self, prediction, batch):
        result = self.native(prediction, batch)
        self.calls += 1
        self.selected_total += int(batch['img'].shape[0])
        if batch.get('_single_view_augmented', False):
            self.augmented_batches += 1
            snapshot = batch.pop('_single_view_bn_before')
            clean = batch.pop('_single_view_clean_img')
            restore_bn(snapshot)
            with torch.no_grad():
                self.trainer.model(clean)
            if len(self.bn_checks) < 3:
                ok = all(int(m.num_batches_tracked) == int(tracked) + 1
                         for m, _, _, tracked in snapshot)
                self.bn_checks.append(ok)
                if not ok:
                    raise AssertionError('Clean-only BN update did not occur exactly once')
        else:
            self.clean_batches += 1
        if self.calls <= 9:
            self.last_stats = dict(step=batch.get('_single_view_step'),
                                   augmented=bool(batch.get('_single_view_augmented')),
                                   clean_batches=self.clean_batches,
                                   augmented_batches=self.augmented_batches)
        return result


def build_single(rt, cfg, path, out, max_updates=None):
    from common import clone, read, write
    if cfg['method_variant'] != 'RGBU_MIX' or cfg['input_mode'] != 'rgb':
        raise ValueError('Wrong single-view arm/input')
    if cfg.get('teacher') is not None or cfg.get('reference') is not None:
        raise ValueError('Single-view control binds no frozen model')

    def no_frozen_model(weight, expected, names):
        if weight is not None:
            raise ValueError('Single-view control cannot load a frozen model')
        return torch.nn.Identity()

    builder = clone(rt.legacy.build_trainer, EvidenceCriterion=SingleViewCriterion,
                    DualLabelRGBIRDataset=rt.TrackedDualLabelRGBIRDataset,
                    load_frozen=no_frozen_model)
    trainer = builder(cfg, path, out, 'weight0', max_steps=max_updates)
    paired_preprocess = trainer.preprocess_batch

    def single_preprocess(batch):
        batch = paired_preprocess(batch)
        step = getattr(trainer, '_subset_current_ni', None)
        if step is None:
            raise RuntimeError('Deterministic global batch index unavailable')
        return prepare_batch(batch, trainer.model, cfg['seed'], step)

    trainer.preprocess_batch = single_preprocess
    setup = trainer._setup_train

    def single_setup():
        setup()
        ready = read(out / 'runtime_ready.json')
        ready.update(input_modality='rgb', native_gt_modality='union',
                     teacher_labels_used_by_kd=False, external_adapter_in_optimizer=False,
                     frozen_models_loaded=False, auxiliary_forward=False,
                     single_view_contract=CONTRACT,
                     paired_geometry='replay_native_RGB_rng_restore_post_RGB_state',
                     ema_update_policy='legacy_every_optimizer_attempt_including_amp_skip')
        write(out / 'runtime_ready.json', ready, False)

    trainer._setup_train = single_setup
    return rt, trainer


def verify_canary(criterion, cfg, out):
    checks = dict(variant=criterion.method_variant == 'RGBU_MIX',
                  native_union=criterion.target_source == 'native_union',
                  sufficient_batches=criterion.calls >= 24,
                  exactly_one_view=criterion.clean_batches + criterion.augmented_batches == criterion.calls,
                  both_views_seen=criterion.clean_batches > 0 and criterion.augmented_batches > 0,
                  clean_only_bn=bool(criterion.bn_checks) and all(criterion.bn_checks))
    receipt = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                   contract=CONTRACT, calls=criterion.calls,
                   clean_batches=criterion.clean_batches,
                   augmented_batches=criterion.augmented_batches,
                   science_config_seed=cfg['seed'])
    (Path(out) / 'single_view_engineering_receipt.json').write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')
    if receipt['status'] != 'PASS':
        raise AssertionError('Single-view canary failed: ' + repr(checks))
    return receipt
