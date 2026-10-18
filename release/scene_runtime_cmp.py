"""Comparison-arm trainer builder (xm_full_r2); dispatched from scene_runtime.build for cmp_criterion.CMP_ARMS.

Same legacy trainer and paired loader as the PVC arms (with xm_full_r1's HSV + paired mosaic). The legacy
teacher/reference slots stay empty (Identity); the criterion itself loads the registered frozen Fusion11n teacher
after the student, EMA and optimizer exist.
"""
import torch
from common import clone, read, write
from cmp_criterion import CMP_ARMS, CMP_CONTRACT, TEACHER, make_type
from paired_full_aug import OBSERVED as FULL_AUG_OBSERVED


def build_cmp(rt, cfg, path, out, max_updates=None):
    if cfg['method_variant'] not in CMP_ARMS or cfg['input_mode'] != 'rgb':
        raise ValueError('Comparison arms deploy RGB input only')

    def no_legacy_model(weight, expected, names):
        if weight is not None:
            raise ValueError('Legacy teacher/reference slots stay empty for comparison arms')
        return torch.nn.Identity()

    criterion = make_type(rt, cfg)
    builder = clone(rt.legacy.build_trainer, EvidenceCriterion=criterion,
                    DualLabelRGBIRDataset=rt.TrackedDualLabelRGBIRDataset, load_frozen=no_legacy_model)
    trainer = builder(cfg, path, out, 'weight0', max_steps=max_updates)
    setup = trainer._setup_train

    def cmp_setup():
        setup()
        ready = read(out / 'runtime_ready.json')
        ready.update(input_modality='rgb', native_gt_modality='union',
                     teacher_labels_used_by_kd=False, external_adapter_in_optimizer=False,
                     frozen_models_loaded=True, auxiliary_forward=True,
                     frozen_teacher=TEACHER, teacher_input='pixelmean_rgb_ir',
                     kd_kind=CMP_ARMS[cfg['method_variant']]['kd'], cmp_contract=CMP_CONTRACT,
                     observed_full_augmentation=dict(FULL_AUG_OBSERVED),
                     paired_geometry='replay_native_RGB_rng_restore_post_RGB_state',
                     ema_update_policy='legacy_every_optimizer_attempt_including_amp_skip')
        write(out / 'runtime_ready.json', ready, False)

    trainer._setup_train = cmp_setup
    return rt, trainer
