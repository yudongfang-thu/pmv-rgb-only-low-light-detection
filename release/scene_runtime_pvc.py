"""PVC trainer builder; dispatched from scene_runtime.build for PVC_ARMS.

Integration (development copy of scene_runtime.build):
    from scene_runtime_pvc import build_pvc
    from pvc_criterion import PVC_ARMS
    ...
    if cfg['method_variant'] in PVC_ARMS:
        return build_pvc(rt, cfg, path, out, max_updates)
"""
import torch
from common import clone, read, write
from pvc_criterion import PVC_ARMS, PVC_CONTRACT, make_type
from paired_full_aug import OBSERVED as FULL_AUG_OBSERVED


def build_pvc(rt, cfg, path, out, max_updates=None):
    if cfg['method_variant'] not in PVC_ARMS or cfg['input_mode'] != 'rgb':
        raise ValueError('PVC arms deploy RGB input only')
    if cfg.get('teacher') is not None or cfg.get('reference') is not None:
        raise ValueError('PVC binds no frozen teacher or reference')

    def no_frozen_model(weight, expected, names):
        if weight is not None:
            raise ValueError('PVC cannot request a frozen model')
        return torch.nn.Identity()

    criterion = make_type(rt, cfg)
    builder = clone(rt.legacy.build_trainer, EvidenceCriterion=criterion,
                    DualLabelRGBIRDataset=rt.TrackedDualLabelRGBIRDataset, load_frozen=no_frozen_model)
    trainer = builder(cfg, path, out, 'weight0', max_steps=max_updates)
    if PVC_ARMS[cfg['method_variant']].get('main') == 'mix':
        # xm_fz2_r1 MIX_PVC: the RGB main input is perturbed in one of each three batches exactly as RGBU_MIX
        # (single_view.prepare_batch, byte-identical module); the criterion fuses the clean RGB with IR.
        from single_view import prepare_batch
        paired_preprocess = trainer.preprocess_batch

        def mix_preprocess(batch):
            batch = paired_preprocess(batch)
            step = getattr(trainer, '_subset_current_ni', None)
            if step is None:
                raise RuntimeError('Deterministic global batch index unavailable')
            return prepare_batch(batch, trainer.model, cfg['seed'], step)

        trainer.preprocess_batch = mix_preprocess
    setup = trainer._setup_train

    def pvc_setup():
        setup()
        ready = read(out / 'runtime_ready.json')
        arm = PVC_ARMS[cfg['method_variant']]
        ready.update(input_modality='rgb', native_gt_modality='union',
                     teacher_labels_used_by_kd=False, external_adapter_in_optimizer=False,
                     frozen_models_loaded=False, auxiliary_forward=True,
                     privileged_view=arm['view'], kd_enabled=arm['kd'], main_view=arm.get('main', 'clean_rgb'),
                     pvc_contract=PVC_CONTRACT, observed_full_augmentation=dict(FULL_AUG_OBSERVED),
                     paired_geometry='replay_native_RGB_rng_restore_post_RGB_state',
                     ema_update_policy='legacy_every_optimizer_attempt_including_amp_skip')
        write(out / 'runtime_ready.json', ready, False)

    trainer._setup_train = pvc_setup
    return rt, trainer
