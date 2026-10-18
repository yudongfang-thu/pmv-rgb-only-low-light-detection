"""Two native endpoints: one paired loader/criterion, RGB or float pixel mean."""
import torch


def _pvc_arm_names():
    """PVC 臂清单的唯一来源是 pvc_criterion.PVC_ARMS。"""
    from pvc_criterion import PVC_ARMS
    return PVC_ARMS

from pathlib import Path
from common import clone, read, sha, scientific_hash, write
from native_recipe import ARMS



def frozen_model(weight, cfg, names, expected_role):
    """绑定一个完整 night seed42 FP32 末轮 EMA; 角色名做同义映射(N<->RGBU, FUSIONN<->FUSIONNU)。"""
    ALIAS = {'N': ('N', 'RGBU'), 'FUSIONN': ('FUSIONN', 'FUSIONNU')}
    path = Path(weight)
    run = path.parent.parent
    receipt, original = read(run / 'completion_receipt.json'), read(run / 'protocol_config.json')
    if (receipt['status'], receipt['last_epoch']) != ('training_completed', 200):
        raise ValueError('Frozen source has no complete E200 endpoint')
    allowed = ALIAS.get(expected_role, (expected_role,))
    if (original['subset'], original['seed']) != ('night', 42) or original['method_variant'] not in allowed:
        raise ValueError('Frozen source scene/role/seed differs: %s not in %s' % (original['method_variant'], allowed))
    if original['population_sha256'] != cfg['population_sha256']:
        raise ValueError('Frozen source population differs')
    if receipt['scientific_config_sha256'] != scientific_hash(original):
        raise ValueError('Frozen source original config receipt differs')
    if str(path) != receipt['checkpoint'] or sha(path) != cfg['teacher_sha256'][str(path)]:
        raise ValueError('Frozen source checkpoint binding differs')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = payload.get('ema')
    if not isinstance(model, torch.nn.Module) or payload.get('model') is not None:
        raise TypeError('Require final EMA-only checkpoint')
    if model.names != names:
        raise ValueError('Frozen source class order differs')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise TypeError('Frozen source endpoint parameters are not FP32')
    model = model.float().eval().requires_grad_(False)
    for part in model.modules():
        part._forward_hooks.clear(); part._forward_pre_hooks.clear(); part._backward_hooks.clear()
    return model

class NativeCriterion:
    def __init__(self, native, teacher, reference, config, arm, trainer, sanity):
        self.native, self.trainer = native, trainer
        self.calls = self.selected_total = self.fallback_batches = 0
        self.gradient_checks, self.last_stats = [], {}
        self.method_variant, self.target_source = config['method_variant'], config['target_source']

    def __deepcopy__(self, memo):
        return None

    def __call__(self, prediction, batch):
        self.calls += 1
        return self.native(prediction, batch)


def _build_direct(rt, cfg, path, out, max_updates):
    """GU/FU: 冻结参照提供 TAL 接收位置, 辅助项目标为 GT 单热或融合教师概率。"""
    from criterion_direct import make_type
    criterion = make_type(rt, cfg)
    loaded = {}
    role = 'FUSIONN' if cfg['method_variant'] == 'FU' else 'N'

    def load(weight, expected, names):
        weight = str(weight)
        roles = {str(cfg['reference']): 'N', str(cfg['teacher']): role}
        if weight not in roles:
            raise ValueError('Unbound frozen source requested')
        if weight not in loaded:
            loaded[weight] = frozen_model(weight, cfg, names, roles[weight])
        return loaded[weight]

    builder = clone(rt.legacy.build_trainer, EvidenceCriterion=criterion,
                    DualLabelRGBIRDataset=rt.TrackedDualLabelRGBIRDataset, load_frozen=load)
    trainer = builder(cfg, path, out, 'weight0', max_steps=max_updates)
    setup = trainer._setup_train

    def direct_setup():
        setup()
        ready = read(out / 'runtime_ready.json')
        ready.update(input_modality='rgb', native_gt_modality='union', teacher_labels_used_by_kd=True,
                     external_adapter_in_optimizer=False, auxiliary_forward=True,
                     receiver_reference='frozen_native_TAL_union_GT',
                     auxiliary_target_source=cfg['target_source'],
                     teacher_forward_batch=cfg.get('teacher_forward_batch', 4),
                     ema_update_policy='legacy_every_optimizer_attempt_including_amp_skip')
        write(out / 'runtime_ready.json', ready, False)

    trainer._setup_train = direct_setup
    return rt, trainer


def build(rt, cfg, path, out, max_updates=None):
    if ARMS.get(cfg['method_variant']) != cfg['input_mode']:
        raise ValueError('Arm/input differs')
    from paired_full_aug import enable
    enable()  # xm_full_r1: HSV + paired Mosaic allowed by the loader contract before any dataset exists
    from cmp_criterion import CMP_ARMS
    if cfg['method_variant'] in CMP_ARMS:
        from scene_runtime_cmp import build_cmp
        return build_cmp(rt, cfg, path, out, max_updates)
    if cfg['method_variant'] in _pvc_arm_names():
        from scene_runtime_pvc import build_pvc
        return build_pvc(rt, cfg, path, out, max_updates)
    if cfg['method_variant'] in ('GU', 'FU'):
        return _build_direct(rt, cfg, path, out, max_updates)

    # Legacy setup calls .to() and optimizer-membership checks on these.
    # Parameterless Identity satisfies both without reading any checkpoint.
    def no_frozen_model(weight, expected, names):
        if weight is not None:
            raise ValueError('Native endpoints cannot request a frozen model')
        return torch.nn.Identity()

    builder = clone(rt.legacy.build_trainer, EvidenceCriterion=NativeCriterion,
        DualLabelRGBIRDataset=rt.TrackedDualLabelRGBIRDataset,
        load_frozen=no_frozen_model)
    trainer = builder(cfg, path, out, 'weight0', max_steps=max_updates)
    paired_preprocess = trainer.preprocess_batch

    def native_preprocess(batch):
        batch = paired_preprocess(batch)
        if cfg['input_mode'] == 'pixelmean':
            # Both inputs have the same replayed augmentation, normalized to
            # float32 /255. Keep the native RGB union cls/bboxes/batch_idx.
            batch['img'] = (batch['img'] + batch['strong_img']) * .5
        elif cfg['input_mode'] == 'ir':
            # 先例: release_v1 scene_runtime `if variant=='IR': batch['img']=batch['strong_img']`
            # 只换像素流; cls/bboxes/batch_idx 仍为 union 标签(与 RGB 臂同一套监督)
            batch['img'] = batch['strong_img']
        return batch

    trainer.preprocess_batch = native_preprocess
    setup = trainer._setup_train

    def native_setup():
        setup()
        ready = read(out / 'runtime_ready.json')
        ready.update(input_modality=cfg['input_mode'], native_gt_modality='union',
            teacher_labels_used_by_kd=False, external_adapter_in_optimizer=False,
            frozen_models_loaded=False, auxiliary_forward=False,
            paired_geometry='replay_native_RGB_rng_restore_post_RGB_state',
            ema_update_policy='legacy_every_optimizer_attempt_including_amp_skip')
        from paired_full_aug import OBSERVED
        ready['observed_full_augmentation'] = dict(OBSERVED)
        write(out / 'runtime_ready.json', ready, False)

    trainer._setup_train = native_setup
    return rt, trainer
