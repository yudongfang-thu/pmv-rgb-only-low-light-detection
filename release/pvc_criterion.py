"""Privileged-View Co-training (PVC v1).

One RGB YOLO11n receives, in every training step, a second forward of the same
augmented image as the privileged pixel-mean view (RGB+IR)/2.  The privileged
view gets its own native union-GT loss; its detached sigmoid scores supervise
the RGB view on every anchor/class (Bernoulli KL, T=1).  BatchNorm running
statistics are restored after the privileged forward, so the deployed RGB model
only ever accumulates RGB statistics.  Deployment architecture and input are
identical to RGBU.

Arms:
  PVC       paired privileged view, KD on
  PVC_SHUF  IR rolled by one image inside the batch (wrong pairing), KD on
  PVC_NOKD  paired privileged view, KD weight 0 (KD still logged)
"""
import contextlib
import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

PVC_ARMS = {
    'PVC': dict(view='paired', kd=True),
    'PVC_SHUF': dict(view='shuffled', kd=True),
    'PVC_NOKD': dict(view='paired', kd=False),
    # v1.2 新增消融臂。二者都不改动上面三条的代码路径（见 verify_v12_equivalence 自检）。
    'PVC_AUG': dict(view='augmented', kd=True),
    'PVC_ACTIVE': dict(view='paired', kd=True, kd_mask='native_tal_fg'),
    # v1.2-AUG: PVC_AUG with KD weight 0 (KD still logged), the PVC_NOKD analogue of PVC_AUG.
    'PVC_AUG_NOKD': dict(view='augmented', kd=False),
    # xm_mv_r1: three-view co-training; 'view' is descriptive only, 'views' drives components_mv.
    'MV_AUG_IR': dict(view='augmented+paired', views=('augmented', 'paired'), kd=True),
    'MV_AUG_SHUF': dict(view='augmented+shuffled', views=('augmented', 'shuffled'), kd=True),
    'MV_AUG_AUG': dict(view='augmented+augmented_b', views=('augmented', 'augmented_b'), kd=True),
    # xm_geo_r1: per-view KD types; 'geo' = DFL box-distribution KD at RGB foreground anchors (no class scores).
    'MV_AUG_IR_GEO': dict(view='augmented+paired', views=('augmented', 'paired'), kd_types=('score', 'geo'), kd=True),
    'MV_AUG_IR_BOTH': dict(view='augmented+paired', views=('augmented', 'paired'), kd_types=('score', 'score+geo'), kd=True),
    'MV_AUG_AUG_GEO': dict(view='augmented+augmented_b', views=('augmented', 'augmented_b'), kd_types=('score', 'geo'), kd=True),
    'MV_AUG_RAWIR_GEO': dict(view='augmented+raw_ir', views=('augmented', 'raw_ir'), kd_types=('score', 'geo'), kd=True),
    'MV_AUG_SHUF_GEO': dict(view='augmented+shuffled', views=('augmented', 'shuffled'), kd_types=('score', 'geo'), kd=True),
    # xm_fuse_r1: fusion-view arms (score self-distillation, same recipe as MV_AUG_IR).
    'MV_AUG_IAF': dict(view='augmented+iaf', views=('augmented', 'iaf'), kd=True),
    'MV_AUG_RAWIR': dict(view='augmented+raw_ir', views=('augmented', 'raw_ir'), kd=True),
    # xm_fz2_r1: two-view fusion + perturbed-RGB arms (single-view PVC code path, one privileged view).
    'MIX_PVC': dict(view='paired_clean', kd=True, main='mix'),
    'PVC_AUGPAIR': dict(view='augpaired', kd=True),
    # xm_full_r1 (94): PVC and PVC_AUG under the ultralytics default augmentation (RandomHSV + Mosaic).
    'PVC_FULL': dict(view='paired', kd=True),
    'AUG_FULL': dict(view='augmented', kd=True),
    # xm_full_r2: IR evidence under the default augmentation (single-view path unless 'views').
    'RAWIR_FULL': dict(view='raw_ir', kd=True),
    'SHUF_FULL': dict(view='shuffled', kd=True),
    'IAF_FULL': dict(view='iaf', kd=True),
    'MV_FULL': dict(view='augmented+paired', views=('augmented', 'paired'), kd=True),
    # xm_hsv2_r1: the same arms under the HSV-on / Mosaic-off recipe.
    'PVC_H': dict(view='paired', kd=True),
    'AUG_H': dict(view='augmented', kd=True),
    'MV_H': dict(view='augmented+paired', views=('augmented', 'paired'), kd=True),
}
MV_SECOND_JITTER_SALT = 1000003
GEO_WEIGHT = 1.0
GEO_TEMPERATURE = 1.0
GEO_DFL_SCALE = 1.5
GEO_REG_MAX = 16
IAF_KERNEL = 31
IAF_TAU = 0.25
PRIVILEGED_NATIVE_WEIGHT = 0.5
KD_WEIGHT = 0.5
ACTIVE_PROBABILITY = 0.05

PVC_CONTRACT = dict(
    version='pvc_v1.2_xm_hsv2_r1',
    deployed_view='rgb', deployed_architecture='yolo11n_unchanged',
    privileged_view='pixelmean_(img+strong_img)*0.5_same_replayed_geometry',
    shuffled_view='strong_img.roll(1, dim=0)_wrong_pair_within_batch',
    bn_running_statistics='rgb_view_only_restored_after_privileged_forward',
    privileged_native='native_union_TAL_loss_on_privileged_view',
    privileged_native_weight=PRIVILEGED_NATIVE_WEIGHT,
    kd='dense_all_anchors_all_5_classes_bernoulli_KL_T1',
    kd_target='stopgrad_sigmoid_of_same_network_privileged_view',
    kd_normalizer='max(sum(privileged_probabilities), 1)',
    kd_weight=KD_WEIGHT,
    total='(native_rgb_total + w_priv*native_priv_total + w_kd*batch_size*kd) / (1 + w_priv)',
    loss_normalizer='1 + w_priv',
    calibration=None, external_teacher=None, reference=None,
    ablation_arms=dict(
        PVC_AUG='deterministic_photometric_jitter_of_same_rgb_same_geometry_no_extra_information',
        PVC_ACTIVE='dense_KD_restricted_to_native_TAL_foreground_anchors',
        PVC_AUG_NOKD='PVC_AUG_view_and_privileged_native_loss_KD_weight_0',
        MV_AUG_IR='clean_rgb_plus_augmented_view_plus_paired_pixelmean_view',
        MV_AUG_SHUF='clean_rgb_plus_augmented_view_plus_shuffled_ir_pixelmean_view',
        MV_AUG_AUG='clean_rgb_plus_two_independent_augmented_views',
        MV_AUG_IR_GEO='clean_rgb_plus_augmented_view_plus_paired_pixelmean_view_geometry_kd_only',
        MV_AUG_IR_BOTH='clean_rgb_plus_augmented_view_plus_paired_pixelmean_view_score_and_geometry_kd',
        MV_AUG_AUG_GEO='clean_rgb_plus_two_independent_augmented_views_geometry_kd_from_second',
        MV_AUG_RAWIR_GEO='clean_rgb_plus_augmented_view_plus_raw_ir_view_geometry_kd_only',
        MV_AUG_SHUF_GEO='clean_rgb_plus_augmented_view_plus_shuffled_ir_pixelmean_view_geometry_kd_only',
        MV_AUG_IAF='clean_rgb_plus_augmented_view_plus_illumination_aware_fused_view',
        MV_AUG_RAWIR='clean_rgb_plus_augmented_view_plus_raw_ir_view',
        MIX_PVC='rgb_main_augmented_in_one_of_three_batches_as_RGBU_MIX_plus_clean_rgb_ir_pixelmean_view',
        PVC_AUGPAIR='clean_rgb_plus_pixelmean_view_of_jittered_rgb_and_ir',
        PVC_FULL='PVC_under_ultralytics_default_hsv_and_mosaic',
        AUG_FULL='PVC_AUG_under_ultralytics_default_hsv_and_mosaic',
        RAWIR_FULL='clean_rgb_plus_raw_ir_view_under_default_augmentation',
        SHUF_FULL='clean_rgb_plus_ir_shuffled_pixelmean_view_under_default_augmentation',
        IAF_FULL='clean_rgb_plus_illumination_aware_fused_view_under_default_augmentation',
        MV_FULL='MV_AUG_IR_under_default_augmentation',
        PVC_H='PVC_FULL_under_hsv_on_mosaic_off',
        AUG_H='AUG_FULL_under_hsv_on_mosaic_off',
        MV_H='MV_FULL_under_hsv_on_mosaic_off'),
    multi_view_total='(native_rgb_total + w_priv*sum_v native_v_total + w_kd*batch_size*sum_v kd_v) / (1 + n_views*w_priv)',
    multi_view_kd='per view dense Bernoulli KL(p_v || p_rgb) / max(sum p_v, 1); every view detached; every privileged forward BN-restored',
    second_jitter='photometric_jitter seed salt 1000003: an independent draw of the same jitter distribution',
    box_or_dfl_kd='xm_geo_r1 geo arms only (geometry_kd)', feature_kd=False, region_or_difficulty_gate=False,
    geometry_kd='per geo view: KL(softmax(priv_dfl/T) || softmax(rgb_dfl/T)) * T^2, mean over 4 box sides and over the RGB view native TAL foreground anchors; privileged detached; class scores untouched',
    geometry_weight=GEO_WEIGHT, geometry_temperature=GEO_TEMPERATURE, geometry_scale='batch_size * 1.5 (ultralytics DFL gain)',
    geometry_total='(native_rgb + w_priv*sum_v native_v + w_kd*b*sum_score_views kd_v + w_geo*b*1.5*sum_geo_views geo_v) / (1 + n_views*w_priv)',
    raw_ir_view='strong_img itself, not averaged with RGB',
    iaf_view='w*rgb + (1-w)*ir, w = clamp(box31(max_c rgb) / 0.25, 0, 1): Retinex illumination estimate decides RGB vs IR per pixel',
    augpaired_view='(photometric_jitter(rgb) + ir) * 0.5 = paired view + the PVC_AUG perturbation (jitter(rgb) - rgb) / 2',
    paired_clean_view='(clean rgb + ir) * 0.5; on a perturbed main step the clean rgb is the pre-perturbation image',
    mix_main='single_view.prepare_batch (byte-identical to release_xm_single_r1): main input is the PVC_AUG augmented view in one of each three batches (sha256 slot schedule); BN restored after it and updated by one no-grad clean forward',

    native_retained=True)


def loss_item_values(items):
    """ultralytics 8.4 returns {'box_loss': t, 'cls_loss': t, 'dfl_loss': t}; older versions a tensor."""
    if isinstance(items, dict):
        return {str(k): float(v) for k, v in items.items()}
    return [float(x) for x in items]


@contextlib.contextmanager
def engineering_numerics():
    """Deterministic, non-TF32 kernels for the algebraic engineering checks only.

    ultralytics' trainer calls unset_deterministic() when train() returns, and
    cudnn TF32 is on by default on Ampere, so checks run after a canary would
    otherwise compare non-reproducible backward passes.  Everything is restored.
    """
    saved = dict(det=torch.are_deterministic_algorithms_enabled(),
                 warn=torch.is_deterministic_algorithms_warn_only_enabled(),
                 cudnn_det=torch.backends.cudnn.deterministic, bench=torch.backends.cudnn.benchmark,
                 cudnn_tf32=torch.backends.cudnn.allow_tf32, mm_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield dict(deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
                   tf32=False, restored_after=saved)
    finally:
        torch.use_deterministic_algorithms(saved['det'], warn_only=saved['warn'])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved['cudnn_det'], saved['bench']
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved['cudnn_tf32'], saved['mm_tf32']
        if saved['cublas'] is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = saved['cublas']


def raw_prediction(prediction):
    prediction = prediction[1] if isinstance(prediction, tuple) else prediction
    if not isinstance(prediction, dict) or not {'scores', 'boxes', 'feats'} <= set(prediction):
        raise TypeError('Bound YOLO11 one-to-many raw dictionary required')
    return prediction


def _bn_layers(model):
    layers = [m for m in model.modules()
              if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    if not layers:
        raise ValueError('Model has no tracked BatchNorm statistics')
    return layers


def bn_state_digest(model):
    h = hashlib.sha256()
    for m in _bn_layers(model):
        for t in (m.running_mean, m.running_var, m.num_batches_tracked):
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def preserved_bn_running_stats(model):
    """Train-mode forward uses batch statistics; buffers are put back bitwise."""
    saved = [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
             for m in _bn_layers(model)]
    try:
        yield len(saved)
    finally:
        # BN autograd nodes save the running buffers; writing through .data keeps
        # their version counters, otherwise backward of both views raises.
        for m, mean, var, tracked in saved:
            m.running_mean.data.copy_(mean); m.running_var.data.copy_(var)
            m.num_batches_tracked.data.copy_(tracked)


def photometric_jitter(rgb, salt=0):
    """E1 用的第二视图：同一张 RGB 的确定性光度抖动。

    几何逐位不变，因此锚点对应不变；不含任何 IR 或其它新信息。
    抖动由图像内容派生种子 → 同一张图在任何 batch 位置得到同一抖动（可复现）。
    """
    b = rgb.shape[0]
    seed = (int(rgb.detach().float().sum().item()) + salt) % (2 ** 31 - 1)
    g = torch.Generator(device=rgb.device).manual_seed(seed)
    bright = (torch.rand(b, 1, 1, 1, device=rgb.device, generator=g) - .5) * .4
    contrast = 1. + (torch.rand(b, 1, 1, 1, device=rgb.device, generator=g) - .5) * .4
    noise = torch.randn(rgb.shape, device=rgb.device, generator=g) * .02
    mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
    out = (rgb - mean) * contrast + mean + bright + noise
    return out.clamp(0., 1.)


def privileged_images(batch, view):
    rgb, ir = batch['img'], batch['strong_img']
    if rgb.shape != ir.shape or rgb.dtype != ir.dtype:
        raise ValueError('Paired RGB/IR tensors differ in shape or dtype')
    if view == 'paired':
        other = ir
    elif view == 'shuffled':
        if rgb.shape[0] < 2:
            raise ValueError('Shuffled privileged view needs at least two images')
        other = ir.roll(1, 0)
    elif view == 'augmented':
        other = photometric_jitter(rgb)
    elif view == 'augmented_b':
        other = photometric_jitter(rgb, salt=MV_SECOND_JITTER_SALT)
    elif view == 'raw_ir':
        return ir
    elif view == 'iaf':
        return illumination_aware_fusion(rgb, ir)
    elif view == 'augpaired':
        return (photometric_jitter(rgb) + ir) * .5
    elif view == 'paired_clean':
        return (batch.get('_single_view_clean_img', rgb) + ir) * .5
    else:
        raise ValueError('Unknown privileged view: ' + str(view))
    # Same expression as the FUSIONNU native arm.
    return (rgb + other) * .5


def dense_privileged_kd(student_logits, privileged_logits, anchor_mask=None):
    """Sum of per-anchor/per-class Bernoulli KL(p_priv || p_rgb) / max(sum p_priv, 1).

    Gradient w.r.t. student logits is (sigmoid(z_rgb) - p_priv) / norm; the
    privileged side is detached, so no gradient reaches it through this term.
    """
    if student_logits.shape != privileged_logits.shape or student_logits.ndim != 3:
        raise ValueError('Expected identical BxCxA raw score tensors')
    # Same BxAxC contiguous layout as the native loss: the two score gradients then
    # accumulate with identical strides, so downstream head-bias reductions keep
    # their order and the zero-weight check stays bitwise on CUDA.
    student = student_logits.permute(0, 2, 1).contiguous()
    teacher_logits = privileged_logits.detach().permute(0, 2, 1).contiguous().float()
    p = teacher_logits.sigmoid()
    # 归一化始终按**全部锚点**计算：掩码只决定哪些锚点贡献分子, 不改变每锚点的权重尺度。
    # 这样 PVC vs PVC_ACTIVE 的唯一差异是"哪些锚点参与 KD", 是干净的密度消融。
    norm = p.sum().clamp_min(1.)
    cross = F.binary_cross_entropy_with_logits(student.float(), p, reduction='none')
    entropy = F.binary_cross_entropy_with_logits(teacher_logits, p, reduction='none')
    if anchor_mask is not None:
        if anchor_mask.shape != student_logits.shape[:1] + student_logits.shape[2:]:
            raise ValueError('Anchor mask must be BxA matching the score grid')
        m = anchor_mask.to(cross.dtype).unsqueeze(-1)      # BxA -> BxAx1
        cross = cross * m
        entropy = entropy * m
    return (cross - entropy).sum() / norm, norm


def illumination_aware_fusion(rgb, ir, kernel=IAF_KERNEL, tau=IAF_TAU):
    """xm_fuse_r1: Retinex-style illumination-aware RGB-IR fusion (training-free, per pixel).

    L = box_filter(max_c rgb) is the local illumination estimate; w = clamp(L / tau, 0, 1) is the RGB weight.
    Lit regions keep RGB, dark regions take IR; the letterbox padding (grey 114/255 in both) stays unchanged.
    """
    if rgb.shape != ir.shape or rgb.ndim != 4:
        raise ValueError('Paired BxCxHxW tensors required')
    light = rgb.max(dim=1, keepdim=True).values
    light = F.avg_pool2d(light, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    w = (light / tau).clamp(0., 1.)
    return w * rgb + (1. - w) * ir


def dense_geometry_kd(student_boxes, privileged_boxes, fg_mask, reg_max=GEO_REG_MAX, temperature=GEO_TEMPERATURE):
    """xm_geo_r1: KL(p_priv || p_rgb) of the DFL box-side distributions at native TAL foreground anchors.

    student/privileged boxes are raw Bx(4*reg_max)xA head outputs (ultralytics 8.4: channel = side * reg_max + bin).
    Mean over the 4 sides, mean over foreground anchors (an exact, graph-connected 0 when there are none).
    The privileged side is detached; no class score enters.
    """
    if student_boxes.shape != privileged_boxes.shape or student_boxes.ndim != 3:
        raise ValueError('Expected identical Bx(4*reg_max)xA raw box tensors')
    b, c, a = student_boxes.shape
    if c != 4 * reg_max:
        raise ValueError('Box channels must equal 4 * reg_max')
    if fg_mask is None or tuple(fg_mask.shape) != (b, a) or fg_mask.dtype != torch.bool:
        raise ValueError('Foreground mask must be a boolean BxA tensor')
    n = int(fg_mask.sum())
    if n == 0:
        return student_boxes.float().sum() * 0., 0
    s = student_boxes.permute(0, 2, 1).contiguous()[fg_mask].view(-1, 4, reg_max).float() / temperature
    t = privileged_boxes.detach().permute(0, 2, 1).contiguous()[fg_mask].view(-1, 4, reg_max).float() / temperature
    kl = (t.softmax(-1) * (t.log_softmax(-1) - s.log_softmax(-1))).sum(-1)
    return kl.mean(-1).sum() / n * temperature ** 2, n


ENGINEERING_BATCH_IMAGES = 16


def engineering_subbatch(batch, k=ENGINEERING_BATCH_IMAGES):
    """xm_geo_r1: the canary's algebraic engineering checks run on the first k images of one real training batch, so
    the canary's peak memory stays inside the project's per-GPU share (the full 3-view check peaked at 22.5G).
    Per-image tensors/lists keep [:k]; per-object cls/bboxes/batch_idx keep objects of those images; nested
    dicts (teacher_batch) are cut by their own batch_idx."""
    b = batch['img'].shape[0]
    if b <= k:
        return batch

    def cut(d):
        idx = d.get('batch_idx')
        keep = None if idx is None else idx < k
        out = {}
        for key, v in d.items():
            if isinstance(v, dict):
                out[key] = cut(v)
            elif key in ('batch_idx', 'cls', 'bboxes'):
                if keep is None or not torch.is_tensor(v) or v.shape[0] != keep.shape[0]:
                    raise ValueError('Per-object field without a matching batch_idx: ' + key)
                out[key] = v[keep]
            elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == b:
                out[key] = v[:k]
            elif isinstance(v, (list, tuple)) and len(v) == b:
                out[key] = type(v)(v[:k])
            else:
                out[key] = v
        return out

    return cut(batch)


def classification_head_parameter_ids(model):
    """Parameters of the classification branch only (ultralytics Detect.cv3; the CPU toy detector's .cls)."""
    layers = getattr(model, 'model', None)
    head = layers[-1] if layers is not None and hasattr(layers[-1], 'cv3') else None
    if head is not None:
        return {id(p) for p in head.cv3.parameters()}
    if hasattr(model, 'cls'):
        return {id(p) for p in model.cls.parameters()}
    return set()


def view_disagreement(student_logits, privileged_logits):
    with torch.no_grad():
        p_rgb = student_logits.detach().float().sigmoid()
        p_priv = privileged_logits.detach().float().sigmoid()
        active = torch.maximum(p_rgb, p_priv) > ACTIVE_PROBABILITY
        n = int(active.sum())
        if n == 0:
            return dict(active_entries=0, abs_diff_mean_active=0., privileged_higher_fraction_active=0.)
        diff = (p_priv - p_rgb)[active]
        return dict(active_entries=n, abs_diff_mean_active=float(diff.abs().mean()),
                    privileged_higher_fraction_active=float((diff > 0).float().mean()))


def parameter_gradients(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach().float() for g, p in zip(gradients, parameters)]


def _dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def gradient_geometry(g_cls, g_rgb, g_priv, g_kd, w_priv, w_kd, scale=None):
    """Norms, pairwise cosines and the effective step relative to an RGBU-like step.

    effective_step_ratio = ||(g_rgb + w_priv g_priv + w_kd g_kd)/(1+w_priv)|| / ||g_rgb||.
    """
    n_cls, n_rgb, n_priv, n_kd = (_dot(g, g) ** .5 for g in (g_cls, g_rgb, g_priv, g_kd))
    scale = 1. + w_priv if scale is None else scale
    total = [(r + w_priv * p + w_kd * k) / scale for r, p, k in zip(g_rgb, g_priv, g_kd)]
    n_total = _dot(total, total) ** .5

    def cos(a, b, na, nb):
        return _dot(a, b) / (na * nb) if na > 0 and nb > 0 else None

    return dict(native_cls_parameter_gradient_l2=n_cls, native_rgb_parameter_gradient_l2=n_rgb,
                privileged_native_parameter_gradient_l2=n_priv, kd_parameter_gradient_l2=n_kd,
                total_parameter_gradient_l2=n_total,
                cos_privileged_native_vs_rgb_native=cos(g_priv, g_rgb, n_priv, n_rgb),
                cos_kd_vs_rgb_native=cos(g_kd, g_rgb, n_kd, n_rgb),
                effective_step_ratio_vs_rgb_native=n_total / n_rgb if n_rgb > 0 else None)


def composition_within_tolerance(actual, operands, coefficients, rtol=3e-4, atol=3e-5):
    """|a - sum c_i g_i| <= atol + rtol * sum |c_i g_i|, elementwise.

    The tolerance scales with the operands, not with their (possibly
    cancelling) sum, so legitimate FP32 reordering cannot fail the check.
    Returns (passed, worst_excess_ratio).
    """
    worst = 0.
    for i, a in enumerate(actual):
        parts = [None if g[i] is None else c * g[i] for g, c in zip(operands, coefficients)]
        parts = [p for p in parts if p is not None]
        if a is None:
            if parts and any(bool((p != 0).any()) for p in parts):
                return False, float('inf')
            continue
        expected = torch.stack(parts).sum(0) if parts else torch.zeros_like(a)
        scale = torch.stack([p.abs() for p in parts]).sum(0) if parts else torch.zeros_like(a)
        ratio = ((a - expected).abs() / (atol + rtol * scale)).max()
        worst = max(worst, float(ratio))
    return worst <= 1., worst


def composition_report(actual, operands, coefficients, names, rtol=3e-4, atol=3e-5):
    """Where a composition check fails: per-parameter worst ratio and cancellation evidence."""
    rows, violating, total_elements, below_atol, violating_below_atol, cancel = [], 0, 0, 0, 0, []
    for i, a in enumerate(actual):
        parts = [c * g[i] for g, c in zip(operands, coefficients) if g[i] is not None]
        if a is None or not parts:
            continue
        expected = torch.stack(parts).sum(0)
        scale = torch.stack([p.abs() for p in parts]).sum(0)
        error = (a - expected).abs()
        bad = error > atol + rtol * scale
        total_elements += a.numel()
        below_atol += int((expected.abs() < atol).sum())
        if bool(bad.any()):
            violating += int(bad.sum())
            violating_below_atol += int((expected.abs()[bad] < atol).sum())
            cancel.append((expected.abs()[bad] / scale[bad].clamp_min(1e-30)).flatten().cpu())
            rows.append(dict(parameter=names[i], violations=int(bad.sum()), numel=a.numel(),
                             worst_ratio=float((error / (atol + rtol * scale)).max()),
                             max_abs_error=float(error.max()), max_operand_scale=float(scale.max())))
    cancel = torch.cat(cancel) if cancel else torch.zeros(0)
    rows.sort(key=lambda r: -r['worst_ratio'])
    return dict(violating_elements=violating, total_elements=total_elements,
                fraction_expected_below_atol=below_atol / max(total_elements, 1),
                fraction_violations_expected_below_atol=violating_below_atol / violating if violating else None,
                violation_cancellation_median=float(cancel.median()) if cancel.numel() else None,
                worst_parameters=rows[:10])


def gradient_difference_report(left, right, names):
    rows = []
    for name, a, b in zip(names, left, right):
        if a is None or b is None or torch.equal(a, b):
            continue
        d = (a - b).abs()
        rows.append(dict(parameter=name, differing=int((d > 0).sum()), numel=a.numel(),
                         max_abs_difference=float(d.max()), max_abs_value=float(a.abs().max())))
    rows.sort(key=lambda r: -r['max_abs_difference'])
    return dict(differing_parameters=len(rows), worst_parameters=rows[:10])


def make_type(rt, cfg):
    variant = cfg['method_variant']
    if variant not in PVC_ARMS:
        raise ValueError('Not a PVC arm: ' + str(variant))
    arm = PVC_ARMS[variant]

    class PVCCriterion:
        def __init__(self, native, teacher, reference, config, arm_name, trainer, sanity):
            for role, model in (('teacher', teacher), ('reference', reference)):
                if model is not None and any(True for _ in model.parameters()):
                    raise ValueError('PVC binds no frozen %s model' % role)
            self.native, self.trainer, self.cfg = native, trainer, cfg
            self.method_variant, self.target_source = cfg['method_variant'], cfg['target_source']
            self.view, self.kd_on = arm['view'], arm['kd']
            self.kd_mask_kind = arm.get('kd_mask')
            self.views = arm.get('views')
            self.kd_types = tuple(arm.get('kd_types', ('score',) * len(self.views))) if self.views else None
            self.w_geo = (float(cfg.get('geo_weight', GEO_WEIGHT))
                          if self.views and any('geo' in t for t in self.kd_types) else 0.)
            self.w_priv = float(cfg.get('privileged_native_weight', PRIVILEGED_NATIVE_WEIGHT))
            self.w_kd = float(cfg.get('kd_weight', KD_WEIGHT)) if self.kd_on else 0.
            self.calls = self.selected_total = self.fallback_batches = 0
            self.gradient_checks, self.last_stats = [], {}
            self.bn_isolation_checks = []
            self.main_mix = arm.get('main') == 'mix'
            self.clean_batches = self.augmented_batches = 0
            self.mix_bn_checks, self.mix_view_checks = [], []

        def __deepcopy__(self, memo):
            return None

        def components(self, prediction, batch, check_bn=False):
            if self.views:
                return self.components_mv(prediction, batch, check_bn)
            model = self.trainer.model
            student = raw_prediction(prediction)
            b = batch['img'].shape[0]
            assigned, losses, items = self.native.get_assigned_targets_and_loss(student, batch)
            # 返回的第 0 项是五元组 (fg_mask, target_gt_idx, target_bboxes, anchor_points,
            # stride_tensor)；原生 TAL 前景掩码是其中的 [0]，形状 (B, A)。
            fg_mask = assigned[0]
            native = losses * b
            images = privileged_images(batch, self.view)
            before = bn_state_digest(model) if check_bn else None
            with preserved_bn_running_stats(model):
                privileged = raw_prediction(model(images))
            if check_bn:
                self.bn_isolation_checks.append(dict(batch=self.calls, unchanged=before == bn_state_digest(model)))
                if not self.bn_isolation_checks[-1]['unchanged']:
                    raise AssertionError('Privileged forward changed BN running statistics')
            if privileged['scores'].shape != student['scores'].shape:
                raise ValueError('Privileged and RGB raw grids differ')
            _, p_losses, p_items = self.native.get_assigned_targets_and_loss(privileged, dict(batch, img=images))
            privileged_native = p_losses * b
            kd_mask = fg_mask if arm.get('kd_mask') == 'native_tal_fg' else None
            kd, norm = dense_privileged_kd(student['scores'], privileged['scores'],
                                           anchor_mask=kd_mask)
            return dict(native=native, native_total=native.sum(), native_cls=native[1], items=items,
                        privileged_native_total=privileged_native.sum(), privileged_items=p_items,
                        kd=kd, batch_kd=b * kd, kd_norm=norm, batch_size=b,
                        student_scores=student['scores'], privileged_scores=privileged['scores'])

        def components_mv(self, prediction, batch, check_bn=False):
            """xm_geo_r1: native RGB loss plus, per privileged view, a BN-restored forward, its native loss and its KD
            (score KD and/or geometry KD by kd_types; existing arms are all-score and keep the xm_mv_r1 operations)."""
            model = self.trainer.model
            student = raw_prediction(prediction)
            b = batch['img'].shape[0]
            assigned, losses, items = self.native.get_assigned_targets_and_loss(student, batch)
            native = losses * b
            fg_mask = assigned[0] if self.w_geo != 0. else None
            p_totals, p_items, kds, norms, p_scores = [], [], [], [], []
            kd_views, geos, geo_views, geo_counts, p_boxes = [], [], [], [], []
            for view, kd_type in zip(self.views, self.kd_types):
                images = privileged_images(batch, view)
                before = bn_state_digest(model) if check_bn else None
                with preserved_bn_running_stats(model):
                    privileged = raw_prediction(model(images))
                if check_bn:
                    self.bn_isolation_checks.append(dict(batch=self.calls, view=view, unchanged=before == bn_state_digest(model)))
                    if not self.bn_isolation_checks[-1]['unchanged']:
                        raise AssertionError('Privileged forward changed BN running statistics: ' + view)
                if privileged['scores'].shape != student['scores'].shape:
                    raise ValueError('Privileged and RGB raw grids differ')
                _, v_losses, v_items = self.native.get_assigned_targets_and_loss(privileged, dict(batch, img=images))
                p_totals.append((v_losses * b).sum()); p_items.append(v_items)
                if 'score' in kd_type:
                    kd_v, norm_v = dense_privileged_kd(student['scores'], privileged['scores'])
                    kds.append(kd_v); norms.append(norm_v); kd_views.append(view)
                if 'geo' in kd_type:
                    geo_v, n_v = dense_geometry_kd(student['boxes'], privileged['boxes'], fg_mask)
                    geos.append(geo_v); geo_views.append(view); geo_counts.append(n_v)
                p_scores.append(privileged['scores']); p_boxes.append(privileged['boxes'])
            kd = torch.stack(kds).sum()
            geo = torch.stack(geos).sum() if geos else None
            return dict(native=native, native_total=native.sum(), native_cls=native[1], items=items,
                        privileged_native_total=torch.stack(p_totals).sum(), privileged_native_totals=p_totals,
                        privileged_items=p_items[0], privileged_items_all=p_items,
                        kd=kd, batch_kd=b * kd, kds=kds, kd_views=kd_views, kd_norm=norms[0], kd_norms=norms, batch_size=b,
                        geo=geo, geos=geos, geo_views=geo_views, geo_counts=geo_counts, fg_mask=fg_mask,
                        batch_geo=None if geo is None else b * GEO_DFL_SCALE * geo,
                        student_scores=student['scores'], student_boxes=student['boxes'],
                        privileged_scores=p_scores[0], privileged_scores_all=p_scores, privileged_boxes_all=p_boxes)

        def normalizer(self):
            return 1. + (len(self.views) if self.views else 1) * self.w_priv

        def total(self, parts):
            if self.views:
                total = parts['native_total'] + self.w_priv * parts['privileged_native_total']
                if self.w_kd != 0.:
                    total = total + self.w_kd * parts['batch_kd']
                if parts.get('batch_geo') is not None:
                    total = total + self.w_geo * parts['batch_geo']
                return total if self.w_priv == 0. else total / self.normalizer()
            total = parts['native_total'] + self.w_priv * parts['privileged_native_total']
            if self.w_kd != 0.:
                total = total + self.w_kd * parts['batch_kd']
            return total if self.w_priv == 0. else total / (1. + self.w_priv)

        def mix_after_step(self, batch):
            """xm_fz2_r1 MIX_PVC, as single_view.SingleViewCriterion: after a perturbed main step restore BN and update it
            with one no-grad clean forward, so the deployed running statistics see clean RGB once per batch."""
            if not batch.get('_single_view_augmented', False):
                self.clean_batches += 1
                return
            from single_view import restore_bn
            self.augmented_batches += 1
            snapshot = batch.pop('_single_view_bn_before')
            clean = batch.pop('_single_view_clean_img')
            if len(self.mix_view_checks) < 3:
                fused = privileged_images(dict(batch, _single_view_clean_img=clean), self.view)
                self.mix_view_checks.append(torch.equal(fused, (clean + batch['strong_img']) * .5)
                                            and not torch.equal(batch['img'], clean))
                if not self.mix_view_checks[-1]:
                    raise AssertionError('MIX_PVC fused view is not built from the clean RGB')
            restore_bn(snapshot)
            with torch.no_grad():
                self.trainer.model(clean)
            if len(self.mix_bn_checks) < 3:
                ok = all(int(m.num_batches_tracked) == int(tracked) + 1 for m, _, _, tracked in snapshot)
                self.mix_bn_checks.append(ok)
                if not ok:
                    raise AssertionError('Clean-only BN update did not occur exactly once')

        def __call__(self, prediction, batch):
            if 'strong_img' not in batch:
                return self.native(prediction, batch)
            self.calls += 1
            due = self.calls <= 3 or self.calls % self.cfg['log_every_batches'] == 0
            parts = self.components(prediction, batch, check_bn=self.calls <= 3)
            total = self.total(parts)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite PVC total')
            self.selected_total += parts['batch_size']
            if self.main_mix:
                self.mix_after_step(batch)
            if due:
                params = tuple(p for p in self.trainer.model.parameters() if p.requires_grad)
                geometry = gradient_geometry(
                    parameter_gradients(parts['native_cls'], params),
                    parameter_gradients(parts['native_total'], params),
                    parameter_gradients(parts['privileged_native_total'], params),
                    parameter_gradients(parts['batch_kd'], params), self.w_priv, self.w_kd, scale=self.normalizer())
                g_native_cls = geometry['native_cls_parameter_gradient_l2']
                g_priv = geometry['privileged_native_parameter_gradient_l2']
                g_kd = geometry['kd_parameter_gradient_l2']
                check = dict(batch=self.calls, epoch=int(self.trainer.epoch), campaign_arm=self.cfg['campaign_arm'],
                             **geometry,
                             weighted_privileged_native_parameter_gradient_l2=self.w_priv * g_priv,
                             weighted_kd_parameter_gradient_l2=self.w_kd * g_kd,
                             weighted_kd_to_native_cls_ratio=self.w_kd * g_kd / g_native_cls if g_native_cls > 0 else None,
                             weighted_privileged_to_native_cls_ratio=self.w_priv * g_priv / g_native_cls if g_native_cls > 0 else None,
                             loss_normalizer=self.normalizer(),
                             privileged_native_weight=self.w_priv, kd_weight=self.w_kd, view=self.view,
                             teacher_has_grad=False, reference_has_grad=False, frozen_models_require_grad=False,
                             actual_batch_size=parts['batch_size'])
                if parts.get('batch_geo') is not None:
                    g_geo_all = parameter_gradients(parts['batch_geo'], params)
                    g_dfl_all = parameter_gradients(parts['native'][2], params)
                    n_geo, n_dfl = _dot(g_geo_all, g_geo_all) ** .5, _dot(g_dfl_all, g_dfl_all) ** .5
                    check.update(geo_parameter_gradient_l2=n_geo, weighted_geo_parameter_gradient_l2=self.w_geo * n_geo,
                                 native_dfl_parameter_gradient_l2=n_dfl, geo_weight=self.w_geo,
                                 weighted_geo_to_native_dfl_ratio=self.w_geo * n_geo / n_dfl if n_dfl > 0 else None,
                                 geo_foreground_anchors=list(parts['geo_counts']))
                if self.calls <= 3:
                    self.gradient_checks.append(check)
                stats = dict(check, native_total=float(parts['native_total'].detach()),
                             native_items=loss_item_values(parts['items']),
                             privileged_native_total=float(parts['privileged_native_total'].detach()),
                             privileged_items=loss_item_values(parts['privileged_items']),
                             kd=float(parts['kd'].detach()), kd_norm=float(parts['kd_norm']),
                             total_loss=float(total.detach()),
                             optimizer_updates=getattr(self.trainer, 'real_updates', None),
                             **view_disagreement(parts['student_scores'], parts['privileged_scores'],
                             ), **(dict(views=list(self.views), kd_per_view=[float(k.detach()) for k in parts['kds']],
                                        privileged_native_per_view=[float(t.detach()) for t in parts['privileged_native_totals']])
                                   if self.views else {}))
                if parts.get('batch_geo') is not None:
                    stats.update(geo=float(parts['geo'].detach()), geo_per_view=[float(g.detach()) for g in parts['geos']],
                                 geo_views=list(parts['geo_views']), kd_views=list(parts['kd_views']), kd_types=list(self.kd_types))
                if self.main_mix:
                    stats.update(mix_clean_batches=self.clean_batches, mix_augmented_batches=self.augmented_batches)
                self.last_stats = stats
                writer = getattr(self.trainer, 'save_dir', None)
                if writer is not None:
                    from common import append
                    append(writer / 'pvc_batches.jsonl', stats)
                    if self.calls <= 3:
                        append(writer / 'gradient_checks.jsonl', check)
            return total, parts['items']

    return PVCCriterion


def verify_pvc_canary(criterion, cfg):
    """Canary acceptance from evidence the criterion actually emitted."""
    if criterion.selected_total <= 0:
        raise AssertionError('No privileged-view batches')
    if not criterion.gradient_checks:
        raise AssertionError('No PVC parameter-gradient checks')
    if len(criterion.bn_isolation_checks) < 3 or not all(r['unchanged'] for r in criterion.bn_isolation_checks):
        raise AssertionError('BN running-statistics isolation not verified')
    for row in criterion.gradient_checks:
        for key in ('native_cls_parameter_gradient_l2', 'native_rgb_parameter_gradient_l2',
                    'privileged_native_parameter_gradient_l2', 'kd_parameter_gradient_l2',
                    'total_parameter_gradient_l2'):
            value = row.get(key)
            if value is None or not value == value or value < 0:
                raise AssertionError('Missing or invalid gradient evidence: ' + key)
        if row.get('cos_privileged_native_vs_rgb_native') is None or row.get('effective_step_ratio_vs_rgb_native') is None:
            raise AssertionError('Missing gradient cosine / effective step evidence')
    if not any(r['privileged_native_parameter_gradient_l2'] > 0 for r in criterion.gradient_checks):
        raise AssertionError('Privileged native loss produced no parameter gradient')
    if PVC_ARMS[cfg['method_variant']]['kd'] and not any(r['weighted_kd_parameter_gradient_l2'] > 0 for r in criterion.gradient_checks):
        raise AssertionError('KD arm produced no weighted KD gradient')
    if not PVC_ARMS[cfg['method_variant']]['kd'] and any(r['kd_weight'] != 0 for r in criterion.gradient_checks):
        raise AssertionError('NOKD arm has nonzero KD weight')
    if getattr(criterion, 'w_geo', 0.) != 0. and not any((r.get('weighted_geo_parameter_gradient_l2') or 0) > 0
                                                         for r in criterion.gradient_checks):
        raise AssertionError('Geometry KD arm produced no weighted geometry gradient')
    mix = {}
    if getattr(criterion, 'main_mix', False):
        if not (criterion.clean_batches > 0 and criterion.augmented_batches > 0):
            raise AssertionError('MIX_PVC: clean and perturbed main batches not both seen')
        if not criterion.mix_bn_checks or not all(criterion.mix_bn_checks):
            raise AssertionError('MIX_PVC: clean-only BN update not verified')
        if not criterion.mix_view_checks or not all(criterion.mix_view_checks):
            raise AssertionError('MIX_PVC: fused view not verified to use the clean RGB')
        mix = dict(mix_clean_batches=criterion.clean_batches, mix_augmented_batches=criterion.augmented_batches,
                   mix_bn_checks=list(criterion.mix_bn_checks), mix_view_checks=list(criterion.mix_view_checks))
    rows = criterion.gradient_checks
    return dict(status='PASS', checks=len(rows),
                bn_isolation_checks=len(criterion.bn_isolation_checks),
                advisory_cos_privileged_vs_rgb=[r['cos_privileged_native_vs_rgb_native'] for r in rows],
                advisory_effective_step_ratio=[r['effective_step_ratio_vs_rgb_native'] for r in rows],
                composition_and_zero_update='separate_engineering_receipt_required', **mix)


def engineering_checks(criterion, model, forward, batch, rtol=3e-4, atol=3e-5):
    """One real batch: zero-weight equivalence, composition, detach, BN isolation, inputs.

    `forward(images)` must run the trainable model on a tensor (train mode).
    Runs under engineering_numerics(); `native_gradient_repeat_bitwise` is the
    control that tells whether bitwise checks are meaningful on this backend.
    """
    with engineering_numerics() as numerics:
        result = _engineering_checks(criterion, model, forward, batch, rtol, atol)
    result['numerics'] = {k: v for k, v in numerics.items() if k != 'restored_after'}
    return result


def _engineering_checks(criterion, model, forward, batch, rtol, atol):
    if getattr(criterion, 'views', None) and getattr(criterion, 'w_geo', 0.) != 0.:
        return _engineering_checks_mv_geo(criterion, model, forward, batch, rtol, atol)
    if getattr(criterion, 'views', None):
        return _engineering_checks_mv(criterion, model, forward, batch, rtol, atol)
    checks = {}
    named = tuple((n, p) for n, p in model.named_parameters() if p.requires_grad)
    names = tuple(n for n, _ in named)
    params = tuple(p for _, p in named)

    def grads(loss):
        return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)

    expected_paired = (batch['img'] + batch['strong_img']) * .5
    checks['privileged_input_equals_FUSIONNU_pixelmean'] = torch.equal(privileged_images(batch, 'paired'), expected_paired)
    rolled = privileged_images(batch, 'shuffled')
    b = batch['img'].shape[0]
    checks['shuffled_view_has_no_self_pair'] = all(
        not torch.equal(rolled[i], expected_paired[i]) for i in range(b)) if b > 1 else False
    if criterion.view == 'augpaired':
        view = privileged_images(batch, 'augpaired')
        augmented = privileged_images(batch, 'augmented')
        checks['augpaired_view_matches_formula'] = torch.equal(view, (photometric_jitter(batch['img']) + batch['strong_img']) * .5)
        checks['augpaired_is_paired_plus_pvc_aug_perturbation'] = torch.allclose(view - expected_paired, augmented - batch['img'],
                                                                                 atol=1e-6)
        checks['augpaired_differs_from_paired_and_augmented'] = (not torch.equal(view, expected_paired)
                                                                 and not torch.equal(view, augmented))
    if criterion.view == 'paired_clean':
        clean = batch.get('_single_view_clean_img', batch['img'])
        checks['mix_fused_view_uses_clean_rgb'] = torch.equal(privileged_images(batch, 'paired_clean'),
                                                              (clean + batch['strong_img']) * .5)
        perturbed = dict(batch, img=privileged_images(dict(batch, img=clean), 'augmented'), _single_view_clean_img=clean)
        checks['mix_fused_view_ignores_perturbed_main'] = (
            torch.equal(privileged_images(perturbed, 'paired_clean'), (clean + batch['strong_img']) * .5)
            and not torch.equal(perturbed['img'], clean))

    bn_before = bn_state_digest(model)
    prediction = forward(batch['img'])
    bn_after_rgb = bn_state_digest(model)
    parts = criterion.components(prediction, batch, check_bn=False)
    checks['bn_unchanged_by_privileged_forward'] = bn_state_digest(model) == bn_after_rgb
    checks['bn_updated_by_rgb_forward'] = bn_after_rgb != bn_before

    def same(left, right):
        return all((a is None and z is None) or (a is not None and z is not None and torch.equal(a, z))
                   for a, z in zip(left, right))

    g_native = grads(parts['native_total'])
    g_native_repeat = grads(parts['native_total'])
    checks['native_gradient_repeat_bitwise'] = same(g_native, g_native_repeat)
    g_priv = grads(parts['privileged_native_total'])
    g_kd = grads(parts['batch_kd'])
    zero = parts['native_total'] + 0. * parts['privileged_native_total'] + 0. * parts['batch_kd']
    checks['zero_weight_exact_loss'] = torch.equal(zero, parts['native_total'])
    g_zero = grads(zero)
    checks['zero_weight_exact_parameter_gradients'] = same(g_native, g_zero)
    total = criterion.total(parts)
    scale = 1. + criterion.w_priv
    g_total = grads(total)
    coefficients = (1. / scale, criterion.w_priv / scale, criterion.w_kd / scale)
    passed, worst = composition_within_tolerance(g_total, (g_native, g_priv, g_kd), coefficients, rtol, atol)
    checks['parameter_gradient_composition_operand_scaled'] = passed
    diagnostics = dict(
        repeat=None if checks['native_gradient_repeat_bitwise'] else gradient_difference_report(g_native, g_native_repeat, names),
        zero_weight=None if checks['zero_weight_exact_parameter_gradients'] else gradient_difference_report(g_native, g_zero, names),
        composition=composition_report(g_total, (g_native, g_priv, g_kd), coefficients, names, rtol, atol))
    kd_to_priv = torch.autograd.grad(parts['batch_kd'], parts['privileged_scores'], retain_graph=True, allow_unused=True)[0]
    checks['kd_has_no_gradient_into_privileged_view'] = kd_to_priv is None or bool((kd_to_priv == 0).all())
    checks['kd_nonzero_parameter_gradient'] = any(g is not None and bool((g != 0).any()) for g in g_kd)
    checks['privileged_native_nonzero_parameter_gradient'] = any(g is not None and bool((g != 0).any()) for g in g_priv)
    same, _ = dense_privileged_kd(parts['student_scores'], parts['student_scores'])
    checks['kd_exact_zero_for_identical_views'] = float(same.detach()) == 0.
    return dict(checks=checks, composition_worst_ratio=worst, diagnostics=diagnostics,
                kd=float(parts['kd'].detach()), privileged_native_total=float(parts['privileged_native_total'].detach()),
                native_total=float(parts['native_total'].detach()))


def _engineering_checks_mv(criterion, model, forward, batch, rtol, atol):
    """xm_mv_r1 engineering checks: per-view inputs, BN isolation, zero weight, (1 + 2n)-operand composition, detach."""
    checks = {}
    named = tuple((n, p) for n, p in model.named_parameters() if p.requires_grad)
    names = tuple(n for n, _ in named)
    params = tuple(p for _, p in named)

    def grads(loss):
        return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)

    def same(left, right):
        return all((a is None and z is None) or (a is not None and z is not None and torch.equal(a, z))
                   for a, z in zip(left, right))

    views = {v: privileged_images(batch, v) for v in criterion.views}
    expected_paired = (batch['img'] + batch['strong_img']) * .5
    b = batch['img'].shape[0]
    if 'paired' in views:
        checks['paired_view_equals_FUSIONNU_pixelmean'] = torch.equal(views['paired'], expected_paired)
    if 'shuffled' in views:
        checks['shuffled_view_has_no_self_pair'] = all(
            not torch.equal(views['shuffled'][i], expected_paired[i]) for i in range(b)) if b > 1 else False
    if 'raw_ir' in views:
        checks['raw_ir_view_equals_strong_img'] = torch.equal(views['raw_ir'], batch['strong_img'])
    if 'iaf' in views:
        checks['iaf_view_matches_formula'] = torch.equal(views['iaf'], illumination_aware_fusion(batch['img'], batch['strong_img']))
        checks['iaf_view_differs_from_pixelmean_and_ir'] = (not torch.equal(views['iaf'], expected_paired)
                                                           and not torch.equal(views['iaf'], batch['strong_img']))
    order = list(criterion.views)
    checks['views_pairwise_distinct'] = all(not torch.equal(views[x], views[y])
                                            for i, x in enumerate(order) for y in order[i + 1:])
    checks['views_same_geometry_as_rgb'] = all(v.shape == batch['img'].shape for v in views.values())

    bn_before = bn_state_digest(model)
    prediction = forward(batch['img'])
    bn_after_rgb = bn_state_digest(model)
    parts = criterion.components(prediction, batch, check_bn=False)
    checks['bn_unchanged_by_privileged_forwards'] = bn_state_digest(model) == bn_after_rgb
    checks['bn_updated_by_rgb_forward'] = bn_after_rgb != bn_before

    g_native = grads(parts['native_total'])
    g_native_repeat = grads(parts['native_total'])
    checks['native_gradient_repeat_bitwise'] = same(g_native, g_native_repeat)
    g_priv = [grads(t) for t in parts['privileged_native_totals']]
    g_kd = [grads(b * k) for k in parts['kds']]
    zero = parts['native_total'] + 0. * parts['privileged_native_total'] + 0. * parts['batch_kd']
    checks['zero_weight_exact_loss'] = torch.equal(zero, parts['native_total'])
    g_zero = grads(zero)
    checks['zero_weight_exact_parameter_gradients'] = same(g_native, g_zero)
    total = criterion.total(parts)
    scale = criterion.normalizer()
    checks['loss_normalizer_is_1_plus_n_views_w'] = scale == 1. + len(order) * criterion.w_priv
    manual = parts['native_total']
    for t in parts['privileged_native_totals']:
        manual = manual + criterion.w_priv * t
    for k in parts['kds']:
        manual = manual + criterion.w_kd * (b * k)
    checks['total_matches_per_view_sum'] = bool(torch.allclose(total, manual / scale, rtol=1e-6, atol=1e-7))
    g_total = grads(total)
    operands = (g_native, *g_priv, *g_kd)
    coefficients = (1. / scale,) + (criterion.w_priv / scale,) * len(g_priv) + (criterion.w_kd / scale,) * len(g_kd)
    passed, worst = composition_within_tolerance(g_total, operands, coefficients, rtol, atol)
    checks['parameter_gradient_composition_operand_scaled'] = passed
    diagnostics = dict(
        repeat=None if checks['native_gradient_repeat_bitwise'] else gradient_difference_report(g_native, g_native_repeat, names),
        zero_weight=None if checks['zero_weight_exact_parameter_gradients'] else gradient_difference_report(g_native, g_zero, names),
        composition=composition_report(g_total, operands, coefficients, names, rtol, atol))
    for i, view in enumerate(order):
        to_view = torch.autograd.grad(b * parts['kds'][i], parts['privileged_scores_all'][i], retain_graph=True, allow_unused=True)[0]
        checks['kd_has_no_gradient_into_privileged_view_' + view] = to_view is None or bool((to_view == 0).all())
        checks['kd_nonzero_parameter_gradient_' + view] = any(g is not None and bool((g != 0).any()) for g in g_kd[i])
        checks['privileged_native_nonzero_parameter_gradient_' + view] = any(g is not None and bool((g != 0).any()) for g in g_priv[i])
    same_kd, _ = dense_privileged_kd(parts['student_scores'], parts['student_scores'])
    checks['kd_exact_zero_for_identical_views'] = float(same_kd.detach()) == 0.
    return dict(checks=checks, composition_worst_ratio=worst, diagnostics=diagnostics,
                kd=float(parts['kd'].detach()), privileged_native_total=float(parts['privileged_native_total'].detach()),
                native_total=float(parts['native_total'].detach()),
                kd_per_view=[float(k.detach()) for k in parts['kds']],
                privileged_native_per_view=[float(t.detach()) for t in parts['privileged_native_totals']])


def _engineering_checks_mv_geo(criterion, model, forward, batch, rtol, atol):
    """xm_geo_r1 engineering checks for multi-view arms: per-view inputs (incl. raw IR), BN isolation, zero weight,
    per-operand composition (native, privileged natives, score KDs, geometry KDs), detach of every target, geometry
    gradient confined to RGB foreground anchors and absent from the classification branch."""
    checks = {}
    named = tuple((n, p) for n, p in model.named_parameters() if p.requires_grad)
    names = tuple(n for n, _ in named)
    params = tuple(p for _, p in named)

    def grads(loss):
        return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)

    def same(left, right):
        return all((a is None and z is None) or (a is not None and z is not None and torch.equal(a, z))
                   for a, z in zip(left, right))

    views = {v: privileged_images(batch, v) for v in criterion.views}
    expected_paired = (batch['img'] + batch['strong_img']) * .5
    b = batch['img'].shape[0]
    if 'paired' in views:
        checks['paired_view_equals_FUSIONNU_pixelmean'] = torch.equal(views['paired'], expected_paired)
    if 'shuffled' in views:
        checks['shuffled_view_has_no_self_pair'] = all(
            not torch.equal(views['shuffled'][i], expected_paired[i]) for i in range(b)) if b > 1 else False
    if 'raw_ir' in views:
        checks['raw_ir_view_equals_strong_img'] = torch.equal(views['raw_ir'], batch['strong_img'])
        checks['raw_ir_view_differs_from_pixelmean'] = not torch.equal(views['raw_ir'], expected_paired)
    order = list(criterion.views)
    checks['views_pairwise_distinct'] = all(not torch.equal(views[x], views[y])
                                            for i, x in enumerate(order) for y in order[i + 1:])
    checks['views_same_geometry_as_rgb'] = all(v.shape == batch['img'].shape for v in views.values())

    bn_before = bn_state_digest(model)
    prediction = forward(batch['img'])
    bn_after_rgb = bn_state_digest(model)
    parts = criterion.components(prediction, batch, check_bn=False)
    checks['bn_unchanged_by_privileged_forwards'] = bn_state_digest(model) == bn_after_rgb
    checks['bn_updated_by_rgb_forward'] = bn_after_rgb != bn_before

    g_native = grads(parts['native_total'])
    g_native_repeat = grads(parts['native_total'])
    checks['native_gradient_repeat_bitwise'] = same(g_native, g_native_repeat)
    g_priv = [grads(t) for t in parts['privileged_native_totals']]
    g_kd = [grads(b * k) for k in parts['kds']]
    g_geo = [grads(b * GEO_DFL_SCALE * g) for g in parts['geos']]
    zero = parts['native_total'] + 0. * parts['privileged_native_total'] + 0. * parts['batch_kd']
    if parts['batch_geo'] is not None:
        zero = zero + 0. * parts['batch_geo']
    checks['zero_weight_exact_loss'] = torch.equal(zero, parts['native_total'])
    g_zero = grads(zero)
    checks['zero_weight_exact_parameter_gradients'] = same(g_native, g_zero)
    total = criterion.total(parts)
    scale = criterion.normalizer()
    checks['loss_normalizer_is_1_plus_n_views_w'] = scale == 1. + len(order) * criterion.w_priv
    manual = parts['native_total']
    for t in parts['privileged_native_totals']:
        manual = manual + criterion.w_priv * t
    for k in parts['kds']:
        manual = manual + criterion.w_kd * (b * k)
    for g in parts['geos']:
        manual = manual + criterion.w_geo * (b * GEO_DFL_SCALE * g)
    checks['total_matches_per_view_sum'] = bool(torch.allclose(total, manual / scale, rtol=1e-6, atol=1e-7))
    g_total = grads(total)
    operands = (g_native, *g_priv, *g_kd, *g_geo)
    coefficients = ((1. / scale,) + (criterion.w_priv / scale,) * len(g_priv) + (criterion.w_kd / scale,) * len(g_kd)
                    + (criterion.w_geo / scale,) * len(g_geo))
    passed, worst = composition_within_tolerance(g_total, operands, coefficients, rtol, atol)
    checks['parameter_gradient_composition_operand_scaled'] = passed
    diagnostics = dict(
        repeat=None if checks['native_gradient_repeat_bitwise'] else gradient_difference_report(g_native, g_native_repeat, names),
        zero_weight=None if checks['zero_weight_exact_parameter_gradients'] else gradient_difference_report(g_native, g_zero, names),
        composition=composition_report(g_total, operands, coefficients, names, rtol, atol))
    for i, view in enumerate(parts['kd_views']):
        j = order.index(view)
        to_view = torch.autograd.grad(b * parts['kds'][i], parts['privileged_scores_all'][j], retain_graph=True, allow_unused=True)[0]
        checks['kd_has_no_gradient_into_privileged_view_' + view] = to_view is None or bool((to_view == 0).all())
        checks['kd_nonzero_parameter_gradient_' + view] = any(g is not None and bool((g != 0).any()) for g in g_kd[i])
    for i, view in enumerate(order):
        checks['privileged_native_nonzero_parameter_gradient_' + view] = any(g is not None and bool((g != 0).any()) for g in g_priv[i])
    cls_ids = classification_head_parameter_ids(model)
    fg = parts['fg_mask']
    if parts['geos']:
        checks['foreground_mask_is_boolean_BxA'] = (fg is not None and fg.dtype == torch.bool
                                                     and tuple(fg.shape) == (b, parts['student_boxes'].shape[-1]))
        checks['foreground_anchors_present'] = int(fg.sum()) > 0
        checks['classification_head_identified'] = len(cls_ids) > 0
    for i, view in enumerate(parts['geo_views']):
        j = order.index(view)
        batch_geo_v = b * GEO_DFL_SCALE * parts['geos'][i]
        to_view = torch.autograd.grad(batch_geo_v, parts['privileged_boxes_all'][j], retain_graph=True, allow_unused=True)[0]
        checks['geo_has_no_gradient_into_privileged_view_' + view] = to_view is None or bool((to_view == 0).all())
        checks['geo_nonzero_parameter_gradient_' + view] = any(g is not None and bool((g != 0).any()) for g in g_geo[i])
        checks['geo_zero_gradient_on_classification_head_' + view] = bool(cls_ids) and all(
            g is None or bool((g == 0).all()) for (n, p), g in zip(named, g_geo[i]) if id(p) in cls_ids)
        to_student = torch.autograd.grad(batch_geo_v, parts['student_boxes'], retain_graph=True, allow_unused=True)[0]
        outside = (~fg).unsqueeze(1).expand_as(to_student)
        checks['geo_gradient_only_on_foreground_anchors_' + view] = bool((to_student[outside] == 0).all())
        to_scores = torch.autograd.grad(batch_geo_v, parts['student_scores'], retain_graph=True, allow_unused=True)[0]
        checks['geo_has_no_gradient_into_student_scores_' + view] = to_scores is None or bool((to_scores == 0).all())
    same_kd, _ = dense_privileged_kd(parts['student_scores'], parts['student_scores'])
    checks['kd_exact_zero_for_identical_views'] = float(same_kd.detach()) == 0.
    if parts['geos']:
        same_geo, _ = dense_geometry_kd(parts['student_boxes'], parts['student_boxes'], fg)
        checks['geo_exact_zero_for_identical_views'] = float(same_geo.detach()) == 0.
    return dict(checks=checks, composition_worst_ratio=worst, diagnostics=diagnostics,
                kd=float(parts['kd'].detach()), privileged_native_total=float(parts['privileged_native_total'].detach()),
                native_total=float(parts['native_total'].detach()),
                kd_per_view=[float(k.detach()) for k in parts['kds']], kd_views=list(parts['kd_views']),
                geo_per_view=[float(g.detach()) for g in parts['geos']], geo_views=list(parts['geo_views']),
                geo_foreground_anchors=list(parts['geo_counts']),
                privileged_native_per_view=[float(t.detach()) for t in parts['privileged_native_totals']])
