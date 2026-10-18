"""Static cross-modal distillation comparison arms, r2 (xm_full_r2; user 2026-10-14: 2-3 comparison methods for the paper).

A frozen Fusion11n teacher distils into the RGB-only YOLO11n student, which keeps its native union-GT loss. The teacher
is MODEL_POLICY asset drone_fusionnu_s42_union_e200_90 (night_FUSIONNU_s42_r3, E200 FP32 EMA), reads the pixel-mean
view (RGB + IR) / 2 of the same replayed geometry (with xm_full_r1 the same HSV draw and the same paired mosaic), runs
in eval mode under no_grad, and stays outside the student, its EMA and the optimizer.
  CMP_KD   logit distillation (Hinton et al. 2015) in its dense-detector form: per anchor and class Bernoulli
           KL(sigmoid teacher || sigmoid student) / max(sum p_teacher, 1) over all anchors, weight 0.5 per image -- the
           same KD term and weight as the shared-weight method (PVC), with a frozen external teacher instead of the
           co-trained fused view
  CMP_CWD  channel-wise distillation (Shu et al., ICCV 2021), the operator of CrossFusionKD, on the three neck outputs
           feeding Detect (P3/P4/P5): per (image, channel) softmax over positions at tau 1, KL(T || S) * tau^2, mean over
           images and channels, summed over levels, weight 1.0 per image; identity alignment (same YOLO11n, same init)
  CMP_LD   localization distillation (Zheng et al., CVPR 2022): KL of the teacher's and the student's DFL box-side
           distributions at the student's native TAL foreground anchors, tau 10, weighted like the native DFL (1.5 / image)
total = native_total + kd_weight * batch_kd; deployment: the unchanged RGB YOLO11n.
r2 vs xm_cmp_r1: CMP_KD added; the gradient-composition engineering check is decided in float64 on copies of the
student, the teacher and 4 images (phase10: the r1 float32 check failed at worst ratio 1.6 / 1.2 because both terms share
one forward graph; in float64 the identity holds to ~1e-13); the float32 ratio is still reported, as advisory.
"""
import copy

import torch
import torch.nn.functional as F

from pvc_criterion import (GEO_DFL_SCALE, _dot, bn_state_digest, classification_head_parameter_ids,
                           composition_within_tolerance, dense_geometry_kd, dense_privileged_kd, engineering_numerics,
                           engineering_subbatch, loss_item_values, parameter_gradients, raw_prediction)

CMP_ARMS = {'CMP_KD': dict(kd='logit'), 'CMP_CWD': dict(kd='cwd'), 'CMP_LD': dict(kd='ld')}
TEACHER = '/mnt/dataY/ydf/rgbt90/active/drone_union_3arm_20261010/runs/night_FUSIONNU_s42_r3/weights/last.pt'
TEACHER_SHA256 = '0573633535dc75af61d1a327614baf70070f3cb2c821bbb043bed3ed3205d313'
CWD_TEMPERATURE = 1.0
LD_TEMPERATURE = 10.0
KD_WEIGHTS = dict(CMP_KD=0.5, CMP_CWD=1.0, CMP_LD=1.0)
FLOAT64_IMAGES = 4

CMP_CONTRACT = dict(
    version='xm_cmp_r2',
    teacher='frozen Fusion11n night_FUSIONNU_s42_r3 E200 FP32 EMA (MODEL_POLICY drone_fusionnu_s42_union_e200_90)',
    teacher_input='(img + strong_img) * 0.5, same replayed geometry, HSV draw and paired mosaic',
    teacher_mode='eval, no_grad, requires_grad False, outside student / EMA / optimizer, BN statistics frozen',
    deployed_view='rgb', deployed_architecture='yolo11n_unchanged', native_retained=True,
    logit='dense_privileged_kd(student scores, teacher scores): Bernoulli KL over all anchors and classes / max(sum p_teacher, 1)',
    cwd='per level P3/P4/P5 neck outputs: tau^2 * KL(softmax_HW(t/tau) || softmax_HW(s/tau)) summed over positions, '
        'mean over images and channels; levels summed; tau 1; identity alignment',
    ld='KL(softmax(t_dfl/tau) || softmax(s_dfl/tau)) * tau^2, mean over 4 sides, mean over student native TAL foreground '
       'anchors; tau 10',
    total='native_total + kd_weight * batch_kd; KD / CWD batch_kd = b * kd; LD batch_kd = b * 1.5 * ld',
    kd_weights=KD_WEIGHTS, cwd_temperature=CWD_TEMPERATURE, ld_temperature=LD_TEMPERATURE,
    composition_check='float64 copies of student, teacher and %d images, rtol 1e-9, atol 1e-12' % FLOAT64_IMAGES,
    references=dict(kd='Hinton et al., Distilling the Knowledge in a Neural Network, 2015',
                    cwd='Shu et al., Channel-wise Knowledge Distillation for Dense Prediction, ICCV 2021',
                    crossfusionkd='CrossFusionKD, TechRxiv 2025 (fusion teacher -> RGB-only student, CWD / MGD)',
                    ld='Zheng et al., Localization Distillation for Dense Object Detection, CVPR 2022'))


def channel_wise_kl(student, teacher, temperature=CWD_TEMPERATURE):
    """One level: tau^2 * KL(softmax_HW(t / tau) || softmax_HW(s / tau)), summed over positions, mean over (image, channel)."""
    if student.shape != teacher.shape or student.ndim != 4:
        raise ValueError('Aligned NxCxHxW student/teacher features required')
    n, c = student.shape[:2]
    s = student.float().flatten(2) / temperature
    t = teacher.detach().float().flatten(2) / temperature
    kl = F.kl_div(s.log_softmax(-1), t.log_softmax(-1), reduction='sum', log_target=True)
    return kl / (n * c) * temperature ** 2


def cwd_loss(student_feats, teacher_feats, temperature=CWD_TEMPERATURE):
    if len(student_feats) != 3 or len(teacher_feats) != 3:
        raise ValueError('P3/P4/P5 neck outputs required')
    return torch.stack([channel_wise_kl(s, t, temperature) for s, t in zip(student_feats, teacher_feats)]).sum()


def detect_head_parameter_ids(model):
    """All Detect parameters (ultralytics model.model[-1]: cv2 box branch + cv3 class branch; the toy detector's heads)."""
    layers = getattr(model, 'model', None)
    if layers is not None and hasattr(layers[-1], 'cv2'):
        return {id(p) for p in layers[-1].parameters()}
    return {id(p) for name in ('cls', 'box') if hasattr(model, name) for p in getattr(model, name).parameters()}


def box_branch_parameter_ids(model):
    """The box branch only (ultralytics Detect.cv2; the toy detector's .box)."""
    layers = getattr(model, 'model', None)
    if layers is not None and hasattr(layers[-1], 'cv2'):
        return {id(p) for p in layers[-1].cv2.parameters()}
    return {id(p) for p in model.box.parameters()} if hasattr(model, 'box') else set()


def load_teacher(cfg, trainer):
    """The registered frozen Fusion11n endpoint through scene_runtime.frozen_model (E200, night s42, population, SHA)."""
    from scene_runtime import frozen_model
    import checkpoint as full
    if cfg.get('fusion_target_teacher') != TEACHER or cfg.get('teacher_sha256') != {TEACHER: TEACHER_SHA256}:
        raise ValueError('Comparison arms bind exactly the registered Fusion11n teacher')
    rng = full.rng_state()
    try:
        model = frozen_model(TEACHER, cfg, trainer.model.names, 'FUSIONNU').to(trainer.device)
    finally:
        full.restore_rng(rng)
    return model


def make_type(rt, cfg):
    variant = cfg['method_variant']
    if variant not in CMP_ARMS:
        raise ValueError('Not a comparison arm: ' + str(variant))
    kind = CMP_ARMS[variant]['kd']

    class CMPCriterion:
        def __init__(self, native, teacher, reference, config, arm_name, trainer, sanity):
            for role, model in (('teacher', teacher), ('reference', reference)):
                if model is not None and any(True for _ in model.parameters()):
                    raise ValueError('Legacy %s slot binds no model; the frozen teacher is loaded here' % role)
            self.native, self.trainer, self.cfg = native, trainer, cfg
            self.method_variant, self.target_source = cfg['method_variant'], cfg['target_source']
            self.kind, self.w = kind, float(cfg['kd_weight'])
            self.calls = self.selected_total = self.fallback_batches = 0
            self.gradient_checks, self.last_stats = [], {}
            self.teacher = load_teacher(cfg, trainer)
            if self.teacher.training or any(p.requires_grad for p in self.teacher.parameters()):
                raise AssertionError('Teacher must be frozen in eval mode')

        def __deepcopy__(self, memo):
            return None

        def teacher_raw(self, batch):
            if self.teacher.training:
                raise AssertionError('Frozen teacher left eval mode')
            images = (batch['img'] + batch['strong_img']) * .5
            with torch.no_grad():
                raw = raw_prediction(self.teacher(images))
            return dict(scores=raw['scores'].detach(), boxes=raw['boxes'].detach(), feats=[f.detach() for f in raw['feats']])

        def components(self, prediction, batch):
            student = raw_prediction(prediction)
            b = batch['img'].shape[0]
            assigned, losses, items = self.native.get_assigned_targets_and_loss(student, batch)
            native = losses * b
            teacher = self.teacher_raw(batch)
            if teacher['scores'].shape != student['scores'].shape or teacher['boxes'].shape != student['boxes'].shape:
                raise ValueError('Teacher and student raw grids differ')
            fg_mask, n = assigned[0], None
            if self.kind == 'logit':
                kd, _ = dense_privileged_kd(student['scores'], teacher['scores'])
                batch_kd = b * kd
            elif self.kind == 'cwd':
                kd = cwd_loss(student['feats'], teacher['feats'])
                batch_kd = b * kd
            else:
                kd, n = dense_geometry_kd(student['boxes'], teacher['boxes'], fg_mask, temperature=LD_TEMPERATURE)
                batch_kd = b * GEO_DFL_SCALE * kd
            return dict(native=native, native_total=native.sum(), native_cls=native[1], items=items, kd=kd,
                        batch_kd=batch_kd, foreground_anchors=n, fg_mask=fg_mask, batch_size=b,
                        student=student, teacher=teacher)

        def total(self, parts):
            return parts['native_total'] + self.w * parts['batch_kd']

        def __call__(self, prediction, batch):
            if 'strong_img' not in batch:
                return self.native(prediction, batch)
            self.calls += 1
            due = self.calls <= 3 or self.calls % self.cfg['log_every_batches'] == 0
            parts = self.components(prediction, batch)
            total = self.total(parts)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite comparison total')
            self.selected_total += parts['batch_size']
            if due:
                params = tuple(p for p in self.trainer.model.parameters() if p.requires_grad)
                g_native = parameter_gradients(parts['native_total'], params)
                g_kd = parameter_gradients(parts['batch_kd'], params)
                n_native, n_kd = _dot(g_native, g_native) ** .5, _dot(g_kd, g_kd) ** .5
                check = dict(batch=self.calls, epoch=int(self.trainer.epoch), campaign_arm=self.cfg['campaign_arm'],
                             kd_kind=self.kind, native_parameter_gradient_l2=n_native, kd_parameter_gradient_l2=n_kd,
                             weighted_kd_parameter_gradient_l2=self.w * n_kd,
                             weighted_kd_to_native_ratio=self.w * n_kd / n_native if n_native > 0 else None,
                             kd_weight=self.w, actual_batch_size=parts['batch_size'],
                             teacher_training=bool(self.teacher.training),
                             teacher_requires_grad=any(p.requires_grad for p in self.teacher.parameters()))
                if self.calls <= 3:
                    self.gradient_checks.append(check)
                stats = dict(check, native_total=float(parts['native_total'].detach()),
                             native_items=loss_item_values(parts['items']), kd=float(parts['kd'].detach()),
                             loss_unweighted=float(parts['kd'].detach()), total_loss=float(total.detach()),
                             foreground_anchors=parts['foreground_anchors'],
                             optimizer_updates=getattr(self.trainer, 'real_updates', None))
                self.last_stats = stats
                writer = getattr(self.trainer, 'save_dir', None)
                if writer is not None:
                    from common import append
                    append(writer / 'cmp_batches.jsonl', stats)
                    if self.calls <= 3:
                        append(writer / 'gradient_checks.jsonl', check)
            return total, parts['items']

    return CMPCriterion


def validate_cmp_objective(c, mode):
    variant = c['method_variant']
    if variant not in CMP_ARMS or c['campaign_arm'] != variant:
        raise ValueError('Unplanned comparison arm identity')
    if c['input_mode'] != 'rgb':
        raise ValueError('Comparison arms deploy RGB input')
    if any(c.get(key) is not None for key in ('teacher', 'reference', 'rgb_target_teacher')):
        raise ValueError('Legacy teacher/reference slots stay empty; the frozen teacher is fusion_target_teacher')
    if c.get('fusion_target_teacher') != TEACHER or c.get('teacher_sha256') != {TEACHER: TEACHER_SHA256}:
        raise ValueError('Comparison arms bind exactly the registered Fusion11n teacher')
    if not any(e['path'] == TEACHER and e['sha256'] == TEACHER_SHA256 for e in c['frozen_inputs']):
        raise ValueError('Teacher checkpoint must be a frozen input')
    if c.get('calibration_receipt') or c.get('evidence_dependencies'):
        raise ValueError('Comparison arms use no calibration or evidence dependency')
    if c['target_source'] != 'frozen_fusion_teacher_' + CMP_ARMS[variant]['kd']:
        raise ValueError('Comparison target source differs')
    if c['localization_coefficient'] != 0 or c['classification_coefficient'] != 0:
        raise ValueError('Comparison arms have no GU/FU-style auxiliary coefficients')
    if c.get('cmp_contract') != CMP_CONTRACT:
        raise ValueError('Comparison contract changed')
    if c.get('kd_weight') != KD_WEIGHTS[variant]:
        raise ValueError('KD weight differs from pre-registration')


def verify_cmp_canary(criterion, cfg):
    """Canary acceptance from evidence the criterion actually emitted."""
    rows = criterion.gradient_checks
    if criterion.selected_total <= 0 or not rows:
        raise AssertionError('No comparison batches or gradient evidence')
    if not any(r['weighted_kd_parameter_gradient_l2'] > 0 for r in rows):
        raise AssertionError('KD produced no weighted parameter gradient')
    if any(r['teacher_training'] or r['teacher_requires_grad'] for r in rows):
        raise AssertionError('Teacher was not frozen during training')
    return dict(status='PASS', checks=len(rows), kd_kind=criterion.kind,
                advisory_weighted_kd_to_native_ratio=[r['weighted_kd_to_native_ratio'] for r in rows])


def float64_composition(criterion, model, batch, k=FLOAT64_IMAGES):
    """grad(native + w * kd) == grad(native) + w * grad(kd) on float64 copies of the student, the teacher and k images.
    Ultralytics' native loss builds target tensors with the default dtype, so it is float64 for the replay only."""
    sub = engineering_subbatch(batch, k)
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        student = copy.deepcopy(model).double().train()
        probe = copy.copy(criterion)
        probe.teacher = copy.deepcopy(criterion.teacher).double().eval()
        if hasattr(student, 'init_criterion'):
            probe.native = student.init_criterion()
        b = {key: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for key, v in sub.items()}
        params = tuple(p for p in student.parameters() if p.requires_grad)
        parts = probe.components(student(b['img']), b)

        def grads(loss):
            return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)

        return composition_within_tolerance(grads(probe.total(parts)), (grads(parts['native_total']), grads(parts['batch_kd'])),
                                            (1., probe.w), rtol=1e-9, atol=1e-12)
    finally:
        torch.set_default_dtype(saved)


def engineering_checks(criterion, model, forward, batch, rtol=3e-4, atol=3e-5):
    """One real batch: frozen teacher on the pixel-mean view, zero weight, composition (float64), detach, gradient support."""
    with engineering_numerics() as numerics:
        result = _engineering_checks(criterion, model, forward, batch, rtol, atol)
    result['numerics'] = {k: v for k, v in numerics.items() if k != 'restored_after'}
    return result


def _engineering_checks(criterion, model, forward, batch, rtol, atol):
    checks = {}
    params = tuple(p for p in model.parameters() if p.requires_grad)
    teacher = criterion.teacher
    optimizer = getattr(criterion.trainer, 'optimizer', None)
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g['params']} if optimizer is not None else set()
    student_ids = {id(p) for p in model.parameters()}
    checks['teacher_frozen_eval_no_grad'] = not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    checks['teacher_outside_optimizer_and_student'] = not any(id(p) in in_optimizer or id(p) in student_ids
                                                              for p in teacher.parameters())
    fused = (batch['img'] + batch['strong_img']) * .5
    teacher_bn = bn_state_digest(teacher)
    with torch.no_grad():
        direct = raw_prediction(teacher(fused))
        rgb_only = raw_prediction(teacher(batch['img']))
    prediction = forward(batch['img'])
    parts = criterion.components(prediction, batch)
    checks['teacher_bn_unchanged'] = bn_state_digest(teacher) == teacher_bn
    checks['teacher_reads_pixelmean_view'] = (torch.equal(parts['teacher']['scores'], direct['scores'])
                                              and not torch.equal(direct['scores'], rgb_only['scores']))
    checks['teacher_outputs_detached'] = not any(t.requires_grad for t in
                                                 [parts['teacher']['scores'], parts['teacher']['boxes'], *parts['teacher']['feats']])

    def grads(loss):
        return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)

    def same(left, right):
        return all((a is None and z is None) or (a is not None and z is not None and torch.equal(a, z))
                   for a, z in zip(left, right))

    g_native = grads(parts['native_total'])
    checks['native_gradient_repeat_bitwise'] = same(g_native, grads(parts['native_total']))
    g_kd = grads(parts['batch_kd'])
    zero = parts['native_total'] + 0. * parts['batch_kd']
    checks['zero_weight_exact_loss'] = torch.equal(zero, parts['native_total'])
    checks['zero_weight_exact_parameter_gradients'] = same(g_native, grads(zero))
    _, worst32 = composition_within_tolerance(grads(criterion.total(parts)), (g_native, g_kd), (1., criterion.w), rtol, atol)
    passed64, worst64 = float64_composition(criterion, model, batch)
    checks['parameter_gradient_composition_float64'] = passed64
    checks['kd_nonzero_parameter_gradient'] = any(g is not None and bool((g != 0).any()) for g in g_kd)
    if criterion.kind == 'logit':
        box = box_branch_parameter_ids(model)
        checks['logit_no_gradient_on_box_branch'] = all(g is None or bool((g == 0).all())
                                                        for p, g in zip(params, g_kd) if id(p) in box)
        self_kd, _ = dense_privileged_kd(parts['student']['scores'], parts['student']['scores'])
        checks['logit_exact_zero_for_identical_scores'] = float(self_kd.detach()) == 0.
    elif criterion.kind == 'cwd':
        head = detect_head_parameter_ids(model)
        checks['cwd_no_gradient_on_detect_head'] = all(g is None or bool((g == 0).all())
                                                       for p, g in zip(params, g_kd) if id(p) in head)
        self_kd = cwd_loss(parts['student']['feats'], [f.detach() for f in parts['student']['feats']])
        checks['cwd_exact_zero_for_identical_features'] = float(self_kd.detach()) == 0.
    else:
        cls = classification_head_parameter_ids(model)
        checks['ld_no_gradient_on_classification_head'] = all(g is None or bool((g == 0).all())
                                                              for p, g in zip(params, g_kd) if id(p) in cls)
        self_kd, _ = dense_geometry_kd(parts['student']['boxes'], parts['student']['boxes'], parts['fg_mask'],
                                       temperature=LD_TEMPERATURE)
        checks['ld_exact_zero_for_identical_boxes'] = float(self_kd.detach()) == 0.
        checks['ld_has_foreground_anchors'] = (parts['foreground_anchors'] or 0) > 0
    return dict(checks=checks, composition_worst_ratio=worst64, composition_worst_ratio_float32_advisory=worst32,
                kd=float(parts['kd'].detach()), native_total=float(parts['native_total'].detach()))
