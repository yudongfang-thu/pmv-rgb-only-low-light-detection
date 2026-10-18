"""Five-class BCE on every native TAL receiver of a frozen RGB reference.

Object-balanced direct classification control: temperature 1, no quality or
teacher-advantage filter, no localization, ordering or feature auxiliary loss.
"""
import torch
import torch.nn.functional as F
from common import append, write


def raw_prediction(prediction):
    prediction = prediction[1] if isinstance(prediction, tuple) else prediction
    if not isinstance(prediction, dict) or not {'scores', 'boxes', 'feats'} <= set(prediction):
        raise TypeError('Bound YOLO11 one-to-many raw dictionary required')
    return prediction


def per_object_aux_bce(student_logits, targets, object_inverse, object_count):
    """Kx5: class mean, within-object position mean, then object mean.

    object_inverse identifies unique (batch image index, native GT index)
    pairs. Empty support gives an attached zero with zero auxiliary gradient.
    """
    if student_logits.ndim != 2 or student_logits.shape[1] != 5 or targets.shape != student_logits.shape:
        raise ValueError('Auxiliary class tensors must have identical Kx5 shape')
    if student_logits.shape[0] != object_inverse.numel(): raise ValueError('Receiver identity length differs')
    if object_count == 0:
        if student_logits.shape[0] != 0: raise ValueError('Positions exist without receiving objects')
        return student_logits.sum() * 0.
    per_position = F.binary_cross_entropy_with_logits(student_logits.float(), targets.detach().float(), reduction='none').mean(1)
    sums = per_position.new_zeros(object_count).index_add(0, object_inverse, per_position)
    counts = per_position.new_zeros(object_count).index_add(0, object_inverse, torch.ones_like(per_position))
    return (sums / counts).mean()


def parameter_gradient_norm(loss, parameters):
    """Unscaled parameter L2 norm; does not modify parameter .grad."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    terms = [g.detach().float().square().sum() for g in gradients if g is not None]
    return torch.stack(terms).sum().sqrt() if terms else loss.detach().new_zeros(())


def aligned_box_iou(boxes, targets):
    intersection = (torch.minimum(boxes[:, 2:], targets[:, 2:]) - torch.maximum(boxes[:, :2], targets[:, :2])).clamp_min(0).prod(1)
    a = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    b = (targets[:, 2:] - targets[:, :2]).clamp_min(0).prod(1)
    return intersection / (a + b - intersection + 1e-7)


def make_type(rt, cfg):
    class DirectCriterion:
        def __init__(self, native, teacher, reference, config, arm, trainer, sanity):
            self.native, self.teacher, self.reference = native, teacher, reference
            self.cfg, self.trainer, self.sanity = cfg, trainer, sanity
            self.method_variant, self.target_source = cfg['method_variant'], cfg['target_source']
            self.calls = self.selected_total = self.fallback_batches = 0
            self.gradient_checks, self.last_stats = [], {}
            self.grid_contract_verified = False
            self._verified_geometries = set()
            self.forward_batch = int(cfg.get('teacher_forward_batch', 4))
            if self.forward_batch < 1: raise ValueError('Invalid frozen eval forward microbatch')
            expected = {'G': 'onehot', 'R': 'rgb_probability', 'F': 'fusion_probability', 'GU': 'onehot', 'FU': 'fusion_probability'}
            if self.target_source != expected[self.method_variant]: raise ValueError('Arm and target source differ')
            self.target_models = {self.method_variant: teacher} if self.method_variant in ('R', 'F') else {}
            self.rgb_teacher = self.target_models.get('R')
            self.fusion_teacher = self.target_models.get('F')
            self.reference_native = reference.init_criterion()
            self._check_head(trainer.model, 'student'); self._check_head(reference, 'reference')
            for source, model in self.target_models.items(): self._check_head(model, source)
            write(trainer.save_dir / 'direct_classification_contract.json', dict(
                reference=cfg['reference'], reference_role='night_N_s42_E200_FP32_EMA',
                class_names=trainer.model.names, levels=['P3', 'P4', 'P5'], strides=[8, 16, 32],
                receiver='reference_native_TAL_on_augmented_RGB_and_native_RGB_GT',
                object_identity=['batch_image_index', 'native_RGB_GT_index_within_image'],
                target_source=self.target_source, temperature=1,
                reduction='mean_objects(mean_object_positions(mean_5_classes(BCEWithLogits)))',
                total='native_total + actual_batch_size * frozen_lambda_arm * auxiliary',
                frozen_teacher_forward_batch=self.forward_batch,
                frozen_teacher_forward_dtype='float32_eval_no_autocast',
                no_IR_GT_pairing=True, no_teacher_advantage_filter=True, no_quality_filter=True))

        def __deepcopy__(self, memo): return None

        def _check_head(self, model, role):
            head = model.model[-1]
            if (head.nc, head.reg_max, tuple(head.f), tuple(float(s) for s in head.stride)) != (5, self.native.reg_max, (16, 19, 22), (8., 16., 32.)):
                raise ValueError('P3/P4/P5 head/grid differs: ' + role)
            if head.end2end or model.names != self.trainer.model.names: raise ValueError('Head or class order differs: ' + role)
            if role != 'student' and any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise AssertionError('Frozen source is trainable or has gradients: ' + role)

        def _ensure_source(self, source):
            if source not in self.target_models:
                from scene_runtime import frozen_model
                import checkpoint as full
                key, role = ('rgb_target_teacher', 'RGBL') if source == 'R' else ('fusion_target_teacher', 'FUSIONL')
                rng = full.rng_state()
                try: model = frozen_model(self.cfg[key], self.cfg, self.trainer.model.names, role).to(self.trainer.device)
                finally: full.restore_rng(rng)
                self._check_head(model, source)
                self.target_models[source] = model
                if source == 'R': self.rgb_teacher = model
                else: self.fusion_teacher = model
            return self.target_models[source]

        def _geometry(self, raw, batch_size):
            from ultralytics.utils.tal import make_anchors
            shapes = tuple(tuple(f.shape[-2:]) for f in raw['feats'])
            if len(shapes) != 3 or raw['scores'].shape != (batch_size, 5, sum(h * w for h, w in shapes)):
                raise ValueError('Raw score layout differs from ordered P3/P4/P5 grids')
            if raw['boxes'].shape != (batch_size, 4 * self.native.reg_max, raw['scores'].shape[2]): raise ValueError('Raw box layout differs')
            anchors, strides = make_anchors(raw['feats'], self.native.stride, 0.5)
            return shapes, anchors, strides

        def _assert_geometry(self, left, right, source):
            if left[0] != right[0]:
                raise ValueError('Anchor order, spatial grids or stride differ')
            key = (source, left[0])
            if key not in self._verified_geometries:
                if not torch.equal(left[1].float(), right[1].float()) or not torch.equal(left[2].float(), right[2].float()):
                    raise ValueError('Anchor order, spatial grids or stride differ')
                self._verified_geometries.add(key)

        def _reference_raw(self, images, geometry):
            """Keep raw heads and shape proxies; release frozen backbone feats."""
            self.reference.eval()
            scores, boxes = [], []
            with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
                for start in range(0, images.shape[0], self.forward_batch):
                    chunk = images[start:start + self.forward_batch].float()
                    raw = raw_prediction(self.reference(chunk))
                    self._assert_geometry(geometry, self._geometry(raw, chunk.shape[0]), 'reference')
                    scores.append(raw['scores'].detach()); boxes.append(raw['boxes'].detach())
                    del raw
                # Native TAL/loss reads feature shape/device/dtype, not values.
                feats = [images.new_empty(1, 1, h, w).expand(images.shape[0], 1, h, w) for h, w in geometry[0]]
                return dict(scores=torch.cat(scores), boxes=torch.cat(boxes), feats=feats)

        def _receivers(self, batch, geometry):
            with torch.no_grad(), torch.autocast(device_type=batch['img'].device.type, enabled=False):
                raw = self._reference_raw(batch['img'], geometry)
                foreground, target_gt_indices, target_boxes, anchors, strides = self.reference_native.get_assigned_targets_and_loss(raw, batch)[0]
                image_indices, anchor_indices = foreground.nonzero(as_tuple=True)
                native_gt_indices = target_gt_indices[image_indices, anchor_indices].long()
                keys, inverse, counts = torch.unique(torch.stack((image_indices, native_gt_indices), 1), dim=0, sorted=True, return_inverse=True, return_counts=True)
                targets = torch.cat((batch['batch_idx'].reshape(-1, 1), batch['cls'].reshape(-1, 1), batch['bboxes']), 1)
                hw = torch.tensor(batch['img'].shape[-2:], device=batch['img'].device, dtype=raw['scores'].dtype)
                gt = self.reference_native.preprocess(targets, batch['img'].shape[0], hw[[1, 0, 1, 0]])
                labels = gt[image_indices, native_gt_indices, 0].long()
                return dict(foreground_mask=foreground.detach(), target_gt_indices=target_gt_indices.detach(),
                            image_indices=image_indices.detach(), native_gt_indices=native_gt_indices.detach(),
                            anchor_indices=anchor_indices.detach(), object_keys=keys.detach(),
                            object_inverse=inverse.detach(), position_counts=counts.detach(), labels=labels.detach(),
                            target_boxes=target_boxes[image_indices, anchor_indices].detach(),
                            object_count=keys.shape[0], position_count=image_indices.numel(),
                            anchor_points=anchors.detach(), stride_tensor=strides.detach())

        def _target(self, source, batch, geometry, support, diagnostics):
            """Gather all receivers while each FP32 teacher chunk is alive."""
            model = self._ensure_source(source); model.eval()
            images = batch['img'] if source == 'R' else (batch['img'] + batch['strong_img']) * .5
            index, anchor_index = support['image_indices'], support['anchor_indices']
            q = images.new_empty((support['position_count'], 5), dtype=torch.float32)
            decoded = images.new_empty((support['position_count'], 4), dtype=torch.float32) if diagnostics else None
            with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
                for start in range(0, images.shape[0], self.forward_batch):
                    chunk = images[start:start + self.forward_batch].float()
                    raw = raw_prediction(model(chunk))
                    shape, anchors, strides = self._geometry(raw, chunk.shape[0])
                    self._assert_geometry(geometry, (shape, anchors, strides), source)
                    selected = ((index >= start) & (index < start + chunk.shape[0])).nonzero(as_tuple=True)[0]
                    local_index, local_anchor = index[selected] - start, anchor_index[selected]
                    q[selected] = raw['scores'].permute(0, 2, 1)[local_index, local_anchor].sigmoid()
                    if diagnostics:
                        distributions = raw['boxes'].permute(0, 2, 1)[local_index, local_anchor]
                        decoded[selected] = self.native.bbox_decode(anchors[local_anchor], distributions.unsqueeze(0)).squeeze(0) * strides[local_anchor]
                    del raw
            detail = {}
            if diagnostics:
                labels = support['labels']; wrong = q.clone(); wrong.scatter_(1, labels[:, None], -1.)
                wrong_probability, wrong_class = wrong.max(1)
                detail = dict(probabilities=q.detach(), correct_class_probability=q.gather(1, labels[:, None]).squeeze(1).detach(),
                              strongest_wrong_probability=wrong_probability.detach(), strongest_wrong_class=wrong_class.detach(),
                              decoded_boxes=decoded.detach(), box_iou_with_assigned_RGB_GT=aligned_box_iou(decoded, support['target_boxes']).detach(),
                              all_receiving_positions_retained=True, quality_filter_applied=False)
            return q.detach(), detail

        def zero_receiver_check(self, scores_B_A_5):
            """Engineering-only empty support through the actual object reducer."""
            if scores_B_A_5.ndim != 3 or scores_B_A_5.shape[-1] != 5:
                raise ValueError('Expected BxAx5 student logits')
            empty = scores_B_A_5.reshape(-1, 5)[:0]
            inverse = torch.empty(0, dtype=torch.long, device=scores_B_A_5.device)
            return per_object_aux_bce(empty, empty.detach(), inverse, 0)

        def components(self, prediction, batch, all_sources=True, diagnostics=True):
            """Calibration graph with no counters/optimizer updates.

            native/native_cls and batch_aux already include actual batch size.
            Sources load lazily once; every diagnostic receiver is preserved.
            """
            student = raw_prediction(prediction)
            geometry = self._geometry(student, batch['img'].shape[0])
            _, losses, items = self.native.get_assigned_targets_and_loss(student, batch)
            b = batch['img'].shape[0]; native = losses * b
            support = self._receivers(batch, geometry)
            logits = student['scores'].permute(0, 2, 1)[support['image_indices'], support['anchor_indices']]
            aux, targets, details = {}, {}, {}
            _sources = ('G', 'R', 'F') if all_sources else (self.method_variant,)
            for source in _sources:
                _key = self.method_variant if source == 'G' and self.method_variant == 'GU' else source
                if source in ('G', 'GU'): q, detail = F.one_hot(support['labels'], num_classes=5).float().detach(), {}
                else: q, detail = self._target(source, batch, geometry, support, diagnostics)
                targets[source], details[_key] = q, detail
                aux[_key] = per_object_aux_bce(logits, q, support['object_inverse'], support['object_count'])
            self.grid_contract_verified = True
            return dict(native=native, native_total=native.sum(), native_cls=native[1], items=items,
                        aux=aux, batch_aux={s: b * v for s, v in aux.items()}, targets=targets,
                        support=support, diagnostics=details, student_logits=logits,
                        native_student_raw=student, batch_size=b)

        def __call__(self, prediction, batch):
            if 'strong_img' not in batch: return self.native(prediction, batch)
            self.calls += 1
            due = self.calls <= 3 or self.calls % self.cfg['log_every_batches'] == 0
            parts = self.components(prediction, batch, all_sources=False, diagnostics=due)
            weight = float(self.cfg['classification_coefficient']); unweighted = parts['batch_aux'][self.method_variant]
            total = parts['native_total'] if weight == 0. else parts['native_total'] + weight * unweighted
            if not bool(torch.isfinite(total)): raise FloatingPointError('Nonfinite direct classification total')
            self.selected_total += parts['support']['object_count']
            if due:
                params = tuple(p for p in self.trainer.model.parameters() if p.requires_grad)
                norms = torch.stack((parameter_gradient_norm(parts['native_cls'], params), parameter_gradient_norm(unweighted, params)))
                if not bool(torch.isfinite(norms).all()): raise FloatingPointError('Nonfinite training gradient dose')
                values = norms.cpu().tolist()
                frozen = (self.reference, *self.target_models.values())
                check = dict(batch=self.calls, epoch=int(self.trainer.epoch), native_cls_parameter_gradient_l2=values[0],
                             batch_aux_parameter_gradient_l2=values[1], weighted_aux_parameter_gradient_l2=weight * values[1],
                             weighted_aux_to_native_cls_ratio=weight * values[1] / values[0] if values[0] > 0 else None,
                             coefficient=weight, actual_batch_size=parts['batch_size'],
                             teacher_has_grad=any(p.grad is not None for m in self.target_models.values() for p in m.parameters()),
                             reference_has_grad=any(p.grad is not None for p in self.reference.parameters()),
                             frozen_models_require_grad=any(p.requires_grad for m in frozen for p in m.parameters()),
                             selected_count=parts['support']['object_count'], receiver_positions=parts['support']['position_count'])
                if check['teacher_has_grad'] or check['reference_has_grad'] or check['frozen_models_require_grad']:
                    raise AssertionError('Frozen source acquired gradients')
                if self.calls <= 3: self.gradient_checks.append(check)
                append(self.trainer.save_dir / 'gradient_checks.jsonl', check)
                stats = dict(check, native_total=float(parts['native_total'].detach()),
                             loss_unweighted=float(parts['aux'][self.method_variant].detach()), batch_aux_total=float(unweighted.detach()),
                             weighted_aux_total=float(unweighted.detach()) * weight, total_loss=float(total.detach()),
                             optimizer_updates=self.trainer.real_updates, object_position_counts=parts['support']['position_counts'].cpu().tolist())
                self.last_stats = stats; append(self.trainer.save_dir / 'kd_batches.jsonl', stats)
            return total, parts['items']

    return DirectCriterion
