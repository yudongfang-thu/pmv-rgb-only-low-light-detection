"""Native pair v3 binding; accepted labels precede any GPU construction."""
import csv
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from native_recipe import RECIPE, ARMS, INITIAL_YOLO11N_SHA256
NATIVE_ARMS = ('RGBU', 'IRU', 'FUSIONNU', 'RGBU_FULL', 'RGBU_H')


def _pvc_arm_names():
    """PVC 臂清单的唯一来源是 pvc_criterion.PVC_ARMS —— 不再在本文件里硬编码臂名。"""
    from pvc_criterion import PVC_ARMS
    return PVC_ARMS


def _cmp_arm_names():
    """xm_full_r2: the comparison arms come from cmp_criterion.CMP_ARMS."""
    from cmp_criterion import CMP_ARMS
    return CMP_ARMS

KD_ARMS = ('GU', 'FU')

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PMV_ROOT', HERE.parent))
PROJECT = Path(os.environ.get('PMV_PROJECT', ROOT))
# Training outputs must stay under this directory (see README).
CAMPAIGN = Path(os.environ.get('PMV_CAMPAIGN', ROOT))
REFERENCE = Path(os.environ.get('PMV_REFERENCE', ROOT / 'reference'))
PYTHON = Path(os.environ.get('PMV_PYTHON', sys.executable))


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value, exclusive=True):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    target = p if exclusive else p.with_name(p.name + '.tmp.' + str(os.getpid()))
    with target.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    if not exclusive:
        target.replace(p)


def append(p, value):
    with Path(p).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def clone(fn, **bindings):
    result = types.FunctionType(fn.__code__, dict(fn.__globals__, **bindings), fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


def runtime():
    sys.path.insert(0, str(REFERENCE))
    import runtime as rt
    if Path(rt.__file__).resolve() != REFERENCE / 'runtime.py':
        raise RuntimeError('Wrong paired runtime')
    return rt


def scientific_hash(cfg):
    ignored = {'canary_receipt', 'recovery_receipt', 'readiness_receipt'}
    return hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k not in ignored}, sort_keys=True).encode()).hexdigest()


def run_key(cfg):
    return f"{cfg['subset']}_{cfg['campaign_arm']}_s{cfg['seed']}"


def source_binding():
    # Reuse trainer/package binding without unrelated previous method roots.
    required = [REFERENCE / 'runtime.py',
        REFERENCE / 'tracked_pair_data.py',
        REFERENCE / 'legacy_oev1/train_object_evidence.py',
        REFERENCE / 'legacy_oev1/paired_rgbir_data.py',
        REFERENCE / 'legacy_oev1/object_evidence_loss.py']
    spec = importlib.util.find_spec('ultralytics')
    if spec is None or not spec.origin:
        raise RuntimeError('Missing pinned ultralytics')
    paths = set(required); paths.update(HERE.glob('*.py'))
    for module in list(sys.modules.values()):
        source = getattr(module, '__file__', None)
        if source:
            p = Path(source).resolve()
            if p.suffix == '.py' and p.is_relative_to(REFERENCE):
                paths.add(p)
    package = Path(spec.origin).parent
    paths.update(package.rglob('*.py')); paths.update(package.rglob('*.yaml'))
    return [{'path': str(p), 'sha256': sha(p)} for p in sorted(paths)]


def check_binding(rows):
    for row in rows:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Frozen input/source changed: ' + row['path'])


def manifest_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def data_contract(cfg):
    if sha(cfg['data_contract']) != cfg['data_contract_sha256']:
        raise ValueError('Data contract changed')
    d = read(cfg['data_contract'])
    if d['status'] != 'READY_FOR_NATIVE_TRAINING' or d['training_labels_accepted'] is not True:
        raise ValueError('Fusion label acceptance is pending')
    if d['official_test_accessed'] is not False:
        raise ValueError('Official test is outside this batch')
    check_binding(d['frozen_inputs'])
    return d


def population(cfg):
    d = data_contract(cfg)
    if sha(cfg['population']) != cfg['population_sha256']:
        raise ValueError('Scene population changed')
    p = read(cfg['population'])
    if p['status'] != 'FROZEN_SCENE_POPULATION' or p.get('official_test_accessed', True):
        raise ValueError('Scene population is not frozen train/dev')
    check_binding([p['scene_manifest']])
    selected = [r for r in manifest_rows(p['scene_manifest']['path']) if r['scene'] in ('Night', 'Dark night')]
    native = manifest_rows(d['train_manifest']) + manifest_rows(d['val_manifest'])
    scene_keys = {(r['split'], r['stem']) for r in selected}
    exclusions = d.get('qc_exclusions', [])
    excluded_keys = {(r['split'], r['stem']) for r in exclusions}
    if len(excluded_keys) != len(exclusions) or not excluded_keys.issubset(scene_keys):
        raise ValueError('QC exclusions are duplicate or outside source population')
    if any(not r.get('reason') for r in exclusions):
        raise ValueError('QC exclusions need attributable reasons')
    keys = {(r['split'], r['stem']) for r in native}
    if keys != scene_keys - excluded_keys or len(keys) != len(native):
        raise ValueError('Native manifest differs from source population minus explicit QC exclusions')
    for split, number in [('train', d['train_images']), ('val', d['val_images'])]:
        if sum(r['split'] == split for r in native) != number:
            raise ValueError('Native manifest count differs: ' + split)
    required = {'rgb_image', 'ir_image', 'rgb_label', 'ir_label', 'union_label', 'union_label_sha256'}
    if any(not required.issubset(r) or any(not r[k] for k in required) for r in native):
        raise ValueError('Native manifest lacks bound images or union labels')
    sub = dict(p['subsets']['night'])
    sub.update(union_yaml=d['union_student_yaml'], union_val_gt=d['union_val_gt'],
        rgb_yaml=d['rgb_evaluation_yaml'], rgb_val_gt=d['source_rgb_val_gt'],
        train_images=d['train_images'], val_images=d['val_images'])
    return p, sub, native


def config(path, mode):
    import yaml
    c = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    variant = c['method_variant']
    if ARMS.get(variant) != c['input_mode'] or c['campaign_arm'] != variant or c['run_identity'] != run_key(c):
        raise ValueError('Unplanned native arm/input/identity')
    for key, value in RECIPE.items():
        if c[key] != value:
            raise ValueError('Matched native recipe changed: ' + key)
    if c['subset'] != 'night' or c['model_scale'] != 'n' or c['model_architecture'] != 'yolo11n':
        raise ValueError('Native endpoint must be night YOLO11n')
    if variant in NATIVE_ARMS:
        if any(c[key] is not None for key in ('teacher', 'reference', 'rgb_target_teacher')):
            raise ValueError('Native endpoint cannot bind any auxiliary teacher')
        if c.get('teacher_sha256') or c.get('calibration_receipt') or c.get('evidence_dependencies'):
            raise ValueError('Native endpoints do not use teacher/calibration evidence')
        if c['target_source'] != 'native_union' or c['classification_coefficient'] != 0 or c['localization_coefficient'] != 0:
            raise ValueError('Native objective differs')
    elif variant in _cmp_arm_names():
        from cmp_criterion import validate_cmp_objective
        validate_cmp_objective(c, mode)
    elif variant in _pvc_arm_names():
        from common_pvc_patch import validate_pvc_objective
        validate_pvc_objective(c, mode)
    else:
        # 蒸馏臂(GU/FU): 移植自 v2 common.py 的 direct classification 校验
        import math as _math
        if c['localization_coefficient'] != 0:
            raise ValueError('Unplanned localization KD')
        coef = c['classification_coefficient']
        if not isinstance(coef, (int, float)) or not _math.isfinite(coef) or coef <= 0:
            raise ValueError('Nonpositive/nonfinite auxiliary coefficient')
        _ctu = dict(reception='frozen_N42_native_TAL_union_GT',
                    target='independent_sigmoid_T1_or_GT_onehot',
                    reduction='objects_then_positions_then_5classes', native_retained=True,
                    calibration_batches=64, initial_gradient_ratio=.10)
        if c['auxiliary_contract'] != _ctu:
            raise ValueError('Direct classification contract changed')
        if c['target_source'] != {'GU': 'onehot', 'FU': 'fusion_probability'}[variant]:
            raise ValueError('Target source differs')
        if c['teacher_forward_batch'] != 4:
            raise ValueError('Frozen teacher forward chunk changed')
        if not c.get('teacher') or not c.get('reference'):
            raise ValueError('Distillation arm requires bound teacher and reference')
        ts = c.get('teacher_sha256') or {}
        for key in ('teacher', 'reference'):
            if str(c[key]) not in ts:
                raise ValueError('Frozen source sha not declared: ' + key)
        receipt = c.get('calibration_receipt')
        if receipt:
            calibrated = read(receipt)
            if sha(receipt) != c['calibration_sha256'] or calibrated['status'] != 'PASS':
                raise ValueError('Calibration receipt differs')
            if calibrated['coefficients'][variant] != coef:
                raise ValueError('Calibrated coefficient differs')
        elif mode == 'train':
            raise ValueError('Formal training requires 64 batch calibration')
    if sha(c['model']) != INITIAL_YOLO11N_SHA256 or c['model_sha256'] != INITIAL_YOLO11N_SHA256:
        raise ValueError('Common initial YOLO11n differs')
    check_binding(c['frozen_inputs'])
    d = data_contract(c)
    expected = dict(student_data_yaml=d['union_student_yaml'], privileged_data_yaml=d['ir_yaml'], paired_train_mapping=d['pair_mapping'])
    if c['paths'] != expected or c['data_mode'] != 'union':
        raise ValueError('Native union/paired data paths differ')
    for name in ('union_student_yaml', 'rgb_evaluation_yaml', 'ir_yaml'):
        ds = yaml.safe_load(Path(d[name]).read_text())
        if 'test' in ds or len(ds['names']) != 5:
            raise ValueError('Native data must contain five classes and train/dev only')
    if (c['expected_train_images'], c['expected_val_images'], c['expected_train_batches']) != (d['train_images'], d['val_images'], (d['train_images'] + 31) // 32):
        raise ValueError('Native train/dev population differs')
    if d['train_images'] <= 0 or d['val_images'] <= 0 or d['union_val_gt'] <= 0:
        raise ValueError('Native training/dev population cannot be empty')
    if c['expected_val_gt'] != d['union_val_gt']:
        raise ValueError('Native union GT count differs')
    expected_eval = dict(endpoint='fixed_E200_last_fp32_ema', epoch=200, split='val',
        primary_subset='night', primary_gt='union', secondary_gt='rgb',
        expected_val_images=d['val_images'], expected_val_gt=d['union_val_gt'],
        metric_units='fraction_0_to_1', difference_units='percentage_points',
        diagnostic_every_epochs=5, fn_fp_conf=0.25, fn_fp_iou=0.5)
    if c['evaluation_contract'] != expected_eval:
        raise ValueError('Native fixed endpoint/readout differs')
    population(c)
    gate = read(c['model_policy_receipt'])
    if sha(c['model_policy_receipt']) != c['model_policy_receipt_sha256'] or gate['status'] != 'MODEL_POLICY_OK_NO_LAUNCH_AUTHORIZATION' or gate.get('launch_authorized') is not False:
        raise ValueError('Independent model policy admission has not passed')
    # Admission is not dispatch authority. Record existing user authorization
    # separately, with accepted labels bound, before GPU engineering/training.
    if c['formal_training_authorized'] is not True or not c['authorization_source']:
        raise ValueError('Native training/engineering authorization is pending')
    check_binding([c['authorization_source']])
    if mode == 'train':
        frozen = read(HERE / 'RELEASE_FROZEN_v3.json')
        binding = HERE / 'RELEASE_BINDING_v3.json'
        if frozen['status'] != 'FROZEN_AFTER_ENGINEERING_PASS' or frozen['source'] != str(HERE) or frozen['source_binding_sha256'] != sha(binding):
            raise ValueError('Native v3 release is not frozen')
        check_binding(read(binding))
        accepted = read(c['engineering_receipt'])
        if accepted['status'] != 'PASS' or not all(accepted['checks'].values()) or accepted['science_hashes'][run_key(c)] != scientific_hash(c):
            raise ValueError('Native engineering acceptance differs')
    return c
