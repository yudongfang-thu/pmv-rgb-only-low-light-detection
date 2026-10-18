"""Config validation branch for PVC arms, to be merged into common.config().

Native recipe and data/population/policy/authorization checks of config() stay
unchanged; this replaces only the variant-specific objective block:

    if variant in NATIVE_ARMS: ...
    elif variant in PVC_ARMS: validate_pvc_objective(c, mode)
    else: ...  # GU/FU

native_recipe.ARMS additions: {'PVC': 'rgb', 'PVC_SHUF': 'rgb', 'PVC_NOKD': 'rgb'}
"""
from pvc_criterion import PVC_ARMS, PVC_CONTRACT, PRIVILEGED_NATIVE_WEIGHT, KD_WEIGHT, GEO_WEIGHT


def validate_pvc_objective(c, mode):
    variant = c['method_variant']
    if variant not in PVC_ARMS or c['campaign_arm'] != variant:
        raise ValueError('Unplanned PVC arm identity')
    if c['input_mode'] != 'rgb':
        raise ValueError('PVC deploys RGB input')
    if any(c.get(key) is not None for key in ('teacher', 'reference', 'rgb_target_teacher', 'fusion_target_teacher')):
        raise ValueError('PVC binds no frozen model')
    if c.get('teacher_sha256') or c.get('calibration_receipt') or c.get('evidence_dependencies'):
        raise ValueError('PVC uses no teacher/calibration evidence')
    if c['target_source'] != 'privileged_view_self':
        raise ValueError('PVC target source differs')
    if c['localization_coefficient'] != 0 or c['classification_coefficient'] != 0:
        raise ValueError('PVC has no GU/FU-style auxiliary coefficients')
    if c.get('pvc_contract') != PVC_CONTRACT:
        raise ValueError('PVC contract changed')
    if c.get('privileged_native_weight') != PRIVILEGED_NATIVE_WEIGHT:
        raise ValueError('Privileged native weight differs from pre-registration')
    expected_kd = KD_WEIGHT if PVC_ARMS[variant]['kd'] else 0.
    if c.get('kd_weight') != expected_kd:
        raise ValueError('KD weight differs from pre-registration')
    expected_geo = GEO_WEIGHT if any('geo' in t for t in PVC_ARMS[variant].get('kd_types', ())) else 0.
    if c.get('geo_weight', 0.) != expected_geo:
        raise ValueError('Geometry KD weight differs from pre-registration')
    if c.get('batch') is not None and c['batch'] < 2:
        raise ValueError('Shuffled control requires batch >= 2')
