"""生成三份 PVC 配置: 以 GU_s42 蒸馏配置为基底, 换目标/权重/合同, 重绑授权与准入回执。"""
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from pvc_criterion import PVC_CONTRACT  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


base = yaml.safe_load((ROOT / "configs" / "night_GU_s42_distill.yaml").read_text(encoding="utf-8"))
DROP_HINTS = ("calibration", "night_FUSIONNU", "night_RGBU", "authorization", "model_policy")

for variant, kd in (("PVC", 0.5), ("PVC_SHUF", 0.5), ("PVC_NOKD", 0.0)):
    c = dict(base)
    c.update(
        method_id="DRONE-UNION-3ARM-20261010-night_%s_s42" % variant,
        method_variant=variant, campaign_arm=variant, arm=variant,
        run_identity="night_%s_s42" % variant,
        source="pvc_v1_privileged_view_cotraining",
        target_source="privileged_view_self", input_mode="rgb",
        teacher=None, reference=None, rgb_target_teacher=None, fusion_target_teacher=None,
        teacher_sha256={}, classification_coefficient=0, localization_coefficient=0,
        auxiliary_contract=None, teacher_forward_batch=None,
        calibration_receipt=None, calibration_sha256=None,
        privileged_native_weight=0.5, kd_weight=kd,
        pvc_contract=dict(PVC_CONTRACT),
        execution_version="pvc_v1_90",
        evidence_dependencies=[],
    )
    keep = [e for e in c.get("frozen_inputs", [])
            if not any(h in e["path"] for h in DROP_HINTS)]
    for rel in ("contract/authorization_pvc_20261011.json", "contract/model_policy_receipt_pvc.json"):
        p = ROOT / rel
        keep.append({"path": str(p), "sha256": sha(p)})
    c["frozen_inputs"] = keep
    policy = ROOT / "contract" / "model_policy_receipt_pvc.json"
    authz = ROOT / "contract" / "authorization_pvc_20261011.json"
    c["model_policy_receipt"] = str(policy)
    c["model_policy_receipt_sha256"] = sha(policy)
    c["formal_training_authorized"] = True
    c["authorization_source"] = {"path": str(authz), "sha256": sha(authz)}
    c["engineering_receipt"] = str(ROOT / "engineering" / "pvc_v1_engineering.json")
    out = ROOT / "configs" / ("night_%s_s42.yaml" % variant)
    out.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("written", out, "frozen_inputs=", len(keep))
print("CONFIGS DONE")
