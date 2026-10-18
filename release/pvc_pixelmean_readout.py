"""DIAGNOSTIC readout: a PVC E200 EMA evaluated on the pixelmean privileged view.

DESIGN.md §5 / R6: tells whether the shared network still holds a strong privileged
view (reference FUSIONNU42). Never a gate. Same validator, roster, union GT and
pixel expression as the FUSIONNU readout; only the PVC config's input_mode is
replaced for this process, so the scientific hash and the formal readout are untouched.
"""
import argparse
from pathlib import Path

import readout
from common import read, sha, write

from pvc_criterion import PVC_ARMS  # 单一来源，避免新增臂漏进来


def main(a):
    validated = readout.config

    def pixelmean_config(path, mode):
        c = dict(validated(path, mode))
        if c['method_variant'] not in PVC_ARMS or c['input_mode'] != 'rgb':
            raise ValueError('Pixelmean diagnostic is defined for PVC RGB-deployed arms only')
        c['input_mode'] = 'pixelmean'
        return c

    run = a.checkpoint.resolve().parent.parent
    completion = read(run / 'completion_receipt.json')
    if (completion['status'], completion['last_epoch'], completion['epochs_configured']) != ('training_completed', 200, 200):
        raise ValueError('Diagnostic readout needs the natural E200 endpoint')
    if a.checkpoint.resolve() != Path(completion['checkpoint']).resolve():
        raise ValueError('Checkpoint is not the bound E200 export')
    readout.config = pixelmean_config
    readout.main(argparse.Namespace(checkpoint=a.checkpoint, config=a.config, output=a.output,
                                    mode='canary', population_scope='subset', gt='union'))
    metrics = read(a.output / 'metrics.json')
    if metrics['status'] != 'DIAGNOSTIC_PASS' or metrics['input_mode'] != 'pixelmean':
        raise ValueError('Diagnostic readout did not use the pixelmean view')
    write(a.output / 'diagnostic_binding.json', dict(
        status='DIAGNOSTIC_PASS', role='pixelmean_view_readout_not_a_gate', arm=metrics['arm'],
        completion=str(run / 'completion_receipt.json'), completion_sha256=sha(run / 'completion_receipt.json'),
        checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint),
        metrics_sha256=sha(a.output / 'metrics.json'), AP50=metrics['AP50'], mAP50_95=metrics['mAP50_95']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
