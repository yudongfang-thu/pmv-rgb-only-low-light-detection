"""Independent evaluator process: immutable EMA export; no train RNG or buffers changed."""
import copy, subprocess, time
import torch
from common import *
import checkpoint as full

def evaluate_epoch(t,out,cfg):
    root=out/'periodic'/f'epoch_{t.epoch+1:04d}'
    root.mkdir(parents=True,exist_ok=False)
    rng=full.rng_state();model_digest=full.digest(t.model.state_dict());ema_digest=full.digest(t.ema.ema.state_dict())
    ema=copy.deepcopy(t.ema.ema).float().cpu()
    if hasattr(ema,'criterion'):ema.criterion=None
    for part in ema.modules():
        part._forward_hooks.clear();part._forward_pre_hooks.clear();part._backward_hooks.clear()
    path=root/'ema.pt'
    full.save(path,dict(epoch=t.epoch,model=None,ema=ema,updates=t.ema.updates,train_args=vars(t.args),version=cfg['ultralytics_version']))
    del ema
    command=[str(PYTHON),'-B',str(HERE/'readout.py'),'--checkpoint',str(path),'--config',str(out/'protocol_config.yaml'),'--output',str(root/'readout'),'--mode','canary']
    started=time.monotonic()
    with (root/'eval.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    checks=dict(rng=full.digest(rng)==full.digest(full.rng_state()),model=model_digest==full.digest(t.model.state_dict()),ema=ema_digest==full.digest(t.ema.ema.state_dict()))
    write(root/'isolation_receipt.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,returncode=result.returncode,seconds=time.monotonic()-started))
    if not all(checks.values()):raise AssertionError('Evaluator contaminated training state')
    # Evaluator failure is recorded without stopping a running training worker.
    append(out/'periodic_status.jsonl',dict(epoch=t.epoch+1,returncode=result.returncode,path=str(root),seconds=time.monotonic()-started))
