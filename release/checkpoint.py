"""Full-precision epoch-boundary state; deterministic worker regeneration."""
import copy
import hashlib
import inspect
import random
import textwrap
from pathlib import Path
import numpy as np
import torch

SCHEMA='subset-full-state-v1'

def digest(value):
    if isinstance(value,torch.Tensor):
        a=value.detach().cpu().contiguous();return (str(a.dtype),list(a.shape),hashlib.sha256(a.numpy().tobytes()).hexdigest())
    if isinstance(value,np.ndarray):return (str(value.dtype),list(value.shape),hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):digest(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [digest(v) for v in value]
    return value

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])

def restart_epoch_loader(loader,seed,epoch):
    # All uninterrupted and resumed arms rebuild workers at every epoch.
    # Prefetched batches from the previous epoch are deliberately discarded.
    generator=torch.Generator().manual_seed(seed+1_000_003*epoch)
    sampler_generator=torch.Generator().manual_seed(seed+1_000_003*epoch+97)
    if not hasattr(loader.sampler,'generator'):raise RuntimeError('Expected random sampler with generator')
    loader.generator=generator;loader.sampler.generator=sampler_generator
    loader.reset()

def install_resume_loop(trainer):
    original=type(trainer)._do_train
    text=textwrap.dedent(inspect.getsource(original))
    if text.count('last_opt_step = -1')!=2:raise RuntimeError('Native loop contract changed')
    text=text.replace('last_opt_step = -1','last_opt_step = getattr(self, "_resume_last_opt_step", -1)',1)
    needle='self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start'
    if text.count(needle)!=1:raise RuntimeError('Native gradient initialization changed')
    text=text.replace(needle,'if not getattr(self, "_full_state_restored", False):\n        self.optimizer.zero_grad()')
    namespace=dict(original.__globals__);exec(compile(text,'<subset_resume_loop>','exec'),namespace)
    trainer._do_train=namespace['_do_train'].__get__(trainer,type(trainer))

def capture(t,science,sources,complete_epoch=True):
    c=t.criterion_ref
    return dict(schema=SCHEMA,science=science,sources=sources,epoch=t.epoch,
        complete_epoch=complete_epoch,model=cpu(t.model.state_dict()),ema=cpu(t.ema.ema.state_dict()),
        optimizer=cpu(t.optimizer.state_dict()),scaler=cpu(t.scaler.state_dict()),scheduler=cpu(t.scheduler.state_dict()),
        gradients={n:cpu(p.grad) if p.grad is not None else None for n,p in t.model.named_parameters()},
        rng=cpu(rng_state()),ema_updates=t.ema.updates,last_opt_step=getattr(t,'_last_opt_step',-1),
        accumulate=t.accumulate,real_updates=t.real_updates,update_attempts=t.update_attempts,
        skipped_amp_updates=t.skipped_amp_updates,batch_visits=t.batch_visits,
        criterion=dict(calls=c.calls,selected_total=c.selected_total,gradient_checks=copy.deepcopy(c.gradient_checks),
                       last_stats=copy.deepcopy(c.last_stats),method_variant=c.method_variant,target_source=c.target_source,fallback_batches=c.fallback_batches),
        metrics=cpu(t.metrics),fitness=t.fitness,best_fitness=t.best_fitness,
        stopper=copy.deepcopy(t.stopper.__dict__),
        auxiliary=cpu(t.auxiliary.state_dict()) if hasattr(t,'auxiliary') else None,
        auxiliary_gradients={n:cpu(p.grad) if p.grad is not None else None for n,p in t.auxiliary.named_parameters()} if hasattr(t,'auxiliary') else None,
        resume_semantics='epoch_boundary_full_state_plus_epoch_deterministic_worker_regeneration_v1')

def save(path,state):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(state,f)

def restore(t,state,science):
    if state['schema']!=SCHEMA or state['science']!=science or not state['complete_epoch']:raise ValueError('Ineligible resume checkpoint')
    t.model.load_state_dict(state['model'],strict=True);t.ema.ema.load_state_dict(state['ema'],strict=True)
    if hasattr(t,'auxiliary'):
        t.auxiliary.load_state_dict(state['auxiliary'],strict=True)
        for name,p in t.auxiliary.named_parameters():
            g=state['auxiliary_gradients'][name];p.grad=None if g is None else g.to(device=p.device,dtype=p.dtype)
    elif state.get('auxiliary') is not None:raise ValueError('Resume has unexpected auxiliary state')
    t.optimizer.load_state_dict(state['optimizer']);t.scaler.load_state_dict(state['scaler']);t.scheduler.load_state_dict(state['scheduler'])
    t.ema.updates=state['ema_updates'];t.accumulate=state['accumulate']
    for key in ('real_updates','update_attempts','skipped_amp_updates','batch_visits','metrics','fitness','best_fitness'):
        setattr(t,key,copy.deepcopy(state[key]))
    for name,p in t.model.named_parameters():
        g=state['gradients'][name];p.grad=None if g is None else g.to(device=p.device,dtype=p.dtype)
    for key,value in state['criterion'].items():setattr(t.criterion_ref,key,copy.deepcopy(value))
    t.stopper.__dict__.update(state['stopper']);t.start_epoch=state['epoch']+1
    t._resume_last_opt_step=t._last_opt_step=state['last_opt_step'];t._full_state_restored=True
    restore_rng(state['rng'])
    checks={k:digest(getattr(t,k).state_dict())==digest(state[k]) for k in ('optimizer','scaler','scheduler')}
    checks.update(model=digest(t.model.state_dict())==digest(state['model']),ema=digest(t.ema.ema.state_dict())==digest(state['ema']))
    if hasattr(t,'auxiliary'):checks['auxiliary']=digest(t.auxiliary.state_dict())==digest(state['auxiliary'])
    if not all(checks.values()):raise AssertionError('Full state restore differs: '+str(checks))
    return checks
