"""Full-population content attribution: deterministic epochs and full state."""
import argparse
import copy
import math
import os
import sys
import time
import traceback
from pathlib import Path
import yaml
import torch


def _pvc_arm_names():
    """PVC 臂清单的唯一来源是 pvc_criterion.PVC_ARMS。"""
    from pvc_criterion import PVC_ARMS
    return PVC_ARMS

from common import *
import checkpoint as full

class CappedLoader:
    def __init__(self,loader,limit):self.loader,self.limit=loader,limit
    def __len__(self):return self.limit
    def __iter__(self):
        iterator=iter(self.loader)
        for _ in range(self.limit):yield next(iterator)
    def __getattr__(self,key):return getattr(self.loader,key)

def build(cfg,path,out,max_updates=None):
    from scene_runtime import build as scene_build
    return scene_build(runtime(),cfg,path,out,max_updates)

def export_last(t,out):
    ema=copy.deepcopy(t.ema.ema).float().cpu()
    if hasattr(ema,'criterion'):ema.criterion=None
    for part in ema.modules():
        part._forward_hooks.clear();part._forward_pre_hooks.clear();part._backward_hooks.clear()
    for value in ema.state_dict().values():
        if value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError('Nonfinite EMA')
    # Export is full-precision final EMA; resume uses full_epoch_* exclusively.
    full.save(out/'weights/last.pt',dict(epoch=t.epoch,model=None,ema=ema,updates=t.ema.updates,
        train_args=vars(t.args),train_metrics=t.metrics,version=str(runtime().legacy.ultralytics.__version__)))

def run(args):
    mode='train' if args.mode=='train' else 'canary'
    cfg=config(args.config,mode);original_cfg=copy.deepcopy(cfg)
    science=scientific_hash(cfg)
    diagnostic=bool(args.diagnostic_epochs)
    if diagnostic:
        cfg['epochs']=args.diagnostic_epochs
        science+=':diagnostic:'+str(args.diagnostic_epochs)+':'+str(args.diagnostic_batches)
    out=args.output.resolve()
    if not str(out).startswith(str(CAMPAIGN)+'/'):raise ValueError('Outside campaign disk')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol_config.json',original_cfg)
    (out/'protocol_config.yaml').write_text(yaml.safe_dump(original_cfg,sort_keys=False))
    rt=runtime();rt.legacy.require_bound_lease_from_environment();torch.cuda.init();torch.set_num_threads(4)
    if str(torch.__version__)!=cfg['torch_version'] or str(rt.legacy.ultralytics.__version__)!=cfg['ultralytics_version']:raise ValueError('Pinned environment differs')
    sources=source_binding();write(out/'source_binding.json',sources)
    max_updates=24 if args.mode=='canary' and not diagnostic else None
    rt,t=build(cfg,args.config,out,max_updates)
    started=time.monotonic();epoch_start=[0.];trainer=None
    resume=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=False) if args.resume_checkpoint else None
    if resume:
        check_binding(resume['sources'])
        assert resume['method_metadata']['method_variant']==cfg['method_variant']
        assert resume['method_metadata']['execution_version']==cfg['execution_version']
    try:
        trainer=t;setup=t._setup_train
        def checked_setup():
            setup()
            ready=read(out/'runtime_ready.json')
            if (ready['train_images'],ready['val_images'],ready['train_batches'],ready['workers'])!=(cfg['expected_train_images'],cfg['expected_val_images'],cfg['expected_train_batches'],4):raise ValueError('Wrong scene train/dev loaders')
            write(out/'initialization.json',dict(model=cfg['model'],state=full.digest(t.model.state_dict()),
                optimizer_empty=not bool(t.optimizer.state),ema_updates=t.ema.updates))
            if t.optimizer.state or t.ema.updates:raise AssertionError('Fresh setup optimizer/EMA differs')
            if diagnostic:t.train_loader=CappedLoader(t.train_loader,args.diagnostic_batches)
            if resume:
                checks=full.restore(t,resume,science)
                write(out/'resume_receipt.json',dict(status='RESTORED_STATE_VERIFIED',source=str(args.resume_checkpoint),
                    source_sha256=sha(args.resume_checkpoint),checks=checks,next_epoch=t.start_epoch,
                    semantics=resume['resume_semantics'],worker_rng='regenerated per epoch; no unrecorded prefetched continuation'))
        t._setup_train=checked_setup
        preprocess=t.preprocess_batch
        def observed(batch):
            result=preprocess(batch)
            if t.batch_visits<=30 or diagnostic or (args.stop_after_epochs and t._subset_batch_index<=30):
                append(out/'sample_stream.jsonl',dict(batch=t.batch_visits,epoch=t.epoch,
                    images=list(result['im_file']),tensors={k:full.digest(result[k]) for k in ('img','strong_img','cls','bboxes','batch_idx')},
                    teacher_labels={k:full.digest(result['teacher_batch'][k]) for k in ('cls','bboxes','batch_idx')}))
            return result
        t.preprocess_batch=observed
        opt=t.optimizer_step
        def optimizer_step():
            verify=t.real_updates<3
            before=[p.detach().clone() for p in t.model.parameters() if p.requires_grad] if verify else None
            aux_before=[p.detach().clone() for p in t.auxiliary.parameters()] if verify and hasattr(t,'auxiliary') else None
            count=t.real_updates
            opt();t._last_opt_step=t._subset_current_ni
            if verify:
                delta=torch.stack([(p.detach().float()-b.float()).square().sum()
                    for p,b in zip((p for p in t.model.parameters() if p.requires_grad),before)]).sum().sqrt()
                changed=float(delta)
                successful=t.real_updates>count
                if successful and not changed>0:raise AssertionError('Optimizer counted an update without changing student parameters')
                if aux_before is not None:
                    aux_delta=float(torch.stack([(p.detach().float()-b.float()).square().sum() for p,b in zip(t.auxiliary.parameters(),aux_before)]).sum().sqrt())
                    if successful and not aux_delta>0:raise AssertionError('Adapter did not update')
                    append(out/'auxiliary_updates.jsonl',dict(attempt=t.update_attempts,successful=successful,delta_l2=aux_delta))
                append(out/'actual_updates.jsonl',dict(attempt=t.update_attempts,successful=successful,
                    student_parameter_delta_l2=changed,auxiliary_present=hasattr(t,'auxiliary'),amp_skipped=not successful,real_updates=t.real_updates))
        t.optimizer_step=optimizer_step
        def epoch_begin(trainer):
            epoch_start[0]=time.monotonic();trainer._subset_batch_index=0
            loader=trainer.train_loader.loader if isinstance(trainer.train_loader,CappedLoader) else trainer.train_loader
            full.restart_epoch_loader(loader,cfg['seed'],trainer.epoch)
        def batch_begin(trainer):
            trainer._subset_current_ni=trainer.epoch*len(trainer.train_loader)+trainer._subset_batch_index
            trainer._subset_batch_index+=1
        t.add_callback('on_train_epoch_start',epoch_begin)
        t.add_callback('on_train_batch_start',batch_begin)
        def save_model():
            for model in (t.model,t.ema.ema):
                if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values() if value.is_floating_point()):
                    raise FloatingPointError('Nonfinite live model/EMA; no repaired checkpoint may be saved')
            complete=t._subset_batch_index==len(t.train_loader)
            state=full.capture(t,science,sources,complete)
            state['method_metadata']={k:original_cfg[k] for k in ('subset','population_sha256','method_variant','target_source','rgb_target_teacher','teacher','reference','seed','model','execution_version','evaluation_contract','loader_policy','expected_train_images','epochs','frozen_inputs')}
            state['method_metadata']['export_format']='FP32_EMA'
            state['method_metadata']['initialization']=read(out/'initialization.json')
            p=out/'weights'/f'full_epoch_{t.epoch+1:04d}.pt';full.save(p,state)
            write(out/'latest_checkpoint.json',dict(path=str(p),sha256=sha(p),completed_epoch=complete,
                optimizer_updates=t.real_updates,epoch=t.epoch+1),False)
            return True
        t.save_model=save_model
        def epoch_end(trainer):
            torch.cuda.synchronize()
            row=dict(epoch=t.epoch+1,seconds=time.monotonic()-epoch_start[0],optimizer_updates=t.real_updates,
                update_attempts=t.update_attempts,amp_skips=t.skipped_amp_updates)
            append(out/'epoch_timing.jsonl',row)
            write(out/'budget_progress.json',dict(completed_epochs=t.epoch+1,elapsed_seconds=time.monotonic()-started,
                remaining_epochs=max(0,cfg['epochs']-t.epoch-1),recent_epoch_seconds=row['seconds'],
                target_hours='use current-batch measured epoch and evaluation cost; no automatic stop'),False)
            if args.stop_after_epochs and t.epoch+1>=args.stop_after_epochs:t.stop=True
        t.add_callback('on_fit_epoch_end',epoch_end)
        if not args.skip_periodic and not max_updates:
            from periodic_eval import evaluate_epoch
            def periodic(trainer):
                every=1 if diagnostic or args.stop_after_epochs else cfg['periodic_eval_every']
                if (t.epoch+1)%every==0 or t.epoch+1==cfg['epochs']:
                    evaluate_epoch(t,out,original_cfg)
            t.add_callback('on_fit_epoch_end',periodic)
        full.install_resume_loop(t)
        torch.cuda.reset_peak_memory_stats();t.train()
        c=t.criterion_ref
        # Fallback is allowed: it uses the identical explicit method objective.
        if max_updates and t.real_updates<24:raise AssertionError('Canary has fewer than 24 successful updates')
        if not max_updates and not args.stop_after_epochs and t.epoch+1!=cfg['epochs']:raise AssertionError('Incomplete epochs')
        if cfg['method_variant'] in _pvc_arm_names() and max_updates:
            from pvc_criterion import verify_pvc_canary, engineering_checks, engineering_subbatch, ENGINEERING_BATCH_IMAGES
            pvc_canary = verify_pvc_canary(c, original_cfg)
            was_training = t.model.training
            t.model.train()
            def fetch_train_batch():
                try:
                    return next(iter(t.train_loader))
                except (StopIteration, RuntimeError):
                    loader = t.train_loader.loader if isinstance(t.train_loader, CappedLoader) else t.train_loader
                    full.restart_epoch_loader(loader, cfg['seed'], t.epoch)
                    return next(iter(loader))
            raw_batch = fetch_train_batch()
            eng_batch = engineering_subbatch(t.preprocess_batch(raw_batch), ENGINEERING_BATCH_IMAGES)
            eng = engineering_checks(c, t.model, lambda im: t.model(im), eng_batch)
            if not was_training:
                t.model.eval()
            failed = [k for k, v in eng['checks'].items() if not v]
            write(out/'pvc_engineering_receipt.json', dict(status='PASS' if not failed else 'FAIL', engineering_batch_images=int(eng_batch['img'].shape[0]),
                checks=eng['checks'], composition_worst_ratio=eng['composition_worst_ratio'], diagnostics=eng.get('diagnostics'), numerics=eng.get('numerics'),
                losses={k: eng[k] for k in ('kd', 'privileged_native_total', 'native_total')},
                canary=pvc_canary))
            if failed:
                raise AssertionError('PVC engineering checks failed: %s' % failed)
        from cmp_criterion import CMP_ARMS
        if cfg['method_variant'] in CMP_ARMS and max_updates:
            from cmp_criterion import verify_cmp_canary, engineering_checks as cmp_engineering_checks
            from pvc_criterion import engineering_subbatch, ENGINEERING_BATCH_IMAGES
            cmp_canary=verify_cmp_canary(c,original_cfg)
            was_training=t.model.training;t.model.train()
            try:raw_batch=next(iter(t.train_loader))
            except (StopIteration,RuntimeError):
                loader=t.train_loader.loader if isinstance(t.train_loader,CappedLoader) else t.train_loader
                full.restart_epoch_loader(loader,cfg['seed'],t.epoch);raw_batch=next(iter(loader))
            eng_batch=engineering_subbatch(t.preprocess_batch(raw_batch),ENGINEERING_BATCH_IMAGES)
            eng=cmp_engineering_checks(c,t.model,lambda im:t.model(im),eng_batch)
            if not was_training:t.model.eval()
            failed=[k for k,v in eng['checks'].items() if not v]
            write(out/'cmp_engineering_receipt.json',dict(status='PASS' if not failed else 'FAIL',engineering_batch_images=int(eng_batch['img'].shape[0]),
                checks=eng['checks'],composition_worst_ratio=eng['composition_worst_ratio'],
                composition_worst_ratio_float32_advisory=eng['composition_worst_ratio_float32_advisory'],numerics=eng.get('numerics'),
                losses={k:eng[k] for k in ('kd','native_total')},canary=cmp_canary))
            if failed:raise AssertionError('Comparison engineering checks failed: %s'%failed)
        if max_updates and cfg['method_variant'] in ('C','M','CWD_FUSION','CWD_RGB','GU','FU'):
            if c.selected_total<=0:raise AssertionError('No real auxiliary selections')
            if cfg['method_variant'].startswith('CWD'):
                if not any(r['kd_parameter_gradient_l2']>0 for r in c.gradient_checks):raise AssertionError('No verified nonzero CWD parameter gradient')
                if not c.gradient_checks or not all(r['total_parameter_gradient_composition_allclose'] for r in c.gradient_checks):raise AssertionError('CWD parameter gradient composition failed')
            else:
                if not any(r['kd_score_gradient_l2']>0 for r in c.gradient_checks):raise AssertionError('No verified nonzero auxiliary KD score gradient')
                if max((r['total_gradient_composition_max_error'] for r in c.gradient_checks),default=0.)>1e-4:raise AssertionError('Gradient composition error above tolerance')
        export_last(t,out);check_binding(sources)
        receipt=dict(status='diagnostic_completed' if diagnostic else ('canary_completed' if max_updates else ('engineering_profile_completed' if args.stop_after_epochs else 'training_completed')),
            **{k:original_cfg[k] for k in ('subset','population_sha256','method_id','method_variant','target_source','rgb_target_teacher','execution_version','campaign_arm','arm','source','seed','dataset','model','teacher','reference','classification_coefficient','localization_coefficient')},
            epochs_configured=original_cfg['epochs'],last_epoch=t.epoch+1,optimizer_updates=t.real_updates,
            optimizer_update_attempts=t.update_attempts,amp_skipped_updates=t.skipped_amp_updates,ema_updates=t.ema.updates,ema_update_policy='legacy_every_optimizer_attempt_including_amp_skip',
            selected_objects=c.selected_total,gradient_checks=c.gradient_checks,fallback_batches=getattr(c,'fallback_batches',0),scientific_config_sha256=scientific_hash(original_cfg),
            sources=sources,gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.monotonic()-started,
            checkpoint=str(out/'weights/last.pt'),latest_full_checkpoint=read(out/'latest_checkpoint.json'),official_test_accessed=False,
            loader_policy=original_cfg['loader_policy'],resumed_from=str(args.resume_checkpoint) if resume else None)
        if getattr(c,'main_mix',False):
            receipt['mix_counts']=dict(clean_batches=c.clean_batches,augmented_batches=c.augmented_batches,
                augmented_fraction=c.augmented_batches/max(c.clean_batches+c.augmented_batches,1),
                bn_checks=c.mix_bn_checks,view_checks=c.mix_view_checks)
        write(out/'completion_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    except BaseException as e:
        write(out/'failure_receipt.json',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started,
            updates=getattr(t,'real_updates',0),latest_checkpoint=read(out/'latest_checkpoint.json') if (out/'latest_checkpoint.json').exists() else None))
        raise
    finally:
        if trainer:
            for key in ('train_loader','test_loader'):
                loader=getattr(trainer,key,None)
                if hasattr(loader,'close'):loader.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('canary','train','diagnostic'),required=True);p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--diagnostic-epochs',type=int,default=0);p.add_argument('--diagnostic-batches',type=int,default=8)
    p.add_argument('--stop-after-epochs',type=int,default=0);p.add_argument('--skip-periodic',action='store_true');run(p.parse_args())
