"""Bound native DetMetrics readout, preserving actual validator order.

Validated union labels are primary; the same endpoints also report original
RGB labels. Only the input pixels differ between the two native endpoints.
"""
import argparse
import gzip
import hashlib
import json
import time
from common import *


def verify_captured_objects(rows, canonical, manifest, captured, gt):
    import numpy as np
    by_stem={r['stem']:r for r in manifest}
    observed=[str(Path(r['image']).resolve()) for r in rows]
    if len(rows)!=len(manifest) or len(set(observed))!=len(rows) or set(observed)!=set(canonical):raise ValueError('Captured unique canonical image set differs')
    if observed!=captured['actual_loader_roster']:raise ValueError('Actual validator order differs')
    if len(by_stem)!=len(manifest):raise ValueError('Duplicate val stems')
    total=0
    for row in rows:
        source=by_stem[Path(row['image']).stem]
        label=Path(source[gt+'_label'])
        if source.get(gt+'_label_sha256') and sha(label)!=source[gt+'_label_sha256']:raise ValueError('GT digest changed')
        values=[line.split() for line in label.read_text().splitlines() if line.strip()]
        if any(len(v)!=5 for v in values):raise ValueError('Malformed YOLO GT')
        native=np.asarray([[float(v) for v in line] for line in values],dtype=np.float32).reshape(-1,5)
        if len(native):
            _,indices=np.unique(native,axis=0,return_index=True)
            if len(indices)<len(native):native=native[indices]
        classes=np.asarray(row['gt_classes'],dtype=np.float64).reshape(-1)
        boxes=np.asarray(row['gt_boxes'],dtype=np.float64).reshape(-1,4)
        if not np.array_equal(classes,native[:,0]):raise ValueError('GT class order/count differs')
        original=tuple(row['original_shape']);canvas=tuple(row['canvas_shape'])
        if original!=(512,640) or canvas!=(544,672):raise ValueError('Canonical validation geometry differs')
        h,w=original;H,W=canvas;px=(W-w)/2;py=(H-h)/2;x=native[:,1:]
        expected=np.column_stack(((x[:,0]-x[:,2]/2)*w+px,(x[:,1]-x[:,3]/2)*h+py,(x[:,0]+x[:,2]/2)*w+px,(x[:,1]+x[:,3]/2)*h+py))
        if not np.allclose(boxes,expected,rtol=0,atol=1e-3):raise ValueError('GT geometry differs')
        total+=len(native)
    return dict(images=len(rows),gt=total,actual_loader_order_sha256=hashlib.sha256('\n'.join(observed).encode()).hexdigest(),gt_modality=gt,order_policy='native_validator_order_preserved')


def input_validator(base, records, input_mode, gt):
    """Same native validator and GT; only replace the paired input pixels."""
    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    by_stem={r['stem']:r for r in records}
    class SceneValidator(base):
        def preprocess(self,batch):
            batch=super().preprocess(batch)
            if input_mode=='rgb':return batch
            canvas=tuple(batch['img'].shape[-2:]);images=[]
            for path in batch['im_file']:
                row=by_stem[Path(path).stem]
                def load(modality):
                    image=cv2.imread(row[modality+'_image'])
                    if image is None or image.shape[:2]!=(512,640):raise ValueError('Invalid canonical paired image')
                    image=LetterBox(new_shape=canvas,auto=False,scaleup=False,stride=32)(image=image)
                    return np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))
                if input_mode=='ir':
                    # r2: IR 臂只换像素流, 与训练侧 scene_runtime 的 IR 分支一致
                    pixels=load('ir').astype(np.float32)
                elif input_mode=='pixelmean':
                    pixels=(load('rgb').astype(np.float32)+load('ir').astype(np.float32))*.5
                else:
                    raise ValueError('Unknown native input mode')
                images.append(pixels)
            batch['img']=torch.from_numpy(np.stack(images)).to(device=batch['img'].device,dtype=batch['img'].dtype)/255
            return batch
    return SceneValidator

def formal_endpoint_binding(checkpoint,cfg):
    """A formal readout must be the bound natural E200 FP32-EMA endpoint."""
    import torch
    checkpoint=Path(checkpoint).resolve()
    run=checkpoint.parent.parent
    if checkpoint.name!='last.pt' or checkpoint.parent.name!='weights':
        raise ValueError('Formal readout requires run/weights/last.pt')
    if not run.is_relative_to(CAMPAIGN/'runs'):
        raise ValueError('Formal checkpoint is outside this campaign runs root')
    completion=read(run/'completion_receipt.json')
    if (completion['status'],completion['last_epoch'],completion['epochs_configured'])!=('training_completed',200,200):
        raise ValueError('Formal run has no complete E200 training receipt')
    if (completion['method_variant'],completion['seed'],completion['execution_version'])!=(cfg['method_variant'],cfg['seed'],cfg['execution_version']):
        raise ValueError('Completion identity differs from formal config')
    if Path(completion['checkpoint']).resolve()!=checkpoint:
        raise ValueError('Completion does not bind this exact checkpoint')
    if completion['scientific_config_sha256']!=scientific_hash(cfg):
        raise ValueError('Completion science hash differs from formal config')
    check_binding(completion['sources'])
    latest=completion['latest_full_checkpoint']
    latest_path=Path(latest['path']).resolve()
    if latest_path!=run/'weights/full_epoch_0200.pt' or latest['epoch']!=200 or latest['completed_epoch'] is not True:
        raise ValueError('No bound complete-epoch E200 recovery state')
    if sha(latest_path)!=latest['sha256']:
        raise ValueError('Final full-state checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    ema=state.get('ema')
    if state.get('model') is not None or ema is None:
        raise ValueError('Final checkpoint is not EMA-only export')
    floating=[v for v in ema.state_dict().values() if torch.is_tensor(v) and v.is_floating_point()]
    if not floating or any(v.dtype!=torch.float32 for v in floating):
        raise ValueError('Final EMA is not fully FP32')
    return dict(run=str(run),completion=str(run/'completion_receipt.json'),
                completion_sha256=sha(run/'completion_receipt.json'),
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                full_state=str(latest_path),full_state_sha256=latest['sha256'],
                scientific_config_sha256=scientific_hash(cfg),ema_floating_tensor_count=len(floating),
                status='FORMAL_E200_FP32_EMA_BOUND')



def main(a):
    cfg=config(a.config,'train' if a.mode=='formal' else 'canary')
    endpoint=formal_endpoint_binding(a.checkpoint,cfg) if a.mode=='formal' else None
    pop,sub,selected=population(cfg)
    if a.population_scope!='subset':raise ValueError('Only the declared native QC population is authorized')
    selected=[r for r in selected if r['split']=='val']
    gt=a.gt or 'union'
    expected_gt=sub[gt+'_val_gt'];data=sub[gt+'_yaml']
    mode=cfg['input_mode']
    rt=runtime();rt.legacy.require_bound_lease_from_environment()
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from evaluate_independent import make_evidence_validator,capture_contract,verify_population
    from diagnose_opportunities import dataset_config,split_images
    torch.set_num_threads(4)
    canonical=[str(p.resolve()) for p in split_images(dataset_config(data),'val')]
    if len(canonical)!=sub['val_images'] or len(set(canonical))!=len(canonical):raise ValueError('Dev roster differs')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    model=YOLO(str(a.checkpoint),task='detect');captured={}
    versions=dict(torch=str(torch.__version__),ultralytics=str(rt.legacy.ultralytics.__version__))
    if versions!=dict(torch=cfg['torch_version'],ultralytics=cfg['ultralytics_version']):raise ValueError('Pinned versions differ')
    model.add_callback('on_val_start',lambda v:captured.update(capture_contract(v,canonical,versions)))
    model.add_callback('on_val_end',lambda v:verify_population(v,captured))
    validator=input_validator(make_evidence_validator(DetectionValidator),selected,mode,gt)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    m=model.val(validator=validator,data=data,split='val',imgsz=640,batch=32,workers=4,device='0',plots=False,save_json=False,verbose=False,project=str(a.output.parent),name=a.output.name,exist_ok=False)
    with gzip.open(a.output/'objects.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
    contract=verify_captured_objects(rows,canonical,selected,captured,gt)
    if contract['gt']!=expected_gt:raise ValueError('Dev GT total differs')
    captured['object_population_verification']=contract;write(a.output/'evaluation_contract.json',captured)
    write(a.output/'metrics.json',dict(status='PASS' if a.mode=='formal' else 'DIAGNOSTIC_PASS',endpoint_binding=endpoint,AP50=float(m.box.map50),AP75=float(m.box.map75),mAP50_95=float(m.box.map),per_class=[dict(class_id=int(ci),name=model.names[int(ci)],AP50=float(m.box.all_ap[j,0]),AP75=float(m.box.all_ap[j,5]),mAP50_95=float(m.box.all_ap[j].mean())) for j,ci in enumerate(m.box.ap_class_index)],checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),seed=cfg['seed'],arm=cfg['campaign_arm'],subset=cfg['subset'],population_scope=a.population_scope,gt_modality=gt,input_mode=mode,images=len(rows),gt=expected_gt,seconds=time.monotonic()-start,sources=source_binding(),metric_units='fraction_0_to_1',object_population_verification=contract,gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,official_test_accessed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('canary','formal'),required=True);p.add_argument('--population-scope',choices=('subset',),default='subset');p.add_argument('--gt',choices=('union','rgb'))
    main(p.parse_args())
