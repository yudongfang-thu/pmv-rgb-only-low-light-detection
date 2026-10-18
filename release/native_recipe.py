"""xm_hsv2_r1 (94): the union recipe with ultralytics default RandomHSV on (h .015 / s .7 / v .4) and Mosaic off (close_mosaic 0)."""
RECIPE = dict(
    ultralytics_version='8.4.115', torch_version='2.10.0+cu128',
    imgsz=640, epochs=200, batch=32, nbs=64, workers=4, seed=42,
    optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937,
    weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
    warmup_bias_lr=0.1, cos_lr=False, close_mosaic=0, patience=0,
    amp=True, deterministic=True, teacher_cache_images=64,
    log_every_batches=100, actual_budget_seconds=None,
    periodic_eval_every=5, loader_policy='epoch_deterministic_v1',
    checkpoint_export_dtype='float32',
    augmentation=dict(mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        degrees=0.0, shear=0.0, perspective=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        erasing=0.0, bgr=0.0),
)
ARMS = {'RGBU': 'rgb', 'IRU': 'ir', 'FUSIONNU': 'pixelmean', 'GU': 'rgb', 'FU': 'rgb',
        'PVC': 'rgb', 'PVC_SHUF': 'rgb', 'PVC_NOKD': 'rgb',
        'PVC_AUG': 'rgb', 'PVC_ACTIVE': 'rgb', 'PVC_AUG_NOKD': 'rgb',
        'MV_AUG_IR': 'rgb', 'MV_AUG_SHUF': 'rgb', 'MV_AUG_AUG': 'rgb',
        'MV_AUG_IR_GEO': 'rgb', 'MV_AUG_IR_BOTH': 'rgb', 'MV_AUG_AUG_GEO': 'rgb', 'MV_AUG_RAWIR_GEO': 'rgb',
        'MV_AUG_SHUF_GEO': 'rgb', 'MV_AUG_IAF': 'rgb', 'MV_AUG_RAWIR': 'rgb',
        'MIX_PVC': 'rgb', 'PVC_AUGPAIR': 'rgb',
        'RGBU_FULL': 'rgb', 'PVC_FULL': 'rgb', 'AUG_FULL': 'rgb',
        'RAWIR_FULL': 'rgb', 'SHUF_FULL': 'rgb', 'IAF_FULL': 'rgb', 'MV_FULL': 'rgb',
        'CMP_KD': 'rgb', 'CMP_CWD': 'rgb', 'CMP_LD': 'rgb',
        'RGBU_H': 'rgb', 'PVC_H': 'rgb', 'AUG_H': 'rgb', 'MV_H': 'rgb'}
INITIAL_YOLO11N_SHA256 = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
