# PMV: Privileged Multi-View Learning for RGB-Only Nighttime Vehicle Detection

Training-time multimodal, deployment-time RGB-only object detection. Registered RGB--infrared pairs are
used **only during training** to form privileged views over one shared-weight YOLO11n. At test time the
deployed model is a plain RGB YOLO11n: one forward pass, no teacher, no extra latency.

Paper: *Privileged Multi-View Learning for RGB-Only Nighttime Vehicle Detection* (AIEIT 2026, under review).

## Method

For every paired training image, three views share the weights of the deployed detector:

| View | Definition | Role |
|---|---|---|
| RGB | `x` | deployment branch |
| Fusion | `(x + x_ir) / 2` | brings infrared evidence into the shared weights |
| Perturbed | `(x + J(x)) / 2`, photometric jitter `J` | stability constraint on what the RGB view learns from the fusion view |

Every view receives the native detection loss (privileged weight 0.5). The class scores of the two
auxiliary views are detached and distilled into the RGB scores (dense KL, weight 0.5). Batch-normalization
running statistics are accumulated from the RGB view only. Deployment is the unmodified RGB branch.

## Repository layout

```
release/     exact frozen research release (training, criterion, readout)
reference/   pinned trainer/loader required by the release
configs/     experiment configuration schema (main recipe HSV)
docs/        data contract and portability notes
```

`release/` contains the code that produced the paper's main table: `pvc_criterion.py` (views, dense score
KD, BN handling), `scene_runtime.py` and `train.py` (training loop), `readout.py` (E200 FP32-EMA
evaluation), `checkpoint.py`, `periodic_eval.py`, and the comparison-arm criteria. The public copy is
identical to the frozen release except `release/common.py`, where the private absolute paths were
replaced by environment-based resolution; the method code is unchanged.

## Environment

The paper used a pinned environment:

- Python 3.10
- PyTorch 2.10.0+cu128
- Ultralytics 8.4.115
- One NVIDIA RTX 4090 (24 GB)

The code checks these versions against the configuration; edit the config if you deliberately use
another environment.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The main experiments use the night-time split of DroneVehicle with union ground truth: all RGB boxes are
kept, and an infrared box is appended only if its IoU with every RGB box is below 0.5. The paper's
development population is 862 validation images with 15,480 union boxes in five classes. LLVIP is used as
a second dataset on its official split (12,025 training pairs; 3,463 test images).

The repository does **not** ship images, labels, weights, or the internal data-contract receipts. See
`docs/data.md` for the expected layout and `docs/portability.md` for what must be generated before a run.

## Training

```bash
export PMV_CAMPAIGN="$(pwd)"          # campaign root; outputs go under $PMV_CAMPAIGN/runs
cd release
python train.py --config ../configs/night_MV_H_s42.schema.yaml \
                --output "$PMV_CAMPAIGN/runs/night_MV_H_s42" --mode train
```

`PMV_CAMPAIGN` defaults to the repository root; training refuses to write outside it, and the readout
expects the run under `$PMV_CAMPAIGN/runs`. `configs/night_MV_H_s42.schema.yaml` is the configuration
**schema**: it documents every field the trainer expects, but its data-contract paths and SHA256 values
must be produced for your own copy of the data.

## Evaluation

The formal endpoint is the last-epoch FP32 EMA at epoch 200, evaluated on the RGB view:

```bash
python readout.py --checkpoint <run>/weights/last.pt \
                  --config <run>/protocol_config.yaml \
                  --output <run>/readout --mode formal
```

The readout reports five-class mAP50-95, AP50, AP75 and the Night / Dark-night split.

## Reproducibility notes

- This is the research release used for the manuscript, not a polished library. It expects the
  campaign's paired-data contract, the pinned environment and the vendored `reference/` trainer.
- The internal resource-lease and run-receipt hooks are replaced by no-op stubs in
  `reference/legacy_oev1/train_object_evidence.py`; they do not affect the optimization.
- The exact paper numbers were produced with frozen campaign artifacts on host 94. The example config
  preserves the main recipe (HSV on, Mosaic off) and the full schema, but is not a turnkey config.

## Citation

```bibtex
@inproceedings{pmv2026,
  title  = {Privileged Multi-View Learning for RGB-Only Nighttime Vehicle Detection},
  author = {Anonymous},
  booktitle = {AIEIT 2026},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.
