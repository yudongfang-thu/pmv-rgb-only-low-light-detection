# Data contract

The repository does not ship images, labels, weights, or the campaign receipts. The trainer expects the
paired union-label layout below; the example configuration in `configs/` shows every field.

## DroneVehicle night split (main experiments)

- 11,200 registered RGB--IR training pairs; 862 validation pairs.
- Five classes: car, freight car, truck, bus, van.
- 640x512 images after the central crop used by the paper (raw images are 840x712).
- Union ground truth: keep every RGB box; append an infrared box only if its IoU with **every** RGB box is
  below 0.5. IR coordinates are used as-is. Validation union GT: 15,480 boxes (13,087 RGB + 2,393 appended).
- The paper's development population is the 862-image validation set; the official test split was not
  used for the main results.

## Expected layout

```
data/
  night_union/
    rgb_union.data.yaml          # student (RGB) dataset
    ir_union.data.yaml           # privileged (IR) dataset
    rgb_train.txt, rgb_val.txt
    ir_train.txt,  ir_val.txt
  contract/
    data_contract.json           # status + frozen inputs + manifests
    population.json              # frozen Night / Dark-night scene population
    rgb_to_infrared_train.json   # paired train mapping
weights/
  yolo11n.pt                     # COCO-pretrained initial weights
```

The trainer binds the files above by SHA256 (`frozen_inputs`, `data_contract_sha256`,
`population_sha256`). For your own copy of the data, generate these receipts and fill the corresponding
fields in the configuration.

## LLVIP (second dataset)

Official split: 12,025 training pairs and 3,463 test images (8,302 persons), one class, images resized to
640x512. The same recipe and fixed E200 endpoint are used; see the paper for the reading rule.

## Citation of the datasets

- DroneVehicle: Sun et al., *DroneVehicle: Drone-based RGB-Infrared Cross-Modality Vehicle Detection*.
- LLVIP: Jia et al., *LLVIP: A Visible-infrared Paired Dataset for Low-light Vision*.
