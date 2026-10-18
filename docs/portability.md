# Portability notes

This repository is the **research release** used for the manuscript. It is honest about two layers:

1. `release/` is the exact frozen code that produced the paper's main table.
2. `reference/` is the pinned trainer/loader that the release calls. It is vendored here so the method is
   readable end to end.

## Environment variables

`release/common.py` resolves paths from the environment (with repository-relative defaults):

| Variable | Default | Meaning |
|---|---|---|
| `PMV_ROOT` | repository root | base for the defaults below |
| `PMV_PROJECT` | `PMV_ROOT` | project root used by the data contract |
| `PMV_CAMPAIGN` | `PMV_ROOT/runs` | writable output root; training refuses to write outside it |
| `PMV_REFERENCE` | `PMV_ROOT/reference` | vendored pinned trainer |
| `PMV_PYTHON` | current interpreter | interpreter used by periodic evaluation |

## What is not shipped

- Images, labels, checkpoints, and the campaign data-contract receipts.
- The internal resource-lease and run-receipt hooks. `reference/legacy_oev1/train_object_evidence.py`
  replaces them with no-op stubs (`require_bound_lease_from_environment` returns one GPU, receipts are
  skipped). These hooks do not affect the optimization or the metrics.
- The exact campaign configs. `configs/night_MV_H_s42.schema.yaml` preserves the recipe and the full
  schema, with data paths and SHA256 values replaced by placeholders.

## What you must provide

1. The paired union-label data described in `docs/data.md`.
2. `weights/yolo11n.pt` (the COCO-pretrained initialisation; the paper used SHA256 starting
   `0ebbc80d`).
3. A data contract and population receipt for your copy of the data, or an adaptation of
   `release/common.py::config` that relaxes the receipt checks for your setting.

## Status

The main text's numbers were produced on a single RTX 4090 with the frozen campaign artifacts. This
release is intended for inspection and adaptation; it is not a turnkey reproduction package. If you need
the exact frozen receipts, contact the authors.
