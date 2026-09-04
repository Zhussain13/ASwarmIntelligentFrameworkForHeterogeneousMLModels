# Paste-Ready Results (True Cross-Backbone Suite)

## Experimental Setup

We ran a strict cross-backbone experiment suite with pretrained encoders and real heavy inference. Each run used 3 agents with distinct backbones (`resnet18`, `mobilenet_v3_small`, `efficientnet_b0`), shared 256-D embeddings, strict matching thresholds (`similarity=0.85`, `confidence=0.8`), peer-first packet lookup, and OpenCV HOG as heavy inference. We evaluated two conditions across 3 seeds each (6 runs total, 80 frames/run, 240 agent-decisions/run): (1) `strict_no_calib` and (2) `strict_calib` with 15-frame cross-backbone projection calibration before inference.

## Results Paragraph

Under strict thresholds, calibration improved transfer quality substantially. Compared with no calibration, the calibrated condition increased cross-agent reuse from 0.00% +/- 0.00% to 73.72% +/- 20.80% (+73.72 pp). Skip rate improved from 98.75% +/- 0.00% to 99.31% +/- 0.24% (+0.56 pp), while mean heavy calls dropped from 3.00 to 1.67 per run. Mean skipped-decision similarity also increased from 0.9861 to 0.9888.

## Table X. Condition-Level Summary (mean +/- std across 3 seeds)

| Condition | Skip rate (%) | Heavy calls/run | Cross-agent reuse (%) | Skipped similarity mean |
|:--|--:|--:|--:|--:|
| Strict no calibration | 98.75 +/- 0.00 | 3.00 +/- 0.00 | 0.00 +/- 0.00 | 0.9861 +/- 0.0002 |
| Strict with calibration | 99.31 +/- 0.24 | 1.67 +/- 0.58 | 73.72 +/- 20.80 | 0.9888 +/- 0.0010 |

## Table Y. Representative Cross-Backbone Transfer Matrix (strict_calib_seed27)

Packet-source backbone (rows) vs receiver backbone (columns), skipped decisions only.

| Source \ Receiver | efficientnet_b0 | mobilenet_v3_small | resnet18 |
|:--|--:|--:|--:|
| mobilenet_v3_small | 3 | 0 | 72 |
| resnet18 | 77 | 79 | 7 |

## Reporting Note

These results use pretrained torchvision backbones, strict similarity/confidence thresholds, and a real heavy detector path. This is suitable as a primary protocol-level evidence block for publication, with the usual recommendation to add larger datasets and external ground-truth evaluation for final camera-ready claims.
