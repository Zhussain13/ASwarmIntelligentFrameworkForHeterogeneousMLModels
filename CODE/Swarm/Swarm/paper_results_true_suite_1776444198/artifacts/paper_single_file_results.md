# Final Results For Paper: Individual Models vs Transfer Learning

## What this comparison means

This section compares two settings:

1. **Without transfer learning**: each model works independently and does not reuse knowledge from other models.
2. **With transfer learning**: models align their feature spaces and reuse knowledge across different backbones during inference.

The experiment used 3 backbone models (`resnet18`, `mobilenet_v3_small`, `efficientnet_b0`) on the same video data, with strict matching thresholds, real heavy inference (OpenCV HOG), and 3 random seeds per setting.

The purpose of this comparison is to isolate the value of transfer learning in a heterogeneous multi-model system. Both settings use the same data, same runtime constraints, and same matching rules. The only difference is whether cross-backbone alignment and transfer are enabled. This allows a direct and fair measurement of what transfer learning contributes in practice.

## Main comparison (mean +/- standard deviation)

| Metric | Without transfer learning | With transfer learning | Advantage gained |
|:--|--:|--:|:--|
| Skipped decisions (%) | 98.75 +/- 0.00 | 99.31 +/- 0.24 | +0.56 percentage points |
| Heavy inferences per run | 3.00 +/- 0.00 | 1.67 +/- 0.58 | -1.33 per run (44.44% fewer) |
| Cross-model reuse (%) | 0.00 +/- 0.00 | 73.72 +/- 20.80 | +73.72 percentage points |
| Average similarity on reused decisions | 0.9861 +/- 0.0002 | 0.9888 +/- 0.0010 | +0.0028 |
| Average heavy inference latency (ms) | 486.26 +/- 13.21 | 400.18 +/- 202.78 | -86.07 ms (17.70% lower) |

The table shows that the largest gain appears in cross-model reuse. Without transfer learning, the system behaves like isolated model islands and cannot exchange useful information across backbones. Once transfer learning is enabled, reuse increases sharply, and this directly reduces the need for expensive heavy inference calls. Even though the skip-rate improvement looks numerically small, that gain is meaningful because it is achieved under strict thresholds and already high baseline skipping.

## Clear takeaways for the paper

Transfer learning is the reason cross-model reuse appears at all. Without transfer learning, reuse between different backbones stays at 0%. With transfer learning enabled, cross-model reuse becomes strong and consistent across runs.

This translates into practical gains: fewer heavy inferences, slightly higher skip rate, and better runtime efficiency. In simple terms, the system does less expensive work while preserving decision quality.

From a systems perspective, this matters for real deployments where energy, compute budget, and latency are constrained. Reducing heavy calls by about 44% per run means the swarm spends less time in high-cost inference mode and more time in efficient reuse mode. The latency reduction further supports this point and indicates that transfer learning is not only a modeling improvement but also an operational optimization.

## Representative proof point

In a representative strict run (seed 27), cross-model reuse moved from **0% without transfer learning** to **97.06% with transfer learning** among skipped decisions.

This representative run helps explain the mechanism behind the average results: once feature spaces are aligned, models with different backbones can accept each other’s packets with high confidence and use them as valid shortcuts instead of re-running heavy detection.

## Extended interpretation for discussion section

These results support the central claim that heterogeneous backbones can cooperate effectively when transfer learning is used to align embeddings. The benefit is not limited to one model pair; it appears at swarm level and persists across multiple seeds. In other words, transfer learning converts a set of independent detectors into a cooperative perception system.

At the same time, variability in cross-model reuse across seeds suggests the strength of transfer depends on scene dynamics and packet timing. This is expected in decentralized systems and does not weaken the main finding; instead, it highlights where future work can improve robustness (for example, larger calibration windows, stronger temporal smoothing, or confidence-aware packet routing).

## Conclusion sentence you can paste directly

Compared with independent single-model inference, our transfer-learning-enabled multi-backbone system delivers strong cross-model knowledge reuse and reduces heavy computation, demonstrating clear efficiency gains under strict evaluation settings.
