# True Experiment Suite Results (Strict Threshold, Pretrained Backbones)

Suite directory: `d:/Swarm/paper_results_true_suite_1776444198`

Conditions: `strict_no_calib` vs `strict_calib` (3 seeds each).

## Per-Run Metrics
| run                    | condition       |   seed |   total_decisions |   heavy_calls |   skipped_calls |   skip_rate_pct |   cross_reuse_rate_pct |   cross_reuse_count |   skip_with_packet_count |   sim_mean |   sim_std |   heavy_latency_ms_mean |   heavy_latency_ms_std |
|:-----------------------|:----------------|-------:|------------------:|--------------:|----------------:|----------------:|-----------------------:|--------------------:|-------------------------:|-----------:|----------:|------------------------:|-----------------------:|
| strict_calib_seed7     | strict_calib    |      7 |               240 |             2 |             238 |         99.1667 |                57.1429 |                 136 |                      238 |     0.9899 |    0.0132 |                265.8346 |               111.9986 |
| strict_calib_seed17    | strict_calib    |     17 |               240 |             1 |             239 |         99.5833 |                66.9456 |                 160 |                      239 |     0.9889 |    0.0152 |                301.2754 |               nan      |
| strict_calib_seed27    | strict_calib    |     27 |               240 |             2 |             238 |         99.1667 |                97.0588 |                 231 |                      238 |     0.9878 |    0.0178 |                633.4368 |               586.5850 |
| strict_no_calib_seed7  | strict_no_calib |      7 |               240 |             3 |             237 |         98.7500 |                 0.0000 |                   0 |                      237 |     0.9859 |    0.0150 |                493.6669 |               485.5563 |
| strict_no_calib_seed17 | strict_no_calib |     17 |               240 |             3 |             237 |         98.7500 |                 0.0000 |                   0 |                      237 |     0.9862 |    0.0142 |                494.0984 |               463.3014 |
| strict_no_calib_seed27 | strict_no_calib |     27 |               240 |             3 |             237 |         98.7500 |                 0.0000 |                   0 |                      237 |     0.9861 |    0.0147 |                471.0032 |               463.3361 |

## Condition Summary (mean +/- std)
| condition       |   runs |   skip_rate_mean_pct |   skip_rate_std_pct |   heavy_calls_mean |   heavy_calls_std |   cross_reuse_rate_mean_pct |   cross_reuse_rate_std_pct |   sim_mean_mean |   sim_mean_std |   heavy_latency_ms_mean |   heavy_latency_ms_std |
|:----------------|-------:|---------------------:|--------------------:|-------------------:|------------------:|----------------------------:|---------------------------:|----------------:|---------------:|------------------------:|-----------------------:|
| strict_calib    |      3 |              99.3056 |              0.2406 |             1.6667 |            0.5774 |                     73.7158 |                    20.8014 |          0.9888 |         0.001  |                 400.182 |               202.78   |
| strict_no_calib |      3 |              98.75   |              0      |             3      |            0      |                      0      |                     0      |          0.9861 |         0.0002 |                 486.256 |                13.2112 |

## Key Deltas (Calibrated - No Calibration)

- Skip rate delta: 0.5556 pp
- Cross-agent reuse delta: 73.7158 pp
- Heavy calls delta (mean): -1.3333
- Similarity mean delta: 0.0028
