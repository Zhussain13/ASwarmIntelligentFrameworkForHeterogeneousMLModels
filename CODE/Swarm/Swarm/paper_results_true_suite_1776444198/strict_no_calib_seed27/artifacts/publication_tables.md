# Publication-Ready Tables

Source: `strict_no_calib_seed27/metrics/` | Baseline assumption: heavy-only inference for all decisions.

## Table 1. Main Results (Baseline vs Proposed)
| metric                | baseline_heavy_only   | proposed_swarm_kt   | gain                     |
|:----------------------|:----------------------|:--------------------|:-------------------------|
| Total decisions       | 240                   | 240                 | Context metric           |
| Heavy inference calls | 240                   | 3                   | 98.75% reduction         |
| Skipped decisions     | 0                     | 237                 | +237                     |
| Skip rate (%)         | 0.00                  | 98.75               | +98.75 pp                |
| Heavy compute factor  | 1.00x                 | 0.01x               | 80.00x fewer heavy calls |

## Table 2. Per-Agent Breakdown
| agent_id   |   total |   heavy |   skipped |   skip_rate_percent |   heavy_rate_percent |
|:-----------|--------:|--------:|----------:|--------------------:|---------------------:|
| agent_1    |      80 |       1 |        79 |               98.75 |                 1.25 |
| agent_2    |      80 |       1 |        79 |               98.75 |                 1.25 |
| agent_3    |      80 |       1 |        79 |               98.75 |                 1.25 |

## Table 3. Similarity Quality (Skipped Decisions)
| stat   |    value |
|:-------|---------:|
| mean   | 0.986091 |
| std    | 0.014672 |
| min    | 0.915102 |
| median | 0.99225  |

## Table 4. Knowledge Transfer Usage
Receiver agent (rows) vs packet source agent (columns).

| agent_id   |   agent_1 |   agent_2 |   agent_3 |
|:-----------|----------:|----------:|----------:|
| agent_1    |        79 |         0 |         0 |
| agent_2    |         0 |        79 |         0 |
| agent_3    |         0 |         0 |        79 |

| transfer_type             |   count |
|:--------------------------|--------:|
| self-packet reuse         |     237 |
| cross-agent reuse         |       0 |
| cross-agent reuse percent |       0 |

## Table 5. Adaptation Log Summary
| file                                  |           timestamp |       loss |   samples_used | backbone           |
|:--------------------------------------|--------------------:|-----------:|---------------:|:-------------------|
| adaptation_log_agent_1_1776424580.csv | 1776424580.39934564 | 0.00030679 |              8 | resnet18           |
| adaptation_log_agent_1_1776424581.csv | 1776424581.61473227 | 0.00015881 |              8 | resnet18           |
| adaptation_log_agent_1_1776424582.csv | 1776424582.79792356 | 0.00010811 |              8 | resnet18           |
| adaptation_log_agent_1_1776424583.csv | 1776424583.98227406 | 0.00007166 |              8 | resnet18           |
| adaptation_log_agent_1_1776424585.csv | 1776424585.11405754 | 0.00006365 |              8 | resnet18           |
| adaptation_log_agent_1_1776424586.csv | 1776424586.27404284 | 0.00004667 |              8 | resnet18           |
| adaptation_log_agent_1_1776424587.csv | 1776424587.46881819 | 0.00004444 |              8 | resnet18           |
| adaptation_log_agent_1_1776424589.csv | 1776424589.33736420 | 0.00003976 |              8 | resnet18           |
| adaptation_log_agent_1_1776424590.csv | 1776424590.84844756 | 0.00003354 |              8 | resnet18           |
| adaptation_log_agent_2_1776424581.csv | 1776424581.29529762 | 0.00026767 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424582.csv | 1776424582.42033553 | 0.00014488 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424583.csv | 1776424583.64001584 | 0.00011101 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424584.csv | 1776424584.77204204 | 0.00007843 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424585.csv | 1776424585.91184902 | 0.00007097 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424587.csv | 1776424587.10870886 | 0.00005083 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424589.csv | 1776424589.07447314 | 0.00003936 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424590.csv | 1776424590.56444001 | 0.00003542 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424591.csv | 1776424591.93745112 | 0.00002919 |              8 | mobilenet_v3_small |
| adaptation_log_agent_3_1776424581.csv | 1776424581.85188222 | 0.00072565 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424584.csv | 1776424584.25467110 | 0.00018525 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424586.csv | 1776424586.64151502 | 0.00013358 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424589.csv | 1776424589.82968092 | 0.00008522 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424592.csv | 1776424592.41969204 | 0.00007062 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424593.csv | 1776424593.55135822 | 0.00004818 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424594.csv | 1776424594.68453050 | 0.00003666 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424595.csv | 1776424595.55308223 | 0.00003411 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424596.csv | 1776424596.52400589 | 0.00002979 |              8 | efficientnet_b0    |

## Caption Notes (Use in Paper)
- Pilot run configuration: 80 frames, 3 agents, 240 agent-decisions.
- Baseline is analytically derived from the same run (heavy-only path), not from a separate hardware-timed baseline run.
- Cross-agent reuse in this run: 0/237 (0.00%).
