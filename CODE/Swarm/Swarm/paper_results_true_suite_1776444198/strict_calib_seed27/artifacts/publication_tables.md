# Publication-Ready Tables

Source: `strict_calib_seed27/metrics/` | Baseline assumption: heavy-only inference for all decisions.

## Table 1. Main Results (Baseline vs Proposed)
| metric                | baseline_heavy_only   | proposed_swarm_kt   | gain                      |
|:----------------------|:----------------------|:--------------------|:--------------------------|
| Total decisions       | 240                   | 240                 | Context metric            |
| Heavy inference calls | 240                   | 2                   | 99.17% reduction          |
| Skipped decisions     | 0                     | 238                 | +238                      |
| Skip rate (%)         | 0.00                  | 99.17               | +99.17 pp                 |
| Heavy compute factor  | 1.00x                 | 0.01x               | 120.00x fewer heavy calls |

## Table 2. Per-Agent Breakdown
| agent_id   |   total |   heavy |   skipped |   skip_rate_percent |   heavy_rate_percent |
|:-----------|--------:|--------:|----------:|--------------------:|---------------------:|
| agent_1    |      80 |       1 |        79 |               98.75 |                 1.25 |
| agent_2    |      80 |       1 |        79 |               98.75 |                 1.25 |
| agent_3    |      80 |       0 |        80 |              100.00 |                 0.00 |

## Table 3. Similarity Quality (Skipped Decisions)
| stat   |    value |
|:-------|---------:|
| mean   | 0.987825 |
| std    | 0.0178   |
| min    | 0.912677 |
| median | 0.99491  |

## Table 4. Knowledge Transfer Usage
Receiver agent (rows) vs packet source agent (columns).

| agent_id   |   agent_1 |   agent_2 |
|:-----------|----------:|----------:|
| agent_1    |         7 |        72 |
| agent_2    |        79 |         0 |
| agent_3    |        77 |         3 |

| transfer_type             |   count |
|:--------------------------|--------:|
| self-packet reuse         |    7    |
| cross-agent reuse         |  231    |
| cross-agent reuse percent |   97.06 |

## Table 5. Adaptation Log Summary
| file                                  |           timestamp |       loss |   samples_used | backbone           |
|:--------------------------------------|--------------------:|-----------:|---------------:|:-------------------|
| adaptation_log_agent_1_1776424612.csv | 1776424612.50831747 | 0.00035399 |              8 | resnet18           |
| adaptation_log_agent_1_1776424613.csv | 1776424613.65920305 | 0.00015414 |              8 | resnet18           |
| adaptation_log_agent_1_1776424614.csv | 1776424614.92928028 | 0.00009810 |              8 | resnet18           |
| adaptation_log_agent_1_1776424616.csv | 1776424616.16218042 | 0.00007431 |              8 | resnet18           |
| adaptation_log_agent_1_1776424617.csv | 1776424617.30331278 | 0.00006882 |              8 | resnet18           |
| adaptation_log_agent_1_1776424618.csv | 1776424618.65765429 | 0.00005189 |              8 | resnet18           |
| adaptation_log_agent_1_1776424619.csv | 1776424619.75295663 | 0.00004192 |              8 | resnet18           |
| adaptation_log_agent_1_1776424620.csv | 1776424620.91680861 | 0.00003794 |              8 | resnet18           |
| adaptation_log_agent_1_1776424622.csv | 1776424622.03159046 | 0.00003630 |              8 | resnet18           |
| adaptation_log_agent_2_1776424613.csv | 1776424613.55657673 | 0.00008173 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424614.csv | 1776424614.66460609 | 0.00005172 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424616.csv | 1776424616.07086277 | 0.00003569 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424617.csv | 1776424617.18827772 | 0.00002351 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424618.csv | 1776424618.51144719 | 0.00002184 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424619.csv | 1776424619.56645179 | 0.00002161 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424620.csv | 1776424620.83341551 | 0.00001791 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424621.csv | 1776424621.91517377 | 0.00001626 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776424623.csv | 1776424623.02244163 | 0.00001506 |              8 | mobilenet_v3_small |
| adaptation_log_agent_3_1776424613.csv | 1776424613.14430761 | 0.00011768 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424615.csv | 1776424615.17256713 | 0.00005201 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424617.csv | 1776424617.05952549 | 0.00004213 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424619.csv | 1776424619.17378139 | 0.00003127 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424621.csv | 1776424621.15086770 | 0.00002661 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424622.csv | 1776424622.90947962 | 0.00002109 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424624.csv | 1776424624.00830770 | 0.00001737 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424624.csv | 1776424624.68861723 | 0.00001917 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424625.csv | 1776424625.48318005 | 0.00001603 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776424626.csv | 1776424626.20035267 | 0.00001570 |              8 | efficientnet_b0    |

## Caption Notes (Use in Paper)
- Pilot run configuration: 80 frames, 3 agents, 240 agent-decisions.
- Baseline is analytically derived from the same run (heavy-only path), not from a separate hardware-timed baseline run.
- Cross-agent reuse in this run: 231/238 (97.06%).
