# Publication-Ready Tables

Source: `paper_results_backbone_transfer_1776441025/metrics/` | Baseline assumption: heavy-only inference for all decisions.

## Table 1. Main Results (Baseline vs Proposed)
| metric                | baseline_heavy_only   | proposed_swarm_kt   | gain                     |
|:----------------------|:----------------------|:--------------------|:-------------------------|
| Total decisions       | 180                   | 180                 | Context metric           |
| Heavy inference calls | 180                   | 2                   | 98.89% reduction         |
| Skipped decisions     | 0                     | 178                 | +178                     |
| Skip rate (%)         | 0.00                  | 98.89               | +98.89 pp                |
| Heavy compute factor  | 1.00x                 | 0.01x               | 90.00x fewer heavy calls |

## Table 2. Per-Agent Breakdown
| agent_id   |   total |   heavy |   skipped |   skip_rate_percent |   heavy_rate_percent |
|:-----------|--------:|--------:|----------:|--------------------:|---------------------:|
| agent_1    |      60 |       1 |        59 |               98.33 |                 1.67 |
| agent_2    |      60 |       1 |        59 |               98.33 |                 1.67 |
| agent_3    |      60 |       0 |        60 |              100.00 |                 0.00 |

## Table 3. Similarity Quality (Skipped Decisions)
| stat   |    value |
|:-------|---------:|
| mean   | 0.546895 |
| std    | 0.344858 |
| min    | 0.008944 |
| median | 0.529894 |

## Table 4. Knowledge Transfer Usage
Receiver agent (rows) vs packet source agent (columns).

| agent_id   |   agent_1 |   agent_2 |
|:-----------|----------:|----------:|
| agent_1    |         0 |        59 |
| agent_2    |        59 |         0 |
| agent_3    |         0 |        60 |

| transfer_type             |   count |
|:--------------------------|--------:|
| self-packet reuse         |       0 |
| cross-agent reuse         |     178 |
| cross-agent reuse percent |     100 |

## Table 5. Adaptation Log Summary
| file                                  |           timestamp |       loss |   samples_used | backbone           |
|:--------------------------------------|--------------------:|-----------:|---------------:|:-------------------|
| adaptation_log_agent_1_1776421234.csv | 1776421234.74552131 | 0.00209293 |              8 | resnet18           |
| adaptation_log_agent_1_1776421236.csv | 1776421236.43560600 | 0.00018567 |              8 | resnet18           |
| adaptation_log_agent_1_1776421238.csv | 1776421238.23608685 | 0.00008310 |              8 | resnet18           |
| adaptation_log_agent_1_1776421240.csv | 1776421240.05674410 | 0.00006145 |              8 | resnet18           |
| adaptation_log_agent_1_1776421242.csv | 1776421242.10165858 | 0.00003033 |              8 | resnet18           |
| adaptation_log_agent_1_1776421244.csv | 1776421244.30902100 | 0.00002174 |              8 | resnet18           |
| adaptation_log_agent_1_1776421246.csv | 1776421246.73868036 | 0.00001946 |              8 | resnet18           |
| adaptation_log_agent_2_1776421234.csv | 1776421234.74452066 | 0.00723106 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776421236.csv | 1776421236.22923255 | 0.00649733 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776421237.csv | 1776421237.74712110 | 0.00578796 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776421239.csv | 1776421239.41841841 | 0.00510903 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776421241.csv | 1776421241.21377444 | 0.00446644 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776421243.csv | 1776421243.21183586 | 0.00386409 |              8 | mobilenet_v3_small |
| adaptation_log_agent_2_1776421245.csv | 1776421245.34856319 | 0.00330085 |              8 | mobilenet_v3_small |
| adaptation_log_agent_3_1776421235.csv | 1776421235.20177627 | 0.00630958 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776421237.csv | 1776421237.21653295 | 0.00514733 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776421239.csv | 1776421239.27367878 | 0.00411451 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776421241.csv | 1776421241.62883925 | 0.00322594 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776421243.csv | 1776421243.80081153 | 0.00248042 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776421246.csv | 1776421246.66419792 | 0.00185497 |              8 | efficientnet_b0    |
| adaptation_log_agent_3_1776421249.csv | 1776421249.45802975 | 0.00133044 |              8 | efficientnet_b0    |

## Caption Notes (Use in Paper)
- Pilot run configuration: 60 frames, 3 agents, 180 agent-decisions.
- Baseline is analytically derived from the same run (heavy-only path), not from a separate hardware-timed baseline run.
- Cross-agent reuse in this run: 178/178 (100.00%).
