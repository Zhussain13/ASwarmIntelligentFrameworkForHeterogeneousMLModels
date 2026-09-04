# Publication-Ready Tables

Source: `paper_results_hetero_1770271738/metrics/` | Baseline assumption: heavy-only inference for all decisions.

## Table 1. Main Results (Baseline vs Proposed)
| metric                | baseline_heavy_only   | proposed_swarm_kt   | gain                     |
|:----------------------|:----------------------|:--------------------|:-------------------------|
| Total decisions       | 60                    | 60                  | Context metric           |
| Heavy inference calls | 60                    | 3                   | 95.00% reduction         |
| Skipped decisions     | 0                     | 57                  | +57                      |
| Skip rate (%)         | 0.00                  | 95.00               | +95.00 pp                |
| Heavy compute factor  | 1.00x                 | 0.05x               | 20.00x fewer heavy calls |

## Table 2. Per-Agent Breakdown
| agent_id   |   total |   heavy |   skipped |   skip_rate_percent |   heavy_rate_percent |
|:-----------|--------:|--------:|----------:|--------------------:|---------------------:|
| agent_1    |      20 |       1 |        19 |               95.00 |                 5.00 |
| agent_2    |      20 |       1 |        19 |               95.00 |                 5.00 |
| agent_3    |      20 |       1 |        19 |               95.00 |                 5.00 |

## Table 3. Similarity Quality (Skipped Decisions)
| stat   |    value |
|:-------|---------:|
| mean   | 0.996608 |
| std    | 0.001942 |
| min    | 0.992554 |
| median | 0.997265 |

## Table 4. Knowledge Transfer Usage
Receiver agent (rows) vs packet source agent (columns).

| agent_id   |   agent_1 |   agent_2 |   agent_3 |
|:-----------|----------:|----------:|----------:|
| agent_1    |        19 |         0 |         0 |
| agent_2    |         0 |        19 |         0 |
| agent_3    |         0 |         0 |        19 |

| transfer_type             |   count |
|:--------------------------|--------:|
| self-packet reuse         |      57 |
| cross-agent reuse         |       0 |
| cross-agent reuse percent |       0 |

## Table 5. Adaptation Log Summary
| file                                  |           timestamp |       loss |   samples_used |
|:--------------------------------------|--------------------:|-----------:|---------------:|
| adaptation_log_agent_1_1770270449.csv | 1770270449.95051575 | 0.00005365 |              8 |
| adaptation_log_agent_1_1770270461.csv | 1770270461.36045456 | 0.00007571 |              8 |
| adaptation_log_agent_2_1770270450.csv | 1770270450.04606438 | 0.00002930 |              8 |
| adaptation_log_agent_2_1770270460.csv | 1770270460.64081359 | 0.00004306 |              8 |
| adaptation_log_agent_3_1770270453.csv | 1770270453.93535519 | 0.00000595 |              8 |
| adaptation_log_agent_3_1770270462.csv | 1770270462.23950863 | 0.00001092 |              8 |

## Caption Notes (Use in Paper)
- Pilot run configuration: 20 frames, 3 agents, 60 agent-decisions.
- Baseline is analytically derived from the same run (heavy-only path), not from a separate hardware-timed baseline run.
