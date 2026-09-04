# Paste-Ready Results Section (Pilot Run)

Use this text directly in your manuscript.

## Results Text (Paragraph Form)

In a pilot heterogeneous swarm run on 20 video frames with 3 agents (60 total agent-decisions), the proposed inference-time knowledge-transfer method substantially reduced heavy inference usage compared with a heavy-only baseline. The system executed heavy inference in only 3/60 decisions (5.0%), while 57/60 decisions (95.0%) were handled by trusted reuse/skip logic, corresponding to a 95.0% reduction in heavy calls (20.0x fewer heavy inferences). The skip-rate 95% Wilson interval was 86.30%-98.29% for this pilot sample size. For skipped decisions, embedding similarity remained high (mean 0.996608, SD 0.001942, median 0.997265, min 0.992554), indicating consistent agreement quality. Per-agent behavior was balanced (each agent: 1 heavy, 19 skipped). Adaptation logs recorded 6 adaptation events with low reconstruction loss (mean 3.643e-05; min 5.95e-06; max 7.571e-05; 8 samples/event). In this pilot run, cross-agent packet reuse was not observed (0/57 skipped decisions), with skips dominated by self-packet reuse.

## Table X. Main Quantitative Results (Baseline vs Proposed)

| Metric | Heavy-only baseline | Proposed Swarm-KT | Gain |
|:--|--:|--:|:--|
| Frames | 20 | 20 | Context |
| Agents | 3 | 3 | Context |
| Total agent-decisions | 60 | 60 | Context |
| Heavy inference calls | 60 | 3 | 95.0% reduction |
| Skipped decisions | 0 | 57 | +57 skipped decisions |
| Heavy rate (%) | 100.0 | 5.0 | -95.0 pp |
| Skip rate (%) | 0.0 | 95.0 | +95.0 pp |
| Relative heavy compute factor | 1.00x | 0.05x | 20.0x fewer heavy calls |
| Skip-rate 95% CI (Wilson) | N/A | 86.30-98.29 | Pilot uncertainty interval |
| Similarity on skipped decisions (mean +/- SD) | N/A | 0.996608 +/- 0.001942 | High agreement consistency |
| Cross-agent reuse rate (%) | N/A | 0.0 (0/57) | No peer-packet reuse in this run |

## Table Y. Per-Agent Breakdown

| Agent | Heavy | Skipped | Total | Skip rate (%) |
|:--|--:|--:|--:|--:|
| agent_1 | 1 | 19 | 20 | 95.0 |
| agent_2 | 1 | 19 | 20 | 95.0 |
| agent_3 | 1 | 19 | 20 | 95.0 |

## Suggested Table Caption

Pilot-run quantitative comparison between a heavy-only baseline and the proposed Swarm-KT inference-time transfer policy on 20 frames and 3 agents (60 agent-decisions). The proposed method reduced heavy inference calls by 95.0% while maintaining high skipped-decision similarity; cross-agent reuse was not observed in this pilot run.

## Methods Note (Recommended Footnote)

Baseline values are analytically derived from the same run under the assumption that every decision would invoke heavy inference (heavy-only policy). These pilot results should be interpreted as early evidence and complemented by full-length multi-seed experiments.
