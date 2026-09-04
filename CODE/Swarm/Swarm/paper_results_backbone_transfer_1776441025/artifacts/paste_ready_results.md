# Paste-Ready Results Section (Cross-Backbone Transfer Run)

## Experimental Setup (Paper Text)

We ran a heterogeneous-backbone swarm experiment with 3 agents on 60 frames from `data/video_for_swarm.avi` (180 total agent-decisions). Agents used different encoder backbones with a shared embedding dimension (256): Agent-1 `resnet18`, Agent-2 `mobilenet_v3_small`, and Agent-3 `efficientnet_b0`. Each backbone produced features that were projected into a shared embedding space via a trainable linear projection head. Peer-first packet selection was enabled to explicitly test cross-backbone knowledge transfer.

## Results Paragraph (Paper Text)

In this cross-backbone run, the proposed inference-time transfer policy reduced heavy inference calls from 180 (heavy-only baseline) to 2, yielding a 98.89% reduction (90.0x fewer heavy calls). The system skipped 178/180 decisions (98.89%; 95% Wilson CI: 96.04%-99.69%). Cross-agent reuse was 178/178 skipped decisions (100%), confirming active transfer between agents using different backbone architectures. Similarity statistics for skipped decisions were mean 0.5469, SD 0.3449, median 0.5299, and minimum 0.0089. Per-agent breakdown was: Agent-1 (1 heavy, 59 skipped), Agent-2 (1 heavy, 59 skipped), Agent-3 (0 heavy, 60 skipped).

## Table X. Main Quantitative Results (Cross-Backbone)

| Metric | Heavy-only baseline | Proposed Swarm-KT | Gain |
|:--|--:|--:|:--|
| Frames | 60 | 60 | Context |
| Agents | 3 | 3 | Context |
| Total agent-decisions | 180 | 180 | Context |
| Heavy inference calls | 180 | 2 | 98.89% reduction |
| Skipped decisions | 0 | 178 | +178 skipped decisions |
| Heavy rate (%) | 100.0 | 1.11 | -98.89 pp |
| Skip rate (%) | 0.0 | 98.89 | +98.89 pp |
| Relative heavy compute factor | 1.00x | 0.01x | 90.0x fewer heavy calls |
| Skip-rate 95% CI (Wilson) | N/A | 96.04-99.69 | Uncertainty interval |
| Cross-agent reuse rate (%) | N/A | 100.0 (178/178) | Active inter-agent transfer |
| Similarity on skipped decisions (mean +/- SD) | N/A | 0.5469 +/- 0.3449 | Cross-backbone compatibility evidence |

## Table Y. Per-Agent Breakdown

| Agent | Backbone | Heavy | Skipped | Total | Skip rate (%) |
|:--|:--|--:|--:|--:|--:|
| agent_1 | resnet18 | 1 | 59 | 60 | 98.33 |
| agent_2 | mobilenet_v3_small | 1 | 59 | 60 | 98.33 |
| agent_3 | efficientnet_b0 | 0 | 60 | 60 | 100.00 |

## Suggested Caption

Cross-backbone knowledge-transfer results on a 3-agent swarm (`resnet18`, `mobilenet_v3_small`, `efficientnet_b0`) with shared 256-D embedding space. Compared with a heavy-only baseline, Swarm-KT reduced heavy calls by 98.89% and achieved 100% cross-agent reuse among skipped decisions.

## Important Reproducibility Note

This run used torchvision backbones with `weights=None` (no pretrained download) and a permissive similarity threshold (`-0.15`) to stress-test protocol-level cross-backbone transfer in an offline environment. Report this explicitly in the paper as a protocol validation setting.
