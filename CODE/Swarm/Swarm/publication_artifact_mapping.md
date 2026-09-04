# Publication Artifact Mapping for Swarm Knowledge-Transfer System

This file lists exactly what to include in your paper submission, with recommended figure/table numbers and captions.

Current available run in this workspace:
- `paper_results_hetero_1770271738/` (pilot run: 20 frames, 3 agents)
- `paper_results_backbone_transfer_1776441025/` (cross-backbone run: 60 frames, 3 agents)
- `paper_results_true_suite_1776444198/` (strict pretrained suite: 6 runs, 80 frames/run, 3 seeds/condition)

Auto-generated publication tables for this run:
- `paper_results_hetero_1770271738/artifacts/publication_tables.md`
- `paper_results_hetero_1770271738/artifacts/publication_tables.tex`
- `paper_results_backbone_transfer_1776441025/artifacts/publication_tables.md`
- `paper_results_backbone_transfer_1776441025/artifacts/publication_tables.tex`
- `paper_results_backbone_transfer_1776441025/artifacts/paste_ready_results.md`
- `paper_results_backbone_transfer_1776441025/artifacts/paste_ready_results.tex`
- `paper_results_true_suite_1776444198/artifacts/suite_summary.md`
- `paper_results_true_suite_1776444198/artifacts/suite_condition_summary.csv`
- `paper_results_true_suite_1776444198/artifacts/suite_run_metrics.csv`
- `paper_results_true_suite_1776444198/artifacts/paste_ready_true_suite_results.md`
- `paper_results_true_suite_1776444198/artifacts/paste_ready_true_suite_results.tex`

## 1. Figures

### Figure 1: System Overview
- (Create a schematic/diagram if not already present; not auto-generated)
- Caption: "Decentralized swarm knowledge-transfer system: agents share KnowledgePackets via a collective memory, enabling low-capacity agents to skip heavy inference by trusting peer embeddings."

### Figure 2: Per-Agent Decision Timeline
- File: `plots/metrics_agent_1_*.png`, `plots/metrics_agent_2_*.png`, `plots/metrics_agent_3_*.png`
- Caption: "Per-agent decision timeline: blue = heavy inference, orange = skipped (trusted peer)."

### Figure 3: Aggregate Swarm Metrics
- File: `plots/swarm_aggregate_*.png`
- Caption: "Aggregate swarm skip rate and cumulative bandwidth usage over all frames."

### Figure 4: Cross-Agent Knowledge Transfer
- File: `plots/cross_agent_transfer_heatmap.png` (or `plots/transfer_usage.png` if present)
- Caption: "Cross-agent transfer matrix: how often each agent used packets from itself vs. others."

### Figure 5: Adaptation/Online Training Evidence
- File: `metrics/adaptation_log_agent_*.csv` (summarize in a plot or table)
- Caption: "Online adaptation: per-agent adaptation loss over time, showing projection head training."

### Figure 6: Sample Frames (Optional)
- File: `plots/samples/frame_*.png`
- Caption: "Sample frames: representative cases of high/low similarity and conflict."

## 2. Tables

### Table 1: Main Results (Baseline vs Proposed)
- File: `artifacts/publication_tables.md` (section: "Table 1. Main Results") or `artifacts/publication_tables.tex`
- Caption: "Compared with a heavy-only baseline, the proposed swarm inference-time transfer reduces heavy inference calls and increases skip rate."

### Table 2: Per-Agent Breakdown
- File: `artifacts/publication_tables.md` (section: "Table 2. Per-Agent Breakdown") or `artifacts/publication_tables.tex`
- Caption: "Per-agent heavy vs skipped decision counts and rates."

### Table 3: Similarity Quality on Skipped Decisions
- File: `artifacts/publication_tables.md` (section: "Table 3. Similarity Quality") or `artifacts/publication_tables.tex`
- Caption: "Distribution statistics of embedding similarity for skipped decisions."

### Table 4: Knowledge Transfer Usage
- File: `artifacts/publication_tables.md` (section: "Table 4. Knowledge Transfer Usage")
- Caption: "Receiver-by-source packet reuse matrix and self-vs-cross transfer counts."

### Table 5: Adaptation Log Summary
- File: `artifacts/publication_tables.md` (section: "Table 5. Adaptation Log Summary")
- Caption: "Adaptation events with loss and sample counts."

## 3. Supplementary/Methods
- File: `artifacts/swarm_kt/`, `artifacts/run_three_agents_local.py`, `artifacts/paper_report.md`
- Caption: "Code snapshot and implementation report for reproducibility."
- Repro script: `tools/run_true_experiment_suite.py` (one-command strict suite with pretrained backbones + summary CSVs).

## 4. How to Reference
- In your paper, refer to figures/tables by number and point to the corresponding file in the results folder.
- For adaptation, include a plot or summary table of adaptation loss (can be generated from the adaptation logs).
- For sample frames, select a few representative images for the appendix or supplement.
- If this pilot run is used in the main paper, explicitly label it as a pilot and include a limitation note that baseline is analytically derived (heavy-only path), not from a separate timed hardware baseline run.

---

This mapping ensures all required evidence, figures, and tables are included and clearly referenced for publication.
