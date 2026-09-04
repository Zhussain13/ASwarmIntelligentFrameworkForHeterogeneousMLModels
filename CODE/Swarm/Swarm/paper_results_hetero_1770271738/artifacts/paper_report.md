# Swarm Knowledge-Transfer System — Implementation Report

Date: 2026-02-05

This document collects the implementation details, experimental artifacts and results produced by the prototype "Swarm Inference-Time Knowledge Transfer" system. It is intended as a near-complete report suitable for inclusion in a Methods / Results appendix of a paper. The report references generated figures and CSVs created by the experiments; those files are in `paper_results_dashcam_full_manual/`.

## Executive summary

- We implemented a decentralized run-time knowledge-transfer system where low-capacity agents consult a shared CollectiveMemory of embeddings (KnowledgePackets) and, when a consensus match is found, skip expensive local inference and reuse the peer-provided label/embedding.
- The prototype was evaluated on a dashcam clip (`data/video_for_swarm.avi`, 795 frames) with a 3-agent local experiment harness using a FakeRedis store. The run artifacts are in `paper_results_dashcam_full_manual/`.
- Key result (one representative run): of 2,385 agent-frame decisions (795 frames × 3 agents), 2,341 were skipped and 44 required heavy inference (skip rate ≈ 98.2%). Skipped-match similarity has median ≈ 0.999; a few low-sim skipped cases (< 0.98) exist and are saved as sample frames for inspection.

## Repository artifacts (where to find figures and CSVs)

- Results (this run): `paper_results_dashcam_full_manual/`
  - `metrics/metrics_agent_*.csv` — per-agent per-frame metrics (columns: `agent_id, frame, decision, sim, stored_label`)
  - `metrics/swarm_aggregate_*.csv` — aggregated swarm CSVs
  - `plots/` — plots produced by `tools/plot_metrics.py` (decision timelines, cumulative bytes) and `plots/samples/` contains per-frame PNG snapshots for manual inspection
  - `artifacts/swarm_kt/` — code snapshot used for this run
  - `logs/` — run logs and plotting logs

Paths referenced in the report below assume the repo root is `/home/dinesh/Documents/Heterogenous-System`.

## Implementation overview (modules)

The code is organized as a small library `swarm_kt/` plus runner scripts:

- `swarm_kt/knowledge_packet.py`
  - KnowledgePacket dataclass that carries: embedding (base64-encoded float32), `stored_label` (e.g., "Person"), provenance fields (model id / timestamp / extra metadata), and optional confidence/trust fields.
  - JSON serialization helpers used by the comms layer.

- `swarm_kt/collective_memory.py`
  - Redis-backed CollectiveMemory abstraction. Stores KnowledgePackets with TTL, supports redundancy checks and `consensus_match()`.
  - `consensus_match(current_embedding, k, accept_weight)` aggregates candidate packets by label weighting them with stored `trust * confidence` and returns a consensus label if the aggregated weight exceeds `accept_weight`.
  - Provides a `get_best_match()` helper used in decision logic.

- `swarm_kt/agent_communication.py`
  - MQTT publish/subscribe wrapper (paho-mqtt). Runs tolerant of a missing broker for local experiments and always writes to the CollectiveMemory so experiments can run offline. In production this is the messaging bridge.

- `swarm_kt/encoder.py`
  - Encoder class which by default uses a ResNet-18 backbone (from torchvision) as a feature extractor plus a small trainable projection head.
  - `get_feature_and_embedding(frame)` returns a lightweight embedding for matching.
  - `adapt_projection(features, target_embeddings, epochs, lr)` implements an online adaptation loop (MSE loss) to align a local projection head to incoming target_embeddings.
  - Fallback non-deep embedding: a color-histogram pipeline, for experiments where PyTorch is not available.

- Runner / harness scripts
  - `run_three_agents_local.py` — in-process threaded 3-agent harness that uses `fakeredis` to simulate shared CollectiveMemory; writes per-agent CSVs and an aggregate CSV.
  - `simulate_swarm.py` — a multi-process simulation harness (older).
  - `tools/plot_metrics.py` — plotting utility for decision timeline and cumulative bytes plots.

## How knowledge transfer works (detailed)

1. Per-frame processing by an agent:
   - The agent computes a lightweight embedding for the current frame using `Encoder.get_feature_and_embedding(frame)`.
   - The agent calls `CollectiveMemory.consensus_match(embedding, k=..., accept_weight=...)`.
     - CollectiveMemory searches recent KnowledgePackets (bounded by TTL and look-back budget), computes cosine similarity to the current embedding, and selects the top-k candidates.
     - It groups candidates by `stored_label` and computes an aggregated score for each label using stored `trust * confidence` (the system stores a `trust` value per source and `confidence` per packet). The aggregated score is the consensus weight.
     - If the top label's aggregated weight ≥ `accept_weight` (and the similarity meets tie-in thresholds), the agent adopts the `stored_label` and marks the decision as `skipped` (no heavy inference).
   - If no adequate consensus exists, the agent runs local heavy inference (its local detector), produces a label (`stored_label`) and optionally publishes a new KnowledgePacket into the CollectiveMemory (publish path).

2. Publishing KnowledgePackets:
   - A KnowledgePacket contains the (projected) embedding, the `stored_label`, detector confidence, model provenance (model name or ID, weights/timestamp), and optional trust score.
   - Packets are serialized (embedding→base64 JSON) and stored in CollectiveMemory with a TTL. The publishing agent may also broadcast via MQTT in real deployments.

3. Online adaptation:
   - When a low-capacity agent accepts an external target embedding (i.e., skips), the pair (its own local feature, the accepted target embedding) can be queued for projection-head adaptation. Periodically the agent trains its projection head to reduce MSE between projected local features and the target embeddings — this reduces domain mismatch and increases future skip likelihood.

Overall effect: the swarm shares compact embedding+label packets that let agents avoid repeated heavy inference for consistent objects across frames or agents.

## Models used by the agents

- Encoder backbone: ResNet-18 (torchvision pretrained weights used by default). The encoder returns a feature vector which is passed through a small projection head to produce the final embedding transported in KnowledgePackets.
- Projection head: small MLP (a few linear layers + ReLU), trainable online via `adapt_projection` (MSE objective to match target embeddings).
- Heavy inference model: each agent runs a local heavy detector when skipping is not possible. The codebase treats this as a configurable detector (not hard-coded). In experiments we used the project's detector that outputs labels such as "Person" (the produced `stored_label` values). If you want an explicit heavy detector (e.g., Faster-RCNN pretrained) we can integrate and evaluate it as the baseline.
- Fallback embedding: when PyTorch is absent, a color-histogram embedding is used as a conservative fallback.

Note: the code stores `stored_label` strings and not full bounding box geometry. The current metrics use label-level correctness; adding bounding-box-level provenance is a recommended improvement.

## Experimental setup (the run reported here)

- Video: `data/video_for_swarm.avi` (795 frames). A public OpenCV test dashcam clip was used for quick evaluation.
- Agents: 3 agents (the `run_three_agents_local.py` harness). The harness uses `fakeredis` so all three agents share one in-memory CollectiveMemory for reproducible local trials.
- Run configuration: default `consensus_k`, `accept_weight` and TTL from `swarm_kt/config.py` (the harness writes a snapshot of code in `paper_results_dashcam_full_manual/artifacts/`). The exact hyperparameter values used for this run are visible in the run logs and can be captured automatically via a manifest (recommended).

## Quantitative results (representative run)

Latest per-agent files used: (latest timestamped CSVs in `paper_results_dashcam_full_manual/metrics/`)

Summary (selected numbers):

| Metric | Value |
|---|---:|
| Frames in video | 795 |
| Agents | 3 |
| Total agent-decisions | 2,385 |
| skipped | 2,341 |
| heavy | 44 |
| overall skip rate | 98.2% |

Per-agent heavy counts (latest full run):
- agent_1: heavy=15, skipped=780
- agent_2: heavy=21, skipped=774
- agent_3: heavy=8, skipped=787

Similarity statistics for skipped decisions (embedding similarity between current embedding and chosen packet):
- mean ≈ 0.99794, std ≈ 0.00354, min ≈ 0.9565, median ≈ 0.99914
- A small number of skipped events have sim < 0.98 (examples saved under `paper_results_dashcam_full_manual/plots/samples/`)

Conflict frames (frames where some agents did heavy and others skipped): examples include frame numbers: 1, 37, 67, 74, 106, 110, 147, 188, 222, 224.

Notes about metrics CSVs: per-agent CSV columns are `agent_id, frame, decision, sim, stored_label`. Heavy decisions contain NaN for `sim` and `stored_label` (the CSVs were emitted before storing heavy-output metadata), while skipped rows contain `sim` and `stored_label`.

## Figures available (and where to put them in a paper)

- Decision timeline (per-agent): `paper_results_dashcam_full_manual/plots/*_decisions.png` — place as a multi-panel figure showing the three agent timelines.
- Cumulative bandwidth (bytes) plot: `paper_results_dashcam_full_manual/plots/*_cum_bytes.png` — place alongside the decision timeline to show savings.
- Similarity distribution plots: generated by `tools/plot_metrics.py` — locate in `plots/` (histograms / boxplots).
- Qualitative examples: `paper_results_dashcam_full_manual/plots/samples/frame_*.png` — use a 2×3 grid of selected frames (correct skip, correct heavy, failure / low-sim skip).

## What can be shown in the paper (concrete list)

1. System diagram and algorithm pseudocode (deterministic description of consensus and accept rule).
2. Per-agent decision timelines and cumulative bytes saved vs a heavy-only baseline.
3. Aggregate summary table (skip rate, heavy counts, sim statistics) for the dashcam run.
4. Qualitative panels showing representative correct skip, correct heavy, and failure modes.
5. Ablation suggestions and preliminary ablation (if you run additional settings): how skip rate and false-accept vary with `accept_weight` and `consensus_k`.
6. (After adding logs) Online adaptation learning curve (projection-head loss vs steps) showing the benefit of adaptation to reduce future heavy inferences.

## Simulated heterogeneity (what we added)

- To demonstrate cross-model behavior without introducing heavy new detector dependencies, we added a "simulated heterogeneity" mode to the local 3-agent harness. In that mode each agent is initialized with a different projection embedding dimension (small/medium/large). This is a fast proxy for true heterogeneity and is useful to exercise the consensus and adaptation plumbing.
- How it's implemented: `run_three_agents_local.py` now accepts `--hetero simulated` and will select per-agent embedding dimensions (e.g., 128 / 256 / 512). The `Encoder` projection head is created per-agent with the specified embedding dimension, and adaptation logs are written to `adaptation_log_<agent>_<ts>.csv` when projection-head adaptation runs.
- Where to run: example command

```bash
.venv/bin/python run_three_agents_local.py --num-agents 3 --mode video --video data/video_for_swarm.avi --max-frames 200 --stagger 0.1 --hetero simulated
```

The simulated heterogeneity mode is described in the results and experiments sections; it is a fast way to validate that the consensus + adaptation loop can align different projection head parameterizations. For the paper we recommend running the true heterogeneity experiments (different backbones) as a follow-up; simulated heterogeneity is a useful intermediate step and is now included in the repo.

## Cleaning up transient files

To make working with the repo tidier we added `scripts/cleanup_artifacts.sh` which moves loose `metrics_*.csv`, `swarm_aggregate_*.csv`, `adaptation_log_*.csv` and tarballs into a timestamped folder `paper_results_cleanup_<ts>/`. This avoids accidental deletion while keeping artifacts grouped for archival.


## Limitations and missing pieces (to address before submission)

- No ground-truth bounding boxes were used for automatic FAR/FRR computation. Add either manual annotation for a sampled subset or run a trusted heavy detector on saved skipped frames.
- Per-decision provenance metadata is limited. Add `packet_id` / `source_agent` / `packet_ts` and `trust_at_use` to the metrics CSV to prove cross-agent transfer.
- Adaptation loss is not logged; implement per-adaptation-step logging and track downstream effect on skip rate/accuracy.
- The heavy inference metadata (label and confidence) is not recorded in the CSV for heavy decisions — record this so you can later compute confusion matrices.
- Experiments need repeats and hyperparameter sweeps to support statistical claims.

## Suggested additions (code edits and experiments)

1. Add provenance to per-decision logs (modify the decision logging call to include `used_packet_id`, `used_packet_source`, `used_packet_ts`, `used_packet_trust`).
2. Add `adaptation_log_agent_<id>_<ts>.csv` writing `step, loss, timestamp, samples_used` during `Encoder.adapt_projection()`.
3. Create `scripts/eval_skipped_with_heavy.py` that loads an off-the-shelf heavy detector (e.g., a torchvision Faster R-CNN or a pre-trained SSD), runs it on all skipped frames or a stratified sample, and compares predicted labels to the `stored_label` used by the agent.
4. Run a parameter sweep across `accept_weight` and `consensus_k` and plot skip-rate vs FAR/FRR.
5. Run 3–5 repeats per configuration and report mean ± std for critical numbers.

## Potential reviewer questions and short answers (to include in rebuttal / appendix)

1. How do you measure correctness of skip? — We'll run a trusted heavy detector on sampled skipped frames to compute FAR/FRR (script recommended above) and include the numbers.
2. Why do some agents do heavy while others skip? — Timing and local memory differences, packet arrival/staleness, and heterogeneity of local detectors. We can add per-packet timestamps + TTL plots to demonstrate packet freshness and show how timing explains the disagreement.
3. How sensitive is the method to hyperparameters? — Provide sweep plots (skip-rate vs threshold) and a simple heuristic to choose thresholds.
4. Is this general to other modalities? — The approach is embedding-agnostic: as long as agents can produce a compact, comparable embedding and provenance metadata, the same consensus rule applies. We will state this as a generalization and provide code hooks.

## Reproducibility & how to reproduce these results

1. Ensure the repository is on a recent commit and you have the project's virtualenv active (`.venv` used in these experiments). Confirm `python --version` and `pip freeze` similar to the run logs.

2. Run the 3-agent dashcam job:

```bash
.venv/bin/python run_three_agents_local.py --num-agents 3 --mode video --video data/video_for_swarm.avi --max-frames 795 --stagger 0.2
```

3. The results will be written in the repo root as `metrics_agent_*.csv` and `swarm_aggregate_*.csv`. Use `tools/plot_metrics.py` to generate/from CSVs.

4. For the exact run above, open `paper_results_dashcam_full_manual/` and inspect `plots/` and `metrics/`.

## Appendix: exact numbers from the representative run (copyable)

- Total agent-frames: 2,385 (3 × 795)
- skipped: 2,341
- heavy: 44
- Per-agent heavy: agent_1=15, agent_2=21, agent_3=8
- Skipped sim: mean ≈ 0.99794, std ≈ 0.00354, min ≈ 0.9565

## Next actions I can implement (pick one and I will run and attach results)

- (A) Add per-decision provenance columns and adaptation-loss logging, then run a 200-frame reproducible run and attach updated CSVs + adaptation plots.
- (B) Run a heavy-detector evaluation on the saved skipped-frame PNGs and produce FAR/FRR and sample corrected labels (I can implement `scripts/eval_skipped_with_heavy.py`).
- (C) Run a small hyperparameter sweep and produce skip-rate vs sim-threshold plots.
- (D) Produce a DOCX version of this report (I can generate it if you prefer `.docx` instead of `.md`).

Please choose one and I will implement it and attach updated artifacts to the `paper_results_...` folder.

---

Report generated automatically from the repository state and the latest run results. If you want a `.docx` file instead of Markdown, I can produce one and place it at `paper_results_dashcam_full_manual/paper_report.docx`.
