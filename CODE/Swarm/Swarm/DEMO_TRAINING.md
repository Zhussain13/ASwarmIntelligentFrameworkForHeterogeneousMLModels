# Training and Adaptation Demo

This project should be presented as an **inference-time knowledge-transfer framework**, not as a full offline CNN training pipeline. The training evidence in this codebase is the **cross-backbone calibration / projection-head adaptation** step that aligns heterogeneous model embeddings into a shared space so agents can reuse each other's knowledge packets.

## What To Say In The Demo

- **Paper intent:** heterogeneous robots cannot directly share raw model internals because their perception models may use different architectures.
- **Implemented idea:** each agent converts perception output into a model-agnostic `KnowledgePacket` with label, confidence, embedding, timestamp, and agent identity.
- **Retraining-ready packet:** each heavy detection now stores label, confidence, bounding-box coordinates, detection id/timestamp, full-frame image payload, source model, and embedding.
- **Collective behavior:** agents publish packets into shared memory and skip redundant heavy inference when a trusted similar packet already exists.
- **Peer validation:** when another agent reuses a peer packet, it can validate the shared image and produce a best-model/best-confidence retraining candidate.
- **Live retraining:** validated packet crops are used to train lightweight person/background student heads on top of frozen backbones.
- **Training/adaptation part:** warm-up calibration frames train the small projection head of non-reference agents so different backbones map features into a common embedding space.
- **Important honesty point:** the full CNN backbones are not retrained here; only the projection layer used for shared embedding alignment is adapted.

## What Is Trained

The calibration path trains the projection head inside the encoder:

- Reference agent generates target embeddings from warm-up frames.
- Other agents extract backbone features from the same warm-up frames.
- Their projection heads are trained to map local features toward the reference embedding space.
- The output proof is `artifacts/calibration_log.csv`, which records status, samples, epochs, learning rate, and final loss.

This matches the paper's emphasis on avoiding raw-data sharing and avoiding full model-parameter exchange while still enabling heterogeneous agents to collaborate.

## What Is Retrained Live

After the calibrated video run, `tools/retrain_from_packets.py` trains one lightweight student detection head per backbone:

- Positive samples come from validated person bounding-box crops.
- Negative samples come from background regions in the same packet images.
- The CNN backbones remain frozen.
- The saved heads are written to `artifacts/student_heads/*.npz`.
- Before/after loss and accuracy are shown in the dashboard.

## Before vs After Calibration

Use the existing strict suite as the cleanest evidence:

- Results file: `paper_results_true_suite_1776444198/artifacts/suite_summary.md`
- No calibration: `0.0%` mean cross-agent reuse.
- With calibration: about `73.7%` mean cross-agent reuse.
- Interpretation: after projection-head calibration, agents with different backbones can reuse peer knowledge packets instead of relying only on self-generated packets.

## Live Frontend Demo

Run the frontend from the repository root:

```powershell
python tools\demo_frontend.py
```

Then open:

```powershell
http://127.0.0.1:7860
```

The frontend asks what real media you have:

- **Video file:** `.avi`, `.mp4`, `.mov`, `.mkv`, or `.webm`
- **Image folder:** `.jpg`, `.png`, `.bmp`, or `.webp` frames/photos, automatically converted into a temporary demo video

From the browser, run:

- **Before vs after calibration comparison** for the strongest demo.
- **Calibration only** when you want to focus on training/adaptation logs.
- **No-calibration baseline only** when you want to show the failure mode.

After a successful run, the frontend refreshes `demo_runs\index.html` with guide-friendly cards, before/after comparison, calibration loss, decision timelines, and transfer matrices.

Useful scenarios:

- `synthetic`: quick heterogeneous-agent demo with simulated heavy inference.
- `video-calib`: video demo with calibration enabled.
- `video-no-calib`: video demo with calibration disabled.
- `compare-video`: runs both video cases and prints a comparison.
- `suite`: runs the full multi-seed strict suite.

## Command-Line Fallback

Use these only if the frontend is not needed.

Video demo with calibration:

```powershell
$env:TORCH_HOME="d:\Swarm\.torch_cache"
python paper_results_hetero_1770271738\artifacts\run_three_agents_local.py --mode video --video data\video_for_swarm.avi --max-frames 80 --num-agents 3 --hetero-backbone diverse --prefer-peer-transfer --use-pretrained-backbone --heavy-detector opencv_hog --output-dir demo_runs\video_calib --calibrate-frames 15 --calibrate-epochs 12
```

Video demo without calibration:

```powershell
$env:TORCH_HOME="d:\Swarm\.torch_cache"
python paper_results_hetero_1770271738\artifacts\run_three_agents_local.py --mode video --video data\video_for_swarm.avi --max-frames 80 --num-agents 3 --hetero-backbone diverse --prefer-peer-transfer --use-pretrained-backbone --heavy-detector opencv_hog --output-dir demo_runs\video_no_calib --calibrate-frames 0 --calibrate-epochs 12
```

Full strict suite:

```powershell
python tools\run_true_experiment_suite.py --output-root d:\Swarm --max-frames 80 --video d:\Swarm\data\video_for_swarm.avi --seeds 7,17,27
```

Default real-video comparison wrapper:

```powershell
python tools\run_training_demo.py --video d:\Swarm\data\video_for_swarm.avi --max-frames 80
```

Build or refresh only the dashboard:

```powershell
python tools\build_demo_dashboard.py
```

## Choosing Videos For Better Calibration Variation

The frontend asks for the type of video/images before running. The calibration and peer-reuse scores will vary more when the input media differs in scene structure. Good options:

- **Stable indoor corridor/classroom:** usually produces high similarity and strong skip rates.
- **Crowded walking scene:** produces more motion and object variation, often lowering similarity.
- **Lighting-change video:** tests robustness and may make calibration improvement more visible.
- **Different camera angle or outdoor scene:** useful for showing heterogeneous backbones behave differently.
- **Mixed scene video:** best for a final showcase because scores are less artificially perfect.

Avoid using synthetic input for the guide demo. Synthetic runs are useful only for debugging because they do not feel like a real robot/perception showcase.

## Files To Show Teachers

- `DEMO_TRAINING.md`: explanation of the training/adaptation story.
- `demo_runs/*/artifacts/run_manifest.json`: exact configuration for each run.
- `demo_runs/*/artifacts/calibration_log.csv`: proof of projection-head training when calibration is enabled.
- `demo_runs/*/metrics/metrics_agent_*.csv`: per-agent heavy/skipped decisions.
- `demo_runs/*/metrics/cross_agent_transfer.csv`: peer reuse matrix.
- `demo_runs/*/metrics/cross_backbone_transfer.csv`: reuse across heterogeneous backbones.
- `demo_runs/*/artifacts/knowledge_packets_*.jsonl`: full rich packets with image payload and coordinates.
- `demo_runs/*/artifacts/packet_validations_*.jsonl`: peer checks of shared packet images.
- `demo_runs/*/artifacts/validated_training_candidates.json`: best metadata selected for future retraining.
- `demo_runs/*/artifacts/retraining_metrics.csv`: before/after metrics for live student-head retraining.
- `demo_runs/*/artifacts/student_heads/*.npz`: saved lightweight retrained heads.
- `paper_results_true_suite_1776444198/artifacts/suite_summary.md`: already-generated multi-seed evidence.

## Suggested Presentation Flow

1. Show the problem from the paper: heterogeneous models cannot share architecture-specific internals.
2. Show the architecture: encoder -> knowledge packet -> collective memory -> peer reuse decision.
3. Show the calibration log: this is the training/adaptation evidence.
4. Show before/after metrics: no calibration vs calibration.
5. Conclude that the system demonstrates model-agnostic knowledge transfer during inference, with lightweight adaptation for heterogeneous backbones.
