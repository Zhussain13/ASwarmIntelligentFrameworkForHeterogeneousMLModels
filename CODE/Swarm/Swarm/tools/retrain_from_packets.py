"""Train lightweight per-backbone student heads from validated swarm packets.

This is intentionally not full CNN fine-tuning. It trains a small detection
head on top of each frozen backbone using validated packet crops:
- positive samples: detected person bounding boxes
- negative samples: background crops from the same packet images

Usage:
    python tools/retrain_from_packets.py --results-dir demo_runs/video_calib
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_CODE = ROOT / "paper_results_hetero_1770271738" / "artifacts"
if str(ARTIFACT_CODE) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_CODE))

from swarm_kt.encoder import Encoder  # noqa: E402


BACKBONES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")


def _decode_image(payload: Dict[str, object]) -> Optional[np.ndarray]:
    if not payload or payload.get("format") != "jpeg" or not payload.get("b64"):
        return None
    try:
        import cv2

        raw = base64.b64decode(str(payload["b64"]).encode("ascii"))
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _clip_box(box: Dict[str, object], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x = max(0, int(float(box.get("x", 0))))
    y = max(0, int(float(box.get("y", 0))))
    w = max(1, int(float(box.get("w", 0))))
    h = max(1, int(float(box.get("h", 0))))
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x2 <= x or y2 <= y:
        return None
    return x, y, x2, y2


def _crop(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def _jitter_box(box: Tuple[int, int, int, int], width: int, height: int, scale: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * scale)
    dy = int(bh * scale)
    return (
        max(0, x1 + random.randint(-dx, dx)),
        max(0, y1 + random.randint(-dy, dy)),
        min(width, x2 + random.randint(-dx, dx)),
        min(height, y2 + random.randint(-dy, dy)),
    )


def _negative_boxes(
    positive: Tuple[int, int, int, int],
    width: int,
    height: int,
    count: int,
) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = positive
    bw = max(24, x2 - x1)
    bh = max(24, y2 - y1)
    boxes = []
    candidates = [
        (0, 0, min(width, bw), min(height, bh)),
        (max(0, width - bw), 0, width, min(height, bh)),
        (0, max(0, height - bh), min(width, bw), height),
        (max(0, width - bw), max(0, height - bh), width, height),
    ]
    for box in candidates:
        bx1, by1, bx2, by2 = box
        if bx2 > bx1 and by2 > by1:
            boxes.append(box)
        if len(boxes) >= count:
            break
    return boxes


def _load_candidates(results_dir: Path) -> List[Dict[str, object]]:
    path = results_dir / "artifacts" / "validated_training_candidates.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _make_training_images(candidates: List[Dict[str, object]]) -> Tuple[List[np.ndarray], List[int]]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    random.seed(7)
    for candidate in candidates:
        label = str(candidate.get("label") or candidate.get("object_label") or "").lower()
        if label != "person":
            continue
        image = _decode_image(candidate.get("image") if isinstance(candidate.get("image"), dict) else {})
        coords = candidate.get("coordinates")
        if image is None or not isinstance(coords, dict):
            continue
        height, width = image.shape[:2]
        box = _clip_box(coords, width, height)
        if box is None:
            continue

        positive_boxes = [box]
        for _ in range(5):
            jittered = _jitter_box(box, width, height, 0.08)
            if jittered[2] > jittered[0] and jittered[3] > jittered[1]:
                positive_boxes.append(jittered)

        for pos_box in positive_boxes:
            crop = _crop(image, pos_box)
            if crop.size:
                images.append(crop)
                labels.append(1)

        for neg_box in _negative_boxes(box, width, height, count=6):
            crop = _crop(image, neg_box)
            if crop.size:
                images.append(crop)
                labels.append(0)
    return images, labels


def _candidate_positive_crops(candidates: List[Dict[str, object]]) -> List[Tuple[Dict[str, object], np.ndarray]]:
    crops: List[Tuple[Dict[str, object], np.ndarray]] = []
    for candidate in candidates:
        label = str(candidate.get("label") or candidate.get("object_label") or "").lower()
        if label != "person":
            continue
        image = _decode_image(candidate.get("image") if isinstance(candidate.get("image"), dict) else {})
        coords = candidate.get("coordinates")
        if image is None or not isinstance(coords, dict):
            continue
        height, width = image.shape[:2]
        box = _clip_box(coords, width, height)
        if box is None:
            continue
        crop = _crop(image, box)
        if crop.size:
            crops.append((candidate, crop))
    return crops


def _extract_embeddings(backbone: str, images: List[np.ndarray], use_pretrained: bool) -> np.ndarray:
    encoder = Encoder(embedding_dim=256, backbone_name=backbone, use_pretrained=use_pretrained)
    return _extract_embeddings_with_encoder(encoder, images)


def _extract_embeddings_with_encoder(encoder: Encoder, images: List[np.ndarray]) -> np.ndarray:
    embeddings = []
    for image in images:
        _, emb = encoder.get_feature_and_embedding(image)
        embeddings.append(np.asarray(emb, dtype="float32"))
    return np.stack(embeddings)


def _train_head(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    eval_x: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    import torch

    torch.manual_seed(7)
    x_t = torch.from_numpy(x.astype("float32"))
    y_t = torch.from_numpy(y.astype("float32")).view(-1, 1)

    model = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        before_logits = model(x_t)
        before_loss = float(loss_fn(before_logits, y_t).item())
        before_acc = float(((torch.sigmoid(before_logits) >= 0.5) == (y_t >= 0.5)).float().mean().item())
        before_eval = None
        if eval_x is not None and len(eval_x):
            before_eval = torch.sigmoid(model(torch.from_numpy(eval_x.astype("float32")))).view(-1).cpu().numpy()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        after_logits = model(x_t)
        after_loss = float(loss_fn(after_logits, y_t).item())
        after_acc = float(((torch.sigmoid(after_logits) >= 0.5) == (y_t >= 0.5)).float().mean().item())
        after_eval = None
        if eval_x is not None and len(eval_x):
            after_eval = torch.sigmoid(model(torch.from_numpy(eval_x.astype("float32")))).view(-1).cpu().numpy()

    metrics = {
        "samples": int(len(y)),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum()),
        "before_loss": before_loss,
        "after_loss": after_loss,
        "loss_delta": before_loss - after_loss,
        "before_accuracy": before_acc,
        "after_accuracy": after_acc,
        "accuracy_delta": after_acc - before_acc,
    }
    weights = {
        "weight": model.weight.detach().cpu().numpy(),
        "bias": model.bias.detach().cpu().numpy(),
    }
    eval_scores = None
    if before_eval is not None and after_eval is not None:
        eval_scores = (before_eval, after_eval)
    return metrics, weights, eval_scores


def retrain(results_dir: Path, epochs: int, lr: float, use_pretrained: bool) -> List[Dict[str, object]]:
    candidates = _load_candidates(results_dir)
    images, labels = _make_training_images(candidates)
    eval_crops = _candidate_positive_crops(candidates)
    eval_images = [crop for _, crop in eval_crops]
    artifacts_dir = results_dir / "artifacts"
    models_dir = artifacts_dir / "student_heads"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    if len(images) < 4 or len(set(labels)) < 2:
        rows.append(
            {
                "backbone": "all",
                "status": "skipped_insufficient_samples",
                "samples": len(images),
            }
        )
    else:
        y = np.asarray(labels, dtype="float32")
        prediction_rows: List[Dict[str, object]] = []
        for backbone in BACKBONES:
            encoder = Encoder(embedding_dim=256, backbone_name=backbone, use_pretrained=use_pretrained)
            x = _extract_embeddings_with_encoder(encoder, images)
            eval_x = _extract_embeddings_with_encoder(encoder, eval_images) if eval_images else None
            metrics, weights, eval_scores = _train_head(x, y, epochs=epochs, lr=lr, eval_x=eval_x)
            np.savez(models_dir / f"{backbone}_student_head.npz", **weights)
            rows.append(
                {
                    "backbone": backbone,
                    "status": "ok",
                    "epochs": epochs,
                    "lr": lr,
                    **metrics,
                    "model_path": str(models_dir / f"{backbone}_student_head.npz"),
                }
            )
            if eval_scores is not None:
                before_scores, after_scores = eval_scores
                for idx, (candidate, _) in enumerate(eval_crops):
                    prediction_rows.append(
                        {
                            "detection_id": candidate.get("detection_id"),
                            "label": candidate.get("label"),
                            "source_model": candidate.get("source_model"),
                            "best_model": candidate.get("best_model"),
                            "agreement_count": candidate.get("agreement_count"),
                            "backbone": backbone,
                            "before_student_confidence": float(before_scores[idx]),
                            "after_student_confidence": float(after_scores[idx]),
                            "delta": float(after_scores[idx] - before_scores[idx]),
                        }
                    )
        pred_json = artifacts_dir / "retraining_predictions.json"
        pred_csv = artifacts_dir / "retraining_predictions.csv"
        pred_json.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
        with pred_csv.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = sorted({key for row in prediction_rows for key in row.keys()})
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)

    out_json = artifacts_dir / "retraining_metrics.json"
    out_csv = artifacts_dir / "retraining_metrics.csv"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain lightweight student heads from swarm packet candidates")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--use-pretrained-backbone", action="store_true")
    args = parser.parse_args()

    rows = retrain(Path(args.results_dir), epochs=args.epochs, lr=args.lr, use_pretrained=args.use_pretrained_backbone)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
