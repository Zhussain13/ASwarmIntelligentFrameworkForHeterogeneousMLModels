"""Embedding encoder used to produce model-agnostic feature vectors.

Supports heterogeneous backbones across agents while projecting into a
shared embedding space for knowledge transfer.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Encoder:
    SUPPORTED_BACKBONES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")

    def __init__(
        self,
        embedding_dim: int = 256,
        use_torch: Optional[bool] = None,
        backbone_name: str = "resnet18",
        use_pretrained: bool = False,
    ):
        self.embedding_dim = int(embedding_dim)
        self._torch_available = False
        self.feature_extractor = None
        self.device = None
        self.backbone_name = (backbone_name or "resnet18").lower().strip()
        self.use_pretrained = bool(use_pretrained)
        self.out_dim = None

        try:
            import torch
            from torchvision import models

            self.torch = torch
            self.feature_extractor, self.out_dim = self._build_backbone(models)
            self.feature_extractor.eval()
            self.device = torch.device("cpu")
            self.feature_extractor.to(self.device)

            # Small trainable projection head into shared embedding space.
            self.proj = torch.nn.Linear(self.out_dim, self.embedding_dim)
            torch.manual_seed(0)
            if self.embedding_dim == self.out_dim:
                try:
                    with torch.no_grad():
                        self.proj.weight.copy_(torch.eye(self.out_dim))
                        self.proj.bias.zero_()
                except Exception:
                    pass
            self._torch_available = True
        except Exception as exc:
            logger.info(
                "Torch/torchvision unavailable or backbone init failed (%s); using fallback encoder",
                exc,
            )
            self.backbone_name = "fallback_histogram"

    def _safe_get_weights(self, models, model_key: str):
        if not self.use_pretrained:
            return None
        candidates = {
            "resnet18": "ResNet18_Weights",
            "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
            "efficientnet_b0": "EfficientNet_B0_Weights",
        }
        enum_name = candidates.get(model_key)
        if not enum_name or not hasattr(models, enum_name):
            return None
        enum_obj = getattr(models, enum_name)
        try:
            return enum_obj.DEFAULT
        except Exception:
            return None

    def _build_backbone(self, models) -> Tuple[object, int]:
        import torch

        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. Supported: {self.SUPPORTED_BACKBONES}"
            )

        weights = self._safe_get_weights(models, self.backbone_name)
        kwargs = {}
        if weights is not None:
            kwargs["weights"] = weights
        else:
            # Older torchvision compatibility.
            kwargs["pretrained"] = False

        if self.backbone_name == "resnet18":
            base = models.resnet18(**kwargs)
            feature_extractor = torch.nn.Sequential(*list(base.children())[:-1])
            return feature_extractor, 512

        if self.backbone_name == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(**kwargs)

            class MobileNetV3Feature(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.features = model.features
                    self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

                def forward(self, x):
                    x = self.features(x)
                    x = self.pool(x)
                    return x

            return MobileNetV3Feature(base), 576

        if self.backbone_name == "efficientnet_b0":
            base = models.efficientnet_b0(**kwargs)

            class EfficientNetFeature(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.features = model.features
                    self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

                def forward(self, x):
                    x = self.features(x)
                    x = self.pool(x)
                    return x

            return EfficientNetFeature(base), 1280

        raise ValueError(f"Unhandled backbone: {self.backbone_name}")

    def _torch_embedding(self, frame: np.ndarray) -> np.ndarray:
        import torch
        import cv2

        img = frame[..., ::-1]  # BGR -> RGB
        img = img.astype("float32") / 255.0
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            feat = self.feature_extractor(t).flatten(1).squeeze(0)
            emb = self.proj(feat)
            arr = emb.cpu().numpy().ravel().astype("float32")
        return arr

    def get_feature_and_embedding(self, frame: np.ndarray):
        """Return (feature_tensor, embedding_numpy).

        feature_tensor is a torch tensor on CPU (shape: out_dim), embedding is a
        normalized numpy float32 vector (shape: embedding_dim).
        """
        if not getattr(self, "_torch_available", False):
            return None, self._fallback_embedding(frame)

        import torch
        import cv2

        img = frame[..., ::-1]
        img = img.astype("float32") / 255.0
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            feat = self.feature_extractor(t).flatten(1).squeeze(0)
            emb = self.proj(feat)
            emb_np = emb.cpu().numpy().ravel().astype("float32")

        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm
        return feat.detach().cpu(), emb_np

    def adapt_projection(self, feature_tensors, target_embeddings, epochs: int = 3, lr: float = 1e-3):
        """Train the projection head to map features -> target embeddings."""
        if not getattr(self, "_torch_available", False):
            raise RuntimeError("Torch not available for adaptation")

        import torch

        self.proj.train()
        device = self.device
        feats = torch.stack(
            [
                f.to(device) if isinstance(f, torch.Tensor) else torch.tensor(f, device=device)
                for f in feature_tensors
            ]
        )
        targets = torch.from_numpy(np.asarray(target_embeddings, dtype="float32")).to(device)
        targets = targets / (targets.norm(dim=1, keepdim=True) + 1e-9)

        opt = torch.optim.Adam(self.proj.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        last_loss = None
        for _ in range(epochs):
            opt.zero_grad()
            preds = self.proj(feats)
            preds = preds / (preds.norm(dim=1, keepdim=True) + 1e-9)
            loss = loss_fn(preds, targets)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        self.proj.eval()
        return last_loss

    def save_proj(self, path: str):
        if not getattr(self, "_torch_available", False):
            raise RuntimeError("Torch not available")
        import torch

        torch.save(self.proj.state_dict(), path)

    def load_proj(self, path: str):
        if not getattr(self, "_torch_available", False):
            raise RuntimeError("Torch not available")
        import torch

        self.proj.load_state_dict(torch.load(path, map_location=self.device))

    def _fallback_embedding(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        chans = cv2.split(img)
        bins = 32
        feats = []
        for c in chans:
            h = cv2.calcHist([c], [0], None, [bins], [0, 256]).ravel()
            feats.append(h)
        arr = np.concatenate(feats).astype("float32")

        if arr.size >= self.embedding_dim:
            out = arr[: self.embedding_dim]
        else:
            out = np.zeros((self.embedding_dim,), dtype="float32")
            out[: arr.size] = arr
        return out

    def get_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Return a 1D float32 embedding vector (L2-normalized)."""
        if getattr(self, "_torch_available", False):
            arr = self._torch_embedding(frame)
        else:
            arr = self._fallback_embedding(frame)

        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.astype("float32")


__all__ = ["Encoder"]
