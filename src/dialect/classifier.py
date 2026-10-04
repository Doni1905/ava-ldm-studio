"""
src/dialect/classifier.py
==========================
Dialect classifier model: wav2vec2 feature extractor + MLP head.

Architecture
------------
We use a pre-trained wav2vec2-base model as a **frozen** feature extractor,
then train only a lightweight MLP classification head on top.  This gives
us:
  - Strong audio representations out-of-the-box (no dialect data needed
    for the backbone).
  - Fast training: only the head (~200K params) is updated.
  - CPU-compatible: wav2vec2-base can run on CPU for both extraction and
    inference.

The approach is deliberately adapted from — not copied from — the TypeScript
processor in ``src/lib/ldm/processor.ts``, which uses lexical dialect markers
(discourse particles: "dei", "machi", "da", etc.) to classify dialect from
text.  Here we instead operate on the **audio signal**, which is the correct
input for a speech-based dialect classifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP Head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Simple MLP classifier head placed on top of wav2vec2 features.

    Parameters
    ----------
    input_dim : int
        Dimension of the feature vector from the backbone (768 for wav2vec2-base).
    num_classes : int
        Number of dialect classes.
    hidden_dims : list[int]
        Sizes of hidden layers (e.g. [256, 128]).
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full Classifier (backbone + head)
# ---------------------------------------------------------------------------

class DialectClassifier(nn.Module):
    """
    Tamil dialect classifier: frozen wav2vec2 backbone + trainable MLP head.

    The backbone is loaded from HuggingFace Hub.  Its weights are frozen
    during training to avoid catastrophic forgetting and to keep training
    feasible on CPU.

    Parameters
    ----------
    config : dict
        Parsed ``configs/dialect.yaml``.
    num_classes : int
        Number of dialect labels.
    """

    def __init__(self, config: Dict[str, Any], num_classes: int) -> None:
        super().__init__()
        model_cfg  = config.get("model", {})
        head_cfg   = model_cfg.get("head", {})

        self.model_id    = model_cfg.get("feature_extractor_id", "facebook/wav2vec2-base")
        self.target_sr   = model_cfg.get("target_sample_rate", 16000)
        self.num_classes = num_classes

        # Hidden dims for MLP
        hidden_dims = head_cfg.get("hidden_dims", [256, 128])
        dropout     = head_cfg.get("dropout", 0.3)

        # Backbone (feature extractor)
        logger.info(f"Loading wav2vec2 backbone: {self.model_id}")
        from transformers import Wav2Vec2Model
        self.backbone = Wav2Vec2Model.from_pretrained(self.model_id)

        # Freeze ALL backbone parameters — we only train the head
        for param in self.backbone.parameters():
            param.requires_grad = False

        backbone_dim = self.backbone.config.hidden_size  # 768 for wav2vec2-base

        # Head
        self.head = MLPHead(
            input_dim=backbone_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        logger.info(
            f"DialectClassifier: backbone_dim={backbone_dim}, "
            f"num_classes={num_classes}, head_params={sum(p.numel() for p in self.head.parameters())}"
        )

    def extract_features(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract mean-pooled representation from wav2vec2 backbone.

        Parameters
        ----------
        waveform : Tensor of shape (batch, time)
        attention_mask : optional Tensor of shape (batch, time)

        Returns
        -------
        Tensor of shape (batch, hidden_dim)
        """
        with torch.no_grad():
            outputs = self.backbone(
                input_values=waveform,
                attention_mask=attention_mask,
            )
        # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
        # Mean-pool over time dimension
        hidden = outputs.last_hidden_state

        if attention_mask is not None:
            # Compute proper masked mean
            # wav2vec2 downsamples by ~320x, so we compute output mask
            out_len = hidden.shape[1]
            in_len  = waveform.shape[1]
            # Approximate: scale mask to output length
            scale   = out_len / in_len
            lengths = attention_mask.sum(dim=1).float() * scale
            lengths = lengths.long().clamp(min=1, max=out_len)
            feat = []
            for i, l in enumerate(lengths):
                feat.append(hidden[i, :l].mean(0))
            return torch.stack(feat)
        else:
            return hidden.mean(dim=1)

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning raw logits (batch, num_classes).
        """
        features = self.extract_features(waveform, attention_mask)
        logits   = self.head(features)
        return logits

    def predict_proba(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return softmax probabilities (batch, num_classes)."""
        logits = self.forward(waveform, attention_mask)
        return F.softmax(logits, dim=-1)

    def save(self, path: str | Path) -> None:
        """Save only the head weights (backbone is reloaded from HF Hub)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state_dict":  self.head.state_dict(),
                "num_classes":      self.num_classes,
                "model_id":         self.model_id,
            },
            path,
        )
        logger.info(f"Saved head checkpoint: {path}")

    def load_head(self, path: str | Path) -> None:
        """Load head weights from a checkpoint."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        self.head.load_state_dict(ckpt["head_state_dict"])
        logger.info(f"Loaded head checkpoint: {path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: Dict[str, Any],
        num_classes: int,
    ) -> "DialectClassifier":
        """Load a DialectClassifier from a saved head checkpoint."""
        model = cls(config=config, num_classes=num_classes)
        model.load_head(checkpoint_path)
        return model
