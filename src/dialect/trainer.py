"""
src/dialect/trainer.py
=======================
Training loop for the Tamil dialect classifier.

Design
------
- Loads audio from the manifest CSV.
- Applies district → dialect label mapping via DialectLabelRegistry.
- Uses wav2vec2 feature extraction (frozen) + MLP head training.
- Supports early stopping, learning rate scheduling, and checkpointing.
- All hyperparameters are read from ``configs/dialect.yaml``.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor

from .classifier import DialectClassifier
from .labels import DialectLabelRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DialectAudioDataset(Dataset):
    """
    PyTorch Dataset that loads audio files from a manifest CSV.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Must contain ``audio_path`` (or ``file``), ``dialect_label`` (int).
    label_registry : DialectLabelRegistry
    target_sr : int
    max_length : int
        Maximum waveform length in samples (clips longer than this are truncated).
    audio_col : str
        Column name containing paths to audio files.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        label_registry: DialectLabelRegistry,
        target_sr: int = 16000,
        max_length: int = 80000,
        audio_col: str = "file",
    ) -> None:
        self.df           = manifest_df.reset_index(drop=True)
        self.registry     = label_registry
        self.target_sr    = target_sr
        self.max_length   = max_length
        self.audio_col    = audio_col

        # Resolve audio column
        if self.audio_col not in self.df.columns:
            for candidate in ("file", "audio_path", "audio_hf_path"):
                if candidate in self.df.columns:
                    self.audio_col = candidate
                    break

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row        = self.df.iloc[idx]
        audio_path = str(row[self.audio_col])
        dialect    = str(row.get("dialect_label", row.get("district", "")))

        # Resolve dialect label
        if dialect in self.registry.label2idx:
            label_idx = self.registry.label_to_idx(dialect)
        else:
            # district → dialect via mapping
            dialect   = self.registry.district_to_dialect(dialect)
            label_idx = self.registry.label_to_idx(dialect)

        # Load audio
        try:
            wave, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)  # mono
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}. Using silence.")
            wave = np.zeros(self.target_sr, dtype=np.float32)

        # Truncate / pad
        wave = wave[:self.max_length]
        waveform = torch.tensor(wave, dtype=torch.float32)

        return waveform, label_idx, dialect

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Pad waveforms to the same length within a batch."""
        waveforms, labels, dialects = zip(*batch)
        max_len = max(w.shape[0] for w in waveforms)

        padded    = torch.zeros(len(waveforms), max_len)
        attn_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)
        for i, w in enumerate(waveforms):
            l = w.shape[0]
            padded[i, :l]    = w
            attn_mask[i, :l] = 1

        label_tensor = torch.tensor(labels, dtype=torch.long)
        return padded, attn_mask, label_tensor, list(dialects)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DialectTrainer:
    """
    Trains the dialect classifier head.

    Parameters
    ----------
    config : dict
        Parsed ``configs/dialect.yaml``.
    label_registry : DialectLabelRegistry
    """

    def __init__(
        self,
        config: Dict[str, Any],
        label_registry: DialectLabelRegistry,
    ) -> None:
        self.config   = config
        self.registry = label_registry
        train_cfg     = config.get("training", {})

        self.batch_size    = train_cfg.get("batch_size", 8)
        self.max_epochs    = train_cfg.get("max_epochs", 30)
        self.lr            = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay  = train_cfg.get("weight_decay", 1e-4)
        self.patience      = train_cfg.get("patience", 5)
        self.max_audio_len = train_cfg.get("max_audio_length", 80000)
        self.num_workers   = train_cfg.get("num_workers", 0)
        self.seed          = train_cfg.get("seed", 42)
        self.ckpt_dir      = Path(config.get("model", {}).get("checkpoint_dir", "models/dialect"))

        device_str = train_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self._set_seed()
        logger.info(f"DialectTrainer: device={self.device}, labels={self.registry.names}")

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _build_loader(
        self,
        manifest_path: str | Path,
        shuffle: bool = True,
        audio_col: str = "file",
    ) -> DataLoader:
        df = pd.read_csv(manifest_path)
        dataset = DialectAudioDataset(
            manifest_df=df,
            label_registry=self.registry,
            target_sr=self.config.get("model", {}).get("target_sample_rate", 16000),
            max_length=self.max_audio_len,
            audio_col=audio_col,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DialectAudioDataset.collate_fn,
        )

    def train(
        self,
        train_manifest: str | Path,
        val_manifest: str | Path,
        audio_col: str = "file",
    ) -> Dict[str, Any]:
        """
        Train the classifier head.

        Returns a dict containing training history and best metrics.
        """
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_loader = self._build_loader(train_manifest, shuffle=True,  audio_col=audio_col)
        val_loader   = self._build_loader(val_manifest,   shuffle=False, audio_col=audio_col)

        model = DialectClassifier(
            config=self.config,
            num_classes=self.registry.num_classes,
        ).to(self.device)

        # Only train the head
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc   = 0.0
        patience_count = 0
        history        = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            # --- Training ---
            model.train()
            train_loss = 0.0
            for waveform, attn_mask, labels, _ in train_loader:
                waveform  = waveform.to(self.device)
                attn_mask = attn_mask.to(self.device)
                labels    = labels.to(self.device)

                optimizer.zero_grad()
                logits = model(waveform, attn_mask)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)
            scheduler.step()

            # --- Validation ---
            val_loss, val_acc = self._evaluate_epoch(model, val_loader, criterion)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | {elapsed:.1f}s"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # --- Save best ---
            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                patience_count = 0
                best_path = self.ckpt_dir / "best_model.pt"
                model.save(best_path)
                logger.info(f"  => New best val_acc={best_val_acc:.4f}. Saved to {best_path}")
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        return {"history": history, "best_val_acc": best_val_acc}

    @torch.no_grad()
    def _evaluate_epoch(
        self,
        model: DialectClassifier,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for waveform, attn_mask, labels, _ in loader:
            waveform  = waveform.to(self.device)
            attn_mask = attn_mask.to(self.device)
            labels    = labels.to(self.device)

            logits = model(waveform, attn_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds  = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total  += labels.size(0)

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
