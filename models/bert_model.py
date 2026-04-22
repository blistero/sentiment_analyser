"""
BERT-based Sentiment Classifier with full per-epoch metric tracking.

Architecture:
    bert-base-uncased  →  [CLS] pooler output  →  Dropout  →  Linear(768→256)
                       →  GELU  →  Dropout  →  Linear(256→3)  →  Softmax

Classes: 0=Negative, 1=Neutral, 2=Positive
"""
import os
import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from utils.logger import logger
from config import Config


# ─── Dataset ────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """PyTorch Dataset for tokenized text + integer label pairs."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long),
            ).squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Model ──────────────────────────────────────────────────────────────────

class BertSentimentClassifier(nn.Module):
    """
    BERT + 2-layer MLP classification head.

    Why BERT?
      • Pre-trained on 3.3B words — rich contextual representations
      • Bidirectional context captures nuance that TF-IDF misses
      • Fine-tuning requires ~3 epochs vs training from scratch
    """

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size   # 768 for base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_labels),
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output          # [CLS] token, shape (B, 768)
        return self.classifier(pooled)          # shape (B, num_labels)


# ─── Trainer ────────────────────────────────────────────────────────────────

class EpochMetrics:
    """Container for all per-epoch training/validation metrics."""

    def __init__(self):
        self.epoch: int = 0
        self.train_loss: float = 0.0
        self.val_loss: float = 0.0
        self.val_accuracy: float = 0.0
        self.val_precision: float = 0.0
        self.val_recall: float = 0.0
        self.val_f1: float = 0.0
        # Per-class
        self.per_class_precision: List[float] = []
        self.per_class_recall: List[float] = []
        self.per_class_f1: List[float] = []
        # Raw predictions for confusion matrix
        self.val_preds: List[int] = []
        self.val_labels: List[int] = []

    def to_dict(self) -> Dict:
        return {
            "epoch":          self.epoch,
            "train_loss":     round(self.train_loss, 6),
            "val_loss":       round(self.val_loss, 6),
            "val_accuracy":   round(self.val_accuracy, 6),
            "val_precision":  round(self.val_precision, 6),
            "val_recall":     round(self.val_recall, 6),
            "val_f1":         round(self.val_f1, 6),
            "val_f1_neg":     round(self.per_class_f1[0], 6) if len(self.per_class_f1) > 0 else 0,
            "val_f1_neu":     round(self.per_class_f1[1], 6) if len(self.per_class_f1) > 1 else 0,
            "val_f1_pos":     round(self.per_class_f1[2], 6) if len(self.per_class_f1) > 2 else 0,
        }


class BertSentimentTrainer:
    """
    Full training pipeline for BertSentimentClassifier.

    Handles:
      - DataLoader construction
      - Weighted cross-entropy (class imbalance)
      - Linear warmup + decay scheduler
      - Per-epoch metric logging (loss, acc, P, R, F1)
      - CSV + JSON export of training history
      - Checkpoint saving (best val loss)
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.BERT_MODEL_NAME)
        self.model = BertSentimentClassifier(
            model_name=self.cfg.BERT_MODEL_NAME,
            num_labels=self.cfg.NUM_LABELS,
        ).to(self.device)

    # ── DataLoaders ──────────────────────────────────────────────────────────

    def build_dataloaders(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        pin = self.device.type == "cuda"
        train_ds = SentimentDataset(df_train["text"].tolist(), df_train["label"].tolist(), self.tokenizer, self.cfg.MAX_LENGTH)
        val_ds   = SentimentDataset(df_val["text"].tolist(),   df_val["label"].tolist(),   self.tokenizer, self.cfg.MAX_LENGTH)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.BATCH_SIZE,      shuffle=True,  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.BATCH_SIZE * 2,  shuffle=False, num_workers=0)
        return train_loader, val_loader

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        class_weights: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        report_dir: Optional[str] = None,
    ) -> List[EpochMetrics]:
        save_path  = save_path  or self.cfg.MODEL_PATH
        report_dir = report_dir or "reports"
        os.makedirs(save_path,  exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        train_loader, val_loader = self.build_dataloaders(df_train, df_val)

        weight_tensor = None
        if class_weights:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        optimizer    = AdamW(self.model.parameters(), lr=self.cfg.LEARNING_RATE, weight_decay=self.cfg.WEIGHT_DECAY)
        total_steps  = len(train_loader) * self.cfg.NUM_EPOCHS
        scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.cfg.WARMUP_STEPS, num_training_steps=total_steps)

        best_val_loss = float("inf")
        history: List[EpochMetrics] = []

        # CSV file for live epoch logging
        csv_path = os.path.join(report_dir, "training_log.csv")
        csv_fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1",
                      "val_f1_neg", "val_f1_neu", "val_f1_pos"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

        logger.info("=" * 70)
        logger.info(f"Starting training | Epochs={self.cfg.NUM_EPOCHS} | Batch={self.cfg.BATCH_SIZE} | LR={self.cfg.LEARNING_RATE}")
        logger.info(f"Train batches={len(train_loader)} | Val batches={len(val_loader)}")
        logger.info("=" * 70)

        for epoch in range(1, self.cfg.NUM_EPOCHS + 1):
            # ── Train phase ──────────────────────────────────────────────────
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.NUM_EPOCHS} [Train]", unit="batch")

            for batch in pbar:
                ids   = batch["input_ids"].to(self.device)
                mask  = batch["attention_mask"].to(self.device)
                ttype = batch["token_type_ids"].to(self.device)
                labs  = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(ids, mask, ttype)
                loss   = criterion(logits, labs)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            avg_train_loss = running_loss / len(train_loader)

            # ── Validation phase ─────────────────────────────────────────────
            em = self._evaluate_full(val_loader, criterion, epoch, avg_train_loss)
            history.append(em)

            # Pretty log
            logger.info(
                f"Epoch {epoch:2d}/{self.cfg.NUM_EPOCHS} | "
                f"train_loss={em.train_loss:.4f} | "
                f"val_loss={em.val_loss:.4f} | "
                f"acc={em.val_accuracy:.4f} | "
                f"P={em.val_precision:.4f} | "
                f"R={em.val_recall:.4f} | "
                f"F1={em.val_f1:.4f}"
            )
            per_cls = list(zip(self.cfg.SENTIMENT_LABELS, em.per_class_f1))
            for lbl, f1 in per_cls:
                logger.info(f"          {lbl:10s}: F1={f1:.4f}")

            # Append to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow({k: em.to_dict()[k] for k in csv_fields})

            # Checkpoint on best val loss
            if em.val_loss < best_val_loss:
                best_val_loss = em.val_loss
                self.save(save_path)
                logger.info(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        # Save full history JSON
        history_data = [em.to_dict() for em in history]
        json_path = os.path.join(report_dir, "training_history.json")
        with open(json_path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"\nTraining complete. Log → {csv_path} | History → {json_path}")
        return history

    def _evaluate_full(self, loader: DataLoader, criterion: nn.Module, epoch: int, train_loss: float) -> EpochMetrics:
        """Run validation and compute all metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds:  List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in loader:
                ids   = batch["input_ids"].to(self.device)
                mask  = batch["attention_mask"].to(self.device)
                ttype = batch["token_type_ids"].to(self.device)
                labs  = batch["label"].to(self.device)

                logits = self.model(ids, mask, ttype)
                loss   = criterion(logits, labs)
                total_loss += loss.item()

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labs.cpu().numpy().tolist())

        val_loss = total_loss / len(loader)

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro",
            labels=[0, 1, 2], zero_division=0
        )
        cls_prec, cls_rec, cls_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None,
            labels=[0, 1, 2], zero_division=0
        )

        em = EpochMetrics()
        em.epoch          = epoch
        em.train_loss     = train_loss
        em.val_loss       = val_loss
        em.val_accuracy   = float(acc)
        em.val_precision  = float(prec)
        em.val_recall     = float(rec)
        em.val_f1         = float(f1)
        em.per_class_precision = cls_prec.tolist()
        em.per_class_recall    = cls_rec.tolist()
        em.per_class_f1        = cls_f1.tolist()
        em.val_preds           = all_preds
        em.val_labels          = all_labels
        return em

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_batch(self, texts: List[str]) -> Tuple[List[int], List[float], np.ndarray]:
        """Return (predicted_label_indices, confidence_scores, all_probs_array)."""
        self.model.eval()
        all_preds, all_confs, all_probs = [], [], []
        chunk = self.cfg.BATCH_SIZE * 2

        for i in range(0, len(texts), chunk):
            batch_texts = texts[i : i + chunk]
            enc = self.tokenizer(
                batch_texts, max_length=self.cfg.MAX_LENGTH,
                padding=True, truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                    enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).to(self.device),
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(probs.argmax(axis=1).tolist())
            all_confs.extend(probs.max(axis=1).tolist())
            all_probs.append(probs)

        return all_preds, all_confs, np.vstack(all_probs) if all_probs else np.array([])

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Saves to `path/`:
          - config.json, vocab.txt, tokenizer files  (HuggingFace format)
          - classifier_head.pt                        (custom head state dict)
          - model_info.json                           (metadata)
        """
        os.makedirs(path, exist_ok=True)
        self.model.bert.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.model.state_dict(), os.path.join(path, "classifier_head.pt"))

        meta = {
            "bert_model_name": self.cfg.BERT_MODEL_NAME,
            "num_labels":      self.cfg.NUM_LABELS,
            "label_map":       self.cfg.ID2LABEL,
            "max_length":      self.cfg.MAX_LENGTH,
            "device":          str(self.device),
        }
        with open(os.path.join(path, "model_info.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Model saved → {path}/")

    def load(self, path: str) -> None:
        """
        Loads from `path/` (must contain classifier_head.pt + tokenizer files).
        Falls back gracefully if head file is missing.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model     = BertSentimentClassifier(model_name=path, num_labels=self.cfg.NUM_LABELS).to(self.device)

        head_path = os.path.join(path, "classifier_head.pt")
        if os.path.exists(head_path):
            state = torch.load(head_path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info(f"Classifier head loaded from {head_path}")

        self.model.eval()
        logger.info(f"Model loaded from {path}")

    def get_checkpoint_info(self, path: str) -> Dict:
        info_path = os.path.join(path, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                return json.load(f)
        return {}
