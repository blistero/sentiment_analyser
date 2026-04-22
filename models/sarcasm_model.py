"""
Sarcasm Detection — DistilBERT classifier with rule-based fallback.

DESIGN NOTES — why the old signal list was wrong
─────────────────────────────────────────────────
Words like "absolutely", "fantastic", "wonderful", "brilliant", "genius"
are POSITIVE INTENSIFIERS in genuine reviews. They only indicate sarcasm
when surrounded by contradictory negative context ("Oh brilliant, it broke
again"). Treating them as stand-alone signals produced false positives
("This is absolutely fantastic!" → wrongly flagged as sarcastic).

The corrected rule-based detector:
  • Only multi-word phrases that are unambiguously sarcastic on their own
  • Single-word sarcasm signals require NEGATIVE context in the same text
  • VADER compound guard: if VADER says compound > +0.30, text is genuinely
    positive — never flag as sarcastic regardless of lexical signals
  • Raised detection threshold: 0.55 (was 0.40)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

from utils.logger import logger


# ── Multi-word phrases that are unambiguously sarcastic ──────────────────────
# These phrases are almost exclusively used ironically in product reviews.
SARCASM_PHRASES = [
    "yeah right",
    "oh great",
    "oh sure",
    "wow thanks",
    "just what i needed",
    "just what i wanted",
    "like that'll work",
    "like that will work",
    "because that makes sense",
    "oh how lovely",
    "big surprise",
    "what a surprise",
    "as if that works",
    "color me surprised",
    "oh what a shock",
    "totally shocked",
    "oh perfect timing",
]

# Phrases for positive word + negative event contradiction (classic sarcasm)
# These fire when combined with a positive word in the same text
SARCASM_EVENTS = [
    "stopped working",
    "stop working",
    "broke again",
    "failed again",
    "doesn't work",
    "does not work",
    "won't work",
    "wont work",
    "won't start",
    "never worked",
    "keeps crashing",
    "keeps failing",
    "stopped charging",
    "stopped responding",
    "instantly broke",
    "immediately broke",
    "died instantly",
    "died immediately",
]

# Single words that ONLY count when paired with negative context words
SARCASM_SINGLE_CONDITIONAL = [
    "obviously", "clearly", "totally", "surely",
]

# Negative context words — required for phrase bonuses and single-word signals
NEGATIVE_CONTEXT = [
    "broke", "broken", "failed", "stop", "stopped", "useless", "waste",
    "terrible", "awful", "horrible", "worst", "bad", "poor", "cheap",
    "junk", "garbage", "trash", "disappointed", "regret", "return",
    "refund", "defective", "defect", "error", "crash", "crashed",
    "died", "dead", "instantly", "immediately", "again",
]


class SarcasmDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SarcasmClassifier(nn.Module):
    """Lightweight DistilBERT-based binary sarcasm classifier."""

    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden = self.distilbert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_out)


class SarcasmDetector:
    def __init__(self, model_path: str = "data/checkpoints/sarcasm_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[SarcasmClassifier] = None
        self.tokenizer = None
        self._rule_only = True

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            logger.info("Sarcasm model not found — using rule-based detector.")
            return False
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = SarcasmClassifier(model_name=self.model_path).to(self.device)
            state = torch.load(
                os.path.join(self.model_path, "sarcasm_head.pt"),
                map_location=self.device,
            )
            self.model.load_state_dict(state)
            self.model.eval()
            self._rule_only = False
            logger.info("Sarcasm model loaded.")
            return True
        except Exception as e:
            logger.warning(f"Sarcasm model load failed ({e}) — using rule-based detector.")
            return False

    def detect(self, text: str) -> Tuple[bool, float]:
        """
        Returns (is_sarcastic: bool, confidence: float 0–1).

        When a trained DistilBERT model is available its score is blended
        70/30 with the rule-based score.  Without a trained model the
        rule-based detector runs alone.
        """
        rule_flag, rule_score = self._rule_based(text)

        if self._rule_only or self.model is None:
            return rule_flag, rule_score

        # Neural score
        self.model.eval()
        enc = self.tokenizer(
            text, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
            neural_score = float(torch.softmax(logits, dim=-1).cpu()[0, 1])

        blended = 0.70 * neural_score + 0.30 * rule_score
        return blended > 0.55, round(blended, 4)

    def detect_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        return [self.detect(t) for t in texts]

    # ── Rule-based detector ──────────────────────────────────────────────────

    def _rule_based(self, text: str) -> Tuple[bool, float]:
        """
        Rule-based sarcasm detection.

        Scoring layers (all capped at 1.0):
          (1) Unambiguous multi-word sarcasm phrase:           +0.45
              + bonus when that phrase sits next to neg ctx:   +0.20
          (2) Positive word + SARCASM_EVENT phrase:            +0.45
          (3) Positive word + negative word contradiction:     +0.30
          (4) Single-word conditional (needs neg ctx):         +0.25
          (5) Punctuation / structural patterns:               +0.15/0.10
          (6) ALL-CAPS proportion:                             +0.20
          (7) VADER compound guard: if compound >= 0.30, cap at 0.35
        Threshold = 0.55
        """
        import re as _re
        text_lower = text.lower()
        score = 0.0

        # Tokenise without punctuation so "amazing," matches "amazing"
        word_tokens = set(_re.findall(r"\b[a-z']+\b", text_lower))

        has_negative_ctx = any(w in text_lower for w in NEGATIVE_CONTEXT)

        # ① Multi-word sarcasm phrases ────────────────────────────────────────
        for phrase in SARCASM_PHRASES:
            if phrase in text_lower:
                score += 0.45
                if has_negative_ctx:
                    score += 0.20   # context bonus — phrase + negative setting
                break               # count once even if multiple phrases match

        # ② Positive word + negative event phrase ─────────────────────────────
        #    "Amazing, stopped working instantly" — event-level contradiction
        pos_words = {"love", "great", "amazing", "wonderful", "perfect",
                     "fantastic", "excellent", "brilliant", "superb", "ideal"}
        has_pos_word = bool(pos_words & word_tokens)

        for event in SARCASM_EVENTS:
            if event in text_lower and has_pos_word:
                score += 0.45
                break

        # ③ Lexical contradiction: positive word + negative word ──────────────
        neg_words = {"hate", "terrible", "awful", "horrible", "worst",
                     "disaster", "broke", "broken", "crash", "crashed",
                     "stopped", "useless", "garbage", "junk", "trash"}
        if has_pos_word and (neg_words & word_tokens):
            score += 0.30

        # ④ Single-word conditional signals ───────────────────────────────────
        if has_negative_ctx:
            for word in SARCASM_SINGLE_CONDITIONAL:
                if word in word_tokens:
                    score += 0.25

        # ⑤ Punctuation / structural patterns ─────────────────────────────────
        excl  = text.count("!")
        quest = text.count("?")
        if excl >= 3:
            score += 0.15
        if excl >= 2 and quest >= 1:
            score += 0.10

        # ⑥ ALL-CAPS proportion ───────────────────────────────────────────────
        words = text.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
        if caps_ratio > 0.35:
            score += 0.20

        score = min(score, 1.0)

        # ⑦ VADER compound guard ──────────────────────────────────────────────
        #    If VADER's lexicon sees clearly positive text, sarcasm is unlikely.
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
            compound = _vader.polarity_scores(text)["compound"]
            if compound >= 0.30:
                score = min(score, 0.35)   # cap below threshold — genuine positive
            elif compound <= -0.30:
                score = min(score + 0.10, 1.0)   # mild boost — text is already negative
        except Exception:
            pass

        return score >= 0.55, round(score, 4)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        texts: List[str],
        labels: List[int],
        save_path: str,
        epochs: int = 3,
        batch_size: int = 32,
    ):
        """Fine-tune sarcasm detector on labeled data (0=not sarcastic, 1=sarcastic)."""
        os.makedirs(save_path, exist_ok=True)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = SarcasmClassifier().to(self.device)

        dataset = SarcasmDataset(texts, labels, tokenizer)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer  = AdamW(model.parameters(), lr=3e-5)
        criterion  = nn.CrossEntropyLoss()
        total_steps = len(loader) * epochs
        scheduler  = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=total_steps
        )

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Sarcasm Epoch {epoch + 1}"):
                optimizer.zero_grad()
                logits = model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                loss = criterion(logits, batch["label"].to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            logger.info(f"Sarcasm Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")

        model.distilbert.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, "sarcasm_head.pt"))
        logger.info(f"Sarcasm model saved → {save_path}")
