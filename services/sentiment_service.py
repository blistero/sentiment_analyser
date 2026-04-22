"""
Hybrid Sentiment Analysis Pipeline
────────────────────────────────────
1. Text cleaning
2. BERT classifier  (primary, when trained model is available)
3. VADER            (corrective layer / fallback)
4. Confidence-weighted blending
5. Sarcasm detection with VADER-compound guard

SARCASM FLIP RULES (corrected)
────────────────────────────────
Old (buggy): flip Positive→Negative whenever sarcasm_score > 0.4 AND conf < 0.85
  Problem  : "absolutely fantastic" scored 0.6 → spurious flip.

New (correct):
  Flip only when ALL three conditions are true:
    (a) sarcasm_conf > 0.65              — high sarcasm signal required
    (b) vader_compound < +0.10           — VADER itself doesn't see positive text
    (c) final_conf < 0.90                — model isn't already very certain
  This prevents genuine positive text from ever being flipped.
"""

import time
import hashlib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, Any, List, Optional, Tuple

from models.bert_model import BertSentimentTrainer
from models.sarcasm_model import SarcasmDetector
from utils.text_cleaner import clean_text
from utils.logger import logger
from config import Config

_PREDICTION_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX = 512


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class SentimentService:

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.vader = SentimentIntensityAnalyzer()
        self.bert_trainer = BertSentimentTrainer(cfg=self.cfg)
        self.sarcasm_detector = SarcasmDetector(model_path=self.cfg.SARCASM_MODEL_PATH)
        self._model_loaded = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_models(self) -> bool:
        try:
            self.bert_trainer.load(self.cfg.MODEL_PATH)
            self._model_loaded = True
            logger.info("BERT model loaded successfully.")
        except FileNotFoundError:
            logger.warning("BERT checkpoint not found — running in VADER-only mode.")
            self._model_loaded = False

        self.sarcasm_detector.load()
        return self._model_loaded

    # ── Single prediction ─────────────────────────────────────────────────────

    def predict(self, text: str, source: str = "api") -> Dict[str, Any]:
        # Cache lookup (keyed on raw text; source is excluded so cached result is reused)
        key = _cache_key(text)
        if key in _PREDICTION_CACHE:
            cached = dict(_PREDICTION_CACHE[key])
            cached["source"] = source
            cached["from_cache"] = True
            return cached

        start = time.time()
        cleaned = clean_text(text)

        # ① VADER scores (always computed — used as corrective layer)
        vader_scores  = self.vader.polarity_scores(cleaned)
        vader_compound = vader_scores["compound"]
        vader_label   = self._vader_label(vader_compound)

        # ② BERT scores (primary when model is trained)
        if self._model_loaded:
            preds, confs, probs_arr = self.bert_trainer.predict_batch([cleaned])
            bert_label_idx = preds[0]
            bert_conf      = confs[0]
            bert_probs     = probs_arr[0].tolist()
        else:
            bert_label_idx = self._vader_label_idx(vader_compound)
            bert_conf      = self._vader_confidence(vader_compound)
            bert_probs     = self._vader_to_probs(vader_compound)

        # ③ Hybrid blend
        final_label, final_conf = self._hybrid_blend(
            bert_label_idx, bert_conf, bert_probs, vader_compound, vader_label
        )

        # ④ Sarcasm detection with corrected flip logic
        is_sarcastic, sarcasm_conf = self.sarcasm_detector.detect(cleaned)
        final_label, final_conf = self._apply_sarcasm_correction(
            final_label, final_conf, is_sarcastic, sarcasm_conf, vader_compound
        )

        elapsed_ms = (time.time() - start) * 1000

        result = {
            "original_text":    text,
            "cleaned_text":     cleaned,
            "predicted_sentiment": final_label,
            "bert_sentiment":   self.cfg.ID2LABEL.get(bert_label_idx, "Unknown"),
            "vader_sentiment":  vader_label,
            "sarcasm_detected": is_sarcastic,
            "sarcasm_confidence": round(sarcasm_conf, 4),
            "confidence_score": round(final_conf, 4),
            "probabilities": {
                "negative": round(bert_probs[0], 4),
                "neutral":  round(bert_probs[1], 4),
                "positive": round(bert_probs[2], 4),
            },
            "vader_scores": {
                "compound": round(vader_compound, 4),
                "positive": round(vader_scores["pos"], 4),
                "negative": round(vader_scores["neg"], 4),
                "neutral":  round(vader_scores["neu"], 4),
            },
            "source": source,
            "processing_time_ms": round(elapsed_ms, 2),
            "from_cache": False,
        }

        # Store in cache (evict oldest entry if full)
        if len(_PREDICTION_CACHE) >= _CACHE_MAX:
            _PREDICTION_CACHE.pop(next(iter(_PREDICTION_CACHE)))
        _PREDICTION_CACHE[key] = result

        return result

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch_texts(self, texts: List[str], source: str = "batch") -> List[Dict[str, Any]]:
        cleaned_texts = [clean_text(t) for t in texts]

        if self._model_loaded:
            preds, confs, probs_arr = self.bert_trainer.predict_batch(cleaned_texts)
        else:
            preds, confs, probs_list = [], [], []
            for ct in cleaned_texts:
                vc = self.vader.polarity_scores(ct)["compound"]
                preds.append(self._vader_label_idx(vc))
                confs.append(self._vader_confidence(vc))
                probs_list.append(self._vader_to_probs(vc))
            probs_arr = np.array(probs_list)

        sarcasm_results = self.sarcasm_detector.detect_batch(cleaned_texts)
        results = []

        for i, (orig, cleaned) in enumerate(zip(texts, cleaned_texts)):
            vader_scores   = self.vader.polarity_scores(cleaned)
            vader_compound = vader_scores["compound"]
            vader_label    = self._vader_label(vader_compound)

            bert_label_idx = int(preds[i])
            bert_conf      = float(confs[i])
            bert_probs     = (
                probs_arr[i].tolist()
                if hasattr(probs_arr[i], "tolist")
                else list(probs_arr[i])
            )

            final_label, final_conf = self._hybrid_blend(
                bert_label_idx, bert_conf, bert_probs, vader_compound, vader_label
            )

            is_sarcastic, sarcasm_conf = sarcasm_results[i]
            final_label, final_conf = self._apply_sarcasm_correction(
                final_label, final_conf, is_sarcastic, sarcasm_conf, vader_compound
            )

            results.append({
                "original_text":    orig,
                "cleaned_text":     cleaned,
                "predicted_sentiment": final_label,
                "bert_sentiment":   self.cfg.ID2LABEL.get(bert_label_idx, "Unknown"),
                "vader_sentiment":  vader_label,
                "sarcasm_detected": is_sarcastic,
                "sarcasm_confidence": round(sarcasm_conf, 4),
                "confidence_score": round(final_conf, 4),
                "probabilities": {
                    "negative": round(bert_probs[0], 4),
                    "neutral":  round(bert_probs[1], 4),
                    "positive": round(bert_probs[2], 4),
                },
                "vader_compound": round(vader_compound, 4),
                "source": source,
            })

        return results

    # ── Sarcasm correction (fixed) ────────────────────────────────────────────

    def _apply_sarcasm_correction(
        self,
        label: str,
        conf: float,
        is_sarcastic: bool,
        sarcasm_conf: float,
        vader_compound: float,
    ) -> Tuple[str, float]:
        """
        Corrected sarcasm flip logic.

        Conditions required to flip Positive → Negative:
          (a) sarcasm_conf > 0.65  — strong sarcasm signal
          (b) vader_compound < 0.10 — VADER doesn't see genuinely positive text
          (c) conf < 0.90          — model isn't extremely confident

        Guard (b) is the critical fix: "absolutely fantastic" has a VADER
        compound ≈ +0.85, so condition (b) fails and the flip never fires.
        """
        if not is_sarcastic:
            return label, conf

        sarcasm_strong   = sarcasm_conf > 0.65
        vader_not_positive = vader_compound < 0.10
        model_uncertain  = conf < 0.90

        if sarcasm_strong and vader_not_positive and model_uncertain:
            if label == "Positive":
                logger.debug(
                    f"Sarcasm correction: Positive→Negative "
                    f"(sarcasm_conf={sarcasm_conf:.3f}, vader={vader_compound:.3f})"
                )
                return "Negative", max(0.50, conf - 0.15)
            if label == "Negative":
                # Sarcasm on negative text → slight confidence drop only
                return label, max(0.50, conf - 0.05)

        return label, conf

    # ── Hybrid blending ───────────────────────────────────────────────────────

    def _hybrid_blend(
        self,
        bert_label_idx: int,
        bert_conf: float,
        bert_probs: List[float],
        vader_compound: float,
        vader_label: str,
    ) -> Tuple[str, float]:
        """
        Blend BERT and VADER predictions.

        Decision tree:
          • bert_conf ≥ 0.80  → trust BERT (high certainty)
          • |vader_compound| ≥ 0.60 AND bert_conf ≤ 0.55
                              → VADER override (strong lexical signal, weak model)
          • otherwise         → weighted soft blend of probability distributions
        """
        bert_label = self.cfg.ID2LABEL.get(bert_label_idx, "Neutral")
        vader_idx  = {"Negative": 0, "Neutral": 1, "Positive": 2}.get(vader_label, 1)
        vader_conf = min(abs(vader_compound) * 1.4, 0.90)

        # High BERT confidence — trust it outright
        if bert_conf >= 0.80:
            return bert_label, round(bert_conf, 4)

        # Strong VADER signal + uncertain BERT → let VADER decide
        if abs(vader_compound) >= 0.60 and bert_conf <= 0.55:
            blended_conf = round((vader_conf * 0.6 + bert_conf * 0.4), 4)
            return vader_label, blended_conf

        # Soft blend: weighted average of probability distributions
        vader_probs = [0.15, 0.15, 0.15]
        vader_probs[vader_idx] = 0.70

        w_bert  = bert_conf
        w_vader = 1.0 - bert_conf
        blended = [
            w_bert * bp + w_vader * vp
            for bp, vp in zip(bert_probs, vader_probs)
        ]

        # Normalize so probabilities sum to 1
        total = sum(blended)
        if total > 0:
            blended = [b / total for b in blended]

        final_idx  = int(np.argmax(blended))
        final_conf = float(max(blended))
        return self.cfg.ID2LABEL[final_idx], round(final_conf, 4)

    # ── VADER helpers ─────────────────────────────────────────────────────────

    def _vader_label(self, compound: float) -> str:
        if compound >= self.cfg.VADER_COMPOUND_POSITIVE:
            return "Positive"
        if compound <= self.cfg.VADER_COMPOUND_NEGATIVE:
            return "Negative"
        return "Neutral"

    def _vader_label_idx(self, compound: float) -> int:
        return self.cfg.LABEL2ID[self._vader_label(compound)]

    def _vader_confidence(self, compound: float) -> float:
        """
        Map VADER compound [-1, +1] to a confidence score.
        Near-zero compound → low confidence (~0.40).
        Extreme compound   → high confidence (capped at 0.92).
        """
        abs_c = abs(compound)
        if abs_c < 0.05:
            return 0.40
        if abs_c < 0.30:
            return 0.40 + abs_c * 0.80          # 0.40 → 0.64
        if abs_c < 0.60:
            return 0.64 + (abs_c - 0.30) * 0.60 # 0.64 → 0.82
        return min(0.40 + abs_c * 0.87, 0.92)   # 0.82 → 0.92

    def _vader_to_probs(self, compound: float) -> List[float]:
        """Convert VADER compound score to a [neg, neu, pos] probability vector."""
        if compound >= 0.05:
            t   = min(compound, 1.0)
            neg = max(0.05, 0.50 - t * 0.45)
            neu = max(0.05, 0.30 - t * 0.25)
            pos = min(0.90, 0.20 + t * 0.70)
        elif compound <= -0.05:
            t   = min(abs(compound), 1.0)
            neg = min(0.90, 0.20 + t * 0.70)
            neu = max(0.05, 0.30 - t * 0.25)
            pos = max(0.05, 0.50 - t * 0.45)
        else:
            neg, neu, pos = 0.20, 0.60, 0.20

        # Normalize
        total = neg + neu + pos
        return [neg / total, neu / total, pos / total]


# ── Singleton ─────────────────────────────────────────────────────────────────

_service: Optional[SentimentService] = None


def get_sentiment_service(cfg: Config = None) -> SentimentService:
    global _service
    if _service is None:
        _service = SentimentService(cfg=cfg)
    return _service
