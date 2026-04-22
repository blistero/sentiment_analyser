"""Unit tests for sentiment service and text cleaner."""
import pytest
from unittest.mock import MagicMock, patch
from utils.text_cleaner import clean_text, normalize_rating_to_sentiment, is_short_text
from services.sentiment_service import SentimentService
from config import Config


class TestTextCleaner:
    def test_clean_url(self):
        text = "Visit http://example.com for more info"
        result = clean_text(text)
        assert "http" not in result

    def test_clean_html(self):
        text = "<p>Great product!</p>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "Great product!" in result

    def test_clean_repeated_chars(self):
        result = clean_text("sooooooo good")
        assert "soo" in result
        assert "sooooooo" not in result

    def test_empty_string(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_unicode_normalize(self):
        result = clean_text("café résumé")
        assert isinstance(result, str)

    def test_rating_to_sentiment(self):
        assert normalize_rating_to_sentiment(1) == 0  # Negative
        assert normalize_rating_to_sentiment(2) == 0
        assert normalize_rating_to_sentiment(3) == 1  # Neutral
        assert normalize_rating_to_sentiment(4) == 2  # Positive
        assert normalize_rating_to_sentiment(5) == 2

    def test_is_short_text(self):
        assert is_short_text("hi there") is True
        assert is_short_text("This is a longer sentence with more words than the threshold") is False


class TestSentimentService:
    @pytest.fixture
    def service(self):
        cfg = Config()
        svc = SentimentService(cfg=cfg)
        # Don't load actual models - use VADER-only mode
        svc._model_loaded = False
        return svc

    def test_predict_positive(self, service):
        result = service.predict("This product is absolutely amazing and fantastic!")
        assert result["predicted_sentiment"] in ["Positive", "Neutral", "Negative"]
        assert 0 <= result["confidence_score"] <= 1
        assert "probabilities" in result
        assert "vader_scores" in result

    def test_predict_negative(self, service):
        result = service.predict("This is terrible, worst product ever. Complete garbage!")
        assert result["predicted_sentiment"] in ["Positive", "Neutral", "Negative"]
        assert result["confidence_score"] > 0

    def test_predict_returns_all_fields(self, service):
        result = service.predict("Okay product.")
        required_keys = [
            "original_text", "cleaned_text", "predicted_sentiment",
            "bert_sentiment", "vader_sentiment", "sarcasm_detected",
            "confidence_score", "probabilities", "vader_scores", "source"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_predict_probabilities_sum_to_one(self, service):
        result = service.predict("Decent product, works fine.")
        probs = result["probabilities"]
        total = probs["positive"] + probs["neutral"] + probs["negative"]
        assert abs(total - 1.0) < 0.05

    def test_predict_batch(self, service):
        texts = ["Great!", "Terrible.", "It's okay."]
        results = service.predict_batch_texts(texts)
        assert len(results) == 3
        for r in results:
            assert r["predicted_sentiment"] in ["Positive", "Negative", "Neutral"]

    def test_empty_text_handling(self, service):
        result = service.predict("")
        assert isinstance(result, dict)

    def test_long_text(self, service):
        long_text = "This product is great. " * 100
        result = service.predict(long_text)
        assert result["predicted_sentiment"] in ["Positive", "Neutral", "Negative"]


class TestMetrics:
    def test_compute_metrics(self):
        from utils.metrics import compute_metrics
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 0, 2]
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "per_class" in metrics
        assert 0 <= metrics["accuracy"] <= 1
