"""Integration tests for Flask API endpoints."""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from config import TestingConfig


@pytest.fixture(scope="module")
def client():
    app = create_app(config_class=TestingConfig)
    with app.test_client() as client:
        with app.app_context():
            from services.sentiment_service import get_sentiment_service
            svc = get_sentiment_service()
            svc._model_loaded = False  # VADER-only for tests
        yield client


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        res = client.post("/api/predict", json={"text": "This is a great product!"})
        assert res.status_code == 200

    def test_predict_response_structure(self, client):
        res = client.post("/api/predict", json={"text": "Excellent service!"})
        data = res.get_json()
        assert "predicted_sentiment" in data
        assert "confidence_score" in data
        assert "probabilities" in data
        assert data["predicted_sentiment"] in ["Positive", "Negative", "Neutral"]

    def test_predict_no_text(self, client):
        res = client.post("/api/predict", json={})
        assert res.status_code == 400

    def test_predict_text_too_long(self, client):
        res = client.post("/api/predict", json={"text": "a" * 10001})
        assert res.status_code == 400

    def test_predict_empty_text(self, client):
        res = client.post("/api/predict", json={"text": "   "})
        assert res.status_code == 400


class TestHistoryEndpoint:
    def test_history_returns_200(self, client):
        res = client.get("/api/history")
        assert res.status_code == 200

    def test_history_structure(self, client):
        res = client.get("/api/history")
        data = res.get_json()
        assert "records" in data
        assert "total" in data
        assert "pages" in data

    def test_history_pagination(self, client):
        res = client.get("/api/history?page=1&per_page=5")
        data = res.get_json()
        assert len(data["records"]) <= 5


class TestAnalyticsEndpoint:
    def test_analytics_returns_200(self, client):
        res = client.get("/api/analytics")
        assert res.status_code == 200

    def test_analytics_structure(self, client):
        res = client.get("/api/analytics")
        data = res.get_json()
        assert "dashboard" in data
        assert "distribution" in data
        assert "trend" in data
        assert "top_keywords" in data

    def test_trend_endpoint(self, client):
        res = client.get("/api/analytics/trend?days=7")
        data = res.get_json()
        assert "dates" in data
        assert "positive" in data
        assert "negative" in data


class TestUIRoutes:
    def test_home_page(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_analyze_page(self, client):
        res = client.get("/analyze")
        assert res.status_code == 200

    def test_bulk_page(self, client):
        res = client.get("/bulk")
        assert res.status_code == 200

    def test_voice_page(self, client):
        res = client.get("/voice")
        assert res.status_code == 200

    def test_history_page(self, client):
        res = client.get("/history")
        assert res.status_code == 200

    def test_analytics_page(self, client):
        res = client.get("/analytics")
        assert res.status_code == 200

    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.get_json()
        assert "status" in data
        assert data["status"] == "ok"
