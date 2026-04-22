# Hybrid BERT-Based Sentiment Analysis System with Sarcasm Detection, Voice Input, and Analytics Dashboard

A full-stack sentiment analysis system: BERT fine-tuned on 60,000 Amazon Reviews,
hybrid BERT+VADER pipeline, sarcasm detection, browser voice input, batch CSV processing,
and a dark-themed analytics dashboard.

**Test Accuracy: 94.4% | Macro F1: 94.5%**

---

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Train (60k samples, 3 epochs -- ~2 hrs CPU or ~20 min GPU)
python train.py

# 3. Run web app
python app.py
# Open http://localhost:5000
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `python train.py` | Train BERT on Amazon Reviews (default: 60k, 3 epochs) |
| `python train.py --max-samples 2000 --epochs 2` | Quick test run |
| `python evaluate.py` | Evaluate checkpoint on test split |
| `python predict.py "text here"` | Single CLI prediction |
| `python predict.py -i` | Interactive CLI mode |
| `pytest tests/ -v` | Run all tests |

---

## Architecture

```
Input (text / CSV / voice)
    |
    v
SentimentService
    |-- [1] MD5 cache lookup (512-entry FIFO)
    |-- [2] clean_text()
    |-- [3] BERT predict_batch() --> label + confidence + probs
    |-- [4] VADER polarity_scores() --> compound
    |-- [5] _hybrid_blend() --> final_label, final_conf
    |-- [6] sarcasm_detector.detect() --> is_sarcastic, sarcasm_conf
    |-- [7] _apply_sarcasm_correction() --> adjusted label/conf
    |
    v
SQLite (PredictionRecord) --> Analytics / History / Export
```

---

## Key Features

| Feature | Details |
|---------|---------|
| 3-class prediction | Positive / Negative / Neutral |
| Hybrid pipeline | BERT confidence-weighted blend with VADER |
| Sarcasm detection | DistilBERT + rule-based fallback with VADER guard |
| Sarcasm UI | Confidence meter with gradient fill |
| Voice input | Browser Web Speech API (no PyAudio needed) |
| Batch processing | Async CSV/Excel with status polling |
| Analytics | Doughnut, trend, confidence histogram, keyword cloud |
| Model summary card | Dataset size, epochs, accuracy, per-class F1 bars |
| Response caching | MD5-keyed in-memory cache |
| Windows-safe logging | UTF-8 handlers, ASCII-only log symbols |
| Dark theme UI | Premium dark theme, mobile responsive |

---

## API Reference

```bash
# Predict single text
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely fantastic!"}'

# Get analytics
curl http://localhost:5000/api/analytics

# Model metadata
curl http://localhost:5000/api/model_info

# Health check
curl http://localhost:5000/health
```

**Response example:**
```json
{
  "predicted_sentiment": "Positive",
  "confidence_score": 0.9823,
  "bert_sentiment": "Positive",
  "vader_sentiment": "Positive",
  "sarcasm_detected": false,
  "sarcasm_confidence": 0.12,
  "probabilities": { "positive": 0.9823, "neutral": 0.0134, "negative": 0.0043 },
  "processing_time_ms": 84.2
}
```

---

## Test Results

| Metric | Score |
|--------|-------|
| Accuracy | 94.36% |
| Macro F1 | 94.48% |
| Negative F1 | 92.26% |
| Neutral F1 | 99.35% |
| Positive F1 | 91.82% |

Trained on: 48,000 samples | Tested on: 6,000 samples | 3 epochs

---

## Files After Training

```
reports/
  training_log.csv          per-epoch metrics (loss, acc, F1)
  training_history.json     full JSON history
  test_metrics.json         final test evaluation
  loss_curve.png            train vs val loss chart
  accuracy_curve.png        accuracy + F1 per epoch
  confusion_matrix.png      normalized confusion matrix
  training_dashboard.png    all-in-one 6-panel figure (use for viva)

data/checkpoints/bert_sentiment/
  model.safetensors         BERT weights (HuggingFace format)
  classifier_head.pt        classification head weights
  model_info.json           metadata
  vocab.txt + tokenizer     tokenizer files
```

---

## Documentation

| File | Contents |
|------|----------|
| `FINAL_REPORT.md` | Full academic report (methodology, results, references) |
| `DATASET_INFO.md` | Dataset transparency and preprocessing details |
| `MODEL_TRAINING.md` | Architecture and training procedure documentation |
| `RESULTS.md` | Results, metrics, and visualization guide |
| `DEPLOYMENT_GUIDE.md` | Local, Docker, and production deployment |

---

## Troubleshooting

**"BERT checkpoint not found" warning**
Run `python train.py` first. App runs in VADER-only mode without it.

**Windows PowerShell Unicode errors**
```powershell
$env:PYTHONIOENCODING = "utf-8"
python train.py
```

**Microphone not working**
Use the browser microphone tab (Chrome or Edge). The Web Speech API transcribes speech directly in the browser — no server audio hardware needed.
