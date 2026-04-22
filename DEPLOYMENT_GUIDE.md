# Deployment Guide — Sentiment Analysis System

---

## Quick Start (Local Development)

```bash
# 1. Clone / navigate to project
cd sentiment_analyser_new

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set environment variables
copy .env.example .env
# Edit .env if needed

# 5. Train the model (first run — takes ~2 hrs CPU or ~20 min GPU)
python train.py --max-samples 60000 --epochs 3

# 6. Start the web server
python app.py
# Open: http://localhost:5000
```

---

## Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Set to `production` for deployment |
| `SECRET_KEY` | auto-generated | Flask session secret — set a strong random value in production |
| `DATABASE_URL` | `sqlite:///instance/sentiment.db` | Switch to PostgreSQL in production |
| `PORT` | `5000` | Server port |
| `MODEL_PATH` | `data/checkpoints/bert_sentiment` | BERT checkpoint directory |

---

## Run Commands Reference

```bash
# Train with default settings (60k samples, 3 epochs)
python train.py

# Quick training for testing (2k samples, 2 epochs)
python train.py --max-samples 2000 --epochs 2

# Evaluate existing checkpoint on test set
python evaluate.py

# Evaluate on custom CSV
python evaluate.py --file data/my_test.csv --text-col text --label-col label

# CLI single prediction
python predict.py "This product is absolutely fantastic!"

# Interactive CLI
python predict.py -i

# Run all tests
pytest tests/ -v

# Start production server (Gunicorn, Linux)
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

---

## Docker Deployment

```bash
# Build image
docker build -t sentiment-analyser .

# Run container
docker run -p 5000:5000 -v $(pwd)/data:/app/data sentiment-analyser

# With environment file
docker run --env-file .env -p 5000:5000 sentiment-analyser
```

---

## Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Set strong `SECRET_KEY`
- [ ] Switch to PostgreSQL (`DATABASE_URL=postgresql://...`)
- [ ] Use Gunicorn (4 workers) behind Nginx reverse proxy
- [ ] Enable HTTPS (Let's Encrypt / Certbot)
- [ ] Set up log rotation (`logs/` directory)
- [ ] Store model checkpoint on persistent volume (not ephemeral container storage)
- [ ] Configure `MAX_CONTENT_LENGTH` for file upload limits

---

## Dependency Install Notes (Windows)

```bash
# If PyTorch GPU is needed (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If pydub is needed for audio conversion
pip install pydub
# Also install ffmpeg: https://ffmpeg.org/download.html

# If speech_recognition file transcription fails
pip install SpeechRecognition
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Single text prediction |
| POST | `/api/batch_predict` | Upload CSV/Excel for batch |
| GET | `/api/batch_status/<id>` | Poll batch job status |
| GET | `/api/batch_download/<id>` | Download batch results CSV |
| POST | `/api/voice_predict` | Voice/transcript prediction |
| GET | `/api/analytics` | Full analytics data |
| GET | `/api/analytics/trend?days=7` | Daily trend data |
| GET | `/api/analytics/keywords` | Top keywords |
| GET | `/api/analytics/export` | Export history CSV |
| GET | `/api/model_info` | Model metadata summary |
| GET | `/api/history` | Paginated prediction history |
| GET | `/health` | Health check |

---

## File Locations After Training

| File | Path | Contents |
|------|------|----------|
| BERT weights | `data/checkpoints/bert_sentiment/` | model.safetensors + classifier_head.pt |
| Training log | `reports/training_log.csv` | Per-epoch metrics |
| Training history | `reports/training_history.json` | Full JSON history |
| Test metrics | `reports/test_metrics.json` | Final evaluation results |
| Split info | `reports/split_info.json` | Dataset split statistics |
| Loss curve | `reports/loss_curve.png` | Train vs val loss chart |
| Accuracy curve | `reports/accuracy_curve.png` | Accuracy + F1 per epoch |
| Confusion matrix | `reports/confusion_matrix.png` | Normalized heatmap |
| Dashboard chart | `reports/training_dashboard.png` | All-in-one 6-panel figure |

---

## Troubleshooting

**"BERT checkpoint not found"**
→ Run `python train.py` first.

**"No module named 'torch'"**
→ Run `pip install torch` or check that venv is activated.

**"Could not capture microphone audio"**
→ Use the browser mic tab (Web Speech API) — not the server-side endpoint.

**Unicode errors in Windows PowerShell logs**
→ Set `PYTHONIOENCODING=utf-8` in PowerShell before running:
```powershell
$env:PYTHONIOENCODING = "utf-8"
python train.py
```

**"Database locked" error**
→ SQLite does not support concurrent writes. Restart the server if stuck.

**Batch job stuck in "processing"**
→ Check `logs/app.log` for errors. The background thread may have crashed.
