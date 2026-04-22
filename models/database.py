import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionRecord(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text)
    predicted_sentiment = db.Column(db.String(20), nullable=False)
    bert_sentiment = db.Column(db.String(20))
    vader_sentiment = db.Column(db.String(20))
    sarcasm_detected = db.Column(db.Boolean, default=False)
    sarcasm_confidence = db.Column(db.Float, default=0.0)
    confidence_score = db.Column(db.Float, nullable=False)
    bert_negative_prob = db.Column(db.Float, default=0.0)
    bert_neutral_prob = db.Column(db.Float, default=0.0)
    bert_positive_prob = db.Column(db.Float, default=0.0)
    vader_compound = db.Column(db.Float, default=0.0)
    source = db.Column(db.String(50), default="api")  # api, batch, voice, web
    batch_id = db.Column(db.String(64))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time_ms = db.Column(db.Float)

    def to_dict(self):
        return {
            "id": self.id,
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "predicted_sentiment": self.predicted_sentiment,
            "bert_sentiment": self.bert_sentiment,
            "vader_sentiment": self.vader_sentiment,
            "sarcasm_detected": self.sarcasm_detected,
            "sarcasm_confidence": round(self.sarcasm_confidence, 4),
            "confidence_score": round(self.confidence_score, 4),
            "probabilities": {
                "negative": round(self.bert_negative_prob, 4),
                "neutral": round(self.bert_neutral_prob, 4),
                "positive": round(self.bert_positive_prob, 4),
            },
            "vader_compound": round(self.vader_compound, 4),
            "source": self.source,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "processing_time_ms": round(self.processing_time_ms, 2) if self.processing_time_ms else None,
        }


class BatchJob(db.Model):
    __tablename__ = "batch_jobs"

    id = db.Column(db.String(64), primary_key=True)
    filename = db.Column(db.String(255))
    status = db.Column(db.String(20), default="pending")  # pending, processing, done, failed
    total_records = db.Column(db.Integer, default=0)
    processed_records = db.Column(db.Integer, default=0)
    positive_count = db.Column(db.Integer, default=0)
    negative_count = db.Column(db.Integer, default=0)
    neutral_count = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    result_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "error_message": self.error_message,
            "result_path": self.result_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class RetrainingJob(db.Model):
    __tablename__ = "retraining_jobs"

    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(20), default="pending")
    dataset_path = db.Column(db.String(255))
    epochs = db.Column(db.Integer, default=3)
    final_accuracy = db.Column(db.Float)
    log_output = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "dataset_path": self.dataset_path,
            "epochs": self.epochs,
            "final_accuracy": self.final_accuracy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
