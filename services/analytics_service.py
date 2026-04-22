import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, Any, List, Optional
import re

from models.database import db, PredictionRecord
from utils.logger import logger


STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "this", "that", "was", "are", "be", "have", "has", "had",
    "not", "with", "from", "by", "as", "its", "my", "i", "we", "you", "they",
    "he", "she", "so", "do", "did", "just", "would", "could", "should",
}


def get_dashboard_stats() -> Dict[str, Any]:
    total = PredictionRecord.query.count()
    pos = PredictionRecord.query.filter_by(predicted_sentiment="Positive").count()
    neg = PredictionRecord.query.filter_by(predicted_sentiment="Negative").count()
    neu = PredictionRecord.query.filter_by(predicted_sentiment="Neutral").count()
    sarcasm = PredictionRecord.query.filter_by(sarcasm_detected=True).count()

    avg_conf_row = db.session.query(db.func.avg(PredictionRecord.confidence_score)).scalar()
    avg_conf = round(float(avg_conf_row or 0), 4)

    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "sarcasm_count": sarcasm,
        "avg_confidence": avg_conf,
        "positive_pct": round(100 * pos / total, 1) if total else 0,
        "negative_pct": round(100 * neg / total, 1) if total else 0,
        "neutral_pct": round(100 * neu / total, 1) if total else 0,
    }


def get_sentiment_distribution() -> Dict[str, Any]:
    stats = get_dashboard_stats()
    return {
        "labels": ["Positive", "Negative", "Neutral"],
        "values": [stats["positive"], stats["negative"], stats["neutral"]],
        "colors": ["#22c55e", "#ef4444", "#f59e0b"],
    }


def get_daily_trend(days: int = 7) -> Dict[str, Any]:
    since = datetime.utcnow() - timedelta(days=days)
    records = PredictionRecord.query.filter(PredictionRecord.timestamp >= since).all()

    df = pd.DataFrame([
        {"date": r.timestamp.date(), "sentiment": r.predicted_sentiment}
        for r in records
    ])

    if df.empty:
        dates = [(datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days - 1, -1, -1)]
        return {
            "dates": dates,
            "positive": [0] * days,
            "negative": [0] * days,
            "neutral": [0] * days,
        }

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    date_range = pd.date_range(end=datetime.utcnow().date(), periods=days)
    pivot = pivot.reindex(date_range, fill_value=0)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in pivot.index],
        "positive": pivot.get("Positive", pd.Series([0] * days)).tolist(),
        "negative": pivot.get("Negative", pd.Series([0] * days)).tolist(),
        "neutral": pivot.get("Neutral", pd.Series([0] * days)).tolist(),
    }


def get_top_keywords(limit: int = 20, sentiment: Optional[str] = None) -> List[Dict[str, Any]]:
    query = PredictionRecord.query
    if sentiment:
        query = query.filter_by(predicted_sentiment=sentiment)

    records = query.order_by(PredictionRecord.timestamp.desc()).limit(500).all()
    texts = " ".join(r.cleaned_text or r.original_text for r in records if r.cleaned_text or r.original_text)

    words = re.findall(r"\b[a-z]{3,}\b", texts.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    top = Counter(filtered).most_common(limit)

    return [{"word": word, "count": cnt} for word, cnt in top]


def get_source_breakdown() -> Dict[str, Any]:
    records = db.session.query(
        PredictionRecord.source,
        PredictionRecord.predicted_sentiment,
        db.func.count(PredictionRecord.id),
    ).group_by(PredictionRecord.source, PredictionRecord.predicted_sentiment).all()

    breakdown = {}
    for source, sentiment, cnt in records:
        if source not in breakdown:
            breakdown[source] = {"Positive": 0, "Negative": 0, "Neutral": 0}
        breakdown[source][sentiment] = cnt

    return breakdown


def get_confidence_histogram(bins: int = 10) -> Dict[str, Any]:
    records = PredictionRecord.query.with_entities(PredictionRecord.confidence_score).all()
    scores = [r[0] for r in records if r[0] is not None]

    if not scores:
        return {"bins": [], "counts": []}

    hist, edges = np.histogram(scores, bins=bins, range=(0, 1))
    bin_labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    return {"bins": bin_labels, "counts": hist.tolist()}


def get_sarcasm_stats() -> Dict[str, Any]:
    total = PredictionRecord.query.count()
    sarcastic = PredictionRecord.query.filter_by(sarcasm_detected=True).count()
    return {
        "total": total,
        "sarcastic": sarcastic,
        "rate": round(100 * sarcastic / total, 2) if total else 0,
    }


def get_history(page: int = 1, per_page: int = 20, source: str = None, sentiment: str = None) -> Dict[str, Any]:
    query = PredictionRecord.query
    if source:
        query = query.filter_by(source=source)
    if sentiment:
        query = query.filter_by(predicted_sentiment=sentiment)

    pagination = query.order_by(PredictionRecord.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)

    return {
        "records": [r.to_dict() for r in pagination.items],
        "total": pagination.total,
        "pages": pagination.pages,
        "current_page": page,
        "per_page": per_page,
    }


def export_history_csv() -> str:
    records = PredictionRecord.query.order_by(PredictionRecord.timestamp.desc()).all()
    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)
    path = "data/uploads/export_history.csv"
    df.to_csv(path, index=False)
    return path
