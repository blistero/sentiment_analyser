import json
import os
from flask import Blueprint, jsonify, request, send_file
from services.analytics_service import (
    get_dashboard_stats,
    get_sentiment_distribution,
    get_daily_trend,
    get_top_keywords,
    get_source_breakdown,
    get_confidence_histogram,
    get_sarcasm_stats,
    export_history_csv,
)
from utils.logger import logger

analytics_bp = Blueprint("analytics", __name__, url_prefix="/api")


@analytics_bp.route("/analytics", methods=["GET"])
def analytics():
    days = int(request.args.get("days", 7))
    return jsonify({
        "dashboard": get_dashboard_stats(),
        "distribution": get_sentiment_distribution(),
        "trend": get_daily_trend(days=days),
        "top_keywords": get_top_keywords(limit=20),
        "source_breakdown": get_source_breakdown(),
        "confidence_histogram": get_confidence_histogram(),
        "sarcasm": get_sarcasm_stats(),
    })


@analytics_bp.route("/analytics/distribution", methods=["GET"])
def distribution():
    return jsonify(get_sentiment_distribution())


@analytics_bp.route("/analytics/trend", methods=["GET"])
def trend():
    days = int(request.args.get("days", 7))
    return jsonify(get_daily_trend(days=days))


@analytics_bp.route("/analytics/keywords", methods=["GET"])
def keywords():
    sentiment = request.args.get("sentiment")
    limit = int(request.args.get("limit", 20))
    return jsonify(get_top_keywords(limit=limit, sentiment=sentiment))


@analytics_bp.route("/analytics/export", methods=["GET"])
def export():
    path = export_history_csv()
    return send_file(path, as_attachment=True, download_name="sentiment_history.csv")


@analytics_bp.route("/model_info", methods=["GET"])
def model_info():
    """Return model metadata for the dashboard summary card."""
    info = {}

    model_info_path = "data/checkpoints/bert_sentiment/model_info.json"
    if os.path.exists(model_info_path):
        with open(model_info_path, encoding="utf-8") as f:
            info.update(json.load(f))

    history_path = "reports/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        if history:
            last = history[-1]
            info["epochs_trained"] = last.get("epoch", len(history))
            info["best_val_accuracy"] = round(last.get("val_accuracy", 0), 4)
            info["best_val_f1"] = round(last.get("val_f1", 0), 4)
            info["final_train_loss"] = round(last.get("train_loss", 0), 4)
            info["training_history"] = history

    test_path = "reports/test_metrics.json"
    if os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as f:
            test_metrics = json.load(f)
        info["test_accuracy"] = round(test_metrics.get("accuracy", 0), 4)
        info["test_macro_f1"] = round(test_metrics.get("macro_f1", 0), 4)
        info["per_class"] = test_metrics.get("per_class", {})

    split_path = "reports/split_info.json"
    if os.path.exists(split_path):
        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        info["dataset_size"] = split.get("total_samples", 0)
        info["train_samples"] = split.get("train_samples", 0)
        info["val_samples"] = split.get("val_samples", 0)
        info["test_samples"] = split.get("test_samples", 0)

    # Checkpoint mtime as last_trained
    ckpt = "data/checkpoints/bert_sentiment/classifier_head.pt"
    if os.path.exists(ckpt):
        import datetime
        mtime = os.path.getmtime(ckpt)
        info["last_trained"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    info["model_loaded"] = os.path.exists(model_info_path)
    return jsonify(info)
