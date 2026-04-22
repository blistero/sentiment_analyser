import os
import uuid
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from models.database import db, RetrainingJob
from utils.logger import logger

retrain_bp = Blueprint("retrain", __name__, url_prefix="/api")


@retrain_bp.route("/retrain", methods=["POST"])
def retrain():
    """Start a retraining job with a new labeled CSV/Excel dataset."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"csv", "xlsx"}:
        return jsonify({"error": "Use csv or xlsx"}), 400

    os.makedirs("data/uploads", exist_ok=True)
    save_path = f"data/uploads/retrain_{uuid.uuid4().hex}.{ext}"
    file.save(save_path)

    epochs = int(request.form.get("epochs", 3))
    text_col = request.form.get("text_col", "text")
    label_col = request.form.get("label_col", "label")

    job = RetrainingJob(dataset_path=save_path, epochs=epochs, status="pending")
    db.session.add(job)
    db.session.commit()
    job_id = job.id

    app = current_app._get_current_object()
    thread = threading.Thread(target=_run_retrain, args=(job_id, save_path, epochs, text_col, label_col, app), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"}), 202


@retrain_bp.route("/retrain/<int:job_id>", methods=["GET"])
def retrain_status(job_id):
    job = RetrainingJob.query.get_or_404(job_id)
    return jsonify(job.to_dict())


def _run_retrain(job_id: int, dataset_path: str, epochs: int, text_col: str, label_col: str, app):
    import sys
    from io import StringIO
    from utils.data_loader import load_custom_csv, split_dataset, compute_class_weights
    from services.sentiment_service import get_sentiment_service

    log_output = StringIO()

    def log(msg):
        logger.info(msg)
        log_output.write(msg + "\n")

    with app.app_context():
        job = RetrainingJob.query.get(job_id)
        job.status = "running"
        db.session.commit()

    try:
        log("Loading dataset...")
        df = load_custom_csv(dataset_path, text_col=text_col, label_col=label_col)
        log(f"Loaded {len(df)} records. Distribution:\n{df['label'].value_counts().to_dict()}")

        df_train, df_val, df_test = split_dataset(df)
        class_weights = compute_class_weights(df_train["label"].tolist())
        log(f"Class weights: {class_weights}")

        service = get_sentiment_service()
        cfg = service.cfg
        cfg.NUM_EPOCHS = epochs

        log("Starting fine-tuning...")
        history = service.bert_trainer.train(df_train, df_val, class_weights=class_weights)
        log(f"Training complete. Final val acc: {history['val_acc'][-1]:.4f}")

        final_acc = history["val_acc"][-1]

        with app.app_context():
            job = RetrainingJob.query.get(job_id)
            job.status = "done"
            job.final_accuracy = final_acc
            job.log_output = log_output.getvalue()
            job.completed_at = datetime.utcnow()
            db.session.commit()

    except Exception as e:
        logger.error(f"Retraining job {job_id} failed: {e}", exc_info=True)
        with app.app_context():
            job = RetrainingJob.query.get(job_id)
            if job:
                job.status = "failed"
                job.log_output = str(e)
                db.session.commit()
