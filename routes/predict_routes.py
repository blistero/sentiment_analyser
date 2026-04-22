import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from models.database import db, PredictionRecord, BatchJob
from services.sentiment_service import get_sentiment_service
from services.batch_service import process_batch_file
from services.voice_service import save_upload_audio, transcribe_audio_file, transcribe_microphone
from utils.logger import logger

predict_bp = Blueprint("predict", __name__, url_prefix="/api")


@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 10000:
        return jsonify({"error": "Text exceeds 10,000 character limit"}), 400

    service = get_sentiment_service()
    result = service.predict(text, source="api")

    # Persist to DB
    try:
        record = PredictionRecord(
            original_text=result["original_text"][:2000],
            cleaned_text=result["cleaned_text"][:2000],
            predicted_sentiment=result["predicted_sentiment"],
            bert_sentiment=result.get("bert_sentiment"),
            vader_sentiment=result.get("vader_sentiment"),
            sarcasm_detected=result["sarcasm_detected"],
            sarcasm_confidence=result["sarcasm_confidence"],
            confidence_score=result["confidence_score"],
            bert_negative_prob=result["probabilities"]["negative"],
            bert_neutral_prob=result["probabilities"]["neutral"],
            bert_positive_prob=result["probabilities"]["positive"],
            vader_compound=result["vader_scores"]["compound"],
            source="api",
            processing_time_ms=result["processing_time_ms"],
        )
        db.session.add(record)
        db.session.commit()
        result["id"] = record.id
    except Exception as e:
        logger.error(f"DB save failed: {e}")

    return jsonify(result)


@predict_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {"csv", "xlsx", "xls"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}. Use csv/xlsx"}), 400

    job_id = str(uuid.uuid4())
    os.makedirs("data/uploads", exist_ok=True)
    save_path = f"data/uploads/{job_id}_{file.filename}"
    file.save(save_path)

    # Create batch job record
    job = BatchJob(id=job_id, filename=file.filename, status="pending")
    db.session.add(job)
    db.session.commit()

    text_col = request.form.get("text_col", "text")
    service = get_sentiment_service()
    process_batch_file(save_path, job_id, service, text_col=text_col, app=current_app._get_current_object())

    return jsonify({"job_id": job_id, "status": "processing", "message": "Batch job started"}), 202


@predict_bp.route("/batch_status/<job_id>", methods=["GET"])
def batch_status(job_id):
    job = BatchJob.query.get_or_404(job_id)
    return jsonify(job.to_dict())


@predict_bp.route("/batch_download/<job_id>", methods=["GET"])
def batch_download(job_id):
    from flask import send_file
    job = BatchJob.query.get_or_404(job_id)

    if job.status != "done" or not job.result_path:
        return jsonify({"error": "Result not ready"}), 404
    if not os.path.exists(job.result_path):
        return jsonify({"error": "Result file missing"}), 404

    return send_file(job.result_path, as_attachment=True, download_name=f"sentiment_{job_id}.csv")


@predict_bp.route("/voice_predict", methods=["POST"])
def voice_predict():
    service = get_sentiment_service()

    # File upload
    if "audio" in request.files:
        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "Empty audio filename"}), 400

        saved_path = save_upload_audio(audio_file, "data/uploads")
        transcript, tconf = transcribe_audio_file(saved_path)

        if not transcript:
            return jsonify({"error": "Could not transcribe audio. Check audio quality."}), 422

        result = service.predict(transcript, source="voice")
        result["transcript"] = transcript
        result["transcript_confidence"] = tconf

        # Persist
        try:
            record = PredictionRecord(
                original_text=transcript[:2000],
                cleaned_text=result["cleaned_text"][:2000],
                predicted_sentiment=result["predicted_sentiment"],
                bert_sentiment=result.get("bert_sentiment"),
                vader_sentiment=result.get("vader_sentiment"),
                sarcasm_detected=result["sarcasm_detected"],
                sarcasm_confidence=result["sarcasm_confidence"],
                confidence_score=result["confidence_score"],
                bert_negative_prob=result["probabilities"]["negative"],
                bert_neutral_prob=result["probabilities"]["neutral"],
                bert_positive_prob=result["probabilities"]["positive"],
                vader_compound=result["vader_scores"]["compound"],
                source="voice",
                processing_time_ms=result.get("processing_time_ms"),
            )
            db.session.add(record)
            db.session.commit()
            result["id"] = record.id
        except Exception as e:
            logger.error(f"Voice DB save failed: {e}")

        return jsonify(result)

    # Browser Web Speech API transcript (primary mic path — no PyAudio needed)
    elif request.json and request.json.get("transcript"):
        transcript = request.json["transcript"].strip()
        if not transcript:
            return jsonify({"error": "Empty transcript"}), 400
        if len(transcript) > 10000:
            return jsonify({"error": "Transcript too long"}), 400

        result = service.predict(transcript, source="voice")
        result["transcript"] = transcript
        result["transcript_confidence"] = 1.0

        try:
            record = PredictionRecord(
                original_text=transcript[:2000],
                cleaned_text=result["cleaned_text"][:2000],
                predicted_sentiment=result["predicted_sentiment"],
                bert_sentiment=result.get("bert_sentiment"),
                vader_sentiment=result.get("vader_sentiment"),
                sarcasm_detected=result["sarcasm_detected"],
                sarcasm_confidence=result["sarcasm_confidence"],
                confidence_score=result["confidence_score"],
                bert_negative_prob=result["probabilities"]["negative"],
                bert_neutral_prob=result["probabilities"]["neutral"],
                bert_positive_prob=result["probabilities"]["positive"],
                vader_compound=result["vader_scores"]["compound"],
                source="voice",
                processing_time_ms=result.get("processing_time_ms"),
            )
            db.session.add(record)
            db.session.commit()
            result["id"] = record.id
        except Exception as e:
            logger.error(f"Voice DB save failed: {e}")

        return jsonify(result)

    # Legacy server-side microphone (requires PyAudio — kept as fallback)
    elif request.json and request.json.get("mode") == "microphone":
        duration = int(request.json.get("duration", 5))
        transcript, tconf = transcribe_microphone(duration=duration)

        if not transcript:
            return jsonify({"error": "Could not capture microphone audio. Use the browser mic tab instead."}), 422

        result = service.predict(transcript, source="voice")
        result["transcript"] = transcript
        result["transcript_confidence"] = tconf
        return jsonify(result)

    return jsonify({"error": "Provide 'audio' file or 'transcript' text"}), 400
