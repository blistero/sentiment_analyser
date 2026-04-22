from flask import Blueprint, jsonify, request
from services.analytics_service import get_history
from models.database import db, PredictionRecord

history_bp = Blueprint("history", __name__, url_prefix="/api")


@history_bp.route("/history", methods=["GET"])
def history():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    source = request.args.get("source")
    sentiment = request.args.get("sentiment")
    return jsonify(get_history(page=page, per_page=per_page, source=source, sentiment=sentiment))


@history_bp.route("/history/<int:record_id>", methods=["GET"])
def history_detail(record_id):
    record = PredictionRecord.query.get_or_404(record_id)
    return jsonify(record.to_dict())


@history_bp.route("/history/<int:record_id>", methods=["DELETE"])
def history_delete(record_id):
    record = PredictionRecord.query.get_or_404(record_id)
    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Record deleted"})


@history_bp.route("/history/clear", methods=["DELETE"])
def history_clear():
    PredictionRecord.query.delete()
    db.session.commit()
    return jsonify({"message": "History cleared"})
