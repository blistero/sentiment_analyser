import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS

from config import get_config
from models.database import init_db
from routes.predict_routes import predict_bp
from routes.analytics_routes import analytics_bp
from routes.history_routes import history_bp
from routes.retrain_routes import retrain_bp
from utils.logger import setup_logger

logger = setup_logger()


def create_app(config_class=None):
    app = Flask(__name__)

    cfg_class = config_class or get_config()
    app.config.from_object(cfg_class)
    app.config["SQLALCHEMY_DATABASE_URI"] = cfg_class.DATABASE_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = cfg_class.MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = cfg_class.UPLOAD_FOLDER

    CORS(app)

    # Database
    init_db(app)

    # Blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(retrain_bp)

    # UI Routes
    @app.route("/")
    def dashboard():
        return render_template("index.html")

    @app.route("/analyze")
    def analyze():
        return render_template("analyze.html")

    @app.route("/bulk")
    def bulk():
        return render_template("bulk.html")

    @app.route("/voice")
    def voice():
        return render_template("voice.html")

    @app.route("/history")
    def history():
        return render_template("history.html")

    @app.route("/analytics")
    def analytics():
        return render_template("analytics.html")

    @app.route("/health")
    def health():
        from services.sentiment_service import get_sentiment_service
        service = get_sentiment_service()
        return jsonify({
            "status": "ok",
            "model_loaded": service._model_loaded,
            "device": str(service.bert_trainer.device),
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        if "api" in str(e):
            return jsonify({"error": "Not found"}), 404
        return render_template("index.html"), 404

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({"error": "File too large. Max 32MB."}), 413

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Server error: {e}")
        return jsonify({"error": "Internal server error"}), 500

    return app


def initialize_models(app):
    """Load ML models after app creation."""
    with app.app_context():
        from services.sentiment_service import get_sentiment_service
        service = get_sentiment_service()
        loaded = service.load_models()
        if loaded:
            logger.info("All models loaded and ready.")
        else:
            logger.warning("Running in VADER-only mode. Train the model first: python train.py")


if __name__ == "__main__":
    app = create_app()
    initialize_models(app)
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    logger.info(f"Starting Flask on port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
