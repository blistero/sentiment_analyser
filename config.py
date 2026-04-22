import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///sentiment.db")

    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "data/checkpoints/bert_sentiment")
    SARCASM_MODEL_PATH = os.getenv("SARCASM_MODEL_PATH", "data/checkpoints/sarcasm_model")
    BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")

    # Training hyperparameters
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 256))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 500))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", 1.0))

    # Dataset
    TRAIN_SIZE = float(os.getenv("TRAIN_SIZE", 0.8))
    VAL_SIZE = float(os.getenv("VAL_SIZE", 0.1))
    TEST_SIZE = float(os.getenv("TEST_SIZE", 0.1))
    MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", 100000))

    # Sentiment labels
    SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
    NUM_LABELS = 3
    LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
    ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

    # VADER hybrid thresholds
    VADER_COMPOUND_POSITIVE = float(os.getenv("VADER_COMPOUND_POSITIVE", 0.05))
    VADER_COMPOUND_NEGATIVE = float(os.getenv("VADER_COMPOUND_NEGATIVE", -0.05))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

    # File uploads
    UPLOAD_FOLDER = "data/uploads"
    ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "wav", "mp3", "ogg", "flac"}
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    DATABASE_URL = "sqlite:///test_sentiment.db"


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}

def get_config():
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
