import os
import uuid
import threading
import pandas as pd
from datetime import datetime
from typing import Optional

from services.sentiment_service import SentimentService
from utils.logger import logger


def process_batch_file(
    file_path: str,
    job_id: str,
    sentiment_service: SentimentService,
    text_col: str = "text",
    app=None,
) -> str:
    """
    Process a CSV/Excel file in a background thread.
    Returns the result file path.
    """
    from models.database import db, BatchJob

    def _run():
        result_path = None
        try:
            # Read file
            if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # Detect text column
            col = text_col if text_col in df.columns else _detect_text_col(df)
            if col is None:
                raise ValueError("No recognizable text column found in the uploaded file.")

            total = len(df)

            with app.app_context():
                job = BatchJob.query.get(job_id)
                job.status = "processing"
                job.total_records = total
                db.session.commit()

            texts = df[col].fillna("").astype(str).tolist()

            # Batch prediction
            CHUNK = 64
            results = []
            for i in range(0, len(texts), CHUNK):
                chunk = texts[i : i + CHUNK]
                chunk_results = sentiment_service.predict_batch_texts(chunk, source="batch")
                results.extend(chunk_results)

                with app.app_context():
                    job = BatchJob.query.get(job_id)
                    job.processed_records = min(i + CHUNK, total)
                    db.session.commit()

            # Build result dataframe
            result_df = df.copy()
            result_df["predicted_sentiment"] = [r["predicted_sentiment"] for r in results]
            result_df["confidence_score"] = [r["confidence_score"] for r in results]
            result_df["sarcasm_detected"] = [r["sarcasm_detected"] for r in results]
            result_df["vader_compound"] = [r["vader_compound"] for r in results]
            result_df["cleaned_text"] = [r["cleaned_text"] for r in results]

            # Save result
            os.makedirs("data/uploads/results", exist_ok=True)
            result_path = f"data/uploads/results/{job_id}_result.csv"
            result_df.to_csv(result_path, index=False)

            pos = sum(1 for r in results if r["predicted_sentiment"] == "Positive")
            neg = sum(1 for r in results if r["predicted_sentiment"] == "Negative")
            neu = sum(1 for r in results if r["predicted_sentiment"] == "Neutral")

            with app.app_context():
                job = BatchJob.query.get(job_id)
                job.status = "done"
                job.processed_records = total
                job.positive_count = pos
                job.negative_count = neg
                job.neutral_count = neu
                job.result_path = result_path
                job.completed_at = datetime.utcnow()
                db.session.commit()

            logger.info(f"Batch job {job_id} completed: {total} records processed.")

            # Persist individual records
            from models.database import PredictionRecord
            with app.app_context():
                for r in results:
                    rec = PredictionRecord(
                        original_text=r["original_text"][:2000],
                        cleaned_text=r["cleaned_text"][:2000],
                        predicted_sentiment=r["predicted_sentiment"],
                        bert_sentiment=r.get("bert_sentiment"),
                        vader_sentiment=r.get("vader_sentiment"),
                        sarcasm_detected=r["sarcasm_detected"],
                        sarcasm_confidence=r["sarcasm_confidence"],
                        confidence_score=r["confidence_score"],
                        bert_negative_prob=r["probabilities"]["negative"],
                        bert_neutral_prob=r["probabilities"]["neutral"],
                        bert_positive_prob=r["probabilities"]["positive"],
                        vader_compound=r["vader_compound"],
                        source="batch",
                        batch_id=job_id,
                    )
                    db.session.add(rec)
                db.session.commit()

        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}", exc_info=True)
            with app.app_context():
                job = BatchJob.query.get(job_id)
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    db.session.commit()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job_id


def _detect_text_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["text", "review", "review_text", "content", "comment", "description", "message", "feedback"]
    for col in candidates:
        if col in df.columns:
            return col
    # Pick the column with the longest average string
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if not str_cols:
        return None
    best = max(str_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())
    return best
