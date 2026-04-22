#!/usr/bin/env python3
"""
CLI prediction script.

Usage:
    python predict.py "This product is amazing!"
    python predict.py --file reviews.csv --output results.csv
    python predict.py --interactive
"""
import argparse
import sys
import json
import pandas as pd

from config import Config
from services.sentiment_service import SentimentService
from utils.logger import setup_logger

logger = setup_logger("predict")


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Prediction CLI")
    parser.add_argument("text", nargs="?", help="Text to analyze (positional)")
    parser.add_argument("--file", type=str, help="CSV file for batch prediction")
    parser.add_argument("--text-col", type=str, default="text", help="Column name for text")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--model-path", type=str, default=None, help="Override model path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    return parser.parse_args()


def load_service(model_path=None):
    cfg = Config()
    if model_path:
        cfg.MODEL_PATH = model_path
    service = SentimentService(cfg=cfg)
    service.load_models()
    return service


def print_result(result: dict, as_json: bool = False):
    if as_json:
        print(json.dumps(result, indent=2))
        return

    sentiment = result["predicted_sentiment"]
    conf = result["confidence_score"]
    sarcasm = "YES" if result["sarcasm_detected"] else "no"
    probs = result["probabilities"]

    icons = {"Positive": "✅", "Negative": "❌", "Neutral": "⚪"}
    icon = icons.get(sentiment, "")

    print(f"\n{'='*50}")
    print(f"  Sentiment   : {icon} {sentiment} (conf={conf:.2%})")
    print(f"  BERT pred   : {result['bert_sentiment']}")
    print(f"  VADER pred  : {result['vader_sentiment']}")
    print(f"  Sarcasm     : {sarcasm}")
    print(f"  Probabilities:")
    print(f"    Negative: {probs['negative']:.2%}")
    print(f"    Neutral : {probs['neutral']:.2%}")
    print(f"    Positive: {probs['positive']:.2%}")
    print(f"  VADER compound: {result['vader_scores']['compound']:.4f}")
    print(f"  Time: {result['processing_time_ms']:.1f}ms")
    print(f"{'='*50}\n")


def main():
    args = parse_args()
    service = load_service(args.model_path)

    if args.interactive:
        print("Sentiment Analyzer — Interactive Mode (type 'quit' to exit)")
        while True:
            try:
                text = input("\n> Enter text: ").strip()
                if text.lower() in {"quit", "exit", "q"}:
                    break
                if not text:
                    continue
                result = service.predict(text)
                print_result(result, as_json=args.json)
            except KeyboardInterrupt:
                break
        return

    if args.file:
        if args.file.endswith(".xlsx"):
            df = pd.read_excel(args.file)
        else:
            df = pd.read_csv(args.file)

        if args.text_col not in df.columns:
            print(f"Error: column '{args.text_col}' not found. Columns: {df.columns.tolist()}")
            sys.exit(1)

        texts = df[args.text_col].fillna("").astype(str).tolist()
        print(f"Predicting {len(texts)} records...")
        results = service.predict_batch_texts(texts)

        df["predicted_sentiment"] = [r["predicted_sentiment"] for r in results]
        df["confidence_score"] = [r["confidence_score"] for r in results]
        df["sarcasm_detected"] = [r["sarcasm_detected"] for r in results]
        df["bert_sentiment"] = [r["bert_sentiment"] for r in results]
        df["vader_compound"] = [r["vader_compound"] for r in results]
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for r in results:
            counts[r["predicted_sentiment"]] = counts.get(r["predicted_sentiment"], 0) + 1
        print(f"\nSummary: {counts}")
        return

    if args.text:
        result = service.predict(args.text)
        print_result(result, as_json=args.json)
        return

    print("Provide text, --file, or --interactive. Use -h for help.")
    sys.exit(1)


if __name__ == "__main__":
    main()
