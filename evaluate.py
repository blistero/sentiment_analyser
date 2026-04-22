#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
EVALUATION SCRIPT — Full Metrics + Visualizations
═══════════════════════════════════════════════════════════════════════════════

Evaluates the trained BERT model on a labeled dataset and produces:
  • Accuracy, Precision, Recall, F1 (macro + weighted + per-class)
  • Confusion matrix (normalized + raw)
  • Classification report
  • reports/eval_metrics.csv      — exportable table
  • reports/eval_*.png            — visualizations

USAGE:
  python evaluate.py                         # Test split from Amazon Reviews
  python evaluate.py --file data/test.csv    # Custom labeled CSV
  python evaluate.py --split val             # Use validation split instead
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import os
import json
import csv

from config import Config
from utils.data_loader import load_amazon_reviews, split_dataset, load_custom_csv
from utils.metrics import compute_metrics, print_metrics
from models.bert_model import BertSentimentTrainer
from utils.logger import setup_logger

logger = setup_logger("evaluate", log_file="logs/evaluate.log")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate BERT Sentiment Model")
    p.add_argument("--model-path",  type=str, default="data/checkpoints/bert_sentiment")
    p.add_argument("--file",        type=str, default=None,       help="Custom labeled CSV/Excel")
    p.add_argument("--text-col",    type=str, default="text")
    p.add_argument("--label-col",   type=str, default="label")
    p.add_argument("--split",       type=str, default="test",     choices=["train","val","test"])
    p.add_argument("--max-samples", type=int, default=60000)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--report-dir",  type=str, default="reports")
    p.add_argument("--no-plots",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    cfg = Config()
    cfg.BATCH_SIZE = args.batch_size

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    trainer = BertSentimentTrainer(cfg=cfg)
    trainer.load(args.model_path)

    # Load data
    if args.file:
        logger.info(f"Loading custom dataset: {args.file}")
        df_eval = load_custom_csv(args.file, text_col=args.text_col, label_col=args.label_col)
        split_name = "custom"
    else:
        logger.info(f"Loading Amazon Reviews — using '{args.split}' split")
        df = load_amazon_reviews(max_samples=args.max_samples)
        df_train, df_val, df_test = split_dataset(df)
        df_eval = {"train": df_train, "val": df_val, "test": df_test}[args.split]
        split_name = args.split

    logger.info(f"Evaluating on {len(df_eval):,} samples ({split_name} set)...")
    logger.info(f"Label distribution: {df_eval['label'].value_counts().sort_index().to_dict()}")

    # Predict
    preds, confs, probs_arr = trainer.predict_batch(df_eval["text"].tolist())
    true_labels = df_eval["label"].tolist()

    # Metrics
    metrics = compute_metrics(true_labels, preds, label_names=cfg.SENTIMENT_LABELS)
    print_metrics(metrics)

    # Print per-class detail
    logger.info("\nDetailed Per-Class Breakdown:")
    logger.info(f"{'Class':10s} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'Support':>9}")
    logger.info("-" * 55)
    for label, vals in metrics["per_class"].items():
        logger.info(f"{label:10s} | {vals['precision']:>10.4f} | {vals['recall']:>8.4f} | {vals['f1']:>8.4f} | {vals['support']:>9,}")

    # Save JSON
    metrics_json = {k: v for k, v in metrics.items() if k != "classification_report"}
    json_path = os.path.join(args.report_dir, f"eval_metrics_{split_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"\nMetrics JSON saved: {json_path}")

    # Save CSV
    csv_path = os.path.join(args.report_dir, f"eval_metrics_{split_name}.csv")
    rows = []
    # Summary row
    rows.append({
        "phase": split_name, "class": "macro_avg",
        "precision": metrics["macro_precision"],
        "recall": metrics["macro_recall"],
        "f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "support": len(true_labels),
        "weighted_f1": metrics["weighted_f1"],
    })
    # Per-class rows
    for label, vals in metrics["per_class"].items():
        rows.append({
            "phase": split_name, "class": label,
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1": vals["f1"],
            "accuracy": "",
            "support": vals["support"],
            "weighted_f1": "",
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase","class","precision","recall","f1","accuracy","support","weighted_f1"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Metrics CSV saved: {csv_path}")

    # Visualizations
    if not args.no_plots:
        from utils.visualizations import plot_confusion_matrix, plot_class_distribution
        logger.info("Generating visualizations...")

        cm_path = plot_confusion_matrix(
            true_labels, preds, args.report_dir,
            title=f"Confusion Matrix ({split_name.capitalize()} Set)"
        )
        dist_path = plot_class_distribution(
            true_labels, args.report_dir,
            title=f"{split_name.capitalize()} Set Class Distribution", split_name=split_name.capitalize()
        )
        logger.info(f"  Confusion matrix : {cm_path}")
        logger.info(f"  Class distribution: {dist_path}")

    # Show where model is loaded from
    info_path = os.path.join(args.model_path, "model_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        logger.info(f"\nModel info: {info}")

    print(f"\n✅ Evaluation complete.")
    print(f"   Accuracy   : {metrics['accuracy']:.4f}")
    print(f"   Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"   Reports    : {args.report_dir}/")


if __name__ == "__main__":
    main()
