#!/usr/bin/env python3
"""
BERT SENTIMENT CLASSIFIER -- TRAINING SCRIPT
Sentiment Analysis System

DATASETS:
  Primary : amazon_polarity (HuggingFace) -- 3.6M Amazon product reviews
            Labels: 0=Negative (1-2 star), 1=Neutral (synthetic/SST-2), 2=Positive (4-5 star)
  Neutral : sst2 (Stanford Sentiment Treebank 2) -- relabeled as neutral bridge samples

MODEL:
  Base    : bert-base-uncased  (HuggingFace transformers)
  Head    : Dropout -> Linear(768->256) -> GELU -> Dropout -> Linear(256->3)
  Loss    : CrossEntropyLoss with inverse-frequency class weights

OUTPUTS (saved to reports/):
  training_log.csv         -- per-epoch loss, accuracy, precision, recall, F1
  training_history.json    -- full JSON history
  loss_curve.png           -- train vs val loss
  accuracy_curve.png       -- val accuracy + macro F1
  per_class_metrics.png    -- per-class P/R/F1 bars
  confusion_matrix.png     -- normalized confusion matrix (test set)
  class_distribution_*.png -- label distribution for each split
  dataset_split.png        -- train/val/test proportion bar
  training_dashboard.png   -- all-in-one 6-panel figure
  all_metrics.csv          -- final exportable metrics table
  test_metrics.json        -- serialized test set metrics

USAGE:
  python train.py                          # default (60k samples, 3 epochs)
  python train.py --max-samples 20000      # quick run for testing
  python train.py --epochs 5 --batch 32   # custom hyperparameters
  python train.py --eval-only             # evaluate existing checkpoint
"""
import argparse
import os
import sys
import json
import time

from config import Config
from utils.data_loader import load_amazon_reviews, split_dataset, compute_class_weights
from utils.metrics import compute_metrics, print_metrics
from utils.visualizations import generate_all_visualizations
from utils.logger import setup_logger

logger = setup_logger("train", log_file="logs/train.log")


# --- CLI -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train BERT Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-samples",  type=int,   default=60000,                            help="Max total samples to load from Amazon Reviews")
    p.add_argument("--epochs",       type=int,   default=3,                                help="Training epochs")
    p.add_argument("--batch",        type=int,   default=16,                               help="Batch size")
    p.add_argument("--lr",           type=float, default=2e-5,                             help="Learning rate")
    p.add_argument("--max-length",   type=int,   default=256,                              help="Max BERT token length")
    p.add_argument("--model",        type=str,   default="bert-base-uncased",              help="HuggingFace model name")
    p.add_argument("--save-path",    type=str,   default="data/checkpoints/bert_sentiment",help="Checkpoint directory")
    p.add_argument("--report-dir",   type=str,   default="reports",                        help="Directory for all outputs")
    p.add_argument("--resume",       action="store_true",                                  help="Resume from existing checkpoint")
    p.add_argument("--eval-only",    action="store_true",                                  help="Skip training; run evaluation only")
    p.add_argument("--no-plots",     action="store_true",                                  help="Skip visualization generation")
    return p.parse_args()


# --- Main ------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs("logs",          exist_ok=True)

    # Configuration
    cfg = Config()
    cfg.MAX_TRAIN_SAMPLES = args.max_samples
    cfg.NUM_EPOCHS        = args.epochs
    cfg.BATCH_SIZE        = args.batch
    cfg.LEARNING_RATE     = args.lr
    cfg.MAX_LENGTH        = args.max_length
    cfg.BERT_MODEL_NAME   = args.model

    print_banner(args, cfg)

    # Data loading
    logger.info("STEP 1/5 -- Loading dataset")
    logger.info("  Source        : amazon_polarity (HuggingFace datasets library)")
    logger.info("  Neutral fill  : sst2 (Stanford Sentiment Treebank)")
    logger.info(f"  Max samples   : {cfg.MAX_TRAIN_SAMPLES:,}")

    df = load_amazon_reviews(max_samples=cfg.MAX_TRAIN_SAMPLES)

    total = len(df)
    label_dist = df["label"].value_counts().sort_index().to_dict()
    logger.info(f"  Total loaded  : {total:,}")
    logger.info(f"  Negative (0)  : {label_dist.get(0, 0):,}")
    logger.info(f"  Neutral  (1)  : {label_dist.get(1, 0):,}")
    logger.info(f"  Positive (2)  : {label_dist.get(2, 0):,}")

    # Split
    logger.info("\nSTEP 2/5 -- Splitting dataset")
    df_train, df_val, df_test = split_dataset(df, train_size=cfg.TRAIN_SIZE, val_size=cfg.VAL_SIZE)

    logger.info(f"  Split ratios  : Train={cfg.TRAIN_SIZE:.0%}  Val={cfg.VAL_SIZE:.0%}  Test={1-cfg.TRAIN_SIZE-cfg.VAL_SIZE:.0%}")
    logger.info(f"  Train samples : {len(df_train):,}")
    logger.info(f"  Val samples   : {len(df_val):,}")
    logger.info(f"  Test samples  : {len(df_test):,}")
    logger.info(f"  Stratified    : Yes  (same label ratios in each split)")

    # Class weights
    logger.info("\nSTEP 3/5 -- Computing class weights (imbalance handling)")
    class_weights = compute_class_weights(df_train["label"].tolist())
    for i, (lbl, w) in enumerate(zip(cfg.SENTIMENT_LABELS, class_weights)):
        logger.info(f"  Weight [{lbl}] : {w:.4f}")

    # Save split info
    split_info = {
        "total_samples": total,
        "label_distribution_full": {cfg.SENTIMENT_LABELS[k]: v for k, v in label_dist.items()},
        "train_samples": len(df_train),
        "val_samples":   len(df_val),
        "test_samples":  len(df_test),
        "train_ratio":   cfg.TRAIN_SIZE,
        "val_ratio":     cfg.VAL_SIZE,
        "test_ratio":    round(1 - cfg.TRAIN_SIZE - cfg.VAL_SIZE, 2),
        "class_weights": {cfg.SENTIMENT_LABELS[i]: round(w, 4) for i, w in enumerate(class_weights)},
        "stratified":    True,
    }
    with open(os.path.join(args.report_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Training
    from models.bert_model import BertSentimentTrainer
    trainer = BertSentimentTrainer(cfg=cfg)

    if args.resume and os.path.exists(args.save_path):
        logger.info(f"\nResuming from checkpoint: {args.save_path}")
        trainer.load(args.save_path)

    history = []
    if not args.eval_only:
        logger.info(f"\nSTEP 4/5 -- Training")
        t0 = time.time()
        history = trainer.train(df_train, df_val, class_weights=class_weights, save_path=args.save_path, report_dir=args.report_dir)
        elapsed = time.time() - t0
        logger.info(f"\n  Training completed in {elapsed/60:.1f} minutes")

    # Evaluation on test set
    logger.info(f"\nSTEP 5/5 -- Evaluating on held-out test set")
    logger.info(f"  Loading best checkpoint from {args.save_path}")
    trainer.load(args.save_path)

    test_texts  = df_test["text"].tolist()
    test_labels = df_test["label"].tolist()
    preds, confs, probs = trainer.predict_batch(test_texts)

    test_metrics = compute_metrics(test_labels, preds, label_names=cfg.SENTIMENT_LABELS)
    print_metrics(test_metrics)

    # Save test metrics
    metrics_save = {k: v for k, v in test_metrics.items() if k != "classification_report"}
    with open(os.path.join(args.report_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_save, f, indent=2)

    # Visualizations
    if not args.no_plots and history:
        logger.info(f"\nGenerating visualizations -> {args.report_dir}/")
        last_em = history[-1]
        history_dicts = []
        for em in history:
            d = em.to_dict()
            history_dicts.append(d)

        last_dict = history_dicts[-1]
        last_dict["val_labels_snapshot"] = last_em.val_labels
        last_dict["val_preds_snapshot"]  = last_em.val_preds

        paths = generate_all_visualizations(
            history_dicts,
            df_train, df_val, df_test,
            y_true_test=test_labels,
            y_pred_test=preds,
            test_metrics=test_metrics,
            save_dir=args.report_dir,
        )
        logger.info("  Saved:")
        for name, path in paths.items():
            logger.info(f"    {name:30s} -> {path}")
    elif not history:
        logger.info("  No training history to plot. Run without --eval-only to generate plots.")

    # Final summary
    print_final_summary(args, test_metrics, history)


# --- Helpers ---------------------------------------------------------------

def print_banner(args, cfg):
    sep = "=" * 70
    logger.info("")
    logger.info(sep)
    logger.info("  BERT SENTIMENT CLASSIFIER -- TRAINING PIPELINE")
    logger.info(sep)
    logger.info(f"  Base model    : {cfg.BERT_MODEL_NAME}")
    logger.info(f"  Num labels    : {cfg.NUM_LABELS} (Negative / Neutral / Positive)")
    logger.info(f"  Max samples   : {cfg.MAX_TRAIN_SAMPLES:,}")
    logger.info(f"  Epochs        : {cfg.NUM_EPOCHS}")
    logger.info(f"  Batch size    : {cfg.BATCH_SIZE}")
    logger.info(f"  Learning rate : {cfg.LEARNING_RATE}")
    logger.info(f"  Max length    : {cfg.MAX_LENGTH} tokens")
    logger.info(f"  Save path     : {args.save_path}")
    logger.info(f"  Report dir    : {args.report_dir}")
    logger.info(sep)
    logger.info("")


def print_final_summary(args, test_metrics, history):
    sep = "=" * 70
    logger.info("")
    logger.info(sep)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info(sep)
    logger.info(f"  Test Accuracy     : {test_metrics['accuracy']:.4f}  ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Macro Precision   : {test_metrics['macro_precision']:.4f}")
    logger.info(f"  Macro Recall      : {test_metrics['macro_recall']:.4f}")
    logger.info(f"  Macro F1-Score    : {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1-Score : {test_metrics['weighted_f1']:.4f}")
    logger.info(sep)
    for lbl, vals in test_metrics["per_class"].items():
        logger.info(f"  {lbl:10s}  P={vals['precision']:.4f}  R={vals['recall']:.4f}  F1={vals['f1']:.4f}  N={vals['support']:>6,}")
    logger.info(sep)
    logger.info(f"  Model checkpoint  : {args.save_path}")
    logger.info(f"  Visualizations    : {args.report_dir}")
    logger.info(sep)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. python app.py          -> Start Flask web app")
    logger.info("  2. python predict.py -i   -> Interactive CLI prediction")
    logger.info("  3. python evaluate.py     -> Re-run evaluation")
    logger.info("  4. Open reports/*.png     -> Screenshots for viva")


if __name__ == "__main__":
    main()
