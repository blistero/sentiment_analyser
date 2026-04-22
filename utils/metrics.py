import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import List, Dict, Any


def compute_metrics(y_true: List[int], y_pred: List[int], label_names: List[str] = None) -> Dict[str, Any]:
    if label_names is None:
        label_names = ["Negative", "Neutral", "Positive"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(label_names))), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    per_class = {}
    for i, label in enumerate(label_names):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0
        ),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"Macro Precision   : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall      : {metrics['macro_recall']:.4f}")
    print("\nPer-Class Metrics:")
    for label, vals in metrics["per_class"].items():
        print(f"  {label:10s} -> P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f} Support={vals['support']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("=" * 60)
