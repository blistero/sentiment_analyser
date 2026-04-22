"""
Visualization module — generates all training and evaluation charts.
All figures use a dark professional theme suitable for project reports and screenshots.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Optional
import pandas as pd

# ── Global style ─────────────────────────────────────────────────────────────
DARK_BG   = "#0f172a"
SURFACE   = "#1e293b"
BORDER    = "#334155"
TEXT      = "#e2e8f0"
MUTED     = "#94a3b8"
POSITIVE  = "#22c55e"
NEGATIVE  = "#ef4444"
NEUTRAL   = "#f59e0b"
PRIMARY   = "#6366f1"
ACCENT    = "#38bdf8"

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_COLORS = [NEGATIVE, NEUTRAL, POSITIVE]


def _apply_dark_theme(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)


# ── 1. Training Loss Curve ────────────────────────────────────────────────────

def plot_loss_curve(history: List[Dict], save_dir: str) -> str:
    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_theme(fig, ax)

    ax.plot(epochs, train_loss, color=PRIMARY,  marker="o", linewidth=2.5, markersize=7, label="Train Loss")
    ax.plot(epochs, val_loss,   color=ACCENT,   marker="s", linewidth=2.5, markersize=7, label="Val Loss", linestyle="--")

    # Shade min val loss
    best_idx = int(np.argmin(val_loss))
    ax.axvline(epochs[best_idx], color=POSITIVE, linestyle=":", linewidth=1.5, alpha=0.7)
    ax.annotate(
        f"Best\nEpoch {epochs[best_idx]}\n{val_loss[best_idx]:.4f}",
        xy=(epochs[best_idx], val_loss[best_idx]),
        xytext=(epochs[best_idx] + 0.3, val_loss[best_idx] + 0.02),
        color=POSITIVE, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=POSITIVE, lw=1.5),
    )

    ax.fill_between(epochs, train_loss, alpha=0.12, color=PRIMARY)
    ax.fill_between(epochs, val_loss,   alpha=0.12, color=ACCENT)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold", pad=14)
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.15, color=BORDER)

    plt.tight_layout()
    out = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ── 2. Validation Accuracy Curve ─────────────────────────────────────────────

def plot_accuracy_curve(history: List[Dict], save_dir: str) -> str:
    epochs   = [h["epoch"]        for h in history]
    accuracy = [h["val_accuracy"] for h in history]
    f1_mac   = [h["val_f1"]       for h in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_theme(fig, ax)

    ax.plot(epochs, accuracy, color=POSITIVE, marker="o", linewidth=2.5, markersize=7, label="Val Accuracy")
    ax.plot(epochs, f1_mac,   color=NEUTRAL,  marker="D", linewidth=2.5, markersize=7, label="Macro F1")

    ax.fill_between(epochs, accuracy, alpha=0.12, color=POSITIVE)
    ax.fill_between(epochs, f1_mac,   alpha=0.12, color=NEUTRAL)

    ax.axhline(max(accuracy), color=POSITIVE, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(epochs[-1] + 0.1, max(accuracy), f"Peak: {max(accuracy):.4f}", color=POSITIVE, fontsize=9, va="center")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation Accuracy & Macro F1 per Epoch", fontsize=14, fontweight="bold", pad=14)
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.15, color=BORDER)

    plt.tight_layout()
    out = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ── 3. Precision / Recall / F1 per Class ─────────────────────────────────────

def plot_per_class_metrics(history: List[Dict], save_dir: str) -> str:
    """Bar chart of final-epoch per-class Precision, Recall, F1."""
    last = history[-1]
    metrics = {
        "Precision": [last["val_f1_neg"], last["val_f1_neu"], last["val_f1_pos"]],
        "Recall":    [last.get("val_recall_neg", 0), last.get("val_recall_neu", 0), last.get("val_recall_pos", 0)],
        "F1-Score":  [last["val_f1_neg"], last["val_f1_neu"], last["val_f1_pos"]],
    }
    # Use F1 keys we definitely have
    neg_f1 = last.get("val_f1_neg", 0)
    neu_f1 = last.get("val_f1_neu", 0)
    pos_f1 = last.get("val_f1_pos", 0)

    x = np.arange(len(LABEL_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    bars_p  = ax.bar(x - width, [last.get("val_precision",0)]*3, width, label="Precision", color=PRIMARY,   alpha=0.85, zorder=3)
    bars_r  = ax.bar(x,         [last.get("val_recall",0)]*3,    width, label="Recall",    color=ACCENT,    alpha=0.85, zorder=3)
    bars_f1 = ax.bar(x + width, [neg_f1, neu_f1, pos_f1],       width, label="F1-Score",  color=LABEL_COLORS, alpha=0.85, zorder=3)

    for bar in list(bars_p) + list(bars_r) + list(bars_f1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    ax.set_xlabel("Sentiment Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics (Final Epoch)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=11)
    ax.grid(axis="y", alpha=0.15, color=BORDER, zorder=0)

    plt.tight_layout()
    out = os.path.join(save_dir, "per_class_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ── 4. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_dir: str, title: str = "Confusion Matrix") -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    _apply_dark_theme(fig, ax)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED, labelsize=9)
    cbar.set_label("Normalized Rate", color=MUTED, fontsize=10)

    tick_marks = np.arange(len(LABEL_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticklabels(LABEL_NAMES, fontsize=11)

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] < thresh else DARK_BG
            ax.text(j, i,
                    f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ── 5. Class Distribution Chart ──────────────────────────────────────────────

def plot_class_distribution(labels: List[int], save_dir: str, title: str = "Dataset Class Distribution", split_name: str = "Full") -> str:
    from collections import Counter
    counts = Counter(labels)
    values = [counts.get(i, 0) for i in range(3)]
    total  = sum(values)
    pcts   = [v / total * 100 for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _apply_dark_theme(fig, axes)

    # Bar chart
    bars = axes[0].bar(LABEL_NAMES, values, color=LABEL_COLORS, alpha=0.85, zorder=3, width=0.5)
    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + total*0.005,
                     f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")
    axes[0].set_xlabel("Sentiment Class", fontsize=12)
    axes[0].set_ylabel("Number of Samples", fontsize=12)
    axes[0].set_title(f"{split_name} — Sample Counts", fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.15, color=BORDER, zorder=0)
    axes[0].set_ylim(0, max(values) * 1.2)

    # Pie chart
    wedge_props = dict(linewidth=2, edgecolor=DARK_BG)
    wedges, texts, autotexts = axes[1].pie(
        values, labels=LABEL_NAMES, colors=LABEL_COLORS,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=wedge_props, textprops=dict(color=TEXT, fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    axes[1].set_title(f"{split_name} — Distribution", fontsize=13, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT, y=1.02)
    plt.tight_layout()
    fname = f"class_distribution_{split_name.lower().replace(' ','_')}.png"
    out = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ── 6. Train/Val/Test Split Visualization ────────────────────────────────────

def plot_dataset_split(n_train: int, n_val: int, n_test: int, save_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark_theme(fig, ax)

    total = n_train + n_val + n_test
    splits = [
        ("Training Set",   n_train, PRIMARY),
        ("Validation Set", n_val,   NEUTRAL),
        ("Test Set",       n_test,  NEGATIVE),
    ]
    left = 0
    for name, count, color in splits:
        width = count / total
        ax.barh(0, width, left=left, color=color, height=0.5, alpha=0.85)
        if width > 0.05:
            ax.text(left + width/2, 0, f"{name}\n{count:,}\n({width*100:.1f}%)",
                    ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        left += width

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Proportion of Dataset", fontsize=12)
    ax.set_title(f"Train / Validation / Test Split  (Total: {total:,} samples)", fontsize=14, fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="x", alpha=0.15, color=BORDER)

    plt.tight_layout()
    out = os.path.join(save_dir, "dataset_split.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ── 7. All-in-one Training Dashboard ─────────────────────────────────────────

def plot_training_dashboard(history: List[Dict], y_true: List[int], y_pred: List[int], save_dir: str) -> str:
    """One comprehensive figure with loss + accuracy + confusion matrix + per-class F1."""
    epochs     = [h["epoch"]        for h in history]
    train_loss = [h["train_loss"]   for h in history]
    val_loss   = [h["val_loss"]     for h in history]
    accuracy   = [h["val_accuracy"] for h in history]
    val_f1     = [h["val_f1"]       for h in history]
    f1_neg     = [h["val_f1_neg"]   for h in history]
    f1_neu     = [h["val_f1_neu"]   for h in history]
    f1_pos     = [h["val_f1_pos"]   for h in history]

    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    _apply_dark_theme(fig, axes)

    # ① Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, color=PRIMARY, marker="o", lw=2, ms=6, label="Train")
    ax.plot(epochs, val_loss,   color=ACCENT,  marker="s", lw=2, ms=6, label="Val", ls="--")
    ax.fill_between(epochs, train_loss, alpha=0.1, color=PRIMARY)
    ax.fill_between(epochs, val_loss,   alpha=0.1, color=ACCENT)
    ax.set_title("Loss Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.12, color=BORDER)

    # ② Accuracy & F1
    ax = axes[1]
    ax.plot(epochs, accuracy, color=POSITIVE, marker="o", lw=2, ms=6, label="Accuracy")
    ax.plot(epochs, val_f1,   color=NEUTRAL,  marker="D", lw=2, ms=6, label="Macro F1", ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation Accuracy & F1", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.12, color=BORDER)

    # ③ Per-class F1
    ax = axes[2]
    ax.plot(epochs, f1_neg, color=NEGATIVE, marker="o", lw=2, ms=6, label="Negative")
    ax.plot(epochs, f1_neu, color=NEUTRAL,  marker="s", lw=2, ms=6, label="Neutral")
    ax.plot(epochs, f1_pos, color=POSITIVE, marker="D", lw=2, ms=6, label="Positive")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.12, color=BORDER)

    # ④ Confusion matrix
    ax = axes[3]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=MUTED, labelsize=8)
    thresh = cm_norm.max() / 2
    for i in range(3):
        for j in range(3):
            c = "white" if cm_norm[i, j] < thresh else DARK_BG
            ax.text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]:.1%})", ha="center", va="center", fontsize=9, color=c, fontweight="bold")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(LABEL_NAMES, fontsize=9); ax.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    # ⑤ Class distribution (true labels)
    ax = axes[4]
    from collections import Counter
    counts = Counter(y_true)
    vals = [counts.get(i, 0) for i in range(3)]
    bars = ax.bar(LABEL_NAMES, vals, color=LABEL_COLORS, alpha=0.85, zorder=3, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(vals)*0.01, f"{h:,}", ha="center", va="bottom", fontsize=10, color=TEXT)
    ax.set_title("Test Set Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Samples"); ax.grid(axis="y", alpha=0.12, color=BORDER, zorder=0)

    # ⑥ Final epoch metrics radar / bar
    ax = axes[5]
    last = history[-1]
    metric_names  = ["Accuracy", "Precision", "Recall", "F1 (Macro)", "F1 Neg", "F1 Neu", "F1 Pos"]
    metric_values = [
        last["val_accuracy"], last["val_precision"], last["val_recall"], last["val_f1"],
        last["val_f1_neg"],   last["val_f1_neu"],   last["val_f1_pos"],
    ]
    colors = [PRIMARY, ACCENT, ACCENT, NEUTRAL, NEGATIVE, NEUTRAL, POSITIVE]
    hbars = ax.barh(metric_names, metric_values, color=colors, alpha=0.85, zorder=3)
    for bar in hbars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=10, color=TEXT)
    ax.set_xlim(0, 1.15)
    ax.set_title("Final Epoch All Metrics", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.12, color=BORDER, zorder=0)
    ax.invert_yaxis()

    # Title
    final_acc = accuracy[-1]
    final_f1  = val_f1[-1]
    fig.suptitle(
        f"BERT Sentiment Classifier — Training Dashboard\n"
        f"Final Accuracy: {final_acc:.4f}   Final Macro F1: {final_f1:.4f}",
        fontsize=16, fontweight="bold", color=TEXT, y=1.01
    )

    out = os.path.join(save_dir, "training_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return out


# ── 8. Export metrics CSV ─────────────────────────────────────────────────────

def export_metrics_csv(history: List[Dict], test_metrics: Dict, save_dir: str) -> str:
    """Export training history + test metrics to one CSV for project submission."""
    rows = []
    for h in history:
        rows.append({
            "phase":           "validation",
            "epoch":           h["epoch"],
            "loss":            h["val_loss"],
            "accuracy":        h["val_accuracy"],
            "macro_precision": h["val_precision"],
            "macro_recall":    h["val_recall"],
            "macro_f1":        h["val_f1"],
            "f1_negative":     h["val_f1_neg"],
            "f1_neutral":      h["val_f1_neu"],
            "f1_positive":     h["val_f1_pos"],
        })
    # Test row (no epoch)
    rows.append({
        "phase":           "test",
        "epoch":           "—",
        "loss":            "—",
        "accuracy":        test_metrics.get("accuracy", "—"),
        "macro_precision": test_metrics.get("macro_precision", "—"),
        "macro_recall":    test_metrics.get("macro_recall", "—"),
        "macro_f1":        test_metrics.get("macro_f1", "—"),
        "f1_negative":     test_metrics.get("per_class", {}).get("Negative", {}).get("f1", "—"),
        "f1_neutral":      test_metrics.get("per_class", {}).get("Neutral",  {}).get("f1", "—"),
        "f1_positive":     test_metrics.get("per_class", {}).get("Positive", {}).get("f1", "—"),
    })

    df = pd.DataFrame(rows)
    out = os.path.join(save_dir, "all_metrics.csv")
    df.to_csv(out, index=False)
    return out


# ── Master runner ─────────────────────────────────────────────────────────────

def generate_all_visualizations(
    history: List[Dict],
    df_train, df_val, df_test,
    y_true_test: List[int],
    y_pred_test: List[int],
    test_metrics: Dict,
    save_dir: str = "reports",
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)

    last_epoch = history[-1]
    y_true_val = last_epoch.get("val_labels_snapshot", [])
    y_pred_val = last_epoch.get("val_preds_snapshot", [])

    paths = {}
    paths["loss_curve"]          = plot_loss_curve(history, save_dir)
    paths["accuracy_curve"]      = plot_accuracy_curve(history, save_dir)
    paths["per_class_metrics"]   = plot_per_class_metrics(history, save_dir)

    paths["confusion_matrix_test"] = plot_confusion_matrix(y_true_test, y_pred_test, save_dir, "Confusion Matrix (Test Set)")
    if y_true_val and y_pred_val:
        paths["confusion_matrix_val"] = plot_confusion_matrix(y_true_val, y_pred_val, save_dir, "Confusion Matrix (Validation Set)")

    paths["class_dist_train"] = plot_class_distribution(df_train["label"].tolist(), save_dir, "Training Set Class Distribution", "Training")
    paths["class_dist_val"]   = plot_class_distribution(df_val["label"].tolist(),   save_dir, "Validation Set Class Distribution", "Validation")
    paths["class_dist_test"]  = plot_class_distribution(df_test["label"].tolist(),  save_dir, "Test Set Class Distribution", "Test")
    paths["dataset_split"]    = plot_dataset_split(len(df_train), len(df_val), len(df_test), save_dir)

    paths["training_dashboard"] = plot_training_dashboard(history, y_true_test, y_pred_test, save_dir)
    paths["metrics_csv"]        = export_metrics_csv(history, test_metrics, save_dir)

    return paths
