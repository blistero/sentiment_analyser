"""
═══════════════════════════════════════════════════════════════════════════════
DATA LOADING PIPELINE — Full Dataset Transparency
═══════════════════════════════════════════════════════════════════════════════

PRIMARY DATASET: amazon_polarity
  Source      : HuggingFace Datasets (McAuley et al.)
  URL         : https://huggingface.co/datasets/amazon_polarity
  Size        : 3,600,000 training + 400,000 test examples
  Columns     : title (str), content (str), label (0=neg, 1=pos)
  Rating map  : 1–2 stars → Negative (label=0), 4–5 stars → Positive (label=2)
  Preprocessing: URL removal, HTML stripping, Unicode normalization,
                 repeated-char fix, whitespace normalization

NEUTRAL SAMPLES: sst2 (Stanford Sentiment Treebank 2)
  Source      : HuggingFace Datasets
  URL         : https://huggingface.co/datasets/sst2
  Size        : 67,349 training sentences
  Rationale   : Amazon reviews have no 3-star (neutral) label natively.
                SST-2 provides short, opinion-light sentences relabeled
                as neutral to balance the 3-class setup.
  Label       : All SST-2 samples → Neutral (label=1)

FALLBACK DATASET: yelp_review_full
  Source      : HuggingFace Datasets
  URL         : https://huggingface.co/datasets/yelp_review_full
  Used when   : amazon_polarity download fails
  Rating map  : 0–1 → Negative, 2 → Neutral, 3–4 → Positive

LABEL ENCODING:
  0 → Negative  (1–2 star reviews, critical feedback)
  1 → Neutral   (3-star / factual / mixed reviews)
  2 → Positive  (4–5 star reviews, praise)

SPLIT STRATEGY:
  Stratified train/val/test split using sklearn.train_test_split
  Ensures same class proportion across all splits.
  Default ratio: 80% train / 10% validation / 10% test

IMBALANCE HANDLING:
  sklearn compute_class_weight("balanced") → inverse frequency weights
  Passed to CrossEntropyLoss weight parameter during training.
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Optional, List
from utils.text_cleaner import clean_text
from utils.logger import logger
from config import Config


# ─── Amazon Reviews (Primary) ────────────────────────────────────────────────

def load_amazon_reviews(max_samples: int = 60000, cache_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load and preprocess Amazon product reviews.

    Pipeline:
      1. Load amazon_polarity from HuggingFace datasets
      2. Sample up to max_samples // 3 from each class
      3. Load neutral samples from sst2
      4. Merge, clean text, shuffle
      5. Cache to parquet for subsequent runs

    Returns:
      DataFrame with columns: text (str), label (int 0/1/2)
    """
    cache_path = os.path.join(cache_dir, f"amazon_reviews_{max_samples}.parquet")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached dataset: {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Cache hit — {len(df):,} samples, distribution: {df['label'].value_counts().to_dict()}")
        return df

    os.makedirs(cache_dir, exist_ok=True)
    samples_per_class = max_samples // 3

    logger.info("Downloading Amazon Reviews dataset (amazon_polarity)...")
    logger.info("  HuggingFace dataset: amazon_polarity")
    logger.info("  Columns used: 'content' (review text), 'label' (0=neg, 1=pos)")
    logger.info(f"  Sampling {samples_per_class:,} per class")

    try:
        ds = load_dataset("amazon_polarity", split="train", trust_remote_code=True)
        df_raw = ds.to_pandas()
        df_raw = df_raw.rename(columns={"content": "text", "label": "polarity"})

        df_neg = (
            df_raw[df_raw["polarity"] == 0]
            .sample(n=min(samples_per_class, len(df_raw[df_raw["polarity"] == 0])), random_state=42)
            .copy()
        )
        df_pos = (
            df_raw[df_raw["polarity"] == 1]
            .sample(n=min(samples_per_class, len(df_raw[df_raw["polarity"] == 1])), random_state=42)
            .copy()
        )
        df_neg["label"] = 0   # Negative
        df_pos["label"] = 2   # Positive
        df_main = pd.concat([df_neg, df_pos], ignore_index=True)[["text", "label"]]

        logger.info(f"  amazon_polarity loaded: {len(df_neg):,} neg + {len(df_pos):,} pos")

    except Exception as e:
        logger.warning(f"amazon_polarity failed ({e}). Using yelp_review_full fallback...")
        df_main = _load_yelp_fallback(samples_per_class)

    # Neutral samples from SST-2
    logger.info("Loading neutral samples from sst2 (Stanford Sentiment Treebank 2)...")
    df_neutral = _load_neutral_from_sst2(samples_per_class)
    logger.info(f"  Neutral samples loaded: {len(df_neutral):,}")

    # Merge
    df_all = pd.concat([df_main, df_neutral], ignore_index=True)[["text", "label"]]

    # Clean text
    logger.info("Cleaning and preprocessing text...")
    df_all["text"] = df_all["text"].astype(str).apply(
        lambda t: clean_text(t, remove_urls=True, remove_html=True, fix_repeated_chars=True)
    )
    df_all = df_all[df_all["text"].str.len() > 10].dropna()

    # Shuffle
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    # Cache
    df_all.to_parquet(cache_path, index=False)
    dist = df_all["label"].value_counts().sort_index().to_dict()
    logger.info(f"Dataset ready — {len(df_all):,} samples | Distribution: {dist}")
    logger.info(f"Cached to: {cache_path}")
    return df_all


def _load_yelp_fallback(samples_per_class: int) -> pd.DataFrame:
    """Fallback: yelp_review_full with 5 star ratings mapped to 3 classes."""
    logger.info("  Loading yelp_review_full (5-class → 3-class mapping)...")
    ds = load_dataset("yelp_review_full", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    # yelp labels 0–4 (stars 1–5)
    df["label"] = df["label"].apply(lambda x: 0 if x <= 1 else (1 if x == 2 else 2))
    rows = []
    for label in [0, 2]:
        sub = df[df["label"] == label]
        rows.append(sub.sample(n=min(samples_per_class, len(sub)), random_state=42))
    return pd.concat(rows, ignore_index=True)[["text", "label"]]


def _load_neutral_from_sst2(n_samples: int) -> pd.DataFrame:
    """Load SST-2 sentences and relabel as Neutral (class 1)."""
    try:
        ds = load_dataset("sst2", split="train", trust_remote_code=True)
        df = ds.to_pandas().rename(columns={"sentence": "text"})
        df["label"] = 1  # Neutral
        df = df.sample(n=min(n_samples, len(df)), random_state=42)[["text", "label"]]
        return df
    except Exception as e:
        logger.warning(f"sst2 failed ({e}). Using synthetic neutral samples.")
        return _synthetic_neutral(n_samples)


def _synthetic_neutral(n: int) -> pd.DataFrame:
    templates = [
        "The product arrived on time and does what it says.",
        "Average quality, nothing special but not bad either.",
        "It works as described. No complaints.",
        "Decent item for the price. Does the job.",
        "Standard product. Met my expectations.",
        "Functional and straightforward, as expected.",
        "The item matches the description.",
        "Works fine. Not exceptional, but acceptable.",
        "Received the order on schedule. Product is okay.",
        "Neither impressed nor disappointed.",
    ]
    texts = (templates * (n // len(templates) + 1))[:n]
    return pd.DataFrame({"text": texts, "label": [1] * n})


# ─── Train / Val / Test Split ─────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.80,
    val_size:   float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / validation / test.

    Args:
        df         : Full dataset with 'text' and 'label' columns
        train_size : Fraction for training (default 0.80)
        val_size   : Fraction for validation (default 0.10)
                     Remaining → test

    Returns:
        (df_train, df_val, df_test)

    Note:
        Stratification ensures each split has the same class ratio
        as the original dataset, preventing biased evaluation.
    """
    test_size = round(1.0 - train_size - val_size, 2)

    df_train, df_temp = train_test_split(
        df, test_size=(val_size + test_size),
        random_state=42, stratify=df["label"]
    )
    relative_test = round(test_size / (val_size + test_size), 4)
    df_val, df_test = train_test_split(
        df_temp, test_size=relative_test,
        random_state=42, stratify=df_temp["label"]
    )

    logger.info(f"Split: {len(df_train):,} train | {len(df_val):,} val | {len(df_test):,} test")
    logger.info(f"  Train distribution : {df_train['label'].value_counts().sort_index().to_dict()}")
    logger.info(f"  Val   distribution : {df_val['label'].value_counts().sort_index().to_dict()}")
    logger.info(f"  Test  distribution : {df_test['label'].value_counts().sort_index().to_dict()}")
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# ─── Class Weights ────────────────────────────────────────────────────────────

def compute_class_weights(labels: List[int]) -> List[float]:
    """
    Compute inverse-frequency class weights.

    Formula: weight[c] = total_samples / (n_classes × count[c])
    Passed to CrossEntropyLoss(weight=...) to handle class imbalance.
    """
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return weights.tolist()


# ─── Custom CSV / Retrain ─────────────────────────────────────────────────────

def load_custom_csv(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Load a user-uploaded labeled file for retraining.

    Accepts:
      - CSV  (.csv)
      - Excel (.xlsx, .xls)

    Label column can be:
      - Integer: 0, 1, 2
      - String:  "negative", "neutral", "positive" (case-insensitive)
    """
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {df.columns.tolist()}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Available: {df.columns.tolist()}")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)

    label_map = {
        "negative": 0, "neg": 0, "bad": 0,
        "neutral":  1, "neu": 1, "ok": 1, "mixed": 1,
        "positive": 2, "pos": 2, "good": 2,
        "0": 0, "1": 1, "2": 2,
    }
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().str.strip().map(label_map)

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    logger.info(f"Custom dataset loaded: {len(df):,} samples | {df['label'].value_counts().to_dict()}")
    return df


# ─── Dataset Stats Reporter ───────────────────────────────────────────────────

def print_dataset_report(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """Print a structured dataset transparency report."""
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print("\n" + "═"*60)
    print("DATASET REPORT")
    print("═"*60)
    print(f"  Total samples    : {len(df):,}")
    print(f"  Feature column   : text (cleaned str)")
    print(f"  Label column     : label (int 0/1/2)")
    print(f"\n  Label Distribution (Full):")
    for lbl, name in label_names.items():
        cnt = (df["label"] == lbl).sum()
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:10s} ({lbl}): {cnt:>7,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Splits (stratified):")
    print(f"    Train : {len(df_train):>7,}  ({len(df_train)/len(df)*100:.1f}%)")
    print(f"    Val   : {len(df_val):>7,}  ({len(df_val)/len(df)*100:.1f}%)")
    print(f"    Test  : {len(df_test):>7,}  ({len(df_test)/len(df)*100:.1f}%)")
    avg_words = df["text"].str.split().str.len().mean()
    max_words = df["text"].str.split().str.len().max()
    print(f"\n  Avg words/review : {avg_words:.1f}")
    print(f"  Max words/review : {max_words}")
    print("═"*60 + "\n")
