# Dataset Information — Sentiment Analyser (Project)

## Overview

This project uses three publicly available, open-source datasets. No proprietary or paid data is used.

---

## 1. Primary Dataset — Amazon Product Reviews (`amazon_polarity`)

| Field | Detail |
|-------|--------|
| **Name** | Amazon Polarity Dataset |
| **Source** | HuggingFace Datasets (`amazon_polarity`) |
| **Original paper** | McAuley & Leskovec, *Hidden factors and hidden topics*, RecSys 2013 |
| **URL** | https://huggingface.co/datasets/amazon_polarity |
| **Total size** | 3,600,000 training + 400,000 test reviews |
| **Domain** | Amazon product reviews across 18+ categories (electronics, books, clothing, etc.) |
| **Language** | English |
| **License** | Open / Research Use |

### Columns Used

| Column | Type | Description |
|--------|------|-------------|
| `content` | string | Full text of the customer review (renamed → `text`) |
| `label` | int (0/1) | Original binary label: 0=negative, 1=positive |
| `title` | string | Review headline (not used — too short) |

### Label Mapping (3-class conversion)

| Original Label | Star Rating | Maps To | Our Label | Index |
|---------------|-------------|---------|-----------|-------|
| 0 (negative) | 1–2 stars | Negative | Negative | 0 |
| — | 3 stars | Neutral | Neutral | 1 |
| 1 (positive) | 4–5 stars | Positive | Positive | 2 |

> **Note:** Amazon Polarity does not contain 3-star (neutral) reviews. Neutral class is supplemented from SST-2 (see below).

### Sample Reviews

```
Negative: "This product broke after 3 days. Complete waste of money. Do not buy."
Positive: "Absolutely love this! Fast shipping, great quality, exactly as described."
```

### Samples Used in This Project

| Subset | Count |
|--------|-------|
| Negative (amazon_polarity label=0) | max_samples // 3 |
| Positive (amazon_polarity label=1) | max_samples // 3 |
| Neutral (SST-2, see below) | max_samples // 3 |
| **Total** | max_samples (default: 60,000) |

---

## 2. Neutral Supplement — Stanford Sentiment Treebank (`sst2`)

| Field | Detail |
|-------|--------|
| **Name** | SST-2 (Stanford Sentiment Treebank, binary) |
| **Source** | HuggingFace Datasets (`sst2`) |
| **Original paper** | Socher et al., *Recursive Deep Models for Semantic Compositionality*, EMNLP 2013 |
| **URL** | https://huggingface.co/datasets/sst2 |
| **Total size** | 67,349 training sentences |
| **Domain** | Movie reviews (Rotten Tomatoes) |
| **Language** | English |

### Why SST-2 for Neutral?

Amazon reviews do not include 3-star reviews. SST-2 contains short, factual movie-review sentences that serve as **neutral sentiment proxies**. All SST-2 samples are relabeled as class 1 (Neutral) in our system.

### Columns Used

| Column | Description |
|--------|-------------|
| `sentence` | Review sentence text (renamed → `text`) |
| `label` | Original binary (ignored — we override with 1=Neutral) |

---

## 3. Fallback Dataset — Yelp Reviews (`yelp_review_full`)

| Field | Detail |
|-------|--------|
| **Name** | Yelp Review Full |
| **Source** | HuggingFace Datasets (`yelp_review_full`) |
| **URL** | https://huggingface.co/datasets/yelp_review_full |
| **Total size** | 650,000 training + 50,000 test reviews |
| **Used when** | `amazon_polarity` download fails |

### Star → Label Mapping

| Stars (0-indexed) | Maps To | Label Index |
|-------------------|---------|-------------|
| 0 (1 star) | Negative | 0 |
| 1 (2 stars) | Negative | 0 |
| 2 (3 stars) | Neutral | 1 |
| 3 (4 stars) | Positive | 2 |
| 4 (5 stars) | Positive | 2 |

---

## 4. Sarcasm Detection Dataset

| Field | Detail |
|-------|--------|
| **Name** | News Headlines Dataset for Sarcasm Detection |
| **Original paper** | Misra & Arora (2019), *Sarcasm Detection using Hybrid Neural Network* |
| **Domain** | News headlines (TheOnion vs HuffPost) |
| **Note** | Used to fine-tune the DistilBERT sarcasm classifier module |

If the sarcasm model is not trained, the system falls back to **rule-based sarcasm detection** using:
- Lexical signals ("yeah right", "oh great", "just what I needed")
- Punctuation patterns (!!!, ???)
- CAPS LOCK ratio
- Sentiment contradiction (positive + negative words co-occurring)

---

## 5. Text Preprocessing Pipeline

Every review passes through these cleaning steps (in order):

| Step | Function | Effect |
|------|----------|--------|
| 1 | `html.unescape` | `&amp;` → `&` |
| 2 | Unicode normalize | NFKD normalization |
| 3 | HTML tag removal | `<br/>`, `<p>` etc removed |
| 4 | URL removal | `http://...` → `` |
| 5 | @mention removal | `@user` → `` |
| 6 | Hashtag expansion | `#great` → `great` |
| 7 | Repeated chars | `sooooo` → `soo` (keeps 2 for emphasis) |
| 8 | Whitespace norm | Multiple spaces → single space |
| 9 | Min length filter | Drop reviews < 10 characters |

---

## 6. Train / Validation / Test Split

| Split | Ratio | Purpose |
|-------|-------|---------|
| **Training** | 80% | Model weight updates |
| **Validation** | 10% | Hyperparameter tuning, early stopping, epoch selection |
| **Test** | 10% | **Final unbiased evaluation — seen only once** |

- Method: `sklearn.model_selection.train_test_split` with `stratify=label`
- Random seed: 42 (reproducible)
- **Stratified**: each split has the same class proportion as the full dataset

### Example Split Sizes (at 60,000 total samples)

| Split | Negative | Neutral | Positive | Total |
|-------|----------|---------|----------|-------|
| Train | ~16,000  | ~16,000 | ~16,000  | 48,000 |
| Val   | ~2,000   | ~2,000  | ~2,000   | 6,000  |
| Test  | ~2,000   | ~2,000  | ~2,000   | 6,000  |

---

## 7. Class Imbalance Handling

**Method:** Inverse-frequency class weighting via `sklearn.utils.class_weight.compute_class_weight("balanced")`

**Formula:**  
```
weight[c] = total_samples / (n_classes × count[c])
```

These weights are passed to `torch.nn.CrossEntropyLoss(weight=class_weights_tensor)` so the model penalizes errors on minority classes more heavily.

---

## 8. Reproducibility

| Item | Value |
|------|-------|
| Random seed (splits) | 42 |
| Random seed (sampling) | 42 |
| Dataset cached | `data/processed/amazon_reviews_{N}.parquet` |
| Cache allows | Exact same dataset on repeated runs |
