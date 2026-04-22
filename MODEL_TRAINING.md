# Model Training Documentation — Sentiment Analyser

## 1. Model Architecture

### 1.1 Why BERT?

| Criterion | Traditional ML (TF-IDF + SVM/RF) | BERT |
|-----------|----------------------------------|------|
| Context understanding | ❌ Bag-of-words — ignores word order | ✅ Bidirectional transformer captures full context |
| Short texts | ❌ Sparse features | ✅ Strong representations even for 5-word reviews |
| Transfer learning | ❌ Train from scratch on limited data | ✅ Pre-trained on 3.3B words — fine-tune in <3 epochs |
| Negation handling | ❌ "not good" ≈ "good" | ✅ Attention captures "not" modifying "good" |
| Domain adaptation | ❌ Requires full retraining | ✅ Fine-tune 2–5% of parameters |
| Ambiguity resolution | ❌ Fixed embeddings | ✅ Contextual — "bank" (financial vs river) differs |

**Decision:** BERT-base-uncased was chosen because:
1. It is the most widely validated transformer for NLP classification tasks
2. Uncased normalizes capitalization (common in reviews)
3. Small enough to fine-tune on consumer hardware (CPU or single GPU)

### 1.2 Full Architecture Diagram

```
Input Text (raw review)
        │
        ▼
┌─────────────────────────────────────────┐
│  Text Preprocessing (utils/text_cleaner) │
│  • HTML removal  • URL removal           │
│  • Unicode norm  • Repeated chars fix    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  BertTokenizer  (bert-base-uncased)      │
│  • WordPiece tokenization                │
│  • Add [CLS] and [SEP] tokens            │
│  • Pad / truncate to max_length=256      │
│  Output: input_ids, attention_mask,      │
│          token_type_ids                  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  BertModel  (bert-base-uncased)          │
│  • 12 transformer encoder layers        │
│  • 12 attention heads per layer         │
│  • Hidden size: 768                     │
│  • Parameters: ~110M                    │
│  Output: pooler_output [CLS] → (B, 768) │
└─────────────────────────────────────────┘
        │  (B, 768)
        ▼
┌─────────────────────────────────────────┐
│  Classification Head (trainable)         │
│  Dropout(p=0.3)                          │
│  Linear(768 → 256)                       │
│  GELU activation                         │
│  Dropout(p=0.15)                         │
│  Linear(256 → 3)                         │
│  Output: logits (B, 3)                   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Softmax → Probabilities (B, 3)         │
│  [P(Negative), P(Neutral), P(Positive)] │
└─────────────────────────────────────────┘
        │
        ▼
  argmax → Predicted class
  max    → Confidence score
```

### 1.3 Why VADER Hybrid Layer?

BERT alone has weaknesses:
- **Domain shift**: A BERT model trained on Amazon reviews may underperform on Twitter slang or formal reports
- **Short texts**: < 5 words may not give enough context for BERT's attention
- **Borderline confidence**: When BERT confidence < 60%, its prediction is unreliable

VADER (Valence Aware Dictionary and sEntiment Reasoner):
- Rule-based lexicon — always provides a signal regardless of domain
- Handles capitalization ("GREAT" > "great"), punctuation ("great!!!")
- Compound score range: -1 (most negative) to +1 (most positive)

**Hybrid Blending Rule:**
```
if BERT_confidence >= 0.80:
    → Trust BERT (sufficient certainty)
elif |VADER_compound| >= 0.5 AND BERT_confidence < 0.60:
    → VADER override (strong lexical signal + uncertain BERT)
else:
    → Soft blend: weighted average of BERT probs and VADER probs
       blended[c] = BERT_conf × BERT_prob[c] + (1-BERT_conf) × VADER_prob[c]
```

### 1.4 Why Sarcasm Detection?

Sarcasm inverts sentiment polarity. Example:
- Text: *"Oh great, another product that breaks in a week. Just what I needed."*
- BERT without sarcasm: → **Positive** (mislead by "great", "needed")
- With sarcasm detection: → **Negative** (sarcasm flag flips confidence)

Sarcasm model: DistilBERT binary classifier (0=not sarcastic, 1=sarcastic)
- 40% fewer parameters than BERT — faster inference
- Rule-based fallback if sarcasm model not trained

---

## 2. Training Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Base model | bert-base-uncased | Widely validated, handles lowercase reviews |
| Number of epochs | 3 | BERT fine-tuning converges fast; >5 risks overfitting |
| Batch size | 16 | Fits in 8GB GPU VRAM; 32 for larger GPUs |
| Learning rate | 2e-5 | Standard for BERT fine-tuning (Devlin et al. 2019) |
| LR scheduler | Linear warmup + decay | Prevents large gradient updates early in training |
| Warmup steps | 500 | ~3% of total steps |
| Weight decay | 0.01 | L2 regularization |
| Max token length | 256 | Covers 95%+ of reviews; 512 doubles memory |
| Dropout (head) | 0.3 | Prevents overfitting in classification head |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Loss function | CrossEntropyLoss | With class weights for imbalance |
| Optimizer | AdamW | Adam with corrected weight decay |

---

## 3. Training Procedure (Step-by-Step)

```
Step 1: Load Dataset
  └─ amazon_polarity (HuggingFace) → sample N per class
  └─ sst2 → neutral supplement
  └─ Clean text → cache to parquet

Step 2: Stratified Split
  └─ train_test_split(stratify=label, seed=42)
  └─ 80% train / 10% val / 10% test

Step 3: Compute Class Weights
  └─ compute_class_weight("balanced") → weight tensor
  └─ Passed to CrossEntropyLoss

Step 4: Initialize Model
  └─ BertTokenizer.from_pretrained("bert-base-uncased")
  └─ BertModel.from_pretrained("bert-base-uncased")
  └─ Custom classification head added
  └─ Move to CUDA (if available) else CPU

Step 5: Training Loop (per epoch)
  ├─ Forward pass: input_ids, attention_mask → logits
  ├─ Compute weighted CrossEntropyLoss
  ├─ loss.backward()
  ├─ Gradient clipping (max_norm=1.0)
  ├─ optimizer.step() + scheduler.step()
  └─ Log train_loss

Step 6: Validation (per epoch)
  ├─ model.eval() + torch.no_grad()
  ├─ Compute: val_loss, accuracy, precision, recall, F1
  ├─ Per-class: precision, recall, F1 for each class
  ├─ Save metrics to training_log.csv
  └─ Save checkpoint if val_loss improved

Step 7: Test Evaluation (once, at end)
  ├─ Load best checkpoint
  ├─ Predict on held-out test set
  ├─ Compute all metrics + confusion matrix
  └─ Save to reports/test_metrics.json

Step 8: Generate Visualizations
  ├─ Loss curve (train + val)
  ├─ Accuracy + F1 curve
  ├─ Per-class metrics bar chart
  ├─ Confusion matrix (normalized)
  ├─ Class distribution charts
  ├─ Dataset split bar
  └─ All-in-one training dashboard
```

---

## 4. Model Checkpoint — Save & Load

### Saved Files

```
data/checkpoints/bert_sentiment/
├── config.json              ← BERT architecture config
├── pytorch_model.bin        ← BERT pre-trained weights
├── vocab.txt                ← Tokenizer vocabulary (30,522 tokens)
├── tokenizer_config.json    ← Tokenizer settings
├── special_tokens_map.json  ← [CLS], [SEP], [PAD] etc.
├── classifier_head.pt       ← Custom head weights (torch state_dict)
└── model_info.json          ← Metadata: num_labels, max_length, etc.
```

### How to Save

```python
trainer.save("data/checkpoints/bert_sentiment")
# Internally calls:
#   self.model.bert.save_pretrained(path)    → HuggingFace format
#   self.tokenizer.save_pretrained(path)     → tokenizer files
#   torch.save(self.model.state_dict(), ...) → full state dict
```

### How to Load

```python
trainer = BertSentimentTrainer(cfg)
trainer.load("data/checkpoints/bert_sentiment")
# Internally:
#   BertTokenizer.from_pretrained(path)
#   BertModel.from_pretrained(path)          → loads BERT weights
#   torch.load("classifier_head.pt")         → loads head weights
#   model.eval()                             → inference mode
```

### Why Not Just Save the Full Model?

HuggingFace `save_pretrained` is the standard format — allows the BERT backbone to be reloaded by other tools. The custom head is saved separately as a PyTorch state dict for flexibility.

---

## 5. Per-Epoch Metrics Logged

Every epoch records:

| Metric | Description | Averaging |
|--------|-------------|-----------|
| `train_loss` | CrossEntropyLoss on training batches | Mean over all batches |
| `val_loss` | CrossEntropyLoss on validation set | Mean over all batches |
| `val_accuracy` | (TP+TN) / Total | Micro |
| `val_precision` | Mean of per-class precision | Macro |
| `val_recall` | Mean of per-class recall | Macro |
| `val_f1` | Mean of per-class F1 | Macro |
| `val_f1_neg` | F1 for Negative class | Per-class |
| `val_f1_neu` | F1 for Neutral class | Per-class |
| `val_f1_pos` | F1 for Positive class | Per-class |

**Saved to:** `reports/training_log.csv` (updated each epoch)

---

## 6. Computing Metrics

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

acc = accuracy_score(y_true, y_pred)

# Macro averaging — equal weight per class
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

# Per-class scores
cls_prec, cls_rec, cls_f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=[0,1,2], zero_division=0
)
```

**Why Macro F1?** It gives equal weight to each class regardless of support. This penalizes models that ignore minority classes (Neutral), making it the right choice for imbalanced multi-class problems.

---

## 7. GPU vs CPU

| Mode | Detection | Typical Speed |
|------|-----------|--------------|
| CUDA GPU | `torch.cuda.is_available()` returns True | ~4 min/epoch (batch=16, 48k samples) |
| CPU only | Fallback | ~90 min/epoch |

The device is printed at startup:
```
Device: cuda  (or cpu)
```

**GPU recommendation:** NVIDIA GPU with ≥6GB VRAM. Use `--batch 32` for GPUs with ≥12GB.
