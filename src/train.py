"""
Person 1 — Baseline & GPU Transformer Training Pipeline
=======================================================
Implements:
1. Fast TF-IDF + Logistic Regression baseline (to anchor neural model evaluation).
2. PyTorch / HuggingFace fine-tuning loop for BERT / Legal-BERT on GPU.
3. Export of validation and test logits (`outputs/legal_bert_{val,test}_logits.npy`,
   `outputs/tfidf_{val,test}_logits.npy`) so Person 2 (Statistical Evaluation & CIs)
   and Person 3 (Bias Audit) can run their analyses without needing to re-train on GPU.

Run from the repository root:
    uv run python -m src.train
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

from src.data import load_ledgar

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


class ContractClauseDataset(Dataset):
    """PyTorch Dataset wrapping contract clauses and integer labels."""

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: Any, max_length: int = 256):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item["labels"] = int(self.labels[idx])
        return item


def train_tfidf_baseline(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    test_texts: List[str],
    max_features: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a fast TF-IDF + L2 Logistic Regression baseline and return (val_logits, test_logits).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    x_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(C=2.0, max_iter=300, random_state=42)
    clf.fit(x_train, train_labels)

    val_logits = clf.decision_function(vectorizer.transform(val_texts)).astype(np.float32)
    test_logits = clf.decision_function(vectorizer.transform(test_texts)).astype(np.float32)
    return val_logits, test_logits


@torch.no_grad()
def evaluate_logits(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run inference over a DataLoader and return (logits, labels, mean_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, labels)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_examples += bs
        all_logits.append(outputs.logits.detach().cpu().numpy().astype(np.float32))
        all_labels.append(labels.detach().cpu().numpy().astype(np.int64))

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
        total_loss / max(1, total_examples),
    )


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer (Hu et al., 2021; Course Session 6, Slides 18-20):
        h = W x + (alpha / r) * B (A x)
    where W in R^{d_out x d_in} is frozen, A in R^{r x d_in} is initialized ~ N(0, 0.02^2),
    and B in R^{d_out x r} is initialized to 0 so that at step 0, B A = 0 and the model
    outputs the exact pretrained representations.
    """

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(r, self.in_features, device=base_layer.weight.device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=base_layer.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    @torch.no_grad()
    def merge_weights(self) -> nn.Linear:
        """Merge W' = W + (alpha / r) B A for zero-overhead inference (Session 6, Slide 19)."""
        self.base_layer.weight.data.add_((self.lora_B @ self.lora_A) * self.scaling)
        return self.base_layer


def apply_lora_to_bert(
    model: nn.Module, r: int = 8, alpha: float = 16.0, target_names: Tuple[str, ...] = ("query", "value")
) -> nn.Module:
    """
    Freeze the pretrained BERT body, inject LoRALinear into every attention (query, value)
    projection (Course Session 6, Slide 20), and keep the classification head trainable.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    for module in model.modules():
        for attr in target_names:
            if hasattr(module, attr) and isinstance(getattr(module, attr), nn.Linear):
                setattr(module, attr, LoRALinear(getattr(module, attr), r=r, alpha=alpha))
    return model


def count_parameters_and_memory(model: nn.Module) -> Dict[str, Any]:
    """
    Count total and trainable parameters and estimate optimizer + weight + gradient memory
    using Course Session 6 (Slides 16-17):
      - Trained parameter (fp32): 16 bytes (4B weight + 4B grad + 4B Adam m + 4B Adam v)
      - Frozen parameter (fp16):   2 bytes (weight only, no grad, no Adam states)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    mem_bytes = trainable * 16 + frozen * 2
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
        "trainable_pct": round(100.0 * trainable / max(1, total), 4),
        "estimated_param_grad_opt_mb": round(mem_bytes / 1e6, 2),
    }


def fine_tune_transformer(
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 3e-5,
    max_length: int = 256,
    seed: int = 42,
    use_class_weights: bool = False,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    run_tfidf: bool = True,
) -> Dict[str, Any]:
    """
    Fine-tune BERT / Legal-BERT on LEDGAR using GPU (CUDA/MPS) and save validation/test logits
    to `outputs/` for the rest of the team.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, label_names = load_ledgar(seed=seed)
    num_classes = len(label_names)

    train_texts = list(dataset["train"]["text"])
    train_labels = np.array(dataset["train"]["label"], dtype=np.int64)
    val_texts = list(dataset["validation"]["text"])
    val_labels = np.array(dataset["validation"]["label"], dtype=np.int64)
    test_texts = list(dataset["test"]["text"])
    test_labels = np.array(dataset["test"]["label"], dtype=np.int64)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUTS_DIR / "val_labels.npy", val_labels)
    np.save(OUTPUTS_DIR / "test_labels.npy", test_labels)
    with open(OUTPUTS_DIR / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    if run_tfidf:
        tfidf_val_logits, tfidf_test_logits = train_tfidf_baseline(
            train_texts, train_labels, val_texts, test_texts
        )
        np.save(OUTPUTS_DIR / "tfidf_val_logits.npy", tfidf_val_logits)
        np.save(OUTPUTS_DIR / "tfidf_test_logits.npy", tfidf_test_logits)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    ).to(device)
    if use_lora:
        apply_lora_to_bert(model, r=lora_r, alpha=lora_alpha)
    param_stats = count_parameters_and_memory(model)
    print(f"Model parameter & memory accounting: {param_stats}")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        ContractClauseDataset(train_texts, train_labels, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        ContractClauseDataset(val_texts, val_labels, tokenizer, max_length),
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        ContractClauseDataset(test_texts, test_labels, tokenizer, max_length),
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    if use_class_weights:
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    best_val_logits: Optional[np.ndarray] = None
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)
        val_logits, val_y, val_loss = evaluate_logits(model, val_loader, device)
        val_acc = float(np.mean(np.argmax(val_logits, axis=1) == val_y))
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_accuracy={val_acc:.4f}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_logits = val_logits
            model.save_pretrained(OUTPUTS_DIR / "best_model")
            tokenizer.save_pretrained(OUTPUTS_DIR / "best_model")

    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUTS_DIR / "best_model").to(device)
    test_logits, _, test_loss = evaluate_logits(best_model, test_loader, device)
    test_acc = float(np.mean(np.argmax(test_logits, axis=1) == test_labels))

    # Save logits using the canonical filenames expected by evaluation.py and bias_audit.py
    np.save(OUTPUTS_DIR / "legal_bert_val_logits.npy", best_val_logits)
    np.save(OUTPUTS_DIR / "legal_bert_test_logits.npy", test_logits)

    summary = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "use_class_weights": use_class_weights,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "epochs_log": history,
    }
    with open(OUTPUTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline and Legal-BERT classifier on LEDGAR")
    parser.add_argument("--model-name", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-sqrt class weights in CrossEntropyLoss")
    parser.add_argument("--lora", action="store_true", help="Use Low-Rank Adaptation (LoRA) on attention query/value projections")
    parser.add_argument("--lora-r", type=int, default=8)
    args = parser.parse_args()
    fine_tune_transformer(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        use_class_weights=args.class_weights,
        use_lora=args.lora,
        lora_r=args.lora_r,
    )
