"""
Person 1 — Baseline & GPU Transformer Training Pipeline
=======================================================
Implements:
1. Fast TF-IDF + Logistic Regression baseline (to anchor neural model evaluation).
2. PyTorch / HuggingFace fine-tuning loop for BERT / Legal-BERT on GPU.
3. Export of validation and test logits (`outputs/val_logits.npy`, `outputs/test_logits.npy`)
   so Person 2 (Statistical Evaluation & CIs) and Person 3 (Bias Audit) can run their analyses
   without needing to re-train the model on GPU.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def fine_tune_transformer(
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-5,
    max_length: int = 256,
    seed: int = 42,
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    ).to(device)
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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_val_logits = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_logits, val_y, val_loss = evaluate_logits(model, val_loader, device)
        val_acc = float(np.mean(np.argmax(val_logits, axis=1) == val_y))
        print(f"Epoch {epoch}: val_loss={val_loss:.4f} | val_accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_logits = val_logits
            model.save_pretrained(OUTPUTS_DIR / "best_model")
            tokenizer.save_pretrained(OUTPUTS_DIR / "best_model")

    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUTS_DIR / "best_model").to(device)
    test_logits, _, test_loss = evaluate_logits(best_model, test_loader, device)

    # Save logits and ground-truth labels for Person 2 & Person 3
    np.save(OUTPUTS_DIR / "val_logits.npy", best_val_logits)
    np.save(OUTPUTS_DIR / "test_logits.npy", test_logits)
    np.save(OUTPUTS_DIR / "val_labels.npy", val_labels)
    np.save(OUTPUTS_DIR / "test_labels.npy", test_labels)
    with open(OUTPUTS_DIR / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    return {
        "model_name": model_name,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_loss": round(test_loss, 4),
    }
