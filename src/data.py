"""
Person 1 — Data Loading, Exploratory Statistics & Tokenizer Comparison
======================================================================
Loads the LexGLUE LEDGAR contract clause dataset (60k train, 10k val, 10k test, 100 classes),
computes class imbalance & length statistics, and compares subword tokenization between
BERT-base, Legal-BERT, and CamemBERT.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


def load_ledgar(
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DatasetDict, List[str]]:
    """
    Load the `coastalcph/lex_glue` (`ledgar`) dataset from Hugging Face.
    Returns (dataset_dict, label_names).
    """
    dataset = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names: List[str] = dataset["train"].features["label"].names

    if max_train is not None and max_train < len(dataset["train"]):
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train))
    if max_val is not None and max_val < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(max_val))
    if max_test is not None and max_test < len(dataset["test"]):
        dataset["test"] = dataset["test"].shuffle(seed=seed).select(range(max_test))

    return dataset, label_names


def get_class_distribution_table(dataset: DatasetDict, label_names: List[str]) -> pd.DataFrame:
    """
    Build a summary table of sample counts per clause category across train, validation, and test splits.
    Highlights rare classes (e.g., 'Books') where small test sample sizes require wide confidence intervals.
    """
    num_classes = len(label_names)
    train_counts = np.bincount(dataset["train"]["label"], minlength=num_classes)
    val_counts = np.bincount(dataset["validation"]["label"], minlength=num_classes)
    test_counts = np.bincount(dataset["test"]["label"], minlength=num_classes)

    rows = []
    for cid, name in enumerate(label_names):
        rows.append(
            {
                "class_id": cid,
                "class_name": name,
                "train_count": int(train_counts[cid]),
                "val_count": int(val_counts[cid]),
                "test_count": int(test_counts[cid]),
            }
        )
    return pd.DataFrame(rows).sort_values("train_count", ascending=False).reset_index(drop=True)


def compare_tokenizers(sample_texts: List[str]) -> Dict[str, Any]:
    """
    Compare subword fertility (tokens per word) and truncation rates (>256 / >512 tokens)
    across Legal-BERT, BERT-base, and CamemBERT on English contract clauses.
    """
    models = [
        "nlpaueb/legal-bert-base-uncased",
        "bert-base-uncased",
        "camembert-base",
    ]
    word_counts = np.array([max(1, len(t.split())) for t in sample_texts])
    results: Dict[str, Any] = {}

    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoded = tokenizer(sample_texts, truncation=False)
        token_lengths = np.array([len(ids) for ids in encoded["input_ids"]])

        results[model_name] = {
            "mean_tokens_per_word": round(float(np.mean(token_lengths / word_counts)), 3),
            "median_token_length": float(np.median(token_lengths)),
            "p95_token_length": float(np.percentile(token_lengths, 95)),
            "pct_truncated_at_256": round(float(100.0 * np.mean(token_lengths > 256)), 2),
            "pct_truncated_at_512": round(float(100.0 * np.mean(token_lengths > 512)), 2),
        }

    return results
