"""
Person 3 — Bias Audit & Failure-Mode Analysis
=============================================
Tasks in this module:
1. Sample-Size / Long-Tail Bias:
   Compare model F1 and 95% CI width across Head classes (n_train >= 1000) vs. Tail classes (n_train < 150).
2. Clause Length & Truncation Bias   [IMPLEMENTE : run_length_truncation_audit]
   Evaluate accuracy (with 95% CIs) across clause word-length buckets (<80, 80-160, 160-260, >260 words)
   to quantify the impact of BERT's 256/512 subword token truncation limit.
3. Semantic Taxonomy Overlap Analysis:
   Analyze top confusion pairs (e.g., 'Governing Laws' vs 'Submission To Jurisdiction', 'Amendments' vs 'Waivers')
   caused by compound boilerplate clauses.
4. US SEC Training Data vs. European Legal-Tech Deployment Bias:
   - Identify US-specific statutory classes in LEDGAR ('Erisa', 'Waiver Of Jury Trials').
   - Run a Counterfactual Perturbation Test (swapping US states/courts/statutes/currencies like 'Delaware', '$', 'SEC'
     with European equivalents like 'France', 'EUR', 'ESMA', 'GDPR') and measure prediction flip rate & confidence drop.

Audit 1 (depuis la racine du projet) :
    python -m src.bias_audit --audit length
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.evaluation import wilson_confidence_interval

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
AUDIT_DIR = OUTPUTS_DIR / "bias_audit"

# (nom, borne basse incluse, borne haute exclue) en nombre de MOTS
LENGTH_BUCKETS = [
    ("<80", 0, 80),
    ("80-160", 80, 160),
    ("160-260", 160, 261),  # 260 mots inclus ici ; ">260" = 261 mots et plus
    (">260", 261, 10**9),
]


def run_length_truncation_audit(
    test_texts: List[str], test_labels: np.ndarray, test_logits: np.ndarray
) -> List[Dict[str, Any]]:
    """Exactitude (accuracy) et IC 95 % de Wilson par tranche de longueur de clause (en mots)."""
    test_labels = np.asarray(test_labels)
    predictions = np.asarray(test_logits).argmax(axis=1)
    if not (len(test_texts) == len(test_labels) == len(predictions)):
        raise ValueError("textes, etiquettes et logits n'ont pas la meme longueur")
    n_words = np.array([len(t.split()) for t in test_texts])
    correct = predictions == test_labels

    rows = []
    for name, low, high in LENGTH_BUCKETS:
        mask = (n_words >= low) & (n_words < high)
        n = int(mask.sum())
        k = int(correct[mask].sum())
        p, lo, hi = wilson_confidence_interval(k, n)
        rows.append({
            "bucket": name,
            "n": n,
            "share_of_test": n / len(n_words),
            "n_correct": k,
            "accuracy": p,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_width": hi - lo,
        })
    return rows


def format_length_table(results_by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = []
    for model, rows in results_by_model.items():
        lines.append(f"\n--- {model} : exactitude par longueur de clause (IC 95 % de Wilson) ---")
        lines.append(f"{'tranche (mots)':<16} {'n':>6} {'% du test':>10}   {'exactitude [IC 95 %]':<28}")
        for r in rows:
            lines.append(
                f"{r['bucket']:<16} {r['n']:>6} {100 * r['share_of_test']:>9.1f}%   "
                f"{r['accuracy']:.3f} [{r['ci_lower']:.3f}; {r['ci_upper']:.3f}]"
            )
    return "\n".join(lines)


def save_length_results(results_by_model: Dict[str, List[Dict[str, Any]]], out_dir: Path = AUDIT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "length_audit.json", "w", encoding="utf-8") as f:
        json.dump({"buckets": [b[0] for b in LENGTH_BUCKETS], "ci": "Wilson 95%", "results": results_by_model},
                  f, indent=2, ensure_ascii=False)
    with open(out_dir / "length_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "bucket", "n", "share_of_test", "accuracy", "ci_lower", "ci_upper"])
        for model, rows in results_by_model.items():
            for r in rows:
                writer.writerow([model, r["bucket"], r["n"], r["share_of_test"], r["accuracy"], r["ci_lower"], r["ci_upper"]])


def run_us_to_eu_counterfactual_audit(
    test_texts: List[str], test_labels: np.ndarray
) -> Dict[str, Any]:
    """TODO (Person 3): Perturb US legal entities/jurisdictions to EU equivalents and measure flip rate."""
    raise NotImplementedError("TODO (Person 3): Implement US->EU counterfactual bias audit.")


def _main_length() -> None:
    from src.data import load_ledgar

    dataset, _ = load_ledgar()  # test COMPLET, ordre d'origine (comme les logits de la personne 1)
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])

    saved_labels = np.load(OUTPUTS_DIR / "test_labels.npy")
    if len(saved_labels) != len(test_labels) or not np.array_equal(saved_labels, test_labels):
        raise SystemExit("ERREUR : l'ordre du test ne correspond pas a outputs/test_labels.npy -> on arrete.")

    results = {}
    for name, file in (("Legal-BERT", "legal_bert_test_logits.npy"), ("TF-IDF", "tfidf_test_logits.npy")):
        logits = np.load(OUTPUTS_DIR / file)
        results[name] = run_length_truncation_audit(test_texts, test_labels, logits)

    print(format_length_table(results))
    save_length_results(results)
    print(f"\nFichiers ecrits dans {AUDIT_DIR}/length_audit.json et .csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audits de biais (personne 3)")
    parser.add_argument("--audit", choices=("length",), required=True)
    args = parser.parse_args()
    if args.audit == "length":
        _main_length()