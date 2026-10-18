"""
Person 3 — Bias Audit & Failure-Mode Analysis
=============================================
Tasks in this module:
1. Sample-Size / Long-Tail Bias:
   Compare model F1 and 95% CI width across Head classes (n_train >= 1000) vs. Tail classes (n_train < 150).
2. Clause Length & Truncation Bias   [IMPLEMENTE : run_length_truncation_audit]
   Evaluate accuracy (with 95% CIs) across clause word-length buckets (<80, 80-160, 160-260, >260 words)
   to quantify the impact of BERT's 256/512 subword token truncation limit.
3. Semantic Taxonomy Overlap Analysis   [IMPLEMENTE : run_taxonomy_overlap_audit]:
   Analyze top confusion pairs (e.g., 'Governing Laws' vs 'Submission To Jurisdiction', 'Amendments' vs 'Waivers')
   caused by compound boilerplate clauses.
4. US SEC Training Data vs. European Legal-Tech Deployment Bias:
   - Identify US-specific statutory classes in LEDGAR ('Erisa', 'Waiver Of Jury Trials').
   - Run a Counterfactual Perturbation Test (swapping US states/courts/statutes/currencies like 'Delaware', '$', 'SEC'
     with European equivalents like 'France', 'EUR', 'ESMA', 'GDPR') and measure prediction flip rate & confidence drop.

Depuis la racine du projet :
    python -m src.bias_audit --audit length
    python -m src.bias_audit --audit taxonomy
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


# ----------------------------------------------------------------------------------------------
# Audit 2 : chevauchement des categories (quelles paires le modele confond-il ?)
# ----------------------------------------------------------------------------------------------

# Paires citees par le README. Mots-cles FIXES AVANT de regarder les resultats : "clause composee" =
# le texte contient a la fois un mot de la categorie A et un mot de la categorie B.
README_PAIRS = [
    ("Governing Laws", "Submission To Jurisdiction", "govern", "jurisdiction"),
    ("Amendments", "Waivers", "amend", "waive"),
]


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """cm[i, j] = nombre de clauses de vraie categorie i predites j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def run_taxonomy_overlap_audit(
    test_texts: List[str], test_labels: np.ndarray, test_logits: np.ndarray, label_names: List[str], top_n: int = 10
) -> Dict[str, Any]:
    """Paires de categories les plus confondues + test de l'hypothese 'clauses composees' du README."""
    y_true = np.asarray(test_labels)
    y_pred = np.asarray(test_logits).argmax(axis=1)
    n_classes = len(label_names)
    cm = _confusion_matrix(y_true, y_pred, n_classes)
    n_errors = int((y_true != y_pred).sum())

    # paire non ordonnee {A, B} : erreurs A->B + erreurs B->A
    pairs = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            a_to_b, b_to_a = int(cm[i, j]), int(cm[j, i])
            if a_to_b + b_to_a > 0:
                pairs.append({
                    "class_a": label_names[i], "class_b": label_names[j],
                    "a_as_b": a_to_b, "b_as_a": b_to_a, "total": a_to_b + b_to_a,
                    "share_of_all_errors": (a_to_b + b_to_a) / max(n_errors, 1),
                })
    pairs.sort(key=lambda r: -r["total"])
    for rank, r in enumerate(pairs, start=1):
        r["rank"] = rank

    top = pairs[:top_n]
    top_share = sum(r["total"] for r in top) / max(n_errors, 1)

    # exemples d'erreurs pour les 3 premieres paires
    examples = []
    for r in pairs[:3]:
        a, b = label_names.index(r["class_a"]), label_names.index(r["class_b"])
        idx = np.where(((y_true == a) & (y_pred == b)) | ((y_true == b) & (y_pred == a)))[0][:3]
        for k in idx:
            examples.append({
                "pair": f"{r['class_a']} / {r['class_b']}", "true": label_names[y_true[k]],
                "predicted": label_names[y_pred[k]], "text_start": test_texts[k][:220],
            })

    # hypothese "clauses composees" sur les paires du README
    lowered = [t.lower() for t in test_texts]
    compound = []
    for name_a, name_b, kw_a, kw_b in README_PAIRS:
        if name_a not in label_names or name_b not in label_names:
            continue
        a, b = label_names.index(name_a), label_names.index(name_b)
        rank = next((r["rank"] for r in pairs if {r["class_a"], r["class_b"]} == {name_a, name_b}), None)
        total = next((r["total"] for r in pairs if {r["class_a"], r["class_b"]} == {name_a, name_b}), 0)
        has_both = np.array([(kw_a in t) and (kw_b in t) for t in lowered])
        confused = ((y_true == a) & (y_pred == b)) | ((y_true == b) & (y_pred == a))
        ok = ((y_true == a) & (y_pred == a)) | ((y_true == b) & (y_pred == b))
        row = {"pair": f"{name_a} / {name_b}", "keywords": [kw_a, kw_b], "rank_among_pairs": rank, "n_confused": int(confused.sum())}
        for label, mask in (("confused", confused), ("correct", ok)):
            n = int(mask.sum())
            k = int(has_both[mask].sum())
            p, lo, hi = wilson_confidence_interval(k, n)
            row[f"share_with_both_keywords_{label}"] = {"n": n, "k": k, "p": p, "ci_lower": lo, "ci_upper": hi}
        compound.append(row)

    return {
        "n_test": int(len(y_true)), "n_errors": n_errors, "top_pairs": top,
        "top_pairs_share_of_errors": top_share, "examples": examples, "readme_pairs": compound,
    }


def format_taxonomy_report(model: str, res: Dict[str, Any]) -> str:
    lines = [f"\n=== {model} : {res['n_errors']} erreurs sur {res['n_test']} clauses de test ==="]
    lines.append(f"Les {len(res['top_pairs'])} paires les plus confondues = {100 * res['top_pairs_share_of_errors']:.1f} % des erreurs")
    lines.append(f"\n{'rang':<5} {'paire':<58} {'A->B':>5} {'B->A':>5} {'total':>6}")
    for r in res["top_pairs"]:
        lines.append(f"{r['rank']:<5} {(r['class_a'] + ' / ' + r['class_b']):<58} {r['a_as_b']:>5} {r['b_as_a']:>5} {r['total']:>6}")
    lines.append("\n--- Paires citees par le README ---")
    for c in res["readme_pairs"]:
        lines.append(f"{c['pair']}: rang {c['rank_among_pairs']}, {c['n_confused']} confusions")
        for label in ("confused", "correct"):
            d = c[f"share_with_both_keywords_{label}"]
            txt = "clauses CONFONDUES" if label == "confused" else "clauses BIEN classees"
            lines.append(f"   contiennent les 2 mots-cles {c['keywords']} parmi les {txt} : "
                         f"{d['k']}/{d['n']} = {d['p']:.3f} [{d['ci_lower']:.3f}; {d['ci_upper']:.3f}]")
    lines.append("\n--- Exemples d'erreurs (3 premieres paires) ---")
    for e in res["examples"]:
        lines.append(f"[{e['pair']}] vrai={e['true']} | predit={e['predicted']}\n    {e['text_start']}")
    return "\n".join(lines)


def _main_taxonomy() -> None:
    from src.data import load_ledgar

    dataset, _ = load_ledgar()
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])
    saved_labels = np.load(OUTPUTS_DIR / "test_labels.npy")
    if len(saved_labels) != len(test_labels) or not np.array_equal(saved_labels, test_labels):
        raise SystemExit("ERREUR : l'ordre du test ne correspond pas a outputs/test_labels.npy -> on arrete.")
    with open(OUTPUTS_DIR / "label_names.json", encoding="utf-8") as f:
        label_names = json.load(f)

    results = {}
    for name, file in (("Legal-BERT", "legal_bert_test_logits.npy"), ("TF-IDF", "tfidf_test_logits.npy")):
        logits = np.load(OUTPUTS_DIR / file)
        results[name] = run_taxonomy_overlap_audit(test_texts, test_labels, logits, label_names)
        print(format_taxonomy_report(name, results[name]))

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "taxonomy_audit.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(AUDIT_DIR / "taxonomy_top_pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "rank", "class_a", "class_b", "a_as_b", "b_as_a", "total", "share_of_all_errors"])
        for model, res in results.items():
            for r in res["top_pairs"]:
                writer.writerow([model, r["rank"], r["class_a"], r["class_b"], r["a_as_b"], r["b_as_a"], r["total"], r["share_of_all_errors"]])
    print(f"\nFichiers ecrits dans {AUDIT_DIR}/taxonomy_audit.json et taxonomy_top_pairs.csv")


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
    parser.add_argument("--audit", choices=("length", "taxonomy"), required=True)
    args = parser.parse_args()
    if args.audit == "length":
        _main_length()
    elif args.audit == "taxonomy":
        _main_taxonomy()