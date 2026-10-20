"""
Person 3 — Lawyer Constraint Queries Retrieval Benchmark
========================================================
Evaluates whether pure dense semantic search (`Sentence-BERT`) fails on exact numerical or
jurisdictional constraints (e.g., "30-day notice" vs. "60-day notice", "State of Delaware"
vs. "State of New York"), and how lexical matching (`BM25`) and Reciprocal Rank Fusion
(`Hybrid` + Category Filter) recover precision.

A clause in the 60,000-clause training bank is a gold hit for a query iff:
    clause_category == target_category   AND   regex_constraint.search(clause_text)

Run from the repository root:
    uv run python -m src.retrieval_lawyer_eval
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from src.retrieval import MODES, ClauseBankRetriever
from src.retrieval_eval import EVAL_DIR, N_BOOT, SEED, TOP_K, bootstrap_mean_ci, per_query_metrics

_DAYS = r"[\s-]+(?:\w+[\s-]+)?days?\b"


def _n(num: str, word: str) -> str:
    """Regle 'N unites' : accepte 30, thirty, thirty (30)..."""
    return rf"\b(?:{num}|{word})(?:\s*\(\s*{num}\s*\))?"


# (categorie visee, requete de l'avocat, regle de contrainte appliquee au texte de la clause)
QUERIES = [
    # --- Terminations : delai de preavis ---
    ("Terminations", "Show me termination clauses with a 30-day notice period", _n("30", "thirty") + _DAYS),
    ("Terminations", "Termination clauses requiring 60 days prior written notice", _n("60", "sixty") + _DAYS),
    ("Terminations", "Termination clauses requiring 90 days prior written notice", _n("90", "ninety") + _DAYS),
    ("Terminations", "Termination for breach if not cured within 10 days", _n("10", "ten") + _DAYS),
    # --- Payments : delai de paiement ---
    ("Payments", "Payment due within 30 days of receipt of invoice", _n("30", "thirty") + _DAYS),
    ("Payments", "Invoices must be paid within 45 days", _n("45", "forty[\\s-]five") + _DAYS),
    ("Payments", "Payment terms net 60 days from the invoice date", _n("60", "sixty") + _DAYS),
    ("Payments", "Amounts are payable within 15 days after the due date", _n("15", "fifteen") + _DAYS),
    # --- Terms : duree du contrat ---
    ("Terms", "This agreement has an initial term of three years", _n("3", "three") + r"[\s-]+years?\b"),
    ("Terms", "This agreement has an initial term of one year", _n("1", "one") + r"[\s-]+years?\b"),
    ("Terms", "This agreement has an initial term of five years", _n("5", "five") + r"[\s-]+years?\b"),
    ("Terms", "This agreement has an initial term of two years", _n("2", "two") + r"[\s-]+years?\b"),
    # --- Notices : delai de reputation de reception ---
    ("Notices", "Notice is deemed received three days after mailing", _n("3", "three") + _DAYS),
    ("Notices", "Notice is deemed received five days after mailing", _n("5", "five") + _DAYS),
    ("Notices", "Notice is deemed received ten days after mailing", _n("10", "ten") + _DAYS),
    ("Notices", "Notice is deemed received 48 hours after being sent", r"\b(?:48|forty[\s-]eight)(?:\s*\(\s*48\s*\))?[\s-]+hours?\b"),
    # --- Governing Laws : etat designe ---
    ("Governing Laws", "This agreement is governed by the laws of the State of New York", r"state\s+of\s+new\s+york"),
    ("Governing Laws", "This agreement is governed by the laws of the State of Delaware", r"state\s+of\s+delaware"),
    ("Governing Laws", "This agreement is governed by the laws of the State of California", r"state\s+of\s+california"),
    ("Governing Laws", "This agreement is governed by the laws of the State of Texas", r"state\s+of\s+texas"),
]

# Configurations evaluated: (display_name, retrieval_mode, use_category_filter)
CONFIGS = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("dense+filtre", "dense", True),
    ("bm25+filtre", "bm25", True),
    ("hybrid+filtre", "hybrid", True),
]


def gold_masks(bank_texts: List[str], bank_labels: np.ndarray, label_names: List[str]) -> np.ndarray:
    """Boolean mask (n_queries, n_bank) indicating whether each bank clause satisfies category + constraint."""
    lowered = [t.lower() for t in bank_texts]
    masks = np.zeros((len(QUERIES), len(bank_texts)), dtype=bool)
    for q, (category, _, rule) in enumerate(QUERIES):
        pattern = re.compile(rule)
        in_cat = bank_labels == label_names.index(category)
        masks[q] = in_cat & np.fromiter((bool(pattern.search(t)) for t in lowered), dtype=bool, count=len(lowered))
    return masks


def run(retriever: ClauseBankRetriever, label_names: List[str], k: int = TOP_K) -> Dict:
    bank_labels = retriever.labels
    gold = gold_masks(retriever.texts, bank_labels, label_names)
    texts = [q[1] for q in QUERIES]
    cats = [label_names.index(q[0]) for q in QUERIES]

    per_query = {}
    for name, mode, use_filter in CONFIGS:
        if use_filter:
            retrieved = np.vstack([retriever.search_batch([t], top_k=k, mode=mode, label_filter=[c]) for t, c in zip(texts, cats)])
        else:
            retrieved = retriever.search_batch(texts, top_k=k, mode=mode)
        valid = retrieved >= 0
        hits = gold[np.arange(len(QUERIES))[:, None], np.where(valid, retrieved, 0)] & valid
        cat_hits = (bank_labels[np.where(valid, retrieved, 0)] == np.array(cats)[:, None]) & valid
        any_hit = hits.any(axis=1)
        per_query[name] = {
            "precision": hits.sum(axis=1) / k,
            "mrr": np.where(any_hit, 1.0 / (hits.argmax(axis=1) + 1), 0.0),
            "hit_rate": any_hit.astype(float),
            "category_only_precision": cat_hits.sum(axis=1) / k,
        }

    out = {
        "n_queries": len(QUERIES),
        "gold_counts": {q[1]: int(g.sum()) for q, g in zip(QUERIES, gold)},
        "modes": {},
        "paired_differences": {},
        "per_query_precision": {n: v["precision"].tolist() for n, v in per_query.items()},
    }
    for name in per_query:
        out["modes"][name] = {m: bootstrap_mean_ci(v, N_BOOT, seed=SEED) for m, v in per_query[name].items()}
    paired_pairs = [
        ("hybrid", "dense"),
        ("bm25", "dense"),
        ("hybrid+filtre", "dense"),
        ("hybrid+filtre", "hybrid"),
        ("hybrid+filtre", "bm25"),
        ("dense+filtre", "dense"),
        ("bm25+filtre", "bm25"),
        ("hybrid+filtre", "bm25+filtre"),
    ]
    for a, b in paired_pairs:
        if a not in per_query or b not in per_query:
            continue
        out["paired_differences"][f"{a}_minus_{b}"] = {}
        for m in ("precision", "mrr"):
            ci = bootstrap_mean_ci(per_query[a][m] - per_query[b][m], N_BOOT, seed=SEED)
            ci["significant"] = bool(ci["ci_lower"] > 0 or ci["ci_upper"] < 0)
            out["paired_differences"][f"{a}_minus_{b}"][m] = ci
    return out


def format_report(res: Dict) -> str:
    lines = ["\n--- Matching gold clauses in the 60k bank per query (if < 5: max Precision@5 < 1.0) ---"]
    for q, c in res["gold_counts"].items():
        warn = "/!\\ " if c < 5 else ""
        lines.append(f"{c:>5}  {warn}{q}")
    lines.append(f"\n--- Results across {res['n_queries']} constraint queries (95% Bootstrap CI over queries) ---")
    lines.append(f"{'mode':<14} {'Precision@5':<24} {'MRR@5':<24} {'HitRate@5':<24} {'(cat. only)':<10}")
    for name, r in res["modes"].items():
        cells = [f"{r[m]['estimate']:.3f} [{r[m]['ci_lower']:.3f}; {r[m]['ci_upper']:.3f}]" for m in ("precision", "mrr", "hit_rate")]
        lines.append(f"{name:<14} " + " ".join(f"{c:<24}" for c in cells) + f" {r['category_only_precision']['estimate']:.3f}")
    lines.append("\n--- Paired bootstrap differences (* = 95% CI excludes 0) ---")
    for name, r in res["paired_differences"].items():
        parts = [f"{m}: {r[m]['estimate']:+.3f} [{r[m]['ci_lower']:+.3f}; {r[m]['ci_upper']:+.3f}]" + ("*" if r[m]["significant"] else "") for m in ("precision", "mrr")]
        lines.append(f"{name:<26} " + "   ".join(parts))
    return "\n".join(lines)


def main() -> None:
    from importlib.metadata import version

    from src.data import load_ledgar

    dataset, label_names = load_ledgar()
    bank = dataset["train"]
    retriever = ClauseBankRetriever(bank["text"], np.array(bank["label"]), label_names)
    res = run(retriever, label_names)

    config = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bank_size": len(bank["text"]),
        "bank_fingerprint": retriever.fingerprint,
        "embedding_model": retriever.model_name,
        "top_k": TOP_K,
        "queries": [{"category": c, "query": q, "rule": r} for c, q, r in QUERIES],
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "confidence": 0.95, "resampling": "queries"},
        "versions": {p: version(p) for p in ("numpy", "scipy", "sentence-transformers")},
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "retrieval_lawyer_queries.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": res}, f, indent=2, ensure_ascii=False)
    print(format_report(res))
    print(f"\nFichier ecrit : {EVAL_DIR / 'retrieval_lawyer_queries.json'}")


if __name__ == "__main__":
    main()
