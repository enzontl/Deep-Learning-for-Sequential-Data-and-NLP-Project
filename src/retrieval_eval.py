"""
Personne 3 - Evaluation chiffree de la recherche clause-a-clause
================================================================
Question posee : quand un avocat colle une clause, les 5 clauses renvoyees sont-elles de la
MEME CATEGORIE ? On mesure :
  - Precision@5 : part des 5 resultats qui ont la bonne categorie ;
  - MRR@5       : 1 / (place du premier bon resultat), 0 s'il n'y en a aucun dans les 5 ;
  - HitRate@5   : au moins un bon resultat dans les 5 (0 ou 1).
Chaque moyenne est donnee avec un intervalle de confiance a 95 % (bootstrap par percentiles,
2 000 tirages, seed 42 : memes reglages que src/evaluation.py de la personne 2). Le tirage se fait
sur les REQUETES (et non sur les 5 resultats, qui sont lies entre eux).

Regle d'honnetete : on regle et on compare sur la VALIDATION ; le TEST n'est utilise qu'une seule
fois, a la fin (option --final obligatoire).

Exemples (depuis la racine du projet) :
    python -m src.retrieval_eval --split validation --max-bank 2000 --n-queries 200   # essai rapide
    python -m src.retrieval_eval --split validation                                   # vraie mesure
    python -m src.retrieval_eval --split test --final                                 # UNE SEULE FOIS
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from src.retrieval import MODES, RETRIEVAL_DIR, ClauseBankRetriever

EVAL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "retrieval_eval"
SEED = 42
N_BOOT = 2000
CONFIDENCE = 0.95
TOP_K = 5
METRICS = ("precision", "mrr", "hit_rate")


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = N_BOOT, confidence: float = CONFIDENCE, seed: int = SEED
) -> Dict[str, float]:
    """Moyenne + intervalle de confiance bootstrap (percentiles) : on retire n_boot fois, avec remise,
    autant de requetes qu'il y en a, et on garde les valeurs entre le 2,5e et le 97,5e percentile."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {"estimate": float(values.mean()), "ci_lower": float(lo), "ci_upper": float(hi)}


def per_query_metrics(
    retrieved: np.ndarray, bank_labels: np.ndarray, query_labels: np.ndarray, k: int = TOP_K
) -> Dict[str, np.ndarray]:
    """Precision@k, MRR@k et HitRate@k de chaque requete. `retrieved` : positions dans la banque (-1 = vide)."""
    retrieved = retrieved[:, :k]
    valid = retrieved >= 0
    hits = (bank_labels[np.where(valid, retrieved, 0)] == query_labels[:, None]) & valid
    any_hit = hits.any(axis=1)
    return {
        "precision": hits.sum(axis=1) / k,
        "mrr": np.where(any_hit, 1.0 / (hits.argmax(axis=1) + 1), 0.0),
        "hit_rate": any_hit.astype(np.float64),
    }


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def evaluate(
    retriever: ClauseBankRetriever,
    bank_texts: Sequence[str],
    bank_labels: np.ndarray,
    query_texts: Sequence[str],
    query_labels: np.ndarray,
    k: int = TOP_K,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Evalue les trois modes sur les memes requetes, avec et sans les requetes dont le texte exact est dans la banque."""
    bank_labels = np.asarray(bank_labels)
    query_labels = np.asarray(query_labels)
    bank_set = {_normalize(t) for t in bank_texts}
    is_duplicate = np.array([_normalize(t) in bank_set for t in query_texts])
    subsets = {"all": np.ones(len(query_texts), dtype=bool), "without_exact_duplicates": ~is_duplicate}

    results: Dict[str, Any] = {
        "n_queries": len(query_texts),
        "n_exact_duplicates_in_bank": int(is_duplicate.sum()),
        "modes": {},
        "paired_differences": {},
    }
    per_query = {}
    for mode in MODES:
        retrieved = retriever.search_batch(query_texts, top_k=k, mode=mode)
        per_query[mode] = per_query_metrics(retrieved, bank_labels, query_labels, k)
        results["modes"][mode] = {
            name: {m: bootstrap_mean_ci(per_query[mode][m][mask], n_boot, seed=seed) for m in METRICS}
            for name, mask in subsets.items()
            if mask.sum() > 0
        }

    # Differences appariees (memes requetes) : significatif si l'intervalle exclut 0
    for a, b in (("hybrid", "dense"), ("bm25", "dense"), ("hybrid", "bm25")):
        results["paired_differences"][f"{a}_minus_{b}"] = {}
        for name, mask in subsets.items():
            if mask.sum() == 0:
                continue
            entry = {}
            for m in ("precision", "mrr"):
                ci = bootstrap_mean_ci((per_query[a][m] - per_query[b][m])[mask], n_boot, seed=seed)
                ci["significant"] = bool(ci["ci_lower"] > 0 or ci["ci_upper"] < 0)
                entry[m] = ci
            results["paired_differences"][f"{a}_minus_{b}"][name] = entry
    return results


def format_table(results: Dict[str, Any]) -> str:
    lines = []
    for subset in ("all", "without_exact_duplicates"):
        title = "TOUTES les requetes" if subset == "all" else "Requetes SANS copie exacte dans la banque"
        lines.append(f"\n--- {title} ---")
        lines.append(f"{'mode':<8} {'Precision@5':<26} {'MRR@5':<26} {'HitRate@5':<26}")
        for mode, per_subset in results["modes"].items():
            if subset not in per_subset:
                continue
            cells = []
            for m in METRICS:
                r = per_subset[subset][m]
                cells.append(f"{r['estimate']:.3f} [{r['ci_lower']:.3f}; {r['ci_upper']:.3f}]")
            lines.append(f"{mode:<8} " + " ".join(f"{c:<26}" for c in cells))
    lines.append("\n--- Differences appariees (toutes requetes) : * = significatif ---")
    for name, per_subset in results["paired_differences"].items():
        r = per_subset["all"]
        parts = [
            f"{m}: {r[m]['estimate']:+.3f} [{r[m]['ci_lower']:+.3f}; {r[m]['ci_upper']:+.3f}]"
            + ("*" if r[m]["significant"] else "")
            for m in ("precision", "mrr")
        ]
        lines.append(f"{name:<22} " + "   ".join(parts))
    return "\n".join(lines)


def save_results(results: Dict[str, Any], config: Dict[str, Any], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results}, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "subset", "metric", "estimate", "ci_lower", "ci_upper"])
        for mode, per_subset in results["modes"].items():
            for subset, per_metric in per_subset.items():
                for m, r in per_metric.items():
                    writer.writerow([mode, subset, m, r["estimate"], r["ci_lower"], r["ci_upper"]])


def main() -> None:
    from src.data import load_ledgar

    parser = argparse.ArgumentParser(description="Evaluation chiffree de la recherche clause-a-clause")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--n-queries", type=int, default=1000, help="nombre de requetes tirees au hasard (seed 42)")
    parser.add_argument("--max-bank", type=int, default=None, help="ESSAI RAPIDE seulement : banque reduite")
    parser.add_argument("--final", action="store_true", help="obligatoire pour --split test (a utiliser UNE SEULE FOIS)")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("Le jeu de TEST ne s'utilise qu'une seule fois, a la fin : ajoute --final si les reglages sont definitifs.")

    dataset, label_names = load_ledgar(max_train=args.max_bank)
    bank, split = dataset["train"], dataset[args.split]
    all_texts, all_labels = split["text"], np.array(split["label"])
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(all_texts), size=min(args.n_queries, len(all_texts)), replace=False))
    query_texts = [all_texts[i] for i in chosen]
    query_labels = all_labels[chosen]

    bank_texts, bank_labels = bank["text"], np.array(bank["label"])
    retriever = ClauseBankRetriever(bank_texts, bank_labels, label_names)
    results = evaluate(retriever, bank_texts, bank_labels, query_texts, query_labels)

    smoke = args.max_bank is not None
    config = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "n_queries": len(query_texts),
        "query_indices_in_split": chosen.tolist(),
        "bank_size": len(bank_texts),
        "bank_fingerprint": retriever.fingerprint,
        "embedding_model": retriever.model_name,
        "top_k": TOP_K,
        "rrf_k": retriever.rrf_k,
        "bm25": {"k1": retriever.bm25.k1, "b": retriever.bm25.b, "epsilon": retriever.bm25.epsilon},
        "bootstrap": {"n_boot": N_BOOT, "confidence": CONFIDENCE, "seed": SEED, "method": "percentile, resampling queries"},
        "versions": {p: version(p) for p in ("numpy", "scipy", "sentence-transformers")},
        "smoke_test_only": smoke,
    }
    stem = f"retrieval_eval_{args.split}" + ("_SMOKE" if smoke else "")
    # Les essais rapides vont dans le dossier ignore par Git : impossible de les envoyer par erreur
    out_dir = RETRIEVAL_DIR / "smoke" if smoke else EVAL_DIR
    save_results(results, config, out_dir, stem)

    print(f"\nBanque : {len(bank_texts)} clauses | requetes ({args.split}) : {len(query_texts)}"
          f" dont {results['n_exact_duplicates_in_bank']} avec une copie exacte dans la banque")
    print(format_table(results))
    if smoke:
        print("\n/!\\ ESSAI RAPIDE (banque reduite) : ces chiffres ne sont PAS des resultats.")
    print(f"\nFichiers ecrits dans {out_dir}/{stem}.json et .csv")


if __name__ == "__main__":
    main()
