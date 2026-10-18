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
4. US SEC Training Data vs. European Legal-Tech Deployment Bias   [IMPLEMENTE : run_us_to_eu_counterfactual_audit]:
   - Identify US-specific statutory classes in LEDGAR ('Erisa', 'Waiver Of Jury Trials').
   - Run a Counterfactual Perturbation Test (swapping US states/courts/statutes/currencies like 'Delaware', '$', 'SEC'
     with European equivalents like 'France', 'EUR', 'ESMA', 'GDPR') and measure prediction flip rate & confidence drop.

Depuis la racine du projet :
    python -m src.bias_audit --audit length
    python -m src.bias_audit --audit taxonomy
    python -m src.bias_audit --audit counterfactual
"""

import argparse
import csv
import json
import re
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


# ----------------------------------------------------------------------------------------------
# Audit 3 : US -> Europe (test contrefactuel)
# ----------------------------------------------------------------------------------------------
# Idee : on prend les clauses de test qui contiennent un element americain (Etat, dollar, SEC, ERISA),
# on le remplace par un equivalent europeen, et on regarde si le modele change d'avis (flip) et s'il
# devient moins sur de lui. Les regles de remplacement sont fixees ICI, avant de voir les resultats.

_US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
]
_STATE_ALT = "|".join(re.escape(x) for x in sorted(_US_STATES, key=len, reverse=True))
# "the State of New York", "Commonwealth of Massachusetts" : groupe 1 = le debut, groupe 2 = le nom de l'Etat
_STATE_RE = re.compile(rf"\b((?:the\s+)?(?:State|Commonwealth)\s+of\s+)({_STATE_ALT})\b", re.IGNORECASE)
# noms seuls, uniquement les 4 plus frequents (et pas "New York Stock Exchange")
_BARE_RE = re.compile(r"\b(Delaware|New York|California|Texas)\b(?!\s+(?:Stock Exchange|Times|Mellon))", re.IGNORECASE)
_DOLLAR_SIGN_RE = re.compile(r"\$\s?(?=\d)")
_DOLLAR_WORD_RE = re.compile(r"\b(?:(?:U\.S\.|US|United States)\s+)?dollars?\b", re.IGNORECASE)
_SEC_LONG_RE = re.compile(r"\bSecurities and Exchange Commission\b", re.IGNORECASE)
_SEC_ABBR_RE = re.compile(r"\bSEC\b")  # sensible a la casse : seulement "SEC" en majuscules
_ERISA_LONG_RE = re.compile(r"\bEmployee Retirement Income Security Act(?: of 1974)?", re.IGNORECASE)
_ERISA_ABBR_RE = re.compile(r"\bERISA\b")

_US_ROTATION = {"delaware": "Ohio", "new york": "California", "california": "Texas", "texas": "Florida"}


def _match_case(source: str, replacement: str) -> str:
    return replacement.upper() if source.isupper() and len(source) > 3 else replacement


def _swap_jurisdiction(text: str, state_phrase: str, country: str) -> str:
    text = _STATE_RE.sub(lambda m: _match_case(m.group(0), state_phrase), text)
    return _BARE_RE.sub(lambda m: _match_case(m.group(0), country), text)


def _swap_currency(text: str) -> str:
    text = _DOLLAR_SIGN_RE.sub("EUR ", text)

    def word(m: "re.Match") -> str:
        plural = m.group(0).lower().endswith("s")
        new = "euros" if plural else "euro"
        return new.capitalize() if m.group(0)[0].isupper() else new

    return _DOLLAR_WORD_RE.sub(word, text)


def _swap_regulator(text: str) -> str:
    text = _SEC_LONG_RE.sub("European Securities and Markets Authority", text)
    return _SEC_ABBR_RE.sub("ESMA", text)


def _swap_statute(text: str) -> str:
    text = _ERISA_LONG_RE.sub("General Data Protection Regulation", text)
    return _ERISA_ABBR_RE.sub("GDPR", text)


def _control_us_swap(text: str) -> str:
    """Temoin : on change d'Etat americain (Delaware -> Ohio...) : un changement 'neutre', reste aux USA."""
    def state(m: "re.Match") -> str:
        return m.group(1) + _match_case(m.group(2), _US_ROTATION.get(m.group(2).lower(), "Ohio"))

    text = _STATE_RE.sub(state, text)
    return _BARE_RE.sub(lambda m: _match_case(m.group(0), _US_ROTATION[m.group(0).lower()]), text)


def _compose(*functions):
    def run(text: str) -> str:
        for f in functions:
            text = f(text)
        return text
    return run


def make_perturbations() -> Dict[str, Any]:
    fr = lambda t: _swap_jurisdiction(t, "the Republic of France", "France")
    de = lambda t: _swap_jurisdiction(t, "the Federal Republic of Germany", "Germany")
    return {
        "control_US_swap": _control_us_swap,
        "juris_FR": fr,
        "juris_DE": de,
        "currency_EUR": _swap_currency,
        "regulator_ESMA": _swap_regulator,
        "all_FR": _compose(fr, _swap_currency, _swap_regulator),
        "all_DE": _compose(de, _swap_currency, _swap_regulator),
        "statute_GDPR": _swap_statute,  # test de stress : l'etiquette n'est plus forcement juste
    }


US_SPECIFIC_CLASSES = ("Erisa", "Waiver Of Jury Trials")


def _bootstrap_mean_ci(values, n_boot: int = 2000, confidence: float = 0.95, seed: int = 42) -> Dict[str, float]:
    """Moyenne + IC bootstrap (percentiles), memes reglages que le reste du projet (2000 tirages, seed 42)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {"estimate": float(values.mean()), "ci_lower": float(lo), "ci_upper": float(hi)}


def _evaluate_variant(
    transform, test_texts, y_true, predict_logits, base_logits, label_names, n_boot: int = 2000, seed: int = 42
) -> Dict[str, Any]:
    from src.evaluation import softmax

    perturbed = [transform(t) for t in test_texts]
    changed = np.array([a != b for a, b in zip(test_texts, perturbed)])
    idx = np.where(changed)[0]
    out: Dict[str, Any] = {"n_changed": int(len(idx)), "share_changed": float(len(idx) / len(test_texts))}
    if len(idx) == 0:
        return out, idx, []
    new_texts = [perturbed[i] for i in idx]
    old_logits, new_logits = base_logits[idx], predict_logits(new_texts)
    p_old, p_new = softmax(old_logits), softmax(new_logits)
    pred_old, pred_new = p_old.argmax(axis=1), p_new.argmax(axis=1)
    y = y_true[idx]
    flips = pred_old != pred_new

    k, n = int(flips.sum()), len(idx)
    p, lo, hi = wilson_confidence_interval(k, n)
    out["flip_rate"] = {"k": k, "n": n, "p": p, "ci_lower": lo, "ci_upper": hi}
    for tag, pred in (("original", pred_old), ("perturbed", pred_new)):
        kk = int((pred == y).sum())
        pp, l, h = wilson_confidence_interval(kk, n)
        out[f"accuracy_{tag}"] = {"k": kk, "n": n, "p": pp, "ci_lower": l, "ci_upper": h}
    out["confidence_drop"] = _bootstrap_mean_ci(p_old.max(axis=1) - p_new.max(axis=1), n_boot, seed=seed)
    out["mean_confidence_original"] = float(p_old.max(axis=1).mean())

    transitions: Dict[Any, int] = {}
    for a, b in zip(pred_old[flips], pred_new[flips]):
        transitions[(label_names[a], label_names[b])] = transitions.get((label_names[a], label_names[b]), 0) + 1
    out["top_flips"] = [
        {"from": a, "to": b, "count": c} for (a, b), c in sorted(transitions.items(), key=lambda kv: -kv[1])[:5]
    ]

    per_class = {}
    for cname in US_SPECIFIC_CLASSES:
        if cname not in label_names:
            continue
        mask = y == label_names.index(cname)
        nn, kk = int(mask.sum()), int(flips[mask].sum())
        pp, l, h = wilson_confidence_interval(kk, nn)
        per_class[cname] = {"n_changed": nn, "n_flips": kk, "p": pp, "ci_lower": l, "ci_upper": h}
    out["us_specific_classes"] = per_class
    return out, idx, new_texts


def run_us_to_eu_counterfactual_audit(
    test_texts: List[str], test_labels: np.ndarray, predict_logits, label_names: List[str]
) -> Dict[str, Any]:
    """Pour chaque remplacement : taux de changement de prediction (flip) et baisse de confiance.
    `predict_logits(textes)` doit renvoyer un tableau (n_textes, n_classes)."""
    y_true = np.asarray(test_labels)
    base_logits = np.asarray(predict_logits(list(test_texts)))
    results: Dict[str, Any] = {"variants": {}}
    exports: Dict[str, Any] = {}
    for name, transform in make_perturbations().items():
        res, idx, new_texts = _evaluate_variant(transform, test_texts, y_true, predict_logits, base_logits, label_names)
        results["variants"][name] = res
        exports[name] = {"indices": [int(i) for i in idx], "texts": new_texts}
    return {"results": results, "exports": exports}


def format_counterfactual_report(model: str, results: Dict[str, Any]) -> str:
    lines = [f"\n=== {model} : test contrefactuel US -> Europe ==="]
    lines.append(f"{'variante':<17} {'clauses changees':>17}   {'flip [IC 95 %]':<24} {'exactitude avant -> apres':<27} {'baisse de confiance [IC 95 %]'}")
    for name, r in results["variants"].items():
        if r["n_changed"] == 0:
            lines.append(f"{name:<17} {0:>17}   (aucune clause concernee)")
            continue
        f, a0, a1, c = r["flip_rate"], r["accuracy_original"], r["accuracy_perturbed"], r["confidence_drop"]
        lines.append(
            f"{name:<17} {r['n_changed']:>6} ({100 * r['share_changed']:>4.1f} %)   "
            f"{f['p']:.3f} [{f['ci_lower']:.3f}; {f['ci_upper']:.3f}]   "
            f"{a0['p']:.3f} -> {a1['p']:.3f}           "
            f"{c['estimate']:+.4f} [{c['ci_lower']:+.4f}; {c['ci_upper']:+.4f}]"
        )
    for name in ("all_FR", "juris_FR", "regulator_ESMA", "statute_GDPR"):
        r = results["variants"].get(name)
        if not r or r["n_changed"] == 0:
            continue
        lines.append(f"\n--- {name} : changements de categorie les plus frequents ---")
        for t in r["top_flips"]:
            lines.append(f"   {t['count']:>4}  {t['from']}  ->  {t['to']}")
        for cname, d in r["us_specific_classes"].items():
            if d["n_changed"] == 0:
                continue
            lines.append(f"   classe americaine {cname}: {d['n_flips']}/{d['n_changed']} changent = {d['p']:.3f} [{d['ci_lower']:.3f}; {d['ci_upper']:.3f}]")
    return "\n".join(lines)


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


def _train_or_load_tfidf(train_texts, train_labels, cache_dir: Path):
    """Reentraine TF-IDF + regression logistique avec EXACTEMENT les reglages de src/train.py (personne 1)."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "tfidf_model.joblib"
    if cache_file.exists():
        return joblib.load(cache_file)
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    x_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(C=2.0, max_iter=300, random_state=42)
    clf.fit(x_train, train_labels)
    joblib.dump((vectorizer, clf), cache_file)
    return vectorizer, clf


def _main_counterfactual() -> None:
    from src.data import load_ledgar

    dataset, _ = load_ledgar()
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])
    saved_labels = np.load(OUTPUTS_DIR / "test_labels.npy")
    if len(saved_labels) != len(test_labels) or not np.array_equal(saved_labels, test_labels):
        raise SystemExit("ERREUR : l'ordre du test ne correspond pas a outputs/test_labels.npy -> on arrete.")
    with open(OUTPUTS_DIR / "label_names.json", encoding="utf-8") as f:
        label_names = json.load(f)

    print("Entrainement de TF-IDF + regression logistique (quelques minutes la 1re fois, ensuite c'est en cache)...")
    vectorizer, clf = _train_or_load_tfidf(dataset["train"]["text"], np.array(dataset["train"]["label"]), AUDIT_DIR / "_cache")

    def predict_logits(texts):
        return clf.decision_function(vectorizer.transform(texts)).astype(np.float32)

    # controle : notre TF-IDF doit retrouver (presque) les memes predictions que celui de la personne 1
    ours = predict_logits(test_texts).argmax(axis=1)
    theirs = np.load(OUTPUTS_DIR / "tfidf_test_logits.npy").argmax(axis=1)
    print(f"Controle : accord avec les predictions TF-IDF de la personne 1 = {100 * (ours == theirs).mean():.2f} %"
          f" | notre exactitude sur le test = {(ours == test_labels).mean():.4f}")

    out = run_us_to_eu_counterfactual_audit(test_texts, test_labels, predict_logits, label_names)
    print(format_counterfactual_report("TF-IDF", out["results"]))

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "counterfactual_audit.json", "w", encoding="utf-8") as f:
        json.dump({"model": "TF-IDF (reentraine, memes reglages que la personne 1)", "results": out["results"]},
                  f, indent=2, ensure_ascii=False)
    # textes modifies : pour que la personne 1 les passe dans Legal-BERT (Colab) et renvoie les logits
    with open(AUDIT_DIR / "counterfactual_perturbed_texts.json", "w", encoding="utf-8") as f:
        json.dump(out["exports"], f, ensure_ascii=False)
    print(f"\nFichiers ecrits dans {AUDIT_DIR}/counterfactual_audit.json et counterfactual_perturbed_texts.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audits de biais (personne 3)")
    parser.add_argument("--audit", choices=("length", "taxonomy", "counterfactual"), required=True)
    args = parser.parse_args()
    if args.audit == "length":
        _main_length()
    elif args.audit == "taxonomy":
        _main_taxonomy()
    elif args.audit == "counterfactual":
        _main_counterfactual()