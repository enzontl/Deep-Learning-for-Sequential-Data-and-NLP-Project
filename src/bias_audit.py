"""
Person 3 — Bias Audit & Failure-Mode Analysis
=============================================
Tasks in this module:
1. Sample-Size / Long-Tail Bias   [IMPLEMENTE : run_long_tail_audit]:
   Compare model F1 and 95% CI width across Head classes (n_train >= 1000), Medium (150 <= n_train < 1000),
   and Tail classes (n_train < 150) across Legal-BERT, TF-IDF, and the Calibrated Ensemble.
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
    python -m src.bias_audit --audit long-tail
    python -m src.bias_audit --audit length
    python -m src.bias_audit --audit taxonomy
    python -m src.bias_audit --audit counterfactual
    python -m src.bias_audit --audit plots
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    lines = [f"\n=== {model} : {res['n_errors']} errors on {res['n_test']} test clauses ==="]
    lines.append(f"Top {len(res['top_pairs'])} confused pairs = {100 * res['top_pairs_share_of_errors']:.1f}% of all errors")
    lines.append(f"\n{'rank':<5} {'pair':<58} {'A->B':>5} {'B->A':>5} {'total':>6}")
    for r in res["top_pairs"]:
        lines.append(f"{r['rank']:<5} {(r['class_a'] + ' / ' + r['class_b']):<58} {r['a_as_b']:>5} {r['b_as_a']:>5} {r['total']:>6}")
    lines.append("\n--- Compound Boilerplate Hypothesis (README pairs) ---")
    for c in res["readme_pairs"]:
        lines.append(f"{c['pair']}: rank {c['rank_among_pairs']}, {c['n_confused']} confusions")
        for label in ("confused", "correct"):
            d = c[f"share_with_both_keywords_{label}"]
            txt = "CONFUSED clauses" if label == "confused" else "CORRECTLY classified clauses"
            lines.append(f"   contain both keywords {c['keywords']} among {txt}: "
                         f"{d['k']}/{d['n']} = {d['p']:.3f} [{d['ci_lower']:.3f}; {d['ci_upper']:.3f}]")
    lines.append("\n--- Error examples (top 3 pairs) ---")
    for e in res["examples"]:
        lines.append(f"[{e['pair']}] true={e['true']} | predicted={e['predicted']}\n    {e['text_start']}")
    return "\n".join(lines)


def _load_model_test_logits(outputs_dir: Path = OUTPUTS_DIR) -> Dict[str, np.ndarray]:
    """Load test logits for Legal-BERT, TF-IDF, and the Calibrated Ensemble."""
    from src.evaluation import build_ensemble

    lb_val = np.load(outputs_dir / "legal_bert_val_logits.npy")
    lb_test = np.load(outputs_dir / "legal_bert_test_logits.npy")
    tf_val = np.load(outputs_dir / "tfidf_val_logits.npy")
    tf_test = np.load(outputs_dir / "tfidf_test_logits.npy")
    y_val = np.load(outputs_dir / "val_labels.npy")
    _, ens_test, _ = build_ensemble(lb_val, tf_val, y_val, lb_test, tf_test)
    return {"Legal-BERT": lb_test, "TF-IDF": tf_test, "Ensemble": ens_test}


def _main_taxonomy() -> None:
    from src.data import load_ledgar

    dataset, _ = load_ledgar()
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])
    saved_labels = np.load(OUTPUTS_DIR / "test_labels.npy")
    if len(saved_labels) != len(test_labels) or not np.array_equal(saved_labels, test_labels):
        raise SystemExit("ERROR: test split order does not match outputs/test_labels.npy.")
    with open(OUTPUTS_DIR / "label_names.json", encoding="utf-8") as f:
        label_names = json.load(f)

    results = {}
    for name, logits in _load_model_test_logits().items():
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
    print(f"\nSaved {AUDIT_DIR}/taxonomy_audit.json and taxonomy_top_pairs.csv")


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

_US_ROTATION = {
    "delaware": "Ohio",
    "new york": "California",
    "california": "Texas",
    "texas": "Florida",
    "ohio": "Delaware",
}


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


_PARTY_ROLE_SWAP_MAP = {
    "lender": "Borrower",
    "lenders": "Borrowers",
    "borrower": "Lender",
    "borrowers": "Lenders",
    "employer": "Employee",
    "employers": "Employees",
    "employee": "Employer",
    "employees": "Employers",
    "landlord": "Tenant",
    "landlords": "Tenants",
    "tenant": "Landlord",
    "tenants": "Landlords",
    "licensor": "Licensee",
    "licensors": "Licensees",
    "licensee": "Licensor",
    "licensees": "Licensors",
}
_PARTY_ROLE_RE = re.compile(
    r"\b(Lenders?|Borrowers?|Employers?|Employees?|Landlords?|Tenants?|Licensors?|Licensees?)\b",
    re.IGNORECASE,
)


def _swap_party_roles(text: str) -> str:
    """Counterfactual party/role symmetry test (Session 7 Slide 24): swap symmetric contractual party roles."""
    def _repl(m: "re.Match") -> str:
        word = m.group(0)
        target = _PARTY_ROLE_SWAP_MAP[word.lower()]
        if word.isupper():
            return target.upper()
        if word[0].isupper():
            return target
        return target.lower()

    return _PARTY_ROLE_RE.sub(_repl, text)


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
        "party_role_swap": _swap_party_roles,
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
    transform,
    test_texts,
    y_true,
    predict_logits,
    base_logits,
    label_names,
    temperature: float = 1.0,
    n_boot: int = 2000,
    seed: int = 42,
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
    p_old, p_new = softmax(old_logits, temperature), softmax(new_logits, temperature)
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
    test_texts: List[str],
    test_labels: np.ndarray,
    predict_logits,
    label_names: List[str],
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate prediction flip rate and confidence drop for each US->EU perturbation variant."""
    y_true = np.asarray(test_labels)
    base_logits = np.asarray(predict_logits(list(test_texts)))
    results: Dict[str, Any] = {"variants": {}}
    exports: Dict[str, Any] = {}
    for name, transform in make_perturbations().items():
        res, idx, new_texts = _evaluate_variant(
            transform, test_texts, y_true, predict_logits, base_logits, label_names, temperature=temperature
        )
        results["variants"][name] = res
        exports[name] = {"indices": [int(i) for i in idx], "texts": new_texts}
    return {"results": results, "exports": exports}


def compute_us_specific_and_cohort_baselines(
    outputs_dir: Path = OUTPUTS_DIR,
    cohort_indices: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Compute baseline accuracy and 95% Wilson CIs on US-specific statutory classes
    ('Erisa', 'Waiver Of Jury Trials') and US perturbation cohorts across all 3 models
    (Legal-BERT, TF-IDF, Calibrated Ensemble) using the saved test logits.
    """
    y_test = np.load(outputs_dir / "test_labels.npy")
    with open(outputs_dir / "label_names.json", encoding="utf-8") as f:
        label_names = json.load(f)
    model_logits = _load_model_test_logits(outputs_dir)

    us_classes: Dict[str, Any] = {}
    for cname in US_SPECIFIC_CLASSES:
        if cname not in label_names:
            continue
        cid = label_names.index(cname)
        mask = y_test == cid
        n = int(mask.sum())
        us_classes[cname] = {"class_id": cid, "n_test": n, "models": {}}
        for mname, logits in model_logits.items():
            k = int((logits.argmax(axis=1)[mask] == cid).sum())
            p, lo, hi = wilson_confidence_interval(k, n)
            us_classes[cname]["models"][mname] = {"k": k, "n": n, "recall": p, "ci_lower": lo, "ci_upper": hi}

    cohort_acc: Dict[str, Any] = {}
    if cohort_indices:
        for variant_name, idx_list in cohort_indices.items():
            idx = np.asarray(idx_list, dtype=np.int64)
            n = len(idx)
            if n == 0:
                continue
            cohort_acc[variant_name] = {"n": n, "models": {}}
            for mname, logits in model_logits.items():
                k = int((logits.argmax(axis=1)[idx] == y_test[idx]).sum())
                p, lo, hi = wilson_confidence_interval(k, n)
                cohort_acc[variant_name]["models"][mname] = {"k": k, "n": n, "accuracy": p, "ci_lower": lo, "ci_upper": hi}

    return {"us_specific_classes_test_summary": us_classes, "cohort_original_accuracy_by_model": cohort_acc}


def format_counterfactual_report(model: str, results: Dict[str, Any]) -> str:
    lines = [f"\n=== {model} : US -> European Law Counterfactual Perturbation Audit ==="]
    lines.append(
        f"{'variant':<17} {'modified clauses':>17}   {'flip rate [95% CI]':<24} "
        f"{'accuracy before -> after':<27} {'confidence drop [95% CI]'}"
    )
    for name, r in results["variants"].items():
        if r["n_changed"] == 0:
            lines.append(f"{name:<17} {0:>17}   (no matching clauses)")
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
        lines.append(f"\n--- {name} : most frequent prediction flips ---")
        for t in r["top_flips"]:
            lines.append(f"   {t['count']:>4}  {t['from']}  ->  {t['to']}")
        for cname, d in r["us_specific_classes"].items():
            if d["n_changed"] == 0:
                continue
            lines.append(
                f"   US-specific class {cname}: {d['n_flips']}/{d['n_changed']} flip = "
                f"{d['p']:.3f} [{d['ci_lower']:.3f}; {d['ci_upper']:.3f}]"
            )
    return "\n".join(lines)


def _main_length() -> None:
    from src.data import load_ledgar

    dataset, _ = load_ledgar()
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])

    saved_labels = np.load(OUTPUTS_DIR / "test_labels.npy")
    if len(saved_labels) != len(test_labels) or not np.array_equal(saved_labels, test_labels):
        raise SystemExit("ERROR: test split order does not match outputs/test_labels.npy.")

    results = {}
    for name, logits in _load_model_test_logits().items():
        results[name] = run_length_truncation_audit(test_texts, test_labels, logits)

    print(format_length_table(results))
    save_length_results(results)
    print(f"\nSaved {AUDIT_DIR}/length_audit.json and .csv")


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
    from src.evaluation import fit_temperature

    dataset, _ = load_ledgar()
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])
    saved_labels = np.load(OUTPUTS_DIR / "test_labels.npy")
    if len(saved_labels) != len(test_labels) or not np.array_equal(saved_labels, test_labels):
        raise SystemExit("ERROR: test split order does not match outputs/test_labels.npy.")
    with open(OUTPUTS_DIR / "label_names.json", encoding="utf-8") as f:
        label_names = json.load(f)

    print("Training/loading TF-IDF + LogisticRegression (cached after first run)...")
    vectorizer, clf = _train_or_load_tfidf(
        dataset["train"]["text"], np.array(dataset["train"]["label"]), AUDIT_DIR / "_cache"
    )

    def predict_logits(texts):
        return clf.decision_function(vectorizer.transform(texts)).astype(np.float32)

    ours = predict_logits(test_texts).argmax(axis=1)
    theirs = np.load(OUTPUTS_DIR / "tfidf_test_logits.npy").argmax(axis=1)
    print(
        f"Verification: agreement with Person 1 saved TF-IDF predictions = {100 * (ours == theirs).mean():.2f}% "
        f"| test accuracy = {(ours == test_labels).mean():.4f}"
    )

    val_labels = np.load(OUTPUTS_DIR / "val_labels.npy")
    tfidf_val_logits = np.load(OUTPUTS_DIR / "tfidf_val_logits.npy")
    tfidf_temp = fit_temperature(tfidf_val_logits, val_labels)

    out = run_us_to_eu_counterfactual_audit(test_texts, test_labels, predict_logits, label_names, temperature=1.0)
    out_cal = run_us_to_eu_counterfactual_audit(
        test_texts, test_labels, predict_logits, label_names, temperature=tfidf_temp
    )
    for vname, vres in out["results"]["variants"].items():
        if vres["n_changed"] > 0:
            vres["confidence_drop_calibrated"] = out_cal["results"]["variants"][vname]["confidence_drop"]

    print(format_counterfactual_report("TF-IDF", out["results"]))

    cohort_indices = {k: v["indices"] for k, v in out["exports"].items()}
    multi_model_baselines = compute_us_specific_and_cohort_baselines(OUTPUTS_DIR, cohort_indices)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "counterfactual_audit.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "TF-IDF (full-text lexical classifier, exact Person 1 hyperparameters) + Multi-Model US Cohort Baselines",
                "results": out["results"],
                **multi_model_baselines,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(AUDIT_DIR / "counterfactual_perturbed_texts.json", "w", encoding="utf-8") as f:
        json.dump(out["exports"], f, ensure_ascii=False)
    print(f"\nSaved {AUDIT_DIR}/counterfactual_audit.json and counterfactual_perturbed_texts.json")


# ----------------------------------------------------------------------------------------------
# Audit 0 (Task 1) : Sample-Size / Long-Tail Bias (Head vs. Medium vs. Tail classes)
# ----------------------------------------------------------------------------------------------

TRAIN_SIZE_BUCKETS = [
    ("Head (n_train >= 1000)", 1000, 10**9),
    ("Medium (150 <= n_train < 1000)", 150, 1000),
    ("Tail (n_train < 150)", 0, 150),
]


def run_long_tail_audit(outputs_dir: Path = OUTPUTS_DIR) -> Dict[str, Any]:
    """
    Compare F1 and 95% CI width across Head (n_train >= 1000), Medium (150 <= n_train < 1000),
    and Tail (n_train < 150) classes for Legal-BERT, TF-IDF, and the Calibrated Ensemble.
    Uses standard csv module so it runs with or without pandas installed.
    """
    with open(outputs_dir / "class_distribution.csv", encoding="utf-8") as f:
        dist_rows = [
            {
                "class_id": int(r["class_id"]),
                "class_name": r["class_name"],
                "train_count": int(r["train_count"]),
                "val_count": int(r["val_count"]),
                "test_count": int(r["test_count"]),
            }
            for r in csv.DictReader(f)
        ]

    eval_dir = outputs_dir / "evaluation"
    models: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for model_name, fname in (
        ("Legal-BERT", "legal_bert_per_class.csv"),
        ("TF-IDF", "tfidf_per_class.csv"),
        ("Ensemble", "ensemble_per_class.csv"),
    ):
        with open(eval_dir / fname, encoding="utf-8") as f:
            models[model_name] = {
                int(r["class_id"]): {
                    "class_id": int(r["class_id"]),
                    "class_name": r["class_name"],
                    "n_test": int(r["n_test"]),
                    "f1": float(r["f1"]),
                    "f1_ci_lower": float(r["f1_ci_lower"]),
                    "f1_ci_upper": float(r["f1_ci_upper"]),
                    "f1_ci_width": float(r["f1_ci_width"]),
                    "tier": r["tier"],
                }
                for r in csv.DictReader(f)
            }

    results_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for model_name, eval_by_id in models.items():
        merged = [{**d, **eval_by_id[d["class_id"]]} for d in dist_rows if d["class_id"] in eval_by_id]
        rows = []
        for bucket_name, low, high in TRAIN_SIZE_BUCKETS:
            sub = [r for r in merged if low <= r["train_count"] < high]
            f1s = np.array([r["f1"] for r in sub], dtype=float)
            widths = np.array([r["f1_ci_width"] for r in sub], dtype=float)
            trains = np.array([r["train_count"] for r in sub], dtype=float)
            tests = np.array([r["n_test"] for r in sub], dtype=float)
            rows.append({
                "bucket": bucket_name,
                "n_classes": int(len(sub)),
                "mean_train_count": float(trains.mean()),
                "mean_test_count": float(tests.mean()),
                "mean_f1": float(f1s.mean()),
                "median_f1": float(np.median(f1s)),
                "min_f1": float(f1s.min()),
                "mean_ci_width": float(widths.mean()),
                "max_ci_width": float(widths.max()),
                "n_zero_f1": int((f1s == 0.0).sum()),
                "zero_f1_classes": [r["class_name"] for r in sub if r["f1"] == 0.0],
                "tier_1_auto": int(sum(r["tier"].startswith("Tier 1") for r in sub)),
                "tier_2_assisted": int(sum(r["tier"].startswith("Tier 2") for r in sub)),
                "tier_3_blocked": int(sum(r["tier"].startswith("Tier 3") for r in sub)),
            })
        results_by_model[model_name] = rows

    # 10 rarest classes in the training split
    tail_10 = sorted(dist_rows, key=lambda r: r["train_count"])[:10]
    tail_comparison = []
    for row in tail_10:
        cid = int(row["class_id"])
        entry: Dict[str, Any] = {
            "class_id": cid,
            "class_name": row["class_name"],
            "n_train": int(row["train_count"]),
            "n_test": int(row["test_count"]),
        }
        for model_name, eval_by_id in models.items():
            m_row = eval_by_id[cid]
            entry[model_name] = {
                "f1": float(m_row["f1"]),
                "ci_lower": float(m_row["f1_ci_lower"]),
                "ci_upper": float(m_row["f1_ci_upper"]),
                "ci_width": float(m_row["f1_ci_width"]),
                "tier": str(m_row["tier"]),
            }
        tail_comparison.append(entry)

    return {"buckets_by_model": results_by_model, "rarest_10_classes": tail_comparison}


def _main_long_tail() -> None:
    res = run_long_tail_audit()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "long_tail_audit.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(AUDIT_DIR / "long_tail_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "bucket", "n_classes", "mean_train_count", "mean_test_count",
            "mean_f1", "median_f1", "min_f1", "mean_ci_width", "max_ci_width",
            "n_zero_f1", "tier_1_auto", "tier_2_assisted", "tier_3_blocked",
        ])
        for model_name, rows in res["buckets_by_model"].items():
            for r in rows:
                writer.writerow([
                    model_name, r["bucket"], r["n_classes"], r["mean_train_count"], r["mean_test_count"],
                    r["mean_f1"], r["median_f1"], r["min_f1"], r["mean_ci_width"], r["max_ci_width"],
                    r["n_zero_f1"], r["tier_1_auto"], r["tier_2_assisted"], r["tier_3_blocked"],
                ])

    for model_name, rows in res["buckets_by_model"].items():
        print(f"\n=== {model_name} : Sample-Size / Long-Tail Bias (Head / Medium / Tail) ===")
        print(f"{'Bucket':<32} {'Classes':>7} {'Mean F1':>9} {'Mean CI Width':>15} {'F1=0':>6} {'T1/T2/T3':>12}")
        for r in rows:
            tiers = f"{r['tier_1_auto']}/{r['tier_2_assisted']}/{r['tier_3_blocked']}"
            print(
                f"{r['bucket']:<32} {r['n_classes']:>7} {r['mean_f1']:>9.3f} "
                f"{r['mean_ci_width']:>15.3f} {r['n_zero_f1']:>6} {tiers:>12}"
            )
    print(f"\nSaved {AUDIT_DIR}/long_tail_audit.json and .csv")


def generate_audit_and_retrieval_plots(outputs_dir: Path = OUTPUTS_DIR) -> None:
    """Generate English PNG figures for Retrieval, Long-Tail Bias, Length Truncation, and US->EU Counterfactual audits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit_dir = outputs_dir / "bias_audit"
    retr_dir = outputs_dir / "retrieval_eval"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # 1) Long-Tail Bias: F1 vs n_train (log scale) and 95% CI width vs n_test (log scale)
    with open(outputs_dir / "class_distribution.csv", encoding="utf-8") as f:
        dist = {int(r["class_id"]): int(r["train_count"]) for r in csv.DictReader(f)}

    def _load_per_class(fname: str) -> List[Dict[str, Any]]:
        with open(outputs_dir / "evaluation" / fname, encoding="utf-8") as f:
            return [
                {
                    "class_id": int(r["class_id"]),
                    "class_name": r["class_name"],
                    "train_count": dist[int(r["class_id"])],
                    "n_test": int(r["n_test"]),
                    "f1": float(r["f1"]),
                    "f1_ci_width": float(r["f1_ci_width"]),
                }
                for r in csv.DictReader(f)
            ]

    m_lb = _load_per_class("legal_bert_per_class.csv")
    m_ens = _load_per_class("ensemble_per_class.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].scatter(
        [r["train_count"] for r in m_lb],
        [r["f1"] for r in m_lb],
        alpha=0.65,
        s=26,
        color="#c0392b",
        label="Legal-BERT (5 classes at F1=0)",
    )
    axes[0].scatter(
        [r["train_count"] for r in m_ens],
        [r["f1"] for r in m_ens],
        alpha=0.75,
        s=26,
        color="#2e8b57",
        label="Calibrated Ensemble (0.45 LB + 0.55 TF-IDF)",
    )
    axes[0].axvline(150, color="grey", linestyle="--", linewidth=1, label="Tail threshold (n_train = 150)")
    axes[0].axvline(1000, color="black", linestyle=":", linewidth=1, label="Head threshold (n_train = 1,000)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training sample size n_train (log scale)")
    axes[0].set_ylabel("Per-class Test F1")
    axes[0].set_ylim(-0.04, 1.03)
    axes[0].set_title("Long-Tail Sample-Size Collapse (Test F1 vs. n_train)")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].scatter(
        [r["n_test"] for r in m_ens],
        [r["f1_ci_width"] for r in m_ens],
        alpha=0.75,
        s=28,
        color="#2b6cb0",
        label="Ensemble per-class 95% CI width",
    )
    offsets = {
        "Books": (6, -8),
        "Assigns": (6, 4),
        "Qualifications": (6, -2),
        "Powers": (6, 4),
        "Sanctions": (-78, -4),
    }
    for r in sorted(m_ens, key=lambda x: x["n_test"])[:5]:
        dx, dy = offsets.get(r["class_name"], (6, 2))
        axes[1].annotate(
            f"{r['class_name']} (n={int(r['n_test'])})",
            (r["n_test"], r["f1_ci_width"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            fontweight="medium",
        )
    axes[1].axhline(0.30, color="#c0392b", linestyle="--", linewidth=1.2, label="Tier-1 Auto-Tag ceiling (CI width <= 0.30)")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.03, 1.08)
    axes[1].set_xlabel("Test sample size n_test (log scale)")
    axes[1].set_ylabel("95% Confidence Interval Width of F1")
    axes[1].set_title("95% CI Width Explosion When n_test < 15")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(audit_dir / "long_tail_bias.png", dpi=150)
    plt.close(fig)

    # 2) Length & Truncation Bias
    if (audit_dir / "length_audit.json").exists():
        with open(audit_dir / "length_audit.json", encoding="utf-8") as f:
            len_data = json.load(f)["results"]
        buckets = [r["bucket"] for r in len_data["Legal-BERT"]]
        x = np.arange(len(buckets))
        model_list = [("Legal-BERT", "#c0392b"), ("TF-IDF", "#2b6cb0")]
        if "Ensemble" in len_data:
            model_list.append(("Ensemble", "#2e8b57"))
        width = 0.8 / len(model_list)
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        for i, (m_name, color) in enumerate(model_list):
            rows = len_data[m_name]
            acc = np.array([r["accuracy"] for r in rows])
            lo = acc - np.array([r["ci_lower"] for r in rows])
            hi = np.array([r["ci_upper"] for r in rows]) - acc
            offset = (i - (len(model_list) - 1) / 2) * width
            ax.bar(x + offset, acc, width=width * 0.92, color=color, alpha=0.88, label=m_name)
            ax.errorbar(x + offset, acc, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b} words\n(n={r['n']:,})" for b, r in zip(buckets, len_data["Legal-BERT"])])
        ax.set_ylim(0.78, 0.94)
        ax.set_ylabel("Test Accuracy (with 95% Wilson CI)")
        ax.set_title("Audit 1 — Clause Length & 256-Token Truncation Bias (>260 words: TF-IDF +2.2 pts)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(audit_dir / "length_truncation_bias.png", dpi=150)
        plt.close(fig)

    # 3) US -> EU Counterfactual Perturbation Audit
    if (audit_dir / "counterfactual_audit.json").exists():
        with open(audit_dir / "counterfactual_audit.json", encoding="utf-8") as f:
            cf = json.load(f)["results"]["variants"]
        order = ["control_US_swap", "party_role_swap", "currency_EUR", "regulator_ESMA", "juris_DE", "juris_FR", "all_FR", "statute_GDPR"]
        labels = [
            f"US Control (Delaware->Ohio)\n(n={cf['control_US_swap']['n_changed']}, k={cf['control_US_swap']['flip_rate']['k']} flips)",
            f"Party Swap (Lender<->Borrower)\n(n={cf['party_role_swap']['n_changed']}, k={cf['party_role_swap']['flip_rate']['k']} flips)",
            f"Currency ($ -> EUR)\n(n={cf['currency_EUR']['n_changed']}, k={cf['currency_EUR']['flip_rate']['k']} flip)",
            f"Regulator (SEC -> ESMA)\n(n={cf['regulator_ESMA']['n_changed']}, k={cf['regulator_ESMA']['flip_rate']['k']} flips)",
            f"German Law (-> Germany)\n(n={cf['juris_DE']['n_changed']}, k={cf['juris_DE']['flip_rate']['k']} flips)",
            f"French Law (-> France)\n(n={cf['juris_FR']['n_changed']}, k={cf['juris_FR']['flip_rate']['k']} flips)",
            f"Full FR Union (FR+EUR+ESMA)\n(n={cf['all_FR']['n_changed']}, k={cf['all_FR']['flip_rate']['k']} flips)",
            f"EU Statute (ERISA -> GDPR)\n(n={cf['statute_GDPR']['n_changed']}, k={cf['statute_GDPR']['flip_rate']['k']} flips)",
        ]
        y = np.arange(len(order))
        flips = np.array([100 * cf[k]["flip_rate"]["p"] for k in order])
        f_lo = flips - np.array([100 * cf[k]["flip_rate"]["ci_lower"] for k in order])
        f_hi = np.array([100 * cf[k]["flip_rate"]["ci_upper"] for k in order]) - flips

        drops = np.array([100 * cf[k]["confidence_drop"]["estimate"] for k in order])
        d_lo = drops - np.array([100 * cf[k]["confidence_drop"]["ci_lower"] for k in order])
        d_hi = np.array([100 * cf[k]["confidence_drop"]["ci_upper"] for k in order]) - drops

        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
        colors = ["#7f8c8d", "#7f8c8d", "#2e8b57", "#2e8b57", "#e69f00", "#e69f00", "#d35400", "#c0392b"]
        axes[0].barh(y, flips, color=colors, alpha=0.88)
        axes[0].errorbar(flips, y, xerr=[f_lo, f_hi], fmt="none", ecolor="black", capsize=3, lw=1.1)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=8.5)
        axes[0].set_xlabel("Prediction Flip Rate (%, 95% Wilson CI)")
        axes[0].set_title("Label Instability Under US -> EU Swaps")

        axes[1].barh(y, drops, color=colors, alpha=0.88)
        axes[1].errorbar(drops, y, xerr=[d_lo, d_hi], fmt="none", ecolor="black", capsize=3, lw=1.1)
        axes[1].set_xlabel("Mean Confidence Drop (percentage points, 95% Bootstrap CI)")
        axes[1].set_title("Confidence Drop (Original - Perturbed Probability)")
        fig.suptitle("Audit 3 — US (SEC) to European Law Counterfactual Perturbation Test", fontsize=11.5)
        fig.tight_layout()
        fig.savefig(audit_dir / "counterfactual_audit.png", dpi=150)
        plt.close(fig)

    # 4) Clause Bank Retrieval Comparison (Clause-to-Clause vs Lawyer Constraint Queries)
    if (retr_dir / "retrieval_eval_test.json").exists() and (retr_dir / "retrieval_lawyer_queries.json").exists():
        with open(retr_dir / "retrieval_eval_test.json", encoding="utf-8") as f:
            r_test = json.load(f)["results"]["modes"]
        with open(retr_dir / "retrieval_lawyer_queries.json", encoding="utf-8") as f:
            r_law = json.load(f)["results"]["modes"]

        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
        # Panel A: Clause-to-Clause (n=1000 test queries)
        m1 = ["dense", "bm25", "hybrid"]
        lbl1 = ["Dense\n(MiniLM-L6)", "BM25\n(Lexical)", "Hybrid\n(RRF)"]
        p1 = np.array([100 * r_test[m]["all"]["precision"]["estimate"] for m in m1])
        p1_lo = p1 - np.array([100 * r_test[m]["all"]["precision"]["ci_lower"] for m in m1])
        p1_hi = np.array([100 * r_test[m]["all"]["precision"]["ci_upper"] for m in m1]) - p1
        x1 = np.arange(len(m1))
        axes[0].bar(x1, p1, color=["#7f8c8d", "#2b6cb0", "#2e8b57"], width=0.52, alpha=0.9)
        axes[0].errorbar(x1, p1, yerr=[p1_lo, p1_hi], fmt="none", ecolor="black", capsize=4, lw=1.2)
        axes[0].set_xticks(x1)
        axes[0].set_xticklabels(lbl1)
        axes[0].set_ylim(68, 82)
        axes[0].set_ylabel("Precision@5 (%, 95% Bootstrap CI)")
        axes[0].set_title(
            "Clause-to-Clause Precedent Search (Test, n=1,000)\n"
            "Hybrid RRF beats Dense by +2.0 pts* (95% CI [+1.0, +3.1])",
            fontsize=10.2,
        )

        # Panel B: Lawyer Constraint Queries (n=20 queries with numbers/states)
        m2 = ["dense", "dense+filtre", "hybrid", "bm25", "hybrid+filtre"]
        lbl2 = ["Dense\nOnly", "Dense\n+ Filter", "Hybrid\nOnly", "BM25\nOnly", "Hybrid\n+ Filter"]
        p2 = np.array([100 * r_law[m]["precision"]["estimate"] for m in m2])
        p2_lo = p2 - np.array([100 * r_law[m]["precision"]["ci_lower"] for m in m2])
        p2_hi = np.array([100 * r_law[m]["precision"]["ci_upper"] for m in m2]) - p2
        x2 = np.arange(len(m2))
        axes[1].bar(x2, p2, color=["#c0392b", "#e69f00", "#2b6cb0", "#34495e", "#2e8b57"], width=0.58, alpha=0.9)
        axes[1].errorbar(x2, p2, yerr=[p2_lo, p2_hi], fmt="none", ecolor="black", capsize=4, lw=1.2)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(lbl2, fontsize=8.8)
        axes[1].set_ylim(0, 95)
        axes[1].set_ylabel("Precision@5 (%, 95% Bootstrap CI)")
        axes[1].set_title(
            "Lawyer Numerical & Jurisdiction Queries (n=20)\n"
            "Dense drops to 45% ('30' vs '60' days); BM25=69%*, Hybrid+Filter=68%*",
            fontsize=10.2,
        )

        fig.tight_layout()
        fig.savefig(retr_dir / "retrieval_comparison.png", dpi=150)
        plt.close(fig)

    print(f"Generated English PNG figures in {audit_dir} and {retr_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audits de biais (personne 3)")
    parser.add_argument(
        "--audit",
        choices=("long-tail", "length", "taxonomy", "counterfactual", "plots", "all"),
        required=True,
    )
    args = parser.parse_args()
    if args.audit in ("long-tail", "all"):
        _main_long_tail()
    if args.audit in ("length", "all"):
        _main_length()
    if args.audit in ("taxonomy", "all"):
        _main_taxonomy()
    if args.audit in ("counterfactual", "all"):
        _main_counterfactual()
    if args.audit in ("plots", "all"):
        generate_audit_and_retrieval_plots()
