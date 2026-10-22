"""
Person 2 — Statistical Evaluation, Confidence Intervals & Calibration
=====================================================================
Consumes the saved predictions exported by Person 1 (`outputs/{legal_bert,tfidf}_{val,test}_logits.npy`,
`outputs/{val,test}_labels.npy`, `outputs/label_names.json`).

Implemented in this module:
1. Non-Parametric Bootstrap 95% Confidence Intervals (B=2,000, seed 42) for Micro / Macro /
   Weighted-F1 and Top-3 accuracy (`evaluate_bootstrap_confidence_intervals`).
2. Per-class F1 bootstrap CIs, Wilson CIs for recall / precision, and 3 Reliability Tiers
   (Auto-Tag / Assisted / Do Not Deploy) so that tiny test classes (e.g. 'Books') get wide CIs.
3. Paired bootstrap of metric differences between two models (`paired_bootstrap_difference`).
4. Calibration: ECE, temperature scaling fit on validation, reliability diagrams
   (`evaluate_calibration`) and a val-weighted calibrated ensemble (`build_ensemble`).
5. Split conformal prediction sets (95%) and auto-tag / shortlist / escalate routing
   (`conformal_prediction_sets`).
6. `run_evaluation()` runs everything and writes CSV / JSON / PNG to `outputs/evaluation/`:
   `uv run python -m src.evaluation`
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest, chi2, norm

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


class _SimpleSeries:
    def __init__(self, values: List[Any]):
        self._values = list(values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)

    @property
    def iloc(self):
        return self._values

    def map(self, mapping: Dict[Any, Any]) -> List[Any]:
        return [mapping[v] for v in self._values]

    def value_counts(self):
        counts: Dict[Any, int] = {}
        for v in self._values:
            counts[v] = counts.get(v, 0) + 1
        sorted_counts = dict(sorted(counts.items(), key=lambda kv: -kv[1]))

        class _VC:
            def to_dict(self_inner):
                return sorted_counts

        return _VC()


class _SimpleDataFrame:
    """Lightweight DataFrame fallback when pandas is not installed in the runtime."""

    def __init__(self, data: Any):
        if isinstance(data, list):
            self.rows: List[Dict[str, Any]] = [dict(r) for r in data]
        elif isinstance(data, dict):
            keys = list(data.keys())
            n = len(data[keys[0]]) if keys else 0
            self.rows = [{k: data[k][i] for k in keys} for i in range(n)]
        else:
            self.rows = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            return _SimpleSeries([r[key] for r in self.rows])
        if isinstance(key, (list, np.ndarray, _SimpleSeries)):
            mask = list(key)
            return _SimpleDataFrame([r for r, m in zip(self.rows, mask) if m])
        raise KeyError(key)

    def sort_values(self, by: str, ascending: bool = True) -> "_SimpleDataFrame":
        return _SimpleDataFrame(sorted(self.rows, key=lambda r: r[by], reverse=not ascending))

    def reset_index(self, drop: bool = False) -> "_SimpleDataFrame":
        return self

    def to_csv(self, path: Any, index: bool = False) -> None:
        if not self.rows:
            return
        fieldnames = list(self.rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: ("" if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()})


def _make_df(data: Any) -> Any:
    return pd.DataFrame(data) if pd is not None else _SimpleDataFrame(data)

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def wilson_confidence_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Wilson score confidence interval for a binomial proportion.
    Returns (p_hat, ci_lower, ci_upper).
    """
    if n <= 0:
        return float("nan"), 0.0, 1.0
    z = norm.ppf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


TIER_AUTO = "Tier 1 (Green - Auto-Tag Ready)"
TIER_ASSISTED = "Tier 2 (Amber - Human-in-the-Loop)"
TIER_BLOCKED = "Tier 3 (Red - Do Not Deploy)"


def assign_reliability_tier(
    f1_lower: float,
    f1_width: float,
    n_test: int,
    auto_lower: float = 0.80,
    auto_max_width: float = 0.30,
    blocked_lower: float = 0.50,
    min_n: int = 10,
) -> str:
    """
    Tier 3 (Red):   F1 CI lower bound < 0.50.
    Tier 1 (Green): F1 CI lower bound >= 0.80, CI width <= 0.30 and n_test >= min_n.
    Tier 2 (Amber): everything else.
    """
    if f1_lower < blocked_lower:
        return TIER_BLOCKED
    if f1_lower >= auto_lower and f1_width <= auto_max_width and n_test >= min_n:
        return TIER_AUTO
    return TIER_ASSISTED


def _class_counts(
    y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray, num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True positives, predicted counts and true counts per class for each resample (B, C)."""
    B = idx.shape[0]
    offset = (np.arange(B) * num_classes)[:, None]
    yt, yp = y_true[idx] + offset, y_pred[idx] + offset
    size = B * num_classes
    n_true = np.bincount(yt.ravel(), minlength=size).reshape(B, num_classes)
    n_pred = np.bincount(yp.ravel(), minlength=size).reshape(B, num_classes)
    tp = np.bincount(yt[yt == yp], minlength=size).reshape(B, num_classes)
    return tp, n_pred, n_true


def _f1_micro_macro_weighted(
    tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class F1 plus Micro / Macro / Weighted F1. Macro averages over classes that appear
    in the true labels or in the predictions (scikit-learn convention)."""
    denom = n_pred + n_true
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    present = denom > 0
    macro = (f1 * present).sum(axis=1) / present.sum(axis=1)
    micro = tp.sum(axis=1) / n_true.sum(axis=1)
    weighted = (f1 * n_true).sum(axis=1) / n_true.sum(axis=1)
    return f1, micro, macro, weighted


def evaluate_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    logits: np.ndarray,
    label_names: List[str],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Any]:
    """
    Global metrics (Micro/Macro/Weighted-F1, Top-3 accuracy) and per-class metrics with
    95% percentile-bootstrap CIs, plus a Reliability Tier for each class.
    When a class has 0 true positives on the test split (TP=0), non-parametric percentile
    bootstrap degenerates to [0.0, 0.0] (width 0.0); we apply a small-sample Wilson score
    upper bound on (support + predicted) so rare zero-hit classes reflect true binomial
    uncertainty rather than artificial zero width.
    Returns (global_results, per_class_dataframe).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    n, num_classes = logits.shape
    y_pred = logits.argmax(axis=1)
    top3 = (np.argpartition(-logits, 2, axis=1)[:, :3] == y_true[:, None]).any(axis=1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1_b, micro_b, macro_b, weighted_b = _f1_micro_macro_weighted(
        *_class_counts(y_true, y_pred, idx, num_classes)
    )
    top3_b = top3[idx].mean(axis=1)

    tp0, np0, nt0 = _class_counts(y_true, y_pred, np.arange(n)[None, :], num_classes)
    f1_0, micro_0, macro_0, weighted_0 = _f1_micro_macro_weighted(tp0, np0, nt0)

    a = (1 - confidence) / 2
    global_results: Dict[str, Any] = {"n_test": int(n), "n_boot": n_boot, "confidence": confidence}
    for name, point, boot in (
        ("micro_f1", micro_0[0], micro_b),
        ("macro_f1", macro_0[0], macro_b),
        ("weighted_f1", weighted_0[0], weighted_b),
        ("top3_accuracy", top3.mean(), top3_b),
    ):
        lo, hi = np.quantile(boot, [a, 1 - a])
        global_results[name] = {"estimate": float(point), "ci_lower": float(lo), "ci_upper": float(hi)}

    f1_lo, f1_hi = np.quantile(f1_b, [a, 1 - a], axis=0)
    rows = []
    for c in range(num_classes):
        support, tp_c, pred_c = int(nt0[0, c]), int(tp0[0, c]), int(np0[0, c])
        recall, rec_lo, rec_hi = wilson_confidence_interval(tp_c, support, confidence)
        precision, prec_lo, prec_hi = wilson_confidence_interval(tp_c, pred_c, confidence)
        ci_lo = float(f1_lo[c])
        ci_hi = float(f1_hi[c])
        if tp_c == 0 and (support + pred_c) > 0:
            # Avoid degenerate [0.0, 0.0] bootstrap CI at the TP=0 boundary on small classes
            _, _, wilson_hi = wilson_confidence_interval(0, support + pred_c, confidence)
            ci_hi = max(ci_hi, float(wilson_hi))
        elif tp_c == support and tp_c == pred_c and support > 0:
            # Avoid degenerate [1.0, 1.0] bootstrap CI at the F1=1.0 boundary
            ci_lo = min(ci_lo, float(rec_lo))
        width = float(ci_hi - ci_lo)
        rows.append(
            {
                "class_id": c,
                "class_name": label_names[c],
                "n_test": support,
                "f1": float(f1_0[0, c]),
                "f1_ci_lower": ci_lo,
                "f1_ci_upper": ci_hi,
                "f1_ci_width": width,
                "recall": recall,
                "recall_wilson_lower": rec_lo,
                "recall_wilson_upper": rec_hi,
                "precision": precision,
                "precision_wilson_lower": prec_lo,
                "precision_wilson_upper": prec_hi,
                "tier": assign_reliability_tier(ci_lo, width, support),
            }
        )
    per_class = _make_df(rows).sort_values("f1_ci_lower", ascending=False).reset_index(drop=True)
    global_results["tier_counts"] = per_class["tier"].value_counts().to_dict()
    return global_results, per_class


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax of logits / temperature."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """
    ECE with equal-width confidence bins (Guo et al., 2017): weighted mean over bins of
    |accuracy - mean confidence|, confidence being the max softmax probability.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    conf_sum = np.bincount(bins, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(bins, weights=correct, minlength=n_bins)
    nonempty = counts > 0
    gap = np.abs(acc_sum[nonempty] - conf_sum[nonempty])  # = count * |acc - conf|
    return float(gap.sum() / len(y_true))


def fit_temperature(
    val_logits: np.ndarray, val_labels: np.ndarray, bounds: Tuple[float, float] = (0.05, 20.0)
) -> float:
    """
    Temperature T > 0 minimising the negative log-likelihood on the validation set.
    Dividing logits by T never changes the predicted class, only the confidence.
    """
    val_labels = np.asarray(val_labels, dtype=np.int64)
    rows = np.arange(len(val_labels))

    def nll(log_t: float) -> float:
        z = val_logits / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-log_probs[rows, val_labels].mean())

    res = minimize_scalar(nll, bounds=tuple(np.log(bounds)), method="bounded")
    return float(np.exp(res.x))


def _confidence_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int):
    """Per-example (bin, confidence, correct) used by the ECE and the reliability diagram."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return bins, confidence, correct


def _ece_bootstrap(
    bins: np.ndarray, confidence: np.ndarray, correct: np.ndarray, idx: np.ndarray, n_bins: int
) -> np.ndarray:
    """ECE of every resample (B,), same bincount-with-offset trick as _class_counts."""
    B, n = idx.shape
    flat = (bins[idx] + (np.arange(B) * n_bins)[:, None]).ravel()
    size = B * n_bins
    conf_sum = np.bincount(flat, weights=confidence[idx].ravel(), minlength=size).reshape(B, n_bins)
    acc_sum = np.bincount(flat, weights=correct[idx].ravel(), minlength=size).reshape(B, n_bins)
    return np.abs(acc_sum - conf_sum).sum(axis=1) / n


def _reliability_table(bins, confidence, correct, n_bins: int) -> Any:
    counts = np.bincount(bins, minlength=n_bins)
    conf_sum = np.bincount(bins, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(bins, weights=correct, minlength=n_bins)
    safe = np.maximum(counts, 1)
    return _make_df(
        {
            "bin_lower": np.arange(n_bins) / n_bins,
            "bin_upper": (np.arange(n_bins) + 1) / n_bins,
            "count": counts,
            "mean_confidence": np.where(counts > 0, conf_sum / safe, np.nan),
            "accuracy": np.where(counts > 0, acc_sum / safe, np.nan),
        }
    )


def evaluate_calibration(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    n_bins: int = 15,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Calibration of one model: temperature fit on val, then ECE / NLL on test before (T=1)
    and after scaling, with percentile-bootstrap CIs and a paired CI on the ECE change.
    ECE is slightly biased upwards on small samples; with n=10k the effect is negligible.
    """
    test_labels = np.asarray(test_labels, dtype=np.int64)
    n = len(test_labels)
    T = fit_temperature(val_logits, val_labels)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    a = (1 - confidence) / 2
    rows = np.arange(n)

    results: Dict[str, Any] = {
        "temperature": T, "n_test": int(n), "n_bins": n_bins, "n_boot": n_boot, "confidence": confidence,
    }
    boots = {}
    for tag, temp in (("before", 1.0), ("after", T)):
        probs = softmax(test_logits, temp)
        bins, conf, correct = _confidence_bins(probs, test_labels, n_bins)
        boot = _ece_bootstrap(bins, conf, correct, idx, n_bins)
        boots[tag] = boot
        point = float(np.abs(
            np.bincount(bins, weights=correct, minlength=n_bins)
            - np.bincount(bins, weights=conf, minlength=n_bins)
        ).sum() / n)
        lo, hi = np.quantile(boot, [a, 1 - a])
        results[f"ece_{tag}"] = {"estimate": point, "ci_lower": float(lo), "ci_upper": float(hi)}
        results[f"nll_{tag}"] = float(-np.log(probs[rows, test_labels] + 1e-12).mean())
        results[f"reliability_bins_{tag}"] = _reliability_table(bins, conf, correct, n_bins)

    delta_b = boots["after"] - boots["before"]
    lo, hi = np.quantile(delta_b, [a, 1 - a])
    results["delta_ece"] = {
        "estimate": results["ece_after"]["estimate"] - results["ece_before"]["estimate"],
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }
    return results


TIER_COLORS = {TIER_AUTO: "#2e8b57", TIER_ASSISTED: "#e69f00", TIER_BLOCKED: "#c0392b"}


def plot_forest_per_class(per_class: Any, path: str, title: str = "") -> None:
    """Forest plot of per-class F1 with 95% bootstrap/Wilson CIs, coloured by reliability tier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = per_class.sort_values("f1_ci_lower", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    colors = df["tier"].map(TIER_COLORS)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(df))))
    ax.hlines(y, np.asarray(df["f1_ci_lower"]), np.asarray(df["f1_ci_upper"]), colors=colors, linewidth=2)
    ax.scatter(np.asarray(df["f1"]), y, c=colors, s=18, zorder=3)
    ax.axvline(0.50, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0.80, color="grey", linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} (n={k})" for n, k in zip(df["class_name"], df["n_test"])], fontsize=6)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-1, len(df))
    ax.set_xlabel("F1 (95% bootstrap / Wilson CI)")
    ax.set_title(title or "Per-class F1 with 95% confidence intervals")
    ax.legend(
        handles=[Line2D([0], [0], color=c, lw=2, label=t) for t, c in TIER_COLORS.items()],
        loc="upper left",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(
    calibration: Dict[str, Any], path: str, model_name: str = "", min_count: int = 30
) -> None:
    """Reliability diagrams before / after temperature scaling. Bins with fewer than
    `min_count` examples are greyed out (unreliable estimates)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, tag, label in zip(axes, ("before", "after"), ("Before scaling (T=1)", "After scaling")):
        table = calibration[f"reliability_bins_{tag}"]
        table = table[np.asarray(table["count"]) > 0]
        bin_lo = np.asarray(table["bin_lower"], dtype=float)
        bin_hi = np.asarray(table["bin_upper"], dtype=float)
        counts = np.asarray(table["count"], dtype=int)
        acc = np.asarray(table["accuracy"], dtype=float)
        width = float(bin_hi[0] - bin_lo[0])
        centers = (bin_lo + bin_hi) / 2
        colors = np.where(counts >= min_count, "#2b6cb0", "#b0b0b0")
        ax.bar(centers, acc, width=width * 0.95, color=colors, edgecolor="white")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ece = calibration[f"ece_{tag}"]["estimate"]
        temp = calibration["temperature"] if tag == "after" else 1.0
        ax.set_title(f"{label}\nT={temp:.3f}, ECE={100 * ece:.2f}%")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle(f"Reliability diagram{' - ' + model_name if model_name else ''} (grey: n<{min_count})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_ensemble(
    val_logits_a: np.ndarray,
    val_logits_b: np.ndarray,
    val_labels: np.ndarray,
    test_logits_a: np.ndarray,
    test_logits_b: np.ndarray,
    weight_grid: np.ndarray = np.linspace(0.0, 1.0, 21),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Weighted average of two models' temperature-calibrated probabilities, with the weight
    on model A chosen on validation by minimum NLL. Returns (val_logits, test_logits, info);
    the returned "logits" are log-probabilities, so argmax and softmax behave as usual.
    """
    val_labels = np.asarray(val_labels, dtype=np.int64)
    t_a = fit_temperature(val_logits_a, val_labels)
    t_b = fit_temperature(val_logits_b, val_labels)
    val_a, val_b = softmax(val_logits_a, t_a), softmax(val_logits_b, t_b)
    test_a, test_b = softmax(test_logits_a, t_a), softmax(test_logits_b, t_b)

    rows = np.arange(len(val_labels))
    nlls = [
        float(-np.log(w * val_a[rows, val_labels] + (1 - w) * val_b[rows, val_labels] + 1e-12).mean())
        for w in weight_grid
    ]
    w = float(weight_grid[int(np.argmin(nlls))])

    def to_logits(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        return np.log(w * p_a + (1 - w) * p_b + 1e-12).astype(np.float32)

    info = {"weight_a": w, "temperature_a": t_a, "temperature_b": t_b, "val_nll": float(min(nlls))}
    return to_logits(val_a, val_b), to_logits(test_a, test_b), info


def conformal_prediction_sets(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    per_class: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Split conformal prediction sets (score = 1 - p(true class), on temperature-scaled probabilities).
    The threshold is calibrated on validation; a test clause gets every class with
    p >= 1 - q_hat, which covers the true class with probability >= 1 - alpha on average
    (marginal guarantee only: no guarantee for rare classes such as 'Books').
    Routing: 1 class = auto-tag, 2-3 classes = shortlist, 0 or 4+ classes = escalate.
    When `per_class` is provided, also reports the effect of hard-blocking singletons predicted
    in a Tier-3 (Red - Do Not Deploy) category.
    """
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    T = fit_temperature(val_logits, val_labels)

    val_probs = softmax(val_logits, T)
    scores = 1.0 - val_probs[np.arange(len(val_labels)), val_labels]
    n_val = len(scores)
    level = min(1.0, np.ceil((n_val + 1) * (1 - alpha)) / n_val)
    q_hat = float(np.quantile(scores, level, method="higher"))

    test_probs = softmax(test_logits, T)
    n = len(test_labels)
    sets = test_probs >= 1.0 - q_hat
    size = sets.sum(axis=1)
    covered = sets[np.arange(n), test_labels].astype(float)
    auto = (size == 1).astype(float)
    shortlist = ((size >= 2) & (size <= 3)).astype(float)
    escalate = 1.0 - auto - shortlist
    pred_top1 = test_probs.argmax(axis=1)
    auto_correct = auto * (pred_top1 == test_labels)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    a = (1 - confidence) / 2

    def ci(point: float, boot: np.ndarray) -> Dict[str, float]:
        lo, hi = np.quantile(boot, [a, 1 - a])
        return {"estimate": float(point), "ci_lower": float(lo), "ci_upper": float(hi)}

    results: Dict[str, Any] = {
        "alpha": alpha, "temperature": T, "q_hat": q_hat, "n_val": int(n_val), "n_test": int(n),
        "coverage": ci(covered.mean(), covered[idx].mean(axis=1)),
        "mean_set_size": ci(size.mean(), size[idx].mean(axis=1)),
        "share_auto_tag": ci(auto.mean(), auto[idx].mean(axis=1)),
        "share_shortlist": ci(shortlist.mean(), shortlist[idx].mean(axis=1)),
        "share_escalate": ci(escalate.mean(), escalate[idx].mean(axis=1)),
    }
    n_auto = auto.sum()
    auto_boot = auto[idx].sum(axis=1)
    results["auto_tag_accuracy"] = ci(
        auto_correct.sum() / max(n_auto, 1),
        auto_correct[idx].sum(axis=1) / np.maximum(auto_boot, 1),
    )
    if per_class is not None:
        cids = list(per_class["class_id"])
        tiers = list(per_class["tier"])
        tier3_ids = {int(c) for c, t in zip(cids, tiers) if str(t).startswith("Tier 3")}
        is_t3_pred = np.isin(pred_top1, list(tier3_ids))
        t3_singletons = (size == 1) & is_t3_pred
        auto_gated = ((size == 1) & ~is_t3_pred).astype(float)
        auto_gated_correct = auto_gated * (pred_top1 == test_labels)
        n_auto_g = auto_gated.sum()
        results["tier3_singletons_diverted"] = int(t3_singletons.sum())
        results["share_auto_tag_tier_gated"] = float(auto_gated.mean())
        results["auto_tag_accuracy_tier_gated"] = float(auto_gated_correct.sum() / max(n_auto_g, 1))
        results["share_escalate_tier_gated"] = float(escalate.mean() + t3_singletons.mean())
    return results


def mcnemar_test(
    y_true: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    McNemar's paired test on classification accuracy disagreements:
      - a: both A and B correct
      - b: only A correct, B wrong
      - c: A wrong, only B correct
      - d: both A and B wrong
    Only disagreements count: acc_A - acc_B = (b - c) / n.
    Under H0, b ~ Binomial(b + c, 1/2), and chi2 = (|b - c| - 1)^2 / (b + c) ~ chi2(df=1)
    with continuity correction (or exact binomial test when b + c < 25).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    ok_a = logits_a.argmax(axis=1) == y_true
    ok_b = logits_b.argmax(axis=1) == y_true
    a = int((ok_a & ok_b).sum())
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    d = int((~ok_a & ~ok_b).sum())
    n_disagree = b + c
    if n_disagree == 0:
        chi2_stat, p_val = 0.0, 1.0
    elif n_disagree < 25:
        chi2_stat = float((abs(b - c) - 1.0) ** 2 / n_disagree)
        p_val = float(binomtest(b, n_disagree, 0.5).pvalue)
    else:
        chi2_stat = float((abs(b - c) - 1.0) ** 2 / n_disagree)
        p_val = float(chi2.sf(chi2_stat, df=1))
    return {
        "a_both_correct": a,
        "b_only_a_correct": b,
        "c_only_b_correct": c,
        "d_both_wrong": d,
        "n_disagreements": n_disagree,
        "chi2": chi2_stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha),
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired bootstrap of the metric difference (A - B): both models are scored on the SAME
    resampled test indices. A difference is significant when its CI excludes 0.
    Also includes McNemar's test on accuracy disagreements.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    n, num_classes = logits_a.shape
    preds = (logits_a.argmax(axis=1), logits_b.argmax(axis=1))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    full = np.arange(n)[None, :]

    a = (1 - confidence) / 2
    results: Dict[str, Any] = {
        "n_test": int(n),
        "n_boot": n_boot,
        "confidence": confidence,
        "mcnemar": mcnemar_test(y_true, logits_a, logits_b, alpha=1.0 - confidence),
    }
    boot, point = [], []
    for y_pred in preds:
        _, mi_b, ma_b, we_b = _f1_micro_macro_weighted(*_class_counts(y_true, y_pred, idx, num_classes))
        _, mi_0, ma_0, we_0 = _f1_micro_macro_weighted(*_class_counts(y_true, y_pred, full, num_classes))
        boot.append((mi_b, ma_b, we_b))
        point.append((mi_0[0], ma_0[0], we_0[0]))

    for k, name in enumerate(("micro_f1", "macro_f1", "weighted_f1")):
        delta_b = boot[0][k] - boot[1][k]
        lo, hi = np.quantile(delta_b, [a, 1 - a])
        results[name] = {
            "estimate_a": float(point[0][k]),
            "estimate_b": float(point[1][k]),
            "delta": float(point[0][k] - point[1][k]),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "significant": bool(lo > 0 or hi < 0),
        }
    return results


def run_evaluation(outputs_dir: Path = OUTPUTS_DIR, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Full Person 2 pipeline on the saved logits: bootstrap CIs and reliability tiers, calibration,
    ensemble, paired model comparisons. Writes CSV / JSON / PNG files to `out_dir`
    (default: outputs/evaluation) and returns the summary dict.
    """
    out_dir = Path(out_dir) if out_dir is not None else outputs_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_val = np.load(outputs_dir / "val_labels.npy")
    y_test = np.load(outputs_dir / "test_labels.npy")
    with open(outputs_dir / "label_names.json", encoding="utf-8") as f:
        label_names = json.load(f)

    logits = {
        m: (np.load(outputs_dir / f"{m}_val_logits.npy"), np.load(outputs_dir / f"{m}_test_logits.npy"))
        for m in ("legal_bert", "tfidf")
    }
    ens_val, ens_test, ens_info = build_ensemble(
        logits["legal_bert"][0], logits["tfidf"][0], y_val, logits["legal_bert"][1], logits["tfidf"][1]
    )
    logits["ensemble"] = (ens_val, ens_test)

    summary: Dict[str, Any] = {"ensemble": ens_info, "models": {}, "paired": {}}
    for name, (val_l, test_l) in logits.items():
        global_res, per_class = evaluate_bootstrap_confidence_intervals(y_test, test_l, label_names)
        calibration = evaluate_calibration(val_l, y_val, test_l, y_test)
        per_class.to_csv(out_dir / f"{name}_per_class.csv", index=False)
        plot_forest_per_class(per_class, str(out_dir / f"{name}_forest.png"), title=name)
        plot_reliability_diagram(calibration, str(out_dir / f"{name}_reliability.png"), model_name=name)
        for tag in ("before", "after"):
            calibration.pop(f"reliability_bins_{tag}").to_csv(
                out_dir / f"{name}_reliability_{tag}.csv", index=False
            )
        conformal = conformal_prediction_sets(val_l, y_val, test_l, y_test, per_class=per_class)
        summary["models"][name] = {"global": global_res, "calibration": calibration, "conformal": conformal}

    for a, b in (("legal_bert", "tfidf"), ("ensemble", "tfidf"), ("ensemble", "legal_bert")):
        summary["paired"][f"{a}_vs_{b}"] = paired_bootstrap_difference(y_test, logits[a][1], logits[b][1])

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    run_evaluation()
