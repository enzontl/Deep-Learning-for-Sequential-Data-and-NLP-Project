"""
Person 2 — Statistical Evaluation, Confidence Intervals & Calibration (TODO)
============================================================================
Consumes the saved predictions (`outputs/val_logits.npy`, `outputs/test_logits.npy`,
`outputs/val_labels.npy`, `outputs/test_labels.npy`) exported by Person 1.

Tasks to implement in this module:
1. Non-Parametric Bootstrap 95% Confidence Intervals (e.g., B=1,000 resamples) for:
   - Global Micro-F1 / Accuracy
   - Global Macro-F1
   - Top-3 Accuracy (to account for compound clauses)
2. Per-Class 95% Confidence Intervals (Bootstrap F1 & Wilson Binomial Recall/Precision)
   - Show how small test sample sizes (e.g., 'Books' with n_test < 10) produce wide CIs.
   - Group the 100 classes into 3 Reliability Tiers (Auto-Tag / Assisted / Do Not Deploy).
3. Probability Calibration (Temperature Scaling & Expected Calibration Error - ECE)
   and/or Split Conformal Prediction sets.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def wilson_confidence_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    TODO (Person 2): Compute the Wilson score 95% confidence interval for a binomial proportion.
    Returns (p_hat, ci_lower, ci_upper).
    """
    raise NotImplementedError("TODO (Person 2): Implement Wilson score confidence interval.")


def evaluate_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    logits: np.ndarray,
    label_names: List[str],
    n_boot: int = 1000,
    confidence: float = 0.95,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    TODO (Person 2): Compute global and per-class metrics with 95% Bootstrap CIs
    and assign each class to a Reliability Tier.
    """
    raise NotImplementedError("TODO (Person 2): Implement Bootstrap 95% CI evaluation.")
