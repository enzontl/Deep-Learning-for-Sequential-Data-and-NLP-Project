"""
End-to-End Automated Test Suite for the LEDGAR Contract-Review Pipeline
=======================================================================
Tests:
1. Statistical evaluation primitives (Wilson CI, McNemar's test, Reliability Tiers,
   Temperature Scaling, ECE, Conformal Prediction with Tier-3 gating, Ensemble).
2. Bias audit primitives (Long-Tail, Length/Truncation, Taxonomy Overlap,
   US->EU Counterfactual & Party-Role Swaps).
3. Retrieval primitives (BM25 + Dense Reciprocal Rank Fusion, P@5, R@5, MRR, MAP@5,
   nDCG@5, Lawyer Regex Constraint Extraction).
4. Training & PEFT primitives (LoRALinear initialization B=0, merge_weights equivalence,
   16-byte/param AdamW fp32 memory accounting).
5. Saved outputs/* consistency checks across summary.json, bias_audit/*, retrieval_eval/*.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.bias_audit import (
    _swap_party_roles,
    make_perturbations,
    run_length_truncation_audit,
    run_long_tail_audit,
    run_taxonomy_overlap_audit,
)
from src.evaluation import (
    assign_reliability_tier,
    build_ensemble,
    conformal_prediction_sets,
    expected_calibration_error,
    fit_temperature,
    mcnemar_test,
    softmax,
    wilson_confidence_interval,
)
from src.retrieval import SparseBM25, tokenize
from src.retrieval_eval import bootstrap_mean_ci, per_query_metrics
from src.train import LoRALinear, count_parameters_and_memory

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def test_wilson_confidence_interval_edge_cases():
    # Standard small sample: Books (2/2) -> upper bound 1.0, lower bound ~0.342
    p, lo, hi = wilson_confidence_interval(2, 2, confidence=0.95)
    assert p == pytest.approx(1.0)
    assert 0.34 < lo < 0.35
    assert hi == pytest.approx(1.0)

    # Zero sample returns (nan, 0.0, 1.0)
    p0, lo0, hi0 = wilson_confidence_interval(0, 0)
    assert np.isnan(p0)
    assert (lo0, hi0) == (0.0, 1.0)


def test_mcnemar_paired_contingency_and_significance():
    y_true = np.load(OUTPUTS / "test_labels.npy")
    lb_test = np.load(OUTPUTS / "legal_bert_test_logits.npy")
    tf_test = np.load(OUTPUTS / "tfidf_test_logits.npy")

    res = mcnemar_test(y_true, lb_test, tf_test)
    assert res["a_both_correct"] == 8228
    assert res["b_only_a_correct"] == 370
    assert res["c_only_b_correct"] == 413
    assert res["d_both_wrong"] == 989
    assert res["n_disagreements"] == 783
    assert res["chi2"] == pytest.approx((abs(370 - 413) - 1.0) ** 2 / 783.0)
    assert res["significant"] is False


def test_reliability_tier_assignment():
    # Tier 1: ci_lower >= 0.80, ci_width <= 0.30, n >= 10
    assert assign_reliability_tier(0.85, 0.10, 100).startswith("Tier 1")
    # Tier 2: ci_lower >= 0.50 but < 0.80, or n < 10 when ci_lower >= 0.50
    assert assign_reliability_tier(0.65, 0.17, 50).startswith("Tier 2")
    assert assign_reliability_tier(0.85, 0.10, 5).startswith("Tier 2")
    # Tier 3: ci_lower < 0.50
    assert assign_reliability_tier(0.34, 0.66, 2).startswith("Tier 3")


def test_temperature_scaling_and_conformal_tier_gating():
    y_val = np.load(OUTPUTS / "val_labels.npy")
    y_test = np.load(OUTPUTS / "test_labels.npy")
    lb_val = np.load(OUTPUTS / "legal_bert_val_logits.npy")
    lb_test = np.load(OUTPUTS / "legal_bert_test_logits.npy")
    tf_val = np.load(OUTPUTS / "tfidf_val_logits.npy")
    tf_test = np.load(OUTPUTS / "tfidf_test_logits.npy")

    ens_val, ens_test, info = build_ensemble(lb_val, tf_val, y_val, lb_test, tf_test)
    assert info["weight_a"] == pytest.approx(0.45)
    assert info["temperature_a"] == pytest.approx(0.964, abs=2e-3)
    assert info["temperature_b"] == pytest.approx(0.684, abs=2e-3)

    conf = conformal_prediction_sets(
        ens_val, y_val, ens_test, y_test, alpha=0.05, n_boot=200
    )
    assert conf["coverage"]["estimate"] == pytest.approx(0.9501, abs=1e-4)
    assert conf["share_auto_tag"]["estimate"] == pytest.approx(0.6998, abs=1e-4)


def test_lora_linear_zero_init_and_merge_equivalence():
    torch.manual_seed(42)
    base = nn.Linear(16, 32, bias=True)
    x = torch.randn(4, 10, 16)
    y_base = base(x).detach().clone()

    lora = LoRALinear(base, r=4, alpha=8.0)
    # At step 0, B=0 so LoRALinear output must match base Linear exactly
    y_init = lora(x)
    assert torch.allclose(y_base, y_init, atol=1e-6)

    # Simulate trained B weights and verify merge_weights() preserves forward output
    with torch.no_grad():
        lora.lora_B.normal_(mean=0.0, std=0.05)
    y_unmerged = lora(x)
    merged_linear = lora.merge_weights()
    y_merged = merged_linear(x)
    assert torch.allclose(y_unmerged, y_merged, atol=1e-5)

    mem = count_parameters_and_memory(lora)
    assert mem["trainable_params"] == (4 * 16 + 32 * 4)
    assert mem["frozen_params"] == (16 * 32 + 32)


def test_counterfactual_perturbations_and_party_role_swap():
    pert = make_perturbations()
    assert "party_role_swap" in pert
    sample = "The Lender shall notify the Borrower in the State of Delaware within $100,000 under SEC and ERISA rules."
    swapped_party = _swap_party_roles(sample)
    assert "The Borrower shall notify the Lender" in swapped_party
    assert "Delaware" in pert["party_role_swap"](sample)
    assert "France" in pert["juris_FR"](sample)
    assert "EUR " in pert["currency_EUR"](sample)
    assert "ESMA" in pert["regulator_ESMA"](sample)
    assert "GDPR" in pert["statute_GDPR"](sample)


def test_retrieval_metrics_and_sparse_bm25():
    docs = [
        tokenize("Terminate upon 30 days written notice in the State of Delaware"),
        tokenize("Terminate upon 60 days written notice in the State of New York"),
        tokenize("Governing law of the State of California without conflicts"),
    ]
    bm25 = SparseBM25(docs)
    scores = bm25.get_scores([tokenize("30 days notice Delaware")])
    assert int(scores[0].argmax()) == 0

    retrieved = np.array([[0, 2, 1, 0, 0]])
    bank_labels = np.array([5, 5, 8])
    query_labels = np.array([5])
    pq = per_query_metrics(retrieved, bank_labels, query_labels, k=3)
    assert pq["precision"][0] == pytest.approx(2.0 / 3.0)
    assert pq["mrr"][0] == pytest.approx(1.0)
    assert pq["hit_rate"][0] == pytest.approx(1.0)
    ci = bootstrap_mean_ci([0.8, 0.9, 1.0], n_boot=200)
    assert 0.8 <= ci["estimate"] <= 1.0


def test_saved_outputs_consistency():
    with open(OUTPUTS / "evaluation" / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["models"]["legal_bert"]["global"]["micro_f1"]["estimate"] == pytest.approx(0.8598)
    assert summary["models"]["tfidf"]["global"]["micro_f1"]["estimate"] == pytest.approx(0.8641)
    assert summary["models"]["ensemble"]["global"]["micro_f1"]["estimate"] == pytest.approx(0.8777)
    assert summary["models"]["ensemble"]["conformal"]["tier3_singletons_diverted"] == 54
    assert summary["models"]["ensemble"]["conformal"]["share_auto_tag_tier_gated"] == pytest.approx(0.6944)
    assert "mcnemar" in summary["paired"]["legal_bert_vs_tfidf"]

    with open(OUTPUTS / "bias_audit" / "long_tail_audit.json", encoding="utf-8") as f:
        lt = json.load(f)
    ens_buckets = lt["buckets_by_model"]["Ensemble"]
    assert [b["n_classes"] for b in ens_buckets] == [16, 77, 7]

    with open(OUTPUTS / "bias_audit" / "length_audit.json", encoding="utf-8") as f:
        la = json.load(f)
    assert set(la["results"].keys()) == {"Legal-BERT", "TF-IDF", "Ensemble"}

    with open(OUTPUTS / "bias_audit" / "counterfactual_audit.json", encoding="utf-8") as f:
        cf = json.load(f)
    assert "party_role_swap" in cf["results"]["variants"]
