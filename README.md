# Contract-Review Assistant — NLP Core, Hybrid Retrieval & Bias Audit

**Course:** Deep Learning for Sequential Data and NLP (Final Project)  
**Authors:** **Namie Pajot**, **Enzo Natali**, and **Guillaume RABEAU**

---

## 1. Overview

This repository is a student final project developed by **Namie Pajot**, **Enzo Natali**, and **Guillaume RABEAU**. It implements, evaluates, and audits the **NLP core of a legal contract-review assistant** designed for corporate due diligence (mergers, financings, supplier audits).

Given a contract clause or a lawyer's natural-language query, the system performs three core tasks:
1. **Clause Classification:** Assigns every contract clause to one of **100 legal clause types** (`Governing Laws`, `Terminations`, `Indemnifications`, `Confidentiality`, etc.) by combining a fine-tuned domain transformer (**Legal-BERT**) with a full-text lexical classifier (**TF-IDF + Logistic Regression**) in a **temperature-scaled soft-voting ensemble**.
2. **Precedent Clause Retrieval:** Searches a **60,000-clause precedent bank** using a **Hybrid Reciprocal Rank Fusion (RRF)** engine that combines dense semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`), sparse lexical matching (`BM25`), and category filtering.
3. **Statistical Uncertainty & Bias Auditing:** Rather than acting as a black box that reports single point estimates, every prediction and evaluation metric is backed by **95% Bootstrap or Wilson confidence intervals**, **Split Conformal Prediction sets** for human-in-the-loop routing, and a **four-part failure-mode and US-to-EU domain-shift audit**.

---

## 2. Dataset & Engineering Challenges

All experiments are conducted on the **LEDGAR** benchmark (`coastalcph/lex_glue`, configuration `ledgar`), extracted from public material contracts filed with the US Securities and Exchange Commission (SEC):
- **Dataset Splits:** `60,000` training clauses, `10,000` validation clauses, and `10,000` held-out test clauses across **100 clause categories**.
- **Extreme Zipfian Imbalance ($138\times$):** The most frequent category (`Governing Laws`) has **3,167** training clauses, whereas the rarest (`Books`) has only **23** training clauses, **0** validation clauses, and **2** test clauses. Overall, **16 Head classes** ($n_{\text{train}} \ge 1,000$) coexist with **77 Medium classes** ($150 \le n_{\text{train}} < 1,000$) and **7 Tail classes** ($n_{\text{train}} < 150$).
- **US Training Data vs. European Deployment Context:** While the assistant targets a European legal-tech setting, LEDGAR consists entirely of English-language US Common Law contracts referencing US states (`Delaware`, `New York`), US regulators (`SEC`), and US federal statutes (`ERISA`, `Waiver Of Jury Trials`).

---

## 3. Repository Structure

```text
├── pyproject.toml                   # Project metadata and dependencies (uv / Python >= 3.10)
├── uv.lock                          # Deterministic lockfile for reproducible environments
├── requirements.txt                 # Pinned dependencies for standard pip installations
├── train_colab_pro.ipynb            # End-to-end GPU training notebook (Legal-BERT + TF-IDF)
├── presentation/
│   ├── deck.html                    # Self-contained interactive 1080p60 presentation deck
│   └── deck.mp4                     # Master 1080p60 video animation of the presentation
├── src/
│   ├── data.py                      # LEDGAR loader, class distribution & tokenizer fertility audit
│   ├── train.py                     # TF-IDF baseline + PyTorch GPU fine-tuning for Legal-BERT
│   ├── evaluation.py                # Bootstrap/Wilson 95% CIs, calibration, ensemble & conformal sets
│   ├── retrieval.py                 # Hybrid retrieval engine (MiniLM-L6-v2 + SparseBM25 + RRF)
│   ├── retrieval_eval.py            # Protocol A: Clause-to-clause precedent retrieval benchmark
│   ├── retrieval_lawyer_eval.py     # Protocol B: 20 lawyer constraint queries benchmark
│   └── bias_audit.py                # Long-tail, length truncation, taxonomy & US->EU counterfactual audits
└── outputs/
    ├── class_distribution.csv       # Per-split sample counts across all 100 clause types
    ├── tokenizer_comparison.json    # Subword fertility & truncation rates (Legal-BERT, BERT, CamemBERT)
    ├── *_val_logits.npy             # Validation logits (10,000 x 100) for calibration & conformal threshold
    ├── *_test_logits.npy            # Held-out test logits (10,000 x 100) for evaluation & audits
    ├── evaluation/                  # Global & per-class CIs, reliability diagrams & forest plots
    ├── retrieval_eval/              # Retrieval benchmarks (CSV/JSON) & comparison plots
    └── bias_audit/                  # Long-tail, length, taxonomy & counterfactual audit outputs & plots
```

---

## 4. Methodology & Key Results

### 4.1 Tokenizer Audit & Model Selection (`src/data.py`, `src/train.py`)

Before fine-tuning, `src/data.py` audits three candidate tokenizers across the corpus (`outputs/tokenizer_comparison.json`):

| Pretrained Model | Pretraining Domain | Mean Tokens / Word | Median Clause Length | Clauses Truncated at 256 Tokens | Clauses Truncated at 512 Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`nlpaueb/legal-bert-base-uncased`** | English US & EU Legal Text | **1.200** | **105.0** | **12.2%** | **1.6%** |
| `bert-base-uncased` | General English Wikipedia/Books | 1.257 | 109.5 | 13.8% | 1.9% |
| `camembert-base` | French Oscar Corpus | 1.924 | 169.0 | 28.4% | 7.2% |

Because `camembert-base` is trained on French, its SentencePiece vocabulary over-fragments English legal terminology by **+60%** (`1.924` vs. `1.200` subwords per word), causing **28.4%** of clauses to be truncated at our 256-token window. We therefore fine-tune **`nlpaueb/legal-bert-base-uncased`** (3 epochs, `AdamW`, learning rate $2\times 10^{-5}$, linear warmup + cosine decay, mixed-precision `fp16`) alongside a full-text **`TF-IDF + Logistic Regression`** baseline (`50,000` unigram/bigram features, `sublinear_tf=True`) and combine their calibrated probabilities into a **Soft-Voting Ensemble** ($0.45\,\text{Legal-BERT} + 0.55\,\text{TF-IDF}$).

### 4.2 Classification Performance & Paired Bootstrap Tests (`src/evaluation.py`)

All global metrics on the `10,000`-clause test set are computed with **2,000 non-parametric bootstrap resamples** (`outputs/evaluation/summary.json`):

| Model | Accuracy / Micro-F1 (95% CI) | Macro-F1 (95% CI) | Weighted-F1 (95% CI) | Top-3 Accuracy (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Legal-BERT (Fine-Tuned)** | 85.98% `[85.29%, 86.66%]` | 75.89% `[74.81%, 77.11%]` | 85.75% `[85.05%, 86.44%]` | 95.88% `[95.48%, 96.26%]` |
| **TF-IDF + Logistic Regression** | 86.41% `[85.71%, 87.08%]` | 79.69% `[78.37%, 80.68%]` | 86.26% `[85.56%, 86.93%]` | 96.02% `[95.63%, 96.40%]` |
| **Calibrated Ensemble** | **87.77%** `[87.12%, 88.40%]` | **80.62%** `[79.00%, 81.76%]` | **87.58%** `[86.92%, 88.22%]` | **96.61%** `[96.25%, 96.95%]` |

- **Paired Bootstrap Hypothesis Testing:** Evaluated on identical bootstrap draws, the **Calibrated Ensemble** outperforms standalone `Legal-BERT` by **+4.74 points on Macro-F1** `[+3.87%, +5.56%]` and **+1.79 points on Accuracy** `[+1.43%, +2.15%]` (both statistically significant). Against `TF-IDF`, the Ensemble achieves a statistically significant **+1.36 point gain in Accuracy** `[+1.06%, +1.67%]`, while the **+0.93 point difference in Macro-F1** `[-0.14%, +1.85%]` crosses zero due to high variance on ultra-rare classes and is reported transparently as statistically unproven.

### 4.3 Probability Calibration & Split Conformal Routing (`src/evaluation.py`)

To ensure predicted probabilities reflect true empirical accuracy before routing clauses to humans:
1. **Temperature Scaling (Validation Set):** Raw `TF-IDF + Logistic Regression` probabilities are severely under-confident (**ECE = 10.99%**). Fitting a single temperature parameter $T = 0.68$ on validation reduces its test ECE to **2.70%**, while `Legal-BERT` ($T=0.96$, **ECE = 1.65%**) and the `Calibrated Ensemble` ($T=0.92$, **ECE = 1.98%**) are already well-calibrated.
2. **Split Conformal Prediction ($\alpha = 0.05$):** Using the validation set to calibrate the non-conformity quantile $\hat{q} = 0.9477$, we construct variable-size prediction sets $C(x) = \{y : \hat{p}(y \mid x) \ge 1 - \hat{q}\}$ on the test set:
   - **Empirical Test Coverage:** **95.01%** `[94.56%, 95.43%]` (matching the 95% target guarantee) with a mean set size of **1.43 labels**.
   - **Green — Auto-Tag (`|C(x)| = 1`):** **70.0%** of all test clauses are tagged automatically with **96.57% accuracy**.
   - **Amber — Paralegal Shortlist (`|C(x)| \in \{2, 3\}`):** **27.3%** of clauses present a 2- or 3-label shortlist for quick human confirmation.
   - **Red — Senior Lawyer Escalation (`|C(x)| \ge 4`):** **2.7%** of highly ambiguous clauses are escalated directly to a senior lawyer.
3. **Per-Class Reliability Tiers:** Using 95% Wilson/Bootstrap lower bounds on each class's F1 score, the 100 clause categories partition into **52 Tier 1 (Green)** classes ($\text{CI}_{\text{lower}} \ge 80\%$, certified for autonomous tagging), **33 Tier 2 (Amber)** classes (human-assisted confirmation), and **15 Tier 3 (Red)** classes ($\text{CI}_{\text{lower}} < 50\%$ or $n_{\text{test}} < 10$, blocked from autonomous tagging).

### 4.4 Hybrid Precedent Retrieval (`src/retrieval.py`, `src/retrieval_eval.py`, `src/retrieval_lawyer_eval.py`)

`ClauseBankRetriever` indexes all `60,000` training clauses using both dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and sparse lexical scores (`SparseBM25`), combined via **Reciprocal Rank Fusion (RRF, $k=60$)** and optional category pre-filtering. We evaluate retrieval under two complementary protocols (`outputs/retrieval_eval/`):

- **Protocol A — Clause-to-Clause Precedent Search ($n=1,000$ test queries vs. 60,000-clause bank):**
  - `Dense (MiniLM-L6-v2)`: Precision@5 = **74.9%** `[72.9%, 77.1%]`, MRR@5 = **85.6%** `[83.6%, 87.5%]`, HitRate@5 = **91.0%**.
  - `BM25 (Lexical)`: Precision@5 = **75.7%** `[73.6%, 77.9%]`, MRR@5 = **85.2%** `[83.1%, 87.2%]`, HitRate@5 = **90.4%**.
  - `Hybrid (RRF)`: Precision@5 = **76.9%** `[74.9%, 79.0%]`, MRR@5 = **85.7%** `[83.7%, 87.8%]`, HitRate@5 = **91.1%** — achieving a statistically significant **+2.0 point** `[+1.0%, +3.1%]` paired bootstrap improvement over Pure Dense (and **+2.0 points** after removing exact boilerplate duplicates).
- **Protocol B — Lawyer Natural-Language Constraint Queries ($n=20$ queries with exact numerical/jurisdictional constraints, e.g., *"termination clauses with a 30-day notice period"*):**
  - `Pure Dense`: Precision@5 drops to **45.0%** `[30.0%, 61.0%]` even though its category-only Precision@5 is **81.0%**—demonstrating that dense embeddings capture the general topic (*termination*) but confuse exact numbers (`30` vs. `60` days).
  - `Hybrid (RRF)`: Precision@5 rises to **61.0%** `[46.0%, 75.0%]` (**+16.0 pts** over Pure Dense).
  - `Hybrid + Category Filter (Shipped)`: Precision@5 reaches **68.0%** `[54.0%, 82.0%]` (**+23.0 pts** `[+14.0%, +31.0%]` over Pure Dense, MRR@5 = **88.3%**, HitRate@5 = **95.0%**), statistically tied with `BM25` (**69.0%**, paired difference `-1.0 pt` `[-12.0%, +10.0%]`) while retaining semantic recall on paraphrased clauses.

### 4.5 Four-Part Bias & Failure-Mode Audit (`src/bias_audit.py`)

1. **Long-Tail Sample-Size Bias (`long_tail_audit.json`):**
   - On **Head classes** ($n_{\text{train}} \ge 1,000$, 16 classes), `Legal-BERT` and the `Ensemble` achieve **92.0%** and **92.2%** mean F1 with a narrow **5.1-point** mean 95% CI width.
   - On **Tail classes** ($n_{\text{train}} < 150$, 7 classes), standalone `Legal-BERT` drops to **34.6%** mean F1 (scoring **0.0% F1** on 5 classes: `Books`, `Assigns`, `Qualifications`, `Jurisdictions`, `Applicable Laws`), whereas the `Ensemble` recovers to **54.3%** mean F1 (**+19.7 pts**). Crucially, the mean 95% CI width on Tail classes widens to **52.2 points** (e.g., `Assigns` with $n_{\text{test}}=4$ spans `[0.0%, 49.0%]`), proving that rare classes cannot be certified without more test data.
2. **Clause Length & 256-Token Truncation Audit (`length_audit.json`):**
   - Across `<80`, `80–160`, `160–260`, and `>260` words, `Legal-BERT` and `TF-IDF` perform identically on short clauses (`86.76%` vs. `86.74%` on `<80` words). On clauses exceeding **260 words** ($n=717$, where `Legal-BERT` truncates after 256 subwords), full-text `TF-IDF` reaches **89.12%** `[86.63%, 91.20%]` vs. **86.89%** `[84.22%, 89.16%]` for `Legal-BERT` (**+2.23 pts**).
3. **Taxonomy Overlap & Compound Clauses (`taxonomy_audit.json`):**
   - Just **10 confusion pairs** out of 4,950 possible class pairs account for **25.5% of all Legal-BERT errors** (`Tax Withholdings` $\leftrightarrow$ `Withholdings`: 60 errors; `Applicable Laws` $\rightarrow$ `Governing Laws`: 42 unidirectional errors; `Defined Terms` $\leftrightarrow$ `Definitions`: 41 errors; `No Waivers` $\leftrightarrow$ `Waivers`: 31 errors). The `Ensemble` reduces total test errors from `1,402` to `1,223`, while conformal shortlists catch ambiguous compound clauses.
4. **US SEC $\rightarrow$ European Law Counterfactual Audit (`counterfactual_audit.json`):**
   - Perturbing real test clauses by swapping US entities for European equivalents reveals where US training priors break down:
     - **US Control (`Delaware` $\rightarrow$ `Ohio`, $n=897$):** **0.67%** prediction flip rate `[0.31%, 1.45%]`.
     - **Currency (`$` $\rightarrow$ `EUR`, $n=376$) & Regulator (`SEC` $\rightarrow$ `ESMA`, $n=159$):** **0.27%** and **0.00%** flip rates (invariant).
     - **European Jurisdictions (US state $\rightarrow$ `Germany` / `France`, $n=915$):** Flip rate rises **3.1×–3.8×** to **2.08%** (Germany) and **2.51%** `[1.68%, 3.74%]` (France).
     - **European Statute (`ERISA` $\rightarrow$ `GDPR`, $n=95$):** Flip rate jumps to **6.32%** overall (**10.91%** `[5.10%, 21.83%]` on true `Erisa` clauses) with a **-18.87 percentage-point drop in model confidence**, because `GDPR` is out-of-vocabulary in US SEC filings.

---

## 5. Deployment Boundaries: What This System Must Not Be Used For

| # | Prohibited Use Case | Empirical Evidence from Our Audit | Enforced Product Guardrail |
| :--- | :--- | :--- | :--- |
| **1** | **Autonomous tagging of the 15 Tier-3 (Red) clause types** | 15 classes have $\text{CI}_{\text{lower}} < 50\%$ or $n_{\text{test}} < 10$ (e.g., `Assigns` CI `[0%, 49%]`, `Applicable Laws` F1 `0.0%`). | Hard-block auto-tagging on all 15 Tier-3 classes; always route to human review. |
| **2** | **Forcing a single label on overlapping or compound clauses** | Top 10 label pairs drive **25.5%** of `Legal-BERT` errors; compound `Amendments`/`Waivers` clauses contain both keywords. | Use 95% Split Conformal sets: route size 2–3 (**27.3%**) to a paralegal shortlist and size $\ge 4$ (**2.7%**) to a senior lawyer. |
| **3** | **Standalone `Legal-BERT` on long clauses ($>260$ words)** | 256-token truncation costs **-2.23 pts** vs. full-text `TF-IDF` on clauses $>260$ words (`86.89%` vs. `89.12%`). | Deploy the calibrated `Legal-BERT + TF-IDF` ensemble (**87.77%** accuracy, **80.62%** Macro-F1). |
| **4** | **Pure dense vector search on numerical/statutory constraints** | Dense-only Precision@5 falls to **45.0%** on lawyer queries (*"30-day notice"* vs. *"60-day notice"*). | Enforce **Hybrid RRF (Dense + BM25) + Category Filter** (**68.0%** Precision@5, **88.3%** MRR@5). |
| **5** | **Unadapted deployment on European Civil Law or EU statutes (`GDPR`)** | Swapping `ERISA` $\rightarrow$ `GDPR` flips **6.3%** of predictions and drops confidence by **-18.9 pts**; US states $\rightarrow$ `France` increases flips **3.8×**. | Restrict to English US-style corporate contracts until fine-tuned and recalibrated on annotated EU clauses. |
| **6** | **Treating conformal coverage as a per-class guarantee or legal advice** | The **95.01%** conformal coverage is a marginal guarantee across the test distribution, not conditional on each rare class. | Present the assistant strictly as a triage and precedent-lookup tool; final legal interpretation remains with a qualified lawyer. |

---

## 6. Installation & Reproducibility

### Using `uv` (recommended)

```bash
uv sync

# 1. Statistical evaluation, calibration, ensemble & conformal prediction sets
uv run python -m src.evaluation

# 2. Precedent retrieval benchmarks (Protocol A: clause-to-clause, Protocol B: lawyer queries)
uv run python -m src.retrieval_eval --split test
uv run python -m src.retrieval_lawyer_eval

# 3. Four-part bias & failure-mode audit + plots
uv run python -m src.bias_audit --audit long-tail
uv run python -m src.bias_audit --audit length
uv run python -m src.bias_audit --audit taxonomy
uv run python -m src.bias_audit --audit counterfactual
uv run python -m src.bias_audit --audit plots
```

### Using `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.evaluation
python -m src.retrieval_lawyer_eval
python -m src.bias_audit --audit all
```

---

## 7. Presentation Deck

The visual defence presentation is located in [`presentation/deck.html`](presentation/deck.html) (with [`presentation/deck.mp4`](presentation/deck.mp4)).  
`presentation/deck.html` is **100% self-contained** and opens directly in any modern web browser with zero on-screen interface:

- **Next animation / step:** `→`, `Space`, `Enter`, `PageDown`, or **Left Click**
- **Previous step:** `←`, `Backspace`, `PageUp`, or **Right Click**
- **Fullscreen:** `F`
- **Restart from beginning:** `Home` or `R`
