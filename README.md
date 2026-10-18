# Final Project — Contract-Review Assistant NLP Core

**Delivery Date:** October 23rd (GitHub Repository + 10-minute Presentation & 5-minute Q&A)

---

## 1. Project Context & Mission

We are the AI engineering team at a **European legal-tech start-up** building the NLP core of a contract-review assistant used during legal due diligence (mergers, financings, supplier audits).

Given a contract, our system must:
1. **Classify every clause** into one of the **100 clause categories** in our review checklist (`Governing Laws`, `Terminations`, `Indemnifications`, `Confidentiality`, etc.).
2. **Retrieve the most similar precedent clauses** from a clause bank, given either a pasted clause or a lawyer's question (*"show me termination clauses with a 30-day notice period"*).
3. **Provide rigorous statistical evidence** of where the model works—and a clear audit of **where it cannot be trusted / what it must not be used for**.

### Dataset (`coastalcph/lex_glue` — `ledgar`)
- **Source:** US SEC (EDGAR) public material contracts.
- **Splits:** `60,000` train / `10,000` validation / `10,000` test clauses across **100 classes**.
- **Key Challenge #1 (Extreme Class Imbalance):** The largest class (`Governing Laws`) has **3,167** training clauses, while the smallest (`Books`) has only **23** training clauses (0 in validation, only **2** in the test set!).
- **Key Challenge #2 (US Data vs. European Legal Context):** Although our startup operates in Europe, the public training dataset (`ledgar`) is 100% English US SEC filings governed by US Common Law (Delaware/New York).

---

## 2. Our Strategic Angle (Why Our Project Stands Out)

> **Core Principle:** *"A number alone is useless."*

Instead of reporting a single point estimate (`Accuracy = 87.5%, Macro-F1 = 80.1%`) like a standard notebook, our project is built around three engineering pillars:
1. **Real & Interpretable 95% Confidence Intervals:**
   - Every global metric (Micro-F1, Macro-F1, Top-3 Accuracy) is anchored against a fast `TF-IDF + Logistic Regression` baseline and reported with a **95% Non-Parametric Bootstrap Confidence Interval**.
   - Every per-class metric is reported with its test sample size $n_c$ and **95% Wilson / Bootstrap Confidence Interval**, proving statistically that rare classes ($n_{\text{test}} < 10$) have huge uncertainty intervals (e.g. $[25\%, 95\%]$) and must be blocked from autonomous tagging.
2. **Calibrated Confidence & Human-in-the-Loop Routing:**
   - Raw neural network softmax probabilities are overconfident. We calibrate probabilities on the validation split (Temperature Scaling / Expected Calibration Error) and use **95% Prediction Sets** to decide when to auto-tag (1 clear class), show a shortlist (2–3 overlapping classes), or escalate to a Senior Partner.
3. **Quantitative Retrieval & US-to-EU Bias Audit:**
   - We evaluate retrieval quantitatively (Precision@$k$, MRR@$k$ with 95% CIs) and show why pure dense embeddings (`Sentence-BERT`) struggle on exact numerical constraints (*"30-day notice"* vs *"60-day notice"*) without lexical (`BM25`) + category filtering.
   - We audit why `CamemBERT` (French pre-trained) should not be blindly trained on English SEC data, and test how our model reacts when US legal entities/jurisdictions (`State of Delaware`, `$`, `SEC`) are replaced with European equivalents (`Republic of France`, `EUR`, `ESMA`, `GDPR`).

---

## 3. Repository Structure & 3-Person Team Split

To keep our codebase clean, modular, and production-ready (using `.py` modules rather than a single giant notebook), the project is split into **3 independent workstreams**:

```text
├── pyproject.toml         # uv project definition (dependencies, Python >= 3.13)
├── uv.lock                # Locked dependency versions
├── .python-version        # Python version used by uv
├── requirements.txt       # Same dependencies for pip users
├── README.md              # Project overview, team roles, task checklist & CTO governance mandate
├── train_colab_pro.ipynb  # [Person 1] Self-contained Colab Pro GPU training notebook (exports outputs.zip)
├── outputs/               # [Person 1] Logits, labels & training stats
│   ├── evaluation/        # [Person 2] 95% Bootstrap/Wilson CIs, calibration, conformal sets & plots
│   ├── retrieval_eval/    # [Person 3] Clause-to-clause & lawyer constraint retrieval metrics & plots
│   └── bias_audit/        # [Person 3] Long-tail, length truncation, taxonomy & US->EU counterfactual audits
└── src/
    ├── data.py                  # [Person 1] Dataset loader, class imbalance stats & tokenizer comparison
    ├── train.py                 # [Person 1] TF-IDF baseline + GPU BERT/Legal-BERT training & logit exporter
    ├── evaluation.py            # [Person 2] 95% Bootstrap CIs, Wilson intervals, Reliability Tiers & Calibration
    ├── retrieval.py             # [Person 3] Sentence-Transformer + SparseBM25 + RRF Hybrid search engine
    ├── retrieval_eval.py        # [Person 3] Clause-to-clause retrieval evaluation with 95% Bootstrap CIs
    ├── retrieval_lawyer_eval.py # [Person 3] 20 Lawyer numerical/jurisdiction constraint queries benchmark
    └── bias_audit.py            # [Person 3] Long-tail, truncation, taxonomy overlap & US->EU counterfactual audits
```

### How the 3 Roles Connect
- **Person 1 (GPU / Data / Model Training)** uploads [`train_colab_pro.ipynb`](train_colab_pro.ipynb) to Google Colab Pro (A100 / L4 GPU), runs the training, and downloads `outputs.zip` containing the validation and test predictions (`outputs/legal_bert_val_logits.npy`, `outputs/legal_bert_test_logits.npy`, `outputs/tfidf_val_logits.npy`, `outputs/tfidf_test_logits.npy`, `outputs/val_labels.npy`, `outputs/test_labels.npy` — only ~4 MB each).
- **Person 2 (Statistical Evaluation & Confidence Intervals)** loads those saved `.npy` files (no GPU needed!) to compute the 95% Bootstrap & Wilson confidence intervals, Reliability Tiers, Calibration curves, Calibrated Ensemble, and 95% Split Conformal Prediction Sets.
- **Person 3 (Retrieval & Bias Audit)** builds the `Sentence-Transformer + SparseBM25` hybrid retrieval index, evaluates both clause-to-clause and lawyer constraint queries with 95% Bootstrap CIs, and runs the 4-part failure-mode & US-to-EU bias audit.

---

## 4. Project Checklist (100% Completed)

### 1. Data, Tokenization & GPU Model Training (Person 1)
- [x] Set up clean repository structure (`src/`, `requirements.txt`, `.gitignore`).
- [x] Implement LEDGAR dataset loader & per-split class imbalance table in [`src/data.py`](src/data.py) (`outputs/class_distribution.csv`).
- [x] Implement tokenizer comparison (`Legal-BERT` vs `BERT-base` vs `CamemBERT`) in [`src/data.py`](src/data.py) to justify model selection (`outputs/tokenizer_comparison.json`):
  - `nlpaueb/legal-bert-base-uncased`: **1.20 tokens/word**, median **105 tokens**, **12.2%** truncated at 256 tokens (**1.6%** at 512).
  - `bert-base-uncased`: **1.257 tokens/word**, median **109.5 tokens**, **13.8%** truncated at 256 tokens.
  - `camembert-base`: **1.924 tokens/word** (+60% subword fertility inflation on English legal terms), median **169 tokens**, **28.4%** truncated at 256 tokens (**7.2%** at 512).
- [x] Implement `TF-IDF + Logistic Regression` baseline and PyTorch GPU training loop (`Legal-BERT`) in [`src/train.py`](src/train.py) and [`train_colab_pro.ipynb`](train_colab_pro.ipynb).
- [x] **Run on Colab GPU & Export Predictions (`outputs/`):**
  - `outputs/legal_bert_{val,test}_logits.npy`: Fine-tuned `nlpaueb/legal-bert-base-uncased` (Test Micro-F1: **85.98%** `[85.29%, 86.66%]`, Macro-F1: **75.89%** `[74.81%, 77.11%]`).
  - `outputs/tfidf_{val,test}_logits.npy`: `TF-IDF + Logistic Regression` baseline (Test Micro-F1: **86.41%** `[85.71%, 87.08%]`, Macro-F1: **79.69%** `[78.37%, 80.68%]`).

### 2. Statistical Evaluation, Confidence Intervals & Calibration (Person 2)
- [x] Implement `wilson_confidence_interval()` in [`src/evaluation.py`](src/evaluation.py) for small-sample binomial proportions.
- [x] Implement `evaluate_bootstrap_confidence_intervals()` in [`src/evaluation.py`](src/evaluation.py) (2,000 bootstrap resamples on the test logits) for Micro-F1, Macro-F1, Weighted-F1, Top-3 Accuracy, and per-class F1 (`outputs/evaluation/summary.json`).
- [x] Compare `Legal-BERT`, `TF-IDF Baseline`, and the `Calibrated Ensemble` (`0.45 Legal-BERT + 0.55 TF-IDF`, reaching **87.77% Micro-F1** `[87.12%, 88.40%]`, **80.62% Macro-F1** `[79.00%, 81.76%]`, and **96.61% Top-3 Accuracy**) using Paired Bootstrap tests (`paired_bootstrap_difference()`).
- [x] Group the 100 clause categories into **3 Reliability Tiers** based on 95% CI lower bound and CI width (for Ensemble: **52 Tier 1 Green**, **33 Tier 2 Amber**, **15 Tier 3 Red**):
  - *Tier 1 (Green — Auto-Tag Ready):* $\text{CI}_{\text{lower}} \ge 80\%$, $\text{CI}_{\text{width}} \le 30\%$, $n_{\text{test}} \ge 10$.
  - *Tier 2 (Amber — Human-in-the-Loop Assisted):* Intermediate certainty; display top predictions for paralegal review.
  - *Tier 3 (Red — Do Not Deploy / Statistically Unverified):* $\text{CI}_{\text{lower}} < 50\%$ or extreme uncertainty (e.g. `Books`, $n_{\text{test}}=2$).
- [x] Compute Expected Calibration Error (ECE), apply Temperature Scaling on validation logits (`TF-IDF` ECE drops from **10.99%** to **2.70%**), and generate forest plots & reliability diagrams (`outputs/evaluation/*_{forest,reliability}.png`).
- [x] Implement Split Conformal 95% Prediction Sets (`conformal_prediction_sets()`): achieves **95.01% empirical test coverage** `[94.56%, 95.43%]` with mean set size **1.43**, routing **70.0%** of clauses to Auto-Tag (at **96.57% accuracy**), **27.3%** to a 2–3 class Shortlist, and **2.7%** to Senior Escalation.

### 3. Clause Bank Retrieval & Bias / Failure-Mode Audit (Person 3)
- [x] Implement `ClauseBankRetriever` in [`src/retrieval.py`](src/retrieval.py) using `sentence-transformers/all-MiniLM-L6-v2` (dense cosine similarity) + `SparseBM25` (vectorized lexical matching) + Reciprocal Rank Fusion (`hybrid`, RRF $k=60$) with pre-ranking category filtering.
- [x] Evaluate retrieval quantitatively with 95% Bootstrap CIs (2,000 query resamples) in [`src/retrieval_eval.py`](src/retrieval_eval.py) and [`src/retrieval_lawyer_eval.py`](src/retrieval_lawyer_eval.py) (`outputs/retrieval_eval/`):
  - **Clause-to-Clause Precedent Search (Test, $n=1,000$ queries vs. 60,000 clause bank):**
    - `Dense`: Precision@5 = **74.9%** `[72.9%, 77.1%]`, MRR@5 = **85.6%** `[83.6%, 87.5%]`, HitRate@5 = **91.0%**.
    - `BM25`: Precision@5 = **75.7%** `[73.6%, 77.9%]`, MRR@5 = **85.2%** `[83.1%, 87.2%]`, HitRate@5 = **90.4%**.
    - `Hybrid (RRF)`: Precision@5 = **76.9%** `[74.9%, 79.0%]`, MRR@5 = **85.7%** `[83.7%, 87.8%]`, HitRate@5 = **91.1%** — Paired bootstrap difference `Hybrid - Dense` = **+2.0 pts** `[+1.0%, +3.1%]` (statistically significant, and **+2.0 pts** even after excluding exact duplicates).
  - **Lawyer Constraint Queries ($n=20$ queries with numerical/jurisdiction constraints, e.g., *"30-day notice"* vs *"60-day notice"*):**
    - `Pure Dense`: Precision@5 collapses to **45.0%** `[30.0%, 61.0%]` (even though its Category-only Precision@5 is **81.0%**!), proving that dense embeddings capture the topic (*termination/payment*) but are blind to exact numbers (`30` vs `60` days).
    - `Hybrid (Dense + BM25)`: Precision@5 jumps to **61.0%** `[46.0%, 75.0%]` (**+16.0 pts** over Dense, significant).
    - `Hybrid + Category Filter`: Precision@5 reaches **68.0%** `[54.0%, 82.0%]` (**+23.0 pts** `[+14.0%, +31.0%]` over Pure Dense, MRR@5 = **88.3%** `[75.0%, 100.0%]`, HitRate@5 = **95.0%**).
- [x] Implement the 4-part Bias & Failure-Mode Audit in [`src/bias_audit.py`](src/bias_audit.py) (`outputs/bias_audit/`):
  - **Audit 0 (Long-Tail / Sample-Size Bias — `long_tail_audit.json`):** On Head classes ($n_{\text{train}} \ge 1,000$, 16 classes), `Legal-BERT` achieves **92.0%** mean F1 with a tight **5.1%** mean 95% CI width. On Tail classes ($n_{\text{train}} < 150$, 7 classes), `Legal-BERT` collapses to **34.6%** mean F1 (scoring **0% F1** on 5 rare/cannibalized classes overall: `Books`, `Assigns`, `Qualifications`, `Jurisdictions`, `Applicable Laws`), while `TF-IDF` (**58.4%** Tail F1) and the `Calibrated Ensemble` (**54.3%** Tail F1, **+19.7 pts** over `Legal-BERT`) recover most rare classes. Critically, the mean 95% CI width explodes from **5.1 pts** (Head) to **45.2 pts** (Tail), proving why point estimates on rare classes are statistically unusable without CIs.
  - **Audit 1 (Length & Truncation Bias — `length_audit.json`):** Across `<80`, `80-160`, `160-260`, and `>260` words, `Legal-BERT` and `TF-IDF` tie on short clauses (`86.76%` vs `86.74%` on `<80` words), but on long clauses (`>260` words, where BERT's 256-token window truncates the tail of the clause), `TF-IDF` outperforms `Legal-BERT` by **+2.23 points** (`89.12% [86.63%, 91.20%]` vs `86.89% [84.22%, 89.16%]`).
  - **Audit 2 (Taxonomy Overlap — `taxonomy_audit.json`):** Just **10 confusion pairs** out of 4,950 possible pairs account for **25.5% of all Legal-BERT errors** (`Tax Withholdings` / `Withholdings`: 60 errors; `Applicable Laws` $\rightarrow$ `Governing Laws`: 42 unidirectional errors; `Defined Terms` / `Definitions`: 41 errors; `No Waivers` / `Waivers`: 31 errors). On `Amendments` / `Waivers`, **100% (`4/4`)** of confused clauses contain both keywords (`amend` and `waive`) vs. only **12.7% (`43/339`)** of correctly classified clauses.
  - **Audit 3 (US SEC $\rightarrow$ European Law Counterfactual Domain Shift — `counterfactual_audit.json`):**
    - Swapping US states with another US state (`control_US_swap`: `Delaware` $\rightarrow$ `Ohio`, $n=897$) causes only a **0.67%** `[0.31%, 1.45%]` prediction flip rate.
    - Swapping US states with European jurisdictions (`juris_FR`: `Republic of France`, $n=915$) increases the flip rate **3.8x** to **2.51%** `[1.68%, 3.74%]` (disjoint 95% CIs) and triples the confidence drop (`-4.80 pts` vs `-1.66 pts`).
    - Swapping US pension statute `ERISA` with European `GDPR` (`statute_GDPR`, $n=95$) triggers a **6.32%** flip rate (**10.91%** `[5.10%, 21.83%]` inside true `Erisa` clauses) and a massive **-18.87 percentage-point drop in model confidence** `[-13.92%, -23.71%]`.
- [x] Summarize the final **"What This Model Must Not Be Used For"** governance mandate for the CTO defense (see Section 5 below).

---

## 5. What This Model Must Not Be Used For (CTO Governance & Deployment Mandate)

Based on our statistical confidence intervals and failure-mode audits, we establish **6 strict operational red lines** for deployment at our European legal-tech startup:

| # | Prohibited Use Case | Statistical Evidence from Our Audit | Mandatory Engineering Guardrail |
| :--- | :--- | :--- | :--- |
| **1** | **Autonomous tagging of Tier 3 (Red) & Tail categories ($n_{\text{test}} < 15$)** | 15 classes remain in Tier 3 ($\text{CI}_{\text{lower}} < 50\%$). `Legal-BERT` alone scores **0.0% F1** on 5 rare classes (`Books`, `Assigns`, `Qualifications`, `Jurisdictions`, `Applicable Laws`), and `Books` ($n_{\text{test}}=2$) has a 95% CI spanning $>60$ pts. | **Hard Block:** Disable auto-tagging for all 15 Tier-3 classes; route exclusively through human review. |
| **2** | **Single-label forced classification on compound boilerplate clauses** | Top 10 overlapping category pairs (`Tax Withholdings`/`Withholdings`, `Applicable Laws`/`Governing Laws`, `No Waivers`/`Waivers`) drive **25.5%** of all `Legal-BERT` errors; **100%** of `Amendments`/`Waivers` errors contain both keywords. | **Conformal Shortlist UI:** Use 95% Split Conformal Prediction Sets (`q_hat = 0.948`). When set size is 2–3 (**27.3%** of clauses), display a multi-label shortlist instead of forcing a single label; escalate set sizes $\ge 4$ (**2.7%**). |
| **3** | **Standalone `Legal-BERT` on long clauses ($>260$ words / $>256$ subwords)** | **12.2%** of clauses exceed 256 `Legal-BERT` tokens. On clauses $>260$ words ($n=717$), `Legal-BERT` trails full-text `TF-IDF` by **-2.23 pts** (`86.89%` vs `89.12%`). | **Hybrid Ensemble:** Deploy the calibrated `0.45 Legal-BERT + 0.55 TF-IDF` ensemble (**87.77% Micro-F1**, **80.62% Macro-F1**), where full-text `TF-IDF` prevents truncation blindness. |
| **4** | **Pure dense vector search (`Sentence-BERT`) for numerical or jurisdictional constraints** | On lawyer constraint queries (*"30-day notice"* vs *"60-day notice"*), Pure Dense Precision@5 collapses to **45.0%** `[30.0%, 61.0%]` despite **81.0%** category accuracy. | **Hybrid RRF + Category Filter:** Enforce `Hybrid (Dense + BM25) + Category Filter`, which boosts Precision@5 by **+23.0 pts** to **68.0%** `[54.0%, 82.0%]` and MRR@5 to **88.3%**. |
| **5** | **Evaluating European Civil Law compliance or EU statutory clauses (`GDPR`, Works Councils)** | LEDGAR is 100% US Common Law (SEC filings) and includes US-only classes (`Erisa`, `Waiver Of Jury Trials` — 7th Amendment). Replacing `ERISA` with `GDPR` drops model confidence by **-18.9 pts** and flips **10.9%** of `Erisa` predictions; replacing US states with `France` increases flips **3.8x** vs. the US control (`2.51%` vs `0.67%`). | **Jurisdiction Gate:** Deprecate US-only labels (`Erisa`, `Waiver Of Jury Trials`) for EU contracts and trigger automatic out-of-distribution (OOD) escalation when EU civil-law statutes (`GDPR`, `ESMA`, `BGB`, `Code civil`) reduce calibrated confidence below $1 - \hat{q}$. |
| **6** | **Applying this English pipeline to French/German contracts—or training `CamemBERT` on English SEC data** | `CamemBERT` fragments English legal text into **1.924 tokens/word** (+60% inflation vs `Legal-BERT`'s **1.20**), truncating **28.4%** of clauses at 256 tokens. Conversely, English models have no alignment with civil-law terminology in French/German. | **Language Router:** Restrict this model strictly to **English-language corporate contracts**; require a dedicated annotated European corpus before deploying a French (`CamemBERT`) or multilingual (`XLM-RoBERTa`) classifier. |

---

## 6. Setup & Reproducibility Instructions

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
uv run python -m src.evaluation                             # [Person 2] Regenerates outputs/evaluation/
uv run python -m src.retrieval_eval --split validation      # [Person 3] Clause-to-clause retrieval eval (val)
uv run python -m src.retrieval_lawyer_eval                  # [Person 3] 20 lawyer constraint queries eval
uv run python -m src.bias_audit --audit long-tail           # [Person 3] Audit 0: Head/Medium/Tail sample-size bias
uv run python -m src.bias_audit --audit length              # [Person 3] Audit 1: Clause length & truncation bias
uv run python -m src.bias_audit --audit taxonomy            # [Person 3] Audit 2: Taxonomy overlap & compound clauses
uv run python -m src.bias_audit --audit counterfactual      # [Person 3] Audit 3: US -> EU counterfactual perturbation
uv run python -m src.bias_audit --audit plots               # [Person 3] Regenerates PNG plots for retrieval & bias audits
```

With pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.evaluation
python -m src.bias_audit --audit long-tail
python -m src.bias_audit --audit plots
```

