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
- **Key Challenge #1 (Extreme Class Imbalance):** The largest class (`Governing Laws`) has **3,167** training clauses, while the smallest (`Books`) has only **23** training clauses (~4 in the test set!).
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
├── requirements.txt       # Project dependencies
├── README.md              # Project overview, team roles & task checklist
├── train_colab_pro.ipynb  # [Person 1] Self-contained Colab Pro GPU training notebook (exports outputs.zip)
└── src/
    ├── data.py            # [Person 1] Dataset loader, class imbalance stats & tokenizer comparison
    ├── train.py           # [Person 1] TF-IDF baseline + GPU BERT/Legal-BERT training & logit exporter
    ├── evaluation.py      # [Person 2] 95% Bootstrap CIs, Wilson intervals, Reliability Tiers & Calibration
    ├── retrieval.py       # [Person 3] Sentence-Transformer + BM25 index & '30-day notice' query benchmark
    └── bias_audit.py      # [Person 3] Truncation audit, taxonomy overlap & US->EU counterfactual audit
```

### How the 3 Roles Connect
- **Person 1 (GPU / Data / Model Training)** uploads [`train_colab_pro.ipynb`](train_colab_pro.ipynb) to Google Colab Pro (A100 / L4 GPU), runs the training, and downloads `outputs.zip` containing the validation and test predictions (`outputs/legal_bert_val_logits.npy`, `outputs/legal_bert_test_logits.npy`, `outputs/tfidf_val_logits.npy`, `outputs/tfidf_test_logits.npy`, `outputs/val_labels.npy`, `outputs/test_labels.npy` — only ~4 MB each).
- **Person 2 (Statistical Evaluation & Confidence Intervals)** loads those saved `.npy` files (no GPU needed!) to compute the 95% Bootstrap & Wilson confidence intervals, Reliability Tiers, and Calibration curves.
- **Person 3 (Retrieval & Bias Audit)** builds the `Sentence-Transformer` retrieval index and runs the failure-mode & US-to-EU bias audits.

---

## 4. Project Checklist (Done vs. To Do)

###  Data, Tokenization & GPU Model Training 
- [x] Set up clean repository structure (`src/`, `requirements.txt`, `.gitignore`).
- [x] Implement LEDGAR dataset loader & per-split class imbalance table in [`src/data.py`](src/data.py).
- [x] Implement tokenizer comparison (`Legal-BERT` vs `BERT-base` vs `CamemBERT`) in [`src/data.py`](src/data.py) to justify model selection (`outputs/tokenizer_comparison.json`).
- [x] Implement `TF-IDF + Logistic Regression` baseline and PyTorch GPU training loop (`Legal-BERT`) in [`src/train.py`](src/train.py) and [`train_colab_pro.ipynb`](train_colab_pro.ipynb).
- [x] **Run on Colab GPU & Export Predictions (`outputs/`):**
  - `outputs/legal_bert_{val,test}_logits.npy`: Fine-tuned `nlpaueb/legal-bert-base-uncased` (Test Micro-F1: **85.98%**, Macro-F1: **75.89%**).
  - `outputs/tfidf_{val,test}_logits.npy`: `TF-IDF + Logistic Regression` baseline (Test Micro-F1: **86.41%**, Macro-F1: **79.69%**).
  - *Note for Person 2 & 3:* Combining `Legal-BERT` + `TF-IDF` probabilities reaches **87.84% Micro-F1** and **81.02% Macro-F1**, proving that `TF-IDF` compensates for `Legal-BERT`'s 256-token truncation (12.2% of clauses) and rare-class underfitting (5 tail classes with 0% F1 in `Legal-BERT`).

### Statistical Evaluation, Confidence Intervals & Calibration
- [ ] Implement `wilson_confidence_interval()` in [`src/evaluation.py`](src/evaluation.py) for small-sample binomial proportions.
- [ ] Implement `evaluate_bootstrap_confidence_intervals()` in [`src/evaluation.py`](src/evaluation.py) (1,000+ bootstrap resamples on `test_logits.npy`) for Micro-F1, Macro-F1, Weighted-F1, Top-3 Accuracy, and per-class F1.
- [ ] Compare `Legal-BERT` vs. `TF-IDF Baseline` using a Paired Bootstrap test (95% CI on $\Delta \text{Macro-F1}$).
- [ ] Group the 100 clause categories into **3 Reliability Tiers** based on 95% CI lower bound and CI width:
  - *Tier 1 (Green — Auto-Tag Ready)*
  - *Tier 2 (Amber — Human-in-the-Loop Assisted)*
  - *Tier 3 (Red — Do Not Deploy / Statistically Unverified, e.g., `Books`)*
- [ ] Compute Expected Calibration Error (ECE), apply Temperature Scaling on `val_logits.npy`, and generate the forest plot & reliability diagram for the presentation.

### Clause Bank Retrieval & Bias / Failure-Mode Audit
- [ ] Implement `ClauseBankRetriever` in [`src/retrieval.py`](src/retrieval.py) using `sentence-transformers/all-MiniLM-L6-v2` (dense cosine similarity) + `BM25` (lexical matching).
- [ ] Evaluate retrieval quantitatively with 95% CIs on:
  - Clause-to-clause precedent search (Precision@5, MRR@5).
  - Lawyer constraint queries (e.g., *"show me termination clauses with a 30-day notice period"*), comparing Pure Dense vs. Hybrid (Dense + BM25 + Category Filter).
- [ ] Implement the Bias & Failure-Mode Audit in [`src/bias_audit.py`](src/bias_audit.py):
  - **Audit 1 (Length & Truncation Bias):** Accuracy + 95% CI across clause length buckets (`<80`, `80-160`, `160-260`, `>260` words) to show the 256/512 token truncation drop.
  - **Audit 2 (Taxonomy Overlap):** Show that top confusion pairs (`Governing Laws` vs `Submission To Jurisdiction`, `Amendments` vs `Waivers`) come from compound boilerplate clauses.
  - **Audit 3 (US SEC $\rightarrow$ European Law Domain Shift):** Identify US-only classes (`Erisa`, `Waiver Of Jury Trials`) and run a counterfactual test swapping US jurisdictions/currencies (`Delaware`, `New York`, `$`, `SEC`) with European equivalents (`France`, `Germany`, `EUR`, `ESMA`, `GDPR`).
- [ ] Summarize the final **"What This Model Must Not Be Used For"** slide for the CTO defense.

---

## 5. Setup Instructions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
