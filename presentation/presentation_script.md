# Presentation Script (`deck.html`)

---

### 1. Title (`#0`)

Good morning. During corporate due diligence, junior lawyers read thousands of contract clauses by hand. Today we present the NLP core of our contract-review assistant—trained on **80,000 SEC clauses across 100 clause types**—and show you not only how it works, but how it knows **when to ask a lawyer**.

**[PRESS →]**

---

### 2. What the assistant must do (`#1` – `#4`)

Take a concrete example. A lawyer pastes this clause into the assistant: *"This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict-of-laws principles."* Our core must do three things.

**[PRESS →]**

First, **classify** it: scan the **100 checklist categories** and tag this clause as **Governing Laws**.

**[PRESS →]**

Second, **retrieve precedents**: project the clause into our **60,000-clause bank** and pull the **five closest clauses**—or answer a lawyer's natural-language question.

**[PRESS →]**

And third—the hardest part for European legal clients—**know when to ask a human**: tag automatically when certain, show a shortlist to a paralegal when two labels compete, or escalate to a lawyer.

**[PRESS →]**

---

### 3. The data: 100 clause types, very unequal (`#5` – `#9`)

Why is that third step so critical? Look at the training data across all **100 clause types**, sorted from most common on the left to rarest on the right.

**[PRESS →]**

**Governing Laws** has **3,167** training examples. At the other end, **Books** has just **23**. That is a **138-to-1 imbalance**.

**[PRESS →]**

In fact, **16 common types** ($\ge 1,000$ clauses, `41.7%` of test) dominate the corpus, **77 sit in the middle** (`57.5%` of test), and **7 rare types** have fewer than 150 examples each (`0.81%` of test).

**[PRESS →]**

If we only report **accuracy**, each type is weighed by its bar height—so a model can completely fail on rare clauses and still look great.

**[PRESS →]**

That is why our primary metric is **Macro-F1**, which gives every single type the **exact same weight**—so the **7 rare types** count just as much per class as the **16 common ones**.

**[PRESS →]**

---

### 4. Why Legal-BERT, not CamemBERT (`#10` – `#11`)

Before training, we audited the tokenizer. Take a **100-word English clause**. Because **Legal-BERT** was pre-trained on English legal text, its WordPiece vocabulary splits those 100 words into **120 pieces** (`1.20` tokens/word). **CamemBERT**—trained on French—shatters English legal words into **192 pieces** (`1.92` tokens/word).

**[PRESS →]**

Why does that matter? Because our transformer window is capped at **256 pieces**—anything past that line is cut and never read. With Legal-BERT, only **12% of clauses** (`12.2%`) are truncated; with CamemBERT, **28% of clauses** (`28.4%`) lose their ending. So we chose **Legal-BERT**.

**[PRESS →]**

---

### 5. Fine-tuning Legal-BERT (`#12` – `#14`)

We split our **80,000 labelled clauses** into **60,000 for training**, **10,000 for validation**, and a strict **10,000-clause held-out test set** never seen during training.

**[PRESS →]**

In the forward pass, a clause is split into pieces with dynamic batch padding (`DataCollatorWithPadding`), flows through **12 transformer layers of Legal-BERT** (`109.5M` parameters), condenses into a single `[CLS]` summary vector, and outputs **100 scores**—picking the highest peak, **Governing Laws**.

**[PRESS →]**

Whenever the prediction is wrong, we adjust every layer using `AdamW` ($\text{lr} = 3\times 10^{-5}$, 10% linear warmup + cosine decay). Across **3 passes** over the 60,000 clauses—**41 minutes on one GPU** (`2,456s`)—validation accuracy climbs from **82.7%** to **85.4%** to **86.1%** (**86.0% test accuracy**, **75.9% Macro-F1**). We also trained a **3-epoch LoRA ($r = 8, \alpha = 16$)** ablation on all query/value projections, cutting trainable parameters by **295×** from `109.5M` (**1.75 GB** in `fp32` AdamW) down to **371,812 weights (`0.34%`, 225 MB)** and reaching **84.7% test accuracy** (`[84.0, 85.4]`) and **75.4% Macro-F1** (`[73.9, 76.5]`)—recovering **99.3% of full fine-tuning's Macro-F1** in **8.2 minutes** (`491s`), while confirming we should retain the full fine-tuned model (`+1.25 pts` accuracy) for production.

**[PRESS →]**

---

### 6. How good is it, and how sure are we? (`#15` – `#17`)

A single test score is not enough. On our **10,000 test clauses**, we redraw 10,000 clauses at random with replacement and score again across **2,000 bootstrap resamples** (400 shown here as dots). Ninety-five percent of Legal-BERT's Macro-F1 scores land between **74.8% and 77.1%**.

**[PRESS →]**

That gives **Legal-BERT 75.9% Macro-F1** (`86.0%` accuracy). A full-text **TF-IDF + Logistic Regression** baseline reaches **79.7% Macro-F1** (`86.4%` accuracy) thanks to exact n-grams on rare classes, and our **calibrated Ensemble of the two reaches 80.6% Macro-F1 and 87.8% accuracy**.

**[PRESS →]**

Is the gap statistically real on the exact same test clauses? Against Legal-BERT, the Ensemble gains **+4.7 points in Macro-F1** (`[+3.4, +5.7]`)—a **real gain** clear of zero. Against TF-IDF on Macro-F1, the **+0.9 point** interval (`[-0.1, +1.8]`) slightly crosses zero, so we report it transparently as **not proven** on Macro-F1. On per-clause **accuracy**, however, **McNemar's paired contingency test** proves the Ensemble's **+1.4 point gain over TF-IDF** is highly significant ($\chi^2 = 50.35, p < 10^{-11}$), because Legal-BERT and TF-IDF complement each other on **783 test clauses** (`370` solved only by Legal-BERT vs. `413` solved only by TF-IDF).

**[PRESS →]**

---

### 7. Calibration: when it says 70%, is it right 70% of the time? (`#18` – `#20`)

To know when to ask a lawyer, confidence must be **honest**—lying on the diagonal. Raw **TF-IDF** is under-confident: when it says **45%**, it is actually right **62%** of the time (and at **63%** confidence, it is right **84%** of the time), giving an average Expected Calibration Error (ECE) gap of **11.0%** (`10.99%`).

**[PRESS →]**

We fix this with temperature scaling—fitting a single number, **$T = 0.68$** (`0.684`), on validation without changing top-1 predictions. The curve snaps to the diagonal and the TF-IDF calibration gap drops **4.1× to 2.7%** (`2.70%`).

**[PRESS →]**

Meanwhile, both **Legal-BERT (`1.3% → 1.7%`)** and our **Ensemble (`2.2% → 2.0%`)** were already honest before scaling.

**[PRESS →]**

---

### 8. Knowing when to ask a human (`#21` – `#24`)

Now, how do we build a formal coverage guarantee? Take this test clause: the model gives **88.9% to Governing Laws** and **10.7% to Applicable Laws**.

**[PRESS →]**

Using **split conformal prediction**, we calibrate a non-conformity threshold ($\hat{q}_{0.95} = 0.9477$) so that **95.0% of validation clauses** keep their true type. Note that calibration sharpens set sizes, while conformal prediction guarantees marginal coverage regardless of the score. At that threshold, **2 types are kept** for this clause.

**[PRESS →]**

The size of that set decides who looks at the clause: **1 type** means we **tag it automatically**; **2 or 3 types** means **a paralegal picks one**; **4 or more** means **a lawyer reads it**.

**[PRESS →]**

Across all **10,000 test clauses**, **70.0% are tagged automatically at 96.6% accuracy**, **27.3% go to a paralegal**, only **2.7% require a lawyer**—and the true type stays inside the set on **95.0% of test clauses** (`95.01%`). When we also hard-block the **15 red (Tier-3) clause types** from autonomous tagging (`54` singletons diverted), **69.4% are auto-tagged at 96.6% accuracy**, leaving **~24 silent errors per 1,000 clauses** (down from `122` without routing).

**[PRESS →]**

---

### 9. Finding similar clauses (`#25` – `#28`)

Second mission: **precedent retrieval**. When a lawyer asks for *"termination clauses with a 30-day notice period,"* a sentence-transformer (`all-MiniLM-L6-v2`) projects that question into our map of **60,000 clauses** and finds the 5 closest neighbours.

**[PRESS →]**

But meaning alone can miss the exact number: dense search finds a termination clause with **sixty days' notice**—right topic, wrong notice period. Keyword search (`BM25`) finds **thirty days**. So we combine both via Reciprocal Rank Fusion in **hybrid search: meaning + exact words**.

**[PRESS →]**

On **20 lawyer constraint questions**, meaning alone gets **45%** relevant results in the top 5 (`Precision@5`). Adding our type filter and hybrid search reaches **68%**—a statistically significant **+23 point gain** (`[14, 31]`), which is what we ship.

**[PRESS →]**

And to be completely transparent: on those 20 short keyword questions—where relevance is checked by a regex on the number or state—**exact words alone tie at 69%** (`-1 point [-12, +10]`). On our broader benchmark of **1,000 clause-to-clause searches**, however, **hybrid search (`76.9%`) significantly beats both meaning (`+2.0 pts [+0.9, +3.1]`) and exact words (`+1.2 pts [+0.2, +2.3]`)**.

**[PRESS →]**

---

### 10. Audit 1: rare clause types (`#29` – `#32`)

Now for our failure-mode and **bias audit**. First: **rare clause types**. Here each dot is one of the 100 clause types, plotted by training clauses on a log scale against test F1 across **rare**, **medium**, and **common** types.

**[PRESS →]**

Adding 95% confidence intervals shows the problem on the left: **Assigns** has only **4 test clauses**, so its F1 lies anywhere from **0% to 49%**.

**[PRESS →]**

Even a medium type can fail when LEDGAR headings overlap: **Applicable Laws** (`352` train, **53 test clauses**) has an **F1 of 0**, because it is almost always predicted as its near-synonym **Governing Laws** (`51/53` clauses). In fact, just **10 near-synonym label pairs** account for **25.5% of all Legal-BERT errors**.

**[PRESS →]**

Across groups, average Ensemble F1 drops from **92% on common types** to **81% on medium**, and down to **54% on rare types** (where mean CI width reaches `52 points`).

**[PRESS →]**

---

### 11. Audit 2: long clauses (`#33` – `#34`)

Second audit: **long clauses**. Legal-BERT cuts every clause at **256 pieces**, which truncates the **`> 260` words** bucket (`7%` of clauses, $n = 717$).

**[PRESS →]**

Comparing Legal-BERT (`86.9%` `[84.2%, 89.2%]`) with **TF-IDF** (`89.1%` `[86.6%, 91.2%]`), which reads every word, shows overlapping Wilson confidence intervals: **the longest clauses are not significantly penalised**—because in SEC contracts, the opening lines already state the clause type.

**[PRESS →]**

---

### 12. Audit 3: does changing parties or laws change the answer? (`#35` – `#36`)

Third audit: **counterfactual swaps on parties and European law**, measured on the full-text **TF-IDF member of the ensemble** (`99.8%` agreement with saved test predictions). When we swap US terms inside real clauses, `Delaware → Ohio` changes the predicted type **0.7%** of the time ($n = 897$), `$ → EUR` **0.3%** ($n = 376$), and `SEC → ESMA` **0.0%** ($n = 159$). Swapping bilateral **contractual party roles** (`Lender → Borrower`, `Employer → Employee`, $n = 2,612$ clauses) also flips only **1.8%** of predictions.

**[PRESS →]**

Swapping a US state for **Germany (`2.1%`)** or **France (`2.5%`)** ($n = 915$) triples the flip rate relative to the US control, and swapping the US statute **ERISA for GDPR** ($n = 95$) flips **6.3%** of predictions (`10.9%` on true ERISA clauses) and drops model confidence by **19 points** (`-18.9 pts [-23.7, -13.9]`), because `GDPR` is unseen in US SEC filings.

**[PRESS →]**

---

### 13. What ships, and what it must not be used for (`#37` – `#38`)

We turn this audit into product rules across all 100 clause types, based on the lower bound of each class's 95% F1 confidence interval ($\text{CI}_{\text{lower}}$), CI width ($\le 30\text{ pts}$), and test support ($n \ge 10$): **52 green types (Tier 1)** are tagged automatically ($\text{CI}_{\text{lower}} \ge 80\%$), **33 yellow types (Tier 2)** require human confirmation, and **15 red types (Tier 3)** are never auto-tagged ($\text{CI}_{\text{lower}} < 50\%$ or $n < 10$).

**[PRESS →]**

And we state clearly what **not to use it for**:
- **non-US or non-English contracts** (`GDPR`, French/German Civil Law), until re-tested and fine-tuned;
- **the 15 red clause types**, without a lawyer (blocking their `54` singletons gives `69.4%` safe auto-tagging);
- **legal advice**: it sorts the pile (with `~24` residual silent errors per `1,000` clauses), a lawyer decides;
- **a per-type guarantee**: the **95% conformal coverage is a marginal average**, not a guarantee on rare classes like `Books`.

**[PRESS →]**

---

### 14. Summary (`#39` — Final Slide)

In summary: **87.8%** of clauses get the right type, **70%** (`69.4%` Tier-3 gated) are tagged with no human at **96.6% accuracy**, **95%** keep the true type inside the shortlist—and we ship an audit that says **where not to trust it**. Thank you, and we are ready for your questions.

---

## Jury Q&A — Quick Defense Reference

| Jury Question | Precise, Evidence-Backed Answer |
| :--- | :--- |
| **1. Why did `Legal-BERT` (`75.9%` Macro-F1) lose to `TF-IDF` (`79.7%` Macro-F1)?** | Under a **138× Zipfian imbalance** (`7` tail classes have `< 150` training examples), neural fine-tuning for 3 epochs underfits ultra-rare classes (`34.6%` Tail F1 for Legal-BERT vs. `53.2%` for TF-IDF), whereas `TF-IDF` (`50k` 1–2 grams, `C=2.0`) memorizes distinctive statutory n-grams immediately. However, **McNemar's test** shows Legal-BERT uniquely solves **370** semantic paraphrases missed by TF-IDF, so combining them reaches **87.8% accuracy** ($p < 10^{-11}$) and **80.6% Macro-F1**. |
| **2. Why stop at 3 epochs and 256 tokens?** | Standard BERT fine-tuning literature (Devlin et al., Chalkidis et al.) recommends 3–4 epochs to avoid overfitting head classes while preserving pre-trained representations; at `256` tokens, `87.8%` of clauses are read in full, and our length audit (`#33`–`#34`) proves that `> 260`-word clauses (`86.9%` `[84.2%, 89.2%]`) are not statistically penalized because SEC clause headers appear in the first sentence. |
| **3. How much GPU memory does `Legal-BERT` require, and what does LoRA change?** | `Legal-BERT` has **109,559,140 parameters** (`23.8M` embeddings, `85.0M` across 12 encoder layers, `76.9K` head). In standard `fp32` `AdamW` (**16 bytes/param**: 4B weight + 4B grad + 4B $m$ + 4B $v$), model states take **1.75 GB**. Our 3-epoch `LoRALinear` run (`src/train.py --lora`, $r=8, \alpha=16$ on $W_Q, W_V$) freezes the backbone in `fp16` (`2B/param`) and trains only **371,812 parameters (`0.34%`, 295× reduction)**, cutting state memory to **224.9 MB** (`7.8×` reduction) and reaching **84.73% test accuracy** and **75.39% Macro-F1** (**99.3% of full fine-tuning's Macro-F1**) in **8.2 minutes** with zero inference latency via `W + (alpha/r) B @ A`. |
| **4. Why do you report both Paired Bootstrap and McNemar's test?** | **McNemar's test** ($\chi^2 = \frac{(|b-c|-1)^2}{b+c}$) tests binary per-clause correctness (`Accuracy` / `Micro-F1`), proving Ensemble vs. TF-IDF is significant at $\chi^2 = 50.35, p = 1.29 \times 10^{-12}$. **Macro-F1** is a non-linear average across 100 classes where no closed-form contingency test applies, so we use **2,000 paired bootstrap resamples** (`+0.94 pts [-0.08, +1.75]`). |
| **5. Does temperature scaling unlock conformal coverage?** | No: **Split Conformal Prediction** guarantees marginal coverage $\mathbb{P}(Y \in C(X)) \ge 1 - \alpha$ (`95.01%` empirical on test) for *any* scoring function by exchangeability. Temperature scaling ($T = 0.684$ on TF-IDF, cutting ECE from `10.99%` to `2.70%`) ensures the probabilities combined in the Ensemble are well-scaled, which sharpens conformal set sizes (`mean 1.43 labels`). |
| **6. Is there a circularity bias in the 20 lawyer queries?** | Yes, and we document it explicitly: Protocol B checks relevance via target category + a regex on the constraint token (`30`, `Delaware`), which favors lexical `BM25` (`69%`) by construction. That is why we also run **Protocol A on 1,000 clause-to-clause queries** without regex bias, where **Hybrid RRF (`76.9%`)** significantly beats both `Dense` (`+2.0 pts [+0.9, +3.1]`) and `BM25` (`+1.2 pts [+0.2, +2.3]`). |
