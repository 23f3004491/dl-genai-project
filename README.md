# 🧠 Smart MCQ Solver — Deep Learning & Generative AI

> **Author:** Tarun Gangwar
> **Roll Number:** 23f3004491
>
> **Deep Learning & Generative AI Project — IIT Madras**
>
> **Final Public Score (MAP@3):** 0.75685 &nbsp;·&nbsp; **Cutoff:** 0.73 &nbsp;·&nbsp; **Guessing baseline:** 0.42

Smart MCQ Solver is a Deep Learning and Generative AI project for **Multiple-Choice Question
Answering**. Given a question and five options (A–E), the system predicts the **top-3 most
likely answers in ranked order**, scored with **MAP@3**.

The project builds five distinct approaches — from a from-scratch lexical baseline to a
fine-tuned transformer — and combines the useful ones with a retrieval strategy. The headline
finding is that **careful data analysis, not a bigger model, is what actually moved the score**:
after stripping wrapper phrases from the prompts, ~98% of the test questions turn out to be
disguised copies of training questions, and a retrieval lookup built around that fact became the
primary predictor.

---

## 📂 Project Structure

```text
.
├── Data/
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
│
├── Final Notebook/
│   └── dl-23f3004491-notebook-t22026.ipynb      # full end-to-end pipeline (final submission)
│
├── Models/
│   ├── 01_tfidf.ipynb                           # TF-IDF + cosine (from scratch)
│   ├── 02_minilm.ipynb                          # MiniLM bi-encoder (pretrained)
│   ├── 03_cross_encoder.ipynb                   # NLI cross-encoder (zero-shot)
│   ├── 04_lora_deberta.ipynb                    # LoRA fine-tuned DeBERTa (fine-tuned)
│   └── 05_retrieval_lookup.ipynb                # retrieval lookup + final ensemble
│
├── Notebooks/
│   ├── milestone-1.ipynb                        # EDA, TF-IDF, MAP@3
│   ├── milestone-2.ipynb                        # transformers, embeddings, zero-shot
│   ├── milestone-4.ipynb                        # LoRA multiple-choice fine-tuning
│   └── milestone-5.ipynb                        # ensembling and TTA
│
├── Reports/
│   └── Project_Report_23f3004491.pdf
│
└── README.md
```

---

## 📊 Dataset

The competition provides multiple-choice questions with five candidate answers.

**train.csv** — 2,000 rows with `id`, `prompt`, options `A`–`E`, and the correct `answer`.
**test.csv** — 500 rows with the same columns except the answer.
**sample_submission.csv** — the required output format: `ID` and a `Prediction` column holding
three space-separated letters (for example, `B A C`).

### The key property of this dataset

Every prompt begins with one of exactly five fixed phrases (for example, *"Pick the best
possible answer:"*) and often ends with a fixed suffix. These wrappers are decoration, not
content — **86.9%** of training prompts and **57.4%** of test prompts carry at least one.
Stripping them reveals that the 2,000 training rows are really only **~415 unique questions**
repeated under different wrappers, and about **98%** of the test questions duplicate a training
question. This is the single most important fact of the project.

---

## 🏆 Evaluation Metric — MAP@3

For each question the model outputs three ranked guesses. The score is:

| Correct answer position | Score |
|-------------------------|-------|
| Ranked 1st              | 1.00  |
| Ranked 2nd              | 0.50  |
| Ranked 3rd              | 0.33  |
| Not in top 3            | 0.00  |

A "guess the common letters (B C A)" baseline scores about **0.42**, which any useful model
must beat.

---

## 🧩 Models Built

| # | Notebook | Model | Category | Val MAP@3 |
|---|----------|-------|----------|-----------|
| 1 | `01_tfidf.ipynb` | TF-IDF + cosine similarity | From scratch | 0.31 |
| 2 | `02_minilm.ipynb` | MiniLM bi-encoder | Pretrained | 0.40 |
| 3 | `03_cross_encoder.ipynb` | NLI cross-encoder | Zero-shot | 0.56 |
| 4 | `04_lora_deberta.ipynb` | LoRA fine-tuned DeBERTa | Fine-tuned | 0.59 |
| 5 | `05_retrieval_lookup.ipynb` | Retrieval lookup + ensemble | Final | **~0.76** |

### The final pipeline

The final prediction ranks options in three tiers:

1. **Retrieval lookup (rank 1)** — three tiers, from most to least precise: an option whose text
   is a known-correct answer anywhere in training; an exact five-option-set match; or a
   core-question match.
2. **Cross-encoder hedge (rank 2)** — the NLI cross-encoder's best remaining option, which
   recovers questions whose answers were altered in the test set.
3. **Length + frequency signal (rank 3)** — fills the final slot.

Every experiment was tracked in **Weights & Biases** and compared on MAP@3, accuracy and F1.

---

## 📈 What Actually Moved the Score

| Stage | Change | Public score |
|-------|--------|--------------|
| Language model alone | Qwen-7B zero-shot | 0.690 |
| + retrieval lookup | lookup first, model fills the rest | 0.736 |
| + cross-encoder hedge | reasoning model at rank 2 | 0.740 |
| tiered lookup | option-set + core matching | 0.753 |
| + answer-text tier | known-correct-answer-text | **0.75685** |

Every gain came from improving the **lookup**, never from a heavier model. A 7B and then a 14B
language model, tried as the rank-2 reasoner, both *lowered* the score — no model can recover the
subset of test questions whose answer keys were deliberately altered.

---

## ⚙️ How to Run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run any single model notebook (Models/), or the full pipeline (Final Notebook/)
#    on Kaggle with the competition dataset attached.

# notebooks 01, 02, 03, 05 run on CPU; 04 (LoRA) needs a GPU.
```

The final notebook loads the competition data, builds all five models, logs to W&B, and writes
`submission.csv` in the required `ID, Prediction` format.

---

## 💡 Key Learnings

1. Read the data before reaching for a bigger model — the decisive gain came from spotting the
   duplication, not from any neural architecture.
2. Clean the text **before** splitting train/validation, or duplicates leak across the split and
   your validation score lies to you. This project splits by unique core question.
3. Always print a do-nothing baseline (0.42 here) next to every real result.
4. A near-perfect validation score is a bug report, not a win.
5. Bigger is not automatically better — the 14B model actually hurt the score.
6. Some errors are structural: altered test answers set a ceiling no method can cross.

---

## 📄 Report

The full write-up, with charts and the complete methodology, is in
`Reports/Project_Report_23f3004491.pdf`.
