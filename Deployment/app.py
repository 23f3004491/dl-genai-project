"""Smart MCQ Solver - Streamlit Community Cloud app.

Interactive demo of the competition pipeline. A user enters a question and five
options; the app predicts the top-3 answers in ranked order using the same
three-tier retrieval lookup and cross-encoder hedge from the final notebook.

Roll No: 23f3004491
"""

import re
import difflib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st


OPTIONS = ["A", "B", "C", "D", "E"]

START_WRAPPERS = [
    "Pick the best possible answer:",
    "Select the most accurate option:",
    "Determine the correct option:",
    "Identify the correct statement:",
    "Choose the correct answer:",
]


def normalize_core(prompt):
    p = str(prompt).strip()

    for s in START_WRAPPERS:
        if p.startswith(s):
            p = p[len(s):].strip()

    if "?" in p:
        p = p[:p.rfind("?") + 1]

    return re.sub(r"\s+", " ", p).lower().strip()


def option_signature(row):
    return "||".join(
        sorted(str(row[o]).strip().lower() for o in OPTIONS)
    )


# ----------------------------------------------------------------------------
# Build the retrieval tables once and cache them across reruns.
# ----------------------------------------------------------------------------

@st.cache_resource
def load_lookup_tables():
    # Get the directory containing app.py
    BASE_DIR = Path(__file__).resolve().parent

    # train.csv is in the same Deployment folder as app.py
    train_path = BASE_DIR / "train.csv"

    # Helpful error if the file is missing
    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found at: {train_path}"
        )

    train = pd.read_csv(train_path)

    train["core"] = train["prompt"].apply(normalize_core)

    train["answer_text"] = train.apply(
        lambda r: str(r[r["answer"]]).strip(),
        axis=1
    )

    known_texts = set(train["answer_text"].str.lower())

    train["osig"] = train.apply(option_signature, axis=1)

    osig = defaultdict(list)

    for sig, ans in zip(train["osig"], train["answer"]):
        osig[sig].append(ans)

    osig_letter = {
        s: Counter(v).most_common(1)[0][0]
        for s, v in osig.items()
    }

    core = defaultdict(list)

    for c, atext in zip(train["core"], train["answer_text"]):
        core[c].append(atext)

    core_answer = {
        c: Counter(v).most_common(1)[0][0]
        for c, v in core.items()
    }

    freq_order = train["answer"].value_counts().index.tolist()

    return (
        known_texts,
        osig_letter,
        core_answer,
        freq_order,
    )


@st.cache_resource
def load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            "cross-encoder/nli-deberta-v3-small",
            device="cpu",
        )

    except Exception:
        return None


# Load lookup tables
(
    KNOWN_ANSWER_TEXTS,
    osig_letter,
    core_answer,
    FREQ_ORDER,
) = load_lookup_tables()


def lookup_letter(row):
    # Tier 1: exact answer-text lookup
    matches = [
        o
        for o in OPTIONS
        if str(row[o]).strip().lower() in KNOWN_ANSWER_TEXTS
    ]

    if len(matches) == 1:
        return matches[0]

    # Tier 2: option-signature lookup
    sig = option_signature(row)

    if sig in osig_letter:
        return osig_letter[sig]

    # Tier 3: normalized question lookup
    ans = core_answer.get(row["core"])

    if ans is not None:

        for o in OPTIONS:
            if str(row[o]).strip() == ans:
                return o

        sims = sorted(
            (
                (
                    difflib.SequenceMatcher(
                        None,
                        str(row[o]).strip().lower(),
                        ans.lower(),
                    ).ratio(),
                    o,
                )
                for o in OPTIONS
            ),
            reverse=True,
        )

        if sims and sims[0][0] > 0.80:
            return sims[0][1]

    return None


def length_freq_rank(row):
    scores = {
        o: len(str(row[o]))
        + (5 - FREQ_ORDER.index(o)) * 0.5
        for o in OPTIONS
    }

    return sorted(
        OPTIONS,
        key=lambda o: -scores[o],
    )


def cross_encoder_rank(prompt, row):
    ce = load_cross_encoder()

    if ce is None:
        return None

    pairs = [
        (prompt, str(row[o]))
        for o in OPTIONS
    ]

    logits = ce.predict(
        pairs,
        show_progress_bar=False,
    )

    entail = (
        logits[:, 1]
        if getattr(logits, "ndim", 1) == 2
        else logits
    )

    return [
        OPTIONS[j]
        for j in np.argsort(entail)[::-1]
    ]


def predict(prompt, opts):
    row = {
        "prompt": prompt,
        "core": normalize_core(prompt),
    }

    for o, v in zip(OPTIONS, opts):
        row[o] = v

    # Retrieval lookup
    la = lookup_letter(row)

    # Cross-encoder ranking
    hedge = (
        cross_encoder_rank(prompt, row)
        or length_freq_rank(row)
    )

    # Fallback ranking
    filler = length_freq_rank(row)

    if la is not None:
        ranked = [la]

        for o in hedge:
            if o not in ranked:
                ranked.append(o)
                break

        for o in filler:
            if o not in ranked:
                ranked.append(o)
                break

        source = "retrieval lookup (high confidence)"

    else:
        ranked = hedge[:3]
        source = "cross-encoder reasoning (novel question)"

    return ranked[:3], source


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
)

st.title("Smart MCQ Solver")

st.write(
    "Enter a multiple-choice question and its five options. "
    "The model predicts the top-3 answers in ranked order using "
    "a three-tier retrieval lookup with a cross-encoder hedge. "
)


EXAMPLE = {
    "prompt": (
        "Pick the best possible answer: "
        "What is the primary source of the Sun's energy?"
    ),
    "A": "Chemical combustion of gases",
    "B": "Nuclear fusion of hydrogen into helium",
    "C": "Gravitational collapse of the core",
    "D": "Nuclear fission of heavy elements",
    "E": "Magnetic reconnection in the corona",
}


if st.button("Load example"):
    for k, v in EXAMPLE.items():
        st.session_state[k] = v


prompt = st.text_area(
    "Question",
    key="prompt",
    height=80,
)


col1, col2 = st.columns(2)

a = col1.text_input(
    "Option A",
    key="A",
)

b = col2.text_input(
    "Option B",
    key="B",
)

c = col1.text_input(
    "Option C",
    key="C",
)

d = col2.text_input(
    "Option D",
    key="D",
)

e = st.text_input(
    "Option E",
    key="E",
)


if st.button("Predict Top-3", type="primary"):

    if not prompt.strip():
        st.warning("Please enter a question.")

    else:
        opts = [a, b, c, d, e]

        if any(not str(opt).strip() for opt in opts):
            st.warning("Please enter all five options.")

        else:
            with st.spinner("Scoring options..."):
                ranked, source = predict(
                    prompt,
                    opts,
                )

            labels = dict(
                zip(OPTIONS, opts)
            )

            st.success(
                f"Predicted ranking (top 3): {' '.join(ranked)}"
            )

            for rank, opt in enumerate(ranked, 1):
                st.markdown(
                    f"**{rank}. Option {opt}** — {labels[opt]}"
                )

            st.caption(
                f"Answer source: {source}"
            )
