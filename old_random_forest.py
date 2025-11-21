import json, re, os, sys
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import text_to_vector as ttv
import tempfile

# path
CLEANED_TRAIN_PATH = "data/training_data_clean.csv"   #
TEST_PATH          = "data/test_data.csv"

OUT_MODEL_JSON         = "rf_trees.json"
OUT_LABEL_JSON         = "label_mapping.json"
OUT_FEATURE_META_JSON  = "feature_meta.json"
OUT_TEXTVEC_JSON       = "text_vectorizer_meta.json" 

LABEL_COL = "label"
RNG = 42

TARGET_TASKS = [
    "Math computations",
    "Writing or debugging code",
    "Data processing or analysis",
    "Explaining complex concepts simply",
]
BEST_TASKS_COL = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
SUBOPT_TASKS_COL = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

ORDINAL_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

FREE_TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?",
]

# helpers - random forest
def select_topk_by_variance(X, k=3000):
    X = np.asarray(X)
    var = X.var(axis=0)
    k = int(min(k, var.shape[0]))
    idx = np.argpartition(var, -k)[-k:]
    idx = idx[np.argsort(-var[idx])]
    return idx

def tfidf_transform_with_vocab(texts: List[str], vocab: Dict[str,int], idf_vec: np.ndarray) -> np.ndarray:
    n = len(texts)
    V = len(vocab)
    X = np.zeros((n, V), dtype=np.float32)
    for i, s in enumerate(texts):
        if not isinstance(s, str):
            s = str(s)
        s = s.lower()
        s = re.sub(r'[^a-z0-9\s]', '', s)
        words = s.split()
        if not words:
            continue
        counts = {}
        for w in words:
            j = vocab.get(w)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        total = float(len(words))
        if total > 0 and counts:
            idxs = list(counts.keys())
            X[i, idxs] = [counts[j] / total for j in idxs]
    return X * idf_vec

def multiselect_onehots(series: pd.Series, choices: List[str]) -> np.ndarray:
    vals = series.fillna("").astype(str).tolist()
    out = np.zeros((len(vals), len(choices)), dtype=np.int16)
    idx = {c:i for i,c in enumerate(choices)}
    for r, cell in enumerate(vals):
        for c in choices:
            if c in cell:
                out[r, idx[c]] = 1
    return out

def keyword_flags(joined: pd.Series) -> np.ndarray:
    txt = joined.fillna("").astype(str).str.lower()
    return np.vstack([
        txt.str.contains(r"\b(?:chatgpt|gpt)\b").astype(int).to_numpy(),
        txt.str.contains(r"\bclaude\b").astype(int).to_numpy(),
        txt.str.contains(r"\bgemini\b").astype(int).to_numpy(),
        txt.str.contains(r"\b(?:python|code|def|class|import)\b").astype(int).to_numpy(),
        txt.str.contains(r"\b(?:math|equation|formula|compute)\b").astype(int).to_numpy(),
        txt.str.contains(r"\b(?:cite|reference|references|source|link)\b").astype(int).to_numpy(),
    ]).T

def build_tabular_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    # ordinals → first digit
    ord_feats = []
    for c in ORDINAL_COLS:
        v = df.get(c)
        if v is None:
            ord_feats.append(np.zeros((len(df),1), dtype=np.int16))
        else:
            ord_feats.append(pd.to_numeric(v.astype(str).str.strip().str[0], errors="coerce")
                               .fillna(0).astype("Int16").values.reshape(-1,1))
    ord_feats = np.hstack(ord_feats) if ord_feats else np.zeros((len(df),0))

    best_hot = multiselect_onehots(df.get(BEST_TASKS_COL, pd.Series([""]*len(df))), TARGET_TASKS)
    subopt_hot = multiselect_onehots(df.get(SUBOPT_TASKS_COL, pd.Series([""]*len(df))), TARGET_TASKS)

    t1 = df.get(FREE_TEXT_COLS[0], pd.Series([""]*len(df))).astype(str)
    t2 = df.get(FREE_TEXT_COLS[1], pd.Series([""]*len(df))).astype(str)
    t3 = df.get(FREE_TEXT_COLS[2], pd.Series([""]*len(df))).astype(str)
    lens = np.vstack([t1.str.len().values, t2.str.len().values, t3.str.len().values]).T
    joined = (t1 + " " + t2 + " " + t3)
    flags = keyword_flags(joined)

    X_tab = np.hstack([ord_feats, best_hot, subopt_hot, lens, flags]).astype(np.float32)
    names = (
        [f"ord_{i+1}" for i in range(ord_feats.shape[1])] +
        [f"best_{t}" for t in TARGET_TASKS] +
        [f"subopt_{t}" for t in TARGET_TASKS] +
        [f"textlen_{i+1}" for i in range(lens.shape[1])] +
        ["flag_gpt","flag_claude","flag_gemini","flag_code","flag_math","flag_refs"]
    )
    return X_tab, names, joined.tolist()

def export_forest_to_json(rf: RandomForestClassifier, classes: List[str], out_path: str) -> None:
    forest = []
    for est in rf.estimators_:
        t = est.tree_
        forest.append({
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "value": t.value.squeeze(axis=1).tolist()
        })
    with open(out_path, "w") as f:
        json.dump({"classes": classes, "trees": forest}, f)

def fast_tune_rf(X, y, rng=42):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.20, random_state=rng, stratify=y
    )
    candidates = [
        (None, 2, 1, None),
        (None, 2, 2, None),
        (16,   2, 1, None),
        (16,   2, 2, None),
        (None, 2, 1, "balanced"),
        (None, 2, 2, "balanced"),
        (16,   2, 1, "balanced"),
        (16,   2, 2, "balanced"),
    ]
    best = (-1.0, None)
    for md, mss, msl, cw in candidates:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=md,
            min_samples_split=mss,
            min_samples_leaf=msl,
            max_features="sqrt",
            class_weight=cw,
            bootstrap=True,
            max_samples=0.8,
            n_jobs=-1,
            random_state=rng,
        )
        rf.fit(X_tr, y_tr)
        acc = rf.score(X_va, y_va)
        if acc > best[0]:
            best = (acc, (md, mss, msl, cw))
    md, mss, msl, cw = best[1]
    rf_final = RandomForestClassifier(
        n_estimators=600,
        max_depth=md,
        min_samples_split=mss,
        min_samples_leaf=msl,
        max_features="sqrt",
        class_weight=cw,
        bootstrap=True,
        max_samples=0.9,
        n_jobs=-1,
        random_state=rng,
    )
    rf_final.fit(X, y)
    return rf_final, {"max_depth": md, "min_samples_split": mss,
                      "min_samples_leaf": msl, "class_weight": cw}

# main
def main():
    print(">>> starting RF training...", flush=True)
    if not os.path.exists(CLEANED_TRAIN_PATH):
        print(f"Missing: {CLEANED_TRAIN_PATH}")
        sys.exit(1)

    df = pd.read_csv(CLEANED_TRAIN_PATH).dropna(subset=[LABEL_COL]).reset_index(drop=True)
    df_train = df.iloc[:743]  
    if LABEL_COL not in df_train.columns:
        print(f"'{LABEL_COL}' not found in {CLEANED_TRAIN_PATH}")
        sys.exit(1)

    # text (train) via text_to_vector — use the same 743 rows as df_train
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_train.to_csv(tmp.name, index=False)

    tfidf_train, y_train, vocab, label_mapping = ttv.process_data_manual(
        tmp.name, FREE_TEXT_COLS, LABEL_COL
    )

    # rebuild train IDF
    train_joined = (df_train[FREE_TEXT_COLS[0]].fillna('') + ' ' +
                    df_train[FREE_TEXT_COLS[1]].fillna('') + ' ' +
                    df_train[FREE_TEXT_COLS[2]].fillna('')).tolist()
    tokenized_docs = [ttv.tokenize(s) for s in train_joined]
    V = len(vocab); num_docs = len(tokenized_docs)
    doc_freq = np.zeros(V, dtype=np.int32)
    # vocab is word->index; ensure index order
    vocab_sorted = [w for w,_ in sorted(vocab.items(), key=lambda kv: kv[1])]
    for i, w in enumerate(vocab_sorted):
        for doc in tokenized_docs:
            if w in doc:
                doc_freq[i] += 1
    idf_vec = (np.log(num_docs / (1.0 + doc_freq)) + 1.0).astype(np.float32)

    # tabular (train)
    X_tab_train, tab_names, _ = build_tabular_features(df_train)

    # stack text and table :>
    tfidf_train = tfidf_train.astype(np.float32)
    # top-K TF-IDF slice for speed/accuracy
    tfidf_idx = select_topk_by_variance(tfidf_train, k=3000)
    X_train = np.hstack([X_tab_train, tfidf_train[:, tfidf_idx]]).astype(np.float32)

    classes = [c for c,_ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    y = y_train.astype(np.int64)

    # tuning + final fit
    rf_final, best_cfg = fast_tune_rf(X_train, y, rng=RNG)
    print({"chosen_params": best_cfg})

    # saving artifacts
    export_forest_to_json(rf_final, classes, OUT_MODEL_JSON)
    with open(OUT_LABEL_JSON, "w") as f:
        json.dump({"classes": classes, "label_to_id": label_mapping}, f)
    feat_meta = {
        "tabular_feature_names": tab_names,
        "text_feature_dim": int(len(tfidf_idx)),
        "free_text_cols": FREE_TEXT_COLS,
        "ordinal_cols": ORDINAL_COLS,
        "target_tasks": TARGET_TASKS,
        "best_tasks_col": BEST_TASKS_COL,
        "subopt_tasks_col": SUBOPT_TASKS_COL,
    }
    with open(OUT_FEATURE_META_JSON, "w") as f:
        json.dump(feat_meta, f)
    with open(OUT_TEXTVEC_JSON, "w") as f:
        json.dump({"vocab": vocab_sorted, "idf": idf_vec.tolist(),
                   "tfidf_idx": tfidf_idx.tolist()}, f)
    print(f"Saved: {OUT_MODEL_JSON}, {OUT_LABEL_JSON}, {OUT_FEATURE_META_JSON}, {OUT_TEXTVEC_JSON}")

    # test time
    if not os.path.exists(TEST_PATH):
        print(f"Test file not found: {TEST_PATH} — skipping eval.")
        return
    df_test = pd.read_csv(TEST_PATH).reset_index(drop=True)
    X_tab_test, _, test_joined = build_tabular_features(df_test)

    vocab_indexed = {w:i for i,w in enumerate(vocab_sorted)}
    X_text_test_full = tfidf_transform_with_vocab(test_joined, vocab_indexed, idf_vec)
    X_text_test = X_text_test_full[:, tfidf_idx]  # apply the same top-K slice

    X_test = np.hstack([X_tab_test, X_text_test.astype(np.float32)])
    y_pred_test = rf_final.predict(X_test)

    if LABEL_COL in df_test.columns and not df_test[LABEL_COL].isna().all():
        y_test_lbl = df_test[LABEL_COL].astype(str)
        known = y_test_lbl.isin(classes)
        if not known.all():
            print(f"Warning: {(~known).sum()} unseen labels in test; excluded from accuracy.")
        if known.any():
            y_test = np.array([label_mapping[v] for v in y_test_lbl[known]], dtype=np.int64)
            acc = accuracy_score(y_test, y_pred_test[known])
            print(f"FINAL TEST ACCURACY: {acc*100:.2f}%")
        else:
            print("No known-class test labels to score.")
    else:
        ids, counts = np.unique(y_pred_test, return_counts=True)
        id2lbl = {i:c for c,i in label_mapping.items()}
        print("Pred counts:", {id2lbl[i]: int(n) for i,n in zip(ids, counts)})

if __name__ == "__main__":
    main()
