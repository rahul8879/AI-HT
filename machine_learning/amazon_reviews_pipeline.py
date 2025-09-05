import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# ------------------------------
# Data loading & preprocessing
# ------------------------------


def _resolve_data_path(data_path: Optional[str]) -> str:
    """Resolve dataset path with sensible fallbacks."""
    candidates = []
    if data_path:
        candidates.append(data_path)
    # Common fallbacks: same folder as module and project path
    here = os.path.dirname(__file__)
    candidates.append(os.path.join(here, "Reviews.csv"))
    candidates.append(os.path.join(here, "..", "machine_learning", "Reviews.csv"))

    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Reviews.csv not found. Tried: {candidates}. Pass data_path explicitly."
    )


def load_dataset(
    data_path: Optional[str] = None,
    n_samples: Optional[int] = None,
    random_state: int = 42,
    drop_neutral: bool = True,
    text_col: str = "Text",
    score_col: str = "Score",
) -> Tuple[pd.Series, pd.Series]:
    """
    Load Amazon Fine Food Reviews from Reviews.csv and produce binary labels.

    Labeling: score >= 4 -> 1 (positive), score <= 2 -> 0 (negative), score==3 -> neutral.
    If drop_neutral is True, rows with score==3 are removed.

    Returns X (text) and y (labels as 0/1).
    """
    path = _resolve_data_path(data_path)
    usecols = [text_col, score_col]

    df = pd.read_csv(path, usecols=usecols)
    df = df.dropna(subset=[text_col, score_col])

    # Map scores to labels
    df["label"] = np.where(df[score_col] >= 4, 1, np.where(df[score_col] <= 2, 0, 3))
    if drop_neutral:
        df = df[df["label"] != 3]

    # Optional downsampling for faster local runs
    if n_samples is not None and len(df) > n_samples:
        # Stratified sampling to keep class balance
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, n_samples // 2)), random_state=random_state))
            .sample(frac=1.0, random_state=random_state)
        )

    X = df[text_col].astype(str)
    y = df["label"].astype(int)
    return X, y


# ------------------------------
# Vectorizers
# ------------------------------


def build_vectorizer(
    vec_type: str = "tfidf",
    ngram: str = "bi",
    max_features: int = 20000,
    min_df: int = 5,
    stop_words: Optional[str] = "english",
):
    """
    Create a text vectorizer.

    vec_type: 'tfidf' or 'bow'
    ngram: 'uni' | 'bi' | 'tri'
    """
    ngram_range = {
        "uni": (1, 1),
        "bi": (1, 2),
        "tri": (1, 3),
    }.get(ngram, (1, 2))

    if vec_type == "bow":
        return CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            stop_words=stop_words,
        )
    elif vec_type == "tfidf":
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            stop_words=stop_words,
            sublinear_tf=True,
        )
    else:
        raise ValueError("vec_type must be 'tfidf' or 'bow'")


# ------------------------------
# Models
# ------------------------------


def available_models() -> Dict[str, BaseEstimator]:
    """
    Returns a dict of name -> estimator covering common text classifiers.
    Includes optional models if installed.
    """
    models: Dict[str, BaseEstimator] = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=None, solver="liblinear"),
        "linear_svc": LinearSVC(),
        "sgd_log": SGDClassifier(loss="log_loss", max_iter=2000),
        "sgd_svm": SGDClassifier(loss="hinge", max_iter=2000),
        "ridge": RidgeClassifier(),
        "nb": MultinomialNB(),
        "bernoulli_nb": BernoulliNB(),
        # Tree ensembles can be slower; keep defaults modest
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    }

    # Optional: XGBoost / LightGBM if available
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgb"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lgbm"] = LGBMClassifier(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


# ------------------------------
# Evaluation
# ------------------------------


@dataclass
class Metrics:
    model: str
    vec_type: str
    ngram: str
    train_secs: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    support_pos: int
    support_neg: int


def _safe_roc_auc(estimator, X_test, y_test) -> Optional[float]:
    try:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_test)[:, 1]
            return float(roc_auc_score(y_test, proba))
        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X_test)
            return float(roc_auc_score(y_test, scores))
    except Exception:
        return None
    return None


def evaluate_pipeline(
    name: str,
    pipeline_estimator: BaseEstimator,
    X_train,
    X_test,
    y_train,
    y_test,
    vec_type: str,
    ngram: str,
) -> Metrics:
    t0 = time.time()
    pipeline_estimator.fit(X_train, y_train)
    train_secs = time.time() - t0

    y_pred = pipeline_estimator.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = _safe_roc_auc(pipeline_estimator, X_test, y_test)

    # Per-class support for quick sanity
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    support_neg = int(cm[0].sum())
    support_pos = int(cm[1].sum())

    return Metrics(
        model=name,
        vec_type=vec_type,
        ngram=ngram,
        train_secs=train_secs,
        accuracy=acc,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc,
        support_pos=support_pos,
        support_neg=support_neg,
    )


# ------------------------------
# Orchestration
# ------------------------------


def run_benchmark(
    data_path: Optional[str] = None,
    models: Optional[List[str]] = None,
    vec_type: str = "tfidf",
    ngram: str = "bi",
    max_features: int = 20000,
    min_df: int = 5,
    stop_words: Optional[str] = "english",
    n_samples: Optional[int] = 100000,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Load data, split, and evaluate multiple models with a chosen vectorizer.

    Returns a pandas DataFrame of metrics (one row per model).
    """
    X, y = load_dataset(data_path=data_path, n_samples=n_samples, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    vec = build_vectorizer(
        vec_type=vec_type, ngram=ngram, max_features=max_features, min_df=min_df, stop_words=stop_words
    )

    all_models = available_models()
    model_names = models if models is not None else list(all_models.keys())

    results: List[Metrics] = []
    for name in model_names:
        if name not in all_models:
            if verbose:
                print(f"[skip] Model '{name}' not available")
            continue
        clf = all_models[name]
        pipe = make_pipeline(vec, clf)
        if verbose:
            print(f"Training {name} with {vec_type}({ngram}) ...")
        try:
            m = evaluate_pipeline(name, pipe, X_train, X_test, y_train, y_test, vec_type, ngram)
            results.append(m)
            if verbose:
                print(
                    f"  -> f1={m.f1:.4f}, acc={m.accuracy:.4f}, roc_auc={m.roc_auc if m.roc_auc is not None else 'NA'} "
                    f"(time {m.train_secs:.1f}s)"
                )
        except Exception as e:
            if verbose:
                print(f"[error] {name}: {e}")

    # Convert to DataFrame for convenient sorting and display
    df = pd.DataFrame([m.__dict__ for m in results])
    if not df.empty:
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Simple CLI usage
    out = run_benchmark(
        data_path=None,  # tries local Reviews.csv automatically
        models=None,  # run all available
        vec_type="tfidf",
        ngram="bi",
        n_samples=80000,  # adjust for your machine
        verbose=True,
    )
    print("\nResults:")
    pd.set_option("display.max_columns", None)
    print(out)

