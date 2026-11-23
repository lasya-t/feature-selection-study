"""1) Which methods find the truly useful features?  2) Does selection help accuracy?
3) The classic mistake: selecting features on the full dataset before cross-validation."""
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import N_INFORMATIVE, N_REDUNDANT, make
from methods import METHODS

K = N_INFORMATIVE + N_REDUNDANT  # 10 useful features exist
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def recovery():
    X, y, kinds = make()
    rows = []
    for name, fn in METHODS.items():
        t0 = time.perf_counter()
        scores = fn(X, y)
        secs = time.perf_counter() - t0
        top = np.argsort(scores)[::-1][:K]
        picked = kinds[top]
        Xs = X[:, top]
        auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xs, y, cv=CV, scoring="roc_auc").mean()
        rows.append({"method": name, "useful in top 10": int((picked != "noise").sum()),
                     "informative": int((picked == "informative").sum()), "redundant": int((picked == "redundant").sum()),
                     "CV AUC (top 10)": auc, "seconds": secs})
    all_auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=CV, scoring="roc_auc").mean()
    return pd.DataFrame(rows), all_auc


def selection_leak(n=200, p=5000, seed=0):
    """Pure noise: no feature is related to y. Honest CV must give AUC ~0.5."""
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(n, p)), rng.integers(0, 2, n)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    # wrong: pick the 20 "best" features using all rows, then cross-validate
    top = np.argsort(f_classif(X, y)[0])[::-1][:20]
    leaky = cross_val_score(model, X[:, top], y, cv=CV, scoring="roc_auc").mean()
    # right: selection is a step inside the pipeline, refit on each training fold
    honest_pipe = make_pipeline(SelectKBest(f_classif, k=20), StandardScaler(), LogisticRegression(max_iter=2000))
    honest = cross_val_score(honest_pipe, X, y, cv=CV, scoring="roc_auc").mean()
    return leaky, honest


if __name__ == "__main__":
    res, all_auc = recovery()
    pd.set_option("display.width", 140)
    print("50 features: 6 informative, 4 redundant (combinations of informative), 40 noise\n")
    print(res.round(3).to_string(index=False))
    print(f"\nall 50 features, CV AUC {all_auc:.3f}")
    leaky, honest = selection_leak()
    print(f"\nPure-noise data (200 rows, 5,000 features, no signal at all):")
    print(f"  select on all rows, then CV  -> AUC {leaky:.3f}   (looks predictive: it is not)")
    print(f"  select inside each CV fold   -> AUC {honest:.3f}   (the truth)")
