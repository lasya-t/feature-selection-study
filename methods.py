"""Feature selection methods, each returning a score per feature (higher = more useful).

filter    - judge each feature on its own: mutual information, ANOVA F-test
wrapper   - search subsets by refitting a model: recursive feature elimination (RFE)
embedded  - read importance off a fitted model: L1 coefficients, random-forest impurity
model-agnostic - permutation importance on held-out data
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mutual_info(X, y):
    return mutual_info_classif(X, y, random_state=0)


def anova_f(X, y):
    return f_classif(X, y)[0]


def rfe_logistic(X, y, keep=10):
    Xs = StandardScaler().fit_transform(X)
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=keep, step=2).fit(Xs, y)
    return -rfe.ranking_.astype(float)  # rank 1 = kept last = best


def l1_logistic(X, y, C=0.05):
    Xs = StandardScaler().fit_transform(X)
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xs, y)
    return np.abs(m.coef_.ravel())


def forest_impurity(X, y):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(X, y).feature_importances_


def permutation(X, y):
    a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(a, c)
    return permutation_importance(rf, b, d, n_repeats=5, random_state=0, n_jobs=-1).importances_mean


METHODS = {
    "mutual information (filter)": mutual_info,
    "ANOVA F (filter)": anova_f,
    "RFE + logistic (wrapper)": rfe_logistic,
    "L1 logistic (embedded)": l1_logistic,
    "forest impurity (embedded)": forest_impurity,
    "permutation (model-agnostic)": permutation,
}
