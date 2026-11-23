import numpy as np

from data import make
from methods import METHODS, anova_f, forest_impurity
from study import selection_leak


def test_data_layout():
    X, y, kinds = make(n=500)
    assert X.shape == (500, 50) and list(kinds[:6]) == ["informative"] * 6 and kinds[-1] == "noise"


def test_every_method_scores_every_feature():
    X, y, _ = make(n=600)
    for name, fn in METHODS.items():
        if "permutation" in name:
            continue  # slow; covered by the study
        s = fn(X, y)
        assert s.shape == (50,) and np.isfinite(s).all(), name


def test_forest_ranks_useful_features_first():
    X, y, kinds = make(n=2000)
    top = np.argsort(forest_impurity(X, y))[::-1][:10]
    assert (kinds[top] != "noise").sum() >= 9


def test_selection_outside_cv_inflates_score_on_noise():
    leaky, honest = selection_leak(n=150, p=2000)
    assert leaky > 0.8
    assert honest < 0.7


def test_anova_noise_scores_low():
    X, y, kinds = make(n=2000)
    s = anova_f(X, y)
    assert np.median(s[kinds == "noise"]) < np.median(s[kinds == "informative"])
