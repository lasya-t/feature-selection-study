"""Data where we know the answer: which features are informative, redundant or noise."""
import numpy as np
from sklearn.datasets import make_classification

N_INFORMATIVE, N_REDUNDANT, N_NOISE = 6, 4, 40


def make(n=3000, seed=0):
    # shuffle=False keeps the column order: informative, then redundant, then noise
    X, y = make_classification(
        n_samples=n, n_features=N_INFORMATIVE + N_REDUNDANT + N_NOISE,
        n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT, n_repeated=0,
        n_clusters_per_class=2, class_sep=0.8, flip_y=0.03, shuffle=False, random_state=seed,
    )
    kinds = ["informative"] * N_INFORMATIVE + ["redundant"] * N_REDUNDANT + ["noise"] * N_NOISE
    return X, y, np.array(kinds)
