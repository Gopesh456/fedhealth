"""Logistic regression from scratch (numpy) - the model federated learning trains here.

    P(response | x) = sigmoid(w . x)        loss = mean log-loss + (l2 / 2) |w|^2

Chosen because it is convex (FedAvg's behaviour is easy to reason about), has one weight per feature (so the
learned coefficients can be compared with the generator's TRUE coefficients), and its entire state is a short
vector of floats - exactly what a hospital may send out.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> float:
    p = np.clip(predict_proba(w, X), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * l2 * float(w @ w))


def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
    return X.T @ (predict_proba(w, X) - y) / len(y) + l2 * w


def train_local(w0: np.ndarray, X: np.ndarray, y: np.ndarray, *, epochs: int, lr: float, batch: int = 64,
                l2: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mini-batch gradient descent starting from `w0` (a hospital's contribution to one FedAvg round)."""
    rng = rng or np.random.default_rng(0)
    w = w0.copy()
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            w -= lr * gradient(w, X[idx], y[idx], l2)
    return w


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) statistic, with tie correction."""
    y = np.asarray(y)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = np.asarray(scores)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
