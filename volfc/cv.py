"""Validation schemes for time-ordered data.

Row i is a sample whose features use data up to day i and whose label covers days i+1 ... i+h.
So the label of row s is fully known only at time s+h.
"""
from __future__ import annotations

import numpy as np


def walk_forward_splits(n: int, initial_train: int, step: int, horizon: int):
    """Expanding-window walk-forward.

    For a test block starting at `start`, the model is fit at time `start`, so it may only use
    rows whose labels are already realised: s + horizon <= start.
    """
    start = initial_train
    while start < n:
        stop = min(start + step, n)
        train = np.arange(0, start - horizon + 1)
        yield train, np.arange(start, stop)
        start = stop


class PurgedKFold:
    """K contiguous test blocks with purging and an embargo (Lopez de Prado style).

    Purge: drop training rows before the test block whose label window overlaps it.
    Embargo: drop training rows after the test block for `embargo` days, because their features
    look back over days that contain the test labels. Use embargo >= longest feature window + horizon.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, embargo: int = 0):
        self.n_splits, self.horizon, self.embargo = n_splits, horizon, embargo

    def split(self, n: int):
        edges = np.linspace(0, n, self.n_splits + 1).astype(int)
        for k in range(self.n_splits):
            a, b = edges[k], edges[k + 1]
            left = np.arange(0, max(a - self.horizon + 1, 0))  # s + h <= a
            right = np.arange(min(b + self.embargo, n), n)
            yield np.concatenate([left, right]), np.arange(a, b)
