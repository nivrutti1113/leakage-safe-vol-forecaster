"""Baselines (naive, rolling mean, HAR, GARCH, GJR-GARCH) and ML models (Ridge, GBM).

All models predict the log of the future mean variance over the next h days.
Hyper-parameters are fixed in advance (not tuned on test data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HAR_COLS


class Passthrough:
    """No-fit baseline: forecast = one feature column."""

    def __init__(self, col: str):
        self.col = col

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.col].to_numpy()


class ColumnModel:
    """Wrap an sklearn estimator so it sees only `cols` (all columns if None)."""

    def __init__(self, est, cols=None):
        self.est, self.cols = est, cols

    def _x(self, X: pd.DataFrame):
        return (X[self.cols] if self.cols else X).to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.est.fit(self._x(X), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est.predict(self._x(X))


class GARCHModel:
    """GARCH(1,1) and GJR-GARCH(1,1,1) baseline using the arch library.

    Fits on daily close-to-close returns within the training window.
    Supports:
      - Raw GARCH: outputs log close-to-close variance directly.
      - Calibrated GARCH: inside the training window, fits a Mincer-Zarnowitz
        regression y_train = a + b * log(garch_in_sample) to absorb level bias
        between close-to-close variance and Garman-Klass intraday variance,
        then applies (a, b) out-of-sample.
      - GJR-GARCH: models leverage asymmetry (negative return shock increases variance).
      - Multi-step h > 1 analytical mean variance forecast.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        horizon: int = 1,
        gjr: bool = False,
        calibrate: bool = True,
    ):
        self.p = p
        self.q = q
        self.horizon = max(horizon, 1)
        self.gjr = gjr
        self.calibrate = calibrate
        self.params = None
        self.a = 0.0
        self.b = 1.0
        self.last_cond_var = None
        self.last_ret = None
        self.pers = 0.95
        self.omega = 0.05
        self.alpha = 0.05
        self.gamma = 0.0
        self.beta = 0.90

    def _get_ret(self, X: pd.DataFrame) -> np.ndarray:
        if "ret" in X.columns:
            return X["ret"].to_numpy(dtype=float)
        return np.where(X["neg_ret"] < 0, X["neg_ret"], X["absret"]).astype(float)

    def _mean_variance(self, s_next2: float) -> float:
        """Analytical mean variance over h days given s_{t+1}^2."""
        h = self.horizon
        pers = self.pers
        if h == 1 or pers >= 1.0:
            return s_next2
        uncond_var = self.omega / max(1.0 - pers, 1e-6)
        geom_sum = (1.0 - pers**h) / max(1.0 - pers, 1e-6)
        return uncond_var + (geom_sum / h) * (s_next2 - uncond_var)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        r = self._get_ret(X) * 100.0  # scale to percentage for numerical stability
        n_tr = len(r)
        try:
            from arch import arch_model

            o_param = 1 if self.gjr else 0
            am = arch_model(r, mean="Zero", vol="Garch", p=self.p, o=o_param, q=self.q)
            res = am.fit(disp="off", show_warning=False)
            self.omega = float(res.params["omega"])
            self.alpha = float(res.params["alpha[1]"])
            self.gamma = float(res.params["gamma[1]"]) if self.gjr else 0.0
            self.beta = float(res.params["beta[1]"])
            self.pers = self.alpha + 0.5 * self.gamma + self.beta

            cond = np.asarray(res.conditional_volatility)
            self.last_cond_var = float(cond[-1] ** 2)
            self.last_ret = float(r[-1])
            self.params = res.params

            if self.calibrate:
                # In-sample 1-step (or h-step) forecasts on the training set
                in_sample = np.empty(n_tr)
                for t in range(n_tr):
                    rt = r[t]
                    s2 = cond[t] ** 2
                    ind = 1.0 if rt < 0.0 else 0.0
                    s_next2 = (
                        self.omega
                        + (self.alpha + self.gamma * ind) * (rt**2)
                        + self.beta * s2
                    )
                    mv = self._mean_variance(s_next2)
                    in_sample[t] = np.log(max(mv / 10000.0, 1e-12))

                y_arr = y.to_numpy(dtype=float)
                mask = np.isfinite(in_sample) & np.isfinite(y_arr)
                if mask.sum() > 30:
                    b, a = np.polyfit(in_sample[mask], y_arr[mask], deg=1)
                    if np.isfinite(b) and b > 0.0:
                        self.a, self.b = float(a), float(b)
                    else:
                        self.a = float(y_arr[mask].mean() - in_sample[mask].mean())
                        self.b = 1.0
                else:
                    self.a, self.b = 0.0, 1.0
        except Exception:
            self.params = None
            self.a, self.b = 0.0, 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        r = self._get_ret(X) * 100.0
        n = len(r)
        if self.params is None or self.last_cond_var is None:
            return X["lv1"].to_numpy() if "lv1" in X.columns else np.full(n, -9.0)

        preds = np.empty(n)
        curr_var = self.last_cond_var
        curr_ret = self.last_ret

        for i in range(n):
            ind = 1.0 if curr_ret < 0.0 else 0.0
            next_var = (
                self.omega
                + (self.alpha + self.gamma * ind) * (curr_ret**2)
                + self.beta * curr_var
            )
            mv = self._mean_variance(next_var)
            raw_logv = np.log(max(mv / 10000.0, 1e-12))
            preds[i] = self.a + self.b * raw_logv if self.calibrate else raw_logv
            curr_var = next_var
            curr_ret = r[i]

        return preds


def make_models(horizon: int = 1) -> dict:
    """Return dict of model factories for out-of-sample walk-forward forecasting."""
    return {
        "naive": lambda: Passthrough("lv1"),
        "roll22": lambda: Passthrough("lv22"),
        "har": lambda: ColumnModel(LinearRegression(), HAR_COLS),
        "garch_raw": lambda: GARCHModel(
            p=1, q=1, horizon=horizon, gjr=False, calibrate=False
        ),
        "garch": lambda: GARCHModel(
            p=1, q=1, horizon=horizon, gjr=False, calibrate=True
        ),
        "gjr_garch": lambda: GARCHModel(
            p=1, q=1, horizon=horizon, gjr=True, calibrate=True
        ),
        "ridge": lambda: ColumnModel(
            make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=0))
        ),
        "gbm": lambda: ColumnModel(
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=200,
                max_depth=3,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
            )
        ),
    }


def make_high_capacity_gbm():
    """Deliberately flexible model used in the leakage demo: it can memorise neighbouring rows."""
    return ColumnModel(
        HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=3,
            l2_regularization=0.0,
            random_state=0,
        )
    )
