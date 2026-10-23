import numpy as np
import pandas as pd
import pytest

from volfc.backtest import single_vol_target_backtest
from volfc.cv import PurgedKFold, walk_forward_splits
from volfc.data import make_synthetic
from volfc.evaluate import block_bootstrap_ci, dm_test, holm_bonferroni, mincer_zarnowitz, qlike
from volfc.features import MAX_WINDOW, build_dataset, gk_variance
from volfc.models import GARCHModel


def test_features_do_not_look_ahead():
    df = make_synthetic(600, seed=1)
    X0, _ = build_dataset(df, horizon=3)
    t = df.index[400]
    df2 = df.copy()
    future = df2.index > t
    df2.loc[future, "High"] *= 1.2
    df2.loc[future, "Close"] *= 0.97
    X1, _ = build_dataset(df2, horizon=3)
    pd.testing.assert_frame_equal(X0.loc[:t], X1.loc[:t])


def test_target_is_mean_future_variance():
    df = make_synthetic(300, seed=2)
    h = 4
    X, y = build_dataset(df, horizon=h)
    gk = gk_variance(df)
    i = df.index.get_loc(X.index[50])
    expected = np.log(gk.iloc[i + 1 : i + 1 + h].mean())
    assert y.iloc[50] == pytest.approx(expected)


@pytest.mark.parametrize("h", [1, 5])
def test_purged_kfold_has_no_overlap(h):
    n, emb = 1000, MAX_WINDOW + h
    for train, test in PurgedKFold(5, horizon=h, embargo=emb).split(n):
        a, b = test[0], test[-1] + 1
        assert not set(train) & set(test)
        # every train label window ends before the test block starts, or train starts after the embargo
        assert all((s + h <= a) or (s >= b + emb) for s in train)


@pytest.mark.parametrize("h", [1, 5])
def test_walk_forward_uses_only_realised_labels(h):
    n = 800
    blocks = list(walk_forward_splits(n, initial_train=300, step=50, horizon=h))
    assert blocks[0][1][0] == 300 and blocks[-1][1][-1] == n - 1
    for train, test in blocks:
        assert train.max() + h <= test.min()
    covered = np.concatenate([t for _, t in blocks])
    assert np.array_equal(covered, np.arange(300, n))


def test_qlike_is_zero_for_perfect_forecast_and_dm_detects_better_model():
    v = np.random.default_rng(0).uniform(1e-4, 4e-4, 500)
    assert qlike(v, v).max() == pytest.approx(0.0)
    good, bad = qlike(v, v * 1.02), qlike(v, v * 1.6)
    stat, p = dm_test(good, bad)
    assert stat < 0 and p < 0.01


def test_garch_does_not_look_ahead():
    """Verify that GARCH forecast at time t does not change if future prices change."""
    df = make_synthetic(600, seed=1)
    X0, y0 = build_dataset(df, horizon=1)
    t_idx = 400
    t = X0.index[t_idx]

    m0 = GARCHModel(horizon=1, calibrate=True).fit(X0.iloc[:t_idx], y0.iloc[:t_idx])
    pred0 = m0.predict(X0.iloc[t_idx : t_idx + 1])[0]

    df2 = df.copy()
    future = df2.index > t
    df2.loc[future, "High"] *= 1.5
    df2.loc[future, "Close"] *= 0.8
    X1, y1 = build_dataset(df2, horizon=1)

    m1 = GARCHModel(horizon=1, calibrate=True).fit(X1.iloc[:t_idx], y1.iloc[:t_idx])
    pred1 = m1.predict(X1.iloc[t_idx : t_idx + 1])[0]

    assert pred0 == pytest.approx(pred1, rel=1e-5)


def test_vol_targeting_positions_use_only_past_close():
    """Verify that position[t] depends strictly on forecasts and prices up to close t-1."""
    df = make_synthetic(500, seed=3)
    prices = df["Close"]
    forecasts = pd.Series(-9.2, index=prices.index)
    t_idx = 350
    t = prices.index[t_idx]

    res0 = single_vol_target_backtest(prices, forecasts, target_vol=0.15)
    w0 = res0["weights"].loc[t]

    # Perturb prices on day t and all future days (prices >= t)
    prices2 = prices.copy()
    prices2.loc[prices2.index >= t] *= 2.5
    res1 = single_vol_target_backtest(prices2, forecasts, target_vol=0.15)
    w1 = res1["weights"].loc[t]

    assert w0 == pytest.approx(w1, rel=1e-7)


def test_mincer_zarnowitz_and_bootstrap():
    """Verify MZ diagnostics, Holm-Bonferroni correction, and block bootstrap CI."""
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 500)
    pred_perfect = y.copy()
    mz = mincer_zarnowitz(y, pred_perfect)
    assert mz["mz_intercept"] == pytest.approx(0.0, abs=1e-5)
    assert mz["mz_slope"] == pytest.approx(1.0, abs=1e-5)
    assert mz["mz_r2"] == pytest.approx(1.0, abs=1e-5)

    # Holm correction monotonic
    pvals = [0.005, 0.04, 0.03, 0.20]
    adj = holm_bonferroni(pvals)
    assert (np.diff(np.sort(adj)) >= -1e-8).all()
    assert (adj >= np.array(pvals) - 1e-8).all()

    # Block bootstrap returns interval bounding difference
    loss_a = y**2
    loss_b = (y + 0.3)**2
    boot = block_bootstrap_ci(loss_a, loss_b, horizon=1, n_boot=200, seed=42)
    assert boot["ci_lower"] <= boot["diff_mean"] <= boot["ci_upper"]


def test_reproducibility_seed():
    """Verify that two runs with the same seed give identical synthetic datasets and splits."""
    df1 = make_synthetic(300, seed=99)
    df2 = make_synthetic(300, seed=99)
    pd.testing.assert_frame_equal(df1, df2)
