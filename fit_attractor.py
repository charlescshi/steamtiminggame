"""Model 1 of the proposal: attraction only, no repulsion.

Each game i chooses one of the 52 weeks of its release year with multinomial-logit
probabilities

    P(week w | x_i) = softmax_w( alpha_w + sum_k x_ik f_k(w) )

alpha_w   common calendar, one value per week (shared by all games);
f_k(w)    calendar curve of tag k, a sum of H Fourier pairs over the 52-week year:
          f_k(w) = sum_h a_kh cos(2 pi h w / 52) + b_kh sin(2 pi h w / 52).
Prior     alpha_w ~ N(0, 10^2);  a_kh, b_kh ~ N(0, (s / h)^2).

A periodic curve with independent Gaussian Fourier coefficients whose variance shrinks
with frequency is a Gaussian process with a periodic (stationary) kernel, written in its
spectral form; truncating at H harmonics keeps it to 2H numbers per tag. s sets how far
a tag's calendar may move from the common one; it is chosen on held-out data.

This script finds the posterior mode (MAP) with L-BFGS and compares held-out
log-likelihood per game against two simpler models:
  uniform    every week equally likely (log(1/52) = -3.951 per game);
  calendar   alpha only, same calendar for every game.

Two held-out designs:
  random     20% of games, all years;
  future     train on 2015-2022, test on 2023-2025 (does the calendar repeat?).

Usage: python scripts/fit_attractor.py [--min-reviews 10] [--harmonics 6]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BLUE, INK, INK_2, N_WEEKS, ORANGE, OUT, load_games, month_ticks, style_axes, tag_matrix, tag_names  # noqa: E402

ALPHA_SD = 10.0


def fourier_basis(H):
    w = np.arange(1, N_WEEKS + 1)
    rows = []
    for h in range(1, H + 1):
        rows += [np.cos(2 * np.pi * h * w / N_WEEKS), np.sin(2 * np.pi * h * w / N_WEEKS)]
    return np.array(rows)  # (2H, 52)


class Model:
    def __init__(self, X, weeks, H, s, use_tags=True):
        self.X, self.y = X, weeks - 1  # 0-based week index
        self.B = fourier_basis(H)
        self.K, self.H, self.use_tags = X.shape[1], H, use_tags
        h = np.repeat(np.arange(1, H + 1), 2)
        self.coef_sd = s / h  # (2H,)
        self.n_alpha = N_WEEKS
        self.n_coef = self.K * 2 * H if use_tags else 0

    def unpack(self, theta):
        alpha = theta[: self.n_alpha]
        C = theta[self.n_alpha:].reshape(self.K, 2 * self.H) if self.use_tags else None
        return alpha, C

    def utilities(self, theta, X=None):
        X = self.X if X is None else X
        alpha, C = self.unpack(theta)
        U = np.broadcast_to(alpha, (X.shape[0], N_WEEKS)).copy()
        if C is not None:
            U += X @ (C @ self.B)
        return U

    def neg_log_post(self, theta):
        alpha, C = self.unpack(theta)
        U = self.utilities(theta)
        logp = log_softmax(U, axis=1)
        n = len(self.y)
        ll = logp[np.arange(n), self.y].sum()
        G = -softmax(U, axis=1)
        G[np.arange(n), self.y] += 1.0  # dLL/dU
        lp = -0.5 * np.sum(alpha ** 2) / ALPHA_SD ** 2
        g_alpha = G.sum(axis=0) - alpha / ALPHA_SD ** 2
        grads = [g_alpha]
        if C is not None:
            lp += -0.5 * np.sum((C / self.coef_sd) ** 2)
            g_C = (self.X.T @ G) @ self.B.T - C / self.coef_sd ** 2
            grads.append(g_C.ravel())
        return -(ll + lp), -np.concatenate(grads)

    def fit(self, theta0=None):
        theta0 = np.zeros(self.n_alpha + self.n_coef) if theta0 is None else theta0
        res = minimize(self.neg_log_post, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "maxfun": 4000})
        self.theta, self.result = res.x, res
        return self

    def loglik_vector(self, X, weeks):
        logp = log_softmax(self.utilities(self.theta, X), axis=1)
        return logp[np.arange(len(weeks)), weeks - 1]

    def mean_loglik(self, X, weeks):
        return self.loglik_vector(X, weeks).mean()

    def report(self, label):
        r = self.result
        print(f"  {label}: converged={r.success} iterations={r.nit} ({r.message})")


def paired(a, b):
    """Mean and standard error of the per-game difference a - b."""
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def split_eval(name, train, test, X_tr, X_te, H, s_grid, rng):
    """Fit calendar-only and tag models on `train`, pick s on a slice of train, score on `test`.

    Also scores every s on the test set as a diagnostic (not used for the choice of s).
    """
    w_te = test["week"].values
    unif = np.full(len(test), np.log(1 / N_WEEKS))
    cal = Model(X_tr, train["week"].values, H, 1.0, use_tags=False).fit()
    cal.report(f"[{name}] calendar")
    ll_cal = cal.loglik_vector(X_te, w_te)
    # choose s on a 15% validation slice of the training games
    idx = rng.permutation(len(train))
    va, tr = idx[: len(idx) * 15 // 100], idx[len(idx) * 15 // 100:]
    best = None
    for s in s_grid:
        m = Model(X_tr[tr], train["week"].values[tr], H, s).fit()
        v = m.mean_loglik(X_tr[va], train["week"].values[va])
        print(f"  [{name}] s={s:<5} validation loglik/game {v:.4f}")
        if best is None or v > best[1]:
            best = (s, v)
    s = best[0]
    full = Model(X_tr, train["week"].values, H, s).fit()
    full.report(f"[{name}] calendar + tags, s={s}")
    ll_tag = full.loglik_vector(X_te, w_te)

    rows = []
    for model, ll, s_used in (("uniform", unif, np.nan), ("calendar", ll_cal, np.nan), ("calendar + tag curves", ll_tag, s)):
        gain, se = paired(ll, unif)
        rows.append({"split": name, "model": model, "s": s_used, "n_test": len(test), "test_loglik": ll.mean(),
                     "gain_vs_uniform": gain, "se_vs_uniform": se})
    d, se = paired(ll_tag, ll_cal)
    rows[-1].update({"gain_vs_calendar": d, "se_vs_calendar": se})

    # diagnostic: test score for every s (fit on all training games)
    diag = []
    for s_d in s_grid:
        m = full if s_d == s else Model(X_tr, train["week"].values, H, s_d).fit()
        d, se = paired(m.loglik_vector(X_te, w_te), ll_cal)
        diag.append({"split": name, "s": s_d, "gain_vs_calendar": d, "se": se})
    return rows, full, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reviews", type=int, default=10)
    ap.add_argument("--harmonics", type=int, default=6)
    ap.add_argument("--min-tag-games", type=int, default=30, help="drop tags on fewer games than this")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(5105)
    s_grid = [0.25, 0.5, 1.0, 2.0, 4.0]

    g = load_games(min_reviews=args.min_reviews)
    X, tagids = tag_matrix(g)
    keep = np.asarray((X > 0).sum(axis=0)).ravel() >= args.min_tag_games
    X, tagids = X[:, keep].tocsr(), tagids[keep]
    print(f"{len(g)} games, {X.shape[1]} tags, {args.harmonics} harmonics, min reviews {args.min_reviews}")

    results, diags = [], []
    # random split
    perm = rng.permutation(len(g))
    te, tr = perm[: len(g) // 5], perm[len(g) // 5:]
    rows, m_random, diag = split_eval("random 20%", g.iloc[tr], g.iloc[te], X[tr], X[te], args.harmonics, s_grid, rng)
    results += rows
    diags += diag
    # future split
    tr_mask = (g["year"] <= 2022).values
    rows, _, diag = split_eval("2023-25 from 2015-22", g[tr_mask], g[~tr_mask], X[tr_mask], X[~tr_mask],
                               args.harmonics, s_grid, rng)
    results += rows
    diags += diag

    res = pd.DataFrame(results)
    tag = f"r{args.min_reviews}_h{args.harmonics}"
    res.to_csv(os.path.join(OUT, f"attractor_fit_{tag}.csv"), index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    diags = pd.DataFrame(diags)
    diags.to_csv(os.path.join(OUT, f"attractor_s_diagnostic_{tag}.csv"), index=False)
    print("\nDiagnostic only: test gain of tag curves over the calendar, for every s\n",
          diags.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Implied release-week profile for the average horror game vs the average game.
    names = tag_names()
    col = {names.get(t, str(t)): i for i, t in enumerate(tagids)}
    alpha, C = m_random.unpack(m_random.theta)
    if "Horror" in col:
        is_h = np.asarray(X[:, col["Horror"]].todense()).ravel() > 0
        xbar_all = np.asarray(X.mean(axis=0)).ravel()
        xbar_h = np.asarray(X[is_h].mean(axis=0)).ravel()
        F = C @ m_random.B
        p_all = softmax(alpha + xbar_all @ F) * 100
        p_h = softmax(alpha + xbar_h @ F) * 100
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
        style_axes(ax)
        wk = np.arange(1, N_WEEKS + 1)
        ax.plot(wk, p_all, color=BLUE, lw=2, label="Average game")
        ax.plot(wk, p_h, color=ORANGE, lw=2, label="Average horror game")
        month_ticks(ax)
        ax.set_ylabel("Predicted % chance of each week", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_title("Fitted attraction: predicted release week (posterior mode)", loc="left", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig_attractor_horror_{tag}.png"))
        plt.close(fig)

    # Which tags move their calendar most? Shares sum to 1 for every game, so adding the same
    # curve to every tag and subtracting it from alpha leaves the likelihood unchanged: only
    # differences between tags are identified. Measure each tag against the average game.
    F = C @ m_random.B
    D = F - np.asarray(X.mean(axis=0)).ravel() @ F
    amp = pd.Series(np.ptp(D, axis=1) * 0.1, index=[names.get(t, str(t)) for t in tagids])
    top = amp.sort_values(ascending=False).head(15)
    peak = pd.Series(D.argmax(axis=1) + 1, index=amp.index)
    out = pd.DataFrame({"swing_at_share_0.1": top.round(3), "peak_iso_week": peak[top.index]})
    out.to_csv(os.path.join(OUT, f"attractor_top_tags_{tag}.csv"))
    print("\nTags whose calendar differs most from the common one:\n", out.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
